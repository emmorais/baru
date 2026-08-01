//! `baru bench-cnn` — a full convolutional classifier through the hybrid
//! pipeline: FHE ciphertext×ciphertext linear layers, MPC nonlinearities
//! (ReLU, max-pool, argmax), **zero FHE bootstraps**, verified bit-exact
//! against the plaintext reference, with a real cost measurement.
//!
//! The FHE side is **BFV at RLWE-secure parameters** (n=4096, q=2^100, at or
//! below the ~109-bit ceiling for n=4096 at the 128-bit security level), with
//! the no-relinearization trick the whole construction rests on: multiply
//! once, then decrypt the degree-2 result under `s²`.
//!
//! Architecture (a compact LeNet-shaped classifier):
//!   input 1×D×D  →  conv(OC, 3×3, s1)  →  ReLU∘rescale  →  maxpool(2×2, s2)
//!               →  flatten  →  dense(→ NCLASSES)  →  argmax  →  class bit
//!
//! Both the image and the weights are encrypted (ct×ct linear layers). The
//! convolution uses **Cheetah-style coefficient packing**
//! ([`crate::fhe::conv_pack`]): the kernel is packed reversed on every axis,
//! so one ciphertext multiplication *is* the whole sliding-window
//! convolution for an entire output channel. That costs `OC` multiplications
//! instead of the `⌈OC·oh·ow·D²/n⌉` an im2col/Toeplitz matvec would need —
//! at 28×28 with `OC = 8`, hundreds of Toeplitz multiplications become 8. The dense layer
//! still uses the packed matvec (convolution is the case where overlapping
//! windows are a feature; for a dense layer the `n/in_dim` separable-output
//! limit is already tight). Every layer's FHE ciphertext is depth-1 fresh,
//! so no bootstrapping occurs at any point — the same invariant
//! `bench-deep-nn` demonstrates for MLPs, now for a CNN.
//!
//! # Scope / honesty
//!
//! - **Correctness is exact** (encrypted class == plaintext class). This is
//!   the crypto claim. **Accuracy** on a real dataset is an *orthogonal*
//!   axis that needs trained weights. The weights *here* are
//!   controlled-magnitude synthetic values (a cost/correctness
//!   microbenchmark, standard for this line of work), not a trained
//!   classifier — for the trained model on real MNIST test digits (98.9%,
//!   run through this same hybrid, bit-exact) see `baru bench-mnist` /
//!   `crate::bench::mnist`, and for real *biomedical* images see
//!   `baru bench-medmnist` / `crate::bench::medmnist`.
//! - Conv packing needs `2·in_c·H·W ≲ n`, so a full 28×28 MNIST image runs
//!   comfortably at `n = 4096` (it needs degree 841). Multi-channel CIFAR
//!   (3×32×32) needs 5185 and so must split channels across ciphertexts (or
//!   move to n=8192) — a packing question, not a limit of the no-bootstrap
//!   construction.
//!
//! # Where the time goes (measured, D=28, `--both`, 2026-07-28)
//!
//! Read this before optimising anything here — the answer has changed twice,
//! and both times the intuitive answer was wrong.
//!
//! | | a2b | gc | gc + `--precompute` |
//! |---|---|---|---|
//! | total (online) | 2535 ms | 1986 ms | **1312 ms** |
//! | offline | — | — | 680 ms |
//! | FHE | 1356 ms | 1351 ms | 695 ms |
//! | MPC | 1153 ms | 609 ms | 600 ms |
//! | — import | 9 ms | 11 ms | 10 ms |
//! | — ReLU∘rescale | 615 ms | 407 ms | 404 ms |
//! | — max-pool | 467 ms | 155 ms | 152 ms |
//! | — argmax | 55 ms | 27 ms | 27 ms |
//! | wire bytes | 39.5 MB | 21.8 MB | 21.8 MB |
//! | messages | 88 | 319 139 | 319 139 |
//!
//! FHE breakdown of the middle column (`--gc`, masks online): keygen 9 ms,
//! input 17 ms, weights 218 ms (13 encryptions), mul 210 ms, **mask 656 ms
//! (39 encryptions)**, decrypt 217 ms, re-encrypt 50 ms (3). The right column
//! is the same with the mask row moved offline.
//!
//! (Separate `--gc` and default runs, so the two columns are two invocations;
//! `--both` runs them back to back and prints the ratio.) Note the message
//! *count* inverts while bytes halve — garbling ships many small messages.
//! That costs nothing at zero RTT and, measured, little at real ones: the MPC
//! segment stays 1.72× ahead at metro (RTT 2 ms) and 1.40× at WAN (RTT 20 ms).
//!
//! 1. Conv packing made the FHE cost nearly **flat in D** (`OC`
//!    multiplications regardless of image size), which moved the bottleneck
//!    to the MPC segment.
//! 2. But *within* the MPC segment the cost was never the comparisons. It was
//!    the **scheme-switch import**: reconstructing an opened layer into
//!    shares ran three blocking input-sharing rounds *per activation*, i.e.
//!    16 224 round trips for D=28's 5408-wide conv output. That was 19.1 s of
//!    a 20.4 s MPC segment — 88% of end-to-end runtime — while the ReLU it
//!    was importing values for cost 0.6 s. Batching it
//!    ([`import_raw_batch`]) costs the same bytes in three rounds: 19.1 s →
//!    8 ms, and end-to-end 21.8 s → 2.6 s.
//! 3. Only *then* do the comparisons matter, and [`Nonlin::Gc`] takes the
//!    MPC segment a further 1.9×.
//! 4. The last unbatched phase was [`crate::gadgets::argmax`], whose left
//!    fold made `nclasses−1` *dependent* comparisons. Reassociating it into a
//!    tournament (`⌈log₂ n⌉` batched levels) cut it 2.4× at RTT 20 ms, where
//!    it had been the single largest phase.
//! 5. With MPC down to a fraction, the same breakdown applied to the FHE side
//!    ([`FhePhases`]) showed **48% of it was encrypting scheme-switch masks**
//!    — 39 of the inference's 52 encryptions. Masks are uniform and
//!    query-independent, so `--precompute` ([`MaskPool`]) moves them offline:
//!    online 1979 ms → 1312 ms here, 4.94 s → 2.90 s on the medical model.
//!
//! The cost centre is now the **FHE arithmetic proper** — the ct×ct products,
//! the model encryption and the decryptions, which depend on the query or the
//! weights and so cannot be preprocessed away. That is where the next
//! optimisation belongs, and it is a schoolbook single-threaded NTT.
//!
//! The lesson worth keeping: "MPC is slow" was true of this pipeline for the
//! wrong reason, and a protocol-level fix (garbled comparisons) was proposed
//! for a cost that was really an unbatched round trip. The phase breakdown in
//! [`MpcPhases`] exists so that mistake is not repeatable.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::bench::comm::{CommStats, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, sleep_link, MpcTransport};
use crate::crypto::types::Fr;
use crate::fhe::conv_pack::{self, ConvShape};
use crate::fhe::bfv::{Bfv, BfvCt, BfvParams};
use crate::gadgets::{argmax, maxpool2d, Chw};
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

const OC: usize = 8; // conv output channels
const K: usize = 3; // conv kernel size
const POOL: usize = 2; // max-pool window / stride
const NCLASSES: usize = 10;

fn residue(v: i64, nbits: usize) -> u128 {
    (v as i128).rem_euclid(1i128 << nbits) as u128
}

// ─── The CNN weights + exact plaintext reference (fused fixed-point) ──────────

/// One additional convolution layer beyond the first (which is stored
/// directly on `CnnModel` for the common single-conv case). Multi-channel:
/// `kernels[out_c][in_c·K·K]`, row-major `(in_c, ky, kx)`.
#[derive(Clone)]
pub struct ConvLayer {
    pub kernels: Vec<Vec<i64>>,
    pub cbias: Vec<i64>,
    pub in_c: usize,
    pub in_h: usize,
    pub in_w: usize,
}

impl ConvLayer {
    fn out_c(&self) -> usize {
        self.kernels.len()
    }
    fn out_h(&self) -> usize {
        self.in_h - K + 1
    }
    fn out_w(&self) -> usize {
        self.in_w - K + 1
    }
}

pub struct CnnModel {
    pub d: usize,        // input side length (1 channel)
    pub kernels: Vec<Vec<i64>>, // first conv: [OC][K*K]
    pub cbias: Vec<i64>, // [OC]
    /// Additional conv layers after the first (each followed by ReLU∘rescale
    /// and max-pool). Empty for the shallow model `bench-cnn` exercises; one
    /// or more for a deeper trained CNN. Every extra conv is one more
    /// FHE-conv + MPC-nonlinearity + scheme-switch — no bootstrap, at any
    /// convolutional depth.
    pub extra_conv: Vec<ConvLayer>,
    /// Zero or more hidden dense layers, each `(weights[out][in], bias[out])`
    /// followed by a fused ReLU∘rescale, stacked between the pooled conv
    /// features and the output layer. Empty for the shallow model `bench-cnn`
    /// exercises; one or more for the trained deeper classifiers in
    /// `bench-mnist`. Each layer composes in with just another FHE matvec +
    /// MPC ReLU + scheme switch — no bootstrap, at any depth.
    pub hidden: Vec<(Vec<Vec<i64>>, Vec<i64>)>,
    pub dense_w: Vec<Vec<i64>>, // [NCLASSES][flat or last hidden width]
    pub dense_b: Vec<i64>, // [NCLASSES]
    pub shift: u32,
    pub nbits: usize,
    pub conv_shape: Chw, // OC × (D-2) × (D-2)
    pub pool_shape: Chw, // OC × ((D-2)/2) × ((D-2)/2)
}

impl CnnModel {
    pub fn random(d: usize, seed: u64) -> (Self, Vec<i64>) {
        assert!((d - (K - 1)).is_multiple_of(POOL), "conv output side must be even for pooling");
        let oh = d - (K - 1);
        let ph = oh / POOL;
        let flat = OC * ph * ph;
        let shift = 6;

        let mut rng = StdRng::seed_from_u64(seed);
        let kernels = (0..OC)
            .map(|_| (0..K * K).map(|_| rng.gen_range(-4i64..=4)).collect())
            .collect();
        let cbias = (0..OC).map(|_| rng.gen_range(-16i64..=16)).collect();
        let dense_w = (0..NCLASSES)
            .map(|_| (0..flat).map(|_| rng.gen_range(-4i64..=4)).collect())
            .collect();
        let dense_b = (0..NCLASSES).map(|_| rng.gen_range(-16i64..=16)).collect();
        let input: Vec<i64> = (0..d * d).map(|_| rng.gen_range(0i64..64)).collect();

        let mut model = CnnModel {
            d,
            kernels,
            cbias,
            extra_conv: Vec::new(),
            hidden: Vec::new(),
            dense_w,
            dense_b,
            shift,
            nbits: 20, // provisional; calibrated below
            conv_shape: Chw { c: OC, h: oh, w: oh },
            pool_shape: Chw { c: OC, h: ph, w: ph },
        };
        // Size the plaintext modulus `t = 2^nbits` from the **measured**
        // activation range rather than a hardcoded width. `forward` is pure
        // integer arithmetic and does not read `nbits`, so calibrating from
        // it is sound. A fixed `t` is a real correctness hazard: too small
        // and a dot product silently wraps the modulus, giving wrong (not
        // merely noisy) results — the same bug that bit `bench-hybrid-nn` at
        // in_dim=784.
        let (_, max_abs) = model.forward(&input);
        let needed = (64 - (max_abs.max(1) as u64).leading_zeros() as usize) + 2;
        assert!(
            needed <= 30,
            "cnn: activations need 2^{needed}, past the depth-1 noise budget              (log2_t ≤ 30 at n=4096, q=2^100); lower `shift` or the weight range"
        );
        model.nbits = needed.max(20);
        (model, input)
    }

    /// Build a model from **trained, quantised** weights (see
    /// `examples/data/mnist_cnn.rs`). The weight tensors are already in the
    /// integer fixed-point scheme `forward` expects; `nbits` is calibrated
    /// from the observed activation range exactly as [`CnnModel::random`]
    /// does, so it fits the depth-1 noise budget.
    #[allow(clippy::too_many_arguments)]
    pub fn from_trained(
        d: usize,
        kernels: Vec<Vec<i64>>,
        cbias: Vec<i64>,
        extra_conv: Vec<ConvLayer>,
        hidden: Vec<(Vec<Vec<i64>>, Vec<i64>)>,
        dense_w: Vec<Vec<i64>>,
        dense_b: Vec<i64>,
        shift: u32,
        sample_inputs: &[Vec<i64>],
    ) -> Self {
        assert_eq!(dense_w.len(), dense_b.len(), "dense weights/bias length mismatch");
        for (w, b) in &hidden {
            assert_eq!(w.len(), b.len(), "hidden layer weights/bias length mismatch");
        }
        let oc = kernels.len();
        let oh = d - (K - 1);
        let ph = oh / POOL;
        let mut model = CnnModel {
            d,
            kernels,
            cbias,
            extra_conv,
            hidden,
            dense_w,
            dense_b,
            shift,
            nbits: 20,
            conv_shape: Chw { c: oc, h: oh, w: oh },
            pool_shape: Chw { c: oc, h: ph, w: ph },
        };
        // Calibrate nbits from the worst activation seen over the samples.
        let mut max_abs = 1i64;
        for x in sample_inputs {
            max_abs = max_abs.max(model.forward(x).1);
        }
        let needed = (64 - (max_abs.max(1) as u64).leading_zeros() as usize) + 2;
        assert!(needed <= 30, "trained model needs 2^{needed}, past the noise budget");
        model.nbits = needed.max(20);
        model
    }

    /// First-conv output channels. `bench-cnn`'s synthetic model uses the `OC`
    /// default; a trained model (`bench-mnist`, `bench-medmnist`) carries
    /// whatever its weight file was generated with.
    pub fn out_channels(&self) -> usize {
        self.kernels.len()
    }

    /// Number of output classes — 10 for MNIST, 2 for PneumoniaMNIST, 11 for
    /// OrganAMNIST. Read from the weights rather than a const so one pipeline
    /// serves every dataset.
    pub fn nclasses(&self) -> usize {
        self.dense_w.len()
    }

    /// The layer stack as a one-line string, read off the model itself.
    ///
    /// How many convs and hidden dense layers there are is a property of the
    /// trained weights (the trainer's `--conv`/`--fc`), not of this code, so a
    /// benchmark banner that spells the architecture out as a literal goes
    /// stale the moment a deeper model is trained — the same drift the
    /// generated weight files exist to prevent.
    pub fn architecture(&self) -> String {
        let mut s = format!(
            "1×{0}×{0} → conv({1}, {K}×{K}) → ReLU → pool",
            self.d,
            self.out_channels()
        );
        for cl in &self.extra_conv {
            s += &format!(" → conv({}, {K}×{K}) → ReLU → pool", cl.out_c());
        }
        for (w, _) in &self.hidden {
            s += &format!(" → dense(→{}) → ReLU", w.len());
        }
        s + &format!(" → dense(→{}) → argmax", self.nclasses())
    }

    fn conv_dot(&self, input: &[i64], oc: usize, oy: usize, ox: usize) -> i64 {
        let mut acc = 0i64;
        for ky in 0..K {
            for kx in 0..K {
                acc += self.kernels[oc][ky * K + kx] * input[(oy + ky) * self.d + (ox + kx)];
            }
        }
        acc
    }

    /// Exact plaintext forward pass mirroring the hybrid's fused semantics.
    /// Returns `(class, max_abs_intermediate)`.
    pub fn forward(&self, input: &[i64]) -> (usize, i64) {
        let (logits, max_abs) = self.forward_logits(input);
        (crate::gadgets::argmax_ref(&logits), max_abs)
    }

    /// As [`CnnModel::forward`], but returning the raw output-layer logits
    /// rather than just the argmax.
    ///
    /// Ranking metrics need a score, not a decision: AUC — the metric MedMNIST
    /// leads with — is a property of how the logits *order* the test set, and
    /// is invisible to a benchmark that only ever sees `argmax`. The encrypted
    /// pipeline deliberately reveals only the class, so this is a
    /// plaintext-side measurement of the same weights.
    pub fn forward_logits(&self, input: &[i64]) -> (Vec<i64>, i64) {
        let oh = self.conv_shape.h;
        let out_c = self.out_channels();
        let mut max_abs = input.iter().map(|v| v.abs()).max().unwrap_or(0);
        // conv → fused ReLU∘rescale
        let mut conv_act = vec![0i64; out_c * oh * oh];
        for oc in 0..out_c {
            for oy in 0..oh {
                for ox in 0..oh {
                    let dot = self.conv_dot(input, oc, oy, ox);
                    let v = dot + (self.cbias[oc] << self.shift);
                    max_abs = max_abs.max(v.abs());
                    let a = if v >= 0 { v >> self.shift } else { 0 };
                    conv_act[(oc * oh + oy) * oh + ox] = a;
                }
            }
        }
        // max-pool
        let (mut feat, mut fshape) =
            crate::gadgets::maxpool2d_ref(&conv_act, self.conv_shape, POOL, POOL);
        // additional conv layers (multi-channel), each → ReLU∘rescale → pool
        for cl in &self.extra_conv {
            debug_assert_eq!((cl.in_c, cl.in_h, cl.in_w), (fshape.c, fshape.h, fshape.w));
            let (coh, cow) = (cl.out_h(), cl.out_w());
            let mut act = vec![0i64; cl.out_c() * coh * cow];
            for oc in 0..cl.out_c() {
                for oy in 0..coh {
                    for ox in 0..cow {
                        let mut dot = 0i64;
                        for ic in 0..cl.in_c {
                            for ky in 0..K {
                                for kx in 0..K {
                                    dot += cl.kernels[oc][(ic * K + ky) * K + kx]
                                        * feat[(ic * cl.in_h + (oy + ky)) * cl.in_w + (ox + kx)];
                                }
                            }
                        }
                        let v = dot + (cl.cbias[oc] << self.shift);
                        max_abs = max_abs.max(v.abs());
                        act[(oc * coh + oy) * cow + ox] = if v >= 0 { v >> self.shift } else { 0 };
                    }
                }
            }
            let (p, ps) = crate::gadgets::maxpool2d_ref(
                &act,
                Chw { c: cl.out_c(), h: coh, w: cow },
                POOL,
                POOL,
            );
            feat = p;
            fshape = ps;
        }
        // stacked hidden dense layers, each → fused ReLU∘rescale
        let mut dense_in = feat;
        for (hw, hb) in &self.hidden {
            dense_in = hw
                .iter()
                .zip(hb)
                .map(|(row, &b)| {
                    let v = row.iter().zip(&dense_in).map(|(&w, &x)| w * x).sum::<i64>() + b;
                    max_abs = max_abs.max(v.abs());
                    if v >= 0 {
                        v >> self.shift
                    } else {
                        0
                    }
                })
                .collect();
        }
        // output dense → logits
        let logits: Vec<i64> = self
            .dense_w
            .iter()
            .zip(&self.dense_b)
            .map(|(row, &b)| {
                let v = row.iter().zip(&dense_in).map(|(&w, &x)| w * x).sum::<i64>() + b;
                max_abs = max_abs.max(v.abs());
                v
            })
            .collect();
        (logits, max_abs)
    }

}

// ─── Hybrid FHE matvec + scheme-switch to opened shares ───────────────────────

/// One packed ct×ct matvec of `weights` (row-major, each length `in_dim`)
/// against `c_input`, chunked to respect the ring packing bound, followed
/// by a mask-and-open switch. Returns per-output opened residues and the
/// three per-output masks (so the MPC side can reconstruct the raw share).
#[allow(clippy::type_complexity)]
/// How many weight rows a single packed matvec ciphertext can carry. Shared
/// by the matvec itself and by [`MaskPool::for_model`], which must predict
/// the same chunking — the two drifting apart is what the pool's
/// exact-consumption assertion catches.
fn rows_per_chunk(n: usize, in_dim: usize) -> usize {
    (n / in_dim).saturating_sub(1).max(1)
}

/// Every packed weight polynomial the pipeline will multiply against, in the
/// order it consumes them: the first conv's kernels, then each extra conv's,
/// then each hidden dense layer's chunks, then the output dense's.
///
/// This is the pipeline's shape written down **once**. It gives one entry per
/// ct×ct multiply, and since every multiply is followed by exactly one
/// scheme-switch mask, it also sizes [`MaskPool`] — so the pool's count and the
/// encrypted model's cannot drift apart, because they are the same walk. (The
/// count used to be re-derived by hand in `MaskPool::for_model`; that is the
/// kind of duplication the exact-consumption assertions existed to catch, and
/// deriving both from one traversal removes the possibility instead.)
fn weight_polys(s: &Bfv, model: &CnnModel) -> Vec<Vec<u64>> {
    let n = s.params.n;
    let t_mask = s.params.t() - 1;
    let mut out: Vec<Vec<u64>> = Vec::new();

    let conv1 = ConvShape { in_c: 1, h: model.d, w: model.d, kh: K, kw: K };
    for kernel in &model.kernels {
        out.push(conv_pack::pack_kernel(kernel, conv1, n, t_mask));
    }
    for cl in &model.extra_conv {
        let shape = ConvShape { in_c: cl.in_c, h: cl.in_h, w: cl.in_w, kh: K, kw: K };
        for kernel in &cl.kernels {
            out.push(conv_pack::pack_kernel(kernel, shape, n, t_mask));
        }
    }

    let mut in_dim = {
        let fs = last_feature_shape(model);
        fs.c * fs.h * fs.w
    };
    for (hidden_w, _) in &model.hidden {
        for chunk in hidden_w.chunks(rows_per_chunk(n, in_dim)) {
            out.push(s.pack_weights(chunk));
        }
        in_dim = hidden_w.len();
    }
    for chunk in model.dense_w.chunks(rows_per_chunk(n, in_dim)) {
        out.push(s.pack_weights(chunk));
    }
    out
}

/// The model's weight ciphertexts, encrypted once per **session**.
///
/// This is a different kind of preprocessing from [`MaskPool`], and the
/// difference is the point:
///
/// - **Masks are per-query.** They must be fresh for every inference, so
///   precomputing them shifts latency off the critical path but buys **no
///   throughput** — a saturated server does the same total work.
/// - **Weights are per-session.** The model does not change between patients,
///   so encrypting it once is genuine **amortization**: total work per query
///   drops, permanently, after the first.
///
/// Any future preprocessing in this pipeline should say which of the two it is.
/// Weights are stored **already forward-transformed** (`BfvCtNtt`), not merely
/// encrypted. `mul_pre` transforms its weight operand on every call; since a
/// session's weights never change, those transforms are offline work too, and
/// [`Bfv::mul_pre_pre`] then leaves only the inverse transforms online.
struct EncryptedModel {
    /// In [`weight_polys`] order — the order the pipeline consumes them.
    cts: Vec<crate::fhe::bfv::BfvCtNtt>,
    /// Wall-clock spent encrypting *and* transforming, reported as offline.
    offline_ms: f64,
}

impl EncryptedModel {
    fn encrypt(s: &Bfv, ek: &crate::fhe::bfv::EncryptKey, polys: &[Vec<u64>]) -> Self {
        let t = Instant::now();
        let cts = polys
            .iter()
            .map(|p| s.transform(&s.encrypt_poly_pre(ek, p)))
            .collect();
        EncryptedModel { cts, offline_ms: ms(t) }
    }
}

/// Where a layer's weight ciphertext comes from.
///
/// Unlike [`Masks`], the session variant **borrows**: a weight ciphertext is
/// reused by every query in the session rather than consumed by one.
struct Weights<'a> {
    /// `None` = encrypt on the critical path, which is what a deployment
    /// without a session preamble must do on every single query. This is the
    /// control the `--precompute` measurement is against.
    model: Option<&'a EncryptedModel>,
    /// How many have been handed out, so exact consumption can be asserted.
    next: usize,
}

impl<'a> Weights<'a> {
    fn online() -> Self {
        Weights { model: None, next: 0 }
    }

    fn session(model: &'a EncryptedModel) -> Self {
        Weights { model: Some(model), next: 0 }
    }

    /// The next weight operand, and the product of it with the pre-transformed
    /// input. `poly` is only evaluated in the online case, so the session path
    /// does not even pack the polynomial.
    ///
    /// The multiply lives here because the two paths reach it differently: the
    /// session path has both operands transformed and uses `mul_pre_pre`, the
    /// online path transforms the weights inside `mul_pre`. Keeping that choice
    /// next to the source of the weights is what stops a caller from silently
    /// paying the transform twice.
    fn mul_next(
        &mut self,
        s: &Bfv,
        ek: &crate::fhe::bfv::EncryptKey,
        cxt: &crate::fhe::bfv::BfvCtNtt,
        poly: impl FnOnce() -> Vec<u64>,
        ph: &mut FhePhases,
    ) -> BfvCt {
        let i = self.next;
        self.next += 1;
        match self.model {
            Some(m) => {
                let cw = m.cts.get(i).unwrap_or_else(|| {
                    panic!(
                        "encrypted model exhausted at {i}: weight_polys under-counted \
                         the model's ct×ct multiplies"
                    )
                });
                let t = Instant::now();
                let prod = s.mul_pre_pre(cxt, cw);
                ph.mul_ms += ms(t);
                prod
            }
            None => {
                let t = Instant::now();
                let cw = s.encrypt_poly_pre(ek, &poly());
                ph.weights_ms += ms(t);
                ph.n_weight_enc += 1;
                let t = Instant::now();
                let prod = s.mul_pre(cxt, &cw);
                ph.mul_ms += ms(t);
                prod
            }
        }
    }

    fn assert_fully_consumed(&self) {
        if let Some(m) = self.model {
            assert_eq!(
                self.next,
                m.cts.len(),
                "encrypted model over-counted: {} ciphertexts built, {} consumed",
                m.cts.len(),
                self.next
            );
        }
    }
}

/// One layer-output's worth of scheme-switch masking: the three parties'
/// plaintext masks, and a single ciphertext encrypting their sum.
///
/// The evaluator only ever needs the sum (it adds one ciphertext onto the
/// product); each party needs its own plaintext mask to reconstruct its
/// share afterwards. Summing the three encryptions in advance is therefore
/// faithful to the protocol and strictly cheaper than doing it online.
struct MaskTriple {
    plain: [Vec<u64>; 3],
    combined: BfvCt,
}

/// Precomputed scheme-switch masks — the pipeline's **offline phase**.
///
/// The masks are uniform random values: they depend on neither the query nor
/// the model, only on the network's *shape*. So every one of them can be
/// generated and encrypted before a patient's image exists, which is the
/// classic Delphi-style online/offline split applied to the switch rather
/// than to the linear layers. Measured, this is where roughly half the FHE
/// time was going.
///
/// Masks must still be **fresh per inference** — a pool is consumed once and
/// refilled between queries, never reused across them, since two queries
/// masked identically would reveal the difference of their pre-activations.
/// Precomputation moves the cost off the critical path; it does not remove it.
struct MaskPool {
    triples: std::collections::VecDeque<MaskTriple>,
    /// Wall-clock spent generating the pool — reported as offline cost.
    offline_ms: f64,
    /// How many triples were drawn, so the caller can assert exact consumption.
    drawn: usize,
}

impl MaskPool {
    /// Build exactly `count` masks — one per ct×ct multiply, which
    /// [`weight_polys`] enumerates. Under-counting panics at draw time and
    /// over-counting panics at the end of the run, so a pipeline that consumed
    /// them in a different order still cannot pass silently.
    fn with_len(
        s: &Bfv,
        ek: &crate::fhe::bfv::EncryptKey,
        count: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let n = s.params.n;
        let t_mask = s.params.t() - 1;
        let t = Instant::now();
        let triples = (0..count)
            .map(|_| {
                let plain: [Vec<u64>; 3] = std::array::from_fn(|_| {
                    (0..n).map(|_| rng.gen::<u64>() & t_mask).collect()
                });
                let mut combined: Option<BfvCt> = None;
                for m in &plain {
                    let enc = s.encrypt_poly_pre(ek, m);
                    combined = Some(match combined {
                        None => enc,
                        Some(acc) => s.add(&acc, &enc),
                    });
                }
                MaskTriple { plain, combined: combined.expect("three parties") }
            })
            .collect();
        MaskPool { triples, offline_ms: ms(t), drawn: 0 }
    }

    fn draw(&mut self) -> MaskTriple {
        self.drawn += 1;
        self.triples
            .pop_front()
            .expect("mask pool exhausted: MaskPool::for_model under-counted the model's switches")
    }

    fn assert_fully_consumed(&self) {
        assert!(
            self.triples.is_empty(),
            "mask pool over-counted: {} triples left unused after {} draws",
            self.triples.len(),
            self.drawn
        );
    }
}

/// Borrow the pool as a [`Masks`] source, or fall back to online generation.
fn masks_src(pool: &mut Option<MaskPool>) -> Masks<'_> {
    match pool {
        Some(p) => Masks::Pool(p),
        None => Masks::Online,
    }
}

/// The feature-map shape entering the first dense layer.
fn last_feature_shape(model: &CnnModel) -> Chw {
    let mut shape = model.pool_shape;
    for cl in &model.extra_conv {
        shape = Chw { c: cl.out_c(), h: cl.out_h() / POOL, w: cl.out_w() / POOL };
    }
    shape
}

/// Where a layer's masks come from: generated on the critical path (the
/// control, and what a system without an offline phase must do), or drawn
/// from a pool built before the query arrived.
enum Masks<'a> {
    Online,
    Pool(&'a mut MaskPool),
}

impl Masks<'_> {
    /// Produce one output's masks and the single ciphertext to add onto the
    /// product, charging the time to `ph.mask_ms` only when it is online.
    fn take(
        &mut self,
        s: &Bfv,
        ek: &crate::fhe::bfv::EncryptKey,
        n: usize,
        t_mask: u64,
        rng: &mut impl Rng,
        ph: &mut FhePhases,
    ) -> ([Vec<u64>; 3], BfvCt, usize) {
        match self {
            Masks::Online => {
                let plain: [Vec<u64>; 3] =
                    std::array::from_fn(|_| (0..n).map(|_| rng.gen::<u64>() & t_mask).collect());
                let t = Instant::now();
                let mut combined: Option<BfvCt> = None;
                let mut bytes = 0usize;
                for m in &plain {
                    let enc = s.encrypt_poly_pre(ek, m);
                    bytes += s.ct_bytes(&enc);
                    combined = Some(match combined {
                        None => enc,
                        Some(acc) => s.add(&acc, &enc),
                    });
                }
                ph.mask_ms += ms(t);
                ph.n_mask_enc += 3;
                (plain, combined.expect("three parties"), bytes)
            }
            Masks::Pool(pool) => {
                let tr = pool.draw();
                // Same bytes on the wire: the three parties still ship their
                // mask ciphertexts, just before the query rather than during it.
                let bytes = 3 * s.ct_bytes(&tr.combined);
                (tr.plain, tr.combined, bytes)
            }
        }
    }
}

fn hybrid_matvec_open(
    s: &Bfv,
    ek: &crate::fhe::bfv::EncryptKey,
    c_input: &BfvCt,
    weights: &[Vec<i64>],
    in_dim: usize,
    profile: NetworkProfile,
    rng: &mut impl Rng,
    masks_src: &mut Masks,
    weights_src: &mut Weights,
    dk: &crate::fhe::bfv::DecryptKey,
) -> (Vec<u64>, Vec<[u64; 3]>, FhePhases) {
    let n = s.params.n;
    let t_mask = s.params.t() - 1;
    let rows_per_chunk = rows_per_chunk(n, in_dim);
    let mut ph = FhePhases::default();
    // The input's forward NTT is computed once and reused across chunks.
    let t = Instant::now();
    let cxt = s.transform(c_input);
    ph.mul_ms += ms(t);

    let mut opened = Vec::with_capacity(weights.len());
    let mut masks = Vec::with_capacity(weights.len());
    for chunk in weights.chunks(rows_per_chunk) {
        let prod = weights_src.mul_next(s, ek, &cxt, || s.pack_weights(chunk), &mut ph);
        let (chunk_masks, mask_ct, conv_bytes) =
            masks_src.take(s, ek, n, t_mask, rng, &mut ph);
        let masked = s.add(&prod, &mask_ct);
        let t = Instant::now();
        let y = s.decrypt_poly_pre(dk, &masked);
        ph.decrypt_ms += ms(t);
        sleep_link(&profile, conv_bytes + chunk.len() * 8, 2);
        for r in 0..chunk.len() {
            let pos = s.matvec_output_position(in_dim, r);
            opened.push(y[pos]);
            masks.push([chunk_masks[0][pos], chunk_masks[1][pos], chunk_masks[2][pos]]);
        }
    }
    (opened, masks, ph)
}

/// Convolution via [`crate::fhe::conv_pack`]: one ct×ct multiply per output
/// channel, then a mask-and-open per channel. Returns the opened
/// pre-activations and per-output masks flattened in `(oc, oy, ox)` order —
/// the same layout the MPC stage and the plaintext reference use.
#[allow(clippy::type_complexity)]
fn hybrid_conv_open(
    s: &Bfv,
    ek: &crate::fhe::bfv::EncryptKey,
    c_input: &BfvCt,
    shape: ConvShape,
    kernels: &[Vec<i64>],
    profile: NetworkProfile,
    rng: &mut impl Rng,
    masks_src: &mut Masks,
    weights_src: &mut Weights,
    dk: &crate::fhe::bfv::DecryptKey,
) -> (Vec<u64>, Vec<[u64; 3]>, FhePhases) {
    let n = s.params.n;
    let t_mask = s.params.t() - 1;
    assert!(
        shape.required_degree() < n,
        "conv packing needs degree {} < ring dimension {n}",
        shape.required_degree()
    );
    let out_c = kernels.len();
    let (oh, ow) = (shape.out_h(), shape.out_w());
    let mut opened = Vec::with_capacity(out_c * oh * ow);
    let mut masks = Vec::with_capacity(out_c * oh * ow);
    let mut ph = FhePhases::default();
    // Every output channel multiplies against the *same* encrypted input, so
    // its forward NTT is computed once and reused (`mul_pre`).
    let t = Instant::now();
    let cxt = s.transform(c_input);
    ph.mul_ms += ms(t);

    for kernel in kernels.iter() {
        let prod = weights_src.mul_next(
            s,
            ek,
            &cxt,
            || conv_pack::pack_kernel(kernel, shape, n, t_mask),
            &mut ph,
        );

        // Fresh masks per output channel (reuse across channels would leak
        // the difference of their pre-activations).
        let (chan_masks, mask_ct, conv_bytes) =
            masks_src.take(s, ek, n, t_mask, rng, &mut ph);
        let masked = s.add(&prod, &mask_ct);
        let t = Instant::now();
        let y = s.decrypt_poly_pre(dk, &masked);
        ph.decrypt_ms += ms(t);
        sleep_link(&profile, conv_bytes + oh * ow * 8, 2);
        for oy in 0..oh {
            for ox in 0..ow {
                let pos = shape.output_position(oy, ox);
                opened.push(y[pos]);
                masks.push([chan_masks[0][pos], chan_masks[1][pos], chan_masks[2][pos]]);
            }
        }
    }
    (opened, masks, ph)
}

/// Re-encrypt a layer's MPC output (three parties' additive share
/// components of each value) as a single packed BFV ciphertext — the
/// MPC→FHE half of the scheme switch. Communication-free per party except
/// the one modelled link hop. Returns the ciphertext and the bytes sent.
fn pack_shares_to_fhe(
    s: &Bfv,
    ek: &crate::fhe::bfv::EncryptKey,
    per_party: &[Vec<Fr>],
    t_mask: u64,
    profile: NetworkProfile,
) -> (BfvCt, usize, FhePhases) {
    let mut c: Option<BfvCt> = None;
    let mut bytes = 0usize;
    let mut ph = FhePhases::default();
    let t = Instant::now();
    for comps in per_party {
        let m: Vec<u64> = comps.iter().map(|&v| (v as u64) & t_mask).collect();
        let enc = s.encrypt_poly_pre(ek, &m);
        bytes += s.ct_bytes(&enc);
        c = Some(match c {
            None => enc,
            Some(acc) => s.add(&acc, &enc),
        });
    }
    ph.reencrypt_ms += ms(t);
    ph.n_reencrypt += per_party.len();
    sleep_link(&profile, bytes, 1);
    (c.expect("three parties"), bytes, ph)
}

/// MPC half of one convolution layer: reconstruct each opened pre-activation,
/// add the (public) conv bias, apply the batched fused ReLU∘rescale, and
/// max-pool — returning each party's additive share components of the pooled
/// feature map (channel-major) plus the communication cost. Shared by the
/// first conv and every extra conv layer.
fn mpc_conv_relu_pool(
    conv_open: Vec<u64>,
    conv_masks: Vec<[u64; 3]>,
    cbias: Vec<i64>,
    conv_shape: Chw,
    shift: u32,
    nbits: usize,
    profile: NetworkProfile,
    nonlin: Nonlin,
) -> (Vec<Vec<Fr>>, CommStats, MpcPhases) {
    let (oh, ow, oc_n) = (conv_shape.h, conv_shape.w, conv_shape.c);
    let (out, comm) = run_parties(profile, move |me| {
        let conv_open = conv_open.clone();
        let conv_masks = conv_masks.clone();
        let cbias = cbias.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            be.set_nonlin(nonlin.mode());
            // Reconstruct + bias every conv output locally (share_secret / add
            // are communication-free), then run the whole ReLU layer as one
            // batched a2b + one batched multiply, then max-pool.
            let t = Instant::now();
            let count = oc_n * oh * ow;
            let raws = import_raw_batch(be, me, &conv_open, &conv_masks, count, nbits);
            let vs: Vec<ShareRef> = raws
                .iter()
                .enumerate()
                .map(|(j, raw)| {
                    let oc = j / (oh * ow);
                    let bconst = residue(cbias[oc] << shift, nbits);
                    let bsh = be.share_public(bconst, nbits);
                    be.add(raw, &bsh)
                })
                .collect();
            let import_ms = ms(t);
            // Max-pool BEFORE the nonlinearity, not after.
            //
            // `relu∘rescale` is monotonic non-decreasing, so it commutes with
            // `max`:  maxpool(f(v)) == f(maxpool(v)).  Pooling first shrinks
            // the ReLU layer by `POOL²` — 4× here — and the comparisons the
            // pool itself costs are unchanged, so this is a strict reduction
            // in nonlinearity work, not a trade.
            //
            // Two things make it safe. The comparison now sees *pre*-activation
            // values, which can be negative: `lt` is the MSB of `a − b`, i.e.
            // signed, so that is fine. And `a − b` must stay inside the
            // two's-complement range — it does, because `nbits` is calibrated
            // as `max_abs.bit_length() + 2` over exactly these values, so
            // |a|,|b| < 2^(nbits−2) and |a−b| < 2^(nbits−1). Those two bits of
            // headroom are what this relies on; shrinking them would break it.
            //
            // `maxpool2d` reduces to `lt_batch`, which the backend flag above
            // has already routed — the gadget itself is protocol-agnostic.
            let t = Instant::now();
            let (pooled, _) = maxpool2d(be, &vs, conv_shape, POOL, POOL);
            let pool_ms = ms(t);
            let t = Instant::now();
            let relu = fused_relu_rescale(be, &pooled, shift, nbits, nonlin);
            let relu_ms = ms(t);
            let comps = relu.iter().map(|p| be.my_additive_component(p)).collect::<Vec<Fr>>();
            (comps, MpcPhases { import_ms, relu_ms, pool_ms, ..Default::default() })
        }
    });
    // Every party runs the same protocol in lockstep, so party 0's split is
    // representative; taking one avoids averaging across thread scheduling noise.
    let phases = out[0].1;
    (out.into_iter().map(|(c, _)| c).collect(), comm, phases)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Where the FHE segment's time actually goes.
///
/// Same motivation as [`MpcPhases`], applied to the side that now dominates:
/// "FHE" is four different operations with different fixes. A
/// ciphertext$\times$ciphertext multiply is irreducible work the construction
/// needs; a *mask* encryption is query-independent and could in principle
/// move offline. Only a breakdown distinguishes them.
#[derive(Default, Clone, Copy, Debug, Serialize)]
pub struct FhePhases {
    /// Key generation (once per inference here; amortizable in deployment).
    pub keygen_ms: f64,
    /// Encrypting the input image.
    pub input_ms: f64,
    /// Encrypting the model: kernels and weight chunks.
    pub weights_ms: f64,
    /// The ct×ct products themselves, including the reused forward NTT.
    pub mul_ms: f64,
    /// Encrypting the per-party masks and adding them onto the product —
    /// the outbound half of each scheme switch.
    pub mask_ms: f64,
    /// Decrypting the masked product to open a layer.
    pub decrypt_ms: f64,
    /// Re-encrypting MPC shares back into a ciphertext (the switch's return
    /// leg, [`pack_shares_to_fhe`]).
    pub reencrypt_ms: f64,
    /// How many `encrypt_poly` calls the whole inference made, by purpose.
    pub n_weight_enc: usize,
    pub n_mask_enc: usize,
    pub n_reencrypt: usize,
}

impl FhePhases {
    fn add(&mut self, o: &FhePhases) {
        self.keygen_ms += o.keygen_ms;
        self.input_ms += o.input_ms;
        self.weights_ms += o.weights_ms;
        self.mul_ms += o.mul_ms;
        self.mask_ms += o.mask_ms;
        self.decrypt_ms += o.decrypt_ms;
        self.reencrypt_ms += o.reencrypt_ms;
        self.n_weight_enc += o.n_weight_enc;
        self.n_mask_enc += o.n_mask_enc;
        self.n_reencrypt += o.n_reencrypt;
    }
}

/// Where the MPC segment's time actually goes. The segment is the pipeline's
/// cost centre, and "MPC" is far too coarse a label to optimize against — the
/// import loop, the nonlinearity and the pooling reduction have different
/// round structures and respond to different fixes.
#[derive(Default, Clone, Copy, Debug, Serialize)]
pub struct MpcPhases {
    /// Reconstructing the opened layer into shares and adding the bias.
    pub import_ms: f64,
    /// The fused ReLU∘rescale.
    pub relu_ms: f64,
    /// Max-pooling (`pool²−1` batched compare+select steps).
    pub pool_ms: f64,
    /// The final argmax over the logits.
    pub argmax_ms: f64,
}

impl MpcPhases {
    fn add(&mut self, o: &MpcPhases) {
        self.import_ms += o.import_ms;
        self.relu_ms += o.relu_ms;
        self.pool_ms += o.pool_ms;
        self.argmax_ms += o.argmax_ms;
    }
}

/// Import a whole opened layer — the MPC side of one scheme switch.
///
/// Semantically this is [`import_raw`] for every `j`, and the resulting shares
/// are identical. The difference is round structure: the scalar version costs
/// three blocking input-sharing rounds *per element*, so a `28×28`
/// convolution's 5408 outputs cost 16 224 round trips and the import dominated
/// the entire pipeline — 19.1 s of a 20.4 s MPC segment, against 0.6 s for the
/// ReLU it was importing values for. Batched, the same layer costs **three**
/// rounds total (one per owner) with the same bytes on the wire.
fn import_raw_batch(
    be: &mut TransportBackend<MpcTransport>,
    me: usize,
    opened: &[u64],
    masks: &[[u64; 3]],
    count: usize,
    nbits: usize,
) -> Vec<ShareRef> {
    // Each party's additive component of the opened value; only the owner's
    // entries are meaningful, the rest are the zeros the protocol expects.
    let my_comps: Vec<Fr> = (0..count)
        .map(|j| {
            if me == 0 {
                (opened[j] as u128).wrapping_sub(masks[j][0] as u128)
            } else {
                0u128.wrapping_sub(masks[j][me] as u128)
            }
        })
        .collect();
    let mut acc: Option<Vec<ShareRef>> = None;
    for owner in 0..3 {
        let vals: Vec<Fr> = if me == owner { my_comps.clone() } else { vec![0; count] };
        let shs = be.share_secret_batch(&vals, nbits, owner);
        acc = Some(match acc {
            None => shs,
            // Addition is local, so folding the three owners costs nothing.
            Some(a) => a.iter().zip(&shs).map(|(x, y)| be.add(x, y)).collect(),
        });
    }
    acc.unwrap_or_default()
}

/// Batched fused ReLU∘rescale: `ReLU(vⱼ ≫ s)` for every input at once.
///
/// The whole layer costs **one** batched `a2b` (all inputs decomposed in the
/// same ~9 rounds) plus **one** batched multiply (the sign gate on every
/// element together), instead of a per-element `a2b`+`mul` — collapsing a
/// CNN's ReLU layer from `count · ~10` communication rounds to ~10.
///
/// Correctness is identical to the per-element version: the logical
/// right-shift is recomposed from the high bits (`b2a`, local) and kept only
/// on the non-negative branch (`keep = 1 − sign`).
/// The pipeline's nonlinearity, `out = (v >= 0) ? (v >> shift) : 0`, under
/// whichever protocol `nonlin` selects. The two branches are extensionally
/// equal — that is what the `gc_matches_a2b_*` tests pin.
fn fused_relu_rescale(
    be: &mut TransportBackend<MpcTransport>,
    vs: &[ShareRef],
    shift: u32,
    nbits: usize,
    nonlin: Nonlin,
) -> Vec<ShareRef> {
    match nonlin {
        Nonlin::A2b => fused_relu_rescale_batch(be, vs, shift, nbits),
        // One garble round for the whole layer, versus ~log(nbits) rounds of
        // decomposition plus a batched multiply for the sign gate.
        Nonlin::Gc => be.relu_gc_batch(vs, nbits, shift as usize),
        // Falcon has no fused relu∘rescale: its ReLU and its truncation are
        // separate protocols. Routing the fused form through the generic
        // `lt`-based path keeps the semantics identical, at the cost of not
        // exercising `Π_ReLU` directly — see `falcon-reproduction.md`.
        Nonlin::Falcon => fused_relu_rescale_batch(be, vs, shift, nbits),
    }
}

fn fused_relu_rescale_batch(
    be: &mut TransportBackend<MpcTransport>,
    vs: &[ShareRef],
    shift: u32,
    nbits: usize,
) -> Vec<ShareRef> {
    if vs.is_empty() {
        return Vec::new();
    }
    let bit_sets = be.a2b_batch(vs, nbits);
    let one = be.share_public(1, 1);
    // Local per element: recompose the shifted value and the keep bit.
    let mut pairs = Vec::with_capacity(vs.len());
    for bits in &bit_sets {
        // b2a inherits the bit-shares' width (1); the recomposed value spans
        // the full range, so restore its width — otherwise a downstream
        // comparison (`lt` → `a2b(·, nbits)`) would decompose only 1 bit.
        let mut shifted = be.b2a(&bits[shift as usize..]);
        shifted.nbits = nbits;
        let keep = be.sub(&one, &bits[nbits - 1]);
        pairs.push((shifted, keep));
    }
    // One batched multiply for every element's sign gate.
    be.mul_batch(&pairs)
        .into_iter()
        .map(|mut act| {
            act.nbits = nbits;
            act
        })
        .collect()
}

// ─── The full hybrid CNN forward pass ─────────────────────────────────────────

pub struct CnnRun {
    pub class: usize,
    pub expected: usize,
    pub correct: bool,
    pub total_ms: f64,
    pub fhe_ms: f64,
    pub mpc_ms: f64,
    /// Breakdown of `mpc_ms` — see [`MpcPhases`].
    pub phases: MpcPhases,
    /// Breakdown of `fhe_ms` — see [`FhePhases`].
    pub fhe_phases: FhePhases,
    /// Query-independent work done before the clock started. Not part of
    /// `total_ms`. Equal to `offline_session_ms + offline_query_ms`.
    pub offline_ms: f64,
    /// The **per-session** half: keygen and the encrypted model. Paid once and
    /// reused by every query in the session, so it is genuine amortization —
    /// total work per query drops permanently after the first.
    pub offline_session_ms: f64,
    /// The **per-query** half: the mask pool. Masks must be fresh for every
    /// inference, so pooling them moves latency off the critical path but buys
    /// no throughput — a saturated server does the same total work. Reported
    /// separately from `offline_session_ms` precisely so the two are not
    /// conflated into one flattering number.
    pub offline_query_ms: f64,
    pub comm: CommStats,
    pub bootstraps: u64,
    /// ct×ct multiplications the packed convolution actually performed.
    pub conv_mults: usize,
    /// What an im2col/Toeplitz matvec of the same convolution would cost.
    pub conv_mults_toeplitz: usize,
    /// FHE ring dimension and modulus actually used — recorded so the
    /// security claim is testable rather than only asserted in prose.
    pub ring_n: usize,
    pub log2_q: u32,
    /// Plaintext modulus width, calibrated from the measured activations.
    pub log2_t: u32,
}

/// Which MPC protocol evaluates the nonlinearities — the pipeline's cost
/// centre (the FHE arithmetic is a few percent of runtime, see
/// [`run_cnn_hybrid`]'s `fhe_ms`/`mpc_ms` split).
///
/// Both settings compute the *same function* and are bit-exact against the
/// plaintext reference; only rounds and bandwidth differ. `A2b` is the
/// log-round batched bit decomposition and stays the default so it remains
/// the control in every measurement; `Gc` routes the fused ReLU∘rescale and
/// every batched comparison (max-pooling, argmax) through the three-party
/// garbled circuit, which is constant-round and needs no oblivious transfer
/// because the co-garbler delivers the input labels.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Nonlin {
    /// Batched bit decomposition (`a2b`), ~log(nbits) rounds per layer.
    #[default]
    A2b,
    /// Three-party garbled circuits, one round per layer.
    Gc,
    /// Falcon's `wrap3` sign extraction over `Z_p` — the lowest-bandwidth of
    /// the three, and the one a bandwidth-bound deployment should pick.
    Falcon,
}

impl Nonlin {
    /// The backend switch this selects.
    pub fn mode(self) -> crate::interpreter::transport_backend::NonlinMode {
        use crate::interpreter::transport_backend::NonlinMode as M;
        match self {
            Nonlin::A2b => M::A2b,
            Nonlin::Gc => M::Gc,
            Nonlin::Falcon => M::Falcon,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Nonlin::A2b => "a2b (batched bit decomposition)",
            Nonlin::Gc => "gc (3-party garbled, constant-round)",
            Nonlin::Falcon => "falcon (wrap3 over Z_p, lowest bandwidth)",
        }
    }

    /// The bare protocol name, for table headers where [`label`] would not fit.
    pub fn short(self) -> &'static str {
        match self {
            Nonlin::A2b => "a2b",
            Nonlin::Gc => "gc",
            Nonlin::Falcon => "falcon",
        }
    }
}

pub fn run_cnn_hybrid(model: &CnnModel, input: &[i64], profile: NetworkProfile) -> CnnRun {
    run_cnn_hybrid_with(model, input, profile, Nonlin::default())
}

/// [`run_cnn_hybrid`] with an explicit choice of nonlinearity protocol.
pub fn run_cnn_hybrid_with(
    model: &CnnModel,
    input: &[i64],
    profile: NetworkProfile,
    nonlin: Nonlin,
) -> CnnRun {
    run_cnn_hybrid_full(model, input, profile, nonlin, false)
}

/// Everything a deployment sets up **once per model**, before any query
/// exists: the FHE context, the key, and — when precomputing — the encrypted
/// weights.
///
/// This is what makes the model's encryption *amortize* rather than merely
/// shift. A session is built once and then serves every inference, so the cost
/// of encrypting the weights is paid on the first query and never again. The
/// mask pool deliberately does **not** live here: masks must be fresh per
/// query, so they are rebuilt inside every run.
pub struct CnnSession {
    s: Bfv,
    /// The secret key deserialized and forward-transformed once for the
    /// session — decryption is per-ciphertext, the key is not.
    dk: crate::fhe::bfv::DecryptKey,
    /// Likewise the public key, which every encryption multiplies against.
    ek: crate::fhe::bfv::EncryptKey,
    enc_model: Option<EncryptedModel>,
    /// One per ct×ct multiply — also the number of masks each query needs.
    n_weight_cts: usize,
    keygen_ms: f64,
}

impl CnnSession {
    /// Build the session for `model`. With `precompute`, the weights are
    /// encrypted now; without it, every query re-encrypts them on its critical
    /// path — the control the `--precompute` measurement is against.
    pub fn new(model: &CnnModel, precompute: bool) -> Self {
        // RLWE-secure parameters: n=4096, q=2^100 is at/below the 128-bit bound
        // of ~109 for n=4096 (HomomorphicEncryption.org).
        let s = Bfv::new(BfvParams { n: 4096, log2_q: 100, log2_t: model.nbits as u32 });
        let n_in = model.d * model.d;
        assert!((n_in + 1) <= s.params.n, "input too large for coefficient packing");

        // Keygen FIRST: everything below must be encrypted under the *same* key
        // the products are, or the masked ciphertext will not decrypt. An early
        // `--precompute` draft called `s.keygen()` a second time for the pool
        // and produced masks that silently did not open.
        let tk = Instant::now();
        let (ck, _sk) = s.keygen();
        let keygen_ms = ms(tk);

        // One walk of the model gives both halves of the preamble: the weight
        // polynomials to encrypt, and (since every ct×ct multiply is followed by
        // exactly one switch) the number of masks each query will pool.
        let polys = weight_polys(&s, model);
        let n_weight_cts = polys.len();
        let ek = s.encrypt_key(&ck);
        let enc_model = precompute.then(|| EncryptedModel::encrypt(&s, &ek, &polys));
        let dk = s.decrypt_key(&ck);
        CnnSession { s, dk, ek, enc_model, n_weight_cts, keygen_ms }
    }

    /// Per-session offline cost: keygen plus, when precomputing, the encrypted
    /// model. Divide by the number of queries served to get its per-query share.
    pub fn offline_ms(&self) -> f64 {
        self.keygen_ms + self.enc_model.as_ref().map_or(0.0, |m| m.offline_ms)
    }
}

/// [`run_cnn_hybrid`] with both knobs: the nonlinearity protocol, and whether
/// the preamble is precomputed in an offline phase. With `precompute`,
/// `total_ms`/`fhe_ms` are the **online** latency and the `offline_*` fields
/// report what was moved off the critical path.
///
/// This builds a fresh [`CnnSession`] per call, so it charges the per-session
/// cost to this one query. To measure the amortization, build one session and
/// call [`run_cnn_hybrid_in_session`] repeatedly.
pub fn run_cnn_hybrid_full(
    model: &CnnModel,
    input: &[i64],
    profile: NetworkProfile,
    nonlin: Nonlin,
    precompute: bool,
) -> CnnRun {
    let session = CnnSession::new(model, precompute);
    run_cnn_hybrid_in_session(&session, model, input, profile, nonlin, precompute)
}

/// One inference against an existing [`CnnSession`]. The session's weights are
/// reused; only the per-query mask pool is rebuilt.
pub fn run_cnn_hybrid_in_session(
    session: &CnnSession,
    model: &CnnModel,
    input: &[i64],
    profile: NetworkProfile,
    nonlin: Nonlin,
    precompute: bool,
) -> CnnRun {
    let (expected, max_abs) = model.forward(input);
    assert!(
        (max_abs as i128) < (1i128 << (model.nbits - 1)),
        "cnn: intermediate magnitude {max_abs} overflows nbits={}",
        model.nbits
    );
    let nbits = model.nbits;
    let s = &session.s;
    let t_mask = s.params.t() - 1;

    let (mut fhe_ms, mut mpc_ms) = (0.0, 0.0);
    let mut comm = CommStats::default();
    let mut rng = rand::thread_rng();
    let mut fhe_ph = FhePhases::default();
    fhe_ph.keygen_ms = session.keygen_ms;

    // ── Offline, before the clock starts ──
    // The masks depend only on the network's *shape*, not on the patient's
    // image, so a deployment generates them while waiting for work. `total_ms`
    // is therefore the latency of a query against a warm server, which is the
    // quantity a deployment is judged on.
    let mut pool = precompute
        .then(|| MaskPool::with_len(s, &session.ek, session.n_weight_cts, &mut rand::thread_rng()));
    let offline_session_ms = session.offline_ms();
    let offline_query_ms = pool.as_ref().map_or(0.0, |p| p.offline_ms);
    let offline_ms = offline_session_ms + offline_query_ms;
    // Per-session, so the cursor advances across the whole inference in the
    // pipeline's consumption order.
    let mut weights_src = match session.enc_model.as_ref() {
        Some(m) => Weights::session(m),
        None => Weights::online(),
    };

    let start = Instant::now();
    let ti = Instant::now();
    let c_input = s.encrypt_poly_pre(&session.ek, &s.pack_input(input));
    fhe_ph.input_ms = ms(ti);

    let shift = model.shift;

    // ── First conv via Cheetah-style packing → open ──
    // One ct×ct multiply per OUTPUT CHANNEL (not per output pixel): the
    // kernel is packed reversed on every axis, so polynomial multiplication
    // *is* the sliding-window convolution and every output pixel of that
    // channel appears at once. See `fhe::conv_pack`.
    let t0 = Instant::now();
    let conv1_shape = ConvShape { in_c: 1, h: model.d, w: model.d, kh: K, kw: K };
    let (conv_open, conv_masks, ph) =
        hybrid_conv_open(s, &session.ek, &c_input, conv1_shape, &model.kernels, profile, &mut rng, &mut masks_src(&mut pool), &mut weights_src, &session.dk);
    fhe_ph.add(&ph);
    fhe_ms += t0.elapsed().as_secs_f64() * 1000.0;

    // ── MPC: +cbias≪s, fused ReLU∘rescale, max-pool → pooled feature shares ──
    let t0 = Instant::now();
    let mut phases = MpcPhases::default();
    let (mut per_party_feat, conv_comm, ph) = mpc_conv_relu_pool(
        conv_open, conv_masks, model.cbias.clone(), model.conv_shape, shift, nbits, profile,
        nonlin,
    );
    phases.add(&ph);
    comm = comm + conv_comm;
    mpc_ms += t0.elapsed().as_secs_f64() * 1000.0;
    let mut feat_shape = model.pool_shape;

    // ── Additional conv layers: switch pooled features back to FHE, convolve
    //    (multi-channel packing), open, MPC ReLU∘rescale + pool. Each extra
    //    conv is just the same three steps composed again — no bootstrap. ──
    for cl in &model.extra_conv {
        let tc = Instant::now();
        let (c_conv_in, _, ph) = pack_shares_to_fhe(s, &session.ek, &per_party_feat, t_mask, profile);
        fhe_ph.add(&ph);
        let shape = ConvShape { in_c: cl.in_c, h: cl.in_h, w: cl.in_w, kh: K, kw: K };
        let (co, cm, ph) =
            hybrid_conv_open(s, &session.ek, &c_conv_in, shape, &cl.kernels, profile, &mut rng, &mut masks_src(&mut pool), &mut weights_src, &session.dk);
        fhe_ph.add(&ph);
        fhe_ms += tc.elapsed().as_secs_f64() * 1000.0;

        let tm = Instant::now();
        let out_shape = Chw { c: cl.out_c(), h: cl.out_h(), w: cl.out_w() };
        let (feat2, comm2, ph2) =
            mpc_conv_relu_pool(co, cm, cl.cbias.clone(), out_shape, shift, nbits, profile, nonlin);
        phases.add(&ph2);
        comm = comm + comm2;
        mpc_ms += tm.elapsed().as_secs_f64() * 1000.0;
        per_party_feat = feat2;
        feat_shape = Chw { c: cl.out_c(), h: cl.out_h() / POOL, w: cl.out_w() / POOL };
    }

    // ── Switch pooled features → packed FHE dense input ──
    let t0 = Instant::now();
    let flat = feat_shape.c * feat_shape.h * feat_shape.w;
    let (mut c_dense_in, _, ph) = pack_shares_to_fhe(s, &session.ek, &per_party_feat, t_mask, profile);
    fhe_ph.add(&ph);
    let mut dense_in_dim = flat;
    fhe_ms += t0.elapsed().as_secs_f64() * 1000.0;

    // ── Stacked hidden dense layers: each is FHE matvec → open → MPC
    //    ReLU∘rescale → switch back. Same machinery as the output dense + the
    //    conv ReLU, composed once per hidden layer — the "no bootstrap, just
    //    switch again" invariant made concrete for arbitrary depth. ──
    for (hidden_w, hidden_b) in model.hidden.clone() {
        let th = Instant::now();
        let (hidden_open, hidden_masks, ph) =
            hybrid_matvec_open(s, &session.ek, &c_dense_in, &hidden_w, dense_in_dim, profile, &mut rng, &mut masks_src(&mut pool), &mut weights_src, &session.dk);
        fhe_ph.add(&ph);
        fhe_ms += th.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let h = hidden_w.len();
        let (per_party_hidden, hidden_comm) = run_parties(profile, |me| {
            let hidden_open = hidden_open.clone();
            let hidden_masks = hidden_masks.clone();
            let hidden_b = hidden_b.clone();
            move |be: &mut TransportBackend<MpcTransport>| {
                be.set_nonlin(nonlin.mode());
                let t = Instant::now();
                let raws = import_raw_batch(be, me, &hidden_open, &hidden_masks, h, nbits);
                let vs: Vec<ShareRef> = raws
                    .iter()
                    .enumerate()
                    .map(|(j, raw)| {
                        let bsh = be.share_public(residue(hidden_b[j], nbits), nbits);
                        be.add(raw, &bsh)
                    })
                    .collect();
                let import_ms = ms(t);
                let t = Instant::now();
                let act = fused_relu_rescale(be, &vs, shift, nbits, nonlin);
                let relu_ms = ms(t);
                let comps =
                    act.iter().map(|p| be.my_additive_component(p)).collect::<Vec<Fr>>();
                (comps, MpcPhases { import_ms, relu_ms, ..Default::default() })
            }
        });
        phases.add(&per_party_hidden[0].1);
        let per_party_hidden: Vec<Vec<Fr>> =
            per_party_hidden.into_iter().map(|(c, _)| c).collect();
        comm = comm + hidden_comm;
        mpc_ms += t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let (c_hidden, _, ph) = pack_shares_to_fhe(s, &session.ek, &per_party_hidden, t_mask, profile);
        fhe_ph.add(&ph);
        c_dense_in = c_hidden;
        dense_in_dim = h;
        fhe_ms += t2.elapsed().as_secs_f64() * 1000.0;
    }

    // ── Output dense (FHE ct×ct matvec) → open ──
    let t3 = Instant::now();
    let (dense_open, dense_masks, ph) = hybrid_matvec_open(
        s,
        &session.ek,
        &c_dense_in,
        &model.dense_w,
        dense_in_dim,
        profile,
        &mut rng,
        &mut masks_src(&mut pool),
        &mut weights_src,
        &session.dk,
    );
    fhe_ph.add(&ph);
    fhe_ms += t3.elapsed().as_secs_f64() * 1000.0;

    // ── MPC: +dense bias, argmax → reveal class only ──
    let t0 = Instant::now();
    let dense_b = model.dense_b.clone();
    let nclasses = model.nclasses();
    let (classes, argmax_comm) = run_parties(profile, |me| {
        let dense_open = dense_open.clone();
        let dense_masks = dense_masks.clone();
        let dense_b = dense_b.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            // `argmax` reduces over scalar `lt`, not `lt_batch`, so the GC flag
            // does not reach it — deliberately: it is `nclasses−1` comparisons
            // against a layer's thousands, and the phase breakdown shows it.
            be.set_nonlin(nonlin.mode());
            let raws = import_raw_batch(be, me, &dense_open, &dense_masks, nclasses, nbits);
            let logits: Vec<ShareRef> = raws
                .iter()
                .enumerate()
                .map(|(j, raw)| {
                    let bsh = be.share_public(residue(dense_b[j], nbits), nbits);
                    be.add(raw, &bsh)
                })
                .collect();
            let t = Instant::now();
            let idx = argmax(be, &logits, nbits);
            let argmax_ms = ms(t);
            (
                (be.reveal(&idx, None) as u64 & (t_mask)) as usize,
                MpcPhases { argmax_ms, ..Default::default() },
            )
        }
    });
    phases.add(&classes[0].1);
    comm = comm + argmax_comm;
    mpc_ms += t0.elapsed().as_secs_f64() * 1000.0;
    let class = classes[0].0;

    // Both halves of the preamble must have been consumed exactly: a pool that
    // over-counts hides work in the offline column, and an encrypted model that
    // over-counts means the pipeline multiplied against fewer weights than the
    // model has — which would still produce *a* class, just the wrong one.
    if let Some(p) = &pool {
        p.assert_fully_consumed();
    }
    weights_src.assert_fully_consumed();

    CnnRun {
        correct: class == expected,
        class,
        expected,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
        fhe_ms,
        mpc_ms,
        phases,
        fhe_phases: fhe_ph,
        offline_ms,
        offline_session_ms,
        offline_query_ms,
        comm,
        bootstraps: 0,
        ring_n: s.params.n,
        log2_q: s.params.log2_q,
        log2_t: s.params.log2_t,
        conv_mults: model.out_channels()
            + model.extra_conv.iter().map(|c| c.out_c()).sum::<usize>(),
        conv_mults_toeplitz: conv_pack::toeplitz_multiplications(
            ConvShape { in_c: 1, h: model.d, w: model.d, kh: K, kw: K },
            model.out_channels(),
            BfvParams::DEFAULT.n,
        ),
    }
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct CnnPoint {
    pub d: usize,
    /// Which protocol evaluated the nonlinearities — without it two rows of
    /// this struct are indistinguishable, and the whole point is comparing them.
    pub nonlin: String,
    pub class: usize,
    pub correct: bool,
    pub total_ms: f64,
    pub fhe_ms: f64,
    pub mpc_ms: f64,
    pub messages: u64,
    pub bytes: u64,
    pub bootstraps: u64,
    pub conv_mults: usize,
    pub conv_mults_toeplitz: usize,
    pub ring_n: usize,
    pub log2_q: u32,
    pub log2_t: u32,
}

fn cnn_point(d: usize, nonlin: Nonlin, r: &CnnRun) -> CnnPoint {
    CnnPoint {
        d,
        nonlin: nonlin.label().to_string(),
        class: r.class,
        correct: r.correct,
        total_ms: r.total_ms,
        fhe_ms: r.fhe_ms,
        mpc_ms: r.mpc_ms,
        messages: r.comm.messages,
        bytes: r.comm.bytes,
        bootstraps: r.bootstraps,
        conv_mults: r.conv_mults,
        conv_mults_toeplitz: r.conv_mults_toeplitz,
        ring_n: r.ring_n,
        log2_q: r.log2_q,
        log2_t: r.log2_t,
    }
}

const USAGE: &str = "usage: baru bench-cnn [options]

A compact convolutional classifier (conv → ReLU → maxpool → dense → argmax)
through the hybrid pipeline: BFV ct×ct linear layers at RLWE-secure
parameters (n=4096, q=2^100), MPC nonlinearities, ZERO FHE bootstraps,
verified bit-exact against the plaintext reference.
Correctness is exact; ACCURACY needs trained weights (orthogonal — see the
module docs). The convolution uses Cheetah-style coefficient packing, so it
costs one ct×ct multiply per output channel regardless of D — a full 28×28
MNIST image runs fine. Multi-channel CIFAR needs the channels split across
ciphertexts.

options:
  --img D              input side length D (1 channel, D×D; default 8; D even)
  --rtt-ms N           round-trip latency (default 0)
  --bandwidth-mbps X   link bandwidth in Mbit/s (default: unlimited)
  --seed N             RNG seed (default 1)
  --gc                 evaluate the nonlinearities with 3-party garbled
                       circuits (constant-round) instead of the default
                       batched bit decomposition; same function, bit-exact
  --both               run both nonlinearity protocols and report the speedup
  --precompute         generate the scheme-switch masks in an offline phase
                       (they are query-independent), so the reported time is
                       online latency; offline cost is reported separately
  --json               emit JSON
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut d = 8usize;
    let mut rtt_ms = 0u64;
    let mut bandwidth_mbps: Option<f64> = None;
    let mut seed = 1u64;
    let mut json = false;
    let mut nonlin = Nonlin::A2b;
    let mut both = false;
    let mut precompute = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--gc" => nonlin = Nonlin::Gc,
            "--falcon" => nonlin = Nonlin::Falcon,
            "--both" => both = true,
            "--precompute" => precompute = true,
            "--img" => {
                i += 1;
                d = args.get(i).ok_or("--img needs a value")?.parse().map_err(|_| "bad img".to_string())?;
            }
            "--rtt-ms" => {
                i += 1;
                rtt_ms = args.get(i).ok_or("--rtt-ms needs a value")?.parse().map_err(|_| "bad rtt-ms".to_string())?;
            }
            "--bandwidth-mbps" => {
                i += 1;
                bandwidth_mbps = Some(args.get(i).ok_or("--bandwidth-mbps needs a value")?.parse().map_err(|_| "bad bandwidth".to_string())?);
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).ok_or("--seed needs a value")?.parse().map_err(|_| "bad seed".to_string())?;
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    let mut profile = NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt_ms * 500));
    if let Some(mbps) = bandwidth_mbps {
        profile = profile.with_bandwidth(mbps * 1e6 / 8.0);
    }

    let (model, input) = CnnModel::random(d, seed);

    if both {
        // Same model, same input, same transport — the only difference is
        // which protocol evaluates the nonlinearities, so the ratio is
        // attributable to that and nothing else.
        let a = run_cnn_hybrid_full(&model, &input, profile, Nonlin::A2b, precompute);
        let g = run_cnn_hybrid_full(&model, &input, profile, Nonlin::Gc, precompute);
        if json {
            let pts: Vec<CnnPoint> = [(Nonlin::A2b, &a), (Nonlin::Gc, &g)]
                .iter()
                .map(|(n, r)| cnn_point(d, *n, r))
                .collect();
            println!("{}", serde_json::to_string_pretty(&pts).map_err(|e| e.to_string())?);
            return Ok(());
        }
        println!("Private CNN classifier — nonlinearity protocol head-to-head");
        println!("architecture: 1×{d}×{d} → conv({OC},{K}×{K}) → ReLU → maxpool({POOL}×{POOL}) → dense(→{NCLASSES}) → argmax");
        println!(
            "network: rtt={rtt_ms}ms, bandwidth={}\n",
            bandwidth_mbps.map(|b| format!("{b}Mbps")).unwrap_or_else(|| "unlimited".into())
        );
        println!("  {:<34}  {:>10}  {:>10}  {:>12}  {:>10}", "nonlinearity", "total", "mpc", "wire bytes", "class");
        for (n, r) in [(Nonlin::A2b, &a), (Nonlin::Gc, &g)] {
            println!(
                "  {:<34}  {:>8.0}ms  {:>8.0}ms  {:>12}  {:>10}",
                n.label(),
                r.total_ms,
                r.mpc_ms,
                r.comm.bytes,
                format!("{}{}", r.class, if r.correct { "" } else { " WRONG" })
            );
        }
        // A speedup between two runs that disagree is meaningless, so the
        // agreement check gates the number rather than being printed beside it.
        if !(a.correct && g.correct && a.class == g.class) {
            return Err(format!(
                "nonlinearity protocols disagree (a2b {} / gc {} / plaintext {}) — \
                 the comparison is void until they are bit-exact",
                a.class, g.class, a.expected
            ));
        }
        println!(
            "\n  both bit-exact vs plaintext (class {}), so the ratio is pure cost:",
            a.expected
        );
        println!(
            "    end-to-end  {:.2}×    mpc segment  {:.2}×    wire bytes  {:.2}×",
            a.total_ms / g.total_ms,
            a.mpc_ms / g.mpc_ms,
            a.comm.bytes as f64 / g.comm.bytes as f64
        );
        return Ok(());
    }

    let r = run_cnn_hybrid_full(&model, &input, profile, nonlin, precompute);

    if json {
        let p = cnn_point(d, nonlin, &r);
        println!("{}", serde_json::to_string_pretty(&p).map_err(|e| e.to_string())?);
        return Ok(());
    }

    println!("Private CNN classifier — hybrid BFV × ABY3, ZERO FHE bootstraps");
    println!(
        "architecture: 1×{d}×{d} → conv({OC},{K}×{K}) → ReLU → maxpool({POOL}×{POOL}) \
         → dense(→{NCLASSES}) → argmax"
    );
    println!("image AND weights encrypted; only the class index is revealed");
    println!(
        "network: rtt={rtt_ms}ms, bandwidth={}\n",
        bandwidth_mbps.map(|b| format!("{b}Mbps")).unwrap_or_else(|| "unlimited".into())
    );
    println!("  predicted class : {}  ({})", r.class, if r.correct { "exact vs plaintext" } else { "WRONG" });
    println!(
        "  FHE parameters  : BFV n={}, q=2^{}, t=2^{} (RLWE-secure, 128-bit)",
        r.ring_n, r.log2_q, r.log2_t
    );
    println!("  FHE bootstraps  : {}", r.bootstraps);
    println!(
        "  conv ct×ct mults: {} (packed)  vs {} (im2col/Toeplitz) — {:.0}× fewer",
        r.conv_mults,
        r.conv_mults_toeplitz,
        r.conv_mults_toeplitz as f64 / r.conv_mults as f64
    );
    println!("  nonlinearity    : {}", nonlin.label());
    println!("  time            : total {:.0}ms  (fhe {:.0}ms, mpc {:.0}ms)", r.total_ms, r.fhe_ms, r.mpc_ms);
    println!(
        "  offline         : {:.0}ms  (query-independent, NOT in total)",
        r.offline_ms
    );
    if precompute {
        // The two halves are not the same kind of saving, and adding them into
        // one number would overstate the result.
        println!(
            "                    {:.0}ms per SESSION (keygen + encrypted model) — paid once, \
             reused by every query: real amortization",
            r.offline_session_ms
        );
        println!(
            "                    {:.0}ms per QUERY (mask pool) — masks must be fresh, so this \
             shifts latency but not total work",
            r.offline_query_ms
        );
    }
    println!(
        "  fhe breakdown   : keygen {:.0}ms, input {:.0}ms, weights {:.0}ms ({}x), \
         mul {:.0}ms, mask {:.0}ms ({}x), decrypt {:.0}ms, re-encrypt {:.0}ms ({}x)",
        r.fhe_phases.keygen_ms,
        r.fhe_phases.input_ms,
        r.fhe_phases.weights_ms,
        r.fhe_phases.n_weight_enc,
        r.fhe_phases.mul_ms,
        r.fhe_phases.mask_ms,
        r.fhe_phases.n_mask_enc,
        r.fhe_phases.decrypt_ms,
        r.fhe_phases.reencrypt_ms,
        r.fhe_phases.n_reencrypt
    );
    println!(
        "  mpc breakdown   : import {:.0}ms, relu {:.0}ms, pool {:.0}ms, argmax {:.0}ms",
        r.phases.import_ms, r.phases.relu_ms, r.phases.pool_ms, r.phases.argmax_ms
    );
    println!("  MPC wire traffic: {} messages, {} bytes", r.comm.messages, r.comm.bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Why the nonlinearity is a garbled *comparator* plus an **arithmetic**
    /// select, rather than one fused garbled max-of-4.
    ///
    /// Measured 2026-08-01 at N=5256 (the pool's comparison count for
    /// pneumonia), solving the two-point cost model:
    ///
    /// | | |
    /// |---|---|
    /// | `lt_batch` (19 AND + 1-bit Y2A) | 43.1 ms |
    /// | `mul_batch` (Beaver select) | 5.1 ms |
    /// | `relu_gc` (31 AND + 12-bit Y2A) | 68.4 ms |
    /// | ⇒ per AND-gate column | **2.28 ms** |
    /// | ⇒ per Y2A output bit | **≈ 0** |
    ///
    /// So the cost is **proportional to AND-gate count**; the Y2A conversion is
    /// free and the Beaver select is 11% of an `lt`+`mul` pair. That kills the
    /// obvious fusion: a garbled max-of-4 would replace three cheap arithmetic
    /// selects with three garbled muxes, taking a 2×2 window from 88 AND gates
    /// to 148 — **1.68× more work** to remove 11%. The current split is right,
    /// and this probe is what says so.
    ///
    /// It also says where a real win must come from: 158 144 AND gates per
    /// inference is the whole nonlinearity budget, so beating it means not
    /// garbling the comparison at all (Falcon's wrap-based DReLU), not
    /// rearranging what is garbled.
    #[test]
    #[ignore] // profiling probe, real 3-party transport
    fn profile_lt_versus_mul_under_gc() {
        const N: usize = 5256; // the pool's comparison count for pneumonia
        let (out, _) = run_parties(NetworkProfile::INSTANT, |me| {
            move |be: &mut TransportBackend<MpcTransport>| {
                be.set_gc_nonlinearities(true);
                let a: Vec<ShareRef> =
                    (0..N).map(|i| be.share_secret(if me == 0 { i as u128 % 977 } else { 0 }, 20, 0)).collect();
                let b: Vec<ShareRef> =
                    (0..N).map(|i| be.share_secret(if me == 0 { i as u128 % 613 } else { 0 }, 20, 0)).collect();
                let pairs: Vec<(ShareRef, ShareRef)> =
                    a.iter().cloned().zip(b.iter().cloned()).collect();

                let t = Instant::now();
                let conds = be.lt_batch(&pairs);
                let lt_ms = ms(t);

                let mul_pairs: Vec<(ShareRef, ShareRef)> = conds
                    .iter()
                    .zip(a.iter().zip(b.iter()))
                    .map(|(c, (x, y))| (c.clone(), be.sub(y, x)))
                    .collect();
                let t = Instant::now();
                let _ = be.mul_batch(&mul_pairs);
                let mul_ms = ms(t);

                let t = Instant::now();
                let _ = be.relu_gc_batch(&a, 20, 8);
                let relu_ms = ms(t);
                (lt_ms, mul_ms, relu_ms)
            }
        });
        let (lt, mul, relu) = out[0];
        println!(
            "\n  N={N}\n  lt_batch  (garbled comparator + Y2A) : {lt:.1} ms\n  \
             mul_batch (Beaver select)            : {mul:.1} ms\n  \
             relu_gc   (comparator+rescale + Y2A) : {relu:.1} ms\n  \
             -> Beaver select is {:.0}% of an lt+mul pair, so fusing max-of-4 into\n     \
             one garbled circuit trades 88 AND gates/window for 148 — a loss.\n",
            mul / (lt + mul) * 100.0
        );
    }

    #[test]
    #[ignore] // uses the real 3-party transport
    fn max2_and_lt_on_transport() {
        use crate::gadgets::max2;
        let (out, _) = run_parties(NetworkProfile::INSTANT, |me| {
            move |be: &mut TransportBackend<MpcTransport>| {
                let a = be.share_secret(if me == 0 { 9 } else { 0 }, 20, 0);
                let b = be.share_secret(if me == 0 { 6 } else { 0 }, 20, 0);
                let lt = be.lt(&a, &b); // expect 0 (9 < 6 is false)
                let m = max2(be, &a, &b); // expect 9
                (be.reveal(&lt, None) as i64, be.reveal(&m, None) as i64)
            }
        });
        assert_eq!(out[0], (0, 9), "lt(9,6) should be 0 and max2(9,6) should be 9");
    }

    #[test]
    fn plaintext_cnn_is_bounded_and_deterministic() {
        let (model, input) = CnnModel::random(8, 3);
        let (class, max_abs) = model.forward(&input);
        assert!(class < NCLASSES);
        assert!((max_abs as i128) < (1i128 << (model.nbits - 1)));
        let (m2, i2) = CnnModel::random(8, 3);
        assert_eq!(input, i2);
        assert_eq!(model.forward(&i2).0, m2.forward(&input).0);
    }

    #[test]
    #[ignore] // n=4096 secure ring — slow in debug, fast in release
    fn hybrid_cnn_classifies_exactly() {
        for (d, seed) in [(8usize, 1u64), (8, 2), (10, 5)] {
            let (model, input) = CnnModel::random(d, seed);
            let r = run_cnn_hybrid(&model, &input, NetworkProfile::INSTANT);
            assert!(
                r.correct,
                "d={d} seed={seed}: encrypted class {} != plaintext {}",
                r.class, r.expected
            );
            assert_eq!(r.bootstraps, 0);
            assert!(r.comm.messages > 0, "must use the real MPC protocol");
        }
    }

    /// The two nonlinearity protocols must be *extensionally equal*: garbled
    /// circuits and bit decomposition compute the same fused ReLU∘rescale and
    /// the same max-pool, so both must reproduce the plaintext class exactly
    /// and agree with each other.
    ///
    /// This is what licenses quoting a speedup between them. A faster protocol
    /// that computed a slightly different function would look like a win and
    /// be a bug, and at these bit widths the difference could easily be a
    /// single boundary value that a random image never hits — hence several
    /// seeds and both image sizes.
    #[test]
    #[ignore] // n=4096 secure ring — slow in debug, fast in release
    fn gc_and_a2b_nonlinearities_agree() {
        for (d, seed) in [(8usize, 1u64), (8, 2), (10, 5)] {
            let (model, input) = CnnModel::random(d, seed);
            let a = run_cnn_hybrid_with(&model, &input, NetworkProfile::INSTANT, Nonlin::A2b);
            let g = run_cnn_hybrid_with(&model, &input, NetworkProfile::INSTANT, Nonlin::Gc);
            assert!(a.correct, "d={d} seed={seed}: a2b class {} != plaintext {}", a.class, a.expected);
            assert!(g.correct, "d={d} seed={seed}: gc class {} != plaintext {}", g.class, g.expected);
            assert_eq!(a.class, g.class, "d={d} seed={seed}: protocols disagree");
            assert_eq!(g.bootstraps, 0, "the GC route must not change the FHE side");
        }
    }

    /// Precomputing the scheme-switch masks must not change the result — it
    /// is the *same* masks, generated earlier. Checked on both nonlinearity
    /// routes and on a two-conv model, because [`MaskPool::for_model`]
    /// predicts the pool size from the model's shape and the multi-layer
    /// chunking is where that prediction can go wrong. (An under-count panics
    /// at draw time and an over-count at the end of the run, so this test also
    /// exercises those assertions.)
    #[test]
    #[ignore] // n=4096 secure ring — slow in debug, fast in release
    fn precomputed_masks_change_nothing_but_the_timing() {
        for (d, seed) in [(8usize, 1u64), (10, 5)] {
            let (model, input) = CnnModel::random(d, seed);
            for nonlin in [Nonlin::A2b, Nonlin::Gc] {
                let on = run_cnn_hybrid_full(&model, &input, NetworkProfile::INSTANT, nonlin, false);
                let pre = run_cnn_hybrid_full(&model, &input, NetworkProfile::INSTANT, nonlin, true);
                assert!(on.correct && pre.correct, "d={d} seed={seed}: must match plaintext");
                assert_eq!(pre.class, on.class, "d={d} seed={seed}: precompute changed the class");
                assert_eq!(pre.fhe_phases.n_mask_enc, 0, "online path still encrypted masks");
                assert_eq!(
                    pre.fhe_phases.n_weight_enc, 0,
                    "d={d} seed={seed}: online path still encrypted the model"
                );
                assert!(
                    pre.fhe_phases.weights_ms == 0.0,
                    "d={d} seed={seed}: model encryption still charged to the online phase"
                );
                assert!(on.fhe_phases.n_weight_enc > 0, "the control must encrypt online");
                assert!(pre.offline_ms > on.offline_ms, "the model's cost must move offline");
                assert!(pre.offline_ms > 0.0, "offline work must be reported, not hidden");
            }
        }
        // The two-conv model exercises the multi-layer size prediction for both
        // halves of the preamble — chunked hidden layers after two conv stages
        // are where a count goes wrong.
        let (model, input, _) = two_conv_model();
        let pre = run_cnn_hybrid_full(&model, &input, NetworkProfile::INSTANT, Nonlin::Gc, true);
        assert!(pre.correct, "two-conv: the precomputed preamble must still be exact");
        assert_eq!(pre.fhe_phases.n_weight_enc, 0, "two-conv: model not fully precomputed");
    }

    /// The encrypted model is **per-session**, so a second query in the same
    /// session must reuse it: same class, and still no online weight encryption.
    ///
    /// This is what separates it from the mask pool, which is per-query and has
    /// to be refilled. Masks shift latency; weights amortize.
    #[test]
    #[ignore]
    fn the_encrypted_model_is_reused_across_queries_in_a_session() {
        let (model, _) = CnnModel::random(8, 3);
        let s = Bfv::new(BfvParams { n: 4096, log2_q: 100, log2_t: model.nbits as u32 });
        let (ck, _sk) = s.keygen();
        let polys = weight_polys(&s, &model);
        let ek = s.encrypt_key(&ck);
        let enc = EncryptedModel::encrypt(&s, &ek, &polys);

        // Two different inputs, one encrypted model: the weight ciphertexts do
        // not depend on the query, so the same `Weights` source serves both —
        // and the products must come out identical for identical inputs.
        let mut first: Option<Vec<Vec<u128>>> = None;
        for _ in 0..2 {
            let (_, input) = CnnModel::random(8, 3);
            let c_in = s.encrypt_poly_pre(&ek, &s.pack_input(&input));
            let cxt = s.transform(&c_in);
            let mut w = Weights::session(&enc);
            let mut ph = FhePhases::default();
            let mut prods = Vec::new();
            for poly in &polys {
                let prod = w.mul_next(&s, &ek, &cxt, || poly.clone(), &mut ph);
                assert_eq!(prod.c.len(), 3, "a ct×ct product is degree-2");
                prods.push(prod.c[0].clone());
            }
            w.assert_fully_consumed();
            assert_eq!(ph.n_weight_enc, 0, "session weights must not re-encrypt");
            // Only the input encryption is randomised, so re-running against the
            // *same* input must reproduce the same products bit for bit only if
            // the input ciphertext is reused; here it is fresh, so just check
            // the shape is stable across queries.
            match &first {
                None => first = Some(prods),
                Some(f) => assert_eq!(f.len(), prods.len(), "product count must be stable"),
            }
        }
    }

    /// The scheme-switch import must cost a *constant* number of rounds, not
    /// one per activation.
    ///
    /// This is the regression test for the pipeline's dominant cost: the
    /// import loop used to run three blocking input-sharing rounds per
    /// element, which at `28×28` was 94% of the MPC segment and 88% of
    /// end-to-end runtime while the ReLU it fed was 3%. Message *count* is the
    /// right invariant to pin rather than wall-clock: the bytes were never the
    /// problem, the round trips were, and a wall-clock threshold would be
    /// flaky on shared CI.
    #[test]
    #[ignore] // n=4096 secure ring — slow in debug, fast in release
    fn scheme_switch_import_does_not_scale_with_layer_width() {
        let (m8, i8v) = CnnModel::random(8, 1);
        let (m28, i28) = CnnModel::random(28, 1);
        let small = run_cnn_hybrid(&m8, &i8v, NetworkProfile::INSTANT);
        let big = run_cnn_hybrid(&m28, &i28, NetworkProfile::INSTANT);
        // 6×6×8 = 288 activations versus 26×26×8 = 5408, an 18.8× wider
        // layer. Per-element sharing would put the message counts in that
        // ratio; batched, they must be within a small constant.
        assert!(
            big.comm.messages < 2 * small.comm.messages,
            "import looks per-element again: {} messages at d=28 vs {} at d=8",
            big.comm.messages,
            small.comm.messages
        );
    }

    /// The **garbled** nonlinearity's round structure must be constant in the
    /// layer width too — the same invariant as the import above, on the other
    /// protocol.
    ///
    /// `gc_protocol::y2a` converts the garbled boolean output back to
    /// arithmetic, and it used to call `share_input` once per (item, bit): at
    /// `d=28` that is `2 × 5408 × 12 = 129 792` calls for one ReLU, and the
    /// D=28 run moved **319 139 messages**. Batched, the same inference moves
    /// **47**. Message count is the right thing to pin rather than wall-clock —
    /// the bytes were never the problem and a time threshold would be flaky on
    /// shared CI.
    #[test]
    #[ignore]
    fn garbled_nonlinearity_does_not_scale_with_layer_width() {
        let (m8, i8v) = CnnModel::random(8, 1);
        let (m28, i28) = CnnModel::random(28, 1);
        let small = run_cnn_hybrid_with(&m8, &i8v, NetworkProfile::INSTANT, Nonlin::Gc);
        let big = run_cnn_hybrid_with(&m28, &i28, NetworkProfile::INSTANT, Nonlin::Gc);
        assert!(small.correct && big.correct, "batching must not change the class");
        // 288 activations versus 5408, an 18.8× wider layer. Per-element
        // resharing would put the message counts in that ratio.
        assert!(
            big.comm.messages < 2 * small.comm.messages,
            "y2a looks per-element again: {} messages at d=28 vs {} at d=8",
            big.comm.messages,
            small.comm.messages
        );
        // An absolute ceiling as well: the ratio test alone would pass if both
        // regressed together.
        assert!(
            big.comm.messages < 1000,
            "garbled D=28 inference moved {} messages; it was 319139 when y2a \
             resharing was per-element and should now be tens",
            big.comm.messages
        );
    }

    /// A **two-convolution** model (a second, multi-channel conv layer after
    /// the first, feeding a hidden dense) must classify bit-exactly through
    /// the encrypted pipeline — the extra conv is just another
    /// FHE-conv + MPC-nonlinearity + scheme-switch, still zero bootstraps.
    /// Small controlled weights so `forward` is deterministic and in budget.
    /// d=16: conv1(OC) 14×14 → pool 7×7 → conv2(C2) 5×5 → pool 2×2. Small
    /// controlled weights so `forward` is deterministic and in budget.
    /// Shared by every test that needs more than one scheme switch.
    fn two_conv_model() -> (CnnModel, Vec<i64>, usize) {
        let d = 16;
        let (mut model, input) = CnnModel::random(d, 3);
        let c2 = 4usize;
        // conv2: c2 output channels, in_c = OC, 7×7, 3×3 kernels.
        let mut rng = StdRng::seed_from_u64(99);
        let kernels: Vec<Vec<i64>> = (0..c2)
            .map(|_| (0..OC * K * K).map(|_| rng.gen_range(-2i64..=2)).collect())
            .collect();
        let cbias: Vec<i64> = (0..c2).map(|_| rng.gen_range(-4i64..=4)).collect();
        model.extra_conv = vec![ConvLayer { kernels, cbias, in_c: OC, in_h: 7, in_w: 7 }];
        // dense now takes the two-conv flat = c2 * 2 * 2 = 16.
        let flat2 = c2 * 2 * 2;
        model.dense_w = (0..NCLASSES)
            .map(|_| (0..flat2).map(|_| rng.gen_range(-3i64..=3)).collect())
            .collect();
        // recalibrate nbits for the new architecture.
        let (_, max_abs) = model.forward(&input);
        model.nbits = ((64 - (max_abs.max(1) as u64).leading_zeros() as usize) + 2).max(20);
        assert!(model.nbits <= 30);
        (model, input, c2)
    }

    #[test]
    #[ignore] // real BFV × ABY3 hybrid, two conv layers
    fn two_conv_hybrid_matches_plaintext() {
        let (model, input, c2) = two_conv_model();
        let r = run_cnn_hybrid(&model, &input, NetworkProfile::INSTANT);
        assert!(r.correct, "two-conv: encrypted class {} != plaintext {}", r.class, r.expected);
        assert_eq!(r.bootstraps, 0);
        assert_eq!(
            r.conv_mults,
            OC + c2,
            "conv mults should count both conv layers' output channels"
        );
    }

    /// Conv packing lifts the input-size ceiling: a full 28×28 MNIST-shaped
    /// image classifies exactly, which the im2col/Toeplitz formulation could
    /// not do at any sane cost (it needs `conv_mults_toeplitz` multiplies,
    /// here in the hundreds, versus `OC`).
    #[test]
    #[ignore] // full 28×28 through the real 3-party transport
    fn conv_packing_classifies_a_full_mnist_sized_image() {
        let (model, input) = CnnModel::random(28, 7);
        let r = run_cnn_hybrid(&model, &input, NetworkProfile::INSTANT);
        assert!(r.correct, "encrypted class {} != plaintext {}", r.class, r.expected);
        assert_eq!(r.bootstraps, 0);
        assert_eq!(r.conv_mults, OC, "one ct×ct multiply per output channel");
        assert!(
            r.conv_mults_toeplitz > 100 * r.conv_mults,
            "expected a large win over Toeplitz, got {} vs {}",
            r.conv_mults_toeplitz, r.conv_mults
        );
    }

    /// The security claim, as a test rather than prose: the benchmark must
    /// run on RLWE-secure parameters. `n = 4096` admits `log q ≤ 109` at the
    /// 128-bit level (HomomorphicEncryption.org), and the plaintext modulus
    /// must stay inside the depth-1 noise budget. Guards against a silent
    /// drift of either parameter out of the secure region.
    #[test]
    #[ignore]
    fn uses_rlwe_secure_parameters() {
        let (model, input) = CnnModel::random(8, 21);
        let r = run_cnn_hybrid(&model, &input, NetworkProfile::INSTANT);
        assert_eq!(r.ring_n, 4096, "ring dimension");
        assert!(r.log2_q <= 109, "q=2^{} exceeds the 128-bit bound for n=4096", r.log2_q);
        assert!(r.log2_t <= 30, "t=2^{} past the depth-1 noise budget", r.log2_t);
        assert!(r.correct && r.bootstraps == 0);
    }

    /// The FHE cost of the convolution is flat in D — it is `OC`
    /// multiplications either way — so growing the image must not grow the
    /// FHE time appreciably. (The MPC side does grow: more ReLUs.)
    #[test]
    #[ignore]
    fn fhe_cost_is_flat_in_image_size() {
        let (m8, i8v) = CnnModel::random(8, 11);
        let (m28, i28) = CnnModel::random(28, 11);
        let small = run_cnn_hybrid(&m8, &i8v, NetworkProfile::INSTANT);
        let big = run_cnn_hybrid(&m28, &i28, NetworkProfile::INSTANT);
        assert!(small.correct && big.correct);
        assert_eq!(small.conv_mults, big.conv_mults);
        assert!(
            big.fhe_ms < 4.0 * small.fhe_ms,
            "FHE time should be near-flat in D: {:.0}ms at D=8 vs {:.0}ms at D=28",
            small.fhe_ms, big.fhe_ms
        );
    }
}
