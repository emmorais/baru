//! `baru bench-cnn` — a full convolutional classifier through the hybrid
//! pipeline: FHE ciphertext×ciphertext linear layers, MPC nonlinearities
//! (ReLU, max-pool, argmax), **zero FHE bootstraps**, verified bit-exact
//! against the plaintext reference, with a real cost measurement.
//!
//! The FHE side is **BFV at RLWE-secure parameters** (n=4096, q=2^100, at or
//! below the ~109-bit ceiling for n=4096 at the 128-bit security level).
//! This benchmark previously ran on SI-NTRU, which our own analysis shows is
//! *not* secure at the parameters this construction needs — not even at
//! depth 1, where the fatigue point sits around 2^59 (adjusted) against a
//! required q ≳ 2^84. See the security note in `crate::fhe::si_ntru`. BFV
//! supports the identical no-relinearization trick (multiply, then decrypt
//! the degree-2 result under s^2), so the construction transfers unchanged.
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
//! | | a2b | gc |
//! |---|---|---|
//! | total | 2535 ms | **1986 ms** |
//! | FHE | 1356 ms | 1351 ms |
//! | MPC | 1153 ms | 609 ms |
//! | — import | 9 ms | 11 ms |
//! | — ReLU∘rescale | 615 ms | 407 ms |
//! | — max-pool | 467 ms | 155 ms |
//! | — argmax | 55 ms | 27 ms |
//! | wire bytes | 39.5 MB | 21.8 MB |
//! | messages | 88 | 319 139 |
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
//!
//! The cost centre is now the **FHE arithmetic** (~68% here, 84% on the
//! trained two-conv medical model, which has four encrypted linear layers to
//! this one's two). That is where the next optimisation belongs.
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
        (crate::gadgets::argmax_ref(&logits), max_abs)
    }

}

// ─── Hybrid FHE matvec + scheme-switch to opened shares ───────────────────────

/// One packed ct×ct matvec of `weights` (row-major, each length `in_dim`)
/// against `c_input`, chunked to respect the ring packing bound, followed
/// by a mask-and-open switch. Returns per-output opened residues and the
/// three per-output masks (so the MPC side can reconstruct the raw share).
#[allow(clippy::type_complexity)]
fn hybrid_matvec_open(
    s: &Bfv,
    ck: &[u8],
    c_input: &BfvCt,
    weights: &[Vec<i64>],
    in_dim: usize,
    profile: NetworkProfile,
    rng: &mut impl Rng,
) -> (Vec<u64>, Vec<[u64; 3]>) {
    let n = s.params.n;
    let t_mask = s.params.t() - 1;
    let rows_per_chunk = (n / in_dim).saturating_sub(1).max(1);
    // The input's forward NTT is computed once and reused across chunks.
    let cxt = s.transform(c_input);

    let mut opened = Vec::with_capacity(weights.len());
    let mut masks = Vec::with_capacity(weights.len());
    for chunk in weights.chunks(rows_per_chunk) {
        let cw = s.encrypt_poly(ck, &s.pack_weights(chunk));
        let prod = s.mul_pre(&cxt, &cw);
        let chunk_masks: Vec<Vec<u64>> = (0..3)
            .map(|_| (0..n).map(|_| rng.gen::<u64>() & t_mask).collect())
            .collect();
        let mut conv_bytes = 0usize;
        let mut masked = prod;
        for m in &chunk_masks {
            let enc = s.encrypt_poly(ck, m);
            conv_bytes += s.ct_bytes(&enc);
            masked = s.add(&masked, &enc);
        }
        let y = s.decrypt_poly(ck, &masked);
        sleep_link(&profile, conv_bytes + chunk.len() * 8, 2);
        for r in 0..chunk.len() {
            let pos = s.matvec_output_position(in_dim, r);
            opened.push(y[pos]);
            masks.push([chunk_masks[0][pos], chunk_masks[1][pos], chunk_masks[2][pos]]);
        }
    }
    (opened, masks)
}

/// Convolution via [`crate::fhe::conv_pack`]: one ct×ct multiply per output
/// channel, then a mask-and-open per channel. Returns the opened
/// pre-activations and per-output masks flattened in `(oc, oy, ox)` order —
/// the same layout the MPC stage and the plaintext reference use.
#[allow(clippy::type_complexity)]
fn hybrid_conv_open(
    s: &Bfv,
    ck: &[u8],
    c_input: &BfvCt,
    shape: ConvShape,
    kernels: &[Vec<i64>],
    profile: NetworkProfile,
    rng: &mut impl Rng,
) -> (Vec<u64>, Vec<[u64; 3]>) {
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
    // Every output channel multiplies against the *same* encrypted input, so
    // its forward NTT is computed once and reused (`mul_pre`).
    let cxt = s.transform(c_input);

    for kernel in kernels.iter() {
        let kpoly = conv_pack::pack_kernel(kernel, shape, n, t_mask);
        let cw = s.encrypt_poly(ck, &kpoly);
        let prod = s.mul_pre(&cxt, &cw);

        // Fresh masks per output channel (reuse across channels would leak
        // the difference of their pre-activations).
        let chan_masks: Vec<Vec<u64>> = (0..3)
            .map(|_| (0..n).map(|_| rng.gen::<u64>() & t_mask).collect())
            .collect();
        let mut conv_bytes = 0usize;
        let mut masked = prod;
        for m in &chan_masks {
            let enc = s.encrypt_poly(ck, m);
            conv_bytes += s.ct_bytes(&enc);
            masked = s.add(&masked, &enc);
        }
        let y = s.decrypt_poly(ck, &masked);
        sleep_link(&profile, conv_bytes + oh * ow * 8, 2);
        for oy in 0..oh {
            for ox in 0..ow {
                let pos = shape.output_position(oy, ox);
                opened.push(y[pos]);
                masks.push([chan_masks[0][pos], chan_masks[1][pos], chan_masks[2][pos]]);
            }
        }
    }
    (opened, masks)
}

/// Re-encrypt a layer's MPC output (three parties' additive share
/// components of each value) as a single packed BFV ciphertext — the
/// MPC→FHE half of the scheme switch. Communication-free per party except
/// the one modelled link hop. Returns the ciphertext and the bytes sent.
fn pack_shares_to_fhe(
    s: &Bfv,
    ck: &[u8],
    per_party: &[Vec<Fr>],
    t_mask: u64,
    profile: NetworkProfile,
) -> (BfvCt, usize) {
    let mut c: Option<BfvCt> = None;
    let mut bytes = 0usize;
    for comps in per_party {
        let m: Vec<u64> = comps.iter().map(|&v| (v as u64) & t_mask).collect();
        let enc = s.encrypt_poly(ck, &m);
        bytes += s.ct_bytes(&enc);
        c = Some(match c {
            None => enc,
            Some(acc) => s.add(&acc, &enc),
        });
    }
    sleep_link(&profile, bytes, 1);
    (c.expect("three parties"), bytes)
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
            be.set_gc_nonlinearities(nonlin == Nonlin::Gc);
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
            let t = Instant::now();
            let relu = fused_relu_rescale(be, &vs, shift, nbits, nonlin);
            let relu_ms = ms(t);
            // `maxpool2d` reduces to `lt_batch`, which the backend flag above
            // has already routed — the gadget itself is protocol-agnostic.
            let t = Instant::now();
            let (pooled, _) = maxpool2d(be, &relu, conv_shape, POOL, POOL);
            let pool_ms = ms(t);
            let comps = pooled.iter().map(|p| be.my_additive_component(p)).collect::<Vec<Fr>>();
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
}

impl Nonlin {
    pub fn label(self) -> &'static str {
        match self {
            Nonlin::A2b => "a2b (batched bit decomposition)",
            Nonlin::Gc => "gc (3-party garbled, constant-round)",
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
    let (expected, max_abs) = model.forward(input);
    assert!(
        (max_abs as i128) < (1i128 << (model.nbits - 1)),
        "cnn: intermediate magnitude {max_abs} overflows nbits={}",
        model.nbits
    );
    let nbits = model.nbits;
    // RLWE-secure parameters: n=4096, q=2^100 is at/below the 128-bit bound
    // of ~109 for n=4096 (HomomorphicEncryption.org). Unlike SI-NTRU there
    // is no fatigue point to worry about — see `fhe::si_ntru`'s security note
    // for why the NTRU instantiation could not be rescued even at depth 1.
    let s = Bfv::new(BfvParams { n: 4096, log2_q: 100, log2_t: nbits as u32 });
    let n_in = model.d * model.d;
    assert!((n_in + 1) <= s.params.n, "input too large for coefficient packing");
    let t_mask = s.params.t() - 1;

    let start = Instant::now();
    let (mut fhe_ms, mut mpc_ms) = (0.0, 0.0);
    let mut comm = CommStats::default();
    let mut rng = rand::thread_rng();

    let (ck, _sk) = s.keygen();
    let c_input = s.encrypt_poly(&ck, &s.pack_input(input));

    let shift = model.shift;

    // ── First conv via Cheetah-style packing → open ──
    // One ct×ct multiply per OUTPUT CHANNEL (not per output pixel): the
    // kernel is packed reversed on every axis, so polynomial multiplication
    // *is* the sliding-window convolution and every output pixel of that
    // channel appears at once. See `fhe::conv_pack`.
    let t0 = Instant::now();
    let conv1_shape = ConvShape { in_c: 1, h: model.d, w: model.d, kh: K, kw: K };
    let (conv_open, conv_masks) =
        hybrid_conv_open(&s, &ck, &c_input, conv1_shape, &model.kernels, profile, &mut rng);
    fhe_ms += t0.elapsed().as_secs_f64() * 1000.0;

    // ── MPC: +cbias≪s, fused ReLU∘rescale, max-pool → pooled feature shares ──
    let t0 = Instant::now();
    let mut phases = MpcPhases::default();
    let (mut per_party_feat, conv_comm, ph) = mpc_conv_relu_pool(
        conv_open, conv_masks, model.cbias.clone(), model.conv_shape, shift, nbits, profile,
        nonlin,
    );
    phases.add(&ph);
    comm = CommStats { messages: comm.messages + conv_comm.messages, bytes: comm.bytes + conv_comm.bytes };
    mpc_ms += t0.elapsed().as_secs_f64() * 1000.0;
    let mut feat_shape = model.pool_shape;

    // ── Additional conv layers: switch pooled features back to FHE, convolve
    //    (multi-channel packing), open, MPC ReLU∘rescale + pool. Each extra
    //    conv is just the same three steps composed again — no bootstrap. ──
    for cl in &model.extra_conv {
        let tc = Instant::now();
        let (c_conv_in, _) = pack_shares_to_fhe(&s, &ck, &per_party_feat, t_mask, profile);
        let shape = ConvShape { in_c: cl.in_c, h: cl.in_h, w: cl.in_w, kh: K, kw: K };
        let (co, cm) =
            hybrid_conv_open(&s, &ck, &c_conv_in, shape, &cl.kernels, profile, &mut rng);
        fhe_ms += tc.elapsed().as_secs_f64() * 1000.0;

        let tm = Instant::now();
        let out_shape = Chw { c: cl.out_c(), h: cl.out_h(), w: cl.out_w() };
        let (feat2, comm2, ph2) =
            mpc_conv_relu_pool(co, cm, cl.cbias.clone(), out_shape, shift, nbits, profile, nonlin);
        phases.add(&ph2);
        comm = CommStats {
            messages: comm.messages + comm2.messages,
            bytes: comm.bytes + comm2.bytes,
        };
        mpc_ms += tm.elapsed().as_secs_f64() * 1000.0;
        per_party_feat = feat2;
        feat_shape = Chw { c: cl.out_c(), h: cl.out_h() / POOL, w: cl.out_w() / POOL };
    }

    // ── Switch pooled features → packed FHE dense input ──
    let t0 = Instant::now();
    let flat = feat_shape.c * feat_shape.h * feat_shape.w;
    let (mut c_dense_in, _) = pack_shares_to_fhe(&s, &ck, &per_party_feat, t_mask, profile);
    let mut dense_in_dim = flat;
    fhe_ms += t0.elapsed().as_secs_f64() * 1000.0;

    // ── Stacked hidden dense layers: each is FHE matvec → open → MPC
    //    ReLU∘rescale → switch back. Same machinery as the output dense + the
    //    conv ReLU, composed once per hidden layer — the "no bootstrap, just
    //    switch again" invariant made concrete for arbitrary depth. ──
    for (hidden_w, hidden_b) in model.hidden.clone() {
        let th = Instant::now();
        let (hidden_open, hidden_masks) =
            hybrid_matvec_open(&s, &ck, &c_dense_in, &hidden_w, dense_in_dim, profile, &mut rng);
        fhe_ms += th.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let h = hidden_w.len();
        let (per_party_hidden, hidden_comm) = run_parties(profile, |me| {
            let hidden_open = hidden_open.clone();
            let hidden_masks = hidden_masks.clone();
            let hidden_b = hidden_b.clone();
            move |be: &mut TransportBackend<MpcTransport>| {
                be.set_gc_nonlinearities(nonlin == Nonlin::Gc);
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
        comm = CommStats {
            messages: comm.messages + hidden_comm.messages,
            bytes: comm.bytes + hidden_comm.bytes,
        };
        mpc_ms += t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let (c_hidden, _) = pack_shares_to_fhe(&s, &ck, &per_party_hidden, t_mask, profile);
        c_dense_in = c_hidden;
        dense_in_dim = h;
        fhe_ms += t2.elapsed().as_secs_f64() * 1000.0;
    }

    // ── Output dense (FHE ct×ct matvec) → open ──
    let t3 = Instant::now();
    let (dense_open, dense_masks) = hybrid_matvec_open(
        &s,
        &ck,
        &c_dense_in,
        &model.dense_w,
        dense_in_dim,
        profile,
        &mut rng,
    );
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
            be.set_gc_nonlinearities(nonlin == Nonlin::Gc);
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
    comm = CommStats { messages: comm.messages + argmax_comm.messages, bytes: comm.bytes + argmax_comm.bytes };
    mpc_ms += t0.elapsed().as_secs_f64() * 1000.0;
    let class = classes[0].0;

    CnnRun {
        correct: class == expected,
        class,
        expected,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
        fhe_ms,
        mpc_ms,
        phases,
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

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--gc" => nonlin = Nonlin::Gc,
            "--both" => both = true,
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
        let a = run_cnn_hybrid_with(&model, &input, profile, Nonlin::A2b);
        let g = run_cnn_hybrid_with(&model, &input, profile, Nonlin::Gc);
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

    let r = run_cnn_hybrid_with(&model, &input, profile, nonlin);

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
        "  mpc breakdown   : import {:.0}ms, relu {:.0}ms, pool {:.0}ms, argmax {:.0}ms",
        r.phases.import_ms, r.phases.relu_ms, r.phases.pool_ms, r.phases.argmax_ms
    );
    println!("  MPC wire traffic: {} messages, {} bytes", r.comm.messages, r.comm.bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A **two-convolution** model (a second, multi-channel conv layer after
    /// the first, feeding a hidden dense) must classify bit-exactly through
    /// the encrypted pipeline — the extra conv is just another
    /// FHE-conv + MPC-nonlinearity + scheme-switch, still zero bootstraps.
    /// Small controlled weights so `forward` is deterministic and in budget.
    #[test]
    #[ignore] // real BFV × ABY3 hybrid, two conv layers
    fn two_conv_hybrid_matches_plaintext() {
        // d=16: conv1(OC) 14×14 → pool 7×7 → conv2(C2) 5×5 → pool 2×2.
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
    /// regression to SI-NTRU, which is *not* secure here at any depth.
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
