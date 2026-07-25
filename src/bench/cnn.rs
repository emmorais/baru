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
//!   classifier — for the trained model on real MNIST test digits (97.6%,
//!   run through this same hybrid, bit-exact) see `baru bench-mnist` /
//!   `crate::bench::mnist`.
//! - Conv packing needs `2·in_c·H·W ≲ n`, so a full 28×28 MNIST image runs
//!   comfortably at `n = 4096` (it needs degree 841). Multi-channel CIFAR
//!   (3×32×32) needs 5185 and so must split channels across ciphertexts (or
//!   move to n=8192) — a packing question, not a limit of the no-bootstrap
//!   construction.
//! - With conv packing the FHE cost is nearly **flat in D** (it is `OC`
//!   multiplications regardless), so the bottleneck moves decisively to the
//!   MPC nonlinearities: at D=28 the ~1352 ReLUs and the max-pool
//!   comparisons dominate. Speeding *those* up (CrypTFlow2-style comparison
//!   and truncation protocols) is now the highest-value optimisation.

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

pub struct CnnModel {
    pub d: usize,        // input side length (1 channel)
    pub kernels: Vec<Vec<i64>>, // [OC][K*K]
    pub cbias: Vec<i64>, // [OC]
    /// Optional hidden dense layer (`[H][flat]` weights, `[H]` bias), with a
    /// fused ReLU∘rescale, sitting between the pooled conv features and the
    /// output layer. `None` for the shallow model `bench-cnn` exercises;
    /// `Some` for the trained deeper classifier in `bench-mnist`.
    pub hidden_w: Option<Vec<Vec<i64>>>,
    pub hidden_b: Option<Vec<i64>>,
    pub dense_w: Vec<Vec<i64>>, // [NCLASSES][flat] or [NCLASSES][H]
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
            hidden_w: None,
            hidden_b: None,
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
        hidden_w: Option<Vec<Vec<i64>>>,
        hidden_b: Option<Vec<i64>>,
        dense_w: Vec<Vec<i64>>,
        dense_b: Vec<i64>,
        shift: u32,
        sample_inputs: &[Vec<i64>],
    ) -> Self {
        assert_eq!(kernels.len(), OC);
        assert_eq!(dense_w.len(), NCLASSES);
        assert_eq!(hidden_w.is_some(), hidden_b.is_some(), "hidden weights and bias must agree");
        let oh = d - (K - 1);
        let ph = oh / POOL;
        let mut model = CnnModel {
            d,
            kernels,
            cbias,
            hidden_w,
            hidden_b,
            dense_w,
            dense_b,
            shift,
            nbits: 20,
            conv_shape: Chw { c: OC, h: oh, w: oh },
            pool_shape: Chw { c: OC, h: ph, w: ph },
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
        let mut max_abs = input.iter().map(|v| v.abs()).max().unwrap_or(0);
        // conv → fused ReLU∘rescale
        let mut conv_act = vec![0i64; OC * oh * oh];
        for oc in 0..OC {
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
        let (pooled, _) = crate::gadgets::maxpool2d_ref(&conv_act, self.conv_shape, POOL, POOL);
        // optional hidden dense → fused ReLU∘rescale
        let dense_in: Vec<i64> = match (&self.hidden_w, &self.hidden_b) {
            (Some(hw), Some(hb)) => hw
                .iter()
                .zip(hb)
                .map(|(row, &b)| {
                    let v = row.iter().zip(&pooled).map(|(&w, &x)| w * x).sum::<i64>() + b;
                    max_abs = max_abs.max(v.abs());
                    if v >= 0 {
                        v >> self.shift
                    } else {
                        0
                    }
                })
                .collect(),
            _ => pooled,
        };
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
    model: &CnnModel,
    profile: NetworkProfile,
    rng: &mut impl Rng,
) -> (Vec<u64>, Vec<[u64; 3]>) {
    let n = s.params.n;
    let t_mask = s.params.t() - 1;
    let shape = ConvShape { in_c: 1, h: model.d, w: model.d, kh: K, kw: K };
    assert!(
        shape.required_degree() < n,
        "conv packing needs degree {} < ring dimension {n}",
        shape.required_degree()
    );
    let (oh, ow) = (shape.out_h(), shape.out_w());
    let mut opened = Vec::with_capacity(OC * oh * ow);
    let mut masks = Vec::with_capacity(OC * oh * ow);
    // Every output channel multiplies against the *same* encrypted image, so
    // its forward NTT is computed once and reused (`mul_pre`).
    let cxt = s.transform(c_input);

    for oc in 0..OC {
        let kpoly = conv_pack::pack_kernel(&model.kernels[oc], shape, n, t_mask);
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

/// Reconstruct this party's share of the raw matvec output `j` from the
/// opened value and the per-party masks (party 0 also holds `y`).
fn import_raw(
    be: &mut TransportBackend<MpcTransport>,
    me: usize,
    opened: &[u64],
    masks: &[[u64; 3]],
    j: usize,
    nbits: usize,
) -> ShareRef {
    let my_comp: Fr = if me == 0 {
        (opened[j] as u128).wrapping_sub(masks[j][0] as u128)
    } else {
        0u128.wrapping_sub(masks[j][me] as u128)
    };
    let mut raw: Option<ShareRef> = None;
    for owner in 0..3 {
        let v = if me == owner { my_comp } else { 0 };
        let sh = be.share_secret(v, nbits, owner);
        raw = Some(match raw {
            None => sh,
            Some(a) => be.add(&a, &sh),
        });
    }
    raw.unwrap()
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

pub fn run_cnn_hybrid(model: &CnnModel, input: &[i64], profile: NetworkProfile) -> CnnRun {
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

    // ── Conv via Cheetah-style packing → open ──
    // One ct×ct multiply per OUTPUT CHANNEL (not per output pixel): the
    // kernel is packed reversed on every axis, so polynomial multiplication
    // *is* the sliding-window convolution and every output pixel of that
    // channel appears at once. See `fhe::conv_pack`.
    let t0 = Instant::now();
    let (conv_open, conv_masks) =
        hybrid_conv_open(&s, &ck, &c_input, model, profile, &mut rng);
    fhe_ms += t0.elapsed().as_secs_f64() * 1000.0;

    // ── MPC: +cbias≪s, fused ReLU∘rescale, max-pool → export pooled acts ──
    let t0 = Instant::now();
    let cbias = model.cbias.clone();
    let conv_shape = model.conv_shape;
    let shift = model.shift;
    let (per_party_pool, conv_comm) = run_parties(profile, |me| {
        let conv_open = conv_open.clone();
        let conv_masks = conv_masks.clone();
        let cbias = cbias.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            let oh = conv_shape.h;
            // Reconstruct + bias every conv output locally (share_secret /
            // add are communication-free), then run the whole ReLU layer as
            // one batched a2b + one batched multiply.
            let vs: Vec<ShareRef> = (0..OC * oh * oh)
                .map(|j| {
                    let raw = import_raw(be, me, &conv_open, &conv_masks, j, nbits);
                    let oc = j / (oh * oh);
                    let bconst = residue(cbias[oc] << shift, nbits);
                    let bsh = be.share_public(bconst, nbits);
                    be.add(&raw, &bsh)
                })
                .collect();
            let relu = fused_relu_rescale_batch(be, &vs, shift, nbits);
            // max-pool (gadget) over the ReLU'd feature maps
            let (pooled, _) = maxpool2d(be, &relu, conv_shape, POOL, POOL);
            pooled.iter().map(|p| be.my_additive_component(p)).collect::<Vec<Fr>>()
        }
    });
    comm = CommStats { messages: comm.messages + conv_comm.messages, bytes: comm.bytes + conv_comm.bytes };
    mpc_ms += t0.elapsed().as_secs_f64() * 1000.0;

    // ── Switch back → packed FHE dense input ──
    let t0 = Instant::now();
    let flat = model.pool_shape.c * model.pool_shape.h * model.pool_shape.w;
    let (mut c_dense_in, _) = pack_shares_to_fhe(&s, &ck, &per_party_pool, t_mask, profile);
    let mut dense_in_dim = flat;
    fhe_ms += t0.elapsed().as_secs_f64() * 1000.0;

    // ── Optional hidden dense layer: FHE matvec → open → MPC ReLU∘rescale →
    //    switch back. Same machinery as the output dense + the conv ReLU,
    //    composed one more time — this is the "no bootstrap, just switch
    //    again" invariant made concrete for an extra depth level. ──
    if let (Some(hidden_w), Some(hidden_b)) = (model.hidden_w.clone(), model.hidden_b.clone()) {
        let th = Instant::now();
        let (hidden_open, hidden_masks) =
            hybrid_matvec_open(&s, &ck, &c_dense_in, &hidden_w, flat, profile, &mut rng);
        fhe_ms += th.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let h = hidden_w.len();
        let (per_party_hidden, hidden_comm) = run_parties(profile, |me| {
            let hidden_open = hidden_open.clone();
            let hidden_masks = hidden_masks.clone();
            let hidden_b = hidden_b.clone();
            move |be: &mut TransportBackend<MpcTransport>| {
                let vs: Vec<ShareRef> = (0..h)
                    .map(|j| {
                        let raw = import_raw(be, me, &hidden_open, &hidden_masks, j, nbits);
                        let bsh = be.share_public(residue(hidden_b[j], nbits), nbits);
                        be.add(&raw, &bsh)
                    })
                    .collect();
                let act = fused_relu_rescale_batch(be, &vs, shift, nbits);
                act.iter().map(|p| be.my_additive_component(p)).collect::<Vec<Fr>>()
            }
        });
        comm = CommStats {
            messages: comm.messages + hidden_comm.messages,
            bytes: comm.bytes + hidden_comm.bytes,
        };
        mpc_ms += t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let (c_hidden, _) = pack_shares_to_fhe(&s, &ck, &per_party_hidden, t_mask, profile);
        c_dense_in = c_hidden;
        dense_in_dim = h;
        // fold this switch's FHE time into the shared accumulator below
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
    let (classes, argmax_comm) = run_parties(profile, |me| {
        let dense_open = dense_open.clone();
        let dense_masks = dense_masks.clone();
        let dense_b = dense_b.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            let logits: Vec<ShareRef> = (0..NCLASSES)
                .map(|j| {
                    let raw = import_raw(be, me, &dense_open, &dense_masks, j, nbits);
                    let bsh = be.share_public(residue(dense_b[j], nbits), nbits);
                    be.add(&raw, &bsh)
                })
                .collect();
            let idx = argmax(be, &logits, nbits);
            (be.reveal(&idx, None) as u64 & (t_mask)) as usize
        }
    });
    comm = CommStats { messages: comm.messages + argmax_comm.messages, bytes: comm.bytes + argmax_comm.bytes };
    mpc_ms += t0.elapsed().as_secs_f64() * 1000.0;
    let class = classes[0];

    CnnRun {
        correct: class == expected,
        class,
        expected,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
        fhe_ms,
        mpc_ms,
        comm,
        bootstraps: 0,
        ring_n: s.params.n,
        log2_q: s.params.log2_q,
        log2_t: s.params.log2_t,
        conv_mults: OC,
        conv_mults_toeplitz: conv_pack::toeplitz_multiplications(
            ConvShape { in_c: 1, h: model.d, w: model.d, kh: K, kw: K },
            OC,
            BfvParams::DEFAULT.n,
        ),
    }
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct CnnPoint {
    pub d: usize,
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
  --json               emit JSON
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut d = 8usize;
    let mut rtt_ms = 0u64;
    let mut bandwidth_mbps: Option<f64> = None;
    let mut seed = 1u64;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
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
    let r = run_cnn_hybrid(&model, &input, profile);

    if json {
        let p = CnnPoint {
            d,
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
        };
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
    println!("  time            : total {:.0}ms  (fhe {:.0}ms, mpc {:.0}ms)", r.total_ms, r.fhe_ms, r.mpc_ms);
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
