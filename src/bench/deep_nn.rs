//! `baru bench-deep-nn` — deep private inference with **zero FHE
//! bootstrapping**, at cost linear in depth.
//!
//! This is the experimental core of the paper claim. An `L`-layer
//! fully-connected ReLU network is evaluated end to end under the hybrid
//! construction: each layer is one BFV ciphertext×ciphertext matvec
//! (weights AND activations encrypted), a scheme switch to a real 3-party
//! ABY3 session for the nonlinearity, and a switch back. The switch-back
//! re-encrypts freshly (`hybrid::shares_to_fhe` → `key_pow = 1`, fresh
//! noise), so **the FHE noise budget is reset at every layer boundary** —
//! the MPC round trip is an *interactive functional bootstrap*. No FHE
//! bootstrapping is ever performed, at any depth.
//!
//! # Why the chain stays exact at arbitrary depth
//!
//! Two independent invariants, both verified empirically (`ok` column =
//! exact bit-equality with the integer plaintext reference at every depth):
//!
//! 1. **Noise never accumulates.** Each layer's FHE ciphertext is depth-1
//!    fresh, so BFV's depth-1 noise budget (measured in `fhe::bfv`'s
//!    `measures_post_multiply_noise_budget`) is the *only* budget that ever
//!    matters, independent of `L`.
//! 2. **Values never overflow.** Layers are fixed-point: the nonlinearity
//!    is the fused `a_next = ReLU(v ≫ s)` with `v = W·a_prev + (b ≪ s)`,
//!    which rescales by `s` bits every layer and so keeps activation
//!    magnitude bounded regardless of `L`. The rescale needs only a
//!    *logical* shift — correct because it is applied solely on the
//!    non-negative (kept) branch of the ReLU — computed from a single
//!    `a2b`, the same `a2b` that yields the ReLU sign bit.
//!
//! # What a pure-FHE competitor would pay instead
//!
//! A leveled-FHE deep net (CKKS/BGV) doing the same ReLUs via polynomial
//! approximation consumes multiplicative levels every layer and must
//! bootstrap when they run out. [`competitor_bootstraps`] estimates that
//! count from *our own measured* per-ReLU level cost
//! (`CkksCompareBackend::levels_for_relu`), with the per-bootstrap level
//! budget as a stated, conservative assumption — the point being the shape
//! (bootstraps grow ∝ depth for them, stay 0 for us), not a fabricated
//! head-to-head timing.
//!
//! # Honesty / scope (carried from the rest of this crate)
//!
//! - The FHE parameters are the secure RLWE ones (`n = 4096`, `q = 2^100`,
//!   inside the 128-bit HomomorphicEncryption.org bound — see `fhe::bfv`).
//!   The *construction* (switch-as-bootstrap) is scheme-agnostic; BFV is
//!   chosen for its cheap relin-free depth-1 multiplication.
//! - The scheme switch simulates threshold decryption with a single key
//!   (`crate::hybrid`); a full instantiation uses distributed-key
//!   threshold FHE.
//! - Weights are encrypted (ct×ct) — a stronger privacy setting than the
//!   plaintext-weight linear layers of Gazelle/Delphi/Cheetah.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::bench::comm::{CommStats, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, sleep_link, MpcTransport};
use crate::fhe::ckks_compare::CkksCompareBackend;
use crate::crypto::types::Fr;
use crate::fhe::bfv::{Bfv, BfvAdapter, BfvCt, BfvParams};
use crate::hybrid::FheAdapter;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

// ─── The integer fixed-point MLP ──────────────────────────────────────────────

#[derive(Clone)]
pub struct Layer {
    /// `out_dim` rows × `in_dim` weights.
    pub w: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
}

pub struct DeepMlp {
    pub layers: Vec<Layer>,
    /// Fixed-point rescale (bits) applied after every layer's matvec.
    pub shift: u32,
    /// Share/plaintext-modulus bit width; every intermediate must stay in
    /// `(-2^(nbits-1), 2^(nbits-1))`.
    pub nbits: usize,
}

impl DeepMlp {
    /// A random square (width→width) `depth`-layer network with weight and
    /// input magnitudes chosen so activations stay bounded across *any*
    /// depth (the rescale `shift` is set to hold the post-matvec magnitude
    /// near the input magnitude). Deterministic in `seed`.
    pub fn random(depth: usize, width: usize, seed: u64) -> (Self, Vec<i64>) {
        // Magnitudes: activations in [0, 2^ACT_BITS), weights in
        // [-2^W_BITS, 2^W_BITS]. Post-matvec |v| ≲ width·2^W_BITS·2^ACT_BITS,
        // so rescaling by that many bits back to ~2^ACT_BITS keeps it flat.
        const ACT_BITS: u32 = 6;
        const W_BITS: u32 = 3;
        let width_bits = (usize::BITS - width.leading_zeros()).max(1);
        let shift = W_BITS + width_bits; // ≈ log2(width·2^W_BITS)
        let nbits = (ACT_BITS + W_BITS + width_bits + 4) as usize; // + headroom

        let mut rng = StdRng::seed_from_u64(seed);
        let wm = 1i64 << W_BITS;
        let layers = (0..depth)
            .map(|_| {
                let w = (0..width)
                    .map(|_| (0..width).map(|_| rng.gen_range(-wm..=wm)).collect())
                    .collect();
                // Bias kept small relative to the pre-rescale accumulator.
                let bias = (0..width)
                    .map(|_| rng.gen_range(-(1i64 << ACT_BITS)..=(1i64 << ACT_BITS)))
                    .collect();
                Layer { w, bias }
            })
            .collect();
        let input: Vec<i64> = (0..width)
            .map(|_| rng.gen_range(0..(1i64 << ACT_BITS)))
            .collect();
        (
            DeepMlp {
                layers,
                shift,
                nbits,
            },
            input,
        )
    }

    /// One layer's fused semantics: `a_next[i] = ReLU((Σ_j W[i][j]·a[j] +
    /// (b[i] ≪ s)) ≫ s)`, where `≫` is exact for the kept (non-negative)
    /// branch. Returns the layer's activation vector.
    fn layer_forward(&self, layer: &Layer, a: &[i64]) -> Vec<i64> {
        layer
            .w
            .iter()
            .zip(&layer.bias)
            .map(|(row, &b)| {
                let dot: i64 = row.iter().zip(a).map(|(&w, &x)| w * x).sum();
                let v = dot + (b << self.shift);
                if v >= 0 {
                    v >> self.shift
                } else {
                    0
                }
            })
            .collect()
    }

    /// Plaintext reference forward pass (the ground truth the encrypted run
    /// must match exactly). Returns the final activation vector.
    pub fn forward(&self, input: &[i64]) -> Vec<i64> {
        let mut a = input.to_vec();
        for layer in &self.layers {
            a = self.layer_forward(layer, &a);
        }
        a
    }

    /// Largest absolute intermediate value over the whole forward pass —
    /// used to assert the fixed-point config never overflows `nbits`.
    fn max_abs_intermediate(&self, input: &[i64]) -> i64 {
        let mut a = input.to_vec();
        let mut m = a.iter().map(|v| v.abs()).max().unwrap_or(0);
        for layer in &self.layers {
            for (row, &b) in layer.w.iter().zip(&layer.bias) {
                let dot: i64 = row.iter().zip(&a).map(|(&w, &x)| w * x).sum();
                let v = dot + (b << self.shift);
                m = m.max(v.abs());
            }
            a = self.layer_forward(layer, &a);
            m = m.max(a.iter().map(|v| v.abs()).max().unwrap_or(0));
        }
        m
    }
}

// ─── MPC middle: import → +bias≪s → fused ReLU-rescale → export ───────────────

fn residue(v: i64, nbits: usize) -> u128 {
    (v as i128).rem_euclid(1i128 << nbits) as u128
}

/// The engine-independent nonlinear step, run on the real ABY3 wire
/// protocol. Given the opened (masked) matvec outputs and the per-party
/// masks, each party reconstructs its share of the raw matvec, adds the
/// public `bias ≪ s`, applies the fused ReLU-rescale, and exports its
/// additive component of the resulting activation. Returns per-party
/// component vectors (index = party) and party-0 wire stats.
#[allow(clippy::type_complexity)]
fn mpc_relu_rescale_layer(
    opened: &[u64],
    masks: &[[u64; 3]],
    bias: &[i64],
    shift: u32,
    nbits: usize,
    profile: NetworkProfile,
) -> (Vec<Vec<Fr>>, CommStats) {
    let width = opened.len();
    let opened = opened.to_vec();
    let masks = masks.to_vec();
    let bias = bias.to_vec();
    run_parties(profile, move |me| {
        let opened = opened.clone();
        let masks = masks.clone();
        let bias = bias.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            // Three phases, so the two interactive steps are issued ONCE for
            // the whole layer instead of once per neuron. The elements are
            // independent, so a per-element loop here would serialise the
            // layer into O(width) rounds for bit-identical output — the same
            // round-structure tax `gadgets::relu` exists to avoid, and the one
            // `bench-hybrid-nn` was paying until it adopted the batched gadget.

            // 1. Import + bias, **batched**: three sharings for the whole
            //    layer (one per owner) plus one for the bias, rather than
            //    `4 * width` interactive calls. Folding is local addition.
            //    Reconstruct this party's component of the masked matvec
            //    output first (party 0 also holds the opened value).
            let my_comps: Vec<Fr> = (0..width)
                .map(|j| {
                    if me == 0 {
                        (opened[j] as u128).wrapping_sub(masks[j][0] as u128)
                    } else {
                        0u128.wrapping_sub(masks[j][me] as u128)
                    }
                })
                .collect();
            let mut raws: Option<Vec<ShareRef>> = None;
            for owner in 0..3 {
                let vals: Vec<Fr> =
                    if me == owner { my_comps.clone() } else { vec![0; width] };
                let shs = be.share_secret_batch(&vals, nbits, owner);
                raws = Some(match raws {
                    None => shs,
                    Some(a) => a.iter().zip(&shs).map(|(x, y)| be.add(x, y)).collect(),
                });
            }
            let bvals: Vec<Fr> = (0..width)
                .map(|j| if me == 0 { residue(bias[j] << shift, nbits) } else { 0 })
                .collect();
            let bshares = be.share_secret_batch(&bvals, nbits, 0);
            let vs: Vec<ShareRef> = raws
                .unwrap_or_default()
                .iter()
                .zip(&bshares)
                .map(|(raw, b)| be.add(raw, b))
                .collect();

            // 2. Fused ReLU∘rescale from ONE **batched** a2b:
            //      bits = a2b(v);  shifted = b2a(bits[s..]) = v≫s;
            //      keep = 1 - sign;  out = shifted · keep.
            //    `a2b_batch` runs the same ~9 rounds as a single decomposition,
            //    just wider; `b2a` is local (Σ bitᵢ·2ⁱ), so it stays per element.
            let all_bits = be.a2b_batch(&vs, nbits);
            let one = be.share_secret(if me == 0 { 1 } else { 0 }, 1, 0);
            let mul_pairs: Vec<(ShareRef, ShareRef)> = all_bits
                .iter()
                .map(|bits| {
                    let shifted = be.b2a(&bits[shift as usize..]);
                    let keep = be.sub(&one, &bits[nbits - 1]);
                    (shifted, keep)
                })
                .collect();

            // 3. One batched multiply for the whole layer.
            be.mul_batch(&mul_pairs)
                .iter()
                .map(|act| be.my_additive_component(act))
                .collect::<Vec<Fr>>()
        }
    })
}

// ─── Per-layer BFV FHE steps ─────────────────────────────────────────────────

/// Ring dimension: the secure `q = 2^100` sits inside the 128-bit RLWE bound
/// (~109) at n=4096, and the reversed packing's `(width+1)·width ≤ n` bound
/// then admits widths up to 63 in a single ciphertext.
const RING_N: usize = 4096;

/// Encrypt an activation vector as a packed BFV input polynomial
/// (value j at coefficient j).
fn pack_activation(s: &Bfv, ck: &[u8], a: &[i64]) -> BfvCt {
    s.encrypt_poly(ck, &s.pack_input(a))
}

/// The full deep inference under the hybrid construction. Returns the
/// revealed final activation vector, per-phase-summed timing, cumulative
/// MPC wire stats, and whether it matched the plaintext reference exactly.
pub struct DeepRun {
    pub output: Vec<i64>,
    pub correct: bool,
    pub total_ms: f64,
    pub fhe_ms: f64,
    pub switch_ms: f64,
    pub mpc_ms: f64,
    pub comm: CommStats,
    pub depth: usize,
    pub width: usize,
    pub nbits: usize,
    pub ct_bytes: usize,
}

pub fn run_deep_hybrid(mlp: &DeepMlp, input: &[i64], profile: NetworkProfile) -> DeepRun {
    let width = input.len();
    let expected = mlp.forward(input);
    // Fixed-point config must never overflow the share width.
    let bound = mlp.max_abs_intermediate(input);
    assert!(
        (bound as i128) < (1i128 << (mlp.nbits - 1)),
        "deep_nn: intermediate magnitude {bound} overflows nbits={}",
        mlp.nbits
    );

    let s = Bfv::new(BfvParams {
        n: RING_N,
        log2_q: 100,
        log2_t: mlp.nbits as u32,
    });
    assert!(
        (width + 1) * width <= s.params.n,
        "deep_nn: width {width} too large to pack in one ciphertext (raise chunking)"
    );
    let nbits = mlp.nbits;
    let t_mask = s.params.t() - 1;
    let adapter = BfvAdapter(&s);

    let start = Instant::now();
    let (mut fhe_ms, mut switch_ms, mut mpc_ms) = (0.0, 0.0, 0.0);
    let mut comm = CommStats::default();
    let mut ct_bytes = 0usize;

    let (ck, _sk) = s.keygen();
    let mut c_act = pack_activation(&s, &ck, input);
    ct_bytes = ct_bytes.max(adapter.ct_bytes(&c_act));
    let mut rng = rand::thread_rng();

    for layer in &mlp.layers {
        // [FHE] one ct×ct matvec: weights reversed-packed, activation
        // packed; row r output at coefficient r·width+width-1. Fresh
        // depth-1 ciphertext (key_pow 1) → product key_pow 2.
        let t0 = Instant::now();
        let c_w = s.encrypt_poly(&ck, &s.pack_weights(&layer.w));
        let c_pre = s.mul(&c_act, &c_w);
        fhe_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // [SWITCH → MPC] mask-and-open: only a uniform-mod-t polynomial is
        // ever decrypted.
        let t0 = Instant::now();
        let masks_poly: Vec<Vec<u64>> = (0..3)
            .map(|_| (0..s.params.n).map(|_| rng.gen::<u64>() & t_mask).collect())
            .collect();
        let mut conv_bytes = 0usize;
        let mut masked = c_pre.clone();
        for m in &masks_poly {
            let enc = s.encrypt_poly(&ck, m);
            conv_bytes += adapter.ct_bytes(&enc);
            masked = s.add(&masked, &enc);
        }
        let opened_poly = s.decrypt_poly(&ck, &masked);
        sleep_link(&profile, conv_bytes + width * 8, 2);
        // Extract the width output positions.
        let mut opened = Vec::with_capacity(width);
        let mut masks = Vec::with_capacity(width);
        for r in 0..width {
            let pos = s.matvec_output_position(width, r);
            opened.push(opened_poly[pos]);
            masks.push([masks_poly[0][pos], masks_poly[1][pos], masks_poly[2][pos]]);
        }
        switch_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // [MPC] fused ReLU-rescale on the real 3-party protocol.
        let t0 = Instant::now();
        let (per_party, layer_comm) =
            mpc_relu_rescale_layer(&opened, &masks, &layer.bias, mlp.shift, nbits, profile);
        comm = CommStats {
            messages: comm.messages + layer_comm.messages,
            bytes: comm.bytes + layer_comm.bytes,
        };
        mpc_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // [SWITCH → FHE] re-encrypt the activation components as the next
        // layer's packed input. This is the interactive bootstrap: the
        // result is a FRESH key_pow-1 ciphertext, noise reset.
        let t0 = Instant::now();
        let mut bytes = 0usize;
        let mut sum: Option<BfvCt> = None;
        for comps in per_party.iter() {
            let m: Vec<u64> = comps.iter().map(|&c| (c as u64) & t_mask).collect();
            let enc = s.encrypt_poly(&ck, &m);
            bytes += adapter.ct_bytes(&enc);
            sum = Some(match sum {
                None => enc,
                Some(acc) => s.add(&acc, &enc),
            });
        }
        sleep_link(&profile, bytes, 1);
        c_act = sum.expect("three parties");
        switch_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }

    // Reveal the final activation vector (coefficients 0..width of the last
    // packed ciphertext) and compare.
    let coeffs = s.decrypt_poly(&ck, &c_act);
    let output: Vec<i64> = (0..width)
        .map(|j| s.to_signed(coeffs[j] & t_mask))
        .collect();

    DeepRun {
        correct: output == expected,
        output,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
        fhe_ms,
        switch_ms,
        mpc_ms,
        comm,
        depth: mlp.layers.len(),
        width,
        nbits,
        ct_bytes,
    }
}

// ─── Competitor bootstrap model (grounded, assumptions stated) ────────────────

/// Estimated FHE bootstraps a leveled-CKKS deep net would need for the same
/// `depth` (ours is always 0). Per layer it spends 1 multiplicative level
/// on the linear map plus `CkksCompareBackend::levels_for_relu(relu_iters)` on the
/// polynomial ReLU; `boot_budget` usable levels are recovered per
/// bootstrap. Returns `(bootstraps, levels_per_layer)`.
pub fn competitor_bootstraps(depth: usize, relu_iters: u32, boot_budget: u32) -> (u64, u32) {
    let per_layer = 1 + CkksCompareBackend::levels_for_relu(relu_iters);
    let total = depth as u64 * per_layer as u64;
    // Start from one fresh budget, bootstrap each time it is exhausted.
    let boots = if total <= boot_budget as u64 {
        0
    } else {
        (total - boot_budget as u64).div_ceil(boot_budget as u64)
    };
    (boots, per_layer)
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct DepthPoint {
    pub depth: usize,
    pub width: usize,
    pub correct: bool,
    pub total_ms: f64,
    pub fhe_ms: f64,
    pub switch_ms: f64,
    pub mpc_ms: f64,
    pub messages: u64,
    pub bytes: u64,
    pub our_bootstraps: u64,
    pub competitor_bootstraps: u64,
}

const USAGE: &str = "usage: baru bench-deep-nn [options]

Deep private inference (fully-connected ReLU network, encrypted weights AND
activations) under the hybrid construction, swept over depth. Demonstrates
ZERO FHE bootstraps at any depth, cost linear in depth, and exact agreement
with the integer plaintext reference at every depth.

options:
  --depths LIST        comma-separated depths to sweep (default 2,4,8,16)
  --width N            layer width (default 16; one ciphertext per layer)
  --relu-iters K       CKKS competitor's polynomial-ReLU iterations (default 2)
  --boot-budget L      CKKS competitor's usable levels per bootstrap (default 20)
  --rtt-ms N           round-trip latency between machines (default 0)
  --bandwidth-mbps X   link bandwidth in Mbit/s (default: unlimited)
  --seed N             RNG seed (default 1)
  --json               emit the sweep as JSON
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut depths = vec![2usize, 4, 8, 16];
    let mut width = 16usize;
    let mut relu_iters = 2u32;
    let mut boot_budget = 20u32;
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
            "--depths" => {
                i += 1;
                depths = args
                    .get(i)
                    .ok_or("--depths needs a value")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().map_err(|_| "bad depth".to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--width" => {
                i += 1;
                width = args.get(i).ok_or("--width needs a value")?.parse().map_err(|_| "bad width".to_string())?;
            }
            "--relu-iters" => {
                i += 1;
                relu_iters = args.get(i).ok_or("--relu-iters needs a value")?.parse().map_err(|_| "bad relu-iters".to_string())?;
            }
            "--boot-budget" => {
                i += 1;
                boot_budget = args.get(i).ok_or("--boot-budget needs a value")?.parse().map_err(|_| "bad boot-budget".to_string())?;
                if boot_budget == 0 {
                    return Err("--boot-budget must be >= 1".into());
                }
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

    if !json {
        println!("Deep private inference — hybrid BFV × ABY3, ZERO FHE bootstraps");
        println!(
            "width={width}, rtt={rtt_ms}ms, bandwidth={}",
            bandwidth_mbps.map(|b| format!("{b}Mbps")).unwrap_or_else(|| "unlimited".into())
        );
        println!(
            "competitor model: CKKS, {} levels/layer (1 linear + {} ReLU@{relu_iters} iters), \
             {boot_budget} levels/bootstrap",
            1 + CkksCompareBackend::levels_for_relu(relu_iters),
            CkksCompareBackend::levels_for_relu(relu_iters)
        );
        println!();
        println!(
            "{:>5}  {:>4}  {:>9}  {:>8} {:>8} {:>8}  {:>8} {:>10}  {:>5}  {:>10}",
            "depth", "ok", "total", "fhe", "switch", "mpc", "msgs", "bytes", "boots", "competitor"
        );
    }

    let mut points = Vec::new();
    for &depth in &depths {
        let (mlp, input) = DeepMlp::random(depth, width, seed);
        let r = run_deep_hybrid(&mlp, &input, profile);
        let (comp_boots, _) = competitor_bootstraps(depth, relu_iters, boot_budget);
        if !json {
            println!(
                "{:>5}  {:>4}  {:>7.0}ms  {:>6.0}ms {:>6.0}ms {:>6.0}ms  {:>8} {:>10}  {:>5}  {:>10}",
                depth,
                if r.correct { "✓" } else { "✗" },
                r.total_ms,
                r.fhe_ms,
                r.switch_ms,
                r.mpc_ms,
                r.comm.messages,
                r.comm.bytes,
                0,
                comp_boots,
            );
        }
        points.push(DepthPoint {
            depth,
            width,
            correct: r.correct,
            total_ms: r.total_ms,
            fhe_ms: r.fhe_ms,
            switch_ms: r.switch_ms,
            mpc_ms: r.mpc_ms,
            messages: r.comm.messages,
            bytes: r.comm.bytes,
            our_bootstraps: 0,
            competitor_bootstraps: comp_boots,
        });
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&points).map_err(|e| e.to_string())?);
    } else {
        println!();
        println!("Every depth: exact match to plaintext, 0 FHE bootstraps.");
        println!("Noise budget is per-layer (depth-1), so it is independent of depth.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plaintext_forward_is_bounded_and_deterministic() {
        let (mlp, input) = DeepMlp::random(8, 16, 1);
        let out = mlp.forward(&input);
        assert_eq!(out.len(), 16);
        let bound = mlp.max_abs_intermediate(&input);
        assert!((bound as i128) < (1i128 << (mlp.nbits - 1)), "config overflows");
        // Determinism.
        let (mlp2, input2) = DeepMlp::random(8, 16, 1);
        assert_eq!(input, input2);
        assert_eq!(out, mlp2.forward(&input2));
    }

    #[test]
    fn competitor_bootstraps_grow_with_depth_and_ours_stay_zero() {
        let (b2, per) = competitor_bootstraps(2, 2, 20);
        let (b16, _) = competitor_bootstraps(16, 2, 20);
        let (b50, _) = competitor_bootstraps(50, 2, 20);
        assert!(per >= 10);
        assert!(b2 < b16 && b16 < b50, "bootstraps must grow with depth: {b2},{b16},{b50}");
    }

    #[test]
    #[ignore] // the n=4096 ring is slow in debug, fast in release
    fn deep_hybrid_exact_at_increasing_depth() {
        for depth in [2usize, 4, 8] {
            let (mlp, input) = DeepMlp::random(depth, 16, 7);
            let r = run_deep_hybrid(&mlp, &input, NetworkProfile::INSTANT);
            assert!(
                r.correct,
                "depth {depth}: encrypted {:?} != plaintext {:?}",
                r.output,
                mlp.forward(&input)
            );
            assert!(r.comm.messages > 0, "must use the real MPC protocol");
        }
    }
}
