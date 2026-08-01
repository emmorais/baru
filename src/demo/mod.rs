//! `baru demo-private-mnist` — the flagship demo: live private handwritten
//! digit recognition where BOTH the image and the model are encrypted, and
//! the only thing that ever leaves the cryptographic domain is the final
//! class bit.
//!
//! # The cast
//!
//! - **Client** holds a 28×28 handwritten digit (a real MNIST sample from
//!   `examples/data/mnist_data.rs`). It must stay private.
//! - **Model owner** holds trained discriminator weights (784 of them +
//!   bias + calibrated decision threshold). They are IP — also private.
//!   (This is the part existing hybrid-inference systems — Gazelle,
//!   Delphi, Cheetah — do NOT protect: their server sees the model in
//!   plaintext. Our linear phase is ciphertext×ciphertext.)
//! - **Evaluator** (an untrusted server) computes on ciphertexts.
//! - **Three MPC parties** (real ABY3 wire protocol) perform the one
//!   nonlinear step — the threshold decision — on secret shares.
//!
//! # The pipeline
//!
//! 1. Client packs all 784 pixels into ONE BFV ciphertext.
//! 2. Model owner packs the reversed weights into ONE ciphertext.
//! 3. Evaluator computes the entire 784-element dot product with **one**
//!    ciphertext×ciphertext multiplication (the reversed-packing trick),
//!    adds the bias as a plaintext Δ·b term — the result decrypts under
//!    s², no relinearization, no bootstrapping, no evaluation keys.
//! 4. Scheme switch: the score crosses into 3-party ABY3 shares via
//!    mask-and-open — the only value ever opened is a uniformly-masked
//!    polynomial that reveals nothing.
//! 5. The parties compare score vs threshold under MPC (real wire
//!    protocol) and reveal ONLY the resulting bit.
//!
//! Crucially, **nobody ever sees the score** — not even the client. A
//! plain-FHE classifier must decrypt the score to decide, and the score
//! leaks information about the input (this is exactly the leak in the
//! reference hom-ham pipeline, whose `DecryptMul` reveals the raw Hamming
//! distance). Thresholding under MPC closes that leak.
//!
//! Same honesty notes as everywhere else in this repo: the conversion's
//! masked decryption stands in for a threshold decryption (single key,
//! in-process — see `crate::hybrid`), so this is a research demonstration,
//! not a product. The FHE parameters themselves are the secure RLWE ones
//! (`n = 4096`, `q = 2^100` — see `crate::fhe::bfv`).

#[path = "../../examples/data/mnist_data.rs"]
mod mnist_data;

use std::time::Instant;

use rand::Rng;

use crate::bench::comm::{CommStats, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, sleep_link, MpcTransport};
use crate::bench::Phase;
use crate::crypto::types::Fr;
use crate::fhe::bfv::{Bfv, BfvAdapter, BfvParams};
use crate::hybrid::FheAdapter;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

/// The model: handwritten-weight linear discriminator from
/// `examples/data/mnist_data.rs` (score = bias + ⟨image, weights⟩), with a
/// decision threshold calibrated as the midpoint of the two known class
/// scores (433_887 for the "0" sample, 365_186 for the "1" sample —
/// verified in `tests/nn.rs`). Score above threshold ⇒ digit 0.
const BIAS: i64 = 80;
const THRESHOLD: i64 = 399_536;

/// Plaintext modulus: scores are < 2^19; one extra bit of headroom plus
/// the sign bit. (BFV's depth-1 noise budget at `n = 4096, q = 2^100`
/// allows up to ~2^30 — see `fhe::bfv`'s measured budget test.)
const LOG2_T: u32 = 21;

/// Ring dimension. The reversed packing needs `(rows + 1)·d ≤ n`, i.e.
/// `2·784 = 1568 ≤ n` for this one-row model; 4096 is also the dimension
/// at which `q = 2^100` is inside the 128-bit RLWE bound.
const RING_N: usize = 4096;

pub struct DemoRun {
    pub predicted: u8,
    pub expected: u8,
    pub phases: Vec<Phase>,
    pub comm: CommStats,
    pub total_ms: f64,
    pub ct_bytes: usize,
}

fn residue(v: i64, nbits: usize) -> u128 {
    (v as i128).rem_euclid(1i128 << nbits) as u128
}

/// Classify one encrypted image against the encrypted model; returns the
/// predicted digit and per-phase timings. `expected` is only used for the
/// correctness check in the report — it never enters the pipeline.
pub fn classify(img: &[i64], expected: u8, profile: NetworkProfile) -> DemoRun {
    let start = Instant::now();
    let mut phases: Vec<Phase> = Vec::new();
    let mut phase = |name: &'static str, start: Instant| {
        phases.push(Phase {
            name,
            micros: start.elapsed().as_micros(),
        });
    };

    let s = Bfv::new(BfvParams {
        n: RING_N,
        log2_q: 100,
        log2_t: LOG2_T,
    });
    let nbits = LOG2_T as usize;
    let t_mask = s.params.t() - 1;
    let adapter = BfvAdapter(&s);

    let t0 = Instant::now();
    let (ck, _sk) = s.keygen();
    phase("keygen", t0);

    // [1] Client: the whole 784-pixel image in ONE ciphertext.
    let t0 = Instant::now();
    let c_img = s.encrypt_poly(&ck, &s.pack_input(img));
    phase("client-encrypt-image", t0);

    // [2] Model owner: reversed-packed weights in ONE ciphertext.
    let t0 = Instant::now();
    let weights_matrix = vec![mnist_data::WEIGHTS.to_vec()];
    let c_w = s.encrypt_poly(&ck, &s.pack_weights(&weights_matrix));
    phase("model-encrypt-weights", t0);

    let ct_bytes = adapter.ct_bytes(&c_img);
    let t0 = Instant::now();
    sleep_link(&profile, 2 * ct_bytes, 2);
    phase("upload", t0);

    // [3] Evaluator: the ENTIRE dot product = one ct×ct multiplication;
    // bias folded in as a plaintext Δ·b addition. Decrypts under s².
    let t0 = Instant::now();
    let pos = s.matvec_output_position(img.len(), 0);
    let mut c_score = s.mul(&c_img, &c_w);
    c_score = s.add_plain_at(&c_score, pos, residue(BIAS, nbits) as u64);
    phase("dot-product(1-mult)", t0);

    // [4] Scheme switch: mask-and-open — the opened polynomial is uniform
    // mod t in every coefficient; the score never appears in the clear.
    let t0 = Instant::now();
    let mut rng = rand::thread_rng();
    let masks: Vec<Vec<u64>> = (0..3)
        .map(|_| (0..s.params.n).map(|_| rng.gen::<u64>() & t_mask).collect())
        .collect();
    let mut conv_bytes = 0usize;
    let mut masked = c_score.clone();
    for m in &masks {
        let enc = s.encrypt_poly(&ck, m);
        conv_bytes += adapter.ct_bytes(&enc);
        masked = s.add(&masked, &enc);
    }
    let opened = s.decrypt_poly(&ck, &masked);
    sleep_link(&profile, conv_bytes + opened.len() * 8, 2);
    phase("switch-to-mpc", t0);

    // [5] Three-party ABY3 (real wire protocol): rebuild the score share,
    // compare against the threshold, reveal ONLY the decision bit.
    let t0 = Instant::now();
    let y = opened[pos];
    let (bits, comm) = run_parties(profile, |me| {
        let my_mask = masks[me][pos];
        move |be: &mut TransportBackend<MpcTransport>| {
            let my_comp: Fr = if me == 0 {
                (y as u128).wrapping_sub(my_mask as u128)
            } else {
                0u128.wrapping_sub(my_mask as u128)
            };
            let mut score: Option<ShareRef> = None;
            for owner in 0..3 {
                let v = if me == owner { my_comp } else { 0 };
                let sref = be.share_secret(v, nbits, owner);
                score = Some(match score {
                    None => sref,
                    Some(a) => be.add(&a, &sref),
                });
            }
            let score = score.unwrap();
            let thr = be.share_secret(
                if me == 0 {
                    residue(THRESHOLD, nbits)
                } else {
                    0
                },
                nbits,
                0,
            );
            let is_zero = be.lt(&thr, &score); // threshold < score ⇒ digit 0
            be.reveal(&is_zero, None) & 1
        }
    });
    phase("mpc-threshold", t0);
    let predicted = if bits[0] == 1 { 0u8 } else { 1u8 };

    DemoRun {
        predicted,
        expected,
        phases,
        comm,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
        ct_bytes,
    }
}

const USAGE: &str = "usage: baru demo-private-mnist [options]

Private handwritten-digit recognition, end to end: the image is encrypted,
the model weights are encrypted, the 784-pixel dot product is ONE
homomorphic multiplication (BFV, decrypted under s² — no
bootstrapping, no evaluation keys), and the class decision is made by a
real 3-party ABY3 threshold comparison that reveals ONLY the final bit —
nobody, including the client, ever sees the score.

options:
  --rtt-ms N           round-trip latency between machines (default 0)
  --bandwidth-mbps X   link bandwidth in Mbit/s (default: unlimited)
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut rtt_ms: u64 = 0;
    let mut bandwidth_mbps: Option<f64> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--rtt-ms" => {
                i += 1;
                rtt_ms = args
                    .get(i)
                    .ok_or("--rtt-ms needs a value")?
                    .parse()
                    .map_err(|_| "--rtt-ms needs a non-negative integer".to_string())?;
            }
            "--bandwidth-mbps" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("--bandwidth-mbps needs a value")?
                    .parse()
                    .map_err(|_| "--bandwidth-mbps needs a number".to_string())?;
                bandwidth_mbps = Some(v);
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }
    let mut profile = NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt_ms * 500));
    if let Some(mbps) = bandwidth_mbps {
        profile = profile.with_bandwidth(mbps * 1e6 / 8.0);
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  PRIVATE MNIST INFERENCE — encrypted image × ENCRYPTED model     ║");
    println!("║  BFV (depth-1, no bootstrapping) + real 3-party ABY3             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("ring X^{RING_N}+1, q=2^100, t=2^{LOG2_T} | 784-pixel image = ONE ciphertext");
    println!(
        "network: rtt={rtt_ms}ms, bandwidth={}",
        bandwidth_mbps
            .map(|b| format!("{b}Mbps"))
            .unwrap_or_else(|| "unlimited".into())
    );
    println!();

    for (label, img, expected) in [
        ("handwritten '0' sample", &mnist_data::IMG_ZERO[..], 0u8),
        ("handwritten '1' sample", &mnist_data::IMG_ONE[..], 1u8),
    ] {
        let r = classify(img, expected, profile);
        println!("── {label} ──");
        for p in &r.phases {
            println!("   {:<24} {:>8.1}ms", p.name, p.micros as f64 / 1000.0);
        }
        println!(
            "   MPC wire traffic: {} messages, {} bytes (threshold decision only)",
            r.comm.messages, r.comm.bytes
        );
        println!(
            "   ciphertext size: {:.1} KB | total: {:.0} ms",
            r.ct_bytes as f64 / 1024.0,
            r.total_ms
        );
        println!(
            "   ⇒ prediction: digit {}   ({})",
            r.predicted,
            if r.predicted == r.expected {
                "correct"
            } else {
                "WRONG"
            }
        );
        println!();
    }

    println!("Who learned what:");
    println!("   evaluator     — ciphertexts and one uniformly-masked polynomial: nothing");
    println!("   MPC parties   — uniform shares and masks: nothing");
    println!("   model owner   — nothing about the image");
    println!("   client        — nothing about the weights, NOT EVEN the score:");
    println!("                   only the single output bit (the predicted digit)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // the n=4096 ring is slow in debug; fast in release
    fn both_repo_samples_classify_correctly_end_to_end() {
        let zero = classify(&mnist_data::IMG_ZERO, 0, NetworkProfile::INSTANT);
        assert_eq!(zero.predicted, 0, "the '0' sample must classify as 0");
        let one = classify(&mnist_data::IMG_ONE, 1, NetworkProfile::INSTANT);
        assert_eq!(one.predicted, 1, "the '1' sample must classify as 1");
        assert!(
            zero.comm.messages > 0,
            "the decision must be a real MPC protocol"
        );
    }
}
