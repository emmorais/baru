//! `baru bench-fhe-innerprod` — the E9 comparison: the **minimal-FHE inner
//! product** (one ciphertext×ciphertext multiply, **no relinearization, no
//! bootstrapping, no evaluation keys**, decrypt the degree-2 product under
//! the squared key) across FHE schemes.
//!
//! The point of the paper's C1 claim is that MPC-assisted inference needs
//! only the cheapest FHE primitive — a single multiply — so the natural
//! question is *which scheme is cheapest for it, among the secure ones*.
//! This harness measures, per scheme, a packed private matvec (both operands
//! encrypted, reversed-coefficient packing → all rows in one multiply):
//! keygen / encrypt / multiply / decrypt time, ciphertext size, and — the
//! honest column — the security status of the parameters used.
//!
//! Schemes:
//!   - **BFV** (`fhe::bfv`): RLWE, **secure** at n=4096, q=2^100 (no fatigue
//!     point). The recommended instantiation.
//!   - **SI-NTRU** (`fhe::si_ntru`): the fast *reference* — cheapest
//!     arithmetic (n=2048) but overstretched/**insecure** (see its docs).
//!
//! Both use the *same* schoolbook-Rust polynomial style and the same
//! reversed-coefficient packing, so the timing difference is the scheme
//! (and its secure ring dimension), not the implementation. (CKKS is
//! measured elsewhere via SEAL — an optimized C++ NTT implementation, so it
//! is not timing-comparable here; its structural cost is noted in the
//! summary.)

use std::time::Instant;

use serde::Serialize;

use crate::fhe::bfv::{Bfv, BfvParams};
use crate::fhe::si_ntru::{SiNtru, SiNtruParams};

#[derive(Debug, Clone, Serialize)]
pub struct SchemeReport {
    pub scheme: String,
    pub secure: bool,
    pub security_note: String,
    pub n: usize,
    pub log2_q: u32,
    pub eval_keys: bool,
    pub keygen_ms: f64,
    pub encrypt_ms: f64,
    pub multiply_ms: f64,
    pub decrypt_ms: f64,
    pub ct_bytes: usize,
    pub correct: bool,
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// A deterministic matvec problem (both operands encrypted).
fn problem(in_dim: usize, out_dim: usize) -> (Vec<i64>, Vec<Vec<i64>>, Vec<i64>) {
    let x: Vec<i64> = (0..in_dim).map(|i| ((i * 7) % 101) as i64 - 50).collect();
    let w: Vec<Vec<i64>> = (0..out_dim)
        .map(|r| (0..in_dim).map(|i| (((r * 13 + i * 7) % 101) as i64) - 50).collect())
        .collect();
    let expected: Vec<i64> = w
        .iter()
        .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
        .collect();
    (x, w, expected)
}

fn bench_bfv(in_dim: usize, out_dim: usize) -> SchemeReport {
    let s = Bfv::new(BfvParams::DEFAULT);
    let (x, w, expected) = problem(in_dim, out_dim);

    let t0 = Instant::now();
    let (ck, pk) = s.keygen();
    let keygen_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let cx = s.encrypt_poly(&pk, &s.pack_input(&x));
    let cw = s.encrypt_poly(&pk, &s.pack_weights(&w));
    let encrypt_ms = ms(t0.elapsed()) / 2.0;
    let ct_bytes = s.ct_bytes(&cx);

    let t0 = Instant::now();
    let prod = s.mul(&cx, &cw);
    let multiply_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let out = s.decrypt_poly(&ck, &prod);
    let decrypt_ms = ms(t0.elapsed());

    let got: Vec<i64> = (0..out_dim)
        .map(|r| s.to_signed(out[s.matvec_output_position(in_dim, r)]))
        .collect();

    SchemeReport {
        scheme: "bfv".into(),
        secure: true,
        security_note: "RLWE, no fatigue point; n=4096, log q=100 ≤ 128-bit bound (~109)".into(),
        n: s.params.n,
        log2_q: s.params.log2_q,
        eval_keys: false,
        keygen_ms,
        encrypt_ms,
        multiply_ms,
        decrypt_ms,
        ct_bytes,
        correct: got == expected,
    }
}

fn bench_si_ntru(in_dim: usize, out_dim: usize) -> SchemeReport {
    let s = SiNtru::new(SiNtruParams::DEFAULT);
    let (x, w, expected) = problem(in_dim, out_dim);

    let t0 = Instant::now();
    let (ck, _sk) = s.keygen();
    let keygen_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let cx = s.encrypt_poly(&ck, &s.pack_input(&x));
    let cw = s.encrypt_poly(&ck, &s.pack_weights(&w));
    let encrypt_ms = ms(t0.elapsed()) / 2.0;
    let ct_bytes = cx.poly.len() * 16 + 4;

    let t0 = Instant::now();
    let prod = s.mul(&cx, &cw);
    let multiply_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let out = s.decrypt_poly(&ck, &prod);
    let decrypt_ms = ms(t0.elapsed());

    let got: Vec<i64> = (0..out_dim)
        .map(|r| s.to_signed(out[s.matvec_output_position(in_dim, r)] & (s.params.t() - 1)))
        .collect();

    SchemeReport {
        scheme: "si-ntru".into(),
        secure: false,
        security_note: "overstretched NTRU; INSECURE even at depth 1 (fast reference only)".into(),
        n: s.params.n,
        log2_q: s.params.log2_q,
        eval_keys: false,
        keygen_ms,
        encrypt_ms,
        multiply_ms,
        decrypt_ms,
        ct_bytes,
        correct: got == expected,
    }
}

const USAGE: &str = "usage: baru bench-fhe-innerprod [options]

Compare FHE schemes on the minimal-FHE inner product: one ciphertext×
ciphertext multiply, NO relinearization / bootstrapping / evaluation keys,
degree-2 product decrypted under the squared key. Reports per-scheme
keygen/encrypt/multiply/decrypt time, ciphertext size, and security status.

options:
  --in D     input dimension (default 8)
  --out M    output dimension / rows (default 4)
  --json     emit JSON
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut in_dim = 8usize;
    let mut out_dim = 4usize;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--in" => { i += 1; in_dim = args.get(i).ok_or("--in needs a value")?.parse().map_err(|_| "bad in".to_string())?; }
            "--out" => { i += 1; out_dim = args.get(i).ok_or("--out needs a value")?.parse().map_err(|_| "bad out".to_string())?; }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    let reports = vec![bench_bfv(in_dim, out_dim), bench_si_ntru(in_dim, out_dim)];

    if json {
        println!("{}", serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?);
        return Ok(());
    }

    println!("Minimal-FHE inner product — one ct×ct multiply, no relin/bootstrap/eval-keys");
    println!("packed matvec({in_dim}x{out_dim}), both operands encrypted, decrypt under squared key\n");
    println!(
        "{:<8} {:>6} {:>4} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10} {:>4}",
        "scheme", "secure", "eval", "n", "keygen", "encrypt", "multiply", "decrypt", "ct-bytes", "ok"
    );
    for r in &reports {
        println!(
            "{:<8} {:>6} {:>4} {:>6} {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>7.1}ms {:>10} {:>4}",
            r.scheme,
            if r.secure { "yes" } else { "NO" },
            if r.eval_keys { "yes" } else { "no" },
            r.n,
            r.keygen_ms,
            r.encrypt_ms,
            r.multiply_ms,
            r.decrypt_ms,
            r.ct_bytes,
            if r.correct { "✓" } else { "✗" },
        );
    }
    println!();
    for r in &reports {
        println!("  {}: {}", r.scheme, r.security_note);
    }
    println!(
        "\n  CKKS (SEAL, not timing-comparable — optimized C++ NTT vs schoolbook Rust here):\n\
         \x20   also RLWE-secure and does the same no-relin depth-1 product; measured in\n\
         \x20   bench-hybrid-nn / bench-ckks-compare. Structurally: secure, no eval keys for\n\
         \x20   the coefficient-packed product, larger ciphertexts than BFV at equal security."
    );
    println!(
        "\n  Takeaway: BFV is the recommended *secure* instantiation. SI-NTRU is faster\n\
         \x20 (smaller ring) but insecure — its speed is the cost of the security gap, which\n\
         \x20 quantifies exactly what a secure deployment pays."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_schemes_compute_the_inner_product_exactly() {
        // Small dims; correctness of the operation (not timing).
        let bfv = bench_bfv(6, 3);
        assert!(bfv.correct, "BFV inner product must be exact");
        assert!(bfv.secure && !bfv.eval_keys);
        let ntru = bench_si_ntru(6, 3);
        assert!(ntru.correct, "SI-NTRU inner product must be exact");
        assert!(!ntru.secure);
    }
}
