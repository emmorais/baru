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
//!   - **BFV** (`fhe::bfv`): RLWE, secure at n=4096, q=2^100.
//!   - **BGV** (`fhe::bgv`): the same construction on the other classic RLWE
//!     encoding, at deliberately identical parameters.
//!
//! Both use the *same* schoolbook-Rust polynomial style and the same
//! reversed-coefficient packing, so any timing difference is the scheme's
//! encoding, not the implementation. (CKKS is
//! measured elsewhere via SEAL — an optimized C++ NTT implementation, so it
//! is not timing-comparable here; its structural cost is noted in the
//! summary.)

use std::time::Instant;

use serde::Serialize;

use crate::fhe::bfv::{Bfv, BfvParams};
use crate::fhe::bgv::{Bgv, BgvParams};
use crate::fhe::poly::{self, Backend};

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
        .map(|r| {
            (0..in_dim)
                .map(|i| (((r * 13 + i * 7) % 101) as i64) - 50)
                .collect()
        })
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
        security_note: "RLWE; n=4096, log q=100 ≤ the 128-bit bound (~109)".into(),
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

/// BGV at **BFV's exact parameters** (n=4096, q=2^100, t=2^20), so the two
/// rows differ only in where the message sits in the ciphertext. The whole
/// point of the row is to isolate the encoding: BGV's multiply is BFV's minus
/// the `t/q` rescale, and its decrypt is a mask rather than a rounded divide.
fn bench_bgv(in_dim: usize, out_dim: usize) -> SchemeReport {
    let s = Bgv::new(BgvParams::DEFAULT);
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
        scheme: "bgv".into(),
        secure: true,
        security_note: "RLWE; same lattice and parameters as BFV".into(),
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
  --poly B   polynomial backend: ntt (default), ntl, or `all` to sweep every
             backend this build has (ntl needs `--features ntl`)
  --reps R   repeat each scheme R times and average (default 1). Run-to-run
             spread is ~1-2%, so comparing schemes that differ by less than
             that needs R > 1 to say anything.
  --json     emit JSON
";

/// Mean of `reps` independent runs, field by field. `correct` is ANDed: one
/// wrong run makes the row wrong, which is the only useful aggregation for a
/// correctness flag.
fn average(runs: Vec<SchemeReport>) -> SchemeReport {
    let k = runs.len() as f64;
    let mut acc = runs[0].clone();
    for r in runs.iter().skip(1) {
        acc.keygen_ms += r.keygen_ms;
        acc.encrypt_ms += r.encrypt_ms;
        acc.multiply_ms += r.multiply_ms;
        acc.decrypt_ms += r.decrypt_ms;
        acc.correct &= r.correct;
    }
    acc.keygen_ms /= k;
    acc.encrypt_ms /= k;
    acc.multiply_ms /= k;
    acc.decrypt_ms /= k;
    acc
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut in_dim = 8usize;
    let mut out_dim = 4usize;
    let mut json = false;
    let mut reps = 1usize;
    let mut sweep: Vec<Backend> = vec![Backend::Ntt];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--in" => {
                i += 1;
                in_dim = args
                    .get(i)
                    .ok_or("--in needs a value")?
                    .parse()
                    .map_err(|_| "bad in".to_string())?;
            }
            "--out" => {
                i += 1;
                out_dim = args
                    .get(i)
                    .ok_or("--out needs a value")?
                    .parse()
                    .map_err(|_| "bad out".to_string())?;
            }
            "--reps" => {
                i += 1;
                reps = args
                    .get(i)
                    .ok_or("--reps needs a value")?
                    .parse()
                    .map_err(|_| "bad reps".to_string())?;
                if reps == 0 {
                    return Err("--reps must be at least 1".into());
                }
            }
            "--poly" => {
                i += 1;
                let v = args.get(i).ok_or("--poly needs a value")?;
                sweep = if v == "all" {
                    poly::available()
                } else {
                    let b = Backend::parse(v)?;
                    if !poly::is_available(b) {
                        return Err(format!(
                            "polynomial backend {v:?} is not in this build (rebuild with --features ntl)"
                        ));
                    }
                    vec![b]
                };
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    let mut runs: Vec<(Backend, Vec<SchemeReport>)> = Vec::new();
    for b in sweep {
        let reports = poly::with_backend(b, || {
            vec![
                average((0..reps).map(|_| bench_bfv(in_dim, out_dim)).collect()),
                average((0..reps).map(|_| bench_bgv(in_dim, out_dim)).collect()),
            ]
        })?;
        runs.push((b, reports));
    }

    if json {
        let payload: Vec<_> = runs
            .iter()
            .map(|(b, r)| serde_json::json!({ "poly_backend": b.name(), "schemes": r }))
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Minimal-FHE inner product — one ct×ct multiply, no relin/bootstrap/eval-keys");
    println!(
        "packed matvec({in_dim}x{out_dim}), both operands encrypted, decrypt under squared key\n"
    );
    for (b, reports) in &runs {
        print_table(*b, reports, runs.len() > 1);
    }
    let reports = &runs.last().expect("at least one backend").1;
    for r in reports {
        println!("  {}: {}", r.scheme, r.security_note);
    }

    // Cross-backend speedup, when more than one was run: the same scheme, the
    // same parameters, only the polynomial arithmetic swapped.
    if runs.len() > 1 {
        let (base_b, base) = &runs[0];
        println!("\n  multiply speedup vs {}:", base_b.name());
        for (b, rs) in runs.iter().skip(1) {
            let line: Vec<String> = rs
                .iter()
                .zip(base)
                .map(|(r, b0)| format!("{} {:.2}x", r.scheme, b0.multiply_ms / r.multiply_ms))
                .collect();
            println!("    {:<4} {}", b.name(), line.join("   "));
        }
    }

    println!(
        "\n  CKKS (SEAL, not timing-comparable — optimized C++ NTT vs schoolbook Rust here):\n\
         \x20   also RLWE-secure and does the same no-relin depth-1 product; measured in\n\
         \x20   bench-hybrid-nn / bench-ckks-compare. Structurally: secure, no eval keys for\n\
         \x20   the coefficient-packed product, larger ciphertexts than BFV at equal security."
    );
    println!(
        "\n  Takeaway: BFV and BGV cost nearly the same — the RLWE encoding is not where the\n\
         \x20 money is. What makes this primitive cheap is the *construction* (one ct×ct\n\
         \x20 multiply, degree-2 decryption, no relinearization or bootstrapping), not the\n\
         \x20 choice of scheme."
    );
    Ok(())
}

fn print_table(b: Backend, reports: &[SchemeReport], label_backend: bool) {
    if label_backend {
        println!("── polynomial backend: {} ──", b.name());
    }
    println!(
        "{:<8} {:>6} {:>4} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10} {:>4}",
        "scheme",
        "secure",
        "eval",
        "n",
        "keygen",
        "encrypt",
        "multiply",
        "decrypt",
        "ct-bytes",
        "ok"
    );
    for r in reports {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scheme_computes_the_inner_product_exactly() {
        // Small dims; correctness of the operation (not timing).
        let bfv = bench_bfv(6, 3);
        assert!(bfv.correct, "BFV inner product must be exact");
        assert!(bfv.secure && !bfv.eval_keys);
        let bgv = bench_bgv(6, 3);
        assert!(bgv.correct, "BGV inner product must be exact");
        assert!(bgv.secure && !bgv.eval_keys);
    }

    /// BGV and BFV are deliberately at identical parameters, so their
    /// ciphertexts must be the same size. If this ever fails, one of the two
    /// parameter sets drifted and the comparison stopped being apples-to-apples.
    #[test]
    fn bgv_and_bfv_are_compared_at_equal_parameters() {
        let bfv = bench_bfv(6, 3);
        let bgv = bench_bgv(6, 3);
        assert_eq!(bgv.n, bfv.n);
        assert_eq!(bgv.log2_q, bfv.log2_q);
        assert_eq!(bgv.ct_bytes, bfv.ct_bytes);
    }
}
