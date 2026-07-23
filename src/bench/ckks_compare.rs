//! `baru bench-ckks-compare` — measures the actual, previously-unmeasured
//! cost of `fhe::ckks_compare`'s polynomial-approximation ReLU: the answer
//! to "CKKS can't do comparisons" is "it can approximate them, at a real,
//! measurable cost in a much deeper (and much slower) parameter set than
//! the shallow CKKS path `bench-nn` otherwise uses." This harness reports
//! that cost directly instead of leaving it as an assumption.
//!
//! For each iteration count (== precision knob, see `ckks_compare` module
//! docs), it:
//!   - picks the smallest ring dimension that can afford the required
//!     modulus chain (`CkksCompareBackend::new` — looked up from SEAL's own
//!     security tables, not guessed),
//!   - keygens, encrypts a handful of values spanning the domain,
//!   - times a single `relu_approx` call,
//!   - decrypts and reports error against BOTH the exact ReLU (the
//!     approximation's real accuracy) and the plain-`f64` reference
//!     `relu_approx_plain` at the same iteration count (isolating actual
//!     SEAL ciphertext noise from the polynomial's own, already-measured-in
//!     approximation error).
//!
//! Every number here is measured on this machine at run time — no part of
//! this module hardcodes an expected timing or error figure.

use std::time::Instant;

use serde::Serialize;

use crate::fhe::ckks_compare::{relu_approx_plain, CkksCompareBackend};

#[derive(Debug, Clone, Serialize)]
pub struct CkksCompareReport {
    pub iterations: u32,
    pub levels: u32,
    pub degree: u64,
    pub keygen_ms: f64,
    pub encrypt_ms: f64,
    pub relu_ms: f64,
    pub decrypt_ms: f64,
    pub ciphertext_bytes: usize,
    /// Worst-case |ckks - exact ReLU| over the sample points — the
    /// approximation's real accuracy, including both polynomial error and
    /// SEAL ciphertext noise.
    pub max_abs_error_vs_exact: f64,
    /// Worst-case |ckks - relu_approx_plain| over the sample points —
    /// isolates SEAL's own ciphertext noise from the (already-quantified,
    /// separately-testable) polynomial approximation error.
    pub max_abs_error_vs_plain_approx: f64,
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Sample points spanning the domain: both signs, near zero (slow-converging,
/// see `ckks_compare` module docs) and far from it.
fn sample_points(bound: f64) -> Vec<f64> {
    vec![
        bound * 0.9,
        bound * 0.5,
        bound * 0.1,
        -bound * 0.1,
        -bound * 0.5,
        -bound * 0.9,
        0.0,
    ]
}

/// Run one (iterations, bound) configuration. Times a single `relu_approx`
/// call on the first sample point; measures accuracy across every sample.
pub fn run_one(iterations: u32, bound: f64) -> CkksCompareReport {
    let levels = CkksCompareBackend::levels_for_relu(iterations);

    let t0 = Instant::now();
    let client_be = CkksCompareBackend::new(levels);
    let (ck, sk) = client_be.keygen();
    let keygen_ms = ms(t0.elapsed());
    let degree = client_be.degree();

    let be = CkksCompareBackend::with_degree_and_server_keys(
        degree.try_into().expect("degree fits DegreeType"),
        levels,
        &sk,
    );

    let samples = sample_points(bound);

    let t0 = Instant::now();
    let cts: Vec<_> = samples.iter().map(|&x| be.encrypt(&ck, x)).collect();
    let encrypt_ms = ms(t0.elapsed()) / samples.len() as f64;
    let ciphertext_bytes = be.serialize_ct(&cts[0]).len();

    let t0 = Instant::now();
    let outs: Vec<_> = cts
        .iter()
        .map(|ct| be.relu_approx(ct, bound, iterations))
        .collect();
    let relu_ms = ms(t0.elapsed()) / samples.len() as f64;

    let t0 = Instant::now();
    let decrypted: Vec<f64> = outs.iter().map(|ct| be.decrypt(&ck, ct)).collect();
    let decrypt_ms = ms(t0.elapsed()) / samples.len() as f64;

    let mut max_abs_error_vs_exact = 0.0f64;
    let mut max_abs_error_vs_plain_approx = 0.0f64;
    for (&x, &got) in samples.iter().zip(&decrypted) {
        max_abs_error_vs_exact = max_abs_error_vs_exact.max((got - x.max(0.0)).abs());
        let plain_approx = relu_approx_plain(x, bound, iterations);
        max_abs_error_vs_plain_approx =
            max_abs_error_vs_plain_approx.max((got - plain_approx).abs());
    }

    CkksCompareReport {
        iterations,
        levels,
        degree,
        keygen_ms,
        encrypt_ms,
        relu_ms,
        decrypt_ms,
        ciphertext_bytes,
        max_abs_error_vs_exact,
        max_abs_error_vs_plain_approx,
    }
}

const USAGE: &str = "usage: baru bench-ckks-compare [options]

Measures the real cost of fhe::ckks_compare's polynomial-approximation
ReLU/comparison (the answer to \"CKKS can't do comparisons\") across
iteration counts, using SEAL's own security tables to pick the smallest
sufficient ring dimension for each.

options:
  --iterations LIST   comma-separated iteration counts (default: 1,2,3)
  --bound N           assumed |pre-activation| bound (default: 5.0)
  --json              emit the report as JSON instead of a table
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut iterations: Vec<u32> = vec![1, 2, 3];
    let mut bound: f64 = 5.0;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--iterations" => {
                i += 1;
                let list = args.get(i).ok_or("--iterations needs a value")?;
                iterations = list
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<u32>()
                            .map_err(|_| format!("--iterations: '{s}' is not a non-negative integer"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--bound" => {
                i += 1;
                bound = args
                    .get(i)
                    .ok_or("--bound needs a value")?
                    .parse()
                    .map_err(|_| "--bound needs a number".to_string())?;
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    eprintln!(
        "warning: this exercises a deep (many-level) CKKS context — each configuration can \
         take from seconds to low minutes, growing fast with --iterations. Use --release."
    );

    let reports: Vec<CkksCompareReport> = iterations.iter().map(|&k| run_one(k, bound)).collect();

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!(
        "{:>4}  {:>6}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>10}  {:>14}  {:>16}",
        "iter",
        "levels",
        "degree",
        "keygen",
        "encrypt",
        "relu",
        "decrypt",
        "ct-bytes",
        "err-vs-exact",
        "err-vs-plain-approx"
    );
    for r in &reports {
        println!(
            "{:>4}  {:>6}  {:>7}  {:>7.1}ms  {:>7.1}ms  {:>7.1}ms  {:>7.1}ms  {:>10}  {:>14.6}  {:>16.2e}",
            r.iterations,
            r.levels,
            r.degree,
            r.keygen_ms,
            r.encrypt_ms,
            r.relu_ms,
            r.decrypt_ms,
            r.ciphertext_bytes,
            r.max_abs_error_vs_exact,
            r.max_abs_error_vs_plain_approx
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // slow: real SEAL keygen + a deep modulus chain
    fn run_one_reports_plausible_numbers() {
        let r = run_one(2, 5.0);
        assert_eq!(r.levels, CkksCompareBackend::levels_for_relu(2));
        assert!(r.degree >= 8192);
        assert!(r.ciphertext_bytes > 0);
        // SEAL noise should be far smaller than the polynomial's own
        // approximation error at this shallow iteration count.
        assert!(r.max_abs_error_vs_plain_approx < 1e-2);
    }
}
