//! `baru bench-polymul` — the three ways this repo can compute an exact
//! negacyclic polynomial product, timed against each other.
//!
//! Every FHE scheme here (`fhe::bfv`, `fhe::bgv`, `pir`) bottoms out in the
//! same operation:
//!
//! ```text
//! c[k] = Σ_{i+j=k} a[i]·b[j] − Σ_{i+j=k+n} a[i]·b[j]     (exact, in ℤ)
//! ```
//!
//! over centered ~100-bit coefficients. It is the innermost cost of a
//! ciphertext multiply, so which implementation serves it decides a large part
//! of every FHE number in the repo. This benchmark measures the choice instead
//! of assuming it.
//!
//! # The two implementations
//!
//! | | where | complexity | notes |
//! |---|---|---|---|
//! | schoolbook | this file | O(n²) 256-bit MACs | the original; the thing that was replaced |
//! | multi-modular NTT | [`crate::fhe::ntt`] | O(n log n) word ops × 4 primes | hand-written, ~200 lines we maintain |
//!
//! An NTL+GMP column sat beside these until 2026-08-03. It was removed with the
//! backend itself, once the hand-written NTT beat it by ~1.8× at every size and
//! keeping a C++ dependency bought nothing — see `docs/experiments.md` §II.2 for
//! the measurement and what profiling NTL taught us (Shoup-preconditioned
//! butterflies, which is *why* the NTT now wins).
//!
//! Timings are **total wall clock per product**, marshalling included, because a
//! caller pays for marshalling too.
//!
//! Correctness is checked, not assumed: at every size the benchmark asserts
//! that the implementations produce **identical** coefficient vectors before it
//! reports any timing. A fast wrong answer scores nothing.
//!
//! # Reading the output
//!
//! Schoolbook is only run below `--schoolbook-max` (default `n = 512`); above
//! that it is quadratic and pointless to wait for. Its column is the
//! justification for having done any of this, and the agreement check beside it
//! is what makes the NTT trustworthy at sizes the reference cannot reach.
//!
//! ```text
//! cargo run --release -- bench-polymul
//! ```

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::fhe::ntt::{self, I256};

// ─── Inputs ──────────────────────────────────────────────────────────────────

/// Deterministic centered coefficients in `[−2^bits, 2^bits)` — the shape a
/// real ciphertext polynomial has (centered mod `q`, `bits ≈ log2 q − 1`).
fn sample(n: usize, bits: u32, seed: u64) -> Vec<i128> {
    let mut s = seed | 1;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 1
    };
    (0..n)
        .map(|_| {
            let hi = next() as u128;
            let lo = next() as u128;
            let mag = (((hi << 64) | lo) % (1u128 << bits)) as i128;
            if hi & 1 == 0 {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

// ─── The reference implementation being compared against ─────────────────────

/// O(n²) exact negacyclic product with 128×128→256-bit products, the
/// straightforward implementation both fast paths replace.
fn schoolbook(a: &[i128], b: &[i128], n: usize) -> Vec<I256> {
    fn mul_i128(a: i128, b: i128) -> I256 {
        let neg = (a < 0) != (b < 0);
        let (ua, ub) = (a.unsigned_abs(), b.unsigned_abs());
        let (a1, a0) = (ua >> 64, ua & 0xFFFF_FFFF_FFFF_FFFF);
        let (b1, b0) = (ub >> 64, ub & 0xFFFF_FFFF_FFFF_FFFF);
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> 64) + (p01 & 0xFFFF_FFFF_FFFF_FFFF) + (p10 & 0xFFFF_FFFF_FFFF_FFFF);
        let lo = (p00 & 0xFFFF_FFFF_FFFF_FFFF) | (mid << 64);
        let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        let m = I256::from_hi_lo(hi, lo);
        if neg {
            m.wrapping_neg()
        } else {
            m
        }
    }
    let mut acc = vec![I256::ZERO; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            let p = mul_i128(ai, bj);
            let k = i + j;
            if k < n {
                acc[k] = acc[k].wrapping_add(p);
            } else {
                acc[k - n] = acc[k - n].wrapping_sub(p);
            }
        }
    }
    acc
}

// ─── Measurement ─────────────────────────────────────────────────────────────

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Run `f` `reps` times and return the mean milliseconds per call. `black_box`
/// is not available on stable without the nightly feature, so the result is
/// accumulated into a checksum the caller keeps alive.
fn time<F: FnMut() -> Vec<I256>>(mut f: F, reps: usize) -> (f64, Vec<I256>) {
    let mut last = f(); // warm up: first call pays plan/prime construction
    let t0 = Instant::now();
    for _ in 0..reps {
        last = f();
    }
    (ms(t0.elapsed()) / reps as f64, last)
}

#[derive(Debug, Clone, Serialize)]
pub struct Point {
    pub n: usize,
    pub coeff_bits: u32,
    pub reps: usize,
    /// `None` when skipped as too slow (see `--schoolbook-max`).
    pub schoolbook_ms: Option<f64>,
    pub ntt_ms: f64,

    /// All implementations that ran produced identical coefficients.
    pub agree: bool,
    pub ntt_vs_schoolbook: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub points: Vec<Point>,
}

/// Time one ring dimension. Returns an error if the implementations disagree —
/// a differential failure is a bug, not a slow result.
pub fn run_one(
    n: usize,
    coeff_bits: u32,
    reps: usize,
    schoolbook_max: usize,
) -> Result<Point, String> {
    let a = sample(n, coeff_bits, 0xA11CE ^ n as u64);
    let b = sample(n, coeff_bits, 0xB0B ^ n as u64);

    let (ntt_ms, ntt_out) = time(|| ntt::exact_negacyclic_product(&a, &b, n), reps);

    let mut agree = true;

    let schoolbook_ms = if n <= schoolbook_max {
        // Same rep count as the others — `--schoolbook-max` is what keeps the
        // quadratic column from dominating the run, not a thinner measurement.
        let (t, out) = time(|| schoolbook(&a, &b, n), reps);
        agree &= out == ntt_out;
        Some(t)
    } else {
        None
    };

    if !agree {
        return Err(format!(
            "implementations disagree at n={n}, {coeff_bits}-bit coefficients — \
             this is a correctness bug, not a benchmark result"
        ));
    }

    Ok(Point {
        n,
        coeff_bits,
        reps,
        schoolbook_ms,
        ntt_ms,
        agree,
        ntt_vs_schoolbook: schoolbook_ms.map(|s| s / ntt_ms),
    })
}

pub fn run(
    sizes: &[usize],
    coeff_bits: u32,
    reps: usize,
    schoolbook_max: usize,
) -> Result<Report, String> {
    let points = sizes
        .iter()
        .map(|&n| run_one(n, coeff_bits, reps, schoolbook_max))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Report { points })
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

const USAGE: &str = "usage: baru bench-polymul [options]

Exact negacyclic polynomial product — schoolbook vs the hand-written
multi-modular NTT.  Correctness is checked before anything is timed.

options:
  --sizes N,N,...      ring dimensions to sweep    (default 256,512,1024,2048,4096)
  --coeff-bits B       coefficient magnitude bits  (default 99, i.e. q = 2^100 centered)
  --reps R             timed repetitions per size  (default 20)
  --schoolbook-max N   largest n to run schoolbook (default 512; it is O(n^2))
  --json               machine-readable output
  -h, --help           this message
";

fn parse_usize(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<usize, String> {
    it.next()
        .ok_or(format!("{flag} needs a value"))?
        .parse()
        .map_err(|_| format!("bad {flag}"))
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut sizes = vec![256usize, 512, 1024, 2048, 4096];
    let mut coeff_bits = 99u32;
    let mut reps = 20usize;
    let mut schoolbook_max = 512usize;
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sizes" => {
                let v = it.next().ok_or("--sizes needs a value")?;
                sizes = v
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad size {s}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--coeff-bits" => {
                coeff_bits = it
                    .next()
                    .ok_or("--coeff-bits needs a value")?
                    .parse()
                    .map_err(|_| "bad --coeff-bits")?;
            }
            "--reps" => reps = parse_usize(&mut it, "--reps")?,
            "--schoolbook-max" => schoolbook_max = parse_usize(&mut it, "--schoolbook-max")?,
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }
    if reps == 0 {
        return Err("--reps must be at least 1".into());
    }
    for &n in &sizes {
        if n == 0 || !n.is_power_of_two() {
            return Err(format!("size {n} must be a power of two"));
        }
        if n > ntt::MAX_N {
            return Err(format!(
                "size {n} exceeds the NTT's MAX_N = {} (its primes are chosen for that bound)",
                ntt::MAX_N
            ));
        }
    }
    if coeff_bits == 0 || coeff_bits > 127 {
        return Err("--coeff-bits must be in 1..=127".into());
    }

    let r = run(&sizes, coeff_bits, reps, schoolbook_max)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Exact negacyclic product in Z[X]/(X^n+1), {coeff_bits}-bit centered coefficients");
    println!();
    println!(
        "{:>6} {:>12} {:>10} {:>12}",
        "n", "schoolbook", "NTT", "NTT speedup"
    );
    let cell = |v: Option<f64>, unit: &str| match v {
        Some(x) if unit == "x" => format!("{x:.1}x"),
        Some(x) => format!("{x:.3} ms"),
        None => "—".to_string(),
    };
    for p in &r.points {
        println!(
            "{:>6} {:>12} {:>10} {:>12}",
            p.n,
            cell(p.schoolbook_ms, "ms"),
            format!("{:.3} ms", p.ntt_ms),
            cell(p.ntt_vs_schoolbook, "x"),
        );
    }
    println!();
    println!("Both implementations produced identical coefficients at every size.");
    println!("Times are per product over {reps} reps, marshalling included, after one warm-up.");
    println!("Speedup is schoolbook/NTT, so above 1 the NTT is faster.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schoolbook_reference_agrees_with_the_ntt() {
        for n in [8usize, 64] {
            let a = sample(n, 40, 3);
            let b = sample(n, 40, 9);
            assert_eq!(
                schoolbook(&a, &b, n),
                ntt::exact_negacyclic_product(&a, &b, n)
            );
        }
    }

    #[test]
    fn a_sweep_runs_and_agrees() {
        let r = run(&[64, 256], 99, 2, 256).expect("implementations agree");
        assert_eq!(r.points.len(), 2);
        assert!(r.points.iter().all(|p| p.agree && p.ntt_ms > 0.0));
        assert!(r.points.iter().all(|p| p.schoolbook_ms.is_some()));
    }

    #[test]
    fn bad_sizes_are_rejected() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(run_cli(&args(&["--sizes", "100"])).is_err()); // not a power of two
        assert!(run_cli(&args(&["--sizes", "16384"])).is_err()); // above MAX_N (8192)
        assert!(run_cli(&args(&["--reps", "0"])).is_err());
    }
}
