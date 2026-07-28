//! `baru bench-ot` — measured oblivious-transfer cost.
//!
//! Two reasons this benchmark exists rather than a note in the OT module:
//!
//! 1. **It is the crossover argument.** Base OT costs public-key operations per
//!    transfer; the extension costs `κ = 128` base OTs once and then a hash and
//!    `κ` bits each. Where those two curves cross is what decides whether a
//!    protocol should do OT directly or extend, and it is a measurement rather
//!    than a matter of opinion.
//! 2. **It is the input the cost model is missing.** `mpcc::protocols` prices
//!    Semi2k and MASCOT triples from *literature estimates* for the offline
//!    phase — flagged as such in `docs/mpc-compiler-roadmap.md`. Those triples
//!    come from OT. Measured OT throughput and bytes-per-OT are what let those
//!    entries become measurements, which is P1 of that roadmap.
//!
//! # What is measured (2026-07-28, M-series, release)
//!
//! ```text
//!   protocol      OTs       us/OT   bytes/OT   rounds
//!       base      128      126.35       64.2        2
//!  extension      128      127.99       48.2        3
//!       base     1024      123.74       64.0        2
//!  extension     1024       18.27       20.0        3
//!       base     8192      123.92       64.0        2
//!  extension     8192        4.67       16.5        3
//! ```
//!
//! Base OT is flat, as it should be — every transfer is its own pair of scalar
//! multiplications, so batching buys nothing beyond reusing `a`. The extension
//! starts *level with it* at 128 OTs, because at that size the batch is exactly
//! the `κ = 128` base OTs it has to pay for first, and then falls away fast:
//! **27× cheaper per OT at 8192**, and 4× smaller.
//!
//! Bytes per OT converge to `κ/8 = 16` from above (48 → 20 → 16.5), which is the
//! protocol's own online cost of one column bit per output OT, with the fixed
//! base phase amortizing on top. That the measured curve lands on the analytic
//! floor rather than near it is the useful part: it says the byte accounting is
//! the protocol's and not an artifact.
//!
//! The per-OT *time* does not fall to zero and should not: the extension does
//! `O(κ)` bit operations per output OT to read a row out of the column-major
//! matrix. This implementation does that with bit lookups rather than a blocked
//! transpose (see `crate::ot::BitColumns`), so the constant here is worse than an
//! optimized library's — the shape of the curve is the claim, not the absolute
//! throughput.

use std::time::Instant;

use rand::{Rng, RngCore};
use serde::Serialize;

use crate::ot::{base, iknp, KAPPA};

#[derive(Debug, Clone, Serialize)]
pub struct OtPoint {
    pub protocol: &'static str,
    pub count: usize,
    /// All transfers agreed with the choice bits.
    pub correct: bool,
    pub total_bytes: usize,
    pub bytes_per_ot: f64,
    pub rounds: usize,
    pub total_ms: f64,
    pub us_per_ot: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct OtReport {
    pub kappa: usize,
    pub points: Vec<OtPoint>,
    /// Batch size at which the extension overtakes base OT on wall clock.
    pub time_crossover: Option<usize>,
}

fn random_bits(n: usize) -> Vec<bool> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<bool>()).collect()
}

fn bench_base(count: usize) -> Result<OtPoint, String> {
    let choices = random_bits(count);
    let mut rng = rand::thread_rng();
    let msgs: Vec<_> = (0..count)
        .map(|_| {
            let mut a = [0u8; 16];
            let mut b = [0u8; 16];
            rng.fill_bytes(&mut a);
            rng.fill_bytes(&mut b);
            (a, b)
        })
        .collect();

    let t0 = Instant::now();
    let (got, traffic) = base::simulate(&choices, &msgs)?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let correct = choices
        .iter()
        .zip(&msgs)
        .zip(&got)
        .all(|((&c, m), g)| *g == if c { m.1 } else { m.0 });
    Ok(OtPoint {
        protocol: "base",
        count,
        correct,
        total_bytes: traffic.total(),
        bytes_per_ot: traffic.total() as f64 / count as f64,
        rounds: traffic.rounds,
        total_ms,
        us_per_ot: total_ms * 1000.0 / count as f64,
    })
}

fn bench_extension(count: usize) -> Result<OtPoint, String> {
    let choices = random_bits(count);
    let t0 = Instant::now();
    let ext = iknp::simulate(&choices)?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let correct = choices.iter().enumerate().all(|(i, &c)| {
        let (y0, y1) = ext.sender[i];
        ext.receiver[i] == if c { y1 } else { y0 }
    });
    Ok(OtPoint {
        protocol: "extension",
        count,
        correct,
        total_bytes: ext.traffic.total(),
        bytes_per_ot: ext.traffic.total() as f64 / count as f64,
        rounds: ext.traffic.rounds,
        total_ms,
        us_per_ot: total_ms * 1000.0 / count as f64,
    })
}

pub fn run(sizes: &[usize]) -> Result<OtReport, String> {
    let mut points = Vec::new();
    for &n in sizes {
        points.push(bench_base(n)?);
        points.push(bench_extension(n)?);
    }
    // Smallest measured batch where the extension is faster per OT.
    let mut time_crossover = None;
    for &n in sizes {
        let b = points
            .iter()
            .find(|p| p.protocol == "base" && p.count == n)
            .map(|p| p.us_per_ot);
        let e = points
            .iter()
            .find(|p| p.protocol == "extension" && p.count == n)
            .map(|p| p.us_per_ot);
        if let (Some(b), Some(e)) = (b, e) {
            if e < b && time_crossover.is_none() {
                time_crossover = Some(n);
            }
        }
    }
    Ok(OtReport {
        kappa: KAPPA,
        points,
        time_crossover,
    })
}

const USAGE: &str = "usage: baru bench-ot [options]

Measured oblivious transfer: Chou-Orlandi base OT against the IKNP extension.
Both semi-honest.  The bytes-per-OT and us-per-OT columns are the inputs the
MPC cost model needs to price Semi2k / MASCOT preprocessing from measurement
instead of from the literature.

options:
  --sizes N,N,...  batch sizes to sweep   (default 128,1024,8192)
  --json           machine-readable output
  -h, --help       this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut sizes = vec![128usize, 1024, 8192];
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
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }
    if sizes.is_empty() {
        return Err("no batch sizes given".to_string());
    }

    let r = run(&sizes)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Oblivious transfer — Chou-Orlandi base OT vs IKNP extension (semi-honest)");
    println!("kappa = {}\n", r.kappa);
    println!(
        "{:>10} {:>8} {:>8} {:>11} {:>10} {:>8} {:>9}",
        "protocol", "OTs", "correct", "us/OT", "bytes/OT", "rounds", "total ms"
    );
    for p in &r.points {
        println!(
            "{:>10} {:>8} {:>8} {:>11.2} {:>10.1} {:>8} {:>9.1}",
            p.protocol,
            p.count,
            if p.correct { "yes" } else { "NO" },
            p.us_per_ot,
            p.bytes_per_ot,
            p.rounds,
            p.total_ms
        );
    }
    println!();
    match r.time_crossover {
        Some(n) => {
            println!("the extension is already cheaper per OT at the smallest batch measured ({n})",)
        }
        None => println!("base OT was cheaper per OT at every batch size measured"),
    }
    println!(
        "extension bytes/OT converge to kappa/8 = {} B — one column bit per output OT",
        r.kappa / 8
    );
    println!("both are SEMI-HONEST: IKNP here has no KOS-style consistency check, so a");
    println!("malicious receiver can send inconsistent columns and extract the sender's s.");

    if r.points.iter().any(|p| !p.correct) {
        return Err("an OT returned the wrong message".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correctness of both layers, and the property that motivates the
    /// extension: its per-OT cost must fall below base OT's.
    #[test]
    fn the_extension_beats_base_ot_per_transfer() {
        let r = run(&[256]).unwrap();
        assert!(r.points.iter().all(|p| p.correct), "{:?}", r.points);
        let base = r.points.iter().find(|p| p.protocol == "base").unwrap();
        let ext = r.points.iter().find(|p| p.protocol == "extension").unwrap();
        assert!(
            ext.us_per_ot < base.us_per_ot,
            "extension {:.2} us/OT vs base {:.2} us/OT",
            ext.us_per_ot,
            base.us_per_ot
        );
        assert!(
            ext.bytes_per_ot < base.bytes_per_ot,
            "extension should also be smaller: {:.1} vs {:.1} B/OT",
            ext.bytes_per_ot,
            base.bytes_per_ot
        );
    }

    /// The online cost of the extension is `κ` bits per OT, so bytes/OT must
    /// approach 16 from above as the base phase amortizes.
    #[test]
    fn extension_bytes_per_ot_approach_the_kappa_bit_floor() {
        let r = run(&[8192]).unwrap();
        let ext = r.points.iter().find(|p| p.protocol == "extension").unwrap();
        let floor = KAPPA as f64 / 8.0;
        assert!(
            ext.bytes_per_ot >= floor && ext.bytes_per_ot < floor * 1.2,
            "{:.2} B/OT is not just above the {floor} B floor",
            ext.bytes_per_ot
        );
    }
}
