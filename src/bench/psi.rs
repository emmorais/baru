//! `baru bench-psi` — what private set intersection costs, and which
//! construction stage 3 should actually use.
//!
//! The research agenda wrote stage 3 as "**PSI** (from OT)". That is an
//! assumption, and this repo's whole thesis is that the cost model should decide
//! rather than the intuition. So this benchmark measures the construction that
//! needs *no* OT ([`crate::psi::dh`]) and prices the OT-based route from the
//! measured OT cost in [`crate::mpcc::ot_cost`] — turning "which PSI?" into a
//! comparison before the expensive one gets built.
//!
//! Three rows, and the insecure one is the point of comparison:
//!
//! | | hides non-shared items? | per item |
//! |---|---|---|
//! | naive hash | **no** — offline dictionary attack recovers the whole set | 32 B, one hash |
//! | DH-PSI | yes (semi-honest, DDH) | 96 B, 2 scalar mults |
//! | OT-based (projected) | yes (semi-honest) | ~`w/κ` × 16 B, symmetric only |
//!
//! # The result (2026-07-28, M-series, release)
//!
//! ```text
//!    items  shared  correct   bytes/item    us/item   naive us/item
//!      256     128      yes         96.0     165.63            0.62
//!     1024     512      yes         96.0     164.57            0.43
//!     4096    2048      yes         96.0     165.36            0.43
//! ```
//!
//! DH-PSI is flat in both, as it must be: three flights of compressed points and
//! a fixed two scalar multiplications per item. **96 B/item and ~165 µs/item.**
//!
//! The insecure baseline is **~380× faster** (0.43 µs/item), which is worth
//! stating plainly because it is why hash-exchange keeps getting shipped. The
//! 380× buys an offline dictionary attack that recovers the sender's entire set.
//!
//! Set that against the projection. KKRT-style OT-based PSI needs a batched OPRF
//! whose code width is `w ≈ 448` bits, i.e. `w/κ = 3.5` OT-extension columns per
//! item, at the measured 16.5 B per column-OT: **≈ 58 B/item**, with no
//! public-key operation at all — so on the order of a microsecond per item rather
//! than 165.
//!
//! So the agenda's assumption is right, and now for a stated reason: OT-based PSI
//! should win on *both* axes, and the decisive one is **compute, not bandwidth**
//! (~1.7× on bytes, but two orders of magnitude on time). DH-PSI's cost is two
//! scalar multiplications per item and no amount of engineering removes them.
//!
//! Two honest qualifications. The 58 B/item is a projection from a measured
//! per-OT cost and a literature code width, not a measurement of a PSI — it
//! belongs in a "what to build next" argument, not in a results table. And
//! DH-PSI stays the right choice in one regime the projection does not overturn:
//! small sets, where 165 µs × n is irrelevant and 100 lines of code that needs no
//! OT infrastructure is worth more than a constant factor.

use std::time::Instant;

use serde::Serialize;

use crate::mpcc::ot_cost;
use crate::psi::plain_intersect;

/// OPRF code width in bits for a KKRT-style batched OPRF. A literature figure,
/// not a measurement — the projection below is only as good as this.
pub const OPRF_CODE_BITS: f64 = 448.0;

#[derive(Debug, Clone, Serialize)]
pub struct PsiPoint {
    /// Which construction — from `PsiProtocol::name`, so a new implementation
    /// appears here without this file changing.
    pub protocol: &'static str,
    /// What it hides, carried alongside the numbers so the table cannot put a
    /// broken protocol in the same column as a real one.
    pub hiding: crate::psi::Hiding,
    pub items: usize,
    pub intersection: usize,
    /// The protocol's output matched the plaintext intersection.
    pub correct: bool,
    pub total_bytes: usize,
    pub bytes_per_item: f64,
    pub rounds: usize,
    pub total_ms: f64,
    pub us_per_item: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PsiReport {
    pub points: Vec<PsiPoint>,
    /// Projected bytes/item for an OT-based PSI, from measured OT cost.
    pub projected_ot_bytes_per_item: f64,
    pub measured_bytes_per_ot: f64,
    pub oprf_code_bits: f64,
}

fn synthetic(n: usize, overlap: usize, tag: &str) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            if i < overlap {
                format!("shared-patient-{i}").into_bytes()
            } else {
                format!("{tag}-patient-{i}").into_bytes()
            }
        })
        .collect()
}

/// Projected per-item communication for a KKRT-style OT-based PSI.
///
/// One item needs `⌈w/κ⌉` OT-extension columns' worth of communication, each
/// costing the measured per-OT bytes. This deliberately ignores cuckoo-hashing
/// expansion (~1.2×) and the OPRF output flight, so it is a *lower* bound on the
/// OT route — which is the conservative direction for a claim that the OT route
/// wins.
pub fn projected_ot_bytes_per_item() -> f64 {
    (OPRF_CODE_BITS / crate::ot::KAPPA as f64) * ot_cost::BYTES_PER_OT
}

pub fn run(sizes: &[usize]) -> Result<PsiReport, String> {
    let mut points = Vec::new();
    for &n in sizes {
        let overlap = n / 2;
        let a = synthetic(n, overlap, "alice");
        let b = synthetic(n, overlap, "bob");
        let want = plain_intersect(&a, &b);

        // Every construction the crate implements, measured identically. Adding
        // one to `psi::all()` adds a row here.
        for proto in crate::psi::all() {
            let t0 = Instant::now();
            let out = proto.intersect(&a, &b)?;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            points.push(PsiPoint {
                protocol: proto.name(),
                hiding: proto.hiding(),
                items: n,
                intersection: out.intersection.len(),
                correct: out.intersection == want,
                total_bytes: out.cost.bytes,
                bytes_per_item: out.cost.bytes as f64 / n as f64,
                rounds: out.cost.rounds,
                total_ms,
                us_per_item: total_ms * 1000.0 / n as f64,
            });
        }
    }
    Ok(PsiReport {
        points,
        projected_ot_bytes_per_item: projected_ot_bytes_per_item(),
        measured_bytes_per_ot: ot_cost::BYTES_PER_OT,
        oprf_code_bits: OPRF_CODE_BITS,
    })
}

const USAGE: &str = "usage: baru bench-psi [options]

Private set intersection: DH-based PSI (measured) against the insecure
hash-exchange baseline, plus a projection for the OT-based construction from
the measured OT cost.  Sets are half-overlapping.

options:
  --sizes N,N,...  set sizes to sweep   (default 256,1024,4096)
  --json           machine-readable output
  -h, --help       this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut sizes = vec![256usize, 1024, 4096];
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
        return Err("no set sizes given".to_string());
    }

    let r = run(&sizes)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Private set intersection — DH-based (Ristretto255), semi-honest, one-sided\n");
    println!(
        "{:>11} {:>8} {:>7} {:>8} {:>12} {:>10} {:>7}  hides",
        "protocol", "items", "shared", "correct", "bytes/item", "us/item", "rounds"
    );
    for p in &r.points {
        println!(
            "{:>11} {:>8} {:>7} {:>8} {:>12.1} {:>10.2} {:>7}  {}",
            p.protocol,
            p.items,
            p.intersection,
            if p.correct { "yes" } else { "NO" },
            p.bytes_per_item,
            p.us_per_item,
            p.rounds,
            match p.hiding {
                crate::psi::Hiding::NonSharedItems => "non-shared items",
                crate::psi::Hiding::Nothing => "NOTHING (insecure)",
            }
        );
    }
    println!();
    println!("Both are flat per item, as they must be: three flights of compressed points");
    println!("and a fixed two scalar multiplications each.\n");
    println!(
        "OT-based PSI, projected from MEASURED OT cost ({:.1} B/OT) and a {:.0}-bit OPRF",
        r.measured_bytes_per_ot, r.oprf_code_bits
    );
    println!(
        "code: ~{:.0} B/item, with no public-key operation at all.",
        r.projected_ot_bytes_per_item
    );
    let pick = |name: &str, f: fn(&PsiPoint) -> f64| -> f64 {
        r.points
            .iter()
            .rfind(|p| p.protocol == name)
            .map(f)
            .unwrap_or(0.0)
    };
    let dh_bytes = pick("dh-psi", |p| p.bytes_per_item);
    let dh_us = pick("dh-psi", |p| p.us_per_item);
    println!(
        "So the OT route should win ~{:.1}x on bytes — and far more on time, since {:.0} us/item",
        dh_bytes / r.projected_ot_bytes_per_item.max(1e-9),
        dh_us
    );
    println!("of scalar multiplication is what DH-PSI cannot engineer away.  The agenda's");
    println!("\"PSI (from OT)\" is therefore right, and the deciding axis is COMPUTE.");
    println!();
    println!(
        "Caveats: the {:.0} B/item is a projection (measured per-OT cost x a literature",
        r.projected_ot_bytes_per_item
    );
    println!("code width), not a measured PSI.  And DH-PSI stays preferable for small sets,");
    println!("where 100 lines needing no OT infrastructure beats a constant factor.");
    let naive_us = pick("naive-hash", |p| p.us_per_item);
    println!(
        "The naive column is the INSECURE baseline, and it is ~{:.0}x faster — which is why",
        dh_us / naive_us.max(1e-9)
    );
    println!("hash-exchange keeps getting shipped.  What that speed buys is an offline");
    println!("dictionary attack recovering the sender's ENTIRE set, not just the overlap.");

    if r.points.iter().any(|p| !p.correct) {
        return Err("a PSI run disagreed with the plaintext intersection".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(r: &PsiReport, name: &str, items: usize) -> PsiPoint {
        r.points
            .iter()
            .find(|p| p.protocol == name && p.items == items)
            .cloned()
            .unwrap_or_else(|| panic!("no {name} row at {items} items"))
    }

    #[test]
    fn every_construction_is_correct_and_flat_per_item() {
        let r = run(&[128, 512]).unwrap();
        assert!(r.points.iter().all(|p| p.correct), "{:?}", r.points);
        // Half-overlapping sets, for every protocol.
        assert!(r.points.iter().all(|p| p.intersection == p.items / 2));

        // Per-item bytes must not depend on set size, for any construction.
        for name in ["dh-psi", "naive-hash"] {
            let (a, b) = (
                row(&r, name, 128).bytes_per_item,
                row(&r, name, 512).bytes_per_item,
            );
            assert!((a - b).abs() < 1e-9, "{name}: {a} vs {b} bytes/item");
        }
        let dh = row(&r, "dh-psi", 128).bytes_per_item;
        assert!(
            (dh - 96.0).abs() < 1e-9,
            "three 32-byte flights per item, got {dh}"
        );
        // Both directions of the digest exchange, so 2 x 32 B.
        let naive = row(&r, "naive-hash", 128).bytes_per_item;
        assert!((naive - 64.0).abs() < 1e-9, "got {naive}");
    }

    /// The benchmark must be driven by the trait, not by a hard-coded list — so
    /// every construction the crate implements has to show up.
    #[test]
    fn the_table_covers_every_implemented_construction() {
        let r = run(&[64]).unwrap();
        for proto in crate::psi::all() {
            assert!(
                r.points.iter().any(|p| p.protocol == proto.name()),
                "{} is implemented but missing from the comparison",
                proto.name()
            );
        }
    }

    /// The projection is the argument for building the OT route, so pin it: it
    /// must come out below DH-PSI's measured bytes, or the recommendation
    /// changes and the docs need rewriting.
    #[test]
    fn the_ot_projection_undercuts_dh_psi() {
        let r = run(&[256]).unwrap();
        let dh = row(&r, "dh-psi", 256).bytes_per_item;
        assert!(
            r.projected_ot_bytes_per_item < dh,
            "projected OT PSI {:.1} B/item should undercut DH-PSI's {dh:.1}",
            r.projected_ot_bytes_per_item
        );
        // And it is derived from the measured OT figure, not a fresh constant.
        assert_eq!(r.measured_bytes_per_ot, ot_cost::BYTES_PER_OT);
    }
}
