//! `baru bench-backend-aware` — the same AI program becomes a *different*
//! circuit for each backend, and using the wrong one costs.
//!
//! This is the claim nobody else in the space makes. MP-SPDZ optimizes rounds
//! and is blind to FHE; FHE compilers optimize multiplicative depth and are
//! blind to MPC; HyCC → Silph → SING assign *protocols* to parts of a program
//! but do not rewrite the circuit differently per target. Baru prices every
//! rewrite against the backend it is headed for, so the circuits genuinely
//! diverge.
//!
//! Asserting divergence is cheap and unconvincing on its own — two circuits
//! could differ in ways nobody pays for. The measurement that matters is the
//! **cross-application matrix**: optimize the program once per backend, then
//! price *every* resulting circuit under *every* backend.
//!
//! ```text
//!                   evaluated under
//!                 shamir  semi-bin   bmr
//!   circuit for
//!   shamir          ✓min      …       …
//!   semi-bin         …       ✓min     …
//!   bmr              …        …      ✓min
//! ```
//!
//! If backend-aware optimization is real, **each row's own circuit wins its
//! own column** — the diagonal is the column minimum — and the off-diagonal
//! entries quantify what a one-size-fits-all compiler would lose. The command
//! fails if the diagonal is ever beaten, because that would mean the
//! per-backend search picked a circuit its own cost model does not prefer.
//!
//! # Why these workloads
//!
//! - `relu-layer` — independent work; the rewrites that matter are local.
//! - `max-fold` — a *dependent* reduction. Rebalancing it into a tournament
//!   trades depth for nothing else, so it is a large win under an interactive
//!   protocol (depth *is* the round count) and worth exactly nothing under a
//!   constant-round garbled protocol. Same rewrite, opposite verdicts.
//! - `dot-const` — a dot product against public constants. Strength reduction
//!   turns `x*c` into shifts and adds: a win for a binary/garbled backend that
//!   pays per AND gate, a loss for an arithmetic one where a multiply by a
//!   constant is already local.

use serde::Serialize;

use crate::interpreter::parse;
use crate::mpcc::cost::{native_domain, Deployment};
use crate::mpcc::lower::LowerConfig;
use crate::mpcc::opt::{objective, optimize};
use crate::mpcc::protocols::find;
use crate::zkp::circuit_compiler::{CircuitCompiler, FlatOp};

/// One backend's optimized circuit and what it cost every backend.
#[derive(Debug, Clone, Serialize)]
pub struct Row {
    /// The backend this circuit was optimized *for*.
    pub optimized_for: String,
    pub ops: usize,
    /// Cheap identity of the emitted circuit. Op *count* is a poor divergence
    /// metric — reassociation changes the shape while preserving it — so
    /// distinct circuits are counted by content.
    pub fingerprint: String,
    /// Rules the search kept for this backend, in order.
    pub rules: Vec<String>,
    /// Predicted seconds under each backend, in the same order as `columns`.
    pub secs: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkloadReport {
    pub workload: String,
    pub n: usize,
    pub columns: Vec<String>,
    pub baseline_secs: Vec<f64>,
    pub rows: Vec<Row>,
    /// Worst penalty, over all backends, for running a circuit that was
    /// optimized for a *different* backend.
    pub worst_wrong_circuit_penalty: f64,
}

fn relu_layer_src(n: usize) -> String {
    let term = |i: usize| format!("(* x{i} (> x{i} 0))");
    let mut e = term(0);
    for i in 1..n {
        e = format!("(+ {e} {})", term(i));
    }
    e
}

/// Maximum by left fold, mux style: `acc + (xᵢ>acc)·(xᵢ−acc)`.
fn max_fold_src(n: usize) -> String {
    let mut e = "x0".to_string();
    for i in 1..n {
        e = format!("(+ {e} (* (> x{i} {e}) (- x{i} {e})))");
    }
    e
}

/// A dot product against public constants with several set bits — the shape
/// strength reduction acts on.
fn dot_const_src(n: usize) -> String {
    let c = [3i64, 5, 9, 17, 100, 12, 33, 7];
    let mut e = format!("(* x0 {})", c[0]);
    for i in 1..n {
        e = format!("(+ {e} (* x{i} {}))", c[i % c.len()]);
    }
    e
}

/// A content hash of a circuit, so "are these the same circuit?" does not
/// reduce to "do they have the same number of operations?".
fn fingerprint(flat: &[FlatOp]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for op in flat {
        format!("{op:?}").hash(&mut h);
    }
    format!("{:x}", h.finish() & 0xffff_ffff)
}

fn compile(src: &str) -> Result<(Vec<FlatOp>, String), String> {
    let expr = parse(src).map_err(|e| format!("parse error: {e}"))?;
    let c = CircuitCompiler::new()
        .compile_fhe(&expr)
        .map_err(|e| format!("compile error: {e}"))?;
    Ok((c.flat_ops, c.output_wire))
}

fn dep_for(id: &str, parties: usize) -> Deployment {
    // Yao is inherently 2-party; everything else gets the requested count.
    Deployment::wan(if id == "yao" { 2 } else { parties })
}

fn cfg_for(id: &str, parties: usize) -> LowerConfig {
    let spec = find(id).expect("known protocol");
    LowerConfig {
        domain: native_domain(spec, 64),
        parties: if id == "yao" { 2 } else { parties },
        bit_width: 32,
        public_vars: Vec::new(),
    }
}

pub fn measure(
    workload: &str,
    src: &str,
    n: usize,
    backends: &[String],
    parties: usize,
) -> Result<WorkloadReport, String> {
    let (flat, out) = compile(src)?;

    // Baseline: the unoptimized circuit priced under each backend.
    let mut baseline_secs = Vec::new();
    for id in backends {
        let spec = find(id).ok_or_else(|| format!("unknown protocol `{id}`"))?;
        let o = objective(
            &flat,
            &out,
            spec,
            &dep_for(id, parties),
            &cfg_for(id, parties),
        )?;
        baseline_secs.push(o.secs);
    }

    // One optimized circuit per backend.
    let mut rows = Vec::new();
    for id in backends {
        let spec = find(id).ok_or_else(|| format!("unknown protocol `{id}`"))?;
        let opt = optimize(
            &flat,
            &out,
            spec,
            &dep_for(id, parties),
            &cfg_for(id, parties),
            true, // differentially verify every circuit we then price
        )?;
        // Price this circuit under every backend, including the others'.
        let mut secs = Vec::new();
        for other in backends {
            let ospec = find(other).expect("known protocol");
            let o = objective(
                &opt.flat,
                &out,
                ospec,
                &dep_for(other, parties),
                &cfg_for(other, parties),
            )?;
            secs.push(o.secs);
        }
        rows.push(Row {
            optimized_for: id.clone(),
            ops: opt.flat.len(),
            fingerprint: fingerprint(&opt.flat),
            rules: opt.applied.iter().map(|a| a.rule.to_string()).collect(),
            secs,
        });
    }

    // The diagonal must win its own column, and the worst off-diagonal
    // penalty is what a backend-blind compiler would pay.
    let mut worst = 1.0f64;
    for (j, _) in backends.iter().enumerate() {
        let own = rows[j].secs[j];
        for (i, row) in rows.iter().enumerate() {
            if i == j {
                continue;
            }
            if row.secs[j] < own * 0.999 {
                return Err(format!(
                    "{workload}: the circuit optimized for `{}` beats `{}`'s own circuit \
                     under `{}` ({:.6}s vs {:.6}s) — the per-backend search chose a circuit \
                     its own cost model does not prefer",
                    row.optimized_for, backends[j], backends[j], row.secs[j], own
                ));
            }
            worst = worst.max(row.secs[j] / own.max(1e-12));
        }
    }

    Ok(WorkloadReport {
        workload: workload.to_string(),
        n,
        columns: backends.to_vec(),
        baseline_secs,
        rows,
        worst_wrong_circuit_penalty: worst,
    })
}

const USAGE: &str = "usage: baru bench-backend-aware [options]

The same program, optimized once per backend, then every resulting circuit
priced under every backend. The diagonal must win its own column; the
off-diagonal entries are what a backend-blind compiler would lose.

options:
  --n N            workload width (default 8)
  --backends a,b   protocols to compare (default shamir,semi-bin,bmr)
  --parties P      computing parties (default 3)
  --json           machine-readable output
  -h, --help       this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut n = 8usize;
    let mut backends: Vec<String> = ["shamir", "semi-bin", "bmr"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut parties = 3usize;
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" => {
                n = it
                    .next()
                    .ok_or("--n needs a value")?
                    .parse()
                    .map_err(|_| "bad --n")?
            }
            "--parties" => {
                parties = it
                    .next()
                    .ok_or("--parties needs a value")?
                    .parse()
                    .map_err(|_| "bad --parties")?
            }
            "--backends" => {
                backends = it
                    .next()
                    .ok_or("--backends needs a value")?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect()
            }
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let reports = vec![
        measure("relu-layer", &relu_layer_src(n), n, &backends, parties)?,
        measure("max-fold", &max_fold_src(n), n, &backends, parties)?,
        measure("dot-const", &dot_const_src(n), n, &backends, parties)?,
    ];

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("The same program becomes a different circuit for each backend");
    println!("(WAN deployment; every optimized circuit differentially verified)\n");
    for r in &reports {
        println!("{} (n={})", r.workload, r.n);
        print!("  {:<22}{:>6}", "circuit optimized for", "ops");
        for c in &r.columns {
            print!("{:>12}", format!("as {c}"));
        }
        println!("     rules kept");
        for (i, row) in r.rows.iter().enumerate() {
            print!("  {:<22}{:>6}", row.optimized_for, row.ops);
            for (j, s) in row.secs.iter().enumerate() {
                let mark = if i == j { "*" } else { " " };
                print!("{:>12}", format!("{:.3}ms{}", s * 1000.0, mark));
            }
            let rules = if row.rules.is_empty() {
                "-".to_string()
            } else {
                let mut r: Vec<String> = row.rules.clone();
                r.dedup();
                r.join(",")
            };
            println!("     {rules}");
        }
        let distinct: std::collections::HashSet<&str> =
            r.rows.iter().map(|x| x.fingerprint.as_str()).collect();
        println!(
            "  → {} distinct circuit(s) from one program; worst penalty for using another \
             backend's circuit: {:.2}x",
            distinct.len(),
            r.worst_wrong_circuit_penalty
        );
        println!();
    }
    println!(
        "* marks each circuit priced under the backend it was optimized for. The diagonal\n\
         wins every column — that is backend-aware optimization doing something a single\n\
         circuit cannot. Where a row keeps no rules, that backend genuinely wanted the\n\
         original circuit, which is a result too: the optimizer declines rather than\n\
         rewriting for its own sake."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backends() -> Vec<String> {
        ["shamir", "semi-bin", "bmr"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// The central claim: optimizing for a backend produces a circuit that
    /// backend prefers to the ones its neighbours chose. `measure` returns an
    /// error if any diagonal entry is beaten, so this also pins that.
    #[test]
    fn each_backend_prefers_its_own_circuit() {
        for (name, src) in [
            ("max-fold", max_fold_src(8)),
            ("dot-const", dot_const_src(8)),
        ] {
            let r = measure(name, &src, 8, &backends(), 3).expect("diagonal must win");
            assert_eq!(r.rows.len(), 3);
        }
    }

    /// Divergence must be real, not nominal: at least one workload has to
    /// yield genuinely different circuits, or "backend-aware" is a label
    /// rather than a behaviour.
    #[test]
    fn some_workload_produces_genuinely_different_circuits() {
        let mut diverged = false;
        for (name, src) in [
            ("relu-layer", relu_layer_src(8)),
            ("max-fold", max_fold_src(8)),
            ("dot-const", dot_const_src(8)),
        ] {
            let r = measure(name, &src, 8, &backends(), 3).unwrap();
            let sizes: std::collections::HashSet<String> =
                r.rows.iter().map(|x| x.fingerprint.clone()).collect();
            let rules: std::collections::HashSet<Vec<String>> =
                r.rows.iter().map(|x| x.rules.clone()).collect();
            if sizes.len() > 1 || rules.len() > 1 {
                diverged = true;
            }
        }
        assert!(
            diverged,
            "no workload produced backend-specific circuits — the per-backend search is \
             not actually discriminating"
        );
    }
}
