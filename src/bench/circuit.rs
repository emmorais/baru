//! `baru bench-circuit` — how good are the circuits Baru generates?
//!
//! [`crate::bench::abstraction`] asks what the *eager* APIs cost (gadgets and
//! the interpreter, which issue operations as they are written). This asks the
//! complementary question about the *compiled* path — `Expr` →
//! [`CircuitCompiler`] → `FlatOp` → [`crate::mpcc`] — where Baru is a circuit
//! language and can rewrite the program before anyone runs it.
//!
//! That is where Baru can beat hand-written code rather than merely match it:
//! a human writing batched MPC calls gets batching only where they remembered
//! to ask for it, whereas an ASAP schedule over the dependence DAG batches
//! *everything* that is independent, for free and without a vectorized IR.
//!
//! # What each column means
//!
//! | column | meaning |
//! |---|---|
//! | `ops` | `FlatOp` count the compiler emits |
//! | `ops/no-cse` | the same with CSE disabled — what the pass is worth |
//! | `interactive` | ops that cost a communication round |
//! | `rounds` | ASAP schedule depth = the online round count |
//!
//! # The two regimes, and why they differ
//!
//! **Independent work batches automatically.** A ReLU layer or a dot product
//! is a set of mutually independent operations, so ASAP puts them all at one
//! level and the round count is *constant in the layer width* — the compiler
//! reproduces, with no vectorization and no user effort, exactly what
//! `bench-abstraction`'s hand-batched route achieves by explicit
//! `lt_batch`/`mul_batch` calls.
//!
//! **Dependent reductions do not.** A fold (`acc = f(acc, xᵢ)`) is a chain,
//! and ASAP is optimal only for a *fixed* lowering — it cannot reassociate.
//! A max-fold therefore costs `O(n)` rounds where a tournament costs
//! `O(log n)`. This is the one place the hand-written gadget library is still
//! ahead of the compiler, and closing it is a rewrite rule, not a scheduling
//! change.
//!
//! # Results (2026-07-28)
//!
//! ```text
//!   program          n      ops ops/no-cse   interactive   rounds
//!   relu-layer       1        3          3             3        3
//!   relu-layer      64      192   255 (1x)           129        3
//!   dot             64      127        127            65        2
//!   max-fold         4       12    52 (4x)             7        7
//!   max-fold         8       28 4372 (156x)            15       15
//! ```
//!
//! Two things to read off it. Rounds are **flat in width** for the layer
//! workloads — 3 at n=1 and 3 at n=64, while the interactive op count grows
//! 3 → 129. And the max-fold is where CSE earns its place in the pipeline:
//! written the obvious way the accumulator subtree is textually repeated at
//! each step, so the flattened circuit grows exponentially — **4372 ops at
//! n=8, collapsing to 28 (156x)**. Before CSE moved into [`CircuitCompiler`],
//! every circuit handed to the FHE and ZKP backends carried that blow-up; it
//! was removed only if the user happened to run `baru mpcc optimize`.
//!
//! On a real workload the same pass takes the MNIST demo circuit from
//! **2365 to 1764 gates (-25%)** with bit-identical scores — and since TFHE
//! evaluates at ~0.2-0.5 s/gate, those 601 gates are minutes per run.

use serde::Serialize;

use crate::interpreter::parse;
use crate::mpcc::ir::Program;
use crate::mpcc::lower::{lower, LowerConfig};
use crate::mpcc::schedule::schedule;
use crate::zkp::circuit_compiler::CircuitCompiler;

#[derive(Debug, Clone, Serialize)]
pub struct CircuitPoint {
    pub program: String,
    pub n: usize,
    pub ops: usize,
    pub ops_no_cse: usize,
    pub interactive: usize,
    pub rounds: usize,
}

/// `ReLU` over `n` independent inputs, summed (the sum is local, so it does
/// not add rounds and does not couple the elements).
fn relu_layer_src(n: usize) -> String {
    let term = |i: usize| format!("(* x{i} (> x{i} 0))");
    let mut e = term(0);
    for i in 1..n {
        e = format!("(+ {e} {})", term(i));
    }
    e
}

/// One dense row with secret weights: `Σ wᵢ·xᵢ`.
fn dot_src(n: usize) -> String {
    let mut e = "(* w0 x0)".to_string();
    for i in 1..n {
        e = format!("(+ {e} (* w{i} x{i}))");
    }
    e
}

/// Maximum by left fold, mux-style: `acc + (xᵢ>acc)·(xᵢ−acc)`. Written the
/// obvious way, so `acc` is textually repeated — which is exactly what makes
/// it the interesting case for CSE.
fn max_fold_src(n: usize) -> String {
    let mut e = "x0".to_string();
    for i in 1..n {
        e = format!("(+ {e} (* (> x{i} {e}) (- x{i} {e})))");
    }
    e
}

fn compile_len(src: &str, cse: bool) -> Result<(usize, Program), String> {
    let expr = parse(src).map_err(|e| format!("parse error: {e}"))?;
    let mut cc = if cse { CircuitCompiler::new() } else { CircuitCompiler::new().without_cse() };
    let circuit = cc.compile_fhe(&expr).map_err(|e| format!("compile error: {e}"))?;
    let program = lower(&circuit.flat_ops, &circuit.output_wire, &LowerConfig::default())?;
    Ok((circuit.flat_ops.len(), program))
}

pub fn measure(name: &str, src: &str, n: usize) -> Result<CircuitPoint, String> {
    let (ops, program) = compile_len(src, true)?;
    let (ops_no_cse, _) = compile_len(src, false)?;
    let sched = schedule(&program);
    let interactive = program.code.iter().filter(|i| program.is_interactive(i)).count();
    Ok(CircuitPoint {
        program: name.to_string(),
        n,
        ops,
        ops_no_cse,
        interactive,
        rounds: sched.round_count(),
    })
}

const USAGE: &str = "usage: baru bench-circuit [options]

How good are the circuits Baru generates? Compiles AI-shaped Baru programs
through CircuitCompiler -> FlatOp -> mpcc and reports circuit size and the
ASAP round count, with and without CSE.

Rounds should be FLAT in width for layer workloads (independent operations
batch automatically under ASAP scheduling) and LINEAR for a fold (a dependent
chain that scheduling alone cannot reassociate).

options:
  --widths a,b,c   layer widths to sweep (default 1,8,64)
  --fold-widths    widths for the dependent max-fold (default 2,4,8; it is
                   exponential without CSE, so keep these small)
  --json           machine-readable output
  -h, --help       this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut widths: Vec<usize> = vec![1, 8, 64];
    let mut fold_widths: Vec<usize> = vec![2, 4, 8];
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--widths" => widths = parse_list(it.next().ok_or("--widths needs a value")?)?,
            "--fold-widths" => {
                fold_widths = parse_list(it.next().ok_or("--fold-widths needs a value")?)?
            }
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let mut pts = Vec::new();
    for &n in &widths {
        pts.push(measure("relu-layer", &relu_layer_src(n), n)?);
        pts.push(measure("dot", &dot_src(n), n)?);
    }
    for &n in &fold_widths {
        pts.push(measure("max-fold", &max_fold_src(n), n)?);
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&pts).map_err(|e| e.to_string())?);
        return Ok(());
    }

    println!("Circuit generation quality — Expr -> CircuitCompiler -> FlatOp -> mpcc\n");
    println!(
        "  {:<12} {:>5} {:>8} {:>10} {:>13} {:>8}",
        "program", "n", "ops", "ops/no-cse", "interactive", "rounds"
    );
    for p in &pts {
        let cse_note = if p.ops_no_cse > p.ops {
            format!("{} ({:.0}x)", p.ops_no_cse, p.ops_no_cse as f64 / p.ops.max(1) as f64)
        } else {
            p.ops_no_cse.to_string()
        };
        println!(
            "  {:<12} {:>5} {:>8} {:>10} {:>13} {:>8}",
            p.program, p.n, p.ops, cse_note, p.interactive, p.rounds
        );
    }

    // The headline property, stated as a check rather than a hope: scheduling
    // must make layer rounds independent of width.
    let layer_rounds: Vec<(usize, usize)> = pts
        .iter()
        .filter(|p| p.program == "relu-layer")
        .map(|p| (p.n, p.rounds))
        .collect();
    if let (Some(first), Some(last)) = (layer_rounds.first(), layer_rounds.last()) {
        if first.1 != last.1 {
            return Err(format!(
                "ReLU layer rounds are not flat in width: {} at n={} but {} at n={} — \
                 the ASAP schedule is no longer batching independent operations",
                first.1, first.0, last.1, last.0
            ));
        }
        println!(
            "\n  ReLU layer: {} rounds at n={} and n={} — round count is FLAT in width.\n  \
             The compiler batches every independent operation with no vectorized IR and\n  \
             no user effort, which is what the hand-written route needs explicit\n  \
             lt_batch/mul_batch calls to achieve.",
            first.1, first.0, last.0
        );
    }
    let fold = pts.iter().filter(|p| p.program == "max-fold").collect::<Vec<_>>();
    if let (Some(a), Some(b)) = (fold.first(), fold.last()) {
        println!(
            "  max-fold:   {} rounds at n={} vs {} at n={} — LINEAR, because a fold is a\n  \
             dependent chain. ASAP is optimal for a fixed lowering but cannot\n  \
             reassociate; a tournament would be ceil(log2 n). This is the gap where the\n  \
             hand-written gadget is still ahead of the compiler.",
            a.rounds, a.n, b.rounds, b.n
        );
    }
    Ok(())
}

fn parse_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|t| t.trim().parse::<usize>().map_err(|_| format!("bad width list `{s}`")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole compiled path rests on: independent operations
    /// cost a constant number of rounds no matter how many there are. If this
    /// breaks, every layer in every network silently serializes.
    #[test]
    fn layer_round_count_is_flat_in_width() {
        let one = measure("relu-layer", &relu_layer_src(1), 1).unwrap();
        let wide = measure("relu-layer", &relu_layer_src(32), 32).unwrap();
        assert_eq!(
            one.rounds, wide.rounds,
            "ReLU layer went from {} rounds at n=1 to {} at n=32",
            one.rounds, wide.rounds
        );
        assert!(wide.interactive > one.interactive, "the wide layer must do more work");
    }

    /// CSE must be part of compilation, not an opt-in pass. A fold written the
    /// obvious way repeats its accumulator subtree and flattens exponentially;
    /// before CSE moved into `CircuitCompiler` this reached the FHE and ZKP
    /// backends intact.
    #[test]
    fn cse_collapses_the_exponential_fold_and_is_on_by_default() {
        let p = measure("max-fold", &max_fold_src(8), 8).unwrap();
        assert!(
            p.ops_no_cse > 20 * p.ops,
            "expected an exponential blow-up without CSE, got {} vs {}",
            p.ops_no_cse,
            p.ops
        );
        assert!(p.ops < 100, "compiled fold should be small, got {} ops", p.ops);
    }
}
