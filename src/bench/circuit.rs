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
//! **Dependent reductions need a rewrite, not a schedule.** A fold
//! (`acc = f(acc, xᵢ)`) is a chain, and ASAP is optimal only for a *fixed*
//! lowering — it cannot reassociate. As written, a max-fold costs `2n−1`
//! rounds. [`crate::mpcc::opt::Rule::ReassociateReduction`] rebalances it into
//! a tournament, giving `2⌈log₂ n⌉+1`: **15 → 7 rounds at n=8, 19 → 9 at
//! n=10**. The `reassoc` column reports it.
//!
//! That rule is what closes the last gap to hand-written code. The gadget
//! library needed `argmax` and `max_reduce` rewritten by hand as tournaments;
//! the compiler now derives the same transformation for any associative
//! operator, including compound ones it has never seen.
//!
//! # Results (2026-07-28)
//!
//! ```text
//!   program          n      ops ops/no-cse   interactive   rounds    reassoc
//!   relu-layer       1        3          3             3        3          3
//!   relu-layer      64      192   255 (1x)           129        3          3
//!   dot             64      127        127            65        2          2
//!   max-fold         4       12    52 (4x)             7        7          5
//!   max-fold         8       28 4372 (156x)            15       15          7
//!   max-fold        10       36 39364 (1093x)          19       19          9
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
use crate::mpcc::opt::reassociate_reductions;
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
    /// Rounds after the compiler rebalances compound reductions
    /// (`Rule::ReassociateReduction`). Equal to `rounds` when there is no
    /// dependent chain to rebalance, which is the common case.
    pub rounds_reassociated: usize,
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

/// Largest fold width the harness will build. The un-let-bound fold repeats
/// its accumulator three times per step, so the **source text** grows as `3^n`
/// — at n=16 that is tens of millions of characters before the compiler is
/// even called. This is a property of writing the fold without a binding, not
/// a compiler limit, but it will hang the benchmark if left unguarded.
const MAX_FOLD_WIDTH: usize = 10;

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

#[allow(clippy::type_complexity)]
fn compile_len(src: &str, cse: bool) -> Result<(usize, Program, Program), String> {
    let expr = parse(src).map_err(|e| format!("parse error: {e}"))?;
    let mut cc = if cse { CircuitCompiler::new() } else { CircuitCompiler::new().without_cse() };
    let circuit = cc.compile_fhe(&expr).map_err(|e| format!("compile error: {e}"))?;
    let program = lower(&circuit.flat_ops, &circuit.output_wire, &LowerConfig::default())?;
    let rebalanced = reassociate_reductions(&circuit.flat_ops, &circuit.output_wire);
    let program_opt = lower(&rebalanced, &circuit.output_wire, &LowerConfig::default())?;
    Ok((circuit.flat_ops.len(), program, program_opt))
}

pub fn measure(name: &str, src: &str, n: usize) -> Result<CircuitPoint, String> {
    let (ops, program, program_opt) = compile_len(src, true)?;
    let (ops_no_cse, _, _) = compile_len(src, false)?;
    let sched = schedule(&program);
    let interactive = program.code.iter().filter(|i| program.is_interactive(i)).count();
    Ok(CircuitPoint {
        program: name.to_string(),
        n,
        ops,
        ops_no_cse,
        interactive,
        rounds: sched.round_count(),
        rounds_reassociated: schedule(&program_opt).round_count(),
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
        if n > MAX_FOLD_WIDTH {
            return Err(format!(
                "--fold-widths {n} is too large: the un-let-bound fold's source text grows \
                 as 3^n, so building it would take ~3^{n} characters. Cap is {MAX_FOLD_WIDTH}."
            ));
        }
        pts.push(measure("max-fold", &max_fold_src(n), n)?);
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&pts).map_err(|e| e.to_string())?);
        return Ok(());
    }

    println!("Circuit generation quality — Expr -> CircuitCompiler -> FlatOp -> mpcc\n");
    println!(
        "  {:<12} {:>5} {:>8} {:>10} {:>13} {:>8} {:>10}",
        "program", "n", "ops", "ops/no-cse", "interactive", "rounds", "reassoc"
    );
    for p in &pts {
        let cse_note = if p.ops_no_cse > p.ops {
            format!("{} ({:.0}x)", p.ops_no_cse, p.ops_no_cse as f64 / p.ops.max(1) as f64)
        } else {
            p.ops_no_cse.to_string()
        };
        println!(
            "  {:<12} {:>5} {:>8} {:>10} {:>13} {:>8} {:>10}",
            p.program, p.n, p.ops, cse_note, p.interactive, p.rounds, p.rounds_reassociated
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
            "  max-fold:   as written, {} rounds at n={} and {} at n={} — LINEAR, because a\n  \
             fold is a dependent chain and ASAP is optimal only for a fixed lowering.\n  \
             Rebalanced into a tournament by Rule::ReassociateReduction: {} and {},\n  \
             i.e. 2*ceil(log2 n)+1. The compiler now does automatically what the\n  \
             hand-written gadget had to be rewritten by hand to do.",
            a.rounds, a.n, b.rounds, b.n, a.rounds_reassociated, b.rounds_reassociated
        );
        if b.rounds_reassociated >= b.rounds {
            return Err(format!(
                "max-fold at n={} was not rebalanced ({} -> {}) — the reduction rule \
                 has stopped firing",
                b.n, b.rounds, b.rounds_reassociated
            ));
        }
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

    /// The reduction rule must turn a fold's *linear* depth into *logarithmic*
    /// depth, and the gap must widen with width — a constant-factor
    /// improvement would mean it is rebalancing only part of the chain, which
    /// is exactly what happened before the depth guard was added (a rebalanced
    /// tree still contains chains, so re-applying the rule could deepen it).
    #[test]
    fn compound_reductions_become_logarithmic_in_depth() {
        let a = measure("max-fold", &max_fold_src(4), 4).unwrap();
        let b = measure("max-fold", &max_fold_src(8), 8).unwrap();
        assert!(a.rounds_reassociated < a.rounds && b.rounds_reassociated < b.rounds);
        // Doubling the width adds one tournament level (2 rounds), not n more.
        assert_eq!(
            b.rounds_reassociated - a.rounds_reassociated,
            2,
            "expected one extra level from n=4 to n=8, got {} -> {}",
            a.rounds_reassociated,
            b.rounds_reassociated
        );
        // ...while the unrebalanced chain doubles.
        assert!(b.rounds > a.rounds + 2, "the chain should grow linearly: {} -> {}", a.rounds, b.rounds);
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
