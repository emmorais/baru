//! `mpcc` — the Baru MPC compiler and protocol-comparison layer.
//!
//! Baru already had the hard part: a front-end that *flattens* a Lisp
//! expression into the `FlatOp` SSA block in
//! [`crate::zkp::circuit_compiler`], shared today by the ZKP and FHE backends.
//! This module takes that flattening and builds the piece MP-SPDZ has and Baru
//! did not — a protocol-neutral **bytecode**, a **round schedule**, and an
//! **analytic cost model** — so that one program can be costed, and eventually
//! run, under many MPC protocols and compared honestly.
//!
//! ```text
//!   Expr ──CircuitCompiler──► FlatOp SSA ──lower──► Bytecode ──schedule──► rounds
//!                                                       │            │
//!                                    protocols::REGISTRY┴──cost──► predicted
//!                                                       │            ├──► calibrate
//!                                             sim engines┴──vm────► measured
//! ```
//!
//! See `docs/mpc-compiler-roadmap.md` for the phasing and the target protocol
//! matrix.  P0 built the framework; P1 (in progress) adds executable protocols
//! and closes the loop — `baru mpcc calibrate` runs a program and reports
//! measured cost against the model's prediction, which is the only thing that
//! makes the estimates for the *unimplemented* protocols worth anything.
//!
//! ## CLI
//!
//! ```text
//! baru mpcc protocols                            # the protocol matrix
//! baru mpcc dump      --expr '(* x y)'           # bytecode listing
//! baru mpcc analyse   --expr '(* x y)'           # + instruction census and schedule
//! baru mpcc compare   --expr '(* x y)' [opts]    # ranked cost estimate per protocol
//! baru mpcc run       --case dot2 --protocol shamir --parties 5
//! baru mpcc calibrate --case dot2                # measured vs predicted
//! baru mpcc optimize  --case dot4-builtin        # backend-aware rewriting
//! ```

pub mod assign;
pub mod cost;
pub mod fhe_cost;
pub mod ir;
pub mod lower;
pub mod opt;
pub mod ot_cost;
pub mod protocols;
pub mod schedule;
pub mod sim;
pub mod vm;

use std::collections::HashMap;

use serde::Serialize;

use crate::zkp::circuit_compiler::CircuitCompiler;

pub use cost::{estimate, CostEstimate, Deployment};
pub use fhe_cost::FheSpec;
pub use ir::{Domain, Program};
pub use lower::{lower, LowerConfig};
pub use opt::{optimize, Optimized};

pub use protocols::{find, ProtocolSpec, Security, REGISTRY};
pub use schedule::{schedule, Schedule};
pub use sim::{DealerEngine, ShamirEngine};
pub use vm::{execute, Measured};

/// Protocols the VM can actually execute today.  Everything else in the
/// registry is analysis-only until its roadmap phase lands.
pub const RUNNABLE: &[&str] = &["dealer", "shamir"];

/// Execute `prog` under the named protocol.
///
/// The program must have been lowered for that protocol's native domain
/// ([`cost::native_domain`]); otherwise a measured run and a predicted one are
/// not comparable, and `calibrate` would be measuring the wrong thing.
pub fn run_protocol(
    id: &str,
    parties: usize,
    prog: &Program,
    sched: &Schedule,
    inputs: &HashMap<String, i64>,
) -> Result<Measured, String> {
    match id {
        "dealer" => {
            let bits = match prog.domain {
                Domain::Ring { k } => k,
                other => {
                    return Err(format!(
                        "dealer computes mod 2^k, but the program was lowered for {}",
                        other
                    ))
                }
            };
            let mut e = DealerEngine::new(parties, bits)?;
            execute(prog, sched, &mut e, id, inputs)
        }
        "shamir" => {
            let mut e = ShamirEngine::new(parties)?;
            execute(prog, sched, &mut e, id, inputs)
        }
        other => Err(format!(
            "`{}` is not executable yet — Baru can run {:?}.  Every other protocol \
             in the registry is costed analytically only; see \
             docs/mpc-compiler-roadmap.md for its phase.",
            other, RUNNABLE
        )),
    }
}

/// One row of `baru mpcc calibrate`: what the model said, what actually happened.
#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    pub protocol: String,
    pub parties: usize,
    pub predicted_rounds: u64,
    pub measured_rounds: u64,
    pub predicted_online_bytes: u64,
    pub measured_online_bytes: u64,
    pub predicted_triples: u64,
    pub measured_triples: u64,
    pub predicted_muls: u64,
    pub measured_muls: u64,
    /// Program output, and the plaintext value it is checked against.
    pub result: Option<i64>,
    pub expected: Option<i64>,
    pub correct: bool,
}

impl Calibration {
    /// Signed relative error of the prediction, as a fraction of the measurement.
    pub fn bytes_error(&self) -> f64 {
        rel_error(self.predicted_online_bytes, self.measured_online_bytes)
    }
    pub fn rounds_error(&self) -> f64 {
        rel_error(self.predicted_rounds, self.measured_rounds)
    }
}

fn rel_error(predicted: u64, measured: u64) -> f64 {
    if measured == 0 {
        if predicted == 0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (predicted as f64 - measured as f64) / measured as f64
    }
}

/// Run `src` under every executable protocol and compare against the model.
pub fn calibrate(
    src: &str,
    inputs: &HashMap<String, i64>,
    parties: usize,
    dep: &Deployment,
    bit_width: u32,
) -> Result<Vec<Calibration>, String> {
    // Ground truth from the plain interpreter, so a protocol bug cannot hide.
    let expected = plain_value(src, inputs);

    let mut out = Vec::new();
    for id in RUNNABLE {
        let spec = find(id).ok_or_else(|| format!("unknown protocol `{}`", id))?;
        if !spec.supports_parties(parties) {
            continue;
        }
        let cfg = LowerConfig {
            domain: cost::native_domain(spec, 64),
            parties,
            bit_width,
            public_vars: Vec::new(),
        };
        let prog = compile(src, &cfg)?;
        let sched = schedule(&prog);

        let mut dep = dep.clone();
        dep.parties = parties;
        dep.allow_dealer = true;
        let predicted = estimate(&prog, &sched, spec, &dep)?;
        let measured = run_protocol(id, parties, &prog, &sched, inputs)?;

        let result = measured.outputs.first().map(|(_, v)| *v);
        out.push(Calibration {
            protocol: (*id).to_string(),
            parties,
            predicted_rounds: predicted.rounds,
            measured_rounds: measured.rounds,
            predicted_online_bytes: predicted.online_bytes,
            measured_online_bytes: measured.online_bytes,
            predicted_triples: predicted.triples,
            measured_triples: measured.triples,
            predicted_muls: predicted.muls,
            measured_muls: measured.muls,
            result,
            expected,
            correct: match (result, expected) {
                (Some(r), Some(e)) => r == e,
                _ => false,
            },
        });
    }
    if out.is_empty() {
        return Err(format!(
            "no executable protocol supports {} parties (dealer needs 2+, shamir 3+)",
            parties
        ));
    }
    Ok(out)
}

/// Evaluate `src` in the plain interpreter to get the ground-truth answer.
fn plain_value(src: &str, inputs: &HashMap<String, i64>) -> Option<i64> {
    let mut names: Vec<&String> = inputs.keys().collect();
    names.sort();
    let bindings: Vec<String> = names
        .iter()
        .map(|k| format!("({} {})", k, inputs[*k]))
        .collect();
    let wrapped = format!("(let ({}) {})", bindings.join(" "), src);
    let expr = crate::interpreter::parse(&wrapped).ok()?;
    match crate::interpreter::Participant::new(0).eval_quiet(&expr) {
        crate::interpreter::Types::Num(n) => Some(n),
        _ => None,
    }
}

/// Parse, flatten and lower a Baru expression to MPC bytecode.
pub fn compile(src: &str, cfg: &LowerConfig) -> Result<Program, String> {
    let expr = crate::interpreter::parse(src).map_err(|e| format!("parse error: {}", e))?;
    let mut cc = CircuitCompiler::new();
    let circuit = cc
        .compile_fhe(&expr)
        .map_err(|e| format!("circuit compile error: {}", e))?;
    lower(&circuit.flat_ops, &circuit.output_wire, cfg)
}

/// A protocol that cannot serve the deployment, and why.
#[derive(Debug, Clone, Serialize)]
pub struct Rejected {
    pub protocol: String,
    pub reason: String,
}

/// The result of costing one program against the whole registry.
#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    pub source: String,
    pub parties: usize,
    pub network: String,
    pub min_security: String,
    pub instructions: u64,
    pub interactive: u64,
    pub rounds: usize,
    pub batching_factor: f64,
    /// Eligible protocols, cheapest first.
    pub ranked: Vec<CostEstimate>,
    pub rejected: Vec<Rejected>,
}

/// Cost `prog` against every protocol in the registry and rank the eligible
/// ones by predicted total time.
pub fn compare(
    source: &str,
    prog: &Program,
    sched: &Schedule,
    dep: &Deployment,
    network: &str,
    only: &[String],
) -> Comparison {
    let mut ranked = Vec::new();
    let mut rejected = Vec::new();
    for spec in REGISTRY {
        if !only.is_empty() && !only.iter().any(|o| o == spec.id) {
            continue;
        }
        match estimate(prog, sched, spec, dep) {
            Ok(e) => ranked.push(e),
            Err(reason) => rejected.push(Rejected {
                protocol: spec.id.to_string(),
                reason,
            }),
        }
    }
    ranked.sort_by(|a, b| {
        a.total_secs
            .partial_cmp(&b.total_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let counts = prog.counts();
    Comparison {
        source: source.to_string(),
        parties: dep.parties,
        network: network.to_string(),
        min_security: dep.min_security.to_string(),
        instructions: counts.instructions,
        interactive: counts.interactive,
        rounds: sched.round_count(),
        batching_factor: sched.batching_factor(),
        ranked,
        rejected,
    }
}

// ─── Formatting ───────────────────────────────────────────────────────────────

fn human_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} B", b)
    } else {
        format!("{:.1} {}", v, UNITS[i])
    }
}

fn human_secs(s: f64) -> String {
    if s < 1e-3 {
        format!("{:.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{:.2} s", s)
    }
}

/// Render a comparison as a table.
pub fn format_comparison(c: &Comparison) -> String {
    let mut out = String::new();
    out.push_str(&format!("program   {}\n", c.source));
    out.push_str(&format!(
        "shape     {} instructions ({} interactive), {} dependence levels, \
         batching {:.1}x\n",
        c.instructions, c.interactive, c.rounds, c.batching_factor
    ));
    out.push_str(&format!(
        "deploy    {} parties, {}, ≥{} security\n\n",
        c.parties, c.network, c.min_security
    ));
    out.push_str(&format!(
        "{:<10} {:>7} {:>10} {:>11} {:>11} {:>11}  {:<10} {}\n",
        "protocol", "rounds", "online", "offline", "online t", "total t", "bottleneck", "status"
    ));
    for e in &c.ranked {
        out.push_str(&format!(
            "{:<10} {:>7} {:>10} {:>11} {:>11} {:>11}  {:<10} {}\n",
            e.protocol,
            e.rounds,
            human_bytes(e.online_bytes),
            human_bytes(e.offline_bytes),
            human_secs(e.online_secs),
            human_secs(e.total_secs),
            e.bottleneck,
            if e.implemented {
                "implemented"
            } else {
                "estimate"
            },
        ));
    }
    if !c.rejected.is_empty() {
        out.push_str("\nnot applicable:\n");
        for r in &c.rejected {
            out.push_str(&format!("  {:<10} {}\n", r.protocol, r.reason));
        }
    }
    // Collect distinct caveats once rather than repeating them per row.
    let mut seen: Vec<&str> = Vec::new();
    for e in &c.ranked {
        for n in &e.notes {
            if !seen.contains(&n.as_str()) {
                seen.push(n.as_str());
            }
        }
    }
    if c.ranked.iter().any(|e| !e.implemented) {
        seen.push(
            "rows marked `estimate` are analytic only — Baru cannot run those protocols yet \
             (see `baru mpcc protocols` for the roadmap phase)",
        );
    }
    if !seen.is_empty() {
        out.push_str("\nnotes:\n");
        for n in seen {
            out.push_str(&format!("  · {}\n", n));
        }
    }
    out
}

/// Render a calibration run: what the model predicted against what happened.
pub fn format_calibration(rows: &[Calibration], source: &str) -> String {
    let mut out = format!("program   {}\n\n", source);
    out.push_str(&format!(
        "{:<10} {:>6} {:>17} {:>21} {:>15} {:>15}  {}\n",
        "protocol",
        "parties",
        "rounds pred/meas",
        "online bytes pred/meas",
        "muls p/m",
        "triples p/m",
        "result"
    ));
    for r in rows {
        out.push_str(&format!(
            "{:<10} {:>6} {:>8}/{:<8} {:>10}/{:<10} {:>7}/{:<7} {:>7}/{:<7}  {}\n",
            r.protocol,
            r.parties,
            r.predicted_rounds,
            r.measured_rounds,
            r.predicted_online_bytes,
            r.measured_online_bytes,
            r.predicted_muls,
            r.measured_muls,
            r.predicted_triples,
            r.measured_triples,
            match (r.correct, r.result, r.expected) {
                (true, Some(v), _) => format!("{} ✓", v),
                (false, Some(v), Some(e)) => format!("{} ✗ (expected {})", v, e),
                (_, Some(v), None) => format!("{} (no plain baseline)", v),
                _ => "no output".to_string(),
            }
        ));
    }
    out.push_str("\nprediction error (signed, relative to the measurement):\n");
    for r in rows {
        out.push_str(&format!(
            "  {:<10} rounds {:+.1}%   bytes {:+.1}%\n",
            r.protocol,
            r.rounds_error() * 100.0,
            r.bytes_error() * 100.0
        ));
    }
    out.push_str(
        "\nnotes:\n  · online rounds and bytes are measured from the simulated protocol; \
         preprocessing\n    is drawn locally and only counted, so triples are trustworthy and \
         offline *time*\n    is not measured at all (see crate::mpcc::vm).\n",
    );
    out
}

/// Render a backend-aware optimization run.
pub fn format_optimizations(rows: &[Optimized], source: &str) -> String {
    let mut out = format!("program   {}\n\n", source);
    out.push_str(&format!(
        "{:<10} {:>13} {:>15} {:>17} {:>9}  {}\n",
        "protocol", "ops b/a", "AND gates b/a", "predicted b/a", "speed-up", "rules kept"
    ));
    for r in rows {
        // Aggregate per rule, and show `rule×n` when a per-site rule was
        // accepted at several sites — a rule taken at 2 of 5 sites is a
        // different answer from one taken everywhere, and the whole point of
        // per-site search is that backends differ in exactly that way.
        let mut order: Vec<&str> = Vec::new();
        let mut hits: HashMap<&str, usize> = HashMap::new();
        for a in &r.applied {
            if !hits.contains_key(a.rule) {
                order.push(a.rule);
            }
            *hits.entry(a.rule).or_insert(0) += 1;
        }
        let rules: Vec<String> = order
            .iter()
            .map(|name| match hits[*name] {
                1 => (*name).to_string(),
                n => format!("{name}×{n}"),
            })
            .collect();
        out.push_str(&format!(
            "{:<10} {:>6}/{:<6} {:>7}/{:<7} {:>8}/{:<8} {:>8.2}x  {}\n",
            r.protocol,
            r.before.ops,
            r.after.ops,
            r.before.and_gates,
            r.after.and_gates,
            human_secs(r.before.secs),
            human_secs(r.after.secs),
            r.speedup(),
            if rules.is_empty() {
                "-".to_string()
            } else {
                rules.join(", ")
            }
        ));
    }
    // Compare the actual instruction sequences, not a proxy: the claim is that
    // the circuits differ, so it should be checked on the circuits.
    let differ = rows.windows(2).any(|w| w[0].flat != w[1].flat);
    out.push_str(&format!(
        "\n{}\n",
        if differ {
            "the optimized circuits differ between backends — which is the point: a \
             backend-blind\noptimizer would have to pick one and be wrong for the others"
        } else {
            "every backend ended up with the same circuit for this program.  Multiplication \
             by a\nconstant with several set bits is where they diverge — try\n  \
             --expr '(+ (* x 100) (* y 100))' --parties 2 --only yao,semi2k"
        }
    ));
    if rows.iter().all(|r| r.verified) && !rows.is_empty() {
        out.push_str("all rewrites differentially checked against the original circuit\n");
    }
    out
}

/// Render a per-operation protocol assignment.
pub fn format_assignment(a: &assign::Assigned, source: &str) -> String {
    // Long generated programs make the table unreadable; the shape is what
    // matters, not 8 KB of s-expression.
    let shown: String = if source.chars().count() > 96 {
        format!(
            "{}… ({} chars)",
            source.chars().take(96).collect::<String>(),
            source.chars().count()
        )
    } else {
        source.to_string()
    };
    let mut out = format!("program   {}\n", shown);
    out.push_str(&format!(
        "split     {} (arithmetic) / {} (binary){}\n\n",
        a.arith_protocol,
        a.binary_protocol,
        match &a.fhe_scheme {
            Some(f) => format!(" / {f} (FHE)"),
            None => " / no FHE".to_string(),
        }
    ));
    out.push_str(&format!(
        "{:<14} {:>8} {:>11} {:>11} {:>12} {:>10}\n",
        "assignment", "rounds", "online", "offline", "conversions", "predicted"
    ));
    let row = |name: &str, c: &assign::MixedCost| {
        format!(
            "{:<14} {:>8} {:>11} {:>11} {:>12} {:>10}\n",
            name,
            c.rounds,
            human_bytes(c.online_bytes),
            human_bytes(c.offline_bytes),
            c.conversions,
            human_secs(c.secs)
        )
    };
    out.push_str(&row("all-arithmetic", &a.all_arith));
    out.push_str(&row("all-binary", &a.all_binary));
    out.push_str(&row("mixed (best)", &a.best));

    out.push('\n');
    if a.is_mixed() {
        out.push_str(&format!(
            "mixed wins {:.2}x over the better pure protocol, by moving:\n",
            a.speedup()
        ));
        for (kind, side) in a.assignment.deviations() {
            out.push_str(&format!("  · {:<10} -> {}\n", kind, side.name()));
        }
        out.push_str(&format!(
            "conversions land at {} dependence level(s), which is what the round count\ncharges — crossings at one level batch into a single round.\n",
            a.best.conversion_levels
        ));
    } else {
        out.push_str(
            "no mixed assignment beat a pure one for this program — which is the right\n\
             answer when one sharing suits every operation.  Mixing pays when a program\n\
             contains both multiplications and comparisons, e.g.\n  \
             --expr '(+ (* a b) (+ (> c d) (> e f)))' --wan --parties 2 --dishonest\n",
        );
    }
    out.push_str(&format!(
        "\n{} assignments priced; the search is greedy over operation kinds, so this is a\nlocal optimum (optimal assignment is NP-hard — Silph uses ILP, SING a GNN).\n",
        a.evaluated
    ));
    out
}

/// Render the protocol registry as a table.
pub fn format_registry() -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<10} {:<38} {:<9} {:<8} {:<10} {:<12} {}\n",
        "id", "protocol", "domain", "parties", "majority", "security", "status"
    ));
    for p in REGISTRY {
        let parties = match p.max_parties {
            Some(m) if m == p.min_parties => format!("{}", m),
            Some(m) => format!("{}-{}", p.min_parties, m),
            None => format!("{}+", p.min_parties),
        };
        out.push_str(&format!(
            "{:<10} {:<38} {:<9} {:<8} {:<10} {:<12} {}\n",
            p.id,
            p.name,
            p.domain.to_string(),
            parties,
            p.majority.to_string(),
            p.security.to_string(),
            p.status
        ));
    }
    out.push_str("\nreferences:\n");
    for p in REGISTRY {
        out.push_str(&format!("  {:<10} {}\n", p.id, p.reference));
    }
    out
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

const USAGE: &str = "\
baru mpcc — MPC compiler and protocol comparison

  baru mpcc protocols
  baru mpcc dump    (--expr EXPR | --case NAME) [lowering opts]
  baru mpcc analyse (--expr EXPR | --case NAME) [lowering opts]
  baru mpcc compare (--expr EXPR | --case NAME) [lowering opts] [deployment opts]
  baru mpcc run     (--expr EXPR | --case NAME) --protocol ID [--input k=v ...]
  baru mpcc calibrate (--expr EXPR | --case NAME) [--input k=v ...] [deployment opts]
  baru mpcc optimize  (--expr EXPR | --case NAME) [--protocol ID] [deployment opts]
  baru mpcc assign    (--expr EXPR | --case NAME) [--protocol ID] [deployment opts]

run/calibrate opts
  --protocol ID                     dealer | shamir  (the executable protocols)
  --input NAME=VAL                  input binding, repeatable; implied by --case

lowering opts
  --domain ring64|ring32|field128   arithmetic domain      (default ring64)
  --bits N                          comparison/decomposition width (default 32)
  --public a,b,c                    variables public to all parties

deployment opts
  --parties N                       number of computing parties (default 3)
  --security semi-honest|malicious  minimum required          (default semi-honest)
  --lan | --wan                     network profile           (default lan)
  --dishonest                       reject honest-majority protocols
  --dealer                          allow the trusted-dealer lower bound
  --only id1,id2                    restrict to these protocols
  --no-offline                      rank by online time only
  --json                            machine-readable output
";

struct Args {
    expr: Option<String>,
    case: Option<String>,
    lower: LowerConfig,
    dep: Deployment,
    network: &'static str,
    only: Vec<String>,
    json: bool,
    protocol: Option<String>,
    inputs: Vec<(String, i64)>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut expr = None;
    let mut case = None;
    let mut lower = LowerConfig::default();
    let mut parties = 3usize;
    let mut wan = false;
    let mut security = Security::SemiHonest;
    let mut dishonest = false;
    let mut dealer = false;
    let mut include_offline = true;
    let mut only: Vec<String> = Vec::new();
    let mut json = false;
    let mut protocol: Option<String> = None;
    let mut inputs: Vec<(String, i64)> = Vec::new();

    let mut i = 0;
    // Values are read as `args[i + 1]`; each arm advances `i` past its value.
    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", flag))
        };
        let takes_value = match flag {
            "--expr" => {
                expr = Some(value()?);
                true
            }
            "--case" => {
                case = Some(value()?);
                true
            }
            "--domain" => {
                let v = value()?;
                lower.domain = match v.as_str() {
                    "ring64" => Domain::Ring { k: 64 },
                    "ring32" => Domain::Ring { k: 32 },
                    "field128" => Domain::Field { bits: 128 },
                    other => return Err(format!("unknown domain `{}`", other)),
                };
                true
            }
            "--bits" => {
                lower.bit_width = value()?
                    .parse()
                    .map_err(|_| "--bits needs an integer".to_string())?;
                true
            }
            "--public" => {
                lower.public_vars = value()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                true
            }
            "--parties" => {
                parties = value()?
                    .parse()
                    .map_err(|_| "--parties needs an integer".to_string())?;
                true
            }
            "--security" => {
                let v = value()?;
                security =
                    Security::parse(&v).ok_or_else(|| format!("unknown security level `{}`", v))?;
                true
            }
            "--protocol" => {
                protocol = Some(value()?);
                true
            }
            "--input" => {
                let v = value()?;
                let (k, n) = v
                    .split_once('=')
                    .ok_or_else(|| format!("--input `{}` is not NAME=VAL", v))?;
                let n: i64 = n
                    .parse()
                    .map_err(|_| format!("--input `{}`: value is not an integer", v))?;
                inputs.push((k.to_string(), n));
                true
            }
            "--only" => {
                only = value()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                true
            }
            "--lan" => {
                wan = false;
                false
            }
            "--wan" => {
                wan = true;
                false
            }
            "--dishonest" => {
                dishonest = true;
                false
            }
            "--dealer" => {
                dealer = true;
                false
            }
            "--no-offline" => {
                include_offline = false;
                false
            }
            "--json" => {
                json = true;
                false
            }
            other => return Err(format!("unknown flag `{}`\n\n{}", other, USAGE)),
        };
        i += if takes_value { 2 } else { 1 };
    }

    if parties == 0 {
        return Err("--parties must be at least 1".to_string());
    }
    // Build the deployment last, so flags are order-independent.
    let mut dep = if wan {
        Deployment::wan(parties)
    } else {
        Deployment::lan(parties)
    };
    dep.min_security = security;
    dep.require_dishonest_majority = dishonest;
    dep.allow_dealer = dealer;
    dep.include_offline = include_offline;
    lower.parties = parties;

    Ok(Args {
        expr,
        case,
        lower,
        dep,
        network: if wan {
            "WAN (100 ms RTT, 100 Mbit/s)"
        } else {
            "LAN (0.1 ms RTT, 10 Gbit/s)"
        },
        only,
        json,
        protocol,
        inputs,
    })
}

/// Input bindings for `run`/`calibrate`: explicit `--input` flags win, and a
/// `--case` supplies the suite's own inputs when none are given.
fn inputs_of(a: &Args) -> Result<HashMap<String, i64>, String> {
    if !a.inputs.is_empty() {
        return Ok(a.inputs.iter().cloned().collect());
    }
    if let Some(name) = &a.case {
        if let Some(c) = crate::bench::SUITE.iter().find(|c| c.name == *name) {
            return Ok(c
                .inputs
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect());
        }
    }
    Err(
        "need --input NAME=VAL for every program input (or use --case, which \
         supplies the suite's own bindings)"
            .to_string(),
    )
}

/// Resolve `--expr` / `--case` to source text.
fn source_of(a: &Args) -> Result<String, String> {
    if let Some(e) = &a.expr {
        return Ok(e.clone());
    }
    if let Some(name) = &a.case {
        let case = crate::bench::SUITE
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| {
                let names: Vec<&str> = crate::bench::SUITE.iter().map(|c| c.name).collect();
                format!("unknown case `{}`; available: {}", name, names.join(", "))
            })?;
        // MPC cannot branch on a secret, so prefer the mux-style rewrite where
        // the suite provides one — that is the program a protocol would run.
        return Ok(case.mpc_expr.unwrap_or(case.expr).to_string());
    }
    Err(format!("need --expr or --case\n\n{}", USAGE))
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    match sub {
        "" | "help" | "--help" | "-h" => {
            print!("{}", USAGE);
            Ok(())
        }
        "protocols" => {
            print!("{}", format_registry());
            Ok(())
        }
        "run" => {
            let a = parse_args(&args[1..])?;
            let src = source_of(&a)?;
            let id = a
                .protocol
                .clone()
                .ok_or_else(|| format!("--protocol is required; Baru can run {:?}", RUNNABLE))?;
            let spec = find(&id).ok_or_else(|| format!("unknown protocol `{}`", id))?;
            let cfg = LowerConfig {
                domain: cost::native_domain(spec, 64),
                parties: a.dep.parties,
                bit_width: a.lower.bit_width,
                public_vars: a.lower.public_vars.clone(),
            };
            let prog = compile(&src, &cfg)?;
            let sched = schedule(&prog);
            let measured = run_protocol(&id, a.dep.parties, &prog, &sched, &inputs_of(&a)?)?;
            if a.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&measured).map_err(|e| e.to_string())?
                );
            } else {
                println!("program   {}", src);
                println!(
                    "protocol  {} over {}, {} parties",
                    measured.protocol, prog.domain, measured.parties
                );
                for (name, value) in &measured.outputs {
                    println!("output    {} = {}", name, value);
                }
                println!(
                    "cost      {} rounds, {} bytes/party online, {} muls, {} opens, \
                     {} triples ({} µs local)",
                    measured.rounds,
                    measured.online_bytes,
                    measured.muls,
                    measured.opens,
                    measured.triples,
                    measured.micros
                );
            }
            Ok(())
        }
        "calibrate" => {
            let a = parse_args(&args[1..])?;
            let src = source_of(&a)?;
            let rows = calibrate(
                &src,
                &inputs_of(&a)?,
                a.dep.parties,
                &a.dep,
                a.lower.bit_width,
            )?;
            if a.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", format_calibration(&rows, &src));
            }
            if rows.iter().any(|r| !r.correct) {
                return Err("a protocol produced the wrong answer".to_string());
            }
            Ok(())
        }
        "assign" => {
            let a = parse_args(&args[1..])?;
            let src = source_of(&a)?;
            let prog = compile(&src, &a.lower)?;
            let sched = schedule(&prog);
            // Default pair: the dishonest-majority ring protocol and the
            // matching binary one, i.e. the classic A/B split.
            let arith_id = a.protocol.clone().unwrap_or_else(|| "semi2k".to_string());
            let arith = find(&arith_id).ok_or_else(|| format!("unknown protocol `{arith_id}`"))?;
            let binary = find("semi-bin").ok_or("semi-bin missing from the registry")?;
            let mut pair = assign::Pair::new(arith, binary)?;
            // FHE joins the assignment unless asked not to — this is the part
            // HyCC / Silph / SING structurally cannot do.
            if !a.only.iter().any(|o| o == "no-fhe") {
                let fhe_id = a
                    .only
                    .iter()
                    .find(|o| fhe_cost::find(o).is_some())
                    .cloned()
                    .unwrap_or_else(|| "bfv".to_string());
                let spec = fhe_cost::find(&fhe_id)
                    .ok_or_else(|| format!("unknown FHE scheme `{fhe_id}`"))?;
                pair = pair.with_fhe(spec);
            }
            let out = assign::assign(&prog, &sched, &pair, &a.dep, a.lower.bit_width)?;
            if a.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", format_assignment(&out, &src));
            }
            Ok(())
        }
        "optimize" | "optimise" => {
            let a = parse_args(&args[1..])?;
            let src = source_of(&a)?;
            let expr =
                crate::interpreter::parse(&src).map_err(|e| format!("parse error: {}", e))?;
            let circuit = CircuitCompiler::new()
                .compile_fhe(&expr)
                .map_err(|e| format!("circuit compile error: {}", e))?;

            // With no --protocol, optimize for every eligible backend, which is
            // the point: the circuits come out different.
            let targets: Vec<&ProtocolSpec> = match &a.protocol {
                Some(id) => vec![find(id).ok_or_else(|| format!("unknown protocol `{}`", id))?],
                None => REGISTRY
                    .iter()
                    .filter(|s| cost::eligibility(s, &a.dep).is_ok())
                    .filter(|s| a.only.is_empty() || a.only.iter().any(|o| o == s.id))
                    .collect(),
            };

            let mut results = Vec::new();
            for spec in targets {
                let cfg = LowerConfig {
                    domain: cost::native_domain(spec, 64),
                    parties: a.dep.parties,
                    bit_width: a.lower.bit_width,
                    public_vars: a.lower.public_vars.clone(),
                };
                match opt::optimize(
                    &circuit.flat_ops,
                    &circuit.output_wire,
                    spec,
                    &a.dep,
                    &cfg,
                    true,
                ) {
                    Ok(o) => results.push(o),
                    Err(e) => eprintln!("{}: {}", spec.id, e),
                }
            }
            if a.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&results).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", format_optimizations(&results, &src));
            }
            Ok(())
        }
        "dump" | "analyse" | "analyze" | "compare" => {
            let a = parse_args(&args[1..])?;
            let src = source_of(&a)?;
            let prog = compile(&src, &a.lower)?;
            let sched = schedule(&prog);
            match sub {
                "dump" => print!("{}", prog.listing()),
                "analyse" | "analyze" => {
                    print!("{}", prog.listing());
                    let counts = prog.counts();
                    println!(
                        "\n{} instructions: {} local, {} interactive",
                        counts.instructions, counts.local, counts.interactive
                    );
                    let mut ops: Vec<(&&str, &u64)> = counts.by_op.iter().collect();
                    ops.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                    for (op, n) in ops {
                        println!("  {:<10} {}", op, n);
                    }
                    println!(
                        "\n{} communication rounds, {:.1} interactive ops per round \
                         (widest round: {})",
                        sched.round_count(),
                        sched.batching_factor(),
                        sched.max_round_ops()
                    );
                }
                _ => {
                    let c = compare(&src, &prog, &sched, &a.dep, a.network, &a.only);
                    if a.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&c).map_err(|e| e.to_string())?
                        );
                    } else {
                        print!("{}", format_comparison(&c));
                    }
                }
            }
            Ok(())
        }
        other => Err(format!("unknown subcommand `{}`\n\n{}", other, USAGE)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_comparison_of_a_relu() {
        let cfg = LowerConfig::default();
        let prog = compile("(* x (> x 0))", &cfg).expect("compile");
        let sched = schedule(&prog);
        let c = compare("relu", &prog, &sched, &Deployment::lan(3), "LAN", &[]);
        assert!(!c.ranked.is_empty());
        // 3-party LAN, semi-honest: ABY3 is eligible; 2-party-only Yao is not.
        assert!(c.ranked.iter().any(|e| e.protocol == "aby3"));
        assert!(c.rejected.iter().any(|r| r.protocol == "yao"));
        // Ranking must be monotone in predicted time.
        for w in c.ranked.windows(2) {
            assert!(w[0].total_secs <= w[1].total_secs);
        }
    }

    #[test]
    fn wan_and_lan_can_disagree_about_the_best_protocol() {
        // A long serial dependence chain: latency-bound on a WAN, so a
        // constant-round protocol should climb the ranking there.
        let cfg = LowerConfig::default();
        let prog = compile("(* (* (* (* x y) x) y) x)", &cfg).expect("compile");
        let sched = schedule(&prog);
        let lan = compare("chain", &prog, &sched, &Deployment::lan(2), "LAN", &[]);
        let wan = compare("chain", &prog, &sched, &Deployment::wan(2), "WAN", &[]);
        let pos = |c: &Comparison, id: &str| c.ranked.iter().position(|e| e.protocol == id);
        let (yao_lan, yao_wan) = (pos(&lan, "yao"), pos(&wan, "yao"));
        assert!(yao_lan.is_some() && yao_wan.is_some(), "yao runs 2-party");
        assert!(
            yao_wan <= yao_lan,
            "constant-round protocols should not lose ground on a WAN: {:?} vs {:?}",
            yao_wan,
            yao_lan
        );
    }

    #[test]
    fn cli_rejects_unknown_flags_and_cases() {
        assert!(run_cli(&["compare".into(), "--nope".into()]).is_err());
        assert!(run_cli(&["compare".into(), "--case".into(), "nosuch".into()]).is_err());
        assert!(run_cli(&["protocols".into()]).is_ok());
    }

    fn bindings(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn simulated_protocols_compute_the_same_answers_as_the_interpreter() {
        let cases: &[(&str, &[(&str, i64)])] = &[
            ("(* x y)", &[("x", -6), ("y", 7)]),
            ("(+ (* x 3) (* y 7))", &[("x", 12), ("y", 5)]),
            ("(* (* (* x y) x) y)", &[("x", 3), ("y", -2)]),
            ("(- (* x x) (* y y))", &[("x", -9), ("y", 4)]),
            (
                "(+ (+ (* a b) (* c d)) (* e f))",
                &[("a", 2), ("b", 3), ("c", 4), ("d", 5), ("e", 6), ("f", 7)],
            ),
        ];
        for parties in [3usize, 5] {
            for (src, ins) in cases {
                let rows = calibrate(src, &bindings(ins), parties, &Deployment::lan(parties), 32)
                    .unwrap_or_else(|e| panic!("{} @ {} parties: {}", src, parties, e));
                assert_eq!(rows.len(), 2, "dealer and shamir both run at {}", parties);
                for r in &rows {
                    assert!(
                        r.correct,
                        "{} under {} with {} parties: got {:?}, expected {:?}",
                        src, r.protocol, parties, r.result, r.expected
                    );
                }
            }
        }
    }

    #[test]
    fn the_cost_model_agrees_with_the_simulator_on_the_online_phase() {
        // Not a tautology: the model derives rounds from the schedule's depth
        // and bytes from per-protocol volume rules, while the simulator counts
        // actual send points during a real execution.  A mistake in either the
        // scheduler or a protocol's volume rule shows up here.
        for (src, ins) in [
            ("(* x y)", vec![("x", 6), ("y", 7)]),
            ("(* (* (* x y) x) y)", vec![("x", 3), ("y", -2)]),
            (
                "(+ (* a b) (* c d))",
                vec![("a", 2), ("b", 3), ("c", 4), ("d", 5)],
            ),
        ] {
            let rows = calibrate(src, &bindings(&ins), 3, &Deployment::lan(3), 32).unwrap();
            for r in rows {
                assert_eq!(
                    r.predicted_rounds, r.measured_rounds,
                    "{} {}",
                    src, r.protocol
                );
                assert_eq!(
                    r.predicted_online_bytes, r.measured_online_bytes,
                    "{} {}",
                    src, r.protocol
                );
                assert_eq!(
                    r.predicted_triples, r.measured_triples,
                    "{} {}",
                    src, r.protocol
                );
            }
        }
    }

    #[test]
    fn independent_multiplications_really_do_collapse_into_one_round() {
        let parallel = calibrate(
            "(+ (+ (* a b) (* c d)) (* e f))",
            &bindings(&[("a", 2), ("b", 3), ("c", 4), ("d", 5), ("e", 6), ("f", 7)]),
            3,
            &Deployment::lan(3),
            32,
        )
        .unwrap();
        let serial = calibrate(
            "(* (* (* x y) x) y)",
            &bindings(&[("x", 3), ("y", -2)]),
            3,
            &Deployment::lan(3),
            32,
        )
        .unwrap();
        // Three multiplications either way; batching turns 3 rounds into 1.
        assert_eq!(parallel[0].measured_triples, 3);
        assert_eq!(serial[0].measured_triples, 3);
        assert!(
            parallel[0].measured_rounds < serial[0].measured_rounds,
            "parallel {} should beat serial {}",
            parallel[0].measured_rounds,
            serial[0].measured_rounds
        );
    }

    #[test]
    fn comparisons_run_and_agree_with_the_interpreter() {
        // Phase P2: the whole point is that these no longer merely *cost*.
        let cases: &[(&str, &[(&str, i64)])] = &[
            ("(* x (> x 0))", &[("x", -7)]),
            ("(* x (> x 0))", &[("x", 12)]),
            (
                "(let ((m (> x y))) (+ (* x m) (* y (- 1 m))))",
                &[("x", 9), ("y", 4)],
            ),
            (
                "(let ((m (> x y))) (+ (* x m) (* y (- 1 m))))",
                &[("x", -9), ("y", -4)],
            ),
            ("(< x y)", &[("x", -3), ("y", 3)]),
            ("(== x y)", &[("x", 5), ("y", 5)]),
            ("(== x y)", &[("x", 5), ("y", 6)]),
            ("(ash-right x 3)", &[("x", 1000)]),
        ];
        for (src, ins) in cases {
            for parties in [3usize, 5] {
                let rows = calibrate(src, &bindings(ins), parties, &Deployment::lan(parties), 32)
                    .unwrap_or_else(|e| panic!("{} @ {}: {}", src, parties, e));
                for r in &rows {
                    assert!(
                        r.correct,
                        "{} under {} with {} parties: got {:?}, expected {:?}",
                        src, r.protocol, parties, r.result, r.expected
                    );
                }
            }
        }
    }

    #[test]
    fn comparison_costs_are_predicted_exactly_too() {
        // The Kogge-Stone borrow chain in `vm` and the `bit_decompose_muls`
        // formula in `cost` are two independent statements of the same
        // algorithm; this is what keeps them from drifting apart.
        for (src, ins) in [
            ("(* x (> x 0))", vec![("x", -7)]),
            ("(== x y)", vec![("x", 5), ("y", 6)]),
            ("(ash-right x 3)", vec![("x", 1000)]),
        ] {
            let rows = calibrate(src, &bindings(&ins), 3, &Deployment::lan(3), 32).unwrap();
            for r in rows {
                assert_eq!(
                    r.predicted_rounds, r.measured_rounds,
                    "{} {}",
                    src, r.protocol
                );
                assert_eq!(r.predicted_muls, r.measured_muls, "{} {}", src, r.protocol);
                assert_eq!(
                    r.predicted_online_bytes, r.measured_online_bytes,
                    "{} {}",
                    src, r.protocol
                );
            }
        }
    }

    #[test]
    fn comparisons_batch_into_one_decomposition_round() {
        // Four independent comparisons at the same dependence level must cost
        // the rounds of *one*, or the whole round-scheduling story is wrong.
        let one = calibrate(
            "(> a b)",
            &bindings(&[("a", 1), ("b", 2)]),
            3,
            &Deployment::lan(3),
            32,
        )
        .unwrap();
        let four = calibrate(
            "(+ (+ (> a b) (> c d)) (+ (> e f) (> g h)))",
            &bindings(&[
                ("a", 1),
                ("b", 2),
                ("c", 3),
                ("d", 4),
                ("e", 5),
                ("f", 6),
                ("g", 7),
                ("h", 8),
            ]),
            3,
            &Deployment::lan(3),
            32,
        )
        .unwrap();
        assert_eq!(
            one[0].measured_rounds, four[0].measured_rounds,
            "batched comparisons must not cost more rounds"
        );
        assert!(four[0].measured_muls > 3 * one[0].measured_muls);
    }

    #[test]
    fn unrunnable_programs_and_protocols_say_why() {
        // Secret division still needs restoring long division — phase P3.
        let cfg = LowerConfig::default();
        let prog = compile("(/ x y)", &cfg).unwrap();
        let sched = schedule(&prog);
        let err =
            run_protocol("dealer", 3, &prog, &sched, &bindings(&[("x", 6), ("y", 3)])).unwrap_err();
        assert!(err.contains("P3"), "{}", err);

        let prog = compile("(* x y)", &cfg).unwrap();
        let sched = schedule(&prog);
        let err =
            run_protocol("mascot", 3, &prog, &sched, &bindings(&[("x", 1), ("y", 2)])).unwrap_err();
        assert!(err.contains("not executable yet"), "{}", err);

        // Shamir has no honest majority with two parties.
        assert!(calibrate(
            "(* x y)",
            &bindings(&[("x", 1), ("y", 2)]),
            2,
            &Deployment::lan(2),
            32
        )
        .unwrap()
        .iter()
        .all(|r| r.protocol == "dealer"));
    }

    #[test]
    fn bench_suite_cases_all_compile_to_bytecode() {
        for case in crate::bench::SUITE {
            let src = case.mpc_expr.unwrap_or(case.expr);
            let prog = compile(src, &LowerConfig::default())
                .unwrap_or_else(|e| panic!("case {} failed to lower: {}", case.name, e));
            let sched = schedule(&prog);
            assert!(
                sched.round_count() > 0,
                "case {} has no interactive rounds",
                case.name
            );
        }
    }
}
