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
//!                                                       │
//!                                    protocols::REGISTRY┴──cost──► ranked comparison
//! ```
//!
//! See `docs/mpc-compiler-roadmap.md` for the phasing and the target protocol
//! matrix.  Phase P0 (this module) implements the framework and validates it
//! against the two backends Baru already has; each later phase adds real
//! protocol implementations that *validate* one row of the registry.
//!
//! ## CLI
//!
//! ```text
//! baru mpcc protocols                         # the protocol matrix
//! baru mpcc dump     --expr '(* x y)'         # bytecode listing
//! baru mpcc analyse  --expr '(* x y)'         # + instruction census and round schedule
//! baru mpcc compare  --expr '(* x y)' [opts]  # ranked cost estimate per protocol
//! ```

pub mod cost;
pub mod ir;
pub mod lower;
pub mod protocols;
pub mod schedule;

use serde::Serialize;

use crate::zkp::circuit_compiler::CircuitCompiler;

pub use cost::{estimate, CostEstimate, Deployment};
pub use ir::{Domain, Program};
pub use lower::{lower, LowerConfig};
pub use protocols::{find, ProtocolSpec, Security, REGISTRY};
pub use schedule::{schedule, Schedule};

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
    })
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
