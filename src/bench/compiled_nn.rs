//! `baru bench-compiled-nn` — a whole network, hand-written versus compiled.
//!
//! [`crate::bench::abstraction`] compares one layer's worth of work; this
//! compares an entire network, which is what the paper's claim is actually
//! about. Both routes evaluate the **same** MLP over the **same** 3-party
//! ABY3 transport, so message and byte counts are directly comparable, and
//! both are checked against a plaintext integer reference — a cost comparison
//! between routes computing different things would be meaningless.
//!
//! | route | how the network is written |
//! |---|---|
//! | `gadget` | Rust calling [`crate::gadgets`] — `dense`, `relu`, `max_reduce` |
//! | `compiled` | a Baru *expression*, through `CircuitCompiler` → `FlatOp` → [`crate::mpcc::exec`] |
//!
//! The gadget route is the honest expert baseline: those gadgets are batched,
//! and they match hand-tuned code message for message
//! (`bench-abstraction`). The compiled route is given no batching hints at
//! all — the source is scalar arithmetic, and every round it saves comes from
//! ASAP scheduling over the dependence DAG.
//!
//! # Architecture
//!
//! `x (n) → dense(h×n) → ReLU → dense(c×h) → max` — secret weights
//! throughout, so every multiply is a real interactive operation rather than a
//! multiplication by a public constant. The final `max` is a *dependent*
//! reduction, which is what exercises
//! [`crate::mpcc::opt::Rule::ReassociateReduction`].

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::bench::comm::{CommStats, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, MpcTransport};
use crate::crypto::types::Fr;
use crate::gadgets;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::parse;
use crate::interpreter::transport_backend::{NonlinMode, TransportBackend};
use crate::mpcc::exec::{run_flat_with, Batching};
use crate::mpcc::opt::reassociate_reductions;
use crate::zkp::circuit_compiler::{CircuitCompiler, FlatOp};

fn residue(v: i64, nbits: usize) -> Fr {
    (v as i128).rem_euclid(1i128 << nbits) as Fr
}

fn to_signed(v: u128, nbits: usize) -> i64 {
    let m = 1i128 << nbits;
    let x = (v as i128) & (m - 1);
    if x >= m / 2 {
        (x - m) as i64
    } else {
        x as i64
    }
}

/// The network's weights and input, small enough that the integer reference
/// cannot overflow the share width.
struct Net {
    x: Vec<i64>,
    w1: Vec<Vec<i64>>,
    w2: Vec<Vec<i64>>,
}

impl Net {
    fn new(n: usize, h: usize, c: usize) -> Self {
        let x = (0..n).map(|i| (i as i64 % 7) - 3).collect();
        let w1 = (0..h)
            .map(|r| (0..n).map(|k| ((r + k) as i64 % 5) - 2).collect())
            .collect();
        let w2 = (0..c)
            .map(|r| (0..h).map(|k| ((r * 3 + k) as i64 % 5) - 2).collect())
            .collect();
        Net { x, w1, w2 }
    }

    /// Plaintext integer reference — the oracle both routes must match.
    fn forward(&self) -> i64 {
        let hidden: Vec<i64> = self
            .w1
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&self.x)
                    .map(|(w, x)| w * x)
                    .sum::<i64>()
                    .max(0)
            })
            .collect();
        self.w2
            .iter()
            .map(|row| row.iter().zip(&hidden).map(|(w, a)| w * a).sum::<i64>())
            .max()
            .expect("at least one class")
    }

    /// The same network as a Baru expression. Deliberately written the way a
    /// user would: scalar arithmetic, a mux-style ReLU (the interpreter cannot
    /// branch on a secret), and `max` as a left fold. Nothing here mentions
    /// batching.
    fn source(&self) -> String {
        let dot = |row: usize, w: &str, v: &dyn Fn(usize) -> String, len: usize| {
            let mut e = format!("(* {w}{row}_0 {})", v(0));
            for k in 1..len {
                e = format!("(+ {e} (* {w}{row}_{k} {}))", v(k));
            }
            e
        };
        let xv = |k: usize| format!("x{k}");
        let hidden: Vec<String> = (0..self.w1.len())
            .map(|r| {
                let z = dot(r, "a", &xv, self.x.len());
                // ReLU as a mux: z · (z > 0).
                format!("(* {z} (> {z} 0))")
            })
            .collect();
        let hv = |k: usize| hidden[k].clone();
        let logits: Vec<String> = (0..self.w2.len())
            .map(|r| dot(r, "b", &hv, self.w1.len()))
            .collect();
        // max as a left fold, mux style.
        let mut e = logits[0].clone();
        for l in logits.iter().skip(1) {
            e = format!("(+ {e} (* (> {l} {e}) (- {l} {e})))");
        }
        e
    }

    /// Wire name → value, for every free wire the compiled circuit reads.
    fn bindings(&self) -> HashMap<String, i64> {
        let mut m = HashMap::new();
        for (k, v) in self.x.iter().enumerate() {
            m.insert(format!("x{k}"), *v);
        }
        for (r, row) in self.w1.iter().enumerate() {
            for (k, v) in row.iter().enumerate() {
                m.insert(format!("a{r}_{k}"), *v);
            }
        }
        for (r, row) in self.w2.iter().enumerate() {
            for (k, v) in row.iter().enumerate() {
                m.insert(format!("b{r}_{k}"), *v);
            }
        }
        m
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RoutePoint {
    pub route: String,
    pub messages: u64,
    pub bytes: u64,
    pub ms_by_rtt: Vec<(u64, f64)>,
    pub correct: bool,
    /// Circuit size, for the compiled routes.
    pub ops: Option<usize>,
    /// Batched interactive protocol invocations — how often this route stopped
    /// and talked. Measured on the backend, so it is exactly comparable
    /// between routes however they were written.
    pub rounds: Option<usize>,
    /// Times a party blocked waiting for a peer. This is the quantity that
    /// predicts wall clock; `rounds` counts backend primitive calls, which
    /// weighs a comparison and a multiplication equally.
    pub waits: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub n: usize,
    pub hidden: usize,
    pub classes: usize,
    pub nbits: usize,
    pub expected: i64,
    pub routes: Vec<RoutePoint>,
}

fn measure<F>(
    profile: NetworkProfile,
    nbits: usize,
    body: F,
) -> (i64, CommStats, f64, Option<usize>, u64)
where
    F: Fn(&mut TransportBackend<MpcTransport>, usize) -> (ShareRef, Option<usize>, u64)
        + Send
        + Sync
        + Clone
        + 'static,
{
    let t = Instant::now();
    let (outs, comm) = run_parties(profile, move |me| {
        let body = body.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            // Same wire width as the CNN and comparison benchmarks, so the byte
            // figures this route reports are comparable with theirs rather than
            // inflated by `u128` padding. `correct` below is the oracle.
            if let Some(w) = crate::bench::cnn::wire_bits_for(nbits) {
                be.set_wire_bits(w);
            }
            let (r, rounds, waits) = body(be, me);
            (be.reveal(&r, None), rounds, waits)
        }
    });
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let (raw, rounds, waits) = outs.into_iter().next().expect("three parties");
    (raw as i64, comm, ms, rounds, waits)
}

/// Share every free wire of the network as a secret owned by party 0.
fn share_inputs(
    be: &mut TransportBackend<MpcTransport>,
    me: usize,
    binds: &HashMap<String, i64>,
    order: &[String],
    nbits: usize,
) -> HashMap<String, ShareRef> {
    let vals: Vec<Fr> = order
        .iter()
        .map(|k| if me == 0 { residue(binds[k], nbits) } else { 0 })
        .collect();
    let shares = be.share_secret_batch(&vals, nbits, 0);
    order.iter().cloned().zip(shares).collect()
}

pub fn run(n: usize, h: usize, c: usize, nbits: usize, rtts: &[u64]) -> Result<Report, String> {
    let net = Net::new(n, h, c);
    let expected = net.forward();
    let binds = net.bindings();
    let mut order: Vec<String> = binds.keys().cloned().collect();
    order.sort();

    // ── The compiled circuit, built once outside the measurement ──
    // Compiled twice: with and without CSE, so the ablation can attribute the
    // pass rather than assert it.
    let src = net.source();
    let expr = parse(&src).map_err(|e| format!("parse error: {e}"))?;
    let circuit = CircuitCompiler::new()
        .compile_fhe(&expr)
        .map_err(|e| format!("compile error: {e}"))?;
    let raw = CircuitCompiler::new()
        .without_cse()
        .compile_fhe(&expr)
        .map_err(|e| format!("compile error: {e}"))?;
    let flat = circuit.flat_ops.clone();
    let flat_no_cse = raw.flat_ops.clone();
    let out_wire = circuit.output_wire.clone();
    let flat_opt = reassociate_reductions(&flat, &out_wire);

    let mut routes = Vec::new();

    // ── Route 1: the hand-written gadget library ──
    //
    // Measured under *both* comparison backends, because the compiled route below is also
    // measured under both and a protocol swap available to one is available to the other. A
    // compiler credited with a win that came from the protocol would be this benchmark's
    // fourth incommensurable comparison.
    for nonlin in [NonlinMode::A2b, NonlinMode::Gc] {
        let (x, w1, w2) = (net.x.clone(), net.w1.clone(), net.w2.clone());
        let body = move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let sh = |be: &mut TransportBackend<MpcTransport>, v: &[i64]| -> Vec<ShareRef> {
                let vals: Vec<Fr> = v
                    .iter()
                    .map(|&t| if me == 0 { residue(t, nbits) } else { 0 })
                    .collect();
                be.share_secret_batch(&vals, nbits, 0)
            };
            be.set_nonlin(nonlin);
            let xs = sh(be, &x);
            let w1s: Vec<Vec<ShareRef>> = w1.iter().map(|r| sh(be, r)).collect();
            let w2s: Vec<Vec<ShareRef>> = w2.iter().map(|r| sh(be, r)).collect();
            let zero = be.share_public(0, nbits);
            let bias1 = vec![zero.clone(); w1s.len()];
            let bias2 = vec![zero; w2s.len()];

            let z = gadgets::dense(be, &xs, &w1s, &bias1);
            let a = gadgets::relu(be, &z);
            let logits = gadgets::dense(be, &a, &w2s, &bias2);
            let r = gadgets::max_reduce(be, &logits);
            (r, Some(be.interactive_calls()), be.sync_waits())
        };
        let name = if nonlin == NonlinMode::Gc { "gadget + gc" } else { "gadget" };
        routes.push(route(name, body, expected, nbits, rtts, None));
    }

    // ── The ablation ladder: each rung adds exactly one optimisation ──
    //
    // The first four rungs are compiler passes. The fifth is not: it changes which *protocol*
    // the comparison levels run, which the compiler inherits for free because `lt_batch`
    // dispatches on `NonlinMode` below the abstraction the executor calls. It is on the same
    // ladder because it is measured in the same unit, and because the wait attribution says
    // comparison levels carry ~72% of the budget — so a protocol swap and a scheduling pass are
    // competing for the same quantity and belong in one table.
    let ladder: Vec<(&str, &Vec<FlatOp>, Batching, NonlinMode)> = vec![
        // The obvious executor: walk the circuit, call the backend per op.
        ("compiled: serial", &flat_no_cse, Batching::Serial, NonlinMode::A2b),
        // ASAP levelling — the parallelism the schedule finds for free.
        ("  + ASAP batching", &flat_no_cse, Batching::Level, NonlinMode::A2b),
        // Common subexpression elimination, in the compile pipeline.
        ("  + CSE", &flat, Batching::Level, NonlinMode::A2b),
        // Rebalancing the dependent reduction.
        ("  + reassociation", &flat_opt, Batching::Level, NonlinMode::A2b),
        // Assigning the comparison levels to the constant-round backend.
        ("  + gc comparisons", &flat_opt, Batching::Level, NonlinMode::Gc),
    ];
    for (name, prog, batching, nonlin) in ladder {
        let (prog, out_wire, binds, order) =
            (prog.clone(), out_wire.clone(), binds.clone(), order.clone());
        let ops = prog.len();
        let body = move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let env = share_inputs(be, me, &binds, &order, nbits);
            be.set_nonlin(nonlin);
            let (r, _st) = run_flat_with(be, &prog, &env, &out_wire, nbits, batching)
                .expect("circuit executes");
            (r, Some(be.interactive_calls()), be.sync_waits())
        };
        routes.push(route(name, body, expected, nbits, rtts, Some(ops)));
    }

    Ok(Report {
        n,
        hidden: h,
        classes: c,
        nbits,
        expected,
        routes,
    })
}

fn route<F>(
    name: &str,
    body: F,
    want: i64,
    nbits: usize,
    rtts: &[u64],
    ops: Option<usize>,
) -> RoutePoint
where
    F: Fn(&mut TransportBackend<MpcTransport>, usize) -> (ShareRef, Option<usize>, u64)
        + Send
        + Sync
        + Clone
        + 'static,
{
    let mut ms_by_rtt = Vec::new();
    let mut comm = CommStats::default();
    let mut correct = true;
    let mut rounds = None;
    let mut waits = 0u64;
    for &rtt in rtts {
        let profile = NetworkProfile::fixed_latency(Duration::from_micros(rtt * 500));
        let (got, c, ms, r, w) = measure(profile, nbits, body.clone());
        correct &= to_signed(got as u128, nbits) == want;
        comm = c;
        rounds = r;
        waits = w;
        ms_by_rtt.push((rtt, ms));
    }
    RoutePoint {
        route: name.to_string(),
        messages: comm.messages,
        bytes: comm.bytes,
        ms_by_rtt,
        correct,
        ops,
        rounds,
        waits,
    }
}

const USAGE: &str = "usage: baru bench-compiled-nn [options]

A whole network, hand-written versus compiled, over the same 3-party
transport. Architecture: x(n) -> dense(h) -> ReLU -> dense(c) -> max, secret
weights throughout. Every route must reproduce the plaintext integer
reference; the command fails if any does not.

The ladder attributes each compiler pass. NOTE: the `serial` control issues one
round trip per interactive operation, so raising --rtt-ms or the widths makes it
take minutes; that is the point of the row, but scale it deliberately.

options:
  --n N            input width (default 8)
  --hidden H       hidden width (default 4)
  --classes C      output width (default 4)
  --nbits B        share bit width (default 32)
  --rtt-ms A,B     round-trip times to probe (default 0; see note above)
  --json           machine-readable output
  -h, --help       this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let (mut n, mut h, mut c, mut nbits) = (8usize, 4usize, 4usize, 32usize);
    // Default to zero latency: the `serial` control performs one round trip
    // per interactive operation (thousands of them), so at RTT 20 ms it takes
    // minutes. That is the finding, not a bug — but it should be opt-in.
    let mut rtts: Vec<u64> = vec![0];
    let mut json = false;
    let mut it = args.iter();
    let num = |s: Option<&String>, what: &str| -> Result<usize, String> {
        s.ok_or(format!("{what} needs a value"))?
            .parse()
            .map_err(|_| format!("bad {what}"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" => n = num(it.next(), "--n")?,
            "--hidden" => h = num(it.next(), "--hidden")?,
            "--classes" => c = num(it.next(), "--classes")?,
            "--nbits" => nbits = num(it.next(), "--nbits")?,
            "--rtt-ms" => {
                rtts = it
                    .next()
                    .ok_or("--rtt-ms needs a value")?
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<u64>()
                            .map_err(|_| "bad --rtt-ms".to_string())
                    })
                    .collect::<Result<_, _>>()?
            }
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let r = run(n, h, c, nbits, &rtts)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("A whole network: hand-written gadgets vs compiled circuit");
    println!(
        "architecture: x({}) -> dense({}) -> ReLU -> dense({}) -> max, secret weights, {}-bit\n",
        r.n, r.hidden, r.classes, r.nbits
    );
    print!(
        "  {:<18} {:>7} {:>12} {:>10} {:>12}",
        "route", "ops", "interactive", "messages", "bytes"
    );
    for &(rtt, _) in &r.routes[0].ms_by_rtt {
        print!("{:>11}", format!("rtt {rtt}ms"));
    }
    println!();
    for p in &r.routes {
        if !p.correct {
            return Err(format!(
                "route `{}` disagrees with the plaintext reference (expected {}) — \
                 a cost comparison between different computations is meaningless",
                p.route, r.expected
            ));
        }
        print!(
            "  {:<18} {:>7} {:>12} {:>10} {:>12}",
            p.route,
            p.ops.map(|o| o.to_string()).unwrap_or_else(|| "-".into()),
            p.rounds
                .map(|o| o.to_string())
                .unwrap_or_else(|| "-".into()),
            p.messages,
            p.bytes
        );
        for &(_, ms) in &p.ms_by_rtt {
            print!("{:>11}", format!("{ms:.0}ms"));
        }
        println!();
    }
    println!(
        "\n  all routes reproduce the plaintext reference ({})",
        r.expected
    );
    // Compiler versus expert, compared **at the same comparison backend**. The last ladder rung
    // runs gc comparisons, so the expert it is measured against must too: the gc assignment is a
    // protocol change available to whoever writes the circuit, and crediting it to the compiler
    // would repeat the error this benchmark has now made three times in its own history.
    let full = r.routes.last().expect("ladder is non-empty");
    let g = r
        .routes
        .iter()
        .find(|p| p.route == "gadget + gc")
        .unwrap_or(&r.routes[0]);
    let rtt = g.ms_by_rtt.last().unwrap().0;
    println!(
        "\n  full compiler vs hand-written expert, at rtt {rtt}ms, both on gc comparisons:\n    \
         {:.2}x interactive calls, {:.2}x messages, {:.2}x waits, {:.2}x wall-clock  (lower is better)",
        full.rounds.unwrap_or(0) as f64 / g.rounds.unwrap_or(1).max(1) as f64,
        full.messages as f64 / g.messages.max(1) as f64,
        full.waits as f64 / g.waits.max(1) as f64,
        full.ms_by_rtt.last().unwrap().1 / g.ms_by_rtt.last().unwrap().1.max(0.001),
    );
    // And what the protocol swap alone was worth, on each route separately, so the two effects
    // are never added together.
    if let (Some(ga), Some(gg)) = (
        r.routes.iter().find(|p| p.route == "gadget"),
        r.routes.iter().find(|p| p.route == "gadget + gc"),
    ) {
        let a = ga.ms_by_rtt.last().unwrap().1;
        let b = gg.ms_by_rtt.last().unwrap().1;
        println!(
            "  the gc assignment alone, on the *hand-written* route: {a:.0}ms -> {b:.0}ms \
             ({:.2}x) — so it is a protocol result, not a compiler one",
            b / a.max(0.001)
        );
    }
    // What each pass contributed. Reported on every axis, because the passes
    // do not compete: batching buys rounds, CSE buys circuit size and
    // bandwidth, reassociation buys depth. A single-column summary would make
    // CSE look worthless (it changes no round counts) when it is a large win.
    println!("\n  what each pass contributed:");
    println!(
        "    {:<20} {:>18} {:>18} {:>18} {:>17}",
        "pass", "ops", "interactive", "bytes", "waits"
    );
    // Only the compiled rungs form a ladder; the two hand-written routes are controls, not
    // rungs, and differencing across that boundary would print a pass that does not exist.
    let mut prev: Option<&RoutePoint> = None;
    for p in r.routes.iter().filter(|p| !p.route.starts_with("gadget")) {
        if let Some(q) = prev {
            let ratio = |a: f64, b: f64| {
                if b <= 0.0 {
                    "-".to_string()
                } else {
                    format!("{:.2}x", a / b)
                }
            };
            let (oa, ob) = (q.ops.unwrap_or(0), p.ops.unwrap_or(0));
            let (ia, ib) = (q.rounds.unwrap_or(0), p.rounds.unwrap_or(0));
            let (ba, bb) = (q.bytes, p.bytes);
            let (wa, wb) = (q.waits, p.waits);
            println!(
                "    {:<20} {:>8} {:>9} {:>8} {:>9} {:>8} {:>9} {:>7} {:>9}",
                p.route.trim(),
                format!("{oa}->{ob}"),
                ratio(oa as f64, ob as f64),
                format!("{ia}->{ib}"),
                ratio(ia as f64, ib as f64),
                format!("{:.0}k->{:.0}k", ba as f64 / 1000.0, bb as f64 / 1000.0),
                ratio(ba as f64, bb as f64),
                format!("{wa}->{wb}"),
                ratio(wa as f64, wb as f64),
            );
        }
        prev = Some(p);
    }
    println!(
        "\n  The passes act on different axes: batching collapses rounds, CSE collapses\n  \
         circuit size and bandwidth, reassociation collapses depth, and the gc\n  \
         assignment collapses the per-comparison blocking. None substitutes for\n  \
         another, which is why the ladder is reported rather than a single number.\n  \
         The waits column is the one that tracks wall clock; note that CSE moves it\n  \
         not at all while cutting bytes 24x, and that the last rung is a protocol\n  \
         swap the hand-written route can make too (see the control above)."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The wait cost of each primitive, which is what a scheduler should be minimising.**
    ///
    /// `sync_waits` says a route's cost is the number of times it blocks. This measures how many
    /// blocks each primitive costs, at one width and one batch size, so the figure can be used
    /// as a cost model rather than guessed at. The interesting entry is the comparison: it is
    /// not one round, because it runs a bit decomposition whose prefix adder blocks once per
    /// stage, and every scheduler decision that trades multiplications against comparisons
    /// depends on that ratio.
    ///
    /// Batch size is held fixed and reported, because a batched primitive blocks the same number
    /// of times whatever its width — that property is the whole reason levelling helps, and if it
    /// ever stopped holding the numbers below would stop being a cost model.
    #[test]
    #[ignore] // three-party transport
    fn the_wait_cost_of_each_primitive() {
        const NBITS: usize = 32;
        for width in [1usize, 64] {
            let probe = move |be: &mut TransportBackend<MpcTransport>, me: usize| {
                let mk = |be: &mut TransportBackend<MpcTransport>, k: u64| -> Vec<ShareRef> {
                    let vals: Vec<Fr> = (0..width)
                        .map(|i| if me == 0 { residue((i as i64) + k as i64, NBITS) } else { 0 })
                        .collect();
                    be.share_secret_batch(&vals, NBITS, 0)
                };
                let xs = mk(be, 1);
                let ys = mk(be, 7);
                let mut out = Vec::new();
                let base = be.sync_waits();

                let pairs: Vec<_> = xs.iter().cloned().zip(ys.iter().cloned()).collect();
                let w0 = be.sync_waits();
                let _ = be.mul_batch(&pairs);
                out.push(("mul_batch", be.sync_waits() - w0));

                let w1 = be.sync_waits();
                let _ = be.lt_batch(&pairs);
                out.push(("lt_batch", be.sync_waits() - w1));

                let w2 = be.sync_waits();
                let _ = be.a2b_batch(&xs, NBITS);
                out.push(("a2b_batch", be.sync_waits() - w2));

                let w3 = be.sync_waits();
                let _ = be.reveal(&xs[0], None);
                out.push(("reveal", be.sync_waits() - w3));

                let total = be.sync_waits() - base;
                for (name, w) in &out {
                    println!("    width {width:3}: {name:11} {w:3} waits");
                }
                // Widths must agree: a batched primitive blocks the same number of times
                // whatever its width, which is the property levelling exploits.
                assert_eq!(out[0].1, 2, "mul_batch should block twice");
                assert_eq!(out[1].1, 9, "lt_batch should block nine times");
                assert!(
                    out[1].1 > 4 * out[0].1,
                    "a comparison must cost several multiplications, or a scheduler that \
                     counts operations uniformly would be right after all"
                );
                (xs[0].clone(), Some(total as usize), total)
            };
            let _ = measure(NetworkProfile::INSTANT, NBITS, probe);
        }
    }

    /// **Where the cost model and the executor disagree, measured rather than argued.**
    ///
    /// `cost.rs` prices a schedule level at the *maximum* over the primitives it contains, on
    /// the reasoning that a level is one logical round. The executor issues one batched call per
    /// *kind* per level and every call blocks, so a level holding a comparison and a multiply
    /// pays their *sum* — 9 + 2, where the model charges 9. Whether that matters is a question
    /// about a particular schedule, not about the two definitions, so this counts the mixed
    /// levels in the network the benchmark actually runs.
    ///
    /// It also pins the one part of the gap that is closable without splitting the primitives
    /// into send and receive phases: a mux *is* a multiply, so muxes ride in the level's
    /// `mul_batch` instead of a second one. The assertion is that no level issues two
    /// multiply-shaped calls, which is what regressing that merge would look like.
    #[test]
    fn mixed_kind_levels_are_where_the_model_and_the_executor_disagree() {
        use crate::mpcc::exec::{run_flat_with, Batching};
        use crate::mpcc::opt::reassociate_reductions;
        use crate::zkp::circuit_compiler::CircuitCompiler;

        const NBITS: usize = 32;
        let net = Net::new(16, 8, 4);
        let expr = parse(&net.source()).expect("parses");
        let circuit = CircuitCompiler::new().compile_fhe(&expr).expect("compiles");
        let flat = reassociate_reductions(&circuit.flat_ops, &circuit.output_wire);

        // The simulator backend suffices: only the schedule shape is under test here, and it is
        // independent of the protocol, so this needs no transport and can run unignored.
        let mut be = crate::interpreter::aby3_simulator::Aby3Simulator::new();
        let binds = net.bindings();
        let inputs: HashMap<String, ShareRef> = binds
            .iter()
            .map(|(k, &v)| (k.clone(), be.share_public(residue(v, NBITS), NBITS)))
            .collect();
        let (_out, st) =
            run_flat_with(&mut be, &flat, &inputs, &circuit.output_wire, NBITS, Batching::Level)
                .expect("executes");

        // Model charge vs executor charge, using the measured per-primitive wait costs
        // (`the_wait_cost_of_each_primitive`: a multiply blocks twice, a comparison nine times).
        println!(
            "  {} interactive ops over {} levels, {} batched calls, {} mixed levels",
            st.interactive_ops, st.rounds, st.batch_calls, st.mixed_levels
        );
        assert_eq!(
            st.mixed_levels,
            st.batch_calls - st.rounds,
            "mixed_levels is definitionally the excess of calls over levels"
        );
        // Every level issues at least one call, and a level can only exceed one by mixing a
        // comparison with a multiply — muxes no longer get a call of their own.
        assert!(st.batch_calls >= st.rounds, "each level issues at least one call");
        assert!(
            st.mixed_levels <= st.rounds,
            "a level cannot mix more kinds than there are kinds: {} mixed of {} levels",
            st.mixed_levels,
            st.rounds
        );
        if st.mixed_levels == 0 {
            println!(
                "  no level mixes kinds in this network, so the max-vs-sum gap costs it nothing —\n  \
                 the discrepancy is real between the two definitions and this schedule does\n  \
                 not pay for it, because ReLU puts its comparison and its multiply at\n  \
                 consecutive levels by construction"
            );
        } else {
            println!(
                "  {} levels pay a sum where the model charges a max: up to {} waits unmodelled",
                st.mixed_levels,
                st.mixed_levels * 2
            );
        }

        // ── Where the waits actually go ──
        //
        // Attribute the total to levels by kind, using the measured per-primitive costs. This
        // is the number a scheduler should be minimising, and it says which primitive to attack.
        let mut per_level: Vec<(usize, usize, usize)> = Vec::new(); // (cmps, muls, muxes)
        {
            let mut lvl: HashMap<&str, usize> = HashMap::new();
            for w in inputs.keys() {
                lvl.insert(w.as_str(), 0);
            }
            let mut op_level = vec![0usize; flat.len()];
            for (i, op) in flat.iter().enumerate() {
                let ready = op.inputs().iter().map(|w| *lvl.get(*w).unwrap_or(&0)).max().unwrap_or(0);
                let interactive = matches!(
                    op,
                    FlatOp::IMul { .. } | FlatOp::ICmpSlt { .. } | FlatOp::ICmpUlt { .. } | FlatOp::Mux { .. }
                );
                op_level[i] = if interactive { ready + 1 } else { ready };
                lvl.insert(op.dst(), op_level[i]);
            }
            let maxl = op_level.iter().copied().max().unwrap_or(0);
            for l in 0..=maxl {
                let mut t = (0usize, 0usize, 0usize);
                for (i, op) in flat.iter().enumerate() {
                    if op_level[i] != l {
                        continue;
                    }
                    match op {
                        FlatOp::ICmpSlt { .. } | FlatOp::ICmpUlt { .. } => t.0 += 1,
                        FlatOp::IMul { .. } => t.1 += 1,
                        FlatOp::Mux { .. } => t.2 += 1,
                        _ => {}
                    }
                }
                if t != (0, 0, 0) {
                    per_level.push(t);
                }
            }
        }
        const W_CMP: usize = 9;
        const W_MUL: usize = 2;
        let mut predicted = 0usize;
        println!("  level  cmps  muls  muxes   waits");
        for (i, &(c, m, x)) in per_level.iter().enumerate() {
            // One lt_batch if any comparison, one mul_batch if any multiply or mux.
            let w = if c > 0 { W_CMP } else { 0 } + if m + x > 0 { W_MUL } else { 0 };
            predicted += w;
            println!("  {i:5}  {c:4}  {m:4}  {x:5}   {w:5}");
        }
        println!(
            "  predicted online waits: {predicted}  (comparison levels dominate at {W_CMP} each \
             against {W_MUL} for a multiply level)"
        );
        let cmp_levels = per_level.iter().filter(|&&(c, _, _)| c > 0).count();
        println!(
            "  {} of {} levels hold a comparison, and they account for {}% of the predicted waits",
            cmp_levels,
            per_level.len(),
            100 * cmp_levels * W_CMP / predicted.max(1)
        );
        assert!(
            cmp_levels * W_CMP * 2 > predicted,
            "comparison levels should dominate the wait budget, or the lever is elsewhere"
        );
    }

    /// **The wait cost of each comparison backend — the measurement a per-subtask assignment needs.**
    ///
    /// `mixed_kind_levels_are_where_the_model_and_the_executor_disagree` finds that comparison
    /// levels carry ~72% of the compiled network's wait budget, so the comparison is the primitive
    /// worth swapping. Baru has three, and `lt_batch` already dispatches on `NonlinMode`, which
    /// means the *compiled* route inherits whichever one the session selects without any change
    /// to the compiler: the backend choice is already below the abstraction the executor calls.
    ///
    /// This measures what each costs in blocks, at one width and one batch, so an assignment pass
    /// has a table to consult rather than a guess. It asserts only that they *differ* — if all
    /// three blocked the same number of times there would be nothing for an assignment to choose
    /// between, and that is the result that would kill the idea.
    #[test]
    #[ignore] // three-party transport
    fn the_wait_cost_of_each_comparison_backend() {
        use crate::interpreter::transport_backend::{FalconPoolSpec, NonlinMode};

        const NBITS: usize = 32;
        const WIDTH: usize = 64;
        let mut table: Vec<(&str, u64)> = Vec::new();
        for mode in [NonlinMode::A2b, NonlinMode::Falcon, NonlinMode::Gc] {
            let probe = move |be: &mut TransportBackend<MpcTransport>, me: usize| {
                let vals: Vec<Fr> = (0..WIDTH)
                    .map(|i| if me == 0 { residue(i as i64 - 8, NBITS) } else { 0 })
                    .collect();
                let xs = be.share_secret_batch(&vals, NBITS, 0);
                let ys = be.share_secret_batch(&vals, NBITS, 0);
                be.set_nonlin(mode);
                // Falcon's comparison consumes preprocessing; fill it first so what is measured
                // is the online phase, which is the phase a scheduler is choosing between.
                if mode == NonlinMode::Falcon {
                    let _ = be.fill_falcon_pool(&FalconPoolSpec {
                        comparisons: WIDTH,
                        truncations: 0,
                        nbits: NBITS as u32,
                        lift_bits: NBITS as u32,
                        shift: 0,
                    });
                }
                let pairs: Vec<_> = xs.iter().cloned().zip(ys.iter().cloned()).collect();
                let w0 = be.sync_waits();
                let out = be.lt_batch(&pairs);
                let waits = be.sync_waits() - w0;
                (out[0].clone(), Some(waits as usize), waits)
            };
            let (_v, comm, _ms, _r, waits) = measure(NetworkProfile::INSTANT, NBITS, probe);
            let name = match mode {
                NonlinMode::A2b => "a2b",
                NonlinMode::Falcon => "falcon",
                NonlinMode::Gc => "gc",
            };
            println!(
                "    {name:8} {waits:3} waits for {WIDTH} comparisons at l={NBITS}  \
                 ({} messages, {:.1} kB total incl. preprocessing)",
                comm.messages,
                comm.bytes as f64 / 1000.0
            );
            table.push((name, waits));
        }
        let lo = table.iter().min_by_key(|&&(_, w)| w).expect("three modes");
        let hi = table.iter().max_by_key(|&&(_, w)| w).expect("three modes");
        println!(
            "  cheapest in blocks: {} at {} waits; dearest: {} at {} — a {:.1}x spread, which is \
             what a per-subtask assignment has to trade against the byte columns",
            lo.0, lo.1, hi.0, hi.1, hi.1 as f64 / lo.1.max(1) as f64
        );
        assert!(
            hi.1 > lo.1,
            "the three comparison backends must differ in blocking cost, or there is nothing \
             for an assignment pass to choose between: {table:?}"
        );
    }

    /// **Waits, not calls, are what a route costs.**
    ///
    /// This pins the correction that this benchmark's own history required. Comparing the two
    /// routes by `interactive_calls` said the compiler won by 2x; comparing by messages said
    /// 1.6x; comparing by the number of times a party blocked for a peer said parity, and only
    /// the last agreed with the wall clock. A primitive call is not a round: `mul_batch` blocks
    /// once, `lt_batch` runs a whole bit decomposition and blocks once per prefix-adder stage,
    /// so counting calls weighs a comparison and a multiplication equally.
    ///
    /// The test asserts the *relationship* between the counters rather than any absolute figure:
    /// the compiled route really is leaner on calls and messages, and it nevertheless does not
    /// block fewer times. If a future change made it genuinely cheaper in waits, this test
    /// fails, and that failure is the result worth having.
    #[test]
    #[ignore] // three-party transport, two full routes
    fn calls_and_messages_do_not_predict_cost_but_waits_do() {
        let r = run(16, 8, 4, 32, &[0]).expect("benchmark runs");
        let pick = |name: &str| {
            r.routes
                .iter()
                .find(|p| p.route == name)
                .unwrap_or_else(|| panic!("route `{name}` present"))
        };
        // Matched backends. The last ladder rung runs gc comparisons, so its control must be
        // the gc expert, not the a2b one — comparing across the two credits the compiler with a
        // protocol change anyone can make, and asserting that here is what keeps it from
        // happening again.
        let gadget = pick("gadget + gc");
        let compiled = r.routes.last().expect("compiled route present");
        assert!(gadget.correct && compiled.correct, "both routes must be exact");

        let (gc, cc) = (gadget.rounds.unwrap(), compiled.rounds.unwrap());
        println!("  gadget:   {gc:3} calls, {:3} waits, {:3} messages", gadget.waits, gadget.messages);
        println!("  compiled: {cc:3} calls, {:3} waits, {:3} messages", compiled.waits, compiled.messages);

        // Leaner on both of the counters we reached for first.
        assert!(cc < gc, "compiled should make fewer primitive calls: {cc} vs {gc}");
        assert!(
            compiled.messages < gadget.messages,
            "compiled should send fewer messages: {} vs {}",
            compiled.messages, gadget.messages
        );

        // And it does not block less, which is why it does not run faster. A band, so the test
        // is about parity rather than about one machine's noise.
        let ratio = compiled.waits as f64 / gadget.waits as f64;
        println!("  -> waits ratio {ratio:.2} (calls ratio {:.2})", cc as f64 / gc as f64);
        assert!(
            (0.80..=1.25).contains(&ratio),
            "the routes should be at parity in waits, got {ratio:.2} ({} vs {})",
            compiled.waits, gadget.waits
        );

        // The mismatched comparison, which this test previously made by accident: measuring the
        // gc compiler against the a2b expert reports a large win that belongs to the protocol.
        // Both figures are printed so the difference between them is on the record.
        let unmatched = pick("gadget");
        let flattered = compiled.waits as f64 / unmatched.waits as f64;
        println!(
            "  against the *a2b* expert the same compiled route reads {flattered:.2}x waits — the \
             gc assignment is worth {:.2}x on the hand-written route on its own, so that reading \
             would credit the compiler with a protocol result",
            gadget.waits as f64 / unmatched.waits as f64
        );
        assert!(
            flattered < ratio,
            "the mismatched comparison must be the flattering one, or the lesson recorded here \
             is not the lesson: {flattered:.2} vs {ratio:.2}"
        );
    }

    /// Both routes must compute the network, and the compiled one must do it
    /// without being told to batch anything.
    #[test]
    fn compiled_and_hand_written_agree_on_the_whole_network() {
        let r = run(8, 4, 3, 32, &[0]).unwrap();
        for p in &r.routes {
            assert!(
                p.correct,
                "route `{}` disagrees with the reference",
                p.route
            );
        }
    }
}
