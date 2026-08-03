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
use crate::interpreter::transport_backend::TransportBackend;
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
) -> (i64, CommStats, f64, Option<usize>)
where
    F: Fn(&mut TransportBackend<MpcTransport>, usize) -> (ShareRef, Option<usize>)
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
            let (r, rounds) = body(be, me);
            (be.reveal(&r, None), rounds)
        }
    });
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let (raw, rounds) = outs.into_iter().next().expect("three parties");
    (raw as i64, comm, ms, rounds)
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
    {
        let (x, w1, w2) = (net.x.clone(), net.w1.clone(), net.w2.clone());
        let body = move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let sh = |be: &mut TransportBackend<MpcTransport>, v: &[i64]| -> Vec<ShareRef> {
                let vals: Vec<Fr> = v
                    .iter()
                    .map(|&t| if me == 0 { residue(t, nbits) } else { 0 })
                    .collect();
                be.share_secret_batch(&vals, nbits, 0)
            };
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
            (r, Some(be.interactive_calls()))
        };
        routes.push(route("gadget", body, expected, nbits, rtts, None));
    }

    // ── The ablation ladder: each rung adds exactly one compiler pass ──
    let ladder: Vec<(&str, &Vec<FlatOp>, Batching)> = vec![
        // The obvious executor: walk the circuit, call the backend per op.
        ("compiled: serial", &flat_no_cse, Batching::Serial),
        // ASAP levelling — the parallelism the schedule finds for free.
        ("  + ASAP batching", &flat_no_cse, Batching::Level),
        // Common subexpression elimination, in the compile pipeline.
        ("  + CSE", &flat, Batching::Level),
        // Rebalancing the dependent reduction.
        ("  + reassociation", &flat_opt, Batching::Level),
    ];
    for (name, prog, batching) in ladder {
        let (prog, out_wire, binds, order) =
            (prog.clone(), out_wire.clone(), binds.clone(), order.clone());
        let ops = prog.len();
        let body = move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let env = share_inputs(be, me, &binds, &order, nbits);
            let (r, _st) = run_flat_with(be, &prog, &env, &out_wire, nbits, batching)
                .expect("circuit executes");
            (r, Some(be.interactive_calls()))
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
    F: Fn(&mut TransportBackend<MpcTransport>, usize) -> (ShareRef, Option<usize>)
        + Send
        + Sync
        + Clone
        + 'static,
{
    let mut ms_by_rtt = Vec::new();
    let mut comm = CommStats::default();
    let mut correct = true;
    let mut rounds = None;
    for &rtt in rtts {
        let profile = NetworkProfile::fixed_latency(Duration::from_micros(rtt * 500));
        let (got, c, ms, r) = measure(profile, nbits, body.clone());
        correct &= to_signed(got as u128, nbits) == want;
        comm = c;
        rounds = r;
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
    let g = &r.routes[0];
    let full = r.routes.last().expect("ladder is non-empty");
    let rtt = g.ms_by_rtt.last().unwrap().0;
    println!(
        "\n  full compiler vs hand-written expert, at rtt {rtt}ms:\n    \
         {:.2}x interactive calls, {:.2}x messages, {:.2}x wall-clock  (lower is better)",
        full.rounds.unwrap_or(0) as f64 / g.rounds.unwrap_or(1).max(1) as f64,
        full.messages as f64 / g.messages.max(1) as f64,
        full.ms_by_rtt.last().unwrap().1 / g.ms_by_rtt.last().unwrap().1.max(0.001),
    );
    // What each pass contributed. Reported on every axis, because the passes
    // do not compete: batching buys rounds, CSE buys circuit size and
    // bandwidth, reassociation buys depth. A single-column summary would make
    // CSE look worthless (it changes no round counts) when it is a large win.
    println!("\n  what each pass contributed:");
    println!(
        "    {:<20} {:>18} {:>18} {:>18}",
        "pass", "ops", "interactive", "bytes"
    );
    let mut prev: Option<&RoutePoint> = None;
    for p in r.routes.iter().skip(1) {
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
            println!(
                "    {:<20} {:>8} {:>9} {:>8} {:>9} {:>8} {:>9}",
                p.route.trim(),
                format!("{oa}->{ob}"),
                ratio(oa as f64, ob as f64),
                format!("{ia}->{ib}"),
                ratio(ia as f64, ib as f64),
                format!("{:.0}k->{:.0}k", ba as f64 / 1000.0, bb as f64 / 1000.0),
                ratio(ba as f64, bb as f64),
            );
        }
        prev = Some(p);
    }
    println!(
        "\n  The passes act on different axes: batching collapses rounds, CSE collapses\n  \
         circuit size and bandwidth, reassociation collapses depth. None substitutes\n  \
         for another, which is why the ladder is reported rather than a single number."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
