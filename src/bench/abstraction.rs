//! `baru bench-abstraction` — what does Baru's abstraction cost?
//!
//! The pipeline benchmarks (`bench-cnn`, `bench-medmnist`) measure
//! hand-written Rust that calls the backend's *batched* entry points
//! directly. The composable API the paper's composability section is about
//! ([`crate::gadgets`]) and the Lisp front end on top of it are what a *user*
//! writes. This benchmark asks whether those cost the same thing.
//!
//! It runs one layer's worth of work three ways over the **same** transport —
//! real serialization, real message passing, real byte counting — so the only
//! difference is which API expressed it:
//!
//! | route | what it is |
//! |---|---|
//! | `naive` | the obvious per-element composition — what the gadgets used to be |
//! | `gadget` | [`crate::gadgets`] — the composable, backend-agnostic library |
//! | `batched` | the backend's batched primitives, as `bench::cnn` uses them |
//!
//! The `naive` route is kept deliberately. It is what a user writes before
//! discovering the batched primitives, it is what the gadget library itself
//! did until 2026-07-28, and keeping it here is the only way the size of that
//! defect stays reproducible after the fix. `gadget` should track `batched`
//! exactly; a test asserts it, so the library cannot silently regress to
//! `naive` under a well-meaning simplification.
//!
//! All three must produce identical values; the harness fails if they do not,
//! because a cost comparison between routes computing different things is
//! meaningless.
//!
//! # Why round *count* is the metric
//!
//! Wall-clock at zero latency understates the difference badly: an in-process
//! transport makes a round trip nearly free, so a round-serial gadget looks
//! fine until it is deployed. Message count is latency-independent and is what
//! the network model turns into time, so it is reported first and the
//! wall-clock is reported at several round-trip times to show the consequence.
//!
//! # Result (2026-07-28, n=64, 20-bit, dense 8x64)
//!
//! | workload | route | messages | wall-clock @ RTT 20 ms |
//! |---|---|---|---|
//! | relu | naive | 771 | 9415 ms |
//! | relu | gadget/batched | 141 | ~1050 ms |
//! | max-reduce | naive | 635 | 8437 ms |
//! | max-reduce | gadget/batched | 65 | ~860 ms |
//! | dense | naive | 549 | 6943 ms |
//! | dense | gadget/batched | 38 | ~150 ms |
//!
//! **Bytes are within 5% across routes** — the naive form moves the same data
//! in more, smaller round trips. That is the whole finding: the cost was never
//! the protocol, it was how the abstraction issued it. See
//! `docs/private-inference-performance.md`.

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::bench::comm::{CommStats, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, MpcTransport};
use crate::crypto::types::Fr;
use crate::gadgets;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

/// Two's-complement residue of a signed value at `nbits`.
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

/// One route's measured cost for one workload.
#[derive(Debug, Clone, Serialize)]
pub struct RoutePoint {
    pub route: String,
    pub messages: u64,
    pub bytes: u64,
    /// Wall-clock at each probed round-trip time, in milliseconds.
    pub ms_by_rtt: Vec<(u64, f64)>,
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkloadReport {
    pub workload: String,
    pub n: usize,
    pub nbits: usize,
    pub routes: Vec<RoutePoint>,
}

/// Share a vector of public values as secrets owned by party 0, batched so
/// the *setup* is not what the comparison measures.
fn share_all(
    be: &mut TransportBackend<MpcTransport>,
    me: usize,
    xs: &[i64],
    nbits: usize,
) -> Vec<ShareRef> {
    let vals: Vec<Fr> = xs
        .iter()
        .map(|&v| if me == 0 { residue(v, nbits) } else { 0 })
        .collect();
    be.share_secret_batch(&vals, nbits, 0)
}

/// Run `body` under the 3-party transport at `profile`, returning party 0's
/// revealed output plus the communication it cost.
fn measure<F>(profile: NetworkProfile, body: F) -> (Vec<i64>, CommStats, f64)
where
    F: Fn(&mut TransportBackend<MpcTransport>, usize) -> Vec<ShareRef>
        + Send
        + Sync
        + Clone
        + 'static,
{
    let t = Instant::now();
    let (outs, comm) = run_parties(profile, move |me| {
        let body = body.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            let out = body(be, me);
            out.iter()
                .map(|r| be.reveal(r, None))
                .collect::<Vec<u128>>()
        }
    });
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let raw = outs.into_iter().next().expect("three parties");
    (raw.into_iter().map(|v| v as i64).collect(), comm, ms)
}

/// ReLU over `n` secret values, three ways.
fn relu_workload(n: usize, nbits: usize, rtts: &[u64]) -> WorkloadReport {
    let xs: Vec<i64> = (0..n).map(|i| (i as i64 % 17) - 8).collect();
    let want: Vec<i64> = gadgets::relu_ref(&xs);

    // The composable library.
    let gadget_body = {
        let xs = xs.clone();
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let s = share_all(be, me, &xs, nbits);
            gadgets::relu(be, &s)
        }
    };
    // The batched primitives, the way `bench::cnn` calls them: one batched
    // comparison for the layer, then one batched multiply.
    let batched_body = {
        let xs = xs.clone();
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let s = share_all(be, me, &xs, nbits);
            let zeros: Vec<ShareRef> = s.iter().map(|x| be.sub(x, x)).collect();
            let pairs: Vec<(ShareRef, ShareRef)> = zeros
                .iter()
                .zip(&s)
                .map(|(z, x)| (z.clone(), x.clone()))
                .collect();
            let pos = be.lt_batch(&pairs); // 1 iff x > 0
            let mul_pairs: Vec<(ShareRef, ShareRef)> = s
                .iter()
                .zip(&pos)
                .map(|(x, p)| (x.clone(), p.clone()))
                .collect();
            be.mul_batch(&mul_pairs)
        }
    };

    // The per-element composition: one comparison and one multiply per
    // element, each its own round trip. This is what `gadgets::relu` was.
    let naive_body = {
        let xs = xs.clone();
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let s = share_all(be, me, &xs, nbits);
            s.iter()
                .map(|x| {
                    let zero = be.sub(x, x);
                    let pos = be.lt(&zero, x);
                    be.mul(x, &pos)
                })
                .collect()
        }
    };

    let routes = vec![
        route("naive", naive_body, &want, nbits, rtts),
        route("gadget", gadget_body, &want, nbits, rtts),
        route("batched", batched_body, &want, nbits, rtts),
    ];
    WorkloadReport {
        workload: "relu".into(),
        n,
        nbits,
        routes,
    }
}

/// Maximum over `n` secret values (a max-pool window taken to the limit).
fn max_workload(n: usize, nbits: usize, rtts: &[u64]) -> WorkloadReport {
    let xs: Vec<i64> = (0..n).map(|i| ((i * 7) as i64 % 29) - 14).collect();
    let want = vec![gadgets::max_reduce_ref(&xs)];

    let gadget_body = {
        let xs = xs.clone();
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let s = share_all(be, me, &xs, nbits);
            vec![gadgets::max_reduce(be, &s)]
        }
    };
    // A tournament over the batched pairwise max: ⌈log₂ n⌉ levels.
    let batched_body = {
        let xs = xs.clone();
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let mut cur = share_all(be, me, &xs, nbits);
            while cur.len() > 1 {
                let m = cur.len() / 2;
                let a: Vec<ShareRef> = (0..m).map(|k| cur[2 * k].clone()).collect();
                let b: Vec<ShareRef> = (0..m).map(|k| cur[2 * k + 1].clone()).collect();
                let mut next = gadgets::max2_batch(be, &a, &b);
                if cur.len() % 2 == 1 {
                    next.push(cur[cur.len() - 1].clone());
                }
                cur = next;
            }
            cur
        }
    };

    // The left fold: each `max2` waits on the previous accumulator, so the
    // comparisons are dependent and cannot batch. This is what
    // `gadgets::max_reduce` was.
    let naive_body = {
        let xs = xs.clone();
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let s = share_all(be, me, &xs, nbits);
            let mut acc = s[0].clone();
            for x in &s[1..] {
                acc = gadgets::max2(be, &acc, x);
            }
            vec![acc]
        }
    };

    let routes = vec![
        route("naive", naive_body, &want, nbits, rtts),
        route("gadget", gadget_body, &want, nbits, rtts),
        route("batched", batched_body, &want, nbits, rtts),
    ];
    WorkloadReport {
        workload: "max-reduce".into(),
        n,
        nbits,
        routes,
    }
}

/// A dense layer with **secret weights**: `out = W·x`, `W` is `out_dim × n`.
fn dense_workload(n: usize, out_dim: usize, nbits: usize, rtts: &[u64]) -> WorkloadReport {
    let xs: Vec<i64> = (0..n).map(|i| (i as i64 % 5) - 2).collect();
    let w: Vec<Vec<i64>> = (0..out_dim)
        .map(|r| (0..n).map(|c| ((r + c) as i64 % 7) - 3).collect())
        .collect();
    let bias: Vec<i64> = (0..out_dim).map(|r| r as i64 - 1).collect();
    let want = gadgets::dense_ref(&xs, &w, &bias);

    let gadget_body = {
        let (xs, w, bias) = (xs.clone(), w.clone(), bias.clone());
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let (x, ws, bs) = share_dense(be, me, &xs, &w, &bias, nbits);
            gadgets::dense(be, &x, &ws, &bs)
        }
    };
    // Every product in the layer is independent, so the whole matrix is one
    // batched multiply; the summation is local.
    let batched_body = {
        let (xs, w, bias) = (xs.clone(), w.clone(), bias.clone());
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let (x, ws, bs) = share_dense(be, me, &xs, &w, &bias, nbits);
            let mut pairs = Vec::with_capacity(ws.len() * x.len());
            for row in &ws {
                for (wi, xi) in row.iter().zip(&x) {
                    pairs.push((wi.clone(), xi.clone()));
                }
            }
            let prods = be.mul_batch(&pairs);
            ws.iter()
                .enumerate()
                .map(|(r, _)| {
                    let mut acc = bs[r].clone();
                    for c in 0..x.len() {
                        acc = be.add(&acc, &prods[r * x.len() + c]);
                    }
                    acc
                })
                .collect()
        }
    };

    // One multiply-accumulate per weight, each its own round trip. This is
    // what `gadgets::dense` was.
    let naive_body = {
        let (xs, w, bias) = (xs.clone(), w.clone(), bias.clone());
        move |be: &mut TransportBackend<MpcTransport>, me: usize| {
            let (x, ws, bs) = share_dense(be, me, &xs, &w, &bias, nbits);
            ws.iter()
                .zip(&bs)
                .map(|(row, b)| {
                    let mut acc = b.clone();
                    for (wi, xi) in row.iter().zip(&x) {
                        let p = be.mul(wi, xi);
                        acc = be.add(&acc, &p);
                    }
                    acc
                })
                .collect()
        }
    };

    let routes = vec![
        route("naive", naive_body, &want, nbits, rtts),
        route("gadget", gadget_body, &want, nbits, rtts),
        route("batched", batched_body, &want, nbits, rtts),
    ];
    WorkloadReport {
        workload: "dense".into(),
        n,
        nbits,
        routes,
    }
}

#[allow(clippy::type_complexity)]
fn share_dense(
    be: &mut TransportBackend<MpcTransport>,
    me: usize,
    xs: &[i64],
    w: &[Vec<i64>],
    bias: &[i64],
    nbits: usize,
) -> (Vec<ShareRef>, Vec<Vec<ShareRef>>, Vec<ShareRef>) {
    let x = share_all(be, me, xs, nbits);
    let ws: Vec<Vec<ShareRef>> = w.iter().map(|row| share_all(be, me, row, nbits)).collect();
    let bs = share_all(be, me, bias, nbits);
    (x, ws, bs)
}

fn route<F>(name: &str, body: F, want: &[i64], nbits: usize, rtts: &[u64]) -> RoutePoint
where
    F: Fn(&mut TransportBackend<MpcTransport>, usize) -> Vec<ShareRef>
        + Send
        + Sync
        + Clone
        + 'static,
{
    let mut ms_by_rtt = Vec::new();
    let mut comm = CommStats::default();
    let mut correct = true;
    for &rtt in rtts {
        let profile = NetworkProfile::fixed_latency(Duration::from_micros(rtt * 500));
        let (got, c, ms) = measure(profile, body.clone());
        let got: Vec<i64> = got.iter().map(|&v| to_signed(v as u128, nbits)).collect();
        correct &= got == want;
        comm = c;
        ms_by_rtt.push((rtt, ms));
    }
    RoutePoint {
        route: name.to_string(),
        messages: comm.messages,
        bytes: comm.bytes,
        ms_by_rtt,
        correct,
    }
}

const USAGE: &str = "usage: baru bench-abstraction [options]

What does Baru's abstraction cost? Runs one layer's work through the
composable gadget library and through the backend's batched primitives, over
the identical 3-party transport, and compares ROUNDS (message count) and
wall-clock at several round-trip times. Both routes must agree bit-exactly.

options:
  --n N          layer width (default 64)
  --out-dim D    dense layer output width (default 8)
  --nbits B      share bit width (default 20)
  --rtt-ms A,B   round-trip times to probe (default 0,2,20)
  --json         machine-readable output
  -h, --help     this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut n = 64usize;
    let mut out_dim = 8usize;
    let mut nbits = 20usize;
    let mut rtts: Vec<u64> = vec![0, 2, 20];
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
            "--out-dim" => {
                out_dim = it
                    .next()
                    .ok_or("--out-dim needs a value")?
                    .parse()
                    .map_err(|_| "bad --out-dim")?
            }
            "--nbits" => {
                nbits = it
                    .next()
                    .ok_or("--nbits needs a value")?
                    .parse()
                    .map_err(|_| "bad --nbits")?
            }
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

    let reports = vec![
        relu_workload(n, nbits, &rtts),
        max_workload(n, nbits, &rtts),
        dense_workload(n, out_dim, nbits, &rtts),
    ];

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("What Baru's abstraction costs — composable gadgets vs batched primitives");
    println!("Same transport, same protocol, same values; only the API differs.\n");
    for r in &reports {
        let width = if r.workload == "dense" {
            format!("{}x{}", out_dim, r.n)
        } else {
            r.n.to_string()
        };
        println!("{} ({} elements, {}-bit)", r.workload, width, r.nbits);
        print!("  {:<10} {:>10} {:>12}", "route", "messages", "bytes");
        for &(rtt, _) in &r.routes[0].ms_by_rtt {
            print!("{:>12}", format!("rtt {rtt}ms"));
        }
        println!();
        for p in &r.routes {
            if !p.correct {
                return Err(format!(
                    "{}/{}: route disagrees with the plaintext reference — \
                     a cost comparison between different computations is meaningless",
                    r.workload, p.route
                ));
            }
            print!("  {:<10} {:>10} {:>12}", p.route, p.messages, p.bytes);
            for &(_, ms) in &p.ms_by_rtt {
                print!("{:>12}", format!("{ms:.0}ms"));
            }
            println!();
        }
        let (nv, g, b) = (&r.routes[0], &r.routes[1], &r.routes[2]);
        if g.messages > 0 {
            println!(
                "  → naive costs {:.0}x the messages and {:.0}x the wall-clock at rtt {}ms",
                nv.messages as f64 / g.messages as f64,
                nv.ms_by_rtt.last().unwrap().1 / g.ms_by_rtt.last().unwrap().1.max(0.001),
                nv.ms_by_rtt.last().unwrap().0
            );
        }
        if g.messages != b.messages {
            return Err(format!(
                "{}: the gadget library sends {} messages where the hand-batched \
                 route sends {} — the composable API has regressed to a \
                 round-serial form",
                r.workload, g.messages, b.messages
            ));
        }
        println!();
    }
    println!(
        "All routes bit-exact against the plaintext reference, and `gadget` matches\n\
         `batched` message for message — composing through the library costs nothing\n\
         over hand-writing the batched calls. The gap to `naive` is round structure,\n\
         not protocol: same data, same protocol, more and smaller round trips."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every route must compute the same thing. This is the precondition for
    /// reporting a ratio between them, and it is checked at a small size so it
    /// runs in the fast suite.
    ///
    /// It also pins the **regression guard**: `gadget` must send exactly as
    /// many messages as the hand-batched route. The composable gadget library
    /// is the API the paper's composability section is about, and until
    /// 2026-07-28 it was round-serial — `relu` issued a comparison and a
    /// multiply per element, `dense` a round per weight. Nothing about the
    /// *results* changes when that regresses, so only a round-count assertion
    /// catches it. If this fires, some gadget went back to a per-element loop.
    #[test]
    fn gadget_route_matches_the_hand_batched_route_and_the_reference() {
        for r in [
            relu_workload(6, 20, &[0]),
            max_workload(6, 20, &[0]),
            dense_workload(4, 3, 20, &[0]),
        ] {
            for p in &r.routes {
                assert!(
                    p.correct,
                    "{}/{} disagrees with the reference",
                    r.workload, p.route
                );
            }
            let (naive, gadget, batched) = (&r.routes[0], &r.routes[1], &r.routes[2]);
            assert_eq!(
                gadget.messages, batched.messages,
                "{}: gadget library is no longer batched ({} messages vs {} hand-batched)",
                r.workload, gadget.messages, batched.messages
            );
            assert!(
                naive.messages > gadget.messages,
                "{}: the naive control must cost more, else it is not measuring the defect",
                r.workload
            );
        }
    }
}
