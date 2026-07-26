//! `baru bench-gc-compare` — do garbled circuits beat bit decomposition for
//! the comparison that dominates the MPC side of the hybrid pipeline?
//!
//! Every ReLU, max-pool and argmax reduces to a comparison, i.e. the sign
//! bit of a subtraction. We currently compute it with a **batched ABY3 bit
//! decomposition** (`a2b_batch`): correct, cheap in bandwidth, but
//! `O(\log b)` communication rounds. A **garbled circuit** evaluates the same
//! comparison in **one round** regardless of bit-width, at the cost of larger
//! garbled tables and per-gate hashing.
//!
//! This benchmark measures both with real work and — critically — **through
//! the identical transport**: the garbled tables and hashing are genuine
//! (`crate::crypto::garbled`), and both the bit decomposition and the garbled
//! circuit pay real bincode serialization, message passing, latency injection
//! and byte counting (`LatencyTransport` + `CountingTransport`). There is no
//! measurement asymmetry — the only differences are the two protocols. We
//! run the head-to-head at a range of round-trip times from LAN to WAN.
//!
//! Rounds (the structural driver):
//! - **a2b**: `4 + \lceil\log_2 b\rceil` — carry-save (2) + prefix-adder
//!   (init 1, `\lceil\log_2 b\rceil` layers, final 1). Nine at `b = 20`.
//! - **GC**: `1` — the garbler(s) send tables + input labels, the evaluator
//!   evaluates locally, both hold a share of the output. In the 3-party
//!   honest-majority setting the evaluator's input labels are delivered by
//!   the co-garbler, so **no oblivious transfer** is needed (the expensive
//!   part of 2-party GC); this is the setting that makes GC competitive here.
//!   The GC path here runs 2-party (garbler+evaluator) as a faithful cost
//!   proxy for that setting: same tables, same labels, same single round.

use std::time::Instant;

use rand::Rng;
use serde::Serialize;

use crate::bench::comm::{CountingTransport, LatencyTransport, NetworkProfile};
use crate::bench::hybrid_nn::run_parties;
use crate::crypto::garbled::{comparator_stats, eval_and, eval_comparator, eval_xor, Garbler, Label};
use crate::crypto::net::{recv_blocking, InprocHandle, Transport};
use crate::interpreter::backend::Backend;

#[derive(Debug, Clone, Serialize)]
pub struct MethodCost {
    pub name: String,
    pub compute_ms: f64,
    pub wire_bytes: usize,
    pub rounds: usize,
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GcCompareReport {
    pub n: usize,
    pub nbits: usize,
    pub a2b: MethodCost,
    pub gc: MethodCost,
}

/// a2b rounds for a `b`-bit decomposition (see module docs).
fn a2b_rounds(b: usize) -> usize {
    4 + (b as f64).log2().ceil() as usize
}

/// Run `n` batched comparisons through the real 3-party bit-decomposition
/// protocol at zero latency, returning (compute time, wire bytes, correct).
fn measure_a2b(n: usize, b: usize) -> (f64, usize, bool) {
    let mut rng = rand::thread_rng();
    let mask = (1i64 << b) - 1;
    // Signed values in [-2^(b-1), 2^(b-1)); lt is MSB of the two's-complement
    // difference, i.e. signed a < b.
    let av: Vec<i64> = (0..n).map(|_| rng.gen_range(-(1 << (b - 2))..(1 << (b - 2)))).collect();
    let bv: Vec<i64> = (0..n).map(|_| rng.gen_range(-(1 << (b - 2))..(1 << (b - 2)))).collect();
    let want: Vec<u128> = (0..n).map(|i| (av[i] < bv[i]) as u128).collect();

    let av2 = av.clone();
    let bv2 = bv.clone();
    let (results, comm) = run_parties(NetworkProfile::INSTANT, move |me| {
        let av = av2.clone();
        let bv = bv2.clone();
        move |be: &mut crate::interpreter::transport_backend::TransportBackend<_>| {
            let pairs: Vec<_> = (0..n)
                .map(|i| {
                    let a = be.share_secret(
                        if me == 0 { (av[i] as i128).rem_euclid(1i128 << b) as u128 } else { 0 },
                        b,
                        0,
                    );
                    let bb = be.share_secret(
                        if me == 0 { (bv[i] as i128).rem_euclid(1i128 << b) as u128 } else { 0 },
                        b,
                        0,
                    );
                    (a, bb)
                })
                .collect();
            // Time only the comparison, not the input sharing.
            let t0 = Instant::now();
            let lt = be.lt_batch(&pairs);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let bits: Vec<u128> = lt.iter().map(|r| be.reveal(r, None) & 1).collect();
            (elapsed, bits)
        }
    });
    let (elapsed, bits) = results.into_iter().next().unwrap();
    let correct = bits == want;
    let _ = mask;
    (elapsed, comm.bytes as usize, correct)
}

/// Garble and evaluate `n` real `b`-bit comparators; return (compute time,
/// wire bytes, correct). Correctness: each garbled output equals the
/// plaintext `MSB(x+y)`.
fn measure_gc(n: usize, b: usize) -> (f64, usize, bool) {
    let mut rng = rand::thread_rng();
    let mask = (1u128 << b) - 1;
    let inputs: Vec<(u128, u128)> = (0..n)
        .map(|_| (rng.gen::<u128>() & mask, rng.gen::<u128>() & mask))
        .collect();

    let t0 = Instant::now();
    let mut correct = true;
    let mut serialized_bytes = 0usize;
    for &(x, y) in &inputs {
        let mut g = Garbler::new(rng.gen());
        let circuit = g.garble_comparator(b);
        // A real garbler serializes the tables + active input labels onto the
        // wire; include that so the compute is not understated relative to
        // the a2b path (which is measured end-to-end through the transport).
        for t in &circuit.tables {
            serialized_bytes += t.tg.len() + t.te.len();
        }
        for i in 0..b {
            serialized_bytes += g.active(&circuit.x0[i], (x >> i) & 1 == 1).len();
            serialized_bytes += g.active(&circuit.y0[i], (y >> i) & 1 == 1).len();
        }
        let observed = eval_comparator(&g, &circuit, x, y, b);
        let got = observed ^ circuit.out_mask;
        let want = (((x + y) & mask) >> (b - 1)) & 1 == 1;
        correct &= got == want;
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    let stats = comparator_stats(n, b);
    debug_assert_eq!(serialized_bytes, stats.wire_bytes, "byte accounting mismatch");
    (elapsed, stats.wire_bytes, correct)
}

/// Garble and evaluate `n` `b`-bit comparators **through the real transport**
/// at network `profile`: the garbler serializes tables + active input labels
/// and sends them in one message; the evaluator deserializes and evaluates.
/// This measures GC the *same way* `a2b` is measured — real serialization,
/// real message passing, real latency injection, real byte counting — so the
/// two are apples-to-apples. Returns (wall-time ms, wire bytes, correct).
///
/// Two parties (garbler=0, evaluator=1). In the honest-majority 3-party
/// setting the evaluator's input labels would be supplied by the co-garbler
/// with no oblivious transfer; the byte count and single round are identical,
/// which is what this measures.
fn measure_gc_transport(n: usize, b: usize, profile: NetworkProfile) -> (f64, usize, bool) {
    let mut rng = rand::thread_rng();
    let mask = (1u128 << b) - 1;
    let inputs: Vec<(u128, u128)> = (0..n)
        .map(|_| (rng.gen::<u128>() & mask, rng.gen::<u128>() & mask))
        .collect();
    let seeds: Vec<u64> = (0..n).map(|_| rng.gen()).collect();

    let handles = InprocHandle::create(2);
    let mut wrapped: Vec<CountingTransport<LatencyTransport<InprocHandle>>> = handles
        .into_iter()
        .map(|h| CountingTransport::new(LatencyTransport::new(h, profile), 2))
        .collect();
    let comm = wrapped[0].handle();
    let evaluator = wrapped.pop().unwrap();
    let garbler = wrapped.pop().unwrap();

    let inputs_g = inputs.clone();
    let seeds_g = seeds.clone();
    let start = Instant::now();

    // Garbler: garble every comparator, serialize tables + active input
    // labels into one flat payload, send. (Gate tweaks are `2i`, reproduced by
    // the evaluator — not sent.)
    let gh = std::thread::spawn(move || {
        let mut buf = Vec::with_capacity(n * ((b - 1) * 32 + 2 * b * 16));
        let mut out_masks = Vec::with_capacity(n);
        for (k, &(x, y)) in inputs_g.iter().enumerate() {
            let mut g = Garbler::new(seeds_g[k]);
            let c = g.garble_comparator(b);
            for t in &c.tables {
                buf.extend_from_slice(&t.tg);
                buf.extend_from_slice(&t.te);
            }
            for i in 0..b {
                buf.extend_from_slice(&g.active(&c.x0[i], (x >> i) & 1 == 1));
                buf.extend_from_slice(&g.active(&c.y0[i], (y >> i) & 1 == 1));
            }
            out_masks.push(c.out_mask);
        }
        garbler.send(1, buf);
        out_masks
    });

    // Evaluator: receive, deserialize, evaluate every comparator.
    let eh = std::thread::spawn(move || {
        let (_from, buf) = recv_blocking(&evaluator, 60_000).expect("gc payload");
        let per = (b - 1) * 32 + 2 * b * 16;
        let mut observed = Vec::with_capacity(n);
        let read = |buf: &[u8], off: usize| -> Label {
            let mut l = [0u8; 16];
            l.copy_from_slice(&buf[off..off + 16]);
            l
        };
        for k in 0..n {
            let base = k * per;
            let mut tables = Vec::with_capacity(b - 1);
            for i in 0..b - 1 {
                let o = base + i * 32;
                tables.push((read(&buf, o), read(&buf, o + 16)));
            }
            let lbase = base + (b - 1) * 32;
            let mut xa = Vec::with_capacity(b);
            let mut ya = Vec::with_capacity(b);
            for i in 0..b {
                xa.push(read(&buf, lbase + (2 * i) * 16));
                ya.push(read(&buf, lbase + (2 * i + 1) * 16));
            }
            // Evaluate the ripple-adder MSB with deterministic tweaks `2i`.
            let mut carry = [0u8; 16];
            let mut carry_const0 = true;
            for i in 0..b - 1 {
                let (u, v) = if carry_const0 {
                    (xa[i], ya[i])
                } else {
                    (eval_xor(&xa[i], &carry), eval_xor(&ya[i], &carry))
                };
                let (tg, te) = tables[i];
                let and = eval_and(&u, &v, &crate::crypto::garbled::AndTable { tg, te }, 2 * i as u64);
                carry = if carry_const0 { and } else { eval_xor(&and, &carry) };
                carry_const0 = false;
            }
            let msb = eval_xor(&eval_xor(&xa[b - 1], &ya[b - 1]), &carry);
            observed.push(msb[0] & 1 == 1);
        }
        observed
    });

    let out_masks = gh.join().unwrap();
    let observed = eh.join().unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Combine the two output shares out-of-band (not a protocol round) and
    // check against the plaintext comparison.
    let mut correct = true;
    for k in 0..n {
        let got = observed[k] ^ out_masks[k];
        let want = (((inputs[k].0 + inputs[k].1) & mask) >> (b - 1)) & 1 == 1;
        correct &= got == want;
    }
    (elapsed, comm.stats().bytes as usize, correct)
}

/// Garble + evaluate `n` fused ReLU∘rescale ops through the transport — the
/// *whole* nonlinearity the pipeline runs, not just the comparison. Returns
/// (wall-time ms, bytes, correct). The output is boolean-shared; a downstream
/// B2A (accounted separately in the CLI) returns it to arithmetic.
fn measure_gc_relu_transport(n: usize, b: usize, shift: usize, profile: NetworkProfile) -> (f64, usize, bool) {
    use crate::crypto::garbled::relu_stats;
    let mut rng = rand::thread_rng();
    let mask = (1u128 << b) - 1;
    let inputs: Vec<(u128, u128)> = (0..n).map(|_| (rng.gen::<u128>() & mask, rng.gen::<u128>() & mask)).collect();
    let seeds: Vec<u64> = (0..n).map(|_| rng.gen()).collect();

    let handles = InprocHandle::create(2);
    let mut wrapped: Vec<CountingTransport<LatencyTransport<InprocHandle>>> = handles
        .into_iter()
        .map(|h| CountingTransport::new(LatencyTransport::new(h, profile), 2))
        .collect();
    let comm = wrapped[0].handle();
    let evaluator = wrapped.pop().unwrap();
    let garbler = wrapped.pop().unwrap();
    let n_and = (b - 1) + (b - shift);
    let width = b - shift;
    let per = n_and * 32 + 2 * b * 16;

    let inputs_g = inputs.clone();
    let seeds_g = seeds.clone();
    let start = Instant::now();
    let gh = std::thread::spawn(move || {
        let mut buf = Vec::with_capacity(n * per);
        let mut masks = Vec::with_capacity(n);
        for (k, &(x, y)) in inputs_g.iter().enumerate() {
            let mut g = Garbler::new(seeds_g[k]);
            let c = g.garble_relu_rescale(b, shift);
            for t in &c.tables {
                buf.extend_from_slice(&t.tg);
                buf.extend_from_slice(&t.te);
            }
            for i in 0..b {
                buf.extend_from_slice(&g.active(&c.x0[i], (x >> i) & 1 == 1));
                buf.extend_from_slice(&g.active(&c.y0[i], (y >> i) & 1 == 1));
            }
            masks.push(c.out_masks);
        }
        garbler.send(1, buf);
        (masks, inputs_g)
    });
    let eh = std::thread::spawn(move || {
        let (_from, buf) = recv_blocking(&evaluator, 60_000).expect("gc relu payload");
        let read = |buf: &[u8], off: usize| -> Label {
            let mut l = [0u8; 16];
            l.copy_from_slice(&buf[off..off + 16]);
            l
        };
        // Re-garble locally to obtain identical circuits is not needed; the
        // evaluator uses the received tables + labels and the deterministic
        // tweaks. We reconstruct the circuit structure from received bytes.
        let mut all = Vec::with_capacity(n);
        for k in 0..n {
            let base = k * per;
            let mut tables = Vec::with_capacity(n_and);
            for i in 0..n_and {
                let o = base + i * 32;
                tables.push(crate::crypto::garbled::AndTable { tg: read(&buf, o), te: read(&buf, o + 16) });
            }
            let lbase = base + n_and * 32;
            let mut xa = Vec::with_capacity(b);
            let mut ya = Vec::with_capacity(b);
            for i in 0..b {
                xa.push(read(&buf, lbase + (2 * i) * 16));
                ya.push(read(&buf, lbase + (2 * i + 1) * 16));
            }
            // adder
            let mut sa = vec![[0u8; 16]; b];
            let mut carry = [0u8; 16];
            let mut cc0 = true;
            let mut ti = 0usize;
            for i in 0..b {
                sa[i] = if cc0 { eval_xor(&xa[i], &ya[i]) } else { eval_xor(&eval_xor(&xa[i], &ya[i]), &carry) };
                if i < b - 1 {
                    let (u, v) = if cc0 { (xa[i], ya[i]) } else { (eval_xor(&xa[i], &carry), eval_xor(&ya[i], &carry)) };
                    let and = eval_and(&u, &v, &tables[ti], 2 * i as u64);
                    ti += 1;
                    carry = if cc0 { and } else { eval_xor(&and, &carry) };
                    cc0 = false;
                }
            }
            let keep = sa[b - 1];
            let mut obs = Vec::with_capacity(width);
            for j in 0..width {
                let g = 2 * (b - 1) as u64 + 2 * j as u64;
                let o = eval_and(&keep, &sa[shift + j], &tables[ti + j], g);
                obs.push(o[0] & 1 == 1);
            }
            all.push(obs);
        }
        all
    });
    let (masks, inputs_used) = gh.join().unwrap();
    let observed = eh.join().unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let mut correct = true;
    let signed = |v: u128| -> i128 { if v >> (b - 1) & 1 == 1 { v as i128 - (1i128 << b) } else { v as i128 } };
    for k in 0..n {
        let s = (inputs_used[k].0 + inputs_used[k].1) & mask;
        let v = signed(s);
        let want = if v >= 0 { (v >> shift) as u128 } else { 0 };
        let mut got = 0u128;
        for (i, (&o, &m)) in observed[k].iter().zip(&masks[k]).enumerate() {
            got |= ((o ^ m) as u128) << i;
        }
        correct &= got == want;
    }
    let _ = relu_stats(n, b, shift);
    (elapsed, comm.stats().bytes as usize, correct)
}

/// Run the `a2b`-based fused ReLU∘rescale through the transport: one
/// `a2b_batch` (all bits) then one `mul_batch` (the sign gate) — matching
/// `bench::cnn::fused_relu_rescale_batch`. Returns (wall-time ms, bytes, ok).
fn measure_a2b_relu_profile(n: usize, b: usize, shift: usize, profile: NetworkProfile) -> (f64, usize, bool) {
    let mut rng = rand::thread_rng();
    let vals: Vec<i64> = (0..n).map(|_| rng.gen_range(-(1 << (b - 2))..(1 << (b - 2)))).collect();
    let want: Vec<i64> = vals.iter().map(|&v| if v >= 0 { v >> shift } else { 0 }).collect();
    let (results, comm) = run_parties(profile, move |me| {
        let vals = vals.clone();
        move |be: &mut crate::interpreter::transport_backend::TransportBackend<_>| {
            let vs: Vec<_> = (0..n)
                .map(|i| be.share_secret(
                    if me == 0 { (vals[i] as i128).rem_euclid(1i128 << b) as u128 } else { 0 }, b, 0))
                .collect();
            let t0 = Instant::now();
            let bit_sets = be.a2b_batch(&vs, b);
            let one = be.share_public(1, 1);
            let mut pairs = Vec::with_capacity(n);
            for bits in &bit_sets {
                let mut shifted = be.b2a(&bits[shift..]);
                shifted.nbits = b;
                let keep = be.sub(&one, &bits[b - 1]);
                pairs.push((shifted, keep));
            }
            let acts = be.mul_batch(&pairs);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let out: Vec<i64> = acts.iter().map(|a| {
                let r = be.reveal(a, None);
                let m = (1i128 << b) - 1;
                let v = (r as i128) & m;
                if v >= (1i128 << (b - 1)) { (v - (1i128 << b)) as i64 } else { v as i64 }
            }).collect();
            (elapsed, out)
        }
    });
    let (elapsed, out) = results.into_iter().next().unwrap();
    (elapsed, comm.bytes as usize, out == want)
}

/// Run `a2b` through the transport at a given `profile` (real latency), for
/// the profile-by-profile head-to-head. Returns (wall-time ms, bytes, ok).
fn measure_a2b_profile(n: usize, b: usize, profile: NetworkProfile) -> (f64, usize, bool) {
    let mut rng = rand::thread_rng();
    let av: Vec<i64> = (0..n).map(|_| rng.gen_range(-(1 << (b - 2))..(1 << (b - 2)))).collect();
    let bv: Vec<i64> = (0..n).map(|_| rng.gen_range(-(1 << (b - 2))..(1 << (b - 2)))).collect();
    let want: Vec<u128> = (0..n).map(|i| (av[i] < bv[i]) as u128).collect();
    let (results, comm) = run_parties(profile, move |me| {
        let av = av.clone();
        let bv = bv.clone();
        move |be: &mut crate::interpreter::transport_backend::TransportBackend<_>| {
            let pairs: Vec<_> = (0..n)
                .map(|i| {
                    let a = be.share_secret(
                        if me == 0 { (av[i] as i128).rem_euclid(1i128 << b) as u128 } else { 0 },
                        b,
                        0,
                    );
                    let bb = be.share_secret(
                        if me == 0 { (bv[i] as i128).rem_euclid(1i128 << b) as u128 } else { 0 },
                        b,
                        0,
                    );
                    (a, bb)
                })
                .collect();
            let t0 = Instant::now();
            let lt = be.lt_batch(&pairs);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let bits: Vec<u128> = lt.iter().map(|r| be.reveal(r, None) & 1).collect();
            (elapsed, bits)
        }
    });
    let (elapsed, bits) = results.into_iter().next().unwrap();
    (elapsed, comm.bytes as usize, bits == want)
}

pub fn run(n: usize, b: usize) -> GcCompareReport {
    let (a_ms, a_bytes, a_ok) = measure_a2b(n, b);
    let (g_ms, g_bytes, g_ok) = measure_gc(n, b);
    GcCompareReport {
        n,
        nbits: b,
        a2b: MethodCost {
            name: "a2b (batched bit-decomposition)".into(),
            compute_ms: a_ms,
            wire_bytes: a_bytes,
            rounds: a2b_rounds(b),
            correct: a_ok,
        },
        gc: MethodCost {
            name: "garbled circuit (half-gates, 3PC no-OT)".into(),
            compute_ms: g_ms,
            wire_bytes: g_bytes,
            rounds: 1,
            correct: g_ok,
        },
    }
}

const USAGE: &str = "usage: baru bench-gc-compare [options]

Garbled circuits vs batched bit decomposition for the comparison that
dominates the MPC nonlinearities. Both measured with real work; the same
network model is applied to each across a range of round-trip times.

options:
  --n N         number of independent comparisons (default 1352, a D=28
                conv layer's worth of ReLUs)
  --nbits B     bit width of the compared values (default 20)
  --json        machine-readable output
  -h, --help    this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut n = 1352usize;
    let mut b = 20usize;
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" => n = it.next().ok_or("--n needs a value")?.parse().map_err(|_| "bad --n")?,
            "--nbits" => b = it.next().ok_or("--nbits needs a value")?.parse().map_err(|_| "bad --nbits")?,
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let r = run(n, b);
    if json {
        println!("{}", serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?);
        return Ok(());
    }

    println!("Comparison primitive: garbled circuits vs batched bit decomposition");
    println!("  workload: {} comparisons of {}-bit values", n, b);
    println!(
        "  structural: a2b {} rounds, {} B  |  GC {} round, {} B  (both bit-exact: {}/{})\n",
        r.a2b.rounds,
        r.a2b.wire_bytes,
        r.gc.rounds,
        r.gc.wire_bytes,
        if r.a2b.correct { "ok" } else { "BAD" },
        if r.gc.correct { "ok" } else { "BAD" },
    );

    // Head-to-head THROUGH THE REAL TRANSPORT at each network profile: both
    // methods pay real serialization, message passing and latency injection.
    let profiles: &[(&str, f64, Option<f64>)] = &[
        ("LAN   (RTT 0.1 ms, 10 Gbps)", 0.1, Some(10.0e9 / 8.0)),
        ("metro (RTT 2 ms, 1 Gbps)", 2.0, Some(1.0e9 / 8.0)),
        ("WAN   (RTT 20 ms, 1 Gbps)", 20.0, Some(1.0e9 / 8.0)),
        ("WAN   (RTT 80 ms, 100 Mbps)", 80.0, Some(100.0e6 / 8.0)),
    ];
    println!(
        "  measured through the real transport (one-way = RTT/2):\n  {:32}  {:>11}  {:>11}  {:>10}",
        "network", "a2b", "GC", "winner"
    );
    for (name, rtt, bw) in profiles {
        let one_way = std::time::Duration::from_secs_f64(rtt / 2.0 / 1000.0);
        let mut profile = NetworkProfile::fixed_latency(one_way);
        if let Some(bps) = bw {
            profile = profile.with_bandwidth(*bps);
        }
        let (la, _, aok) = measure_a2b_profile(n, b, profile);
        let (lg, _, gok) = measure_gc_transport(n, b, profile);
        assert!(aok && gok, "correctness must hold at every profile");
        let (winner, factor) = if la < lg { ("a2b", lg / la) } else { ("GC", la / lg) };
        println!("  {:32}  {:8.1} ms  {:8.1} ms  {} {:.1}x", name, la, lg, winner, factor);
    }
    // The pipeline's actual operation is the *fused ReLU∘rescale*, not a bare
    // comparison. Garbled circuits do the whole thing in one round; the a2b
    // path is a2b (9 rounds) + one sign-gate multiply (1 round).
    let shift = 8usize;
    println!("\n  fused ReLU∘rescale layer ({} ops, {}-bit, shift {}) through the transport:", n, b, shift);
    println!("  {:32}  {:>11}  {:>11}  {:>10}", "network", "a2b relu", "GC relu", "winner");
    for (name, rtt, bw) in profiles {
        let one_way = std::time::Duration::from_secs_f64(rtt / 2.0 / 1000.0);
        let mut profile = NetworkProfile::fixed_latency(one_way);
        if let Some(bps) = bw {
            profile = profile.with_bandwidth(*bps);
        }
        let (la, _, aok) = measure_a2b_relu_profile(n, b, shift, profile);
        let (mut lg, _, gok) = measure_gc_relu_transport(n, b, shift, profile);
        // The garbled output is boolean-shared; charge one B2A round to return
        // it to arithmetic for the next FHE switch (the a2b path already lands
        // in arithmetic). Modeled as one extra one-way hop.
        lg += rtt / 2.0;
        assert!(aok && gok, "relu correctness must hold at every profile");
        let (winner, factor) = if la < lg { ("a2b", lg / la) } else { ("GC", la / lg) };
        println!("  {:32}  {:8.1} ms  {:8.1} ms  {} {:.1}x", name, la, lg, winner, factor);
    }

    println!(
        "\n  note: both paths run through the identical transport (bincode +\n  \
         channels + latency injection). GC is 2-party (garbler+evaluator); in\n  \
         the honest-majority 3-party setting the evaluator's input labels are\n  \
         supplied by the co-garbler with no oblivious transfer — same bytes,\n  \
         same single round. The GC ReLU output is boolean-shared; one modeled\n  \
         B2A round is charged to return it to arithmetic for the next switch."
    );
    Ok(())
}
