//! `baru bench-malicious` — what does active security actually cost?
//!
//! The malicious ABY3 layer ([`crate::crypto::aby3_verify`]) is a drop-in
//! replacement for the semi-honest one, so the two can be run on identical
//! workloads through the **identical transport**, with the same latency
//! injection and byte counting. The only difference between the two legs is the
//! protocol, which is what makes the comparison worth quoting.
//!
//! # The prediction this tests
//!
//! Every value carries a MAC, so multiplications send twice the bytes. But both
//! halves batch into the *same* round, so the round count should not move at
//! all. This repository's recurring finding is that round structure rather than
//! arithmetic is what costs — four separate times a segment-level measurement
//! said "MPC is slow" while a per-phase breakdown said "one unbatched loop
//! inside it is slow". If that lesson holds, the overhead here lands almost
//! entirely in bytes and barely in wall-clock on a high-latency link.
//!
//! Bit decomposition is the exception, and the interesting one: it additionally
//! spends one batched round on a pool of random pairs and one batched opening to
//! attach MACs to its leaves ([`crate::crypto::aby3_verify::MacPair`]). Those
//! are two extra rounds *per decomposition batch*, not per value — so the
//! penalty should shrink as the batch widens.
//!
//! The numbers this prints are measurements, not estimates. Where they
//! contradict the prediction above, the prediction is what is wrong.

use std::time::Instant;

use serde::Serialize;

use crate::bench::comm::{CommStats, CountingTransport, LatencyTransport, NetworkProfile};
use crate::crypto::aby3_verify::{headroom_bits, product_width};
use crate::crypto::net::InprocHandle;
use crate::interpreter::backend::Backend;
use crate::interpreter::malicious_backend::MaliciousBackend;
use crate::interpreter::transport_backend::TransportBackend;

/// One protocol's cost on one workload.
#[derive(Debug, Clone, Serialize)]
pub struct Leg {
    pub name: String,
    pub wall_ms: f64,
    pub wire_bytes: u64,
    pub messages: u64,
    /// Batched interactive invocations — the round-structure proxy that is
    /// comparable across backends.
    pub interactive_calls: usize,
    pub correct: bool,
    /// Malicious leg only: did the deferred verification pass?
    pub verified: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    pub workload: String,
    pub semi_honest: Leg,
    pub malicious: Leg,
}

impl Comparison {
    pub fn byte_ratio(&self) -> f64 {
        self.malicious.wire_bytes as f64 / self.semi_honest.wire_bytes.max(1) as f64
    }
    pub fn call_ratio(&self) -> f64 {
        self.malicious.interactive_calls as f64 / self.semi_honest.interactive_calls.max(1) as f64
    }
    pub fn time_ratio(&self) -> f64 {
        self.malicious.wall_ms / self.semi_honest.wall_ms.max(1e-9)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MaliciousReport {
    pub profile: String,
    pub comparisons: Vec<Comparison>,
    /// Bits of statistical security the ring sacrifice actually has at the
    /// widths this workload uses — the Stage 1 claim, measured.
    pub sacrifice_headroom_bits: u32,
    pub widest_intermediate_bits: u32,
}

/// Drive three semi-honest parties through the counted transport.
fn run_semi<R, F>(profile: NetworkProfile, mk: impl Fn(usize) -> F) -> (Vec<R>, CommStats)
where
    R: Send + 'static,
    F: FnOnce(&mut TransportBackend<CountingTransport<LatencyTransport<InprocHandle>>>) -> R
        + Send
        + 'static,
{
    let handles = InprocHandle::create(3);
    let mut joins = Vec::with_capacity(3);
    let mut comm = None;
    for (me, h) in handles.into_iter().enumerate() {
        let counted = CountingTransport::new(LatencyTransport::new(h, profile), 3);
        if me == 0 {
            comm = Some(counted.handle());
        }
        let f = mk(me);
        joins.push(std::thread::spawn(move || {
            let mut be = TransportBackend::new(me, counted);
            f(&mut be)
        }));
    }
    let results = joins
        .into_iter()
        .map(|j| j.join().expect("party thread panicked"))
        .collect();
    (results, comm.unwrap().stats())
}

/// Drive three malicious parties through the *same* transport stack, taking the
/// deferred verdict inside the measured region so its cost is counted.
fn run_malicious<R, F>(
    profile: NetworkProfile,
    mk: impl Fn(usize) -> F,
) -> (Vec<(R, bool)>, CommStats)
where
    R: Send + 'static,
    F: FnOnce(&mut MaliciousBackend<CountingTransport<LatencyTransport<InprocHandle>>>) -> R
        + Send
        + 'static,
{
    let handles = InprocHandle::create(3);
    let mut joins = Vec::with_capacity(3);
    let mut comm = None;
    for (me, h) in handles.into_iter().enumerate() {
        let counted = CountingTransport::new(LatencyTransport::new(h, profile), 3);
        if me == 0 {
            comm = Some(counted.handle());
        }
        let f = mk(me);
        joins.push(std::thread::spawn(move || {
            let mut be = MaliciousBackend::new(me, counted);
            let out = f(&mut be);
            let verified = be.finish().is_ok();
            (out, verified)
        }));
    }
    let results = joins
        .into_iter()
        .map(|j| j.join().expect("party thread panicked"))
        .collect();
    (results, comm.unwrap().stats())
}

/// A dot product of `n` shared pairs — the multiplication-dominated shape.
fn dot_product(profile: NetworkProfile, n: usize, nbits: usize) -> Comparison {
    let expected: u128 = (0..n as u128).map(|i| (i + 1) * (i + 2)).sum();

    let t0 = Instant::now();
    let (semi_out, semi_comm) = run_semi(profile, move |_me| {
        move |be: &mut TransportBackend<_>| {
            // Default 128, so the headline ratio stays an **equal-width** comparison
            // against the verified leg — which cannot narrow, because
            // `aby3_verify::headroom_bits` is `128 − used_bits` and its statistical
            // security *is* that slack. `BARU_WIRE_BITS=auto` narrows this leg only,
            // giving the other meaningful ratio: active security measured against a
            // baseline that packs, which is what a deployment would actually run.
            if let Some(w) = crate::bench::cnn::wire_bits_override(nbits) {
                be.set_wire_bits(w);
            }
            // Batched input sharing, as every real workload in this repo does:
            // the scalar path costs one synchronisation per element and would
            // measure the loop rather than the protocol.
            let xv: Vec<u128> = (0..n as u128).map(|i| i + 1).collect();
            let yv: Vec<u128> = (0..n as u128).map(|i| i + 2).collect();
            let xs = be.share_secret_batch(&xv, nbits, 0);
            let ys = be.share_secret_batch(&yv, nbits, 1);
            let pairs: Vec<_> = xs.into_iter().zip(ys).collect();
            let prods = be.mul_batch(&pairs);
            let mut acc = prods[0].clone();
            for p in &prods[1..] {
                acc = be.add(&acc, p);
            }
            (be.reveal(&acc, None), be.interactive_calls())
        }
    });
    let semi_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let (mal_out, mal_comm) = run_malicious(profile, move |_me| {
        move |be: &mut MaliciousBackend<_>| {
            // Batched input sharing, as every real workload in this repo does:
            // the scalar path costs one synchronisation per element and would
            // measure the loop rather than the protocol.
            let xv: Vec<u128> = (0..n as u128).map(|i| i + 1).collect();
            let yv: Vec<u128> = (0..n as u128).map(|i| i + 2).collect();
            let xs = be.share_secret_batch(&xv, nbits, 0);
            let ys = be.share_secret_batch(&yv, nbits, 1);
            let pairs: Vec<_> = xs.into_iter().zip(ys).collect();
            let prods = be.mul_batch(&pairs);
            let mut acc = prods[0].clone();
            for p in &prods[1..] {
                acc = be.add(&acc, p);
            }
            (be.reveal(&acc, None), be.interactive_calls())
        }
    });
    let mal_ms = t1.elapsed().as_secs_f64() * 1e3;

    let mask = if nbits >= 128 {
        u128::MAX
    } else {
        (1u128 << nbits) - 1
    };
    Comparison {
        workload: format!("dot product, n={n}, {nbits}-bit"),
        semi_honest: Leg {
            name: "ABY3 (semi-honest)".into(),
            wall_ms: semi_ms,
            wire_bytes: semi_comm.bytes,
            messages: semi_comm.messages,
            interactive_calls: semi_out[0].1,
            correct: semi_out.iter().all(|(v, _)| v & mask == expected & mask),
            verified: None,
        },
        malicious: Leg {
            name: "ABY3 (malicious, with abort)".into(),
            wall_ms: mal_ms,
            wire_bytes: mal_comm.bytes,
            messages: mal_comm.messages,
            interactive_calls: mal_out[0].0 .1,
            correct: mal_out
                .iter()
                .all(|((v, _), _)| v & mask == expected & mask),
            verified: Some(mal_out.iter().all(|(_, ok)| *ok)),
        },
    }
}

/// A batch of comparisons — the shape every ReLU, max-pool and argmax reduces
/// to, and the one that exercises bit decomposition.
fn comparisons(profile: NetworkProfile, n: usize, nbits: usize) -> Comparison {
    let t0 = Instant::now();
    let (semi_out, semi_comm) = run_semi(profile, move |_me| {
        move |be: &mut TransportBackend<_>| {
            // Default 128, so the headline ratio stays an **equal-width** comparison
            // against the verified leg — which cannot narrow, because
            // `aby3_verify::headroom_bits` is `128 − used_bits` and its statistical
            // security *is* that slack. `BARU_WIRE_BITS=auto` narrows this leg only,
            // giving the other meaningful ratio: active security measured against a
            // baseline that packs, which is what a deployment would actually run.
            if let Some(w) = crate::bench::cnn::wire_bits_override(nbits) {
                be.set_wire_bits(w);
            }
            let xv: Vec<u128> = (0..n).map(|i| (i % 17) as u128).collect();
            let yv: Vec<u128> = (0..n).map(|i| (i % 11) as u128).collect();
            let xs = be.share_secret_batch(&xv, nbits, 0);
            let ys = be.share_secret_batch(&yv, nbits, 1);
            let pairs: Vec<_> = xs.into_iter().zip(ys).collect();
            let lts = be.lt_batch(&pairs);
            let opened: Vec<u128> = lts.iter().map(|s| be.reveal(s, None) & 1).collect();
            (opened, be.interactive_calls())
        }
    });
    let semi_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let (mal_out, mal_comm) = run_malicious(profile, move |_me| {
        move |be: &mut MaliciousBackend<_>| {
            let xv: Vec<u128> = (0..n).map(|i| (i % 17) as u128).collect();
            let yv: Vec<u128> = (0..n).map(|i| (i % 11) as u128).collect();
            let xs = be.share_secret_batch(&xv, nbits, 0);
            let ys = be.share_secret_batch(&yv, nbits, 1);
            let pairs: Vec<_> = xs.into_iter().zip(ys).collect();
            let lts = be.lt_batch(&pairs);
            let opened: Vec<u128> = lts.iter().map(|s| be.reveal(s, None) & 1).collect();
            (opened, be.interactive_calls())
        }
    });
    let mal_ms = t1.elapsed().as_secs_f64() * 1e3;

    let want: Vec<u128> = (0..n).map(|i| ((i % 17) < (i % 11)) as u128).collect();
    Comparison {
        workload: format!("comparisons, n={n}, {nbits}-bit"),
        semi_honest: Leg {
            name: "ABY3 (semi-honest)".into(),
            wall_ms: semi_ms,
            wire_bytes: semi_comm.bytes,
            messages: semi_comm.messages,
            interactive_calls: semi_out[0].1,
            correct: semi_out.iter().all(|(v, _)| *v == want),
            verified: None,
        },
        malicious: Leg {
            name: "ABY3 (malicious, with abort)".into(),
            wall_ms: mal_ms,
            wire_bytes: mal_comm.bytes,
            messages: mal_comm.messages,
            interactive_calls: mal_out[0].0 .1,
            correct: mal_out.iter().all(|((v, _), _)| *v == want),
            verified: Some(mal_out.iter().all(|(_, ok)| *ok)),
        },
    }
}

pub fn run(profile: NetworkProfile, profile_name: &str, n: usize, nbits: usize) -> MaliciousReport {
    let comparisons_report = comparisons(profile, n.min(32), nbits);
    let dot = dot_product(profile, n, nbits);
    let widest = product_width(nbits, n);
    MaliciousReport {
        profile: profile_name.to_string(),
        comparisons: vec![dot, comparisons_report],
        sacrifice_headroom_bits: headroom_bits(widest),
        widest_intermediate_bits: widest,
    }
}

pub fn main(args: &[String]) {
    let mut n = 64usize;
    let mut nbits = 32usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                n = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(n);
                i += 2;
            }
            "--bits" => {
                nbits = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(nbits);
                i += 2;
            }
            _ => i += 1,
        }
    }

    for (name, profile) in [
        ("LAN (instant)", NetworkProfile::INSTANT),
        // 25 ms one-way — a representative inter-region WAN hop, which is
        // where round structure rather than bytes decides the cost.
        (
            "WAN (25ms one-way)",
            NetworkProfile::fixed_latency(std::time::Duration::from_millis(25)),
        ),
    ] {
        let r = run(profile, name, n, nbits);
        println!("\n=== the cost of active security — {} ===", r.profile);
        println!(
            "ring sacrifice: values occupy {} bits of the 128-bit ring, leaving {} bits \
             of statistical security at no widening cost",
            r.widest_intermediate_bits, r.sacrifice_headroom_bits
        );
        for c in &r.comparisons {
            println!("\n{}", c.workload);
            println!(
                "  {:<32} {:>9} {:>12} {:>10} {:>9}",
                "protocol", "wall ms", "wire bytes", "messages", "calls"
            );
            for leg in [&c.semi_honest, &c.malicious] {
                println!(
                    "  {:<32} {:>9.2} {:>12} {:>10} {:>9}{}",
                    leg.name,
                    leg.wall_ms,
                    leg.wire_bytes,
                    leg.messages,
                    leg.interactive_calls,
                    match leg.verified {
                        Some(true) => "  [verified]",
                        Some(false) => "  [ABORTED]",
                        None => "",
                    }
                );
                assert!(leg.correct, "{} computed the wrong answer", leg.name);
            }
            println!(
                "  overhead: {:.2}x bytes, {:.2}x interactive calls, {:.2}x wall clock",
                c.byte_ratio(),
                c.call_ratio(),
                c.time_ratio()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both legs must compute the same, correct answer — otherwise the cost
    /// comparison is meaningless.
    #[test]
    fn both_protocols_agree_and_the_malicious_one_verifies() {
        let r = run(NetworkProfile::INSTANT, "test", 8, 16);
        for c in &r.comparisons {
            assert!(c.semi_honest.correct, "{}: semi-honest wrong", c.workload);
            assert!(c.malicious.correct, "{}: malicious wrong", c.workload);
            assert_eq!(
                c.malicious.verified,
                Some(true),
                "{}: an honest run must verify",
                c.workload
            );
        }
    }

    /// The headline structural claim: MACs cost bytes, not rounds. Multiplying
    /// batches the value and MAC halves into the same round, so the
    /// interactive-call count should not grow with the upgrade.
    #[test]
    fn the_multiplication_overhead_is_bytes_not_rounds() {
        let r = run(NetworkProfile::INSTANT, "test", 16, 16);
        let dot = &r.comparisons[0];
        assert!(
            dot.malicious.wire_bytes > dot.semi_honest.wire_bytes,
            "MACs must cost bytes"
        );
        assert!(
            dot.call_ratio() <= 2.0,
            "multiplication must not multiply rounds: {:.2}x calls",
            dot.call_ratio()
        );
    }

    /// The result that matters for private inference: on the comparison
    /// workload — what every ReLU, max-pool and argmax reduces to — the
    /// malicious upgrade must cost bytes, not rounds.
    ///
    /// Bit decomposition adds two rounds per *batch* (the random-pair pool and
    /// the MAC-attaching opening), so the ratio must fall toward 1 as the batch
    /// widens rather than staying constant.
    #[test]
    fn comparisons_pay_in_bytes_not_rounds() {
        let r = run(NetworkProfile::INSTANT, "test", 32, 16);
        let cmp = &r.comparisons[1];
        assert!(
            cmp.byte_ratio() > 1.5,
            "MACs must show up in bytes: {:.2}x",
            cmp.byte_ratio()
        );
        assert!(
            cmp.call_ratio() < 1.3,
            "the two extra rounds must amortise over the batch: {:.2}x calls",
            cmp.call_ratio()
        );
    }

    /// The ring-slack claim from Stage 1, checked at the widths a real workload
    /// uses rather than asserted.
    #[test]
    fn the_sacrifice_has_real_headroom_at_realistic_widths() {
        let r = run(NetworkProfile::INSTANT, "test", 8, 16);
        assert!(
            r.sacrifice_headroom_bits >= 40,
            "only {} bits of headroom at {} used bits",
            r.sacrifice_headroom_bits,
            r.widest_intermediate_bits
        );
    }
}
