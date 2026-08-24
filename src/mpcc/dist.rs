//! A **distributed** two-party engine: one party per thread, one share each,
//! talking over a transport that really waits.
//!
//! # Why this exists
//!
//! Every other [`ShareEngine`](crate::mpcc::vm::ShareEngine) in this crate runs
//! all parties in one process and holds every party's share, which makes results
//! and communication *counts* real but makes wall clock a construction: rounds
//! and bytes must be put through a network model, and local compute has to be
//! divided by the party count because the parties ran serially. A composed figure
//! is not a running time, and a comparison against a system whose latency is
//! elapsed wall clock cannot rest on one.
//!
//! This engine removes that. Each party holds a single `u64` per shared value,
//! executes the same bytecode in the same schedule, and exchanges messages over a
//! [`Transport`]. Run one party per thread against
//! [`LatencyTransport`](crate::bench::comm::LatencyTransport), whose worker
//! *sleeps* until a message's delivery time, and the elapsed time of the slowest
//! party is a measurement: real compute, really overlapped, really waiting for
//! the link.
//!
//! # Why the trait did not have to change
//!
//! [`ShareEngine`] turns out to be exactly the right boundary. Its interactive
//! methods — `input`, `mul_batch`, `open_batch` — are precisely the points at
//! which a party must speak, and everything else is local. So
//! [`vm::execute`](crate::mpcc::vm::execute), `bit_decompose_batch` and
//! `random_bits_batch` run over this engine unmodified, which also means the
//! distributed run computes the *same* program as the in-process one and can be
//! checked against it.
//!
//! # What is real here and what is not
//!
//! Real: the arithmetic, the sharing, the messages, the round structure, the
//! concurrency, and the waiting. Elapsed time includes each party's own compute
//! and its idle time on the link.
//!
//! Not real: the **triples**. This engine takes correlated randomness from a
//! seed both parties hold, which is the trusted-dealer model — the online-phase
//! lower bound, and the right thing to measure when the question is a warm
//! server's latency, since preprocessing is query-independent. It is *not* a cold
//! figure, and a cold figure needs the oblivious transfer or the
//! somewhat-homomorphic generation to run over this same transport. That is
//! deliberately not simulated here: an engine that pretended to pay for
//! preprocessing without sending its bytes would reintroduce exactly the
//! composed number this module exists to eliminate.
//!
//! Also not a deployment: both parties are threads of one process on one machine,
//! so they contend for cores and the link is emulated rather than a socket.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::crypto::net::{PartyId, Transport};

use super::vm::{Counters, Modulus, ShareEngine};

/// One party's additive share of a value.
pub type Share = u64;

/// A batch of ring elements with the sequence number of the exchange it belongs
/// to. Both parties execute the same program in the same order, so the sequence
/// is a desynchronisation check and not a demultiplexer: a mismatch means the two
/// sides disagree about the protocol, which is a bug rather than a race.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Exchange {
    seq: u64,
    vals: Vec<u64>,
}

/// Correlated randomness from a seed both parties hold.
///
/// This is the dealer model made concrete: each party derives its own share of
/// every triple from the same seed, so no communication is needed and the online
/// phase is isolated. See the module docs on what that does and does not measure.
struct Dealer {
    m: Modulus,
    state: u64,
    me: PartyId,
}

impl Dealer {
    fn new(m: Modulus, seed: u64, me: PartyId) -> Dealer {
        Dealer {
            m,
            state: seed | 1,
            me,
        }
    }

    /// SplitMix64, so both parties walk the same stream without a shared RNG.
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn elem(&mut self) -> u64 {
        let v = self.next();
        match self.m {
            Modulus::Ring { k } if k >= 64 => v,
            Modulus::Ring { k } => v & ((1u64 << k) - 1),
            Modulus::Prime(p) => v % p,
        }
    }

    /// This party's share of a fresh Beaver triple.
    fn triple(&mut self) -> (Share, Share, Share) {
        // Both parties derive every component, then keep only their own; the
        // arithmetic is identical on both sides, so the shares are consistent.
        let a0 = self.elem();
        let a1 = self.elem();
        let b0 = self.elem();
        let b1 = self.elem();
        let c0 = self.elem();
        let a = self.m.add(a0, a1);
        let b = self.m.add(b0, b1);
        let c1 = self.m.sub(self.m.mul(a, b), c0);
        if self.me == 0 {
            (a0, b0, c0)
        } else {
            (a1, b1, c1)
        }
    }
}

/// A party of a two-party additive-sharing protocol over a real transport.
pub struct DistEngine<T: Transport> {
    m: Modulus,
    me: PartyId,
    net: Arc<T>,
    counters: Counters,
    dealer: Dealer,
    /// This party's **private** randomness, used to mask its own inputs.
    ///
    /// Kept strictly separate from the dealer stream. Drawing an input mask from
    /// the dealer would desynchronise the two parties' streams — the owner draws
    /// and the receiver does not — so every triple after the first input would
    /// differ between them, and the run silently computes nonsense. Correlated
    /// randomness must be consumed in lockstep; private randomness must not be
    /// consumed from it at all.
    rng: Dealer,
    seq: u64,
    /// Messages received out of turn, kept rather than dropped.
    pending: Vec<(u64, Vec<u64>)>,
    timeout_ms: u64,
    /// When set, shared random bits are taken from the dealer stream instead of
    /// being generated by the exclusive-or protocol.
    ///
    /// This is the difference between two configurations that must not be
    /// conflated. Random bits are **query-independent**: a deployment generates
    /// them before the query arrives, exactly as an FHE deployment generates keys
    /// before the query arrives. Hoisting them measures the *online phase*, which
    /// is the figure comparable to a warm FHE server. Generating them inline
    /// measures a server that was handed a query with nothing prepared. Both are
    /// real elapsed times; they answer different questions, and this flag is which
    /// one is being asked.
    hoist_bits: bool,
    /// Rescale with truncation pairs instead of a bit decomposition.
    trunc_pairs: bool,
    /// Evaluate bilinear forms by opening each operand once.
    bilinear: bool,
}

impl<T: Transport> DistEngine<T> {
    /// `seed` must be the same at both parties: it is the dealer's stream.
    pub fn new(me: PartyId, net: Arc<T>, bits: u32, seed: u64) -> Result<Self, String> {
        if me > 1 {
            return Err(format!("dist: two-party engine, got party {me}"));
        }
        let m = Modulus::Ring { k: bits };
        Ok(DistEngine {
            m,
            me,
            net,
            counters: Counters::default(),
            dealer: Dealer::new(m, seed, me),
            // A per-party stream, so the two parties' masks differ and neither
            // disturbs the shared one.
            rng: Dealer::new(m, seed ^ (0xA5A5_5A5A_u64 << 1) ^ (me as u64 + 1), me),
            seq: 0,
            pending: Vec::new(),
            timeout_ms: 60_000,
            hoist_bits: false,
            trunc_pairs: false,
            bilinear: false,
        })
    }

    /// Take shared random bits from preprocessing rather than generating them.
    pub fn with_hoisted_bits(mut self, hoist: bool) -> Self {
        self.hoist_bits = hoist;
        self
    }

    /// Rescale with truncation pairs: one opening instead of a decomposition, at
    /// the cost of a stochastic carry in the last bit. See
    /// [`super::vm::ShareEngine::trunc_pair_mode`].
    pub fn with_trunc_pairs(mut self, on: bool) -> Self {
        self.trunc_pairs = on;
        self
    }

    /// Evaluate convolutions and matrix products by opening each operand once rather
    /// than once per product. See [`super::vm::bilinear_batch`].
    pub fn with_bilinear(mut self, on: bool) -> Self {
        self.bilinear = on;
        self
    }

    fn peer(&self) -> PartyId {
        1 - self.me
    }

    /// Send this party's half of an exchange and block for the peer's.
    ///
    /// This is the only place the engine communicates, so it is also the only
    /// place a round is spent: every interactive primitive below is one call.
    fn exchange(&mut self, vals: Vec<u64>) -> Result<Vec<u64>, String> {
        let seq = self.seq;
        self.seq += 1;
        let bytes = self.m.element_bytes() * vals.len() as u64;
        self.counters.online_bytes += bytes;

        let msg = bincode::serialize(&Exchange {
            seq,
            vals: vals.clone(),
        })
        .map_err(|e| format!("dist: serialize failed: {e}"))?;
        self.net.send(self.peer(), msg);

        // A message already buffered for this sequence needs no waiting.
        if let Some(i) = self.pending.iter().position(|(s, _)| *s == seq) {
            return Ok(self.pending.swap_remove(i).1);
        }
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_millis(self.timeout_ms);
        loop {
            if let Some((from, raw)) = self.net.recv() {
                if from != self.peer() {
                    // Two parties: nothing else may speak. Refusing rather than
                    // ignoring, since a third sender means the run is not what it
                    // claims to be.
                    return Err(format!("dist: message from unexpected party {from}"));
                }
                let ex: Exchange = bincode::deserialize(&raw)
                    .map_err(|e| format!("dist: malformed exchange: {e}"))?;
                if ex.seq == seq {
                    return Ok(ex.vals);
                }
                self.pending.push((ex.seq, ex.vals));
                continue;
            }
            if std::time::Instant::now() > deadline {
                return Err(format!(
                    "dist: party {} timed out waiting for exchange {seq}",
                    self.me
                ));
            }
            std::hint::spin_loop();
        }
    }
}

impl<T: Transport> ShareEngine for DistEngine<T> {
    type Share = Share;

    fn modulus(&self) -> Modulus {
        self.m
    }
    fn parties(&self) -> usize {
        2
    }
    fn counters(&self) -> &Counters {
        &self.counters
    }
    fn counters_mut(&mut self) -> &mut Counters {
        &mut self.counters
    }

    /// `value` is used **only** when this party owns the input; the other party
    /// receives its component and never sees the secret. That is what makes the
    /// run a real two-party execution and not a simulation of one.
    fn input(&mut self, value: u64, party: usize) -> Share {
        let mine = if party == self.me {
            let r = self.rng.elem();
            let other = self.m.sub(value, r);
            let got = self
                .exchange(vec![other])
                .expect("dist: input exchange failed");
            debug_assert_eq!(got.len(), 1);
            r
        } else {
            // Send nothing of substance, receive the owner's component. The empty
            // payload keeps the exchange symmetric so both parties stay in step.
            let got = self
                .exchange(Vec::new())
                .expect("dist: input exchange failed");
            *got.first().unwrap_or(&0)
        };
        self.counters.opens += 0;
        mine
    }

    /// One exchange for the whole layer of inputs.
    ///
    /// Each party sends the components of the inputs **it owns**, in index order;
    /// both sides know from `party` which indices the peer owns, so both consume
    /// the peer's vector in the same order without any tagging.
    fn input_batch(&mut self, values: &[(u64, usize)]) -> Vec<Share> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut mine: Vec<Option<Share>> = Vec::with_capacity(values.len());
        let mut out = Vec::new();
        for (v, party) in values {
            if *party == self.me {
                let r = self.rng.elem();
                out.push(self.m.sub(*v, r));
                mine.push(Some(r));
            } else {
                mine.push(None);
            }
        }
        let peer = self
            .exchange(out)
            .expect("dist: input batch exchange failed");
        let mut it = peer.into_iter();
        mine.into_iter()
            .map(|slot| match slot {
                Some(r) => r,
                None => it.next().unwrap_or(0),
            })
            .collect()
    }

    fn constant(&mut self, value: u64) -> Share {
        // One party carries the constant; both together sum to it.
        if self.me == 0 {
            value
        } else {
            0
        }
    }

    fn add(&mut self, a: &Share, b: &Share) -> Share {
        self.m.add(*a, *b)
    }
    fn sub(&mut self, a: &Share, b: &Share) -> Share {
        self.m.sub(*a, *b)
    }
    fn add_const(&mut self, a: &Share, c: u64) -> Share {
        if self.me == 0 {
            self.m.add(*a, c)
        } else {
            *a
        }
    }
    fn mul_const(&mut self, a: &Share, c: u64) -> Share {
        self.m.mul(*a, c)
    }

    /// Beaver multiplication: one exchange for the whole batch, whatever its
    /// width. This is the property that makes round count track dependence depth
    /// and not operation count.
    fn mul_batch(&mut self, pairs: &[(Share, Share)]) -> Vec<Share> {
        if pairs.is_empty() {
            return Vec::new();
        }
        self.counters.muls += pairs.len() as u64;
        let mut triples = Vec::with_capacity(pairs.len());
        let mut out_msg = Vec::with_capacity(pairs.len() * 2);
        for (x, y) in pairs {
            let (a, b, c) = self.dealer.triple();
            self.counters.triples += 1;
            out_msg.push(self.m.sub(*x, a));
            out_msg.push(self.m.sub(*y, b));
            triples.push((a, b, c));
        }
        let peer = self.exchange(out_msg.clone()).expect("dist: mul exchange");
        assert_eq!(peer.len(), out_msg.len(), "dist: mul batch width mismatch");

        let mut out = Vec::with_capacity(pairs.len());
        for (i, (a, b, c)) in triples.into_iter().enumerate() {
            let d = self.m.add(out_msg[2 * i], peer[2 * i]);
            let e = self.m.add(out_msg[2 * i + 1], peer[2 * i + 1]);
            // z = c + d*b + e*a, plus the public d*e at exactly one party.
            let mut z = self.m.add(c, self.m.mul(d, b));
            z = self.m.add(z, self.m.mul(e, a));
            if self.me == 0 {
                z = self.m.add(z, self.m.mul(d, e));
            }
            out.push(z);
        }
        out
    }

    fn open_batch(&mut self, values: &[Share]) -> Vec<u64> {
        if values.is_empty() {
            return Vec::new();
        }
        self.counters.opens += values.len() as u64;
        let peer = self
            .exchange(values.to_vec())
            .expect("dist: open exchange failed");
        assert_eq!(peer.len(), values.len(), "dist: open batch width mismatch");
        values
            .iter()
            .zip(peer)
            .map(|(mine, theirs)| self.m.add(*mine, theirs))
            .collect()
    }

    fn trunc_pair_mode(&self) -> bool {
        self.trunc_pairs
    }

    fn bilinear_triple(
        &mut self,
        n_in: usize,
        n_w: usize,
        terms: &[Vec<(u32, u32)>],
    ) -> Option<super::vm::BilinearTriple<Share>> {
        if !self.bilinear {
            return None;
        }
        // Both parties walk the same stream, so both derive additive shares of the same
        // `a`, `b` and `F(a, b)` without exchanging anything. `a` and `b` are drawn in
        // the clear from the shared stream and then split; the form is evaluated on the
        // clear values and split the same way. A real preprocessing phase would produce
        // this from oblivious transfer or a homomorphic convolution — which is what
        // Cheetah's offline phase is — and the accounting note in the module docs applies
        // here exactly as it does to scalar triples.
        let a_clear: Vec<u64> = (0..n_in).map(|_| self.dealer.elem()).collect();
        let b_clear: Vec<u64> = (0..n_w).map(|_| self.dealer.elem()).collect();
        let mut a = Vec::with_capacity(n_in);
        let mut b = Vec::with_capacity(n_w);
        let mut c = Vec::with_capacity(terms.len());
        for v in &a_clear {
            let s0 = self.dealer.elem();
            a.push(if self.me == 0 { s0 } else { self.m.sub(*v, s0) });
        }
        for v in &b_clear {
            let s0 = self.dealer.elem();
            b.push(if self.me == 0 { s0 } else { self.m.sub(*v, s0) });
        }
        for term in terms {
            let mut acc = 0u64;
            for &(i, w) in term {
                acc = self
                    .m
                    .add(acc, self.m.mul(a_clear[i as usize], b_clear[w as usize]));
            }
            let s0 = self.dealer.elem();
            c.push(if self.me == 0 {
                s0
            } else {
                self.m.sub(acc, s0)
            });
        }
        // Charged like any other correlated randomness: query-independent, so its
        // generation cost belongs in the offline column, and the online traffic is the
        // two operand openings that `vm::bilinear_batch` performs.
        self.counters.triples += terms.iter().map(|t| t.len() as u64).sum::<u64>();
        Some((a, b, c))
    }

    fn random_bits(&mut self, n: usize) -> Vec<Share> {
        if self.hoist_bits {
            // Preprocessed: both parties derive consistent shares of the same
            // uniform bit from the shared stream, so no message is sent and the
            // measurement isolates the online phase.
            self.counters.random_bits += n as u64;
            return (0..n)
                .map(|_| {
                    let b = self.dealer.next() & 1;
                    let r = self.dealer.elem();
                    if self.me == 0 {
                        r
                    } else {
                        self.m.sub(b, r)
                    }
                })
                .collect();
        }
        // Generated: each party contributes a private bit and the shared bit is
        // their exclusive-or, one multiplication and therefore one triple per bit.
        // Identical construction to every other engine, so counts stay comparable.
        super::vm::random_bits_batch(self, n)
    }
}

// ─── Running two parties concurrently ─────────────────────────────────────────

/// What a distributed run measured. Every field is observed, none is composed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DistMeasured {
    /// Elapsed wall clock of the **slowest** party, which is when the
    /// computation is actually finished.
    pub elapsed_secs: f64,
    /// Each party's own elapsed time, so asymmetry is visible.
    pub per_party_secs: Vec<f64>,
    /// Bytes this party sent, taken from the engine's own counter; the busiest
    /// party's figure is reported.
    pub online_bytes: u64,
    pub muls: u64,
    pub opens: u64,
    pub triples: u64,
    pub random_bits: u64,
    /// Interactive exchanges performed, which is the number of times a party
    /// blocked on its peer.
    pub exchanges: u64,
    pub outputs: Vec<(String, i64)>,
}

/// How a distributed run is configured — chiefly *which accounting question it
/// answers*, since the two flags below move cost across the online/offline
/// boundary rather than reducing it.
#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    /// Ring width the parties compute in.
    pub bits: u32,
    /// Shared seed for the correlated-randomness stream.
    pub seed: u64,
    /// Take shared random bits from preprocessing rather than generating them
    /// inline. `true` measures the online phase — the figure comparable against a
    /// warm homomorphic server; `false` measures a server handed a query with
    /// nothing prepared. Both are real elapsed times for different deployments.
    pub hoist_bits: bool,
    /// Rescale with truncation pairs rather than a bit decomposition. See
    /// [`super::vm::ShareEngine::trunc_pair_mode`]; this changes the computed
    /// function, not only its cost.
    pub trunc_pairs: bool,
    /// Evaluate bilinear forms by opening each operand once. See
    /// [`super::vm::bilinear_batch`]; this changes cost only, not the answer.
    pub bilinear: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            bits: 64,
            seed: 0xC0FFEE,
            hoist_bits: false,
            trunc_pairs: false,
            bilinear: false,
        }
    }
}

/// One party's measured tally, as returned across the thread boundary.
type PartyTally = (u64, u64, u64, u64, u64, u64, Vec<(String, i64)>);

/// Execute `prog` with two parties on two threads over `mk(party)`'s transport.
///
/// `inputs` is supplied to both parties, but [`DistEngine::input`] reads a value
/// only when the party owns it, so neither party's secrets are used by the other.
/// The seed is shared: it is the dealer's stream (see the module docs).
pub fn run_two_party<T, F>(
    prog: &super::ir::Program,
    sched: &super::schedule::Schedule,
    inputs: &std::collections::HashMap<String, i64>,
    cfg: RunConfig,
    mk: F,
) -> Result<DistMeasured, String>
where
    T: Transport + 'static,
    F: Fn(usize) -> T,
{
    let nets: Vec<Arc<T>> = (0..2).map(|i| Arc::new(mk(i))).collect();
    let mut joins = Vec::with_capacity(2);
    for (me, net) in nets.into_iter().enumerate() {
        let prog = prog.clone();
        let sched = sched.clone();
        let inputs = inputs.clone();
        joins.push(std::thread::spawn(move || -> Result<_, String> {
            let mut e = DistEngine::new(me, net, cfg.bits, cfg.seed)?
                .with_hoisted_bits(cfg.hoist_bits)
                .with_trunc_pairs(cfg.trunc_pairs)
                .with_bilinear(cfg.bilinear);
            // The readiness barrier keeps connection setup out of the timing.
            e.net.wait_ready(30_000);
            let t0 = std::time::Instant::now();
            let m = super::vm::execute(&prog, &sched, &mut e, "dist", &inputs)?;
            let elapsed = t0.elapsed().as_secs_f64();
            let c = e.counters();
            Ok((
                elapsed,
                c.online_bytes,
                c.muls,
                c.opens,
                c.triples,
                c.random_bits,
                e.seq,
                m.outputs,
            ))
        }));
    }
    let mut per_party = Vec::with_capacity(2);
    let mut best: Option<PartyTally> = None;
    for j in joins {
        let (elapsed, bytes, muls, opens, triples, bits_n, seq, outs) = j
            .join()
            .map_err(|_| "dist: a party thread panicked".to_string())??;
        per_party.push(elapsed);
        // Keep the busiest party's counters, matching the convention elsewhere:
        // a deployment pays what its heaviest link carries.
        let take = best.as_ref().map(|b| bytes > b.0).unwrap_or(true);
        if take {
            best = Some((bytes, muls, opens, triples, bits_n, seq, outs));
        }
    }
    let (bytes, muls, opens, triples, rbits, seq, outputs) =
        best.ok_or("dist: no party completed")?;
    Ok(DistMeasured {
        elapsed_secs: per_party.iter().cloned().fold(0.0, f64::max),
        per_party_secs: per_party,
        online_bytes: bytes,
        muls,
        opens,
        triples,
        random_bits: rbits,
        exchanges: seq,
        outputs,
    })
}

#[cfg(test)]
mod tests {

    /// Fusing a level into a bilinear form must not change what the program computes.
    ///
    /// The isolated test above exercises the primitive on a hand-built form. This one
    /// exercises the *automatic* path: a real compiled convolutional network, where
    /// `vm::multiply_level` has to recover the structure from operand registers that the
    /// scalar lowering left behind, and decide per level whether to fuse. A bug in that
    /// recovery — a mis-indexed operand pool, a level fused when it should not be —
    /// would show up here as a wrong answer and nowhere else.
    #[test]
    fn fusing_a_convolution_does_not_change_what_it_computes() {
        use crate::bench::conv_nn::{debug_case, Cnn};
        use crate::ml::Activation;
        use crate::mpcc::{compile, schedule, LowerConfig};
        use std::sync::Mutex;

        let net = Cnn::tiny(8, Activation::Square);
        let (src, ins) = debug_case(&net, 0);
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let prog = compile(&src, &cfg).expect("compiles");
        let sched = schedule(&prog);

        let mut seen = Vec::new();
        for bilinear in [false, true] {
            let hs = Mutex::new(crate::crypto::net::InprocHandle::create(2));
            let m = run_two_party(
                &prog,
                &sched,
                &ins,
                RunConfig {
                    bits: 64,
                    seed: 0x5EED,
                    hoist_bits: true,
                    trunc_pairs: true,
                    bilinear,
                },
                |i| hs.lock().unwrap()[i].clone(),
            )
            .expect("runs");
            seen.push((m.outputs, m.online_bytes, m.exchanges));
        }
        let (beaver_out, beaver_bytes, beaver_exch) = &seen[0];
        let (fused_out, fused_bytes, fused_exch) = &seen[1];

        assert_eq!(
            beaver_out, fused_out,
            "the fusion changed the answer: {beaver_out:?} against {fused_out:?}"
        );
        // Fusion is a traffic optimisation and must not cost a round.
        assert_eq!(
            beaver_exch, fused_exch,
            "fusion should not change the round count"
        );
        // On this deliberately tiny network the saving may be small or absent — the
        // threshold exists precisely so that a level without reuse is left alone — so the
        // claim here is only that fusion never makes the traffic *worse*.
        assert!(
            fused_bytes <= beaver_bytes,
            "fusion increased traffic: {fused_bytes} against {beaver_bytes}"
        );
    }

    /// A bilinear form must give the same answer either way, and cost far less one way.
    ///
    /// The shape is a convolution's: a small kernel multiplied against a feature map at
    /// every position, so each operand is reused many times. That reuse is the entire
    /// source of the saving, and a test on unrelated products would show none — which is
    /// the point, and why the assertion below is on the *ratio* and not a constant.
    #[test]
    fn opening_each_operand_once_is_cheaper_and_gives_the_same_answer() {
        use crate::mpcc::vm::{bilinear_batch, Modulus};
        use std::sync::Mutex;

        // A 16x16 map and a 3x3 kernel: 196 output positions, 1764 products, but only
        // 256 + 9 = 265 operand values.
        let (h, w, k) = (16usize, 16usize, 3usize);
        let (oh, ow) = (h - k + 1, w - k + 1);
        let terms: Vec<Vec<(u32, u32)>> = (0..oh * ow)
            .map(|p| {
                let (y, x) = (p / ow, p % ow);
                (0..k * k)
                    .map(|t| {
                        let (a, b) = (t / k, t % k);
                        (((y + a) * w + (x + b)) as u32, t as u32)
                    })
                    .collect()
            })
            .collect();
        let n_prod: usize = terms.iter().map(|t| t.len()).sum();
        assert_eq!(n_prod, oh * ow * k * k);

        let map: Vec<i64> = (0..h * w).map(|i| (i as i64 % 17) - 8).collect();
        let ker: Vec<i64> = (0..k * k).map(|i| (i as i64 % 5) - 2).collect();

        // The plaintext reference, computed directly from the definition.
        let want: Vec<i64> = terms
            .iter()
            .map(|t| {
                t.iter()
                    .map(|&(i, j)| map[i as usize] * ker[j as usize])
                    .sum()
            })
            .collect();

        let mut seen: Vec<(Vec<i64>, u64, u64)> = Vec::new();
        for bilinear in [false, true] {
            let hs = Mutex::new(crate::crypto::net::InprocHandle::create(2));
            let nets: Vec<_> = (0..2)
                .map(|i| Arc::new(hs.lock().unwrap()[i].clone()))
                .collect();
            let mut joins = Vec::new();
            let terms_c = terms.clone();
            let map_c = map.clone();
            let ker_c = ker.clone();
            for (me, net) in nets.into_iter().enumerate() {
                let terms = terms_c.clone();
                let map = map_c.clone();
                let ker = ker_c.clone();
                joins.push(std::thread::spawn(move || -> Result<_, String> {
                    let mut e = DistEngine::new(me, net, 64, 0xB111)?.with_bilinear(bilinear);
                    e.net.wait_ready(30_000);
                    // Party 0 owns the map, party 1 the kernel: a real two-party split.
                    let xs =
                        e.input_batch(&map.iter().map(|v| (*v as u64, 0usize)).collect::<Vec<_>>());
                    let ys =
                        e.input_batch(&ker.iter().map(|v| (*v as u64, 1usize)).collect::<Vec<_>>());
                    let out = bilinear_batch(&mut e, &xs, &ys, &terms);
                    let opened = e.open_batch(&out);
                    let c = e.counters();
                    Ok((opened, c.online_bytes, c.opens))
                }));
            }
            let mut got = None;
            for j in joins {
                let r = j.join().map_err(|_| "panicked").unwrap().unwrap();
                got = Some(r);
            }
            let (opened, bytes, opens) = got.unwrap();
            let m = Modulus::Ring { k: 64 };
            seen.push((
                opened.iter().map(|v| m.to_i64(*v, 64)).collect(),
                bytes,
                opens,
            ));
        }

        let (scalar_out, scalar_bytes, scalar_opens) = &seen[0];
        let (bilin_out, bilin_bytes, bilin_opens) = &seen[1];

        // Same answer, and both right.
        assert_eq!(scalar_out, &want, "scalar Beaver route");
        assert_eq!(bilin_out, &want, "operand-opening route");

        // And the traffic collapses.
        //
        // Scalar Beaver sends two masked values per product — 1764 products, so 3528
        // elements — while the bilinear route sends the 265 operand values once. Both
        // routes additionally share the operands in and open the results out, which is
        // why the ratio is a little under the ratio of products to operands rather than
        // equal to it.
        //
        // The assertion is on the *ratio* rather than a constant, because the saving is
        // entirely a function of operand reuse: a form over unrelated products would show
        // none at all, and hard-coding a byte count would hide that.
        assert!(
            *bilin_bytes * 3 < *scalar_bytes,
            "expected far fewer bytes: {bilin_bytes} against {scalar_bytes} \
             ({scalar_opens} / {bilin_opens} explicit openings)"
        );
    }
    use super::*;
    use crate::bench::comm::{LatencyTransport, NetworkProfile};
    use crate::crypto::net::InprocHandle;
    use crate::mpcc::{compile, schedule, LowerConfig};
    use std::collections::HashMap;

    fn bind(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn run(src: &str, ins: &HashMap<String, i64>, profile: NetworkProfile) -> DistMeasured {
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let prog = compile(src, &cfg).expect("compile");
        let sched = schedule(&prog);
        let handles = InprocHandle::create(2);
        let shared: std::sync::Mutex<Vec<InprocHandle>> = std::sync::Mutex::new(handles);
        run_two_party(&prog, &sched, ins, RunConfig::default(), |i| {
            let h = shared.lock().unwrap()[i].clone();
            LatencyTransport::new(h, profile, 2)
        })
        .expect("distributed run")
    }

    /// **The gate.** A distributed run must compute exactly what the in-process
    /// engine computes; otherwise its timing describes a different program.
    #[test]
    fn a_distributed_run_agrees_with_the_in_process_engine() {
        let cases: &[(&str, &[(&str, i64)])] = &[
            ("(+ (* x y) z)", &[("x", 7), ("y", -3), ("z", 5)]),
            ("(* x (> x 0))", &[("x", -9)]),
            ("(* x (> x 0))", &[("x", 12)]),
            ("(ash-right (* x y) 8)", &[("x", -300), ("y", 7)]),
            ("(== x y)", &[("x", 5), ("y", 5)]),
            (
                "(let ((m (> x y))) (+ (* x m) (* y (- 1 m))))",
                &[("x", -9), ("y", -4)],
            ),
        ];
        for (src, pairs) in cases {
            let ins = bind(pairs);
            let d = run(src, &ins, NetworkProfile::INSTANT);
            let cfg = LowerConfig {
                parties: 2,
                bit_width: 32,
                ..Default::default()
            };
            let prog = compile(src, &cfg).unwrap();
            let sched = schedule(&prog);
            let m = crate::mpcc::run_protocol("dealer", 2, &prog, &sched, &ins).unwrap();
            assert_eq!(
                d.outputs, m.outputs,
                "{src}: distributed {:?} != in-process {:?}",
                d.outputs, m.outputs
            );
        }
    }

    /// Latency must actually be paid. With a one-way delay of `L` and `R`
    /// exchanges on the critical path, elapsed time must be at least `R*L` —
    /// which is the property that makes this a measurement rather than a model.
    #[test]
    fn elapsed_time_really_includes_the_link() {
        let ins = bind(&[("x", -9)]);
        let fast = run("(* x (> x 0))", &ins, NetworkProfile::INSTANT);
        let slow = run(
            "(* x (> x 0))",
            &ins,
            NetworkProfile::fixed_latency(std::time::Duration::from_millis(2)),
        );
        assert_eq!(fast.outputs, slow.outputs, "the answer must not change");
        let floor = slow.exchanges as f64 * 0.002;
        assert!(
            slow.elapsed_secs >= floor * 0.8,
            "with {} exchanges at 2 ms one-way, elapsed {:.3} s should be near or \
             above {:.3} s — if it is not, the transport is not waiting",
            slow.exchanges,
            slow.elapsed_secs,
            floor
        );
        assert!(
            slow.elapsed_secs > fast.elapsed_secs,
            "adding latency must cost time: {:.4} vs {:.4}",
            slow.elapsed_secs,
            fast.elapsed_secs
        );
    }

    /// One exchange per interactive level, whatever its width: a batch of
    /// independent comparisons must not cost more exchanges than one.
    #[test]
    fn a_batch_costs_the_exchanges_of_one() {
        let one = run(
            "(> a b)",
            &bind(&[("a", 1), ("b", 2)]),
            NetworkProfile::INSTANT,
        );
        let four = run(
            "(+ (+ (> a b) (> c d)) (+ (> e f) (> g h)))",
            &bind(&[
                ("a", 1),
                ("b", 2),
                ("c", 3),
                ("d", 1),
                ("e", 5),
                ("f", 9),
                ("g", 2),
                ("h", 2),
            ]),
            NetworkProfile::INSTANT,
        );
        assert_eq!(
            one.exchanges, four.exchanges,
            "four independent comparisons cost {} exchanges against one's {}",
            four.exchanges, one.exchanges
        );
    }
}
