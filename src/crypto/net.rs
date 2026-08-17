use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

pub type Message = Vec<u8>;
pub type PartyId = usize;

/// Minimal transport trait used by MPC protocols in this crate.
/// Implementations may be in-process (testing) or real networked.
pub trait Transport: Send + Sync {
    /// id of this participant (0..n-1)
    fn id(&self) -> PartyId;
    /// send a message to a specific party (non-blocking)
    fn send(&self, to: PartyId, msg: Message);
    /// broadcast a message to all other parties (non-blocking)
    fn broadcast(&self, msg: Message);
    /// receive the next message addressed to this participant, if any.
    /// Non-blocking: returns None if no message is available.
    fn recv(&self) -> Option<(PartyId, Message)>;

    /// Block until this participant has a live connection to every peer (in
    /// both directions), or until `timeout_ms` elapses.  Returns `true` on full
    /// connectivity.  This is the readiness barrier that lets protocols start
    /// without startup `sleep`s.  In-process transports are always ready.
    fn wait_ready(&self, _timeout_ms: u64) -> bool {
        true
    }
}

/// Simple MPC message types for simulator usage.
///
/// Each variant that belongs to a specific in-flight operation carries an
/// `op_id` so that concurrent operations (e.g., multiple multiplications in
/// the same expression) can be correctly demultiplexed.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MpcMsg {
    // ---- legacy untagged (kept for tcp_party demo) ----
    D(u128),
    E(u128),
    PublicD(u128),
    PublicE(u128),

    // ---- tagged protocol messages ----
    /// One share component sent during input sharing.
    /// `share_id` identifies the share; `value` is the field element for the
    /// receiving party.
    InputShare {
        share_id: u64,
        value: u128,
    },

    /// A whole *layer* of input sharings in one message — the batched form of
    /// [`MpcMsg::InputShare`]. `lefts[k]`/`rights[k]` are the two replicated
    /// components for share `base_id + k`.
    ///
    /// The unbatched variant costs a blocking round trip **per element**, and
    /// a scheme switch imports thousands of activations at once; batching does
    /// not change the protocol, the randomness, or the bytes on the wire —
    /// only how many round trips carry them.
    InputShareBatch {
        base_id: u64,
        lefts: Vec<u128>,
        rights: Vec<u128>,
    },

    /// One party's additive share component sent during a reveal round.
    RevealShare {
        share_id: u64,
        value: u128,
    },

    /// A whole batch of reveal contributions in one message — the batched form
    /// of [`MpcMsg::RevealShare`], used by the malicious layer.
    ///
    /// Bit decomposition attaches a MAC to every leaf, and each attachment
    /// needs one opening. Unbatched that would be one round *per leaf* —
    /// thousands for a single layer — which would dwarf the protocol it is
    /// protecting. Batched it is one round for the whole decomposition.
    RevealShareBatch {
        base_id: u64,
        values: Vec<u128>,
    },

    /// One party's accumulated pairwise transcript hash, exchanged once at the
    /// deferred check point (`crate::crypto::aby3_verify`).
    ///
    /// `toward_next` says which of the sender's two transcripts this is: the one
    /// it keeps with its successor (`true`) or its predecessor (`false`). The
    /// receiver compares it against the matching transcript of its own.
    ViewHash {
        toward_next: bool,
        digest: [u8; 32],
    },

    /// Commitment to this party's contribution to the verification coin flip.
    /// Sent by every party before any seed is opened, so the public challenge
    /// coefficients cannot be biased by whoever speaks last.
    CoinCommit {
        digest: [u8; 32],
    },

    /// Opening of a [`MpcMsg::CoinCommit`].
    CoinOpen {
        seed: [u8; 32],
    },

    /// One-time PRF key handshake. Party `i` samples `k_i` and sends it to
    /// party `i+1`, so each adjacent pair ends up sharing a key that the
    /// third party does not know. Those keys generate the zero-sum
    /// correlated randomness ABY3 multiplication needs, with no dealer and
    /// no per-multiplication communication.
    PrfKey {
        key: [u8; 32],
    },

    /// Resharing values for a batch of ABY3 multiplications: party `i` sends
    /// its `z_i` components to party `i−1`. One ring element per party per
    /// multiplication — replacing `TripleBatch` (6 elements dealt per
    /// multiplication) plus `TaggedDEBatch` (2 opened).
    TaggedZBatch {
        base_op_id: u64,
        /// Width each value is packed at. The protocol works modulo `2^bits`,
        /// so anything wider is padding — and for the bit-level multiplications
        /// inside `a2b` the honest width is **1**, not 128.
        bits: u32,
        data: Vec<u8>,
    },

    /// Resharing values for a batch of multiplications over the small prime
    /// field `Z_p` (`crate::crypto::fp_protocol`). Same one-directed-message
    /// pattern as [`MpcMsg::TaggedZBatch`], but **one byte per element** rather
    /// than sixteen: Falcon's `p` is 37 and this crate's is 67, so a field
    /// element is a byte.
    ///
    /// That width is not a micro-optimisation — Falcon's headline advantage over
    /// prior work is communication volume, so carrying its field traffic in
    /// `u128` slots would overstate its cost by 16× and make any comparison
    /// against it meaningless.
    TaggedFpBatch {
        base_op_id: u64,
        values: Vec<u8>,
    },

    /// A batched reveal whose payload is packed at the share's **actual** width
    /// rather than padded to `u128`.
    ///
    /// [`MpcMsg::RevealShareBatch`] carries `Vec<u128>` — 16 bytes per element
    /// whatever `nbits` is, so at `nbits = 20` it is 84% padding and at `ℓ = 32`
    /// still 75%. That is invisible in a protocol whose traffic is dominated by
    /// something else, and badly misleading in one whose traffic *is* the
    /// reveals. Falcon's wrap3 opens one ring value per DReLU, so the width has
    /// to be honest for the comparison to mean anything.
    RevealPackedBatch {
        base_id: u64,
        bits: u32,
        data: Vec<u8>,
    },

    /// One garbler's contribution in the 3-party (two-garbler) garbled-circuit
    /// protocol (`crate::crypto::gc_protocol`). Both garblers derive an
    /// identical circuit from their shared PRF key; garbler 0 ships the garbled
    /// tables plus the active labels for its input wires, garbler 1 ships the
    /// active labels for its wires, and the evaluator combines both. `op_id`
    /// demultiplexes concurrent conversions; `garbler` (0 or 1) says which
    /// contribution this is. No oblivious transfer: every input wire's value is
    /// known to the garbler that sends its label.
    GcFromGarbler {
        op_id: u64,
        garbler: usize,
        payload: Vec<u8>,
    },
}

/// A party's two pairwise PRF keys: `mine` is shared with the next party,
/// `prev` with the previous one.
#[derive(Clone, Copy, Debug)]
pub struct PrfKeys {
    pub mine: [u8; 32],
    pub prev: [u8; 32],
}

impl PrfKeys {
    /// Zero-sum correlated randomness for operation `ctr`:
    /// `αᵢ = F(kᵢ, ctr) − F(kᵢ₋₁, ctr)`. Summed over the three parties each
    /// PRF output appears once positively and once negatively, so
    /// `α₀+α₁+α₂ = 0` — while no single party can predict any other's α.
    pub fn alpha(&self, ctr: u64) -> u128 {
        prf(&self.mine, ctr).wrapping_sub(prf(&self.prev, ctr))
    }

    /// The same zero-sum correlated randomness, but in `Z_p` — what a
    /// replicated multiplication over a *prime field* needs
    /// (`crate::crypto::fp_protocol`, for Falcon's private compare).
    ///
    /// [`Self::alpha`] cannot be reused by reduction: three values summing to
    /// zero mod `2^128` do **not** sum to zero mod `p`. What carries over is the
    /// *construction* — the telescoping difference `F(kᵢ) − F(kᵢ₋₁)` sums to
    /// zero in any group, so the subtraction is simply done in the field.
    ///
    /// The reduction of a uniform `u128` mod `p` is biased by about `2^-121`,
    /// which is negligible and is the same bias the rest of the crate accepts
    /// when deriving small values from a PRF.
    pub fn alpha_mod(&self, ctr: u64, p: u64) -> u64 {
        let a = (prf(&self.mine, ctr) % p as u128) as u64;
        let b = (prf(&self.prev, ctr) % p as u128) as u64;
        if a >= b {
            a - b
        } else {
            a + p - b
        }
    }

    /// A replicated sharing of a **fresh uniform value nobody knows**, for
    /// counter `ctr`, with *no communication*.
    ///
    /// The same pairwise keys that make [`Self::alpha`] free also make this
    /// free, and it is the reason the malicious upgrade needs no dealer and no
    /// extra round to obtain its MAC key.
    ///
    /// The component assignment is forced by what each key is shared with.
    /// `prev` is the key party `i` shares with `i−1`, so `F(prev, ctr)` is known
    /// to exactly `{P_{i−1}, P_i}` — which is precisely the pair that must hold
    /// additive component `i` under replicated sharing (`P_i` as `left`,
    /// `P_{i−1}` as `right`). Symmetrically `F(mine, ctr)` is component `i+1`,
    /// this party's `right`. So
    ///
    /// ```text
    /// rᵢ = F(k_{i−1,i}, ctr)      known to P_{i−1}, P_i
    /// ```
    ///
    /// is a consistent replicated sharing across all three parties, and no
    /// single party can predict the component it does not hold.
    ///
    /// `ctr` must not collide with counters used for [`Self::alpha`]; callers
    /// take their counters from disjoint domains (see
    /// `aby3_verify::VERIFY_CTR_BASE`).
    pub fn random_share(&self, ctr: u64) -> (u128, u128) {
        (prf(&self.prev, ctr), prf(&self.mine, ctr))
    }

    /// [`Self::random_share`] in `Z_p`.
    ///
    /// Note the contrast with [`Self::alpha_mod`], which is easy to conflate:
    /// that one is a *difference* of the two PRFs, because it must sum to zero
    /// across parties; this one is the two PRFs themselves, because it must
    /// reconstruct to a uniform value nobody knows. Using the zero-sum form here
    /// would produce a sharing of zero.
    pub fn random_share_mod(&self, ctr: u64, p: u64) -> (u64, u64) {
        (
            (prf(&self.prev, ctr) % p as u128) as u64,
            (prf(&self.mine, ctr) % p as u128) as u64,
        )
    }
}

/// Keyed PRF: the low 16 bytes of `SHA-256(key ‖ ctr)`.
fn prf(key: &[u8; 32], ctr: u64) -> u128 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(key);
    h.update(ctr.to_le_bytes());
    let d = h.finalize();
    let mut b = [0u8; 16];
    b.copy_from_slice(&d[..16]);
    u128::from_le_bytes(b)
}

/// Serialize a message using bincode.
pub fn serialize_msg<T: Serialize>(msg: &T) -> Message {
    bincode::serialize(msg).expect("serialize_msg failed")
}

/// Deserialize a message using bincode.
pub fn deserialize_msg<T: for<'de> serde::Deserialize<'de>>(bytes: &Message) -> T {
    bincode::deserialize(bytes).expect("deserialize_msg failed")
}

/// [`deserialize_msg`] for bytes that came from a peer, which may be malformed.
///
/// The panicking form is fine for a party's own round trips and wrong for
/// anything a peer chose: a corrupt party could otherwise crash an honest one
/// with a single malformed payload. A crash is not the abort this protocol
/// promises — it bypasses the deferred check that is meant to be the only place a
/// run can stop, and it poisons the mutexes the surviving threads hold.
pub fn try_deserialize_msg<T: for<'de> serde::Deserialize<'de>>(bytes: &Message) -> Option<T> {
    bincode::deserialize(bytes).ok()
}

/// In-process network: simple shared queues per party. Not secure; for tests.
struct InprocNetwork {
    queues: Vec<Mutex<VecDeque<(PartyId, Message)>>>,
}

impl InprocNetwork {
    fn new(nparties: usize) -> Self {
        let mut queues = Vec::with_capacity(nparties);
        for _ in 0..nparties {
            queues.push(Mutex::new(VecDeque::new()));
        }
        InprocNetwork { queues }
    }
}

/// A per-party handle to an `InprocNetwork` that implements `Transport`.
#[derive(Clone)]
pub struct InprocHandle {
    id: PartyId,
    net: Arc<InprocNetwork>,
    nparties: usize,
}

impl InprocHandle {
    /// Create `nparties` handles connected to the same in-process network.
    pub fn create(nparties: usize) -> Vec<InprocHandle> {
        let net = Arc::new(InprocNetwork::new(nparties));
        (0..nparties)
            .map(|id| InprocHandle {
                id,
                net: net.clone(),
                nparties,
            })
            .collect()
    }
}

impl Transport for InprocHandle {
    fn id(&self) -> PartyId {
        self.id
    }

    fn send(&self, to: PartyId, msg: Message) {
        if to >= self.nparties {
            return;
        }
        if let Ok(mut q) = self.net.queues[to].lock() {
            q.push_back((self.id, msg));
        }
    }

    fn broadcast(&self, msg: Message) {
        for p in 0..self.nparties {
            if p == self.id {
                continue;
            }
            if let Ok(mut q) = self.net.queues[p].lock() {
                q.push_back((self.id, msg.clone()));
            }
        }
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        if let Ok(mut q) = self.net.queues[self.id].lock() {
            q.pop_front()
        } else {
            None
        }
    }
}

/// Blocking helpers that poll `recv()` in a spin-sleep loop until a message
/// matching a predicate arrives. These are used by protocol helpers that need
/// to wait for a peer's contribution before proceeding.
pub fn recv_blocking<T: Transport>(t: &T, timeout_ms: u64) -> Option<(PartyId, Message)> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    loop {
        if let Some(m) = t.recv() {
            return Some(m);
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// Block indefinitely until a message arrives.  Use this only for
/// the outer wait between REPL expressions so parties 1/2 don't
/// time out while the human is still typing.
pub fn recv_forever<T: Transport>(t: &T) -> (PartyId, Message) {
    loop {
        if let Some(m) = t.recv() {
            return m;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// A buffering demultiplexer layered over a `Transport`.
///
/// Protocol steps routinely need a *specific* message (identified by op/share
/// id) while messages belonging to *later* steps may arrive first — a plain TCP
/// send carries no cross-message ordering guarantee.  The previous receive
/// helpers polled the transport directly and silently **dropped** any message
/// that did not match what they were currently waiting for; a message that
/// raced ahead of its consumer was lost, and the later step blocked until it
/// timed out.  (The `sleep` hacks existed to bias arrival order so this would
/// happen less often.)
///
/// `Mailbox` fixes the root cause: `recv_matching` returns the first message
/// satisfying the caller's predicate and **buffers every other message** so it
/// stays available to a subsequent step.  The buffer persists across calls, so
/// the protocols become order-independent and no timing hacks are required.
pub struct Mailbox<T: Transport> {
    transport: T,
    /// Pairwise PRF keys, established lazily on first use (see
    /// `prf_keys`). `Mutex` for interior mutability behind `&self`.
    prf: Mutex<Option<PrfKeys>>,
    // `Mutex` (rather than `RefCell`) so `Mailbox`/`TransportBackend` stay
    // `Send + Sync` as the `Backend` trait requires; a party drives its mailbox
    // single-threaded, so the lock is uncontended.
    buf: Mutex<VecDeque<(PartyId, MpcMsg)>>,
    /// Times this party blocked on the wire for a message it did not already
    /// hold, and times a `recv_matching` was served from the buffer instead.
    ///
    /// The first is the quantity that predicts wall clock under the latency
    /// model of [`crate::bench::comm`], which charges a one-way delay per
    /// message in send order: what costs time is not how many messages a party
    /// sends, nor how many backend primitives it calls, but how many times it
    /// stops and waits. Neither of the other two counters is a substitute, and
    /// on the compiled-versus-hand-written comparison they disagree — one route
    /// makes half the primitive calls and sends 0.61x the messages while taking
    /// the same wall clock, which is only explicable in terms of waits.
    ///
    /// # It is almost, but not quite, deterministic
    ///
    /// A receive counts as a wait only when it actually polls the transport; one
    /// whose message has already arrived is served from `buf` and counts as a
    /// hit instead. So the figure is *the number of times a party had to stop*,
    /// which depends slightly on how the three parties' threads interleave.
    ///
    /// Measured on `bench-compiled-nn` at `16/8/4` over eight runs spanning two
    /// round-trip times:
    ///
    /// | route | observed |
    /// |---|---|
    /// | hand-written gadgets | 32–33 |
    /// | hand-written, gc cmp | 11–12 |
    /// | + ASAP levelling | 43 (stable) |
    /// | + CSE | 42–43 |
    /// | + reassociation | 33 (stable) |
    /// | + gc comparisons | 11–12 |
    ///
    /// So four of six rows move by one and two do not, and the network profile
    /// perturbs the interleaving as much as repetition does. **Quote waits with a
    /// ±1 and never build an argument on a difference of one count.**
    ///
    /// We got this wrong twice, in the same direction each time. First, two equal
    /// readings of 33 were written up as "exact parity" between the compiler and a
    /// hand-written expert. Then, after four repetitions at one RTT, the caveat was
    /// narrowed to "only the a2b route is unstable" — which the next run, at a
    /// different RTT, contradicted. The temptation both times was to quote the
    /// equality that had happened to be observed.
    ///
    /// What survives is everything resting on a factor rather than a count: 136x
    /// for levelling, ~2.8x for the garbled-circuit assignment, 214x for the
    /// fitted local-cost constant.
    ///
    /// [`Self::buffer_hits`] is the other half of the pair, and a route whose
    /// waits jitter is one whose hits jitter by the same amount.
    waits: std::sync::atomic::AtomicU64,
    hits: std::sync::atomic::AtomicU64,
}

impl<T: Transport> Mailbox<T> {
    /// Wrap a transport in a fresh (empty) mailbox.
    pub fn new(transport: T) -> Self {
        Mailbox {
            transport,
            prf: Mutex::new(None),
            buf: Mutex::new(VecDeque::new()),
            waits: std::sync::atomic::AtomicU64::new(0),
            hits: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Times this party blocked on the wire — the round count that predicts
    /// wall clock. See the field's documentation.
    pub fn sync_waits(&self) -> u64 {
        self.waits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Times a receive was satisfied from the buffer, costing nothing.
    ///
    /// Reported beside [`Mailbox::sync_waits`] because a high ratio of hits to
    /// waits is what a well-batched protocol looks like: the messages of one
    /// round arrive together and are consumed without further blocking.
    pub fn buffer_hits(&self) -> u64 {
        self.hits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// This party's pairwise PRF keys, performing the one-time handshake on
    /// first call. `mine` is shared with party `me+1`, `prev` with `me−1`;
    /// no party holds all three, which is what makes the derived randomness
    /// sum to zero without revealing it.
    pub fn prf_keys(&self, me: PartyId) -> PrfKeys {
        {
            let held = self.prf.lock().expect("prf key lock poisoned");
            if let Some(k) = *held {
                return k;
            }
        }
        let mut mine = [0u8; 32];
        rand::Rng::fill(&mut rand::thread_rng(), &mut mine);
        self.send((me + 1) % 3, serialize_msg(&MpcMsg::PrfKey { key: mine }));
        // Only the predecessor may supply this key. Accepting it from anyone let a
        // successor race its own key in, after which it would know *both* of this
        // party's pairwise keys and could predict every zero-sum masking term the
        // party contributes for the rest of the session.
        let prev_party = (me + 2) % 3;
        let (_, msg) = self
            .recv_matching_from(prev_party, 30_000, |m| matches!(m, MpcMsg::PrfKey { .. }))
            .expect("prf key handshake timed out");
        let prev = match msg {
            MpcMsg::PrfKey { key } => key,
            _ => unreachable!("predicate guarantees PrfKey"),
        };
        let keys = PrfKeys { mine, prev };
        *self.prf.lock().expect("prf key lock poisoned") = Some(keys);
        keys
    }

    /// This participant's id.
    pub fn id(&self) -> PartyId {
        self.transport.id()
    }

    /// Borrow the underlying transport (e.g. for raw sends).
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Send a message to a specific party.
    pub fn send(&self, to: PartyId, msg: Message) {
        self.transport.send(to, msg);
    }

    /// Broadcast a message to all other parties.
    pub fn broadcast(&self, msg: Message) {
        self.transport.broadcast(msg);
    }

    /// Receive the next message **from party `from`** satisfying `pred`.
    ///
    /// # Why the sender has to be part of the predicate
    ///
    /// [`Self::recv_matching`] matches on message *content* only. Every protocol
    /// step that expects a particular party to speak must therefore say so, or a
    /// different party can answer in its place — the messages carry no binding to
    /// their sender, so nothing downstream notices.
    ///
    /// That is not hypothetical. Input sharing waits for two `InputShare`
    /// messages carrying the right `share_id` and takes them in arrival order; a
    /// party that is *not* the owner can race one in, and the receiver adopts an
    /// adversary-chosen component. The three components then no longer sum to the
    /// owner's value, and nothing detects it — the MACs are computed over the
    /// corrupted share, so the run is consistent and wrong. The same omission let
    /// any peer supply the pairwise PRF key that should come from a party's
    /// predecessor, handing the sender both of that party's keys and with them
    /// every masking term it will ever contribute.
    ///
    /// Use this wherever the protocol names the sender; reserve
    /// [`Self::recv_matching`] for the genuinely sender-agnostic case.
    pub fn recv_matching_from<F>(
        &self,
        from: PartyId,
        timeout_ms: u64,
        pred: F,
    ) -> Option<(PartyId, MpcMsg)>
    where
        F: Fn(&MpcMsg) -> bool,
    {
        self.recv_where(timeout_ms, |sender, m| sender == from && pred(m))
    }

    /// Receive the next message satisfying `pred`, buffering any others so they
    /// remain available to later steps.  Blocks up to `timeout_ms`; returns
    /// `None` only on timeout.
    ///
    /// **This does not constrain the sender.** Prefer
    /// [`Self::recv_matching_from`] whenever the protocol knows who should speak.
    pub fn recv_matching<F>(&self, timeout_ms: u64, pred: F) -> Option<(PartyId, MpcMsg)>
    where
        F: Fn(&MpcMsg) -> bool,
    {
        self.recv_where(timeout_ms, |_, m| pred(m))
    }

    /// Receive the next message whose **sender and content** both satisfy `pred`.
    ///
    /// The general form. Use it where the set of acceptable senders is not a single
    /// party — most importantly when collecting one contribution from each peer,
    /// where the requirement is that the contributions come from *distinct*
    /// parties. Accepting two from the same party lets one peer supply another's
    /// share of a reveal, or both halves of a coin flip.
    pub fn recv_where_from<F>(&self, timeout_ms: u64, pred: F) -> Option<(PartyId, MpcMsg)>
    where
        F: Fn(PartyId, &MpcMsg) -> bool,
    {
        self.recv_where(timeout_ms, pred)
    }

    /// The shared implementation: a predicate over `(sender, message)`.
    ///
    /// Non-matching messages are buffered rather than dropped, which is what lets
    /// a sender-bound wait coexist with out-of-order arrivals — a message from the
    /// wrong party is kept for whoever *is* waiting for it instead of being
    /// discarded.
    fn recv_where<F>(&self, timeout_ms: u64, pred: F) -> Option<(PartyId, MpcMsg)>
    where
        F: Fn(PartyId, &MpcMsg) -> bool,
    {
        // 1) Serve from the buffer if a matching message is already held.
        {
            let mut buf = self.buf.lock().expect("mailbox buffer poisoned");
            if let Some(pos) = buf.iter().position(|(f, m)| pred(*f, m)) {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return buf.remove(pos);
            }
        }
        // 2) Poll the transport until a match arrives or we time out. Reaching
        //    here is a synchronization point: the party has nothing to do until
        //    a peer speaks, so this is what a round costs.
        self.waits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            if let Some((from, bytes)) = self.transport.recv() {
                // A malformed payload is dropped, not fatal. See
                // `try_deserialize_msg`.
                let Some(m) = try_deserialize_msg::<MpcMsg>(&bytes) else {
                    continue;
                };
                if pred(from, &m) {
                    return Some((from, m));
                }
                self.buf
                    .lock()
                    .expect("mailbox buffer poisoned")
                    .push_back((from, m));
            } else if std::time::Instant::now() >= deadline {
                return None;
            } else {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

/// Largest frame a peer may announce, in bytes.
///
/// Chosen against measured traffic rather than picked round. The biggest single
/// message this artifact sends is a batched ring resharing: a `16x20x20 -> 8`
/// convolution issues 373 248 multiplications in one `TaggedZBatch` at 16 bytes
/// each, about 6 MB (`bench-conv`). 64 MiB leaves an order of magnitude over that
/// while keeping one hostile frame far below what would trouble a host, and a peer
/// needing more is malfunctioning whether or not it is malicious.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// TCP wire frames.  `Hello` announces the sender on a freshly-opened
/// persistent connection (used by the readiness barrier); `Msg` carries a
/// payload.  Both are length-prefixed on the wire.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) enum Frame {
    Hello(PartyId),
    Msg(PartyId, Message),
}

use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

/// TCP-based `Transport` using **one persistent, ordered connection per peer**
/// in each direction.
///
/// * Outbound: a lazily-established `TcpStream` per peer, reused for every send,
///   so messages to a given peer are delivered in send order (this closes the
///   reordering hazards the old connect-per-message design had).
/// * Inbound: a listener accepts a persistent connection per peer and a reader
///   thread streams length-prefixed frames from it into the shared inbox.
/// * Readiness: each side sends a `Hello` on connect; `wait_ready` blocks until
///   this participant is connected outbound to all peers and has seen inbound
///   `Hello`s from all peers — removing the need for startup `sleep`s.
pub struct TcpTransport {
    id: PartyId,
    addrs: Vec<String>,
    inbox: Arc<Mutex<VecDeque<(PartyId, Message)>>>,
    /// Persistent outbound stream per peer (index == peer id; own slot unused).
    outbound: Vec<Mutex<Option<TcpStream>>>,
    /// Set of peers whose inbound `Hello` we have received.
    peers_seen: Arc<Mutex<HashSet<PartyId>>>,
}

/// Write a length-prefixed bincode-serialized frame to any writable stream.
/// Generic over the stream type so both plaintext TCP and TLS reuse it.
pub(crate) fn write_frame<S: Write>(s: &mut S, frame: &Frame) -> std::io::Result<()> {
    let bytes = bincode::serialize(frame)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let len = (bytes.len() as u32).to_be_bytes();
    s.write_all(&len)
        .and_then(|_| s.write_all(&bytes))
        .and_then(|_| s.flush())
}

/// Read length-prefixed frames from a peer connection until EOF/error, routing
/// `Hello` into `peers_seen` and `Msg` into `inbox`.  Generic over the stream
/// type so both plaintext TCP and TLS reuse it.
pub(crate) fn read_frames<S: Read>(
    mut s: S,
    inbox: Arc<Mutex<VecDeque<(PartyId, Message)>>>,
    peers_seen: Arc<Mutex<HashSet<PartyId>>>,
) {
    loop {
        let mut lenbuf = [0u8; 4];
        if s.read_exact(&mut lenbuf).is_err() {
            return;
        }
        let len = u32::from_be_bytes(lenbuf) as usize;
        // A peer controls this length, and the allocation below happens before a
        // single payload byte is read. Without a bound, one 4-byte header claiming
        // `u32::MAX` costs the receiver a 4 GiB allocation, and a peer that repeats
        // it exhausts the host — a denial of service needing no valid message and
        // arriving before any authentication.
        if len > MAX_FRAME_BYTES {
            return;
        }
        let mut buf = vec![0u8; len];
        if s.read_exact(&mut buf).is_err() {
            return;
        }
        match bincode::deserialize::<Frame>(&buf) {
            Ok(Frame::Hello(from)) => {
                if let Ok(mut p) = peers_seen.lock() {
                    p.insert(from);
                }
            }
            Ok(Frame::Msg(from, payload)) => {
                if let Ok(mut q) = inbox.lock() {
                    q.push_back((from, payload));
                }
            }
            Err(_) => { /* skip malformed frame */ }
        }
    }
}

impl TcpTransport {
    /// Create a new TcpTransport for participant `id` with a list of addresses
    /// (one address per party, length == nparties). This starts a listener
    /// thread bound to `addrs[id]`.
    pub fn new(id: PartyId, addrs: Vec<String>) -> Self {
        let inbox = Arc::new(Mutex::new(VecDeque::new()));
        let peers_seen = Arc::new(Mutex::new(HashSet::new()));
        let listener_addr = addrs[id].clone();
        let inbox_clone = inbox.clone();
        let peers_clone = peers_seen.clone();

        // Listener: accept persistent connections and stream frames from each.
        thread::spawn(move || {
            if let Ok(listener) = TcpListener::bind(&listener_addr) {
                for stream in listener.incoming().flatten() {
                    let inbox = inbox_clone.clone();
                    let peers = peers_clone.clone();
                    thread::spawn(move || read_frames(stream, inbox, peers));
                }
            }
        });

        let outbound = (0..addrs.len()).map(|_| Mutex::new(None)).collect();
        TcpTransport {
            id,
            addrs,
            inbox,
            outbound,
            peers_seen,
        }
    }

    /// Ensure a persistent outbound connection to `to` exists, establishing it
    /// (with retry + `Hello`) if necessary.  Returns whether we are connected.
    fn ensure_connected(&self, to: PartyId) -> bool {
        let mut slot = match self.outbound[to].lock() {
            Ok(s) => s,
            Err(_) => return false,
        };
        if slot.is_some() {
            return true;
        }
        for _ in 0..8 {
            if let Ok(mut s) = TcpStream::connect(&self.addrs[to]) {
                if write_frame(&mut s, &Frame::Hello(self.id)).is_ok() {
                    *slot = Some(s);
                    return true;
                }
            }
            thread::sleep(std::time::Duration::from_millis(100));
        }
        false
    }
}

impl Transport for TcpTransport {
    fn id(&self) -> PartyId {
        self.id
    }

    fn send(&self, to: PartyId, msg: Message) {
        if to >= self.addrs.len() || to == self.id {
            return;
        }
        if !self.ensure_connected(to) {
            return; // best-effort: peer unreachable
        }
        if let Ok(mut slot) = self.outbound[to].lock() {
            if let Some(s) = slot.as_mut() {
                // On write failure, drop the stream so the next send reconnects.
                if write_frame(s, &Frame::Msg(self.id, msg)).is_err() {
                    *slot = None;
                }
            }
        }
    }

    fn broadcast(&self, msg: Message) {
        for p in 0..self.addrs.len() {
            if p == self.id {
                continue;
            }
            self.send(p, msg.clone());
        }
    }

    fn wait_ready(&self, timeout_ms: u64) -> bool {
        let npeers = self.addrs.len().saturating_sub(1);
        // Establish outbound connections (each sends our Hello).
        for p in 0..self.addrs.len() {
            if p != self.id {
                self.ensure_connected(p);
            }
        }
        // Wait until we have seen inbound Hellos from all peers.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            let seen = self.peers_seen.lock().map(|p| p.len()).unwrap_or(0);
            if seen >= npeers {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        if let Ok(mut q) = self.inbox.lock() {
            q.pop_front()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    /// **A peer cannot make us allocate what it names.**
    ///
    /// The frame header is four attacker-chosen bytes and the buffer is allocated before any
    /// payload arrives, so an unbounded length is a remote out-of-memory needing no valid
    /// message and no authentication. This feeds an oversize header to the real reader and
    /// requires it to give up rather than allocate: a reader that honoured the length would
    /// abort the process or hang, and either fails the test.
    #[test]
    fn an_oversize_frame_header_is_refused_rather_than_allocated() {
        use std::collections::HashSet;
        use std::io::Cursor;
        let mut framed = Vec::new();
        framed.extend_from_slice(&u32::MAX.to_be_bytes()); // 4 GiB if believed
        let inbox = Arc::new(Mutex::new(VecDeque::new()));
        let seen = Arc::new(Mutex::new(HashSet::new()));
        read_frames(Cursor::new(framed), inbox.clone(), seen);
        assert!(
            inbox.lock().unwrap().is_empty(),
            "an oversize frame must deliver nothing"
        );
        // The bound must still admit real traffic: the largest single message measured is a
        // batched resharing of about 6 MB (`bench-conv`). Checked at compile time, so
        // shrinking the cap below what the artifact sends fails the build rather than a run.
        const _: () = assert!(MAX_FRAME_BYTES >= 16 * 1024 * 1024);
    }

    /// **A malformed payload is dropped, not fatal.**
    ///
    /// `deserialize_msg` panics by design and is right to for a party's own round trips. On the
    /// receive path it would let one corrupt peer crash an honest party with a single junk
    /// payload — not the abort this protocol promises, since a crash skips the deferred check
    /// that is meant to be the only place a run stops.
    #[test]
    fn a_malformed_payload_does_not_deserialize_and_does_not_panic() {
        let junk: Message = vec![0xff; 7];
        assert!(
            try_deserialize_msg::<MpcMsg>(&junk).is_none(),
            "junk must fail to parse"
        );
    }

    /// **A receive bound to a sender ignores the same message from anyone else, and keeps it.**
    ///
    /// The property the input-sharing and PRF-handshake fixes rest on. The wrong sender's
    /// message must be *buffered* rather than discarded — a later step may legitimately want it
    /// — and the wait must continue until the named party speaks.
    #[test]
    fn a_sender_bound_receive_ignores_and_keeps_the_wrong_sender() {
        let handles = InprocHandle::create(3);
        let mb0 = Mailbox::new(handles[0].clone());

        // Party 1 races in with the right content but is the wrong sender.
        handles[1].send(0, serialize_msg(&MpcMsg::PrfKey { key: [1u8; 32] }));
        handles[2].send(0, serialize_msg(&MpcMsg::PrfKey { key: [2u8; 32] }));

        let (from, msg) = mb0
            .recv_matching_from(2, 5_000, |m| matches!(m, MpcMsg::PrfKey { .. }))
            .expect("party 2 did speak");
        assert_eq!(from, 2, "must be the party we named");
        match msg {
            MpcMsg::PrfKey { key } => {
                assert_eq!(
                    key, [2u8; 32],
                    "must be that party's value, not the racer's"
                )
            }
            other => panic!("unexpected {other:?}"),
        }

        // Party 1's message is still there — dropping it would turn a sender check into
        // message loss.
        let leftover = mb0
            .recv_matching(1_000, |m| matches!(m, MpcMsg::PrfKey { .. }))
            .expect("the wrong sender's message was buffered, not discarded");
        assert_eq!(leftover.0, 1);
    }
}
