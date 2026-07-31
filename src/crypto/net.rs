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
        values: Vec<u128>,
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
}

impl<T: Transport> Mailbox<T> {
    /// Wrap a transport in a fresh (empty) mailbox.
    pub fn new(transport: T) -> Self {
        Mailbox {
            transport,
            prf: Mutex::new(None),
            buf: Mutex::new(VecDeque::new()),
        }
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
        let (_, msg) = self
            .recv_matching(30_000, |m| matches!(m, MpcMsg::PrfKey { .. }))
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

    /// Receive the next message satisfying `pred`, buffering any others so they
    /// remain available to later steps.  Blocks up to `timeout_ms`; returns
    /// `None` only on timeout.
    pub fn recv_matching<F>(&self, timeout_ms: u64, pred: F) -> Option<(PartyId, MpcMsg)>
    where
        F: Fn(&MpcMsg) -> bool,
    {
        // 1) Serve from the buffer if a matching message is already held.
        {
            let mut buf = self.buf.lock().expect("mailbox buffer poisoned");
            if let Some(pos) = buf.iter().position(|(_, m)| pred(m)) {
                return buf.remove(pos);
            }
        }
        // 2) Poll the transport until a match arrives or we time out.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            if let Some((from, bytes)) = self.transport.recv() {
                let m: MpcMsg = deserialize_msg(&bytes);
                if pred(&m) {
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
    /// thread bound to addrs[id].
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
