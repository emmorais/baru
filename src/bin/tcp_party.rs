//! Multi-party node: an MPC REPL/evaluator plus optional consensus, p2p
//! messaging, and a REST control plane.
//!
//! # Running
//!
//! In-process demo (no arguments) — a handful of example expressions across 3
//! threaded parties sharing an in-process network:
//!
//!   tcp_party
//!
//! TCP mode — one process per party, all given the *same* address roster and
//! distinguished only by the leading party id:
//!
//!   tcp_party 0 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
//!   tcp_party 1 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
//!   tcp_party 2 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
//!
//! Party 0 drives a full REPL (colours, tab-completion, history); parties 1/2
//! are silent workers.  Party 0 broadcasts each expression's text over the
//! distribution transport and every party evaluates it jointly via
//! `TransportBackend`, so all parties run in lockstep.
//!
//! # Transports
//!
//! Each subsystem runs on its own transport, on a port range derived from the
//! MPC base port so they never interfere:
//!
//!   MPC           base           ABY3 protocol traffic   (e.g. 7000–7002)
//!   distribution  base + 100     expression text from party 0
//!   consensus     base + 200     epoch commits           (BARU_CONSENSUS)
//!   messaging     base + 300     chat + forum            (BARU_CHAT)
//!   settlement    base + 400     per-job MPC co-signing  (BARU_SETTLE)
//!
//! In two-argument mode the distribution roster is derived automatically by
//! bumping each port by 100; a fourth CLI argument overrides it explicitly.
//!
//! # Environment toggles
//!
//!   BARU_TLS         use PQ-TLS transports (requires `--features pq-tls`);
//!                    each party mints one long-term Ed25519 identity shared
//!                    across all its channels.
//!   BARU_CONSENSUS   enable a consensus backend on the consensus transport.
//!   BARU_POUW        with consensus on, elect the epoch leader by reputation
//!                    (`ReputationPoUW`) instead of the fixed `SingleLeader`
//!                    baseline (party 0).
//!   BARU_POUW_REP    deterministic genesis reputation as "party:delta,…"
//!                    (every party must set the same value).
//!   BARU_CHAT        enable p2p chat + forum on the messaging transport.
//!   BARU_SETTLE      settle each evaluated MPC expression as a job: parties
//!                    co-sign the result and settle it via `baru::mpc_market`,
//!                    accruing reputation per job (fraud is excluded live).
//!   BARU_REST[=port] serve the per-node REST control plane; party `i` listens
//!                    on `port + i` (default base 8090).  See `baru::rest`.
//!
//! # REPL commands (party 0)
//!
//!   :help                     language reference
//!   :examples [category]      list examples (optionally filtered)
//!   :example <n>              run example n
//!   :verbose [on|off]         toggle per-step evaluation tracing
//!   :commit                   run one live consensus epoch across the parties
//!   :chat <party> <message>   send an end-to-end encrypted direct message
//!   :post <message>           broadcast a signed forum post
//!   :quit                     signal all parties to exit
//!
//! Anything else is parsed as a Baru expression, broadcast, and evaluated
//! jointly; a parse/runtime error on party 0 broadcasts `:abort` so the workers
//! re-sync instead of hanging.

use std::env;
use std::panic;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use colored::*;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use serde::{Deserialize, Serialize};

use sha2::{Digest, Sha256};

use baru::consensus::{ConsensusBackend, RootSet, SingleLeader};
use baru::crypto::identity::Identity;
use baru::crypto::net::{
    deserialize_msg, recv_blocking, recv_forever, serialize_msg, InprocHandle, MpcMsg,
    TcpTransport, Transport,
};
use baru::crypto::identity::IdentityId;
use baru::fhe::backend::FheBackend;
use baru::fhe::lifecycle::{Effect, Stake};
use baru::fhe::market::{evaluate_job, verify_receipt, FheJob, Verdict as FheVerdict, WorkReceipt};
use baru::fhe::mock_backend::MockFheBackend;
use baru::interpreter::{parse, Participant, Types};
use baru::messaging::{open, seal, ChatSecret, Contact, Forum, ForumPost};
use baru::mpc_market::{
    cosign, result_commitment, verify_cosignature, CoSignature, MpcJob, MpcReceipt, Verdict,
};
use baru::storage::{self, ChallengeResponse, StorageDeal, StoredData};
use baru::zkp::circuit_compiler::CircuitCompiler;
use baru::reputation::{ReputationLedger, ReputationPoUW};
use baru::rest::{serve_blocking, AppState, NodeState};
use baru::repl::{print_examples_for, print_examples_list, print_help, ReplHelper, EXAMPLES};

// ---------------------------------------------------------------------------
// Expression broadcast helpers (use share_id=0 for len + sequential ids for chunks)
// ---------------------------------------------------------------------------

fn broadcast_expr<T: Transport>(src: &str, transport: &T) {
    let bytes = src.as_bytes();
    transport.broadcast(serialize_msg(&MpcMsg::RevealShare {
        share_id: 0,
        value: bytes.len() as u128,
    }));
    for (i, chunk) in bytes.chunks(16).enumerate() {
        let mut v: u128 = 0;
        for (j, &b) in chunk.iter().enumerate() {
            v |= (b as u128) << (j * 8);
        }
        transport.broadcast(serialize_msg(&MpcMsg::RevealShare {
            share_id: (i + 1) as u64,
            value: v,
        }));
    }
}

fn receive_expr<T: Transport>(transport: &T) -> String {
    // Wait indefinitely for the length header — no timeout between expressions.
    let len = loop {
        let (_from, msg) = recv_forever(transport);
        if let MpcMsg::RevealShare {
            share_id: 0,
            value: l,
        } = deserialize_msg(&msg)
        {
            break l as usize;
        }
        // Any other message that arrived before the length header is ignored;
        // this can happen if a stray MPC message lingers in the queue.
    };
    let chunks = len.div_ceil(16);
    let mut bytes = vec![0u8; len];
    let mut received = 0usize;
    while received < chunks {
        if let Some((_from, msg)) = recv_blocking(transport, 5_000) {
            if let MpcMsg::RevealShare { share_id, value } = deserialize_msg(&msg) {
                if share_id >= 1 {
                    let start = (share_id as usize - 1) * 16;
                    for j in 0..16 {
                        if start + j < len {
                            bytes[start + j] = ((value >> (j * 8)) & 0xff) as u8;
                        }
                    }
                    received += 1;
                }
            }
        } else {
            panic!("receive_expr: timeout waiting for chunk");
        }
    }
    String::from_utf8(bytes).expect("expression not valid UTF-8")
}

// ---------------------------------------------------------------------------
// In-process 3-party demo
//
// Each party runs in its own thread.  We print results inside the thread to
// avoid moving !Send SharedValue across thread boundaries.
// Two InprocHandle networks are created: one for expression distribution,
// one for MPC operations.
// ---------------------------------------------------------------------------

fn run_inproc_demo() {
    println!("Running in-process 3-party MPC demo...\n");

    let demos: &[&str] = &[
        "(reconstruct (share 7 8))",
        "(reconstruct (+ (share 10 8) (share 32 8)))",
        "(let ((w (share -1 8)) (x (share 3 8))) (reconstruct (* w x)))",
        "(let ((z (+ (share -2 8) (share 5 8)))) (reconstruct (if (> z 0) z 0)))",
    ];

    for &src in demos {
        println!("Expression: {}", src);
        // one network for MPC, one for expression distribution
        let mpc_handles = InprocHandle::create(3);
        let dist_handles = InprocHandle::create(3);

        let mut joins = vec![];
        for i in 0..3usize {
            let mpc_h = mpc_handles[i].clone();
            let dist_h = dist_handles[i].clone();
            let src_str = src.to_string();
            joins.push(thread::spawn(move || {
                let participant = Participant::new_transport(i, mpc_h);
                let resolved_src = if i == 0 {
                    broadcast_expr(&src_str, &dist_h);
                    src_str
                } else {
                    receive_expr(&dist_h)
                };
                let expr = parse(&resolved_src).expect("parse failed");
                let result = participant.eval(&expr);
                // print inside thread to avoid Send bound on SharedValue
                println!("  party {} => {:?}", i, result);
            }));
        }
        for j in joins {
            j.join().expect("thread panicked");
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// TCP multi-party mode
// ---------------------------------------------------------------------------

fn history_path() -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".baru_tcp_history");
        Some(p)
    } else {
        None
    }
}

/// Render a result (revealing a `Share` **exactly once**, sign-extended through
/// 32 bits) and return its numeric value for settlement — `None` for
/// non-numeric results, which don't settle.
fn display_result(result: &Types, participant: &Participant) -> (String, Option<i64>) {
    match result {
        Types::Share(sr) => {
            let signed = (participant.reveal(sr) as u32) as i32 as i64;
            (signed.to_string().bright_green().to_string(), Some(signed))
        }
        Types::Num(n) => {
            let signed = (*n as u32) as i32 as i64;
            (signed.to_string().bright_green().to_string(), Some(signed))
        }
        other => (format!("{:?}", other).bright_green().to_string(), None),
    }
}

/// Shift the port of a `host:port` address by `delta` (used to derive the
/// dist / consensus address sets from the MPC one).
fn bump_port(addr: &str, delta: u16) -> String {
    if let Some(colon) = addr.rfind(':') {
        let port: u16 = addr[colon + 1..].parse().unwrap_or(7000);
        format!("{}:{}", &addr[..colon], port.wrapping_add(delta))
    } else {
        format!("{}:{}", addr, 7000 + delta)
    }
}

/// Build the transports (plaintext TCP, or PQ-TLS when compiled with
/// `--features pq-tls` and `BARU_TLS` is set) and run the party.  A dedicated
/// consensus transport is added when `BARU_CONSENSUS` is set.
fn launch(me: usize, mpc_addrs: Vec<String>, dist_addrs: Vec<String>) {
    let want_consensus = std::env::var("BARU_CONSENSUS").is_ok();
    let want_messaging = std::env::var("BARU_CHAT").is_ok();
    let want_settlement = std::env::var("BARU_SETTLE").is_ok();
    let cons_addrs: Vec<String> = mpc_addrs.iter().map(|a| bump_port(a, 200)).collect();
    let msg_addrs: Vec<String> = mpc_addrs.iter().map(|a| bump_port(a, 300)).collect();
    let settle_addrs: Vec<String> = mpc_addrs.iter().map(|a| bump_port(a, 400)).collect();

    #[cfg(feature = "pq-tls")]
    {
        if std::env::var("BARU_TLS").is_ok() {
            use baru::crypto::tls::TlsTransport;
            // One long-term identity for this party, shared by all channels.
            let id = Identity::generate();
            let secret = id.to_secret_bytes();
            eprintln!(
                "{}",
                format!("[party {}] identity {}", me, id.id()).bright_magenta()
            );
            let mpc = TlsTransport::new(me, mpc_addrs, id).expect("tls mpc transport");
            let dist = TlsTransport::new(me, dist_addrs, Identity::from_secret_bytes(secret))
                .expect("tls dist transport");
            let cons = want_consensus.then(|| {
                TlsTransport::new(me, cons_addrs, Identity::from_secret_bytes(secret))
                    .expect("tls consensus transport")
            });
            let msg = want_messaging.then(|| {
                TlsTransport::new(me, msg_addrs, Identity::from_secret_bytes(secret))
                    .expect("tls messaging transport")
            });
            let settle = want_settlement.then(|| {
                TlsTransport::new(me, settle_addrs, Identity::from_secret_bytes(secret))
                    .expect("tls settlement transport")
            });
            run_tcp(me, mpc, dist, cons, msg, settle);
            return;
        }
    }
    let mpc = TcpTransport::new(me, mpc_addrs);
    let dist = TcpTransport::new(me, dist_addrs);
    let cons = want_consensus.then(|| TcpTransport::new(me, cons_addrs));
    let msg = want_messaging.then(|| TcpTransport::new(me, msg_addrs));
    let settle = want_settlement.then(|| TcpTransport::new(me, settle_addrs));
    run_tcp(me, mpc, dist, cons, msg, settle);
}

/// Deterministic long-term identity for a party (demo stand-in for a
/// self-generated key).  This one keypair is used both to co-sign MPC job
/// results and — via its id — as the reputation-roster entry, so co-signing
/// workers earn reputation under the same identity leader-election weights.
fn party_identity(party: usize) -> Identity {
    let mut h = Sha256::new();
    h.update(b"baru-party-id-v1");
    h.update((party as u64).to_le_bytes());
    Identity::from_secret_bytes(h.finalize().into())
}

/// The reputation-roster id of a party: the id of its [`party_identity`].
fn roster_id(party: usize) -> [u8; 32] {
    party_identity(party).id().0
}

/// Number of parties in this demo committee.
const N_PARTIES: usize = 3;
/// Co-signatures required to finalize an MPC job (honest majority of 3).
const SETTLE_THRESHOLD: usize = 2;
/// Reputation + stake a co-signing worker earns per finalized MPC job.
const SETTLE_REWARD: Stake = 10;

/// Build the consensus backend: `ReputationPoUW` when `BARU_POUW` is set (leader
/// elected by the shared reputation ledger), else the `SingleLeader` baseline.
fn build_consensus<T: Transport + 'static>(
    me: usize,
    n: usize,
    transport: T,
    ledger: Arc<Mutex<ReputationLedger>>,
) -> Box<dyn ConsensusBackend> {
    if std::env::var("BARU_POUW").is_ok() {
        let roster: Vec<[u8; 32]> = (0..n).map(roster_id).collect();
        Box::new(ReputationPoUW::new(me, n, roster, ledger, transport, [0u8; 32]))
    } else {
        Box::new(SingleLeader::new(me, n, 0, transport))
    }
}

/// Seed the ledger with an optional deterministic genesis reputation
/// (`BARU_POUW_REP="party:delta,..."`).  All parties set the same env, so all
/// build the same ledger.
fn seed_genesis_reputation(ledger: &Arc<Mutex<ReputationLedger>>, n: usize) {
    if let Ok(spec) = std::env::var("BARU_POUW_REP") {
        if let Ok(mut l) = ledger.lock() {
            for entry in spec.split(',') {
                if let Some((p, amt)) = entry.split_once(':') {
                    if let (Ok(p), Ok(amt)) = (p.trim().parse::<usize>(), amt.trim().parse::<i128>())
                    {
                        if p < n {
                            l.apply(&Effect::Reputation {
                                id: roster_id(p),
                                delta: amt,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Run one consensus epoch committing this participant's current environment
/// root, and advance the epoch counter.  No-op (with a log) if consensus is off.
fn commit_state(
    consensus: &mut Option<Box<dyn ConsensusBackend>>,
    epoch: &mut u64,
    participant: &Participant,
    me: usize,
    node_state: &Option<Arc<Mutex<NodeState>>>,
    ledger: &Arc<Mutex<ReputationLedger>>,
) {
    let Some(cb) = consensus.as_mut() else {
        eprintln!("[party {}] :commit ignored — consensus disabled", me);
        return;
    };
    let leader = cb
        .epoch_leader(*epoch)
        .map(|l| l.to_string())
        .unwrap_or_else(|| "?".to_string());
    let root = participant.env_root();
    let roots = RootSet {
        private_root: [0u8; 32],
        public_root: root,
        state_root: root,
    };
    match cb.commit_epoch(*epoch, roots) {
        Ok(commit) => {
            let group = hex::encode(&commit.group_root[..8]);
            let rep = hex::encode(&commit.reputation_root[..8]);
            let line = format!(
                "[party {}] epoch {} committed via {} — leader={} group_root={}… rep_root={}…",
                me,
                *epoch,
                cb.name(),
                leader,
                group,
                rep
            );
            if me == 0 {
                println!("{}", line.bright_green());
            } else {
                eprintln!("{}", line.bright_black());
            }
            *epoch += 1;

            // Reputation now accrues per MPC *job* (see `settle_expression`),
            // not per epoch, so committing only snapshots the ledger for the
            // REST view.
            let rep_snapshot = ledger.lock().ok().map(|l| l.clone());

            // Reflect the committed state in the REST control plane.
            if let Some(ns) = node_state {
                if let Ok(mut s) = ns.lock() {
                    s.epoch = *epoch;
                    s.state_root = participant.env_root();
                    if let Some(r) = rep_snapshot {
                        s.reputation = r;
                    }
                }
            }
        }
        Err(e) => eprintln!("[party {}] consensus error: {}", me, e),
    }
}

// ── P2P chat + forum ───────────────────────────────────────────────────────────
//
// Delivery for baru::messaging over a dedicated transport.  For the demo each
// party derives its identity and chat key deterministically from its index, so
// everyone can address everyone without a discovery protocol (in production
// keys are self-generated and contacts exchanged over the overlay).

/// Envelope for messages carried on the messaging transport.
#[derive(Serialize, Deserialize)]
enum MsgEnvelope {
    Chat(baru::messaging::ChatMessage),
    Post(ForumPost),
}

fn msg_identity(party: usize) -> Identity {
    let mut h = Sha256::new();
    h.update(b"baru-msg-id-v1");
    h.update((party as u64).to_le_bytes());
    Identity::from_secret_bytes(h.finalize().into())
}

fn msg_chat(party: usize) -> ChatSecret {
    let mut h = Sha256::new();
    h.update(b"baru-msg-chat-v1");
    h.update((party as u64).to_le_bytes());
    ChatSecret::from_bytes(h.finalize().into())
}

fn msg_contact(party: usize) -> Contact {
    Contact {
        identity: msg_identity(party).public(),
        chat_pk: msg_chat(party).public(),
    }
}

/// Background loop: decrypt incoming chat addressed to us and append forum posts.
fn messaging_receiver<T: Transport>(me: usize, transport: Arc<T>, forum: Arc<Mutex<Forum>>) {
    let my_id = msg_identity(me);
    let my_chat = msg_chat(me);
    loop {
        if let Some((_from, bytes)) = transport.recv() {
            match bincode::deserialize::<MsgEnvelope>(&bytes) {
                Ok(MsgEnvelope::Chat(msg)) => match open(&my_id, &my_chat, &msg) {
                    Ok(pt) => eprintln!(
                        "{}",
                        format!(
                            "[party {}] 📩 chat from {}: {}",
                            me,
                            msg.from.id(),
                            String::from_utf8_lossy(&pt)
                        )
                        .bright_cyan()
                    ),
                    Err(baru::messaging::ChatError::WrongRecipient) => {}
                    Err(e) => eprintln!("[party {}] chat error: {}", me, e),
                },
                Ok(MsgEnvelope::Post(post)) => {
                    if let Ok(mut f) = forum.lock() {
                        if f.append(post.clone()).is_ok() {
                            eprintln!(
                                "{}",
                                format!(
                                    "[party {}] 📰 forum {}: {}  (root={})",
                                    me,
                                    post.author.id(),
                                    post.body,
                                    hex::encode(&f.root()[..6])
                                )
                                .bright_blue()
                            );
                        }
                    }
                }
                Err(_) => {}
            }
        } else {
            thread::sleep(Duration::from_millis(5));
        }
    }
}

/// Chat + forum delivery for this party.
struct Messaging<T: Transport> {
    me: usize,
    n: usize,
    transport: Arc<T>,
    forum: Arc<Mutex<Forum>>,
    post_seq: u64,
}

impl<T: Transport + 'static> Messaging<T> {
    fn new(me: usize, n: usize, transport: T) -> Self {
        let transport = Arc::new(transport);
        let _ = transport.wait_ready(10_000);
        let forum = Arc::new(Mutex::new(Forum::new()));
        let rx_t = transport.clone();
        let rx_f = forum.clone();
        thread::spawn(move || messaging_receiver(me, rx_t, rx_f));
        Messaging {
            me,
            n,
            transport,
            forum,
            post_seq: 0,
        }
    }

    /// Seal an end-to-end encrypted DM and unicast it to `to`.
    fn send_chat(&self, to: usize, text: &str) {
        if to >= self.n || to == self.me {
            println!("{}", "chat: pick another party's index".bright_red());
            return;
        }
        let msg = seal(
            &msg_identity(self.me),
            &msg_chat(self.me),
            &msg_contact(to),
            text.as_bytes(),
        );
        self.transport
            .send(to, serialize_msg(&MsgEnvelope::Chat(msg)));
        println!(
            "{}",
            format!("🔒 encrypted chat sent to party {}", to).bright_cyan()
        );
    }

    /// Broadcast a signed forum post and append it locally.
    fn post(&mut self, text: &str) {
        self.post_seq += 1;
        let post = ForumPost::create(&msg_identity(self.me), self.post_seq, text);
        self.transport
            .broadcast(serialize_msg(&MsgEnvelope::Post(post.clone())));
        if let Ok(mut f) = self.forum.lock() {
            let _ = f.append(post);
            println!(
                "{}",
                format!("📰 posted to forum (root={})", hex::encode(&f.root()[..6])).bright_blue()
            );
        }
    }
}

// ── Per-job market settlement (MPC / FHE / storage) ────────────────────────────
//
// After a unit of verifiable work, the parties settle it over a dedicated
// transport and fold the reward into the shared reputation ledger:
//   * MPC     — after each evaluated expression, parties co-sign the result;
//               co-signatures that disagree (a deviant's) are excluded, and a
//               threshold receipt settles via `baru::mpc_market`.
//   * FHE     — a worker evaluates a homomorphic job and broadcasts its receipt;
//               every party re-executes to verify it (deterministic) and settles.
//   * storage — a provider answers a Merkle retrievability challenge; every
//               party verifies the proof against the known root and settles.
// A wrong result / bad proof simply fails verification, so its author earns
// nothing while the honest parties still settle.

/// Messages on the settlement transport.
#[derive(Serialize, Deserialize)]
enum SettleMsg {
    Mpc {
        seq: u64,
        from: usize,
        cosig: CoSignature,
    },
    Fhe {
        seq: u64,
        worker: usize,
        receipt: WorkReceipt,
    },
    Storage {
        seq: u64,
        provider: usize,
        response: ChallengeResponse,
    },
}

/// A deterministic request commitment for the `seq`-th MPC job.
fn mpc_request(seq: u64) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"baru-mpc-job-v1");
    h.update(seq.to_le_bytes());
    h.finalize().into()
}

/// A fixed, deterministic FHE job `(+ 10 32)` on the mock backend.  Every party
/// builds the identical job, so any party can re-execute it to verify a receipt.
fn build_fhe_job() -> (FheJob, MockFheBackend) {
    let be = MockFheBackend;
    let (ck, _) = be.keygen();
    let expr = parse("(+ x y)").expect("parse fhe expr");
    let mut circuit = CircuitCompiler::new().compile_fhe(&expr).expect("compile fhe");
    circuit.public_inputs.clear();
    circuit.encrypted_inputs = vec!["x".into(), "y".into()];
    let mut enc = std::collections::HashMap::new();
    enc.insert("x".to_string(), be.serialize_ct(&be.encrypt(&ck, 10)));
    enc.insert("y".to_string(), be.serialize_ct(&be.encrypt(&ck, 32)));
    let job = FheJob {
        circuit,
        public_inputs: std::collections::HashMap::new(),
        encrypted_inputs: enc,
        backend_id: be.backend_id().to_string(),
    };
    (job, be)
}

/// A party's deterministic stored dataset (16 high-entropy chunks).  Every party
/// can recompute any provider's Merkle root, so it can verify its proofs.
fn build_stored_data(party: usize) -> StoredData {
    StoredData::new(
        (0..16u64)
            .map(|i| {
                let mut h = Sha256::new();
                h.update(b"baru-storage-v1");
                h.update((party as u64).to_le_bytes());
                h.update(i.to_le_bytes());
                h.finalize().to_vec()
            })
            .collect(),
    )
}

/// Fold settlement effects into the ledger and refresh the REST view (no nested
/// locks: snapshot the ledger, release it, then write `node_state`).
fn fold_and_sync(
    effects: &[Effect],
    ledger: &Arc<Mutex<ReputationLedger>>,
    node_state: &Option<Arc<Mutex<NodeState>>>,
) {
    let snapshot = ledger.lock().ok().map(|mut l| {
        l.apply_all(effects);
        l.clone()
    });
    if let (Some(ns), Some(snap)) = (node_state, snapshot) {
        if let Ok(mut s) = ns.lock() {
            s.reputation = snap;
        }
    }
}

/// Per-job settlement over a dedicated transport.
struct Settlement<T: Transport> {
    me: usize,
    transport: T,
    mpc_seq: u64,
    fhe_seq: u64,
    storage_seq: u64,
    /// Messages for future rounds that arrived early.
    buf: Vec<SettleMsg>,
}

impl<T: Transport> Settlement<T> {
    fn new(me: usize, transport: T) -> Self {
        let _ = transport.wait_ready(10_000);
        Settlement {
            me,
            transport,
            mpc_seq: 0,
            fhe_seq: 0,
            storage_seq: 0,
            buf: Vec::new(),
        }
    }

    /// Receive the next settlement message satisfying `pred`, buffering others.
    fn recv_matching(
        &mut self,
        deadline: std::time::Instant,
        pred: impl Fn(&SettleMsg) -> bool,
    ) -> Option<SettleMsg> {
        if let Some(pos) = self.buf.iter().position(&pred) {
            return Some(self.buf.remove(pos));
        }
        loop {
            match self.transport.recv() {
                Some((_from, bytes)) => {
                    let m: SettleMsg = deserialize_msg(&bytes);
                    if pred(&m) {
                        return Some(m);
                    }
                    self.buf.push(m);
                }
                None if std::time::Instant::now() >= deadline => return None,
                None => thread::sleep(Duration::from_millis(2)),
            }
        }
    }

    /// MPC: co-sign the result, collect agreeing co-signatures, settle.
    fn settle_expression(
        &mut self,
        result_val: i64,
        ledger: &Arc<Mutex<ReputationLedger>>,
        node_state: &Option<Arc<Mutex<NodeState>>>,
    ) {
        self.mpc_seq += 1;
        let seq = self.mpc_seq;
        let request = mpc_request(seq);
        let result_root = result_commitment(result_val);

        let my_cosig = cosign(&party_identity(self.me), &request, &result_root);
        self.transport.broadcast(serialize_msg(&SettleMsg::Mpc {
            seq,
            from: self.me,
            cosig: my_cosig.clone(),
        }));

        use std::collections::HashSet;
        let mut cosigners = vec![my_cosig];
        let mut seen: HashSet<usize> = HashSet::from([self.me]);
        let deadline = std::time::Instant::now() + Duration::from_millis(4_000);
        while cosigners.len() < N_PARTIES {
            let Some(SettleMsg::Mpc { from, cosig, .. }) = self.recv_matching(deadline, |m| {
                matches!(m, SettleMsg::Mpc { seq: s, .. } if *s == seq)
            }) else {
                break;
            };
            if seen.insert(from) && verify_cosignature(&cosig, &request, &result_root) {
                cosigners.push(cosig);
            }
        }

        let committee: Vec<IdentityId> = (0..N_PARTIES).map(|p| party_identity(p).id()).collect();
        let job = MpcJob {
            request,
            owner: committee[0],
            committee,
            threshold: SETTLE_THRESHOLD,
        };
        let receipt = MpcReceipt {
            request,
            result_root,
            cosigners,
        };
        let (verdict, effects) = baru::mpc_market::settle(&job, &receipt, SETTLE_REWARD);
        if verdict == Verdict::Valid {
            fold_and_sync(&effects, ledger, node_state);
        } else {
            eprintln!(
                "[party {}] MPC job {} not finalized: {:?}",
                self.me, seq, verdict
            );
        }
    }

    /// FHE: a worker evaluates the job and broadcasts its receipt; every party
    /// re-executes to verify it and settles (rewarding the worker).
    fn settle_fhe(
        &mut self,
        ledger: &Arc<Mutex<ReputationLedger>>,
        node_state: &Option<Arc<Mutex<NodeState>>>,
    ) {
        self.fhe_seq += 1;
        let seq = self.fhe_seq;
        let worker = (seq as usize) % N_PARTIES;
        let (job, be) = build_fhe_job();

        let receipt = if self.me == worker {
            let (_handle, r) = evaluate_job(&job, &be).expect("fhe eval");
            self.transport.broadcast(serialize_msg(&SettleMsg::Fhe {
                seq,
                worker,
                receipt: r.clone(),
            }));
            Some(r)
        } else {
            let deadline = std::time::Instant::now() + Duration::from_millis(4_000);
            self.recv_matching(deadline, |m| matches!(m, SettleMsg::Fhe { seq: s, .. } if *s == seq))
                .and_then(|m| match m {
                    SettleMsg::Fhe { receipt, .. } => Some(receipt),
                    _ => None,
                })
        };

        let Some(receipt) = receipt else {
            eprintln!("[party {}] FHE job {} timed out", self.me, seq);
            return;
        };
        // Verify by deterministic re-execution — no decryption.
        if verify_receipt(&job, &receipt, &be) == FheVerdict::Valid {
            let id = party_identity(worker).id().0;
            let effects = vec![
                Effect::Release {
                    to: id,
                    amount: SETTLE_REWARD,
                },
                Effect::Reputation {
                    id,
                    delta: SETTLE_REWARD as i128,
                },
            ];
            fold_and_sync(&effects, ledger, node_state);
            if self.me == 0 {
                println!(
                    "{}",
                    format!(
                        "🔐 FHE job {} verified by re-execution → worker {} rewarded",
                        seq, worker
                    )
                    .bright_cyan()
                );
            }
        } else {
            eprintln!(
                "[party {}] FHE job {} FRAUD (re-execution mismatch)",
                self.me, seq
            );
        }
    }

    /// Storage: a provider answers a Merkle retrievability challenge; every party
    /// verifies the proof against the known root and settles.
    fn settle_storage(
        &mut self,
        ledger: &Arc<Mutex<ReputationLedger>>,
        node_state: &Option<Arc<Mutex<NodeState>>>,
    ) {
        self.storage_seq += 1;
        let seq = self.storage_seq;
        let provider = (seq as usize) % N_PARTIES;
        let data = build_stored_data(provider);
        let deal = StorageDeal {
            data_root: data.root(),
            n_chunks: data.len(),
            owner: party_identity(0).id().0,
            provider: party_identity(provider).id().0,
            reward_per_epoch: SETTLE_REWARD,
            until_epoch: u64::MAX,
        };
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&seq.to_le_bytes());
        let index = deal.challenge(seq, &seed);

        let response = if self.me == provider {
            let r = data.answer(index).expect("answer challenge");
            self.transport.broadcast(serialize_msg(&SettleMsg::Storage {
                seq,
                provider,
                response: r.clone(),
            }));
            Some(r)
        } else {
            let deadline = std::time::Instant::now() + Duration::from_millis(4_000);
            self.recv_matching(deadline, |m| {
                matches!(m, SettleMsg::Storage { seq: s, .. } if *s == seq)
            })
            .and_then(|m| match m {
                SettleMsg::Storage { response, .. } => Some(response),
                _ => None,
            })
        };

        let (proven, effects) = storage::settle_epoch(&deal, seq, &seed, response.as_ref());
        if proven {
            fold_and_sync(&effects, ledger, node_state);
            if self.me == 0 {
                println!(
                    "{}",
                    format!(
                        "💾 storage challenge {} (chunk {}) proven → provider {} rewarded",
                        seq, index, provider
                    )
                    .bright_cyan()
                );
            }
        } else {
            eprintln!("[party {}] storage challenge {} unproven", self.me, seq);
        }
    }
}

fn run_tcp<T: Transport + 'static>(
    me: usize,
    mpc_transport: T,
    dist_transport: T,
    consensus_transport: Option<T>,
    messaging_transport: Option<T>,
    settlement_transport: Option<T>,
) {
    // Readiness barrier: block until fully connected to all peers on both
    // transports before doing any protocol work.  Replaces startup `sleep`s.
    let mpc_ok = mpc_transport.wait_ready(10_000);
    let dist_ok = dist_transport.wait_ready(10_000);
    eprintln!(
        "{}",
        format!(
            "[party {}] transports ready (mpc={}, dist={})",
            me, mpc_ok, dist_ok
        )
        .bright_yellow()
    );

    // Reputation ledger shared with the consensus backend: seeded with an
    // optional genesis, then folded with each committed epoch's real MPC work.
    let ledger = Arc::new(Mutex::new(ReputationLedger::new()));
    seed_genesis_reputation(&ledger, 3);

    // Optional consensus backend over its dedicated transport: single-leader
    // parity by default, reputation-weighted (ReputationPoUW) with BARU_POUW.
    let mut consensus: Option<Box<dyn ConsensusBackend>> = consensus_transport.map(|ct| {
        let _ = ct.wait_ready(10_000);
        build_consensus(me, 3, ct, ledger.clone())
    });
    let mut epoch: u64 = 0;

    // Optional p2p chat + forum over a dedicated transport (spawns a receiver).
    let mut messaging = messaging_transport.map(|mt| Messaging::new(me, 3, mt));

    // Optional per-job MPC settlement over a dedicated transport.
    let mut settlement = settlement_transport.map(|st| Settlement::new(me, st));

    // Optional REST control plane (BARU_REST=<base-port>, default 8090; this
    // party listens on base+id).  Exposes health / state / reputation / forum /
    // programs; the shared NodeState is refreshed as the node commits epochs.
    let node_state: Option<Arc<Mutex<NodeState>>> = std::env::var("BARU_REST").ok().map(|spec| {
        let base: u16 = spec.trim().parse().unwrap_or(8090);
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], base + me as u16));
        let ns = Arc::new(Mutex::new(NodeState::new()));
        let app = AppState::new(me, &msg_identity(me), ns.clone());
        thread::spawn(move || serve_blocking(addr, app));
        eprintln!(
            "{}",
            format!("[party {}] REST control plane on http://{}", me, addr).bright_magenta()
        );
        ns
    });

    let participant = Participant::new_transport(me, mpc_transport);

    if me == 0 {
        // ── Party 0: full interactive REPL ──────────────────────────────────
        println!(
            "{}",
            "ABY3 distributed MPC — party 0 (evaluator + broadcaster)".bright_green()
        );
        println!("{}", "Type ':help' for language reference, ':examples' to browse examples, or ':quit' to exit.".bright_yellow());

        let helper = ReplHelper::new_tcp();
        let mut rl = Editor::new().expect("failed to create line editor");
        rl.set_helper(Some(helper));

        if let Some(p) = history_path() {
            if p.exists() {
                let _ = rl.load_history(&p);
            }
        }

        let mut verbose = false;

        loop {
            let readline = rl.readline(&format!(
                "{}{}{}",
                "baru[party-0]".bright_cyan(),
                "> ".bright_white(),
                ""
            ));
            let src = match readline {
                Ok(line) => line.trim().to_string(),
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) | Err(_) => {
                    broadcast_expr(":quit", &dist_transport);
                    break;
                }
            };

            if src.is_empty() || src.starts_with(';') {
                continue;
            }
            let _ = rl.add_history_entry(&src);

            // ── Special commands ────────────────────────────────────────────
            if src.starts_with(':') || src == "quit" || src == "exit" {
                let mut parts = src.splitn(3, ' ');
                let cmd = parts.next().unwrap_or("");
                match cmd {
                    ":quit" | "quit" | "exit" => {
                        broadcast_expr(":quit", &dist_transport);
                        println!("{}", "Signalled all parties to quit.".bright_yellow());
                        break;
                    }
                    ":help" => {
                        print_help(true);
                        continue;
                    }
                    ":examples" => {
                        match parts.next() {
                            None => print_examples_list(),
                            Some(cat) => print_examples_for(cat),
                        }
                        continue;
                    }
                    ":example" => {
                        match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                            Some(n) if n >= 1 && n <= EXAMPLES.len() => {
                                let ex = &EXAMPLES[n - 1];
                                println!();
                                println!(
                                    "{} [{}] {} ({})",
                                    "Running example".bright_yellow(),
                                    n.to_string().bright_white(),
                                    ex.name.bright_cyan(),
                                    ex.category
                                );
                                println!(
                                    "{} {}",
                                    "Code:    ".bright_yellow(),
                                    ex.code.bright_green()
                                );
                                println!(
                                    "{} {}",
                                    "Expected:".bright_yellow(),
                                    ex.expected.bright_magenta()
                                );
                                println!();
                                // broadcast and fall through to normal eval below
                                participant.reset_per_expression();
                                broadcast_expr(ex.code, &dist_transport);
                                match parse(ex.code) {
                                    Ok(expr) => {
                                        let result =
                                            panic::catch_unwind(panic::AssertUnwindSafe(|| {
                                                if verbose {
                                                    participant.eval_quiet_verbose(&expr)
                                                } else {
                                                    participant.eval_quiet(&expr)
                                                }
                                            }));
                                        match result {
                                            Ok(val) => {
                                                let (shown, num) =
                                                    display_result(&val, &participant);
                                                println!("{} {}", "=>".bright_yellow(), shown);
                                                if let (Some(v), Some(s)) =
                                                    (num, settlement.as_mut())
                                                {
                                                    s.settle_expression(v, &ledger, &node_state);
                                                }
                                            }
                                            Err(e) => {
                                                let msg = if let Some(s) =
                                                    e.downcast_ref::<String>()
                                                {
                                                    s.clone()
                                                } else if let Some(s) = e.downcast_ref::<&str>() {
                                                    s.to_string()
                                                } else {
                                                    "unknown runtime error".to_string()
                                                };
                                                println!(
                                                    "{} {}",
                                                    "runtime error:".bright_red(),
                                                    msg
                                                );
                                                broadcast_expr(":abort", &dist_transport);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        println!("{} {}", "parse error:".bright_red(), e);
                                        broadcast_expr(":abort", &dist_transport);
                                    }
                                }
                            }
                            _ => println!(
                                "usage: :example <n>  (use ':examples' to list, 1–{})",
                                EXAMPLES.len()
                            ),
                        }
                        continue;
                    }
                    ":verbose" => {
                        match parts.next() {
                            Some("on") => {
                                verbose = true;
                                println!("{}", "verbose: on".bright_green());
                            }
                            Some("off") => {
                                verbose = false;
                                println!("{}", "verbose: off".bright_yellow());
                            }
                            _ => {
                                verbose = !verbose;
                                println!(
                                    "verbose: {}",
                                    if verbose {
                                        "on".bright_green()
                                    } else {
                                        "off".bright_yellow()
                                    }
                                );
                            }
                        }
                        continue;
                    }
                    ":commit" => {
                        // Run one live consensus epoch over the network: signal
                        // the workers, then commit everyone's state roots.
                        if consensus.is_some() {
                            broadcast_expr(":commit", &dist_transport);
                            commit_state(&mut consensus, &mut epoch, &participant, me, &node_state, &ledger);
                        } else {
                            println!(
                                "{}",
                                "consensus disabled — set BARU_CONSENSUS=1 on all parties"
                                    .bright_yellow()
                            );
                        }
                        continue;
                    }
                    ":chat" => {
                        // Send an end-to-end encrypted DM to a peer.
                        if let Some(m) = messaging.as_ref() {
                            let to = parts.next().and_then(|s| s.parse::<usize>().ok());
                            let text = parts.next().unwrap_or("");
                            match to {
                                Some(p) if !text.is_empty() => m.send_chat(p, text),
                                _ => println!("usage: :chat <party-index> <message>"),
                            }
                        } else {
                            println!(
                                "{}",
                                "chat disabled — set BARU_CHAT=1 on all parties".bright_yellow()
                            );
                        }
                        continue;
                    }
                    ":post" => {
                        // Broadcast a signed post to the forum.
                        if let Some(m) = messaging.as_mut() {
                            let body = src.strip_prefix(":post").map(|s| s.trim()).unwrap_or("");
                            if body.is_empty() {
                                println!("usage: :post <message>");
                            } else {
                                m.post(body);
                            }
                        } else {
                            println!(
                                "{}",
                                "forum disabled — set BARU_CHAT=1 on all parties".bright_yellow()
                            );
                        }
                        continue;
                    }
                    ":fhe" => {
                        // Post an FHE job: a worker evaluates, all verify by
                        // re-execution and settle.
                        if let Some(s) = settlement.as_mut() {
                            broadcast_expr(":fhe", &dist_transport);
                            s.settle_fhe(&ledger, &node_state);
                        } else {
                            println!(
                                "{}",
                                "settlement disabled — set BARU_SETTLE=1 on all parties"
                                    .bright_yellow()
                            );
                        }
                        continue;
                    }
                    ":storage" => {
                        // Issue a storage retrievability challenge; the provider
                        // answers, all verify the Merkle proof and settle.
                        if let Some(s) = settlement.as_mut() {
                            broadcast_expr(":storage", &dist_transport);
                            s.settle_storage(&ledger, &node_state);
                        } else {
                            println!(
                                "{}",
                                "settlement disabled — set BARU_SETTLE=1 on all parties"
                                    .bright_yellow()
                            );
                        }
                        continue;
                    }
                    _ => {
                        println!(
                            "{} {}  (type '{}' for reference)",
                            "unknown command:".bright_red(),
                            cmd,
                            ":help".bright_cyan()
                        );
                        continue;
                    }
                }
            }

            // ── Normal expression ────────────────────────────────────────────
            // Reset per-expression backend state so all parties' ID counters start
            // at the same value for this expression (crucial after error recovery).
            participant.reset_per_expression();
            broadcast_expr(&src, &dist_transport);
            match parse(&src) {
                Err(e) => {
                    println!("{} {}", "parse error:".bright_red(), e);
                    // Workers already received the expression and got the same parse error.
                    // Broadcast :abort so they skip any pending reveal and return to wait.
                    broadcast_expr(":abort", &dist_transport);
                }
                Ok(expr) => {
                    // Wrap evaluation in catch_unwind so a panic (e.g. MPC protocol
                    // timeout, out-of-field value) doesn't crash the whole process.
                    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        if verbose {
                            participant.eval_quiet_verbose(&expr)
                        } else {
                            participant.eval_quiet(&expr)
                        }
                    }));
                    match result {
                        Ok(val) => {
                            let (shown, num) = display_result(&val, &participant);
                            println!("{} {}", "=>".bright_yellow(), shown);
                            // Settle this MPC job (co-sign + exchange + reward).
                            if let (Some(v), Some(s)) = (num, settlement.as_mut()) {
                                s.settle_expression(v, &ledger, &node_state);
                            }
                        }
                        Err(e) => {
                            let msg = if let Some(s) = e.downcast_ref::<String>() {
                                s.clone()
                            } else if let Some(s) = e.downcast_ref::<&str>() {
                                s.to_string()
                            } else {
                                "unknown runtime error".to_string()
                            };
                            println!("{} {}", "runtime error:".bright_red(), msg);
                            // Broadcast :abort so workers re-sync to the wait loop.
                            broadcast_expr(":abort", &dist_transport);
                        }
                    }
                }
            }
        }

        if let Some(p) = history_path() {
            let _ = rl.save_history(&p);
        }
    } else {
        // ── Parties 1/2: silent workers ──────────────────────────────────────
        eprintln!(
            "{}",
            format!(
                "[party {}] ready, waiting for expressions from party 0...",
                me
            )
            .bright_yellow()
        );
        loop {
            let src = receive_expr(&dist_transport);
            let src = src.trim().to_string();
            if src == ":quit" {
                eprintln!(
                    "{}",
                    format!("[party {}] received :quit, exiting.", me).bright_yellow()
                );
                break;
            }
            if src == ":abort" {
                // Party 0 had a parse or runtime error; skip this round.
                eprintln!(
                    "{}",
                    format!("[party {}] received :abort, skipping expression.", me).bright_yellow()
                );
                continue;
            }
            if src == ":commit" {
                // Party 0 triggered a consensus epoch; participate.
                commit_state(&mut consensus, &mut epoch, &participant, me, &node_state, &ledger);
                continue;
            }
            if src == ":fhe" {
                if let Some(s) = settlement.as_mut() {
                    s.settle_fhe(&ledger, &node_state);
                }
                continue;
            }
            if src == ":storage" {
                if let Some(s) = settlement.as_mut() {
                    s.settle_storage(&ledger, &node_state);
                }
                continue;
            }
            eprintln!(
                "{}",
                format!("[party {}] evaluating: {}", me, src).bright_black()
            );
            match parse(&src) {
                Ok(expr) => {
                    // Wrap in catch_unwind so a protocol panic doesn't crash the worker.
                    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        participant.reset_per_expression();
                        participant.eval_quiet(&expr)
                    }));
                    match result {
                        Ok(val) => {
                            // Mirror party 0's reveal (exactly once) to stay in sync.
                            let (shown, num) = display_result(&val, &participant);
                            eprintln!(
                                "{}",
                                format!(
                                    "[party {}] => {}  (waiting for next expression)",
                                    me, shown
                                )
                                .bright_black()
                            );
                            // Settle this MPC job in lockstep with party 0.
                            if let (Some(v), Some(s)) = (num, settlement.as_mut()) {
                                s.settle_expression(v, &ledger, &node_state);
                            }
                        }
                        Err(e) => {
                            let msg = if let Some(s) = e.downcast_ref::<String>() {
                                s.clone()
                            } else if let Some(s) = e.downcast_ref::<&str>() {
                                s.to_string()
                            } else {
                                "unknown runtime error".to_string()
                            };
                            eprintln!(
                                "{}",
                                format!(
                                    "[party {}] runtime error: {}  (recovered, waiting for next)",
                                    me, msg
                                )
                                .bright_red()
                            );
                            // Wait for the :abort that party 0 will broadcast.
                            let abort = receive_expr(&dist_transport);
                            if abort.trim() == ":quit" {
                                break;
                            }
                            // any other message (including :abort) — continue the loop
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{}",
                        format!("[party {}] parse error: {}  (waiting for :abort)", me, e)
                            .bright_red()
                    );
                    // Wait for :abort from party 0 before re-entering the wait loop.
                    let abort = receive_expr(&dist_transport);
                    if abort.trim() == ":quit" {
                        break;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Legacy D-sum protocol (used by the integration test).
/// Each party broadcasts its d_value and prints `SUM: <total>`.
fn run_d_sum(me: usize, addrs: Vec<String>, d_val: u128) {
    use std::time::Duration as D;
    thread::sleep(D::from_millis(50 * me as u64));
    let transport = TcpTransport::new(me, addrs);
    transport.broadcast(serialize_msg(&MpcMsg::D(d_val)));

    let mut got = [false; 3];
    got[me] = true;
    let mut d_shares = [0u128; 3];
    d_shares[me] = d_val;
    let mut pending = 2;
    let start = std::time::Instant::now();
    while pending > 0 && start.elapsed() < Duration::from_secs(8) {
        if let Some((from, msg)) = transport.recv() {
            if let MpcMsg::D(v) = deserialize_msg(&msg) {
                if !got[from] {
                    got[from] = true;
                    d_shares[from] = v;
                    pending -= 1;
                }
            }
        } else {
            thread::sleep(Duration::from_millis(10));
        }
    }
    if pending > 0 {
        eprintln!("[party {}] timed out", me);
        std::process::exit(1);
    }
    let sum: u128 = d_shares.iter().fold(0, |a, &v| a.wrapping_add(v));
    println!("SUM: {}", sum);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // no-args: in-process demo
    if args.len() < 3 {
        run_inproc_demo();
        return;
    }

    let me: usize = args[1]
        .parse()
        .expect("first argument must be party id (0/1/2)");
    let addrs: Vec<String> = args[2].split(',').map(|s| s.to_string()).collect();
    if addrs.len() != 3 {
        eprintln!("expected 3 addresses");
        std::process::exit(1);
    }

    // legacy integration-test mode: <id> <addrs> <d_value>
    if args.len() >= 4 {
        if let Ok(d_val) = args[3].parse::<u128>() {
            run_d_sum(me, addrs, d_val);
            return;
        }
        // fourth arg is dist_addrs (MPC interactive mode)
        let dist_addrs: Vec<String> = args[3].split(',').map(|s| s.to_string()).collect();
        launch(me, addrs, dist_addrs);
        return;
    }

    // two args: derive dist_addrs by bumping port by 100
    let dist_addrs: Vec<String> = addrs
        .iter()
        .map(|addr| {
            if let Some(colon) = addr.rfind(':') {
                let port: u16 = addr[colon + 1..].parse().unwrap_or(7000);
                format!("{}:{}", &addr[..colon], port + 100)
            } else {
                format!("{}:7100", addr)
            }
        })
        .collect();
    launch(me, addrs, dist_addrs);
}
