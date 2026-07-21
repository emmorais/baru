//! Pluggable consensus over per-epoch state roots.
//!
//! Time is split into **epochs**.  At each boundary participants agree on the
//! committed roots for that epoch: every participant contributes a [`RootSet`]
//! — its `(private_root, public_root, state_root)` from the hash-consed store —
//! and consensus produces an [`EpochCommit`] that all honest participants
//! adopt, including a single **group root** (a Merkle commitment over everyone's
//! roots) that succinctly names the whole network's agreed state.
//!
//! The mechanism is swappable behind [`ConsensusBackend`].  [`SingleLeader`] is
//! the parity baseline — it reproduces today's "party 0 collects and dictates"
//! behavior — and is the migration bridge for later BFT / reputation-weighted
//! backends, which implement the same trait.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::crypto::net::{deserialize_msg, serialize_msg, PartyId, Transport};
use crate::interpreter::merkle::merkle_root;

/// A 32-byte content commitment (same shape as the store's `ZHash`).
pub type Hash = [u8; 32];

/// Monotonic epoch counter.
pub type Epoch = u64;

/// The roots a participant commits for one epoch.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct RootSet {
    /// Commitment to this participant's private data tree.
    pub private_root: Hash,
    /// Commitment to this participant's public data tree.
    pub public_root: Hash,
    /// Commitment to this participant's compute state `(env, expr[, cont])`.
    pub state_root: Hash,
}

impl RootSet {
    /// A single leaf hash summarizing this root set, used to build the group root.
    pub fn leaf(&self) -> Hash {
        let mut h = Sha256::new();
        h.update(self.private_root);
        h.update(self.public_root);
        h.update(self.state_root);
        h.finalize().into()
    }
}

/// The agreed outcome of one epoch.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct EpochCommit {
    pub epoch: Epoch,
    /// Agreed root set per participant (indexed by party id).
    pub party_roots: Vec<RootSet>,
    /// Merkle commitment over all participants' root sets — one hash naming the
    /// entire network's agreed state for this epoch.
    pub group_root: Hash,
}

impl EpochCommit {
    /// Build a commit from the collected per-party roots, computing the group root.
    pub fn new(epoch: Epoch, party_roots: Vec<RootSet>) -> Self {
        let leaves: Vec<Hash> = party_roots.iter().map(RootSet::leaf).collect();
        let group_root = merkle_root(&leaves);
        EpochCommit {
            epoch,
            party_roots,
            group_root,
        }
    }
}

/// Wire messages for the consensus transport.
#[derive(Clone, Serialize, Deserialize)]
enum ConsensusMsg {
    /// A follower's root set for an epoch, sent to the leader.
    Submit {
        epoch: Epoch,
        from: PartyId,
        roots: RootSet,
    },
    /// The leader's decided commit for an epoch, broadcast to all.
    Commit(EpochCommit),
}

/// Consensus failures.
#[derive(Debug, PartialEq, Eq)]
pub enum ConsensusError {
    /// Agreement did not complete within the timeout.
    Timeout,
}

impl std::fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConsensusError::Timeout => write!(f, "consensus timed out"),
        }
    }
}

impl std::error::Error for ConsensusError {}

/// A swappable consensus mechanism.  Implementors agree on the committed roots
/// for an epoch; every honest participant returns the same [`EpochCommit`].
pub trait ConsensusBackend {
    /// Run agreement for `epoch` given this participant's roots.  Blocking.
    fn commit_epoch(&mut self, epoch: Epoch, my: RootSet) -> Result<EpochCommit, ConsensusError>;

    /// Human-readable backend name.
    fn name(&self) -> &'static str;
}

/// Single-leader consensus: one designated leader collects every participant's
/// root set, forms the [`EpochCommit`], and broadcasts it; followers submit and
/// adopt.  Reproduces the current centralized behavior as a parity baseline.
pub struct SingleLeader<T: Transport> {
    me: PartyId,
    n: usize,
    leader: PartyId,
    transport: T,
    timeout_ms: u64,
    /// Buffer for consensus messages that arrived out of turn (e.g. a later
    /// epoch's), so they are not dropped.
    buf: VecDeque<(PartyId, ConsensusMsg)>,
}

impl<T: Transport> SingleLeader<T> {
    /// Create a single-leader backend over a dedicated consensus transport.
    pub fn new(me: PartyId, n: usize, leader: PartyId, transport: T) -> Self {
        SingleLeader {
            me,
            n,
            leader,
            transport,
            timeout_ms: 10_000,
            buf: VecDeque::new(),
        }
    }

    /// Override the agreement timeout (default 10s).
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Receive the next consensus message satisfying `pred`, buffering others.
    fn recv_matching<F>(&mut self, deadline: Instant, pred: F) -> Option<(PartyId, ConsensusMsg)>
    where
        F: Fn(&ConsensusMsg) -> bool,
    {
        if let Some(pos) = self.buf.iter().position(|(_, m)| pred(m)) {
            return self.buf.remove(pos);
        }
        loop {
            if let Some((from, bytes)) = self.transport.recv() {
                let m: ConsensusMsg = deserialize_msg(&bytes);
                if pred(&m) {
                    return Some((from, m));
                }
                self.buf.push_back((from, m));
            } else if Instant::now() >= deadline {
                return None;
            } else {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

impl<T: Transport> ConsensusBackend for SingleLeader<T> {
    fn commit_epoch(&mut self, epoch: Epoch, my: RootSet) -> Result<EpochCommit, ConsensusError> {
        let deadline = Instant::now() + Duration::from_millis(self.timeout_ms);

        if self.me == self.leader {
            // Collect every follower's root set for this epoch.
            let mut party_roots = vec![RootSet::default(); self.n];
            party_roots[self.me] = my;
            let mut have = vec![false; self.n];
            have[self.me] = true;
            let mut remaining = self.n - 1;

            while remaining > 0 {
                match self.recv_matching(deadline, |m| {
                    matches!(m, ConsensusMsg::Submit { epoch: e, .. } if *e == epoch)
                }) {
                    Some((_from, ConsensusMsg::Submit { from, roots, .. })) => {
                        if from < self.n && !have[from] {
                            party_roots[from] = roots;
                            have[from] = true;
                            remaining -= 1;
                        }
                    }
                    _ => return Err(ConsensusError::Timeout),
                }
            }

            let commit = EpochCommit::new(epoch, party_roots);
            self.transport
                .broadcast(serialize_msg(&ConsensusMsg::Commit(commit.clone())));
            Ok(commit)
        } else {
            // Submit to the leader and adopt its decision.
            self.transport.send(
                self.leader,
                serialize_msg(&ConsensusMsg::Submit {
                    epoch,
                    from: self.me,
                    roots: my,
                }),
            );
            match self.recv_matching(deadline, |m| {
                matches!(m, ConsensusMsg::Commit(c) if c.epoch == epoch)
            }) {
                Some((_from, ConsensusMsg::Commit(c))) => Ok(c),
                _ => Err(ConsensusError::Timeout),
            }
        }
    }

    fn name(&self) -> &'static str {
        "single-leader"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;
    use std::sync::mpsc;
    use std::thread;

    fn roots_for(party: usize) -> RootSet {
        let b = party as u8 + 1;
        RootSet {
            private_root: [b; 32],
            public_root: [b.wrapping_mul(2); 32],
            state_root: [b.wrapping_mul(3); 32],
        }
    }

    /// Run `epochs` rounds of single-leader consensus across `n` in-process
    /// parties and return each party's sequence of commits.
    fn run(n: usize, leader: usize, epochs: u64) -> Vec<Vec<EpochCommit>> {
        let handles = InprocHandle::create(n);
        let (tx, rx) = mpsc::channel();
        let mut joins = Vec::new();
        for (me, h) in handles.into_iter().enumerate() {
            let tx = tx.clone();
            joins.push(thread::spawn(move || {
                let mut backend = SingleLeader::new(me, n, leader, h).with_timeout(5_000);
                let mut commits = Vec::new();
                for e in 0..epochs {
                    let c = backend.commit_epoch(e, roots_for(me)).expect("commit");
                    commits.push(c);
                }
                tx.send((me, commits)).unwrap();
            }));
        }
        drop(tx);
        for j in joins {
            j.join().unwrap();
        }
        let mut by_party: Vec<Option<Vec<EpochCommit>>> = (0..n).map(|_| None).collect();
        for (me, commits) in rx {
            by_party[me] = Some(commits);
        }
        by_party.into_iter().map(|c| c.unwrap()).collect()
    }

    #[test]
    fn all_parties_agree_on_the_same_commit() {
        let all = run(3, 0, 1);
        let reference = &all[0][0];
        for party in &all {
            assert_eq!(&party[0], reference, "parties disagree");
        }
    }

    #[test]
    fn commit_records_each_partys_roots() {
        let all = run(3, 0, 1);
        let commit = &all[0][0];
        for party in 0..3 {
            assert_eq!(commit.party_roots[party], roots_for(party));
        }
    }

    #[test]
    fn group_root_matches_independent_merkle() {
        let all = run(3, 0, 1);
        let commit = &all[1][0];
        let leaves: Vec<Hash> = (0..3).map(|p| roots_for(p).leaf()).collect();
        assert_eq!(commit.group_root, merkle_root(&leaves));
    }

    #[test]
    fn multiple_epochs_stay_in_agreement() {
        let all = run(3, 0, 3);
        for epoch in 0..3usize {
            let reference = &all[0][epoch];
            assert_eq!(reference.epoch, epoch as u64);
            for party in &all {
                assert_eq!(&party[epoch], reference);
            }
        }
    }

    #[test]
    fn different_roots_change_the_group_root() {
        // Sanity: an epoch's group root depends on the contributed roots.
        let leaves_a: Vec<Hash> = (0..3).map(|p| roots_for(p).leaf()).collect();
        let mut leaves_b = leaves_a.clone();
        leaves_b[1] = RootSet::default().leaf();
        assert_ne!(merkle_root(&leaves_a), merkle_root(&leaves_b));
    }
}
