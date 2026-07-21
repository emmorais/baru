//! Storage market — Proof of Retrievability grounded on the Merkle tree.
//!
//! A provider agrees to hold a dataset whose content is committed by a Merkle
//! [`root`](StoredData::root) over its chunks.  Storage is *continuous*, not
//! one-shot: each epoch the network issues a fresh, unpredictable challenge — a
//! chunk index derived from epoch randomness (a VRF / the consensus group root)
//! — and the provider must answer with the **actual chunk bytes plus an
//! inclusion proof** to the committed root.  Verification is a `log n` Merkle
//! check; because the provider must produce the real bytes (not just the root),
//! this is a spot-check of retrievability.  Passing an epoch accrues reputation,
//! so a reliable provider earns a *continuous* stream — settled through the same
//! [`Effect`]s as the FHE market, into the same reputation ledger.
//!
//! Design notes carried over from the market design:
//! * The challenge is derived from epoch randomness known only at the epoch, so
//!   a provider cannot pre-fetch just the challenged chunk — it must hold all of
//!   them.  High-entropy data (e.g. FHE ciphertexts) also defeats the
//!   regenerate-on-demand attack.
//! * This proves *retrievability from a provider*, not distinct physical
//!   replicas.  Redundancy is achieved by rewarding **distinct provider
//!   identities** holding the same root, not by proving one identity holds N
//!   copies (true Proof-of-Replication is future work).

use sha2::{Digest, Sha256};

use crate::fhe::lifecycle::{Effect, Epoch, Id, Stake};
use crate::interpreter::merkle::{merkle_proof, merkle_root, verify_proof, ProofStep};

/// 32-byte commitment.
pub type Hash = [u8; 32];

/// Hash a chunk into its Merkle leaf.
fn chunk_leaf(chunk: &[u8]) -> Hash {
    let mut h = Sha256::new();
    h.update(chunk);
    h.finalize().into()
}

/// A dataset stored as an ordered list of chunks, committed by a Merkle root.
#[derive(Clone, Debug)]
pub struct StoredData {
    chunks: Vec<Vec<u8>>,
}

impl StoredData {
    pub fn new(chunks: Vec<Vec<u8>>) -> Self {
        StoredData { chunks }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    fn leaves(&self) -> Vec<Hash> {
        self.chunks.iter().map(|c| chunk_leaf(c)).collect()
    }

    /// The retrievability commitment for this dataset.
    pub fn root(&self) -> Hash {
        merkle_root(&self.leaves())
    }

    /// Answer a challenge for chunk `index`: the chunk bytes and its inclusion
    /// proof.  Returns `None` if the index is out of range — i.e. the provider
    /// does not hold that chunk.
    pub fn answer(&self, index: usize) -> Option<ChallengeResponse> {
        let chunk = self.chunks.get(index)?.clone();
        let proof = merkle_proof(&self.leaves(), index)?;
        Some(ChallengeResponse { chunk, proof })
    }
}

/// A provider's response to a storage challenge.
#[derive(Clone, Debug)]
pub struct ChallengeResponse {
    pub chunk: Vec<u8>,
    pub proof: Vec<ProofStep>,
}

/// The chunk index a provider must prove this epoch: derived from the committed
/// `root`, the `epoch`, and epoch randomness `seed` (a VRF / consensus group
/// root stand-in), so it is unpredictable before the epoch.
pub fn challenge_index(root: &Hash, epoch: Epoch, seed: &[u8; 32], n_chunks: usize) -> usize {
    if n_chunks == 0 {
        return 0;
    }
    let mut h = Sha256::new();
    h.update(seed);
    h.update(root);
    h.update(epoch.to_le_bytes());
    let digest: [u8; 32] = h.finalize().into();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&digest[..8]);
    (u64::from_le_bytes(buf) as usize) % n_chunks
}

/// Verify a provider's challenge response against the committed `root`: the
/// chunk must hash to the leaf that the proof places at the challenged index.
pub fn verify_response(root: &Hash, response: &ChallengeResponse) -> bool {
    verify_proof(&chunk_leaf(&response.chunk), &response.proof, root)
}

/// A continuous storage agreement: `provider` holds the data committed by
/// `data_root` (with `n_chunks` chunks) and earns `reward_per_epoch` for each
/// epoch it answers the challenge, up to `until_epoch`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageDeal {
    pub data_root: Hash,
    pub n_chunks: usize,
    pub owner: Id,
    pub provider: Id,
    pub reward_per_epoch: Stake,
    pub until_epoch: Epoch,
}

impl StorageDeal {
    pub fn is_active(&self, epoch: Epoch) -> bool {
        epoch < self.until_epoch
    }

    /// The index the provider must prove this epoch.
    pub fn challenge(&self, epoch: Epoch, seed: &[u8; 32]) -> usize {
        challenge_index(&self.data_root, epoch, seed, self.n_chunks)
    }
}

/// Settle one epoch of a deal.  Verifies the provider answered *this epoch's*
/// challenge index and emits the settlement effects: a reward and a reputation
/// credit on success (continuous accrual), nothing on a miss.  Returns whether
/// the epoch was proven.
pub fn settle_epoch(
    deal: &StorageDeal,
    epoch: Epoch,
    seed: &[u8; 32],
    response: Option<&ChallengeResponse>,
) -> (bool, Vec<Effect>) {
    if !deal.is_active(epoch) {
        return (false, vec![]);
    }
    let expected = deal.challenge(epoch, seed);
    // The response must be for *this epoch's* challenged index (so a provider
    // can't replay one chunk's proof for every challenge) *and* verify against
    // the committed root.
    let proven = match response {
        Some(r) => {
            proof_matches_index(&r.proof, expected, deal.n_chunks)
                && verify_response(&deal.data_root, r)
        }
        None => false,
    };
    if proven {
        (
            true,
            vec![
                Effect::Release {
                    to: deal.provider,
                    amount: deal.reward_per_epoch,
                },
                Effect::Reputation {
                    id: deal.provider,
                    delta: deal.reward_per_epoch as i128,
                },
            ],
        )
    } else {
        (false, vec![])
    }
}

/// Check that an inclusion proof is for leaf position `index` in a tree of
/// `n_chunks` leaves.  Replicates `merkle_proof`'s index walk exactly —
/// including the promote-lone-node rule that *skips* a step at a level with no
/// sibling — so the expected sequence of sibling sides is reconstructed and
/// compared.  This binds a response to the challenged index.
fn proof_matches_index(proof: &[ProofStep], index: usize, n_chunks: usize) -> bool {
    let mut idx = index;
    let mut level_len = n_chunks;
    let mut expected_sides = Vec::new();
    while level_len > 1 {
        if idx ^ 1 < level_len {
            // A sibling exists at this level; sibling is on the right iff our
            // node is the (even) left child.
            expected_sides.push(idx.is_multiple_of(2));
        }
        idx /= 2;
        level_len = level_len.div_ceil(2);
    }
    proof.len() == expected_sides.len()
        && proof
            .iter()
            .zip(expected_sides)
            .all(|(step, side)| step.sibling_on_right == side)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> StoredData {
        // Distinct, high-entropy-ish chunks.
        StoredData::new(
            (0..n)
                .map(|i| {
                    let mut h = Sha256::new();
                    h.update(b"chunk");
                    h.update((i as u64).to_le_bytes());
                    h.finalize().to_vec()
                })
                .collect(),
        )
    }

    fn id(n: u8) -> Id {
        [n; 32]
    }

    #[test]
    fn honest_answer_verifies() {
        let d = data(8);
        let root = d.root();
        for i in 0..8 {
            let resp = d.answer(i).unwrap();
            assert!(verify_response(&root, &resp), "index {i}");
        }
    }

    #[test]
    fn wrong_chunk_or_root_rejected() {
        let d = data(8);
        let root = d.root();
        let mut resp = d.answer(3).unwrap();
        assert!(verify_response(&root, &resp));
        // Tamper the chunk bytes → different leaf → fails.
        resp.chunk[0] ^= 0xff;
        assert!(!verify_response(&root, &resp));
        // Wrong root → fails.
        let good = d.answer(3).unwrap();
        assert!(!verify_response(&[0u8; 32], &good));
    }

    #[test]
    fn provider_without_the_chunk_cannot_answer() {
        let d = data(4);
        // Out-of-range index: the provider does not hold this chunk.
        assert!(d.answer(4).is_none());
    }

    #[test]
    fn challenge_is_deterministic_and_in_range() {
        let d = data(16);
        let root = d.root();
        let seed = [9u8; 32];
        for epoch in 0..20 {
            let i = challenge_index(&root, epoch, &seed, 16);
            assert!(i < 16);
            assert_eq!(i, challenge_index(&root, epoch, &seed, 16));
        }
    }

    #[test]
    fn answering_a_stale_index_does_not_settle() {
        // A provider that replays last epoch's proof (wrong index) is not paid.
        let d = data(16);
        let deal = StorageDeal {
            data_root: d.root(),
            n_chunks: 16,
            owner: id(1),
            provider: id(2),
            reward_per_epoch: 100,
            until_epoch: 100,
        };
        let seed = [5u8; 32];
        let epoch = 7;
        let this_idx = deal.challenge(epoch, &seed);
        let stale_idx = deal.challenge(epoch + 1, &seed);
        if this_idx == stale_idx {
            return; // rare collision; nothing to test this seed/epoch
        }
        let stale = d.answer(stale_idx).unwrap();
        let (proven, effects) = settle_epoch(&deal, epoch, &seed, Some(&stale));
        assert!(!proven);
        assert!(effects.is_empty());
    }

    #[test]
    fn works_and_binds_index_for_non_power_of_two() {
        for n in [3usize, 5, 7, 9, 13] {
            let d = data(n);
            let deal = StorageDeal {
                data_root: d.root(),
                n_chunks: n,
                owner: id(1),
                provider: id(2),
                reward_per_epoch: 10,
                until_epoch: 100,
            };
            let seed = [4u8; 32];
            for epoch in 0..10u64 {
                let idx = deal.challenge(epoch, &seed);
                let good = d.answer(idx).unwrap();
                assert!(settle_epoch(&deal, epoch, &seed, Some(&good)).0, "n={n} epoch={epoch}");
                // A valid proof for a *different* index must not settle.
                let other = (idx + 1) % n;
                let wrong = d.answer(other).unwrap();
                assert!(
                    !settle_epoch(&deal, epoch, &seed, Some(&wrong)).0,
                    "n={n} epoch={epoch}: index binding failed"
                );
            }
        }
    }

    #[test]
    fn continuous_storage_accrues_reputation() {
        use crate::reputation::ReputationLedger;

        let d = data(32);
        let deal = StorageDeal {
            data_root: d.root(),
            n_chunks: 32,
            owner: id(1),
            provider: id(2),
            reward_per_epoch: 50,
            until_epoch: 10,
        };
        let seed = [3u8; 32];
        let mut ledger = ReputationLedger::new();

        // Provider proves storage every epoch → continuous reputation accrual.
        for epoch in 0..10 {
            let idx = deal.challenge(epoch, &seed);
            let resp = d.answer(idx).unwrap();
            let (proven, effects) = settle_epoch(&deal, epoch, &seed, Some(&resp));
            assert!(proven, "epoch {epoch} should prove");
            ledger.apply_all(&effects);
        }
        assert_eq!(ledger.reputation(&id(2)), 50 * 10);
        assert_eq!(ledger.balance(&id(2)), 50 * 10);

        // A missed epoch earns nothing (no reward, no reputation).
        let (proven, effects) = settle_epoch(&deal, 5, &seed, None);
        assert!(!proven);
        assert!(effects.is_empty());
    }

    #[test]
    fn storage_reputation_makes_a_provider_committee_eligible() {
        use crate::reputation::ReputationLedger;

        // Storage work feeds the same reputation system as FHE work: a provider
        // that proves storage becomes eligible for the consensus committee.
        let d = data(16);
        let deal = StorageDeal {
            data_root: d.root(),
            n_chunks: 16,
            owner: id(1),
            provider: id(7),
            reward_per_epoch: 100,
            until_epoch: 5,
        };
        let seed = [1u8; 32];
        let mut ledger = ReputationLedger::new();
        for epoch in 0..5 {
            let idx = deal.challenge(epoch, &seed);
            let resp = d.answer(idx).unwrap();
            let (_ok, effects) = settle_epoch(&deal, epoch, &seed, Some(&resp));
            ledger.apply_all(&effects);
        }
        // The storage provider is now selectable as a committee member.
        for s in 0u8..20 {
            let committee = ledger.select_committee(1, &[s; 32]);
            assert_eq!(committee, vec![id(7)]);
        }
    }
}
