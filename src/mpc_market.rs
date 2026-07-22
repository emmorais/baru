//! MPC job market — the `CoSigned` work-receipt model.
//!
//! An MPC job has one **owner** (who contributes the private data) and a
//! **committee** of workers who jointly run the protocol (for ABY3, 1 owner +
//! 2 workers).  Unlike an FHE job — checkable by *re-executing* a deterministic
//! circuit — an MPC computation is interactive and jointly produced, so no lone
//! verifier can reproduce it.  Correctness is instead attested by a **threshold
//! co-signature**: a threshold of the committee signs the result commitment.
//!
//! Because a dishonest minority cannot forge the honest majority's signatures,
//! a wrong result cannot reach the threshold — the honest workers simply won't
//! sign it.  A valid receipt is thus a `t`-of-`n` set of committee signatures
//! over the same `(request, result)`.  Settlement credits each co-signing worker
//! through the same [`Effect`]s as the FHE and storage markets, into the one
//! reputation ledger.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::crypto::identity::{Identity, IdentityId, IdentitySig, PublicIdentity};
use crate::fhe::lifecycle::{Effect, Stake};

/// 32-byte commitment.
pub type Hash = [u8; 32];

/// An MPC job: a program over the owner's private inputs, to be run by a
/// committee that co-signs the result.
#[derive(Clone, Debug)]
pub struct MpcJob {
    /// Commitment to the program + inputs (what the committee agreed to run).
    pub request: Hash,
    /// The data owner.
    pub owner: IdentityId,
    /// The worker committee (their identities).
    pub committee: Vec<IdentityId>,
    /// How many committee co-signatures make a result valid.
    pub threshold: usize,
}

/// One committee member's signature over a result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoSignature {
    pub signer: PublicIdentity,
    pub sig: IdentitySig,
}

/// A completed MPC job: the result commitment and the collected co-signatures.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MpcReceipt {
    pub request: Hash,
    /// Commitment to the (reconstructed / public) result.
    pub result_root: Hash,
    pub cosigners: Vec<CoSignature>,
}

/// The bytes a committee member signs to attest a result.
fn cosign_bytes(request: &Hash, result_root: &Hash) -> Vec<u8> {
    let mut v = Vec::with_capacity(64 + 16);
    v.extend_from_slice(b"baru-mpc-cosign-v1");
    v.extend_from_slice(request);
    v.extend_from_slice(result_root);
    v
}

/// A worker's commitment to the value it computed.  Every honest committee
/// member reconstructs the same result, so they all produce the same root.
pub fn result_commitment(value: i64) -> Hash {
    let mut h = Sha256::new();
    h.update(b"baru-mpc-result-v1");
    h.update(value.to_le_bytes());
    h.finalize().into()
}

/// A committee member co-signs a result.
pub fn cosign(worker: &Identity, request: &Hash, result_root: &Hash) -> CoSignature {
    CoSignature {
        signer: worker.public(),
        sig: worker.sign(&cosign_bytes(request, result_root)),
    }
}

/// Check that `cosig` is a valid signature over `(request, result_root)`.  A
/// collector uses this to keep only co-signatures that agree with the honest
/// result — a worker that computed (or signed) a *different* result is silently
/// excluded, so it earns nothing while the honest majority still settles.
pub fn verify_cosignature(cosig: &CoSignature, request: &Hash, result_root: &Hash) -> bool {
    cosig
        .signer
        .verify(&cosign_bytes(request, result_root), &cosig.sig)
}

/// Verdict on an MPC receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// A threshold of valid committee signatures over the result.
    Valid,
    /// The receipt does not answer this job.
    WrongRequest,
    /// A signer is not a committee member.
    NonCommitteeSigner,
    /// The same signer appears twice.
    DuplicateSigner,
    /// A signature did not verify.
    BadSignature,
    /// Not enough valid co-signatures.
    InsufficientCosigners { have: usize, need: usize },
}

/// Verify an MPC receipt: every co-signature must be a distinct committee
/// member with a valid signature over `(request, result_root)`, and their count
/// must meet the threshold.
pub fn verify_receipt(job: &MpcJob, receipt: &MpcReceipt) -> Verdict {
    if receipt.request != job.request {
        return Verdict::WrongRequest;
    }
    let committee: HashSet<IdentityId> = job.committee.iter().copied().collect();
    let msg = cosign_bytes(&receipt.request, &receipt.result_root);
    let mut seen: HashSet<IdentityId> = HashSet::new();
    let mut valid = 0usize;
    for cs in &receipt.cosigners {
        let id = cs.signer.id();
        if !committee.contains(&id) {
            return Verdict::NonCommitteeSigner;
        }
        if !seen.insert(id) {
            return Verdict::DuplicateSigner;
        }
        if !cs.signer.verify(&msg, &cs.sig) {
            return Verdict::BadSignature;
        }
        valid += 1;
    }
    if valid >= job.threshold {
        Verdict::Valid
    } else {
        Verdict::InsufficientCosigners {
            have: valid,
            need: job.threshold,
        }
    }
}

/// Settle an MPC job: on a valid receipt, credit each co-signing worker its
/// reward and reputation.  Returns the verdict and the settlement effects.
pub fn settle(job: &MpcJob, receipt: &MpcReceipt, reward_per_worker: Stake) -> (Verdict, Vec<Effect>) {
    let verdict = verify_receipt(job, receipt);
    if verdict != Verdict::Valid {
        return (verdict, vec![]);
    }
    let mut effects = Vec::new();
    for cs in &receipt.cosigners {
        let id = cs.signer.id().0;
        effects.push(Effect::Release {
            to: id,
            amount: reward_per_worker,
        });
        effects.push(Effect::Reputation {
            id,
            delta: reward_per_worker as i128,
        });
    }
    (Verdict::Valid, effects)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(n: u8) -> Identity {
        Identity::from_secret_bytes([n; 32])
    }

    fn job_of(committee: &[&Identity], threshold: usize) -> MpcJob {
        MpcJob {
            request: [1u8; 32],
            owner: committee[0].id(),
            committee: committee.iter().map(|w| w.id()).collect(),
            threshold,
        }
    }

    #[test]
    fn threshold_of_valid_cosignatures_is_valid() {
        let (a, b, c) = (worker(1), worker(2), worker(3));
        let job = job_of(&[&a, &b, &c], 2);
        let result = result_commitment(42);
        let receipt = MpcReceipt {
            request: job.request,
            result_root: result,
            cosigners: vec![
                cosign(&a, &job.request, &result),
                cosign(&b, &job.request, &result),
            ],
        };
        assert_eq!(verify_receipt(&job, &receipt), Verdict::Valid);
    }

    #[test]
    fn below_threshold_is_rejected() {
        let (a, b, c) = (worker(1), worker(2), worker(3));
        let job = job_of(&[&a, &b, &c], 2);
        let result = result_commitment(42);
        let receipt = MpcReceipt {
            request: job.request,
            result_root: result,
            cosigners: vec![cosign(&a, &job.request, &result)],
        };
        assert_eq!(
            verify_receipt(&job, &receipt),
            Verdict::InsufficientCosigners { have: 1, need: 2 }
        );
    }

    #[test]
    fn a_dishonest_minority_cannot_forge_a_different_result() {
        // The committee agreed on 42; a lone cheat wants 99 accepted.  It can
        // only sign its own — it cannot forge the others', so 99 never reaches
        // the threshold.
        let (a, b, c) = (worker(1), worker(2), worker(3));
        let job = job_of(&[&a, &b, &c], 2);
        let bogus = result_commitment(99);
        let receipt = MpcReceipt {
            request: job.request,
            result_root: bogus,
            cosigners: vec![cosign(&c, &job.request, &bogus)],
        };
        assert!(matches!(
            verify_receipt(&job, &receipt),
            Verdict::InsufficientCosigners { .. }
        ));
    }

    #[test]
    fn non_committee_and_duplicate_and_bad_sig_rejected() {
        let (a, b, c) = (worker(1), worker(2), worker(3));
        let outsider = worker(9);
        let job = job_of(&[&a, &b, &c], 2);
        let result = result_commitment(7);

        // Outsider signer.
        let r1 = MpcReceipt {
            request: job.request,
            result_root: result,
            cosigners: vec![
                cosign(&a, &job.request, &result),
                cosign(&outsider, &job.request, &result),
            ],
        };
        assert_eq!(verify_receipt(&job, &r1), Verdict::NonCommitteeSigner);

        // Duplicate signer.
        let r2 = MpcReceipt {
            request: job.request,
            result_root: result,
            cosigners: vec![
                cosign(&a, &job.request, &result),
                cosign(&a, &job.request, &result),
            ],
        };
        assert_eq!(verify_receipt(&job, &r2), Verdict::DuplicateSigner);

        // Tampered signature.
        let mut r3 = MpcReceipt {
            request: job.request,
            result_root: result,
            cosigners: vec![
                cosign(&a, &job.request, &result),
                cosign(&b, &job.request, &result),
            ],
        };
        r3.cosigners[1].sig.0[0] ^= 0xff;
        assert_eq!(verify_receipt(&job, &r3), Verdict::BadSignature);
    }

    #[test]
    fn settlement_credits_each_worker() {
        use crate::reputation::ReputationLedger;
        let (a, b, c) = (worker(1), worker(2), worker(3));
        let job = job_of(&[&a, &b, &c], 2);
        let result = result_commitment(42);
        let receipt = MpcReceipt {
            request: job.request,
            result_root: result,
            cosigners: vec![
                cosign(&a, &job.request, &result),
                cosign(&b, &job.request, &result),
            ],
        };
        let (verdict, effects) = settle(&job, &receipt, 100);
        assert_eq!(verdict, Verdict::Valid);
        let mut ledger = ReputationLedger::new();
        ledger.apply_all(&effects);
        assert_eq!(ledger.reputation(&a.id().0), 100);
        assert_eq!(ledger.reputation(&b.id().0), 100);
        // The non-signing member earned nothing.
        assert_eq!(ledger.reputation(&c.id().0), 0);
    }

    /// End-to-end grounding on a real ABY3 computation: three in-process
    /// parties jointly compute `6 * 7`, each reconstructs `42`, each co-signs
    /// the result, and a 2-of-3 threshold receipt verifies and settles.
    #[test]
    fn real_aby3_computation_cosigned_and_settled() {
        use crate::crypto::net::InprocHandle;
        use crate::interpreter::{parse, Participant, Types};
        use crate::reputation::ReputationLedger;
        use std::sync::mpsc;
        use std::thread;

        let request = [7u8; 32];
        let handles = InprocHandle::create(3);
        let (tx, rx) = mpsc::channel();
        let mut joins = Vec::new();
        for (i, h) in handles.into_iter().enumerate() {
            let tx = tx.clone();
            joins.push(thread::spawn(move || {
                let participant = Participant::new_transport(i, h);
                let expr = parse("(reconstruct (* (share 6 32) (share 7 32)))").unwrap();
                let value: i64 = match participant.eval_quiet(&expr) {
                    Types::Num(n) => (n as u32) as i32 as i64,
                    Types::Share(sr) => (participant.reveal(&sr) as u32) as i32 as i64,
                    other => panic!("unexpected result: {:?}", other),
                };
                // Each worker co-signs the result it reconstructed.
                let me = Identity::from_secret_bytes([100 + i as u8; 32]);
                let root = result_commitment(value);
                tx.send((i, cosign(&me, &request, &root), value)).unwrap();
            }));
        }
        drop(tx);
        for j in joins {
            j.join().unwrap();
        }
        let mut collected: Vec<_> = rx.iter().collect();
        collected.sort_by_key(|(i, _, _)| *i);

        // All parties agreed on 42.
        assert!(collected.iter().all(|(_, _, v)| *v == 42));
        let result_root = result_commitment(42);

        // Committee = the three party identities; require 2-of-3.
        let committee: Vec<IdentityId> = (0..3)
            .map(|i| Identity::from_secret_bytes([100 + i as u8; 32]).id())
            .collect();
        let job = MpcJob {
            request,
            owner: committee[0],
            committee,
            threshold: 2,
        };
        // Parties 0 and 1 co-sign (deterministic after the sort).
        let receipt = MpcReceipt {
            request,
            result_root,
            cosigners: collected.iter().take(2).map(|(_, cs, _)| cs.clone()).collect(),
        };

        assert_eq!(verify_receipt(&job, &receipt), Verdict::Valid);
        let (_v, effects) = settle(&job, &receipt, 50);
        let mut ledger = ReputationLedger::new();
        ledger.apply_all(&effects);
        // The co-signing workers earned reputation for the real MPC job.
        assert!(ledger.reputation(&job.committee[0].0) > 0);
    }
}
