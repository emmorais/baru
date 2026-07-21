//! Reputation ledger — the ReputationPoUW tie-in between verified useful work
//! and consensus weight.
//!
//! Settling a job (see [`crate::fhe::lifecycle`]) emits [`Effect`]s: reputation
//! deltas and stake moves.  This ledger folds those effects into per-identity
//! balances that every node computes identically from the agreed event log, so
//! its [`root`](ReputationLedger::root) is a group commitment consensus agrees
//! on each epoch.  Reputation — earned only from work others requested and that
//! survived re-execution — then **weights committee selection**
//! ([`ReputationLedger::select_committee`]), so doing genuine cryptographic work
//! is what buys a say in the network.  Reputation **decays** each epoch, so
//! standing must be continually re-earned rather than accumulated once.
//!
//! Folding is integer-only and order-independent over a `BTreeMap`, so all nodes
//! derive the same ledger and the same root.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::fhe::lifecycle::{Effect, Id, Stake};
use crate::interpreter::merkle::merkle_root;

/// 32-byte commitment.
pub type Hash = [u8; 32];

/// Per-identity reputation and stake, plus the treasury.
#[derive(Debug, Default, Clone)]
pub struct ReputationLedger {
    reputation: BTreeMap<Id, i128>,
    balance: BTreeMap<Id, Stake>,
    treasury: Stake,
}

impl ReputationLedger {
    pub fn new() -> Self {
        ReputationLedger::default()
    }

    /// Fold one settlement effect into the ledger.
    pub fn apply(&mut self, effect: &Effect) {
        match effect {
            Effect::Reputation { id, delta } => {
                *self.reputation.entry(*id).or_insert(0) += *delta;
            }
            Effect::Release { to, amount } | Effect::Refund { to, amount } => {
                *self.balance.entry(*to).or_insert(0) += *amount;
            }
            Effect::Slash { .. } => {
                // Slashed stake was escrowed (already out of the payer's balance);
                // it leaves circulation except for any challenger cut, which is
                // paid via a separate Release effect.
            }
            Effect::Treasury { amount } => {
                self.treasury += *amount;
            }
        }
    }

    /// Fold a batch of effects (e.g. one job's settlement).
    pub fn apply_all(&mut self, effects: &[Effect]) {
        for e in effects {
            self.apply(e);
        }
    }

    /// Decay all reputation by `retain_bps / 10_000` (applied once per epoch).
    /// Prevents rich-get-richer lock-in and forces continued participation.
    pub fn decay(&mut self, retain_bps: u32) {
        for r in self.reputation.values_mut() {
            *r = *r * retain_bps as i128 / 10_000;
        }
    }

    /// Reputation of `id` (0 if unknown; may be negative after slashing).
    pub fn reputation(&self, id: &Id) -> i128 {
        self.reputation.get(id).copied().unwrap_or(0)
    }

    /// Credited balance of `id`.
    pub fn balance(&self, id: &Id) -> Stake {
        self.balance.get(id).copied().unwrap_or(0)
    }

    pub fn treasury(&self) -> Stake {
        self.treasury
    }

    /// Canonical Merkle commitment over the ledger: one leaf per identity
    /// `SHA-256(id ‖ reputation ‖ balance)`, in id order (plus the treasury).
    pub fn root(&self) -> Hash {
        let ids: std::collections::BTreeSet<Id> = self
            .reputation
            .keys()
            .chain(self.balance.keys())
            .copied()
            .collect();
        let mut leaves: Vec<Hash> = Vec::with_capacity(ids.len() + 1);
        for id in &ids {
            let mut h = Sha256::new();
            h.update(id);
            h.update(self.reputation(id).to_le_bytes());
            h.update(self.balance(id).to_le_bytes());
            leaves.push(h.finalize().into());
        }
        // Fold the treasury in as a final leaf so it is committed too.
        let mut th = Sha256::new();
        th.update(b"treasury");
        th.update(self.treasury.to_le_bytes());
        leaves.push(th.finalize().into());
        merkle_root(&leaves)
    }

    /// Select a committee of up to `size` identities, sampled **without
    /// replacement, weighted by reputation**, deterministically from `seed`
    /// (a VRF/epoch-randomness stand-in).  Only positive-reputation identities
    /// are eligible.
    pub fn select_committee(&self, size: usize, seed: &[u8; 32]) -> Vec<Id> {
        let mut pool: Vec<(Id, u128)> = self
            .reputation
            .iter()
            .filter(|(_, r)| **r > 0)
            .map(|(id, r)| (*id, *r as u128))
            .collect();

        let mut chosen = Vec::new();
        let mut draw: u64 = 0;
        while chosen.len() < size && !pool.is_empty() {
            let total: u128 = pool.iter().map(|(_, w)| *w).sum();
            let pick = draw_below(seed, draw, total);
            draw += 1;
            let mut acc = 0u128;
            let mut idx = 0;
            for (i, (_, w)) in pool.iter().enumerate() {
                acc += *w;
                if pick < acc {
                    idx = i;
                    break;
                }
            }
            chosen.push(pool[idx].0);
            pool.remove(idx);
        }
        chosen
    }
}

/// Deterministic pseudo-random value in `[0, bound)` from `(seed, counter)`.
fn draw_below(seed: &[u8; 32], counter: u64, bound: u128) -> u128 {
    if bound == 0 {
        return 0;
    }
    let mut h = Sha256::new();
    h.update(seed);
    h.update(counter.to_le_bytes());
    let digest: [u8; 32] = h.finalize().into();
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&digest[..16]);
    u128::from_le_bytes(buf) % bound
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::backend::{FheBackend, FheCircuit};
    use crate::fhe::lifecycle::{
        apply, on_epoch, verdict_to_adjudication, Event, JobHeader, JobState, MarketConfig, Outcome,
    };
    use crate::fhe::market::{evaluate_job, verify_receipt, FheJob, WorkReceipt};
    use crate::fhe::mock_backend::MockFheBackend;
    use crate::interpreter::parse;
    use crate::zkp::circuit_compiler::CircuitCompiler;
    use std::collections::HashMap;

    fn id(n: u8) -> Id {
        [n; 32]
    }

    fn bps(amount: Stake, bps: u32) -> Stake {
        amount * bps as u128 / 10_000
    }

    fn fhe_job(be: &MockFheBackend, ck: &[u8], x: u64, y: u64) -> FheJob {
        let expr = parse("(+ x y)").unwrap();
        let mut circuit: FheCircuit = CircuitCompiler::new().compile_fhe(&expr).unwrap();
        circuit.public_inputs.clear();
        circuit.encrypted_inputs = vec!["x".into(), "y".into()];
        let mut enc = HashMap::new();
        enc.insert("x".to_string(), be.serialize_ct(&be.encrypt(ck, x)));
        enc.insert("y".to_string(), be.serialize_ct(&be.encrypt(ck, y)));
        FheJob {
            circuit,
            public_inputs: HashMap::new(),
            encrypted_inputs: enc,
            backend_id: be.backend_id().to_string(),
        }
    }

    fn header(job: &FheJob, requester: Id, reward: Stake) -> JobHeader {
        JobHeader {
            id: [0u8; 32],
            requester,
            reward,
            request: job.request_hash(),
            compute_ttl: 4,
            posted: 0,
        }
    }

    #[test]
    fn folds_reputation_and_treasury() {
        let mut led = ReputationLedger::new();
        led.apply_all(&[
            Effect::Reputation { id: id(1), delta: 100 },
            Effect::Reputation { id: id(1), delta: 50 },
            Effect::Reputation { id: id(2), delta: -30 },
            Effect::Treasury { amount: 20 },
            Effect::Release { to: id(1), amount: 500 },
        ]);
        assert_eq!(led.reputation(&id(1)), 150);
        assert_eq!(led.reputation(&id(2)), -30);
        assert_eq!(led.balance(&id(1)), 500);
        assert_eq!(led.treasury(), 20);
    }

    #[test]
    fn fold_is_order_independent() {
        let effects = [
            Effect::Reputation { id: id(1), delta: 10 },
            Effect::Reputation { id: id(2), delta: 20 },
            Effect::Treasury { amount: 5 },
        ];
        let mut a = ReputationLedger::new();
        a.apply_all(&effects);
        let mut b = ReputationLedger::new();
        for e in effects.iter().rev() {
            b.apply(e);
        }
        assert_eq!(a.root(), b.root());
    }

    #[test]
    fn decay_reduces_reputation() {
        let mut led = ReputationLedger::new();
        led.apply(&Effect::Reputation { id: id(1), delta: 1000 });
        led.decay(9000); // retain 90%
        assert_eq!(led.reputation(&id(1)), 900);
    }

    #[test]
    fn committee_selection_is_deterministic_and_weighted() {
        let mut led = ReputationLedger::new();
        led.apply(&Effect::Reputation { id: id(1), delta: 1000 }); // heavy
        led.apply(&Effect::Reputation { id: id(2), delta: 10 }); // light
        led.apply(&Effect::Reputation { id: id(3), delta: -5 }); // slashed → ineligible

        let seed = [7u8; 32];
        // Deterministic.
        assert_eq!(led.select_committee(2, &seed), led.select_committee(2, &seed));
        // Slashed identity is never selected.
        for s in 0u8..50 {
            let seed = [s; 32];
            let c = led.select_committee(3, &seed);
            assert!(!c.contains(&id(3)), "slashed id selected with seed {s}");
        }
        // Heavy identity wins the first slot far more often than the light one.
        let (mut heavy, mut light) = (0, 0);
        for s in 0u8..100 {
            let seed = [s; 32];
            match led.select_committee(1, &seed).first() {
                Some(x) if *x == id(1) => heavy += 1,
                Some(x) if *x == id(2) => light += 1,
                _ => {}
            }
        }
        assert!(heavy > light, "reputation weighting failed: heavy={heavy} light={light}");
    }

    /// The full ReputationPoUW loop: real FHE jobs → lifecycle settlement →
    /// reputation → committee weight.  Honest workers gain standing; a worker
    /// caught by re-execution loses it and drops out of the committee.
    #[test]
    fn reputation_pouw_end_to_end() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let cfg = MarketConfig::default();
        let (honest_a, honest_b, cheat) = (id(11), id(12), id(13));
        let requester = id(1);
        let challenger = id(2);
        let claim_bond = bps(1_000, cfg.claim_bond_bps);
        let chal_bond = bps(1_000, cfg.challenge_bond_bps);

        let mut led = ReputationLedger::new();

        // Two honest jobs settle cleanly (challenge window elapses).
        for worker in [honest_a, honest_b] {
            let job = fhe_job(&be, &ck, 10, 32);
            let hdr = header(&job, requester, 1_000);
            let (_h, receipt) = evaluate_job(&job, &be).unwrap();
            let (st, _) = apply(&hdr, JobState::Open { since: 0 }, Event::Claim {
                worker, bond: claim_bond, requested_ttl: 4,
            }, &cfg, 1).unwrap();
            let (st, _) = apply(&hdr, st, Event::Deliver {
                by: worker, output_root: receipt.output_root,
            }, &cfg, 2).unwrap();
            let (st, effects) = on_epoch(&hdr, st, &cfg, 2 + cfg.challenge_window);
            assert_eq!(st, JobState::Settled { outcome: Outcome::Paid });
            led.apply_all(&effects);
        }

        // A cheating worker delivers a bad result; re-execution catches it.
        {
            let job = fhe_job(&be, &ck, 10, 32);
            let hdr = header(&job, requester, 1_000);
            let lie = WorkReceipt { request: job.request_hash(), output_root: [0xaa; 32] };
            let (st, _) = apply(&hdr, JobState::Open { since: 0 }, Event::Claim {
                worker: cheat, bond: claim_bond, requested_ttl: 4,
            }, &cfg, 1).unwrap();
            let (st, _) = apply(&hdr, st, Event::Deliver {
                by: cheat, output_root: lie.output_root,
            }, &cfg, 2).unwrap();
            let (st, _) = apply(&hdr, st, Event::Challenge {
                challenger, bond: chal_bond,
            }, &cfg, 3).unwrap();
            let verdict = verify_receipt(&job, &lie, &be);
            let (st, effects) = apply(&hdr, st, verdict_to_adjudication(&verdict), &cfg, 3).unwrap();
            assert_eq!(st, JobState::Settled { outcome: Outcome::Slashed });
            led.apply_all(&effects);
        }

        // Honest workers earned reputation; the cheat lost it.
        assert!(led.reputation(&honest_a) > 0);
        assert!(led.reputation(&honest_b) > 0);
        assert!(led.reputation(&cheat) < 0);

        // The committee is drawn from those who did verified useful work.
        for s in 0u8..30 {
            let committee = led.select_committee(2, &[s; 32]);
            assert!(!committee.contains(&cheat));
            for m in committee {
                assert!(m == honest_a || m == honest_b);
            }
        }
    }
}
