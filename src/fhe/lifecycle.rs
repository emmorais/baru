//! FHE job lifecycle: the replicated state machine that wraps the raw
//! evaluate/verify core ([`crate::fhe::market`]) in economics.
//!
//! A job moves `Open → Claimed → Delivered → {Settled | Challenged → Settled}`.
//! Escrow (the reward) and a worker claim bond are held; a challenge window lets
//! anyone dispute a delivered result; disputes are adjudicated by **deterministic
//! re-execution** (a [`Verdict`] from [`crate::fhe::market::verify_receipt`]), so
//! a wrong result is provable fraud.  Settlement emits [`Effect`]s — stake moves
//! and reputation deltas — that a consensus ledger folds at each epoch boundary.
//!
//! Everything is integer-only (basis points) so every node folds the same
//! transitions deterministically.  The [`MarketConfig`] lives in agreed state,
//! so parameters are changed by governance, not a recompile.
//!
//! This machine is adjudicated by, but not tied to, FHE: it consumes an
//! `Adjudicated { worker_correct }` verdict, which an MPC co-signature or a
//! storage proof could equally produce.

use crate::fhe::market::Verdict;

/// Stake amount (smallest unit).
pub type Stake = u128;
/// Monotonic epoch counter.
pub type Epoch = u64;
/// Participant identity (hash of a public key).
pub type Id = [u8; 32];
/// 32-byte content commitment (job request / output roots).
pub type Hash = [u8; 32];

/// All market tunables.  Ratios are basis points (`10_000` = 100%) so folding is
/// integer-only and deterministic across nodes.  Lives in agreed state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarketConfig {
    // timeouts, in epochs
    pub claim_ttl: Epoch,
    pub compute_ttl_max: Epoch,
    pub challenge_window: Epoch,
    pub adjudicate_ttl: Epoch,
    // bonds / escrow, in bps of the reward R
    pub claim_bond_bps: u32,
    pub challenge_bond_bps: u32,
    pub verification_tax_bps: u32,
    // slashing splits, in bps of the slashed bond
    pub fraud_challenger_bps: u32,
    pub abandon_requester_bps: u32,
    // reputation credited per unit R on clean finalize
    pub reward_reputation_bps: u32,
    pub min_reward: Stake,
}

impl Default for MarketConfig {
    fn default() -> Self {
        MarketConfig {
            claim_ttl: 4,
            compute_ttl_max: 16,
            challenge_window: 8,
            adjudicate_ttl: 2,
            claim_bond_bps: 15_000,     // 150% of R
            challenge_bond_bps: 2_500,  //  25% of R
            verification_tax_bps: 200,  //   2% of R
            fraud_challenger_bps: 5_000,
            abandon_requester_bps: 5_000,
            reward_reputation_bps: 10_000,
            min_reward: 1_000,
        }
    }
}

impl MarketConfig {
    /// Reject configs that break the fraud-is-unprofitable invariant
    /// `B_w ≥ R + B_c`.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.claim_bond_bps < 10_000 + self.challenge_bond_bps {
            return Err("claim_bond must be ≥ R + challenge_bond (fraud would be +EV)");
        }
        if self.fraud_challenger_bps > 10_000 || self.abandon_requester_bps > 10_000 {
            return Err("split ratios exceed 100%");
        }
        Ok(())
    }
}

/// Integer basis-points multiply.
fn bps(amount: Stake, bps: u32) -> Stake {
    amount * bps as u128 / 10_000
}

/// Immutable job header, committed once at post time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobHeader {
    pub id: Hash,
    pub requester: Id,
    pub reward: Stake,
    /// Commitment to the evaluation request (see `FheJob::request_hash`).
    pub request: Hash,
    /// Worker's requested compute budget, clamped to `compute_ttl_max`.
    pub compute_ttl: Epoch,
    pub posted: Epoch,
}

/// Live job state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobState {
    /// Awaiting a worker; `since` is the epoch it last entered Open.
    Open {
        since: Epoch,
    },
    Claimed {
        worker: Id,
        claim_bond: Stake,
        deliver_by: Epoch,
    },
    Delivered {
        worker: Id,
        claim_bond: Stake,
        output_root: Hash,
        challengeable_until: Epoch,
    },
    Challenged {
        worker: Id,
        claim_bond: Stake,
        output_root: Hash,
        challenger: Id,
        challenge_bond: Stake,
        adjudicate_by: Epoch,
    },
    Settled {
        outcome: Outcome,
    },
}

/// Terminal outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Paid,
    Slashed,
    Forfeited,
    Cancelled,
}

/// Explicit events applied to a job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Claim {
        worker: Id,
        bond: Stake,
        requested_ttl: Epoch,
    },
    Deliver {
        by: Id,
        output_root: Hash,
    },
    Challenge {
        challenger: Id,
        bond: Stake,
    },
    /// Committee verdict from re-execution: did the worker's output match?
    Adjudicated {
        worker_correct: bool,
    },
}

/// Map a re-execution [`Verdict`] to the adjudication bit the machine consumes.
pub fn verdict_to_adjudication(v: &Verdict) -> Event {
    Event::Adjudicated {
        worker_correct: matches!(v, Verdict::Valid),
    }
}

/// Deterministic stake / reputation movements a transition emits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    Release { to: Id, amount: Stake },
    Refund { to: Id, amount: Stake },
    Slash { from: Id, amount: Stake },
    Treasury { amount: Stake },
    Reputation { id: Id, delta: i128 },
}

/// Why an explicit event was rejected (a no-op that never breaks consensus).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejected {
    Underbonded,
    BadTransition,
}

/// Clean finalize: worker gets `R + bond` back minus the audit tax, and earns
/// reputation proportional to `R`.
fn payout(hdr: &JobHeader, worker: Id, claim_bond: Stake, cfg: &MarketConfig) -> Vec<Effect> {
    let tax = bps(hdr.reward, cfg.verification_tax_bps);
    vec![
        Effect::Release {
            to: worker,
            amount: hdr.reward - tax + claim_bond,
        },
        Effect::Treasury { amount: tax },
        Effect::Reputation {
            id: worker,
            delta: bps(hdr.reward, cfg.reward_reputation_bps) as i128,
        },
    ]
}

/// Effects of a proven-fraud settlement: worker's bond slashed (split to the
/// challenger and treasury), reward refunded, reputation debited.
fn fraud_settlement(
    hdr: &JobHeader,
    worker: Id,
    claim_bond: Stake,
    challenger: Id,
    challenge_bond: Stake,
    cfg: &MarketConfig,
) -> Vec<Effect> {
    let to_challenger = bps(claim_bond, cfg.fraud_challenger_bps);
    vec![
        Effect::Slash {
            from: worker,
            amount: claim_bond,
        },
        Effect::Release {
            to: challenger,
            amount: challenge_bond + to_challenger,
        },
        Effect::Treasury {
            amount: claim_bond - to_challenger,
        },
        Effect::Refund {
            to: hdr.requester,
            amount: hdr.reward,
        },
        Effect::Reputation {
            id: worker,
            delta: -(hdr.reward as i128),
        },
    ]
}

/// Apply an explicit event.  Rejected events are no-ops (never break consensus).
pub fn apply(
    hdr: &JobHeader,
    state: JobState,
    event: Event,
    cfg: &MarketConfig,
    now: Epoch,
) -> Result<(JobState, Vec<Effect>), Rejected> {
    use Event::*;
    use JobState::*;
    match (state, event) {
        // Worker claims an open job, posting its bond and a compute budget.
        (
            Open { .. },
            Claim {
                worker,
                bond,
                requested_ttl,
            },
        ) => {
            if bond < bps(hdr.reward, cfg.claim_bond_bps) {
                return Err(Rejected::Underbonded);
            }
            let ttl = requested_ttl.min(cfg.compute_ttl_max);
            Ok((
                Claimed {
                    worker,
                    claim_bond: bond,
                    deliver_by: now + ttl,
                },
                vec![],
            ))
        }
        // Worker delivers a result → opens the challenge window.
        (
            Claimed {
                worker, claim_bond, ..
            },
            Deliver { by, output_root },
        ) if by == worker => Ok((
            Delivered {
                worker,
                claim_bond,
                output_root,
                challengeable_until: now + cfg.challenge_window,
            },
            vec![],
        )),
        // Anyone opens a dispute during the window, posting a challenge bond.
        (
            Delivered {
                worker,
                claim_bond,
                output_root,
                ..
            },
            Challenge { challenger, bond },
        ) => {
            if bond < bps(hdr.reward, cfg.challenge_bond_bps) {
                return Err(Rejected::Underbonded);
            }
            Ok((
                Challenged {
                    worker,
                    claim_bond,
                    output_root,
                    challenger,
                    challenge_bond: bond,
                    adjudicate_by: now + cfg.adjudicate_ttl,
                },
                vec![],
            ))
        }
        // Committee re-execution verdict resolves a dispute.
        (
            Challenged {
                worker,
                claim_bond,
                challenger,
                challenge_bond,
                ..
            },
            Adjudicated { worker_correct },
        ) => {
            if worker_correct {
                // Challenger loses its bond to the (vindicated) worker; job pays.
                let mut fx = payout(hdr, worker, claim_bond, cfg);
                fx.push(Effect::Release {
                    to: worker,
                    amount: challenge_bond,
                });
                Ok((Settled { outcome: Outcome::Paid }, fx))
            } else {
                Ok((
                    Settled {
                        outcome: Outcome::Slashed,
                    },
                    fraud_settlement(hdr, worker, claim_bond, challenger, challenge_bond, cfg),
                ))
            }
        }
        _ => Err(Rejected::BadTransition),
    }
}

/// Timeouts, evaluated once per job at each epoch boundary.
pub fn on_epoch(
    hdr: &JobHeader,
    state: JobState,
    cfg: &MarketConfig,
    now: Epoch,
) -> (JobState, Vec<Effect>) {
    use JobState::*;
    match state {
        // Unclaimed past the claim TTL → cancel, refund escrow.
        Open { since } if now >= since + cfg.claim_ttl => (
            Settled {
                outcome: Outcome::Cancelled,
            },
            vec![Effect::Refund {
                to: hdr.requester,
                amount: hdr.reward,
            }],
        ),
        // Claimed but not delivered in time → forfeit bond, re-open.
        Claimed {
            worker,
            claim_bond,
            deliver_by,
        } if now >= deliver_by => {
            let to_req = bps(claim_bond, cfg.abandon_requester_bps);
            (
                Open { since: now },
                vec![
                    Effect::Slash {
                        from: worker,
                        amount: claim_bond,
                    },
                    Effect::Refund {
                        to: hdr.requester,
                        amount: to_req,
                    },
                    Effect::Treasury {
                        amount: claim_bond - to_req,
                    },
                ],
            )
        }
        // Challenge window elapsed clean → finalize and pay.
        Delivered {
            worker,
            claim_bond,
            challengeable_until,
            ..
        } if now >= challengeable_until => (
            Settled { outcome: Outcome::Paid },
            payout(hdr, worker, claim_bond, cfg),
        ),
        // Worker failed to defend a dispute in time → treated as fraud.
        Challenged {
            worker,
            claim_bond,
            challenger,
            challenge_bond,
            adjudicate_by,
            ..
        } if now >= adjudicate_by => (
            Settled {
                outcome: Outcome::Slashed,
            },
            fraud_settlement(hdr, worker, claim_bond, challenger, challenge_bond, cfg),
        ),
        other => (other, vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::market::{evaluate_job, verify_receipt, FheJob, WorkReceipt};
    use crate::fhe::mock_backend::MockFheBackend;
    use crate::fhe::backend::{FheBackend, FheCircuit};
    use crate::interpreter::parse;
    use crate::zkp::circuit_compiler::CircuitCompiler;
    use std::collections::HashMap;

    const REQUESTER: Id = [1u8; 32];
    const WORKER: Id = [2u8; 32];
    const CHALLENGER: Id = [3u8; 32];

    fn cfg() -> MarketConfig {
        let c = MarketConfig::default();
        c.validate().unwrap();
        c
    }

    /// A real FHE job: (+ x y) with both encrypted.
    fn fhe_job(be: &MockFheBackend, ck: &[u8], x: u64, y: u64) -> FheJob {
        let expr = parse("(+ x y)").unwrap();
        let mut circuit: FheCircuit = CircuitCompiler::new().compile_fhe(&expr).unwrap();
        circuit.public_inputs.clear();
        circuit.encrypted_inputs = vec!["x".into(), "y".into()];
        let mut encrypted_inputs = HashMap::new();
        encrypted_inputs.insert("x".to_string(), be.serialize_ct(&be.encrypt(ck, x)));
        encrypted_inputs.insert("y".to_string(), be.serialize_ct(&be.encrypt(ck, y)));
        FheJob {
            circuit,
            public_inputs: HashMap::new(),
            encrypted_inputs,
            backend_id: be.backend_id().to_string(),
        }
    }

    fn header(job: &FheJob, reward: Stake) -> JobHeader {
        JobHeader {
            id: [9u8; 32],
            requester: REQUESTER,
            reward,
            request: job.request_hash(),
            compute_ttl: 4,
            posted: 0,
        }
    }

    #[test]
    fn config_default_is_valid() {
        assert!(MarketConfig::default().validate().is_ok());
        // Fraud must be unprofitable: B_w ≥ R + B_c.
        let bad = MarketConfig {
            claim_bond_bps: 10_000,
            challenge_bond_bps: 2_500,
            ..MarketConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn honest_job_pays_worker_and_credits_reputation() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job = fhe_job(&be, &ck, 10, 32);
        let hdr = header(&job, 1_000);
        let cfg = cfg();

        // Worker actually evaluates the FHE job.
        let (_handle, receipt) = evaluate_job(&job, &be).unwrap();

        // Open → Claim → Deliver.
        let (st, _) = apply(&hdr, JobState::Open { since: 0 }, Event::Claim {
            worker: WORKER,
            bond: bps(hdr.reward, cfg.claim_bond_bps),
            requested_ttl: 4,
        }, &cfg, 1).unwrap();
        let (st, _) = apply(&hdr, st, Event::Deliver {
            by: WORKER,
            output_root: receipt.output_root,
        }, &cfg, 2).unwrap();

        // Challenge window elapses with no dispute → clean finalize.
        let finalize_at = 2 + cfg.challenge_window;
        let (st, effects) = on_epoch(&hdr, st, &cfg, finalize_at);
        assert_eq!(st, JobState::Settled { outcome: Outcome::Paid });

        // Worker receives R + bond − tax, and earns reputation.
        let tax = bps(hdr.reward, cfg.verification_tax_bps);
        let bond = bps(hdr.reward, cfg.claim_bond_bps);
        assert!(effects.contains(&Effect::Release {
            to: WORKER,
            amount: hdr.reward - tax + bond,
        }));
        assert!(effects.iter().any(|e| matches!(
            e,
            Effect::Reputation { id, delta } if *id == WORKER && *delta > 0
        )));
    }

    #[test]
    fn fraudulent_delivery_is_caught_and_slashed_via_real_reexecution() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job = fhe_job(&be, &ck, 10, 32);
        let hdr = header(&job, 1_000);
        let cfg = cfg();

        // A lying worker delivers a bogus output root.
        let lie = WorkReceipt {
            request: job.request_hash(),
            output_root: [0xabu8; 32],
        };

        let claim_bond = bps(hdr.reward, cfg.claim_bond_bps);
        let challenge_bond = bps(hdr.reward, cfg.challenge_bond_bps);

        let (st, _) = apply(&hdr, JobState::Open { since: 0 }, Event::Claim {
            worker: WORKER,
            bond: claim_bond,
            requested_ttl: 4,
        }, &cfg, 1).unwrap();
        let (st, _) = apply(&hdr, st, Event::Deliver {
            by: WORKER,
            output_root: lie.output_root,
        }, &cfg, 2).unwrap();
        let (st, _) = apply(&hdr, st, Event::Challenge {
            challenger: CHALLENGER,
            bond: challenge_bond,
        }, &cfg, 3).unwrap();

        // Adjudicate by REAL re-execution of the FHE circuit.
        let verdict = verify_receipt(&job, &lie, &be);
        assert!(matches!(verdict, Verdict::Fraudulent { .. }));
        let (st, effects) = apply(&hdr, st, verdict_to_adjudication(&verdict), &cfg, 3).unwrap();

        assert_eq!(st, JobState::Settled { outcome: Outcome::Slashed });
        assert!(effects.contains(&Effect::Slash {
            from: WORKER,
            amount: claim_bond,
        }));
        // Challenger is made whole plus a cut of the slash.
        let cut = bps(claim_bond, cfg.fraud_challenger_bps);
        assert!(effects.contains(&Effect::Release {
            to: CHALLENGER,
            amount: challenge_bond + cut,
        }));
        // Requester refunded.
        assert!(effects.contains(&Effect::Refund {
            to: REQUESTER,
            amount: hdr.reward,
        }));
    }

    #[test]
    fn honest_worker_survives_a_frivolous_challenge() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job = fhe_job(&be, &ck, 6, 7);
        let hdr = header(&job, 1_000);
        let cfg = cfg();

        // Honest delivery.
        let (_h, receipt) = evaluate_job(&job, &be).unwrap();
        let claim_bond = bps(hdr.reward, cfg.claim_bond_bps);
        let challenge_bond = bps(hdr.reward, cfg.challenge_bond_bps);

        let (st, _) = apply(&hdr, JobState::Open { since: 0 }, Event::Claim {
            worker: WORKER, bond: claim_bond, requested_ttl: 4,
        }, &cfg, 1).unwrap();
        let (st, _) = apply(&hdr, st, Event::Deliver {
            by: WORKER, output_root: receipt.output_root,
        }, &cfg, 2).unwrap();
        let (st, _) = apply(&hdr, st, Event::Challenge {
            challenger: CHALLENGER, bond: challenge_bond,
        }, &cfg, 3).unwrap();

        // Re-execution vindicates the worker.
        let verdict = verify_receipt(&job, &receipt, &be);
        assert_eq!(verdict, Verdict::Valid);
        let (st, effects) = apply(&hdr, st, verdict_to_adjudication(&verdict), &cfg, 3).unwrap();

        assert_eq!(st, JobState::Settled { outcome: Outcome::Paid });
        // Worker gets the challenger's forfeited bond on top of the payout.
        assert!(effects.contains(&Effect::Release {
            to: WORKER,
            amount: challenge_bond,
        }));
    }

    #[test]
    fn abandoned_claim_forfeits_bond_and_reopens() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job = fhe_job(&be, &ck, 1, 1);
        let hdr = header(&job, 1_000);
        let cfg = cfg();
        let claim_bond = bps(hdr.reward, cfg.claim_bond_bps);

        let (st, _) = apply(&hdr, JobState::Open { since: 0 }, Event::Claim {
            worker: WORKER, bond: claim_bond, requested_ttl: 4,
        }, &cfg, 1).unwrap();

        // Deliver deadline passes.
        let (st, effects) = on_epoch(&hdr, st, &cfg, 1 + cfg.compute_ttl_max + 1);
        assert_eq!(st, JobState::Open { since: 1 + cfg.compute_ttl_max + 1 });
        assert!(effects.contains(&Effect::Slash {
            from: WORKER,
            amount: claim_bond,
        }));
    }

    #[test]
    fn unclaimed_job_cancels_and_refunds() {
        let hdr = JobHeader {
            id: [9u8; 32],
            requester: REQUESTER,
            reward: 1_000,
            request: [7u8; 32],
            compute_ttl: 4,
            posted: 0,
        };
        let cfg = cfg();
        let (st, effects) = on_epoch(&hdr, JobState::Open { since: 0 }, &cfg, cfg.claim_ttl);
        assert_eq!(st, JobState::Settled { outcome: Outcome::Cancelled });
        assert!(effects.contains(&Effect::Refund {
            to: REQUESTER,
            amount: 1_000,
        }));
    }
}
