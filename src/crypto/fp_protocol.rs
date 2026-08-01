//! Replicated secret sharing over the small prime field `Z_p`, on the real
//! 3-party transport.
//!
//! The mirror of [`crate::crypto::aby3_protocol`], which does the same over
//! `Z_{2^k}`. It exists because Falcon's private compare needs a field (see
//! [`crate::crypto::field`]), and because the *point* of reproducing Falcon is
//! to measure its cost — which means its messages have to be real messages,
//! counted by the same transport every other protocol in this crate is counted
//! by.
//!
//! Two things differ from the ring version, and only two:
//!
//! - **Zero-sum randomness.** `PrfKeys::alpha` sums to zero mod `2^128`, which
//!   says nothing mod `p`. [`PrfKeys::alpha_mod`](crate::crypto::net::PrfKeys::alpha_mod)
//!   applies the same telescoping construction in the field instead.
//! - **Wire width.** A field element is one byte, not sixteen. Falcon's claim
//!   over prior work is communication volume, so this has to be right or the
//!   comparison is worthless.
//!
//! Everything else — the local cross terms, the single directed resharing
//! message, the α counter discipline — is the ring protocol unchanged.

use crate::crypto::field::{Fp, FpShare, P};
use crate::crypto::net::{serialize_msg, Mailbox, MpcMsg, PartyId, Transport};

const N: usize = 3;
const TIMEOUT_MS: u64 = 5_000;

/// Distinct α counter per product, matching `aby3_protocol`'s discipline: each
/// `op_id` owns a disjoint `2^40` slot, so no two multiplications anywhere
/// share randomness. Reusing an α would expose the difference of two products
/// on the wire.
fn alpha_ctr(op_id: u64, i: usize) -> u64 {
    (op_id << 40).wrapping_add(i as u64)
}

/// Share a field element held by `owner`, with no communication, by placing it
/// in one additive component: component `owner` is held as `left` by party
/// `owner` and as `right` by party `owner−1`.
pub fn share_component(value: Fp, owner: usize, me: PartyId) -> FpShare {
    FpShare {
        left: if me == owner { value } else { Fp::ZERO },
        right: if me == (owner + N - 1) % N { value } else { Fp::ZERO },
    }
}

/// A replicated sharing of a fresh uniform field element **nobody knows**, free
/// of communication — the field analogue of the ring version, and what supplies
/// `Π_PC`'s blinding bit and mask in deployment.
/// This must use the **raw** PRF outputs, not
/// [`alpha_mod`](crate::crypto::net::PrfKeys::alpha_mod): the zero-sum form is
/// for masking a product, and using it here would share the value zero. The two
/// are one character apart at the call site and behave completely differently,
/// which is why `random_shares_are_consistent_and_uniform` exists.
pub fn random_share<T: Transport>(ctr: u64, me: PartyId, mbx: &Mailbox<T>) -> FpShare {
    let keys = mbx.prf_keys(me);
    // `prev` is shared with party me−1 and so names additive component `me`
    // (this party's `left`); `mine` is shared with me+1 and names component
    // `me+1` (this party's `right`). Same assignment as the ring version.
    let (left, right) = keys.random_share_mod(ctr, P);
    FpShare { left: Fp::new(left), right: Fp::new(right) }
}

/// Batched multiplication over `Z_p`: one directed message per party for the
/// whole batch.
///
/// Local cross terms re-randomised by a zero-sum field α, then each party sends
/// its component to its predecessor so the 3-out-of-3 result becomes replicated
/// again — structurally identical to the ring protocol, and one round.
pub fn mul_batch<T: Transport>(
    pairs: &[(FpShare, FpShare)],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<FpShare> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let keys = mbx.prf_keys(me);
    let z_mine: Vec<u8> = pairs
        .iter()
        .enumerate()
        .map(|(i, (a, b))| {
            let t = a.left * b.left + a.left * b.right + a.right * b.left
                + Fp::new(keys.alpha_mod(alpha_ctr(base_op_id, i), P));
            t.value() as u8
        })
        .collect();

    mbx.send(
        (me + N - 1) % N,
        serialize_msg(&MpcMsg::TaggedFpBatch { base_op_id, values: z_mine.clone() }),
    );
    let (_, msg) = mbx
        .recv_matching(TIMEOUT_MS, |m| {
            matches!(m, MpcMsg::TaggedFpBatch { base_op_id: b, .. } if *b == base_op_id)
        })
        .unwrap_or_else(|| panic!("fp mul_batch: timeout on z for op_id={base_op_id}"));
    let z_next = match msg {
        MpcMsg::TaggedFpBatch { values, .. } => values,
        _ => unreachable!("predicate guarantees TaggedFpBatch"),
    };
    assert_eq!(z_next.len(), z_mine.len(), "fp batch length mismatch in resharing");

    z_mine
        .into_iter()
        .zip(z_next)
        .map(|(l, r)| FpShare { left: Fp::new(l as u64), right: Fp::new(r as u64) })
        .collect()
}

/// Open a batch of field shares to every party — one directed message each.
///
/// Each party already holds two of the three summands, so it needs only the one
/// it is missing, which its successor supplies.
pub fn reveal_batch<T: Transport>(
    shares: &[FpShare],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<Fp> {
    if shares.is_empty() {
        return Vec::new();
    }
    let mine: Vec<u8> = shares.iter().map(|s| s.left.value() as u8).collect();
    mbx.send(
        (me + 1) % N,
        serialize_msg(&MpcMsg::TaggedFpBatch { base_op_id, values: mine.clone() }),
    );
    let (_, msg) = mbx
        .recv_matching(TIMEOUT_MS, |m| {
            matches!(m, MpcMsg::TaggedFpBatch { base_op_id: b, .. } if *b == base_op_id)
        })
        .unwrap_or_else(|| panic!("fp reveal_batch: timeout for op_id={base_op_id}"));
    let missing = match msg {
        MpcMsg::TaggedFpBatch { values, .. } => values,
        _ => unreachable!("predicate guarantees TaggedFpBatch"),
    };
    shares
        .iter()
        .zip(missing)
        .map(|(s, m)| s.left + s.right + Fp::new(m as u64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;
    use std::thread;

    /// Drive a 3-party in-process run.
    fn run3<F, R>(f: F) -> Vec<R>
    where
        F: Fn(PartyId, &Mailbox<InprocHandle>) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let handles = InprocHandle::create(3);
        let f = std::sync::Arc::new(f);
        let joins: Vec<_> = handles
            .into_iter()
            .map(|h| {
                let f = f.clone();
                thread::spawn(move || {
                    let mbx = Mailbox::new(h);
                    let me = mbx.id();
                    f(me, &mbx)
                })
            })
            .collect();
        joins.into_iter().map(|j| j.join().expect("party panicked")).collect()
    }

    /// A free random sharing must actually be a *consistent* replicated sharing
    /// — party `i`'s `right` is party `i+1`'s `left` — and must not be a sharing
    /// of zero.
    ///
    /// The second half is the real check. Deriving this from the zero-sum
    /// `alpha_mod` instead of the raw PRFs is a one-word slip that yields a
    /// perfectly consistent sharing *of zero*, which every structural assertion
    /// would pass.
    #[test]
    #[ignore] // real 3-party transport
    fn random_shares_are_consistent_and_uniform() {
        let out = run3(|me, mbx| {
            (0..64u64).map(|c| random_share(c, me, mbx)).collect::<Vec<_>>()
        });
        let mut nonzero = 0;
        for i in 0..64usize {
            for p in 0..3usize {
                assert_eq!(
                    out[p][i].right, out[(p + 1) % 3][i].left,
                    "party {p}'s right must be party {}'s left",
                    (p + 1) % 3
                );
            }
            let v = out[0][i].left + out[1][i].left + out[2][i].left;
            if !v.is_zero() {
                nonzero += 1;
            }
        }
        assert!(nonzero > 55, "shared values must not be a sharing of zero ({nonzero}/64 non-zero)");
    }

    /// The transport multiplication must agree with the field arithmetic — the
    /// same product, obtained over real messages.
    #[test]
    #[ignore] // real 3-party transport
    fn transport_multiplication_matches_the_field() {
        let cases: Vec<(u64, u64)> = (0..P).flat_map(|a| (0..P).map(move |b| (a, b))).collect();
        let expected: Vec<Fp> = cases.iter().map(|&(a, b)| Fp::new(a) * Fp::new(b)).collect();
        let out = run3(move |me, mbx| {
            // Party 0 owns both inputs; sharing a held value is free.
            let pairs: Vec<(FpShare, FpShare)> = cases
                .iter()
                .map(|&(a, b)| {
                    (
                        share_component(Fp::new(a), 0, me),
                        share_component(Fp::new(b), 0, me),
                    )
                })
                .collect();
            let prod = mul_batch(&pairs, 1, me, mbx);
            reveal_batch(&prod, 2, me, mbx)
        });
        for party in &out {
            assert_eq!(party, &expected, "every party must open the same products");
        }
    }
}
