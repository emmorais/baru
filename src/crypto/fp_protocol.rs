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
//! - **Wire width.** A field element is **6 bits**, bit-packed — not a byte and
//!   certainly not sixteen. Falcon counts `k = 6` bits per `Z_p` element, and
//!   this network's traffic is essentially all field elements, so byte-padding
//!   would inflate the reproduction by `8/6` and byte-16 padding by 16×. The
//!   comparison is the point, so the encoding has to match.
//!
//! Everything else — the local cross terms, the single directed resharing
//! message, the α counter discipline — is the ring protocol unchanged.

use crate::crypto::field::{Fp, FpShare, ELEM_BITS, P};
use crate::crypto::net::{serialize_msg, Mailbox, MpcMsg, PartyId, Transport};

const N: usize = 3;
const TIMEOUT_MS: u64 = 5_000;

/// Bit-pack field elements at [`ELEM_BITS`] each.
///
/// Not a micro-optimisation. Falcon's published communication counts `k` bits
/// per `Z_p` element with `k = 6`; a byte per element would be `8/6` of that,
/// and since this network's traffic is essentially all field elements, padding
/// would inflate the reproduction by a third and make it look worse than the
/// system it is reproducing. The comparison is the point, so the encoding has to
/// match.
pub fn pack(values: &[Fp]) -> Vec<u8> {
    let mut out = vec![0u8; (values.len() * ELEM_BITS as usize).div_ceil(8)];
    for (i, v) in values.iter().enumerate() {
        let bit = i * ELEM_BITS as usize;
        let val = v.value();
        for b in 0..ELEM_BITS as usize {
            if (val >> b) & 1 == 1 {
                out[(bit + b) / 8] |= 1 << ((bit + b) % 8);
            }
        }
    }
    out
}

/// Inverse of [`pack`]; `n` is how many elements to read back.
pub fn unpack(bytes: &[u8], n: usize) -> Vec<Fp> {
    (0..n)
        .map(|i| {
            let bit = i * ELEM_BITS as usize;
            let mut val = 0u64;
            for b in 0..ELEM_BITS as usize {
                let idx = bit + b;
                if bytes[idx / 8] >> (idx % 8) & 1 == 1 {
                    val |= 1 << b;
                }
            }
            Fp::new(val)
        })
        .collect()
}

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
        right: if me == (owner + N - 1) % N {
            value
        } else {
            Fp::ZERO
        },
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
    FpShare {
        left: Fp::new(left),
        right: Fp::new(right),
    }
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
    let z_mine: Vec<Fp> = pairs
        .iter()
        .enumerate()
        .map(|(i, (a, b))| {
            a.left * b.left
                + a.left * b.right
                + a.right * b.left
                + Fp::new(keys.alpha_mod(alpha_ctr(base_op_id, i), P))
        })
        .collect();

    mbx.send(
        (me + N - 1) % N,
        serialize_msg(&MpcMsg::TaggedFpBatch {
            base_op_id,
            values: pack(&z_mine),
        }),
    );
    let (_, msg) = mbx
        .recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::TaggedFpBatch { base_op_id: b, .. } if *b == base_op_id),
        )
        .unwrap_or_else(|| panic!("fp mul_batch: timeout on z for op_id={base_op_id}"));
    let z_next = match msg {
        MpcMsg::TaggedFpBatch { values, .. } => unpack(&values, z_mine.len()),
        _ => unreachable!("predicate guarantees TaggedFpBatch"),
    };

    z_mine
        .into_iter()
        .zip(z_next)
        .map(|(left, right)| FpShare { left, right })
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
    let mine: Vec<Fp> = shares.iter().map(|s| s.left).collect();
    mbx.send(
        (me + 1) % N,
        serialize_msg(&MpcMsg::TaggedFpBatch {
            base_op_id,
            values: pack(&mine),
        }),
    );
    let (_, msg) = mbx
        .recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::TaggedFpBatch { base_op_id: b, .. } if *b == base_op_id),
        )
        .unwrap_or_else(|| panic!("fp reveal_batch: timeout for op_id={base_op_id}"));
    let missing = match msg {
        MpcMsg::TaggedFpBatch { values, .. } => unpack(&values, shares.len()),
        _ => unreachable!("predicate guarantees TaggedFpBatch"),
    };
    shares
        .iter()
        .zip(missing)
        .map(|(s, m)| s.left + s.right + m)
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
        joins
            .into_iter()
            .map(|j| j.join().expect("party panicked"))
            .collect()
    }

    /// How many bytes a ring multiplication actually puts on the wire, per
    /// meaningful bit — the comparison that says whether Baru's own reported
    /// traffic is encoding-limited.
    ///
    /// `aby3_protocol` reshares through `MpcMsg::TaggedZBatch { values: Vec<u128> }`
    /// and bincode encodes `u128` as a fixed 16 bytes, whatever `nbits` is. At
    /// the CNN benchmarks' `nbits = 20` that is 128 wire bits for 20 meaningful
    /// ones. This test measures the ratio rather than asserting it is fine.
    #[test]
    #[ignore] // real 3-party transport
    fn ring_traffic_is_dominated_by_the_u128_encoding() {
        use crate::bench::comm::CountingTransport;
        use crate::crypto::aby3_protocol::{
            mul_batch_party, share_component as ring_component, PartyShare,
        };
        const N: usize = 4096;
        const NBITS: u32 = 20;

        let handles = InprocHandle::create(3);
        let mut comm = None;
        let mut joins = Vec::new();
        for h in handles {
            let counted = CountingTransport::new(h, 3);
            if comm.is_none() {
                comm = Some(counted.handle());
            }
            joins.push(thread::spawn(move || {
                let mbx = Mailbox::new(counted);
                let me = mbx.id();
                let pairs: Vec<(PartyShare, PartyShare)> = (0..N)
                    .map(|i| {
                        (
                            ring_component(i as u128 % (1 << NBITS), 0, me),
                            ring_component((i as u128 * 7) % (1 << NBITS), 0, me),
                        )
                    })
                    .collect();
                let _ = mul_batch_party(&pairs, 1, me, &mbx);
            }));
        }
        for j in joins {
            j.join().expect("panic");
        }
        let s = comm.unwrap().stats();
        let bytes_per_elem = s.bytes as f64 / N as f64;
        let useful = NBITS as f64 / 8.0;

        println!(
            "\n  ABY3 ring mul_batch, n={N}, nbits={NBITS} — per party\n    \
             bytes on the wire : {}\n    per element       : {bytes_per_elem:.2} B\n    \
             meaningful        : {useful:.2} B ({NBITS} bits)\n    \
             encoding overhead : {:.1}x\n    \
             (for contrast, the Z_p path packs at {} bits/element)\n",
            s.bytes,
            bytes_per_elem / useful,
            crate::crypto::field::ELEM_BITS
        );
        assert!(
            bytes_per_elem > 8.0,
            "expected the u128 encoding to dominate; got {bytes_per_elem:.2} B/element"
        );
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
            (0..64u64)
                .map(|c| random_share(c, me, mbx))
                .collect::<Vec<_>>()
        });
        let mut nonzero = 0;
        // Indexed on purpose: each step reads the *same* position across all
        // three parties' vectors, which an iterator over any one of them
        // cannot express.
        #[allow(clippy::needless_range_loop)]
        for i in 0..64usize {
            for p in 0..3usize {
                assert_eq!(
                    out[p][i].right,
                    out[(p + 1) % 3][i].left,
                    "party {p}'s right must be party {}'s left",
                    (p + 1) % 3
                );
            }
            let v = out[0][i].left + out[1][i].left + out[2][i].left;
            if !v.is_zero() {
                nonzero += 1;
            }
        }
        assert!(
            nonzero > 55,
            "shared values must not be a sharing of zero ({nonzero}/64 non-zero)"
        );
    }

    /// The transport multiplication must agree with the field arithmetic — the
    /// same product, obtained over real messages.
    #[test]
    #[ignore] // real 3-party transport
    fn transport_multiplication_matches_the_field() {
        let cases: Vec<(u64, u64)> = (0..P).flat_map(|a| (0..P).map(move |b| (a, b))).collect();
        let expected: Vec<Fp> = cases
            .iter()
            .map(|&(a, b)| Fp::new(a) * Fp::new(b))
            .collect();
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
