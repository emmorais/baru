//! ABY3-specific protocol helpers for additive 3-party secret sharing.
//!
//! This module contains the ABY3 protocol primitives: PartyShare representation,
//! share input/output, multiplication via Beaver triples, a2b decomposition,
//! and related operations. These are specific to the ABY3 scheme and should not
//! be used directly by the evaluator or generic backends.

use rand::Rng;

use crate::crypto::arith::gen_beaver_triple;
use crate::crypto::net::{serialize_msg, Mailbox, MpcMsg, PartyId, Transport};
use crate::crypto::types::{Fr, Share3};

const N: usize = 3;
const TIMEOUT_MS: u64 = 5_000;

/// The two field elements that party `id` holds for a given shared value.
/// In ABY3, party i holds (s_i, s_{(i+1) mod 3}).
///
/// This is ABY3-specific and should not be exposed to the interpreter.
#[derive(Clone, Debug)]
pub struct PartyShare {
    pub left: Fr,  // s_i
    pub right: Fr, // s_{(i+1) mod 3}
}

/// Distribute a secret input `value` to all three parties.
///
/// - Party `owner` calls this with the actual `value`.
/// - All other parties call this with `value = 0` (ignored).
///
/// Returns the `PartyShare` for the *calling* party.
pub fn share_input<T: Transport>(
    value: Fr,
    share_id: u64,
    owner: PartyId,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> PartyShare {
    if me == owner {
        let mut rng = rand::thread_rng();
        let s: [Fr; 3] = {
            let s0: Fr = rng.gen();
            let s1: Fr = rng.gen();
            let s2 = value.wrapping_sub(s0).wrapping_sub(s1);
            [s0, s1, s2]
        };
        for i in 0..N {
            let right_idx = (i + 1) % N;
            if i != me {
                mbx.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[i],
                    }),
                );
                mbx.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[right_idx],
                    }),
                );
            }
        }
        PartyShare {
            left: s[me],
            right: s[(me + 1) % N],
        }
    } else {
        let mut components: Vec<Fr> = Vec::new();
        while components.len() < 2 {
            if let Some((_from, m)) = mbx.recv_matching(
                TIMEOUT_MS,
                |m| matches!(m, MpcMsg::InputShare { share_id: sid, .. } if *sid == share_id),
            ) {
                if let MpcMsg::InputShare { value: v, .. } = m {
                    components.push(v);
                }
            } else {
                panic!(
                    "share_input: timeout waiting for InputShare share_id={}",
                    share_id
                );
            }
        }
        PartyShare {
            left: components[0],
            right: components[1],
        }
    }
}

/// Share a public constant locally (no communication).  Party 0 gets the
/// value in its left component; others get 0 so that reconstruction = value.
///
/// The replicated layout must be consistent: party `i` holds
/// `(s_i, s_{(i+1) mod 3})` for the additive sharing `s = (value, 0, 0)`.
/// Party 2's *right* component is therefore `s_0 = value`, not 0 — an
/// earlier version zeroed every right component, which reconstructs fine
/// (reveal sums only the lefts) but corrupts any *multiplication* of the
/// public value, since the multiplication protocol relies on the replicated
/// pairing. (Latent until a public nonzero constant was fed into a
/// product — e.g. `crate::gadgets` argmax indices / ReLU `keep` mask.)
pub fn share_public(value: Fr, me: PartyId) -> PartyShare {
    let s = |i: usize| if i == 0 { value } else { 0 };
    PartyShare {
        left: s(me),
        right: s((me + 1) % 3),
    }
}

/// Add two party shares locally (no communication required).
pub fn add_party(a: &PartyShare, b: &PartyShare) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_add(b.left),
        right: a.right.wrapping_add(b.right),
    }
}

/// Subtract two party shares locally.
pub fn sub_party(a: &PartyShare, b: &PartyShare) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_sub(b.left),
        right: a.right.wrapping_sub(b.right),
    }
}

/// Multiply a party share by a public scalar locally.
pub fn scalar_mul_party(a: &PartyShare, k: Fr) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_mul(k),
        right: a.right.wrapping_mul(k),
    }
}

/// Reveal a shared value to `reveal_to`.
///
/// - If `reveal_to == Some(p)`, all parties send their left component to p.
/// - If `reveal_to == None`, all parties broadcast; everyone reconstructs.
pub fn reveal<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    reveal_to: Option<PartyId>,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Fr {
    // Collect `N - 1` RevealShare contributions for `share_id`, summing them
    // into `start`.  Buffering means peer contributions can arrive in any order.
    fn collect<T: Transport>(mbx: &Mailbox<T>, share_id: u64, start: Fr) -> Fr {
        let mut sum = start;
        let mut received = 0usize;
        while received < N - 1 {
            if let Some((_from, m)) = mbx.recv_matching(
                TIMEOUT_MS,
                |m| matches!(m, MpcMsg::RevealShare { share_id: sid, .. } if *sid == share_id),
            ) {
                if let MpcMsg::RevealShare { value: v, .. } = m {
                    sum = sum.wrapping_add(v);
                    received += 1;
                }
            } else {
                panic!(
                    "reveal: timeout waiting for RevealShare share_id={}",
                    share_id
                );
            }
        }
        sum
    }

    match reveal_to {
        Some(target) => {
            if me != target {
                mbx.send(
                    target,
                    serialize_msg(&MpcMsg::RevealShare {
                        share_id,
                        value: share.left,
                    }),
                );
                return 0;
            }
            collect(mbx, share_id, share.left)
        }
        None => {
            mbx.broadcast(serialize_msg(&MpcMsg::RevealShare {
                share_id,
                value: share.left,
            }));
            collect(mbx, share_id, share.left)
        }
    }
}

/// Beaver triple from the dealer's perspective: distribute triple components
/// to each party.
pub fn deal_triple<T: Transport>(
    triple_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> (PartyShare, PartyShare, PartyShare) {
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();

    let x_parts = party_pairs(&x_sh);
    let y_parts = party_pairs(&y_sh);
    let z_parts = party_pairs(&z_sh);

    for i in 0..N {
        if i == me {
            continue;
        }
        mbx.send(
            i,
            serialize_msg(&MpcMsg::TripleComponent {
                triple_id,
                left: x_parts[i].left,
                right: x_parts[i].right,
            }),
        );
        mbx.send(
            i,
            serialize_msg(&MpcMsg::TripleComponent {
                triple_id: triple_id + 1,
                left: y_parts[i].left,
                right: y_parts[i].right,
            }),
        );
        mbx.send(
            i,
            serialize_msg(&MpcMsg::TripleComponent {
                triple_id: triple_id + 2,
                left: z_parts[i].left,
                right: z_parts[i].right,
            }),
        );
    }

    (
        x_parts[me].clone(),
        y_parts[me].clone(),
        z_parts[me].clone(),
    )
}

/// Receive a triple that was dealt by another party.
pub fn receive_triple<T: Transport>(
    triple_id: u64,
    mbx: &Mailbox<T>,
) -> (PartyShare, PartyShare, PartyShare) {
    let mut x: Option<PartyShare> = None;
    let mut y: Option<PartyShare> = None;
    let mut z: Option<PartyShare> = None;

    while x.is_none() || y.is_none() || z.is_none() {
        if let Some((_from, m)) = mbx.recv_matching(TIMEOUT_MS, |m| {
            matches!(m, MpcMsg::TripleComponent { triple_id: tid, .. }
                if (triple_id..=triple_id + 2).contains(tid))
        }) {
            if let MpcMsg::TripleComponent {
                triple_id: tid,
                left,
                right,
            } = m
            {
                if tid == triple_id {
                    x = Some(PartyShare { left, right });
                } else if tid == triple_id + 1 {
                    y = Some(PartyShare { left, right });
                } else if tid == triple_id + 2 {
                    z = Some(PartyShare { left, right });
                }
            }
        } else {
            panic!("receive_triple: timeout for triple_id={}", triple_id);
        }
    }
    (x.unwrap(), y.unwrap(), z.unwrap())
}

/// ABY3 multiplication using Beaver triples.
pub fn mul_party<T: Transport>(
    a: &PartyShare,
    b: &PartyShare,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> PartyShare {
    let triple_id = op_id + 1;

    // Party 0 deals the triple; the others receive it.  The `Mailbox` buffers
    // any D/E messages that race ahead of the triple, so no ordering hack (the
    // former `sleep`) is needed on the receiving parties.
    let (x, y, z) = if me == 0 {
        deal_triple(triple_id, me, mbx)
    } else {
        receive_triple(triple_id, mbx)
    };

    let d_i = a.left.wrapping_sub(x.left);
    let e_i = b.left.wrapping_sub(y.left);

    mbx.broadcast(serialize_msg(&MpcMsg::TaggedD { op_id, value: d_i }));
    mbx.broadcast(serialize_msg(&MpcMsg::TaggedE { op_id, value: e_i }));

    let mut d_shares = [0u128; N];
    let mut e_shares = [0u128; N];
    d_shares[me] = d_i;
    e_shares[me] = e_i;
    let mut got_d = [false; N];
    let mut got_e = [false; N];
    got_d[me] = true;
    got_e[me] = true;
    let mut remaining = (N - 1) * 2;

    while remaining > 0 {
        if let Some((from, m)) = mbx.recv_matching(TIMEOUT_MS, |m| {
            matches!(m,
                MpcMsg::TaggedD { op_id: oid, .. } | MpcMsg::TaggedE { op_id: oid, .. }
                    if *oid == op_id)
        }) {
            match m {
                MpcMsg::TaggedD { value: v, .. } if !got_d[from] => {
                    d_shares[from] = v;
                    got_d[from] = true;
                    remaining -= 1;
                }
                MpcMsg::TaggedE { value: v, .. } if !got_e[from] => {
                    e_shares[from] = v;
                    got_e[from] = true;
                    remaining -= 1;
                }
                _ => {}
            }
        } else {
            panic!("mul_party: timeout collecting D/E for op_id={}", op_id);
        }
    }

    let d: Fr = d_shares.iter().fold(0u128, |acc, &v| acc.wrapping_add(v));
    let e: Fr = e_shares.iter().fold(0u128, |acc, &v| acc.wrapping_add(v));

    let de_pub = share_public(d.wrapping_mul(e), me);
    let dy = scalar_mul_party(&y, d);
    let ex = scalar_mul_party(&x, e);
    let t = add_party(&z, &dy);
    let t = add_party(&t, &ex);
    add_party(&t, &de_pub)
}

/// Deal Beaver triples for `count` independent multiplications at once,
/// starting at `base_triple_id` (each occupying 3 consecutive ids — the same
/// per-triple convention as [`deal_triple`] — so batched and single
/// multiplications can freely interleave).  Sends one [`MpcMsg::TripleBatch`]
/// per peer instead of `3*count` separate [`MpcMsg::TripleComponent`]s.
///
/// Each of the `count` triples is freshly and independently random — only the
/// *network I/O* is batched, not the randomness — so this is exactly as
/// secure as `count` separate `deal_triple` calls.
pub fn deal_triple_batch<T: Transport>(
    base_triple_id: u64,
    count: usize,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<(PartyShare, PartyShare, PartyShare)> {
    let mut mine = Vec::with_capacity(count);
    let mut per_peer: Vec<Vec<u128>> = (0..N).map(|_| Vec::with_capacity(count * 6)).collect();

    for _ in 0..count {
        let (x_sh, y_sh, z_sh) = gen_beaver_triple();
        let x_parts = party_pairs(&x_sh);
        let y_parts = party_pairs(&y_sh);
        let z_parts = party_pairs(&z_sh);
        for (i, peer_buf) in per_peer.iter_mut().enumerate() {
            if i == me {
                continue;
            }
            peer_buf.push(x_parts[i].left);
            peer_buf.push(x_parts[i].right);
            peer_buf.push(y_parts[i].left);
            peer_buf.push(y_parts[i].right);
            peer_buf.push(z_parts[i].left);
            peer_buf.push(z_parts[i].right);
        }
        mine.push((
            x_parts[me].clone(),
            y_parts[me].clone(),
            z_parts[me].clone(),
        ));
    }

    for (i, components) in per_peer.into_iter().enumerate() {
        if i == me {
            continue;
        }
        mbx.send(
            i,
            serialize_msg(&MpcMsg::TripleBatch {
                base_triple_id,
                components,
            }),
        );
    }
    mine
}

/// Receive a batch of triples dealt by [`deal_triple_batch`].
pub fn receive_triple_batch<T: Transport>(
    base_triple_id: u64,
    count: usize,
    mbx: &Mailbox<T>,
) -> Vec<(PartyShare, PartyShare, PartyShare)> {
    let Some((_from, m)) = mbx.recv_matching(
        TIMEOUT_MS,
        |m| matches!(m, MpcMsg::TripleBatch { base_triple_id: id, .. } if *id == base_triple_id),
    ) else {
        panic!("receive_triple_batch: timeout for base_triple_id={base_triple_id}");
    };
    let MpcMsg::TripleBatch { components, .. } = m else {
        unreachable!("recv_matching predicate guarantees TripleBatch");
    };
    assert_eq!(
        components.len(),
        count * 6,
        "receive_triple_batch: expected {} components for count={count}, got {}",
        count * 6,
        components.len()
    );
    (0..count)
        .map(|i| {
            let b = i * 6;
            (
                PartyShare {
                    left: components[b],
                    right: components[b + 1],
                },
                PartyShare {
                    left: components[b + 2],
                    right: components[b + 3],
                },
                PartyShare {
                    left: components[b + 4],
                    right: components[b + 5],
                },
            )
        })
        .collect()
}

/// Batched ABY3 multiplication: `pairs.len()` independent multiplications
/// sharing **one** communication round (Beaver-triple dealing + D/E opening),
/// instead of `pairs.len()` separate rounds for that many [`mul_party`]
/// calls.  This is the wire-level fix for MPC's lack of multiplication
/// batching — the same per-element math as `mul_party`, just with the
/// network I/O collapsed from O(n) messages to O(1).
///
/// `base_op_id` must not be reused until this call returns (mirroring
/// `mul_party`'s `op_id`/`op_id+1` convention, generalized to
/// `base_op_id+1 ..= base_op_id + 3*pairs.len()` for the triple range).
pub fn mul_batch_party<T: Transport>(
    pairs: &[(PartyShare, PartyShare)],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let count = pairs.len();
    if count == 0 {
        return Vec::new();
    }
    let base_triple_id = base_op_id + 1;

    let triples = if me == 0 {
        deal_triple_batch(base_triple_id, count, me, mbx)
    } else {
        receive_triple_batch(base_triple_id, count, mbx)
    };

    let d_values: Vec<u128> = pairs
        .iter()
        .zip(&triples)
        .map(|((a, _b), (x, _y, _z))| a.left.wrapping_sub(x.left))
        .collect();
    let e_values: Vec<u128> = pairs
        .iter()
        .zip(&triples)
        .map(|((_a, b), (_x, y, _z))| b.left.wrapping_sub(y.left))
        .collect();

    mbx.broadcast(serialize_msg(&MpcMsg::TaggedDEBatch {
        base_op_id,
        d_values: d_values.clone(),
        e_values: e_values.clone(),
    }));

    let mut d_shares: Vec<[u128; N]> = vec![[0u128; N]; count];
    let mut e_shares: Vec<[u128; N]> = vec![[0u128; N]; count];
    for i in 0..count {
        d_shares[i][me] = d_values[i];
        e_shares[i][me] = e_values[i];
    }
    let mut got = [false; N];
    got[me] = true;
    let mut remaining = N - 1;

    while remaining > 0 {
        let Some((from, m)) = mbx.recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::TaggedDEBatch { base_op_id: oid, .. } if *oid == base_op_id),
        ) else {
            panic!("mul_batch_party: timeout collecting D/E batch for base_op_id={base_op_id}");
        };
        if got[from] {
            continue;
        }
        let MpcMsg::TaggedDEBatch {
            d_values: dv,
            e_values: ev,
            ..
        } = m
        else {
            unreachable!("recv_matching predicate guarantees TaggedDEBatch");
        };
        assert_eq!(dv.len(), count, "mul_batch_party: d_values length mismatch");
        assert_eq!(ev.len(), count, "mul_batch_party: e_values length mismatch");
        for i in 0..count {
            d_shares[i][from] = dv[i];
            e_shares[i][from] = ev[i];
        }
        got[from] = true;
        remaining -= 1;
    }

    (0..count)
        .map(|i| {
            let d: Fr = d_shares[i]
                .iter()
                .fold(0u128, |acc, &v| acc.wrapping_add(v));
            let e: Fr = e_shares[i]
                .iter()
                .fold(0u128, |acc, &v| acc.wrapping_add(v));
            let (x, y, z) = &triples[i];
            let de_pub = share_public(d.wrapping_mul(e), me);
            let dy = scalar_mul_party(y, d);
            let ex = scalar_mul_party(x, e);
            let t = add_party(z, &dy);
            let t = add_party(&t, &ex);
            add_party(&t, &de_pub)
        })
        .collect()
}

/// Arithmetic-to-bit decomposition.
pub fn a2b_party<T: Transport>(
    a: &PartyShare,
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&op_id.to_le_bytes());
    use rand::SeedableRng;
    let mut seeded_rng = rand::rngs::StdRng::from_seed(rng_seed);
    let r: Fr = seeded_rng.gen::<u128>() & ((1u128 << nbits) - 1);

    let mut r_bits: Vec<PartyShare> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let bit = ((r >> i) & 1) != 0;
        let bv: Fr = if bit { 1 } else { 0 };
        r_bits.push(share_public(bv, me));
    }

    let mut r_sh = share_public(0, me);
    for (i, r_bit) in r_bits.iter().enumerate() {
        let term = scalar_mul_party(r_bit, 1u128 << i);
        r_sh = add_party(&r_sh, &term);
    }

    let d_i = a.left.wrapping_sub(r_sh.left);
    mbx.broadcast(serialize_msg(&MpcMsg::TaggedD { op_id, value: d_i }));

    let mut d_shares = [0u128; N];
    d_shares[me] = d_i;
    let mut got_d = [false; N];
    got_d[me] = true;
    let mut remaining = N - 1;

    while remaining > 0 {
        if let Some((from, m)) = mbx.recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::TaggedD { op_id: oid, .. } if *oid == op_id),
        ) {
            if let MpcMsg::TaggedD { value: v, .. } = m {
                if !got_d[from] {
                    d_shares[from] = v;
                    got_d[from] = true;
                    remaining -= 1;
                }
            }
        } else {
            panic!("a2b_party: timeout collecting D for op_id={}", op_id);
        }
    }

    let d: Fr = d_shares.iter().fold(0u128, |acc, &v| acc.wrapping_add(v));

    let mut out: Vec<PartyShare> = Vec::with_capacity(nbits);
    let mut carry = share_public(0, me);

    for (i, r_bit) in r_bits.iter().enumerate().take(nbits) {
        let d_bit: Fr = (d >> i) & 1;
        let r_bit_val: Fr = (r >> i) & 1;

        let s1 = xor_public_party(r_bit, &share_public(d_bit, me), me);
        let sum = xor_public_party(&s1, &carry, me);
        out.push(sum);

        let r_and_d: Fr = r_bit_val.wrapping_mul(d_bit);
        let r_xor_d: Fr = r_bit_val ^ d_bit;
        let xor_times_carry = scalar_mul_party(&carry, r_xor_d);
        let carry_base = share_public(r_and_d, me);
        carry = add_party(&carry_base, &xor_times_carry);
    }

    out
}

/// XOR two party shares.
fn xor_public_party(a: &PartyShare, b: &PartyShare, _me: PartyId) -> PartyShare {
    let two_ab = PartyShare {
        left: a.left.wrapping_mul(2).wrapping_mul(b.left),
        right: a.right.wrapping_mul(2).wrapping_mul(b.right),
    };
    let ab = add_party(a, b);
    sub_party(&ab, &two_ab)
}

/// Reveal a bit share to all parties.
pub fn reveal_bit_to_all<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> bool {
    let v = reveal(share, share_id, None, me, mbx);
    v % 2 != 0
}

fn party_pairs(s: &Share3) -> Vec<PartyShare> {
    vec![
        PartyShare {
            left: s.0,
            right: s.1,
        },
        PartyShare {
            left: s.1,
            right: s.2,
        },
        PartyShare {
            left: s.2,
            right: s.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;

    /// Run `f` (given `(me, mailbox)`) concurrently across 3 in-process
    /// parties and collect their results in party-index order.
    fn run_parties<R: Send + 'static>(
        f: impl Fn(PartyId, Mailbox<InprocHandle>) -> R + Send + Sync + Clone + 'static,
    ) -> Vec<R> {
        let handles = InprocHandle::create(3);
        let joins: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(me, h)| {
                let f = f.clone();
                std::thread::spawn(move || f(me, Mailbox::new(h)))
            })
            .collect();
        joins.into_iter().map(|j| j.join().unwrap()).collect()
    }

    /// Secret-share `values[i]` (owned by party 0) for i in 0..values.len(),
    /// returning this party's `PartyShare` for each.
    fn share_all<T: Transport>(values: &[i64], me: PartyId, mbx: &Mailbox<T>) -> Vec<PartyShare> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let fr = if me == 0 { v as u128 } else { 0 };
                share_input(fr, i as u64 * 10, 0, me, mbx)
            })
            .collect()
    }

    #[test]
    fn mul_batch_matches_sequential_mul_party() {
        let a_vals = [3i64, -5, 7, 0, 12];
        let b_vals = [4i64, 6, -2, 9, 11];
        let expected: Vec<i64> = a_vals.iter().zip(&b_vals).map(|(a, b)| a * b).collect();

        let results = run_parties(move |me, mbx| {
            let a_shares = share_all(&a_vals, me, &mbx);
            let b_shares = share_all(&b_vals, me, &mbx);
            let pairs: Vec<_> = a_shares.into_iter().zip(b_shares).collect();

            let out = mul_batch_party(&pairs, 1000, me, &mbx);

            // Reveal every product to party 0.
            out.into_iter()
                .enumerate()
                .map(|(i, ps)| reveal(&ps, 2000 + i as u64, Some(0), me, &mbx))
                .collect::<Vec<Fr>>()
        });

        // Only party 0 (the reveal target) has the real sums; the others'
        // `reveal` calls return 0 by design (see `reveal`'s doc).
        let got: Vec<i64> = results[0].iter().map(|&v| v as i128 as i64).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn mul_batch_matches_sequential_mul_party_for_single_pair() {
        // count=1 must behave identically to a plain `mul_party` call.
        let results = run_parties(|me, mbx| {
            let a = share_input(if me == 0 { 6 } else { 0 }, 0, 0, me, &mbx);
            let b = share_input(if me == 0 { 7 } else { 0 }, 1, 0, me, &mbx);
            let batched = mul_batch_party(&[(a.clone(), b.clone())], 100, me, &mbx)
                .into_iter()
                .next()
                .unwrap();
            reveal(&batched, 200, Some(0), me, &mbx) as i128 as i64
        });
        assert_eq!(results[0], 42);
    }

    #[test]
    fn mul_batch_uses_one_round_instead_of_n() {
        use crate::bench::comm::CountingTransport;

        let n = 8usize;
        let a_vals: Vec<i64> = (0..n as i64).collect();
        let b_vals: Vec<i64> = (0..n as i64).map(|i| i + 1).collect();

        let handles = InprocHandle::create(3);
        let joins: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(me, h)| {
                let a_vals = a_vals.clone();
                let b_vals = b_vals.clone();
                std::thread::spawn(move || {
                    let counted = CountingTransport::new(h, 3);
                    let handle = counted.handle();
                    let mbx = Mailbox::new(counted);

                    let a_shares = share_all(&a_vals, me, &mbx);
                    let b_shares = share_all(&b_vals, me, &mbx);
                    let pairs: Vec<_> = a_shares.into_iter().zip(b_shares).collect();

                    let batched_stats_before = handle.stats();
                    let _ = mul_batch_party(&pairs, 1000, me, &mbx);
                    let batched_stats = handle.stats().saturating_sub(&batched_stats_before);

                    // Sequential baseline: n separate mul_party calls on fresh shares.
                    let a2 = share_all(&a_vals, me, &mbx);
                    let b2 = share_all(&b_vals, me, &mbx);
                    let seq_before = handle.stats();
                    for (i, (a, b)) in a2.iter().zip(&b2).enumerate() {
                        let _ = mul_party(a, b, 5000 + i as u64 * 10, me, &mbx);
                    }
                    let seq_stats = handle.stats().saturating_sub(&seq_before);

                    (batched_stats, seq_stats)
                })
            })
            .collect();
        let results: Vec<_> = joins.into_iter().map(|j| j.join().unwrap()).collect();

        let (batched, seq) = results[0];
        assert!(
            batched.messages < seq.messages,
            "batched ({}) should use far fewer messages than sequential ({}) for n={n}",
            batched.messages,
            seq.messages
        );
        // Sequential does n independent rounds -> message count scales with n;
        // batched does it in one round -> message count is O(1) in n.
        assert!(
            seq.messages >= batched.messages * (n as u64) / 2,
            "expected roughly n-fold more messages sequentially: batched={} seq={} n={n}",
            batched.messages,
            seq.messages
        );
    }
}
