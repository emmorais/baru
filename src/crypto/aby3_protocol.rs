//! ABY3-specific protocol helpers for additive 3-party secret sharing.
//!
//! This module contains the ABY3 protocol primitives: PartyShare representation,
//! share input/output, multiplication via Beaver triples, a2b decomposition,
//! and related operations. These are specific to the ABY3 scheme and should not
//! be used directly by the evaluator or generic backends.

use std::thread::sleep;

use rand::Rng;

use crate::crypto::arith::gen_beaver_triple;
use crate::crypto::net::{
    deserialize_msg, recv_blocking, serialize_msg, MpcMsg, PartyId, Transport,
};
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
    transport: &T,
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
                transport.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[i],
                    }),
                );
                transport.send(
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
            if let Some((_from, msg)) = recv_blocking(transport, TIMEOUT_MS) {
                let m: MpcMsg = deserialize_msg(&msg);
                if let MpcMsg::InputShare {
                    share_id: sid,
                    value: v,
                } = m
                {
                    if sid == share_id {
                        components.push(v);
                    }
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
pub fn share_public(value: Fr, me: PartyId) -> PartyShare {
    let left = if me == 0 { value } else { 0 };
    let right = 0;
    PartyShare { left, right }
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
    transport: &T,
) -> Fr {
    match reveal_to {
        Some(target) => {
            if me != target {
                transport.send(
                    target,
                    serialize_msg(&MpcMsg::RevealShare {
                        share_id,
                        value: share.left,
                    }),
                );
                return 0;
            }
            let mut sum = share.left;
            let mut received = 0usize;
            while received < N - 1 {
                if let Some((_from, msg)) = recv_blocking(transport, TIMEOUT_MS) {
                    let m: MpcMsg = deserialize_msg(&msg);
                    if let MpcMsg::RevealShare {
                        share_id: sid,
                        value: v,
                    } = m
                    {
                        if sid == share_id {
                            sum = sum.wrapping_add(v);
                            received += 1;
                        }
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
        None => {
            transport.broadcast(serialize_msg(&MpcMsg::RevealShare {
                share_id,
                value: share.left,
            }));
            let mut sum = share.left;
            let mut received = 0usize;
            while received < N - 1 {
                if let Some((_from, msg)) = recv_blocking(transport, TIMEOUT_MS) {
                    let m: MpcMsg = deserialize_msg(&msg);
                    if let MpcMsg::RevealShare {
                        share_id: sid,
                        value: v,
                    } = m
                    {
                        if sid == share_id {
                            sum = sum.wrapping_add(v);
                            received += 1;
                        }
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
    }
}

/// Beaver triple from the dealer's perspective: distribute triple components
/// to each party.
pub fn deal_triple<T: Transport>(
    triple_id: u64,
    me: PartyId,
    transport: &T,
) -> (PartyShare, PartyShare, PartyShare) {
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();

    let x_parts = party_pairs(&x_sh);
    let y_parts = party_pairs(&y_sh);
    let z_parts = party_pairs(&z_sh);

    for i in 0..N {
        if i == me {
            continue;
        }
        transport.send(
            i,
            serialize_msg(&MpcMsg::TripleComponent {
                triple_id,
                left: x_parts[i].left,
                right: x_parts[i].right,
            }),
        );
        transport.send(
            i,
            serialize_msg(&MpcMsg::TripleComponent {
                triple_id: triple_id + 1,
                left: y_parts[i].left,
                right: y_parts[i].right,
            }),
        );
        transport.send(
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
    transport: &T,
) -> (PartyShare, PartyShare, PartyShare) {
    let mut x: Option<PartyShare> = None;
    let mut y: Option<PartyShare> = None;
    let mut z: Option<PartyShare> = None;

    while x.is_none() || y.is_none() || z.is_none() {
        if let Some((_from, msg)) = recv_blocking(transport, TIMEOUT_MS) {
            let m: MpcMsg = deserialize_msg(&msg);
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
    transport: &T,
) -> PartyShare {
    println!("mul_party: op_id={}, me={}", op_id, me);
    let triple_id = op_id + 1;

    let (x, y, z) = if me == 0 {
        deal_triple(triple_id, me, transport)
    } else {
        sleep(std::time::Duration::from_millis(500));
        receive_triple(triple_id, transport)
    };

    let d_i = a.left.wrapping_sub(x.left);
    let e_i = b.left.wrapping_sub(y.left);

    transport.broadcast(serialize_msg(&MpcMsg::TaggedD { op_id, value: d_i }));
    println!("broadcast d_i for op_id={}", op_id);
    transport.broadcast(serialize_msg(&MpcMsg::TaggedE { op_id, value: e_i }));
    println!("broadcast e_i for op_id={}", op_id);

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
        println!("remaining = {}", remaining);
        if let Some((from, msg)) = recv_blocking(transport, TIMEOUT_MS) {
            println!(
                "mul_party: op_id={}, me={}, received from {}",
                op_id, me, from
            );
            let m: MpcMsg = deserialize_msg(&msg);
            println!("m: {:?}", m);
            match m {
                MpcMsg::TaggedD {
                    op_id: oid,
                    value: v,
                } if oid == op_id => {
                    if !got_d[from] {
                        d_shares[from] = v;
                        got_d[from] = true;
                        remaining -= 1;
                    }
                }
                MpcMsg::TaggedE {
                    op_id: oid,
                    value: v,
                } if oid == op_id && !got_e[from] => {
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

/// Arithmetic-to-bit decomposition.
pub fn a2b_party<T: Transport>(
    a: &PartyShare,
    nbits: usize,
    op_id: u64,
    me: PartyId,
    transport: &T,
) -> Vec<PartyShare> {
    println!("a2b_party: op_id={}, me={}", op_id, me);
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
    transport.broadcast(serialize_msg(&MpcMsg::TaggedD { op_id, value: d_i }));

    let mut d_shares = [0u128; N];
    d_shares[me] = d_i;
    let mut got_d = [false; N];
    got_d[me] = true;
    let mut remaining = N - 1;

    while remaining > 0 {
        if let Some((from, msg)) = recv_blocking(transport, TIMEOUT_MS) {
            println!(
                "a2b_party: op_id={}, me={}, received from {}",
                op_id, me, from
            );
            let m: MpcMsg = deserialize_msg(&msg);
            if let MpcMsg::TaggedD {
                op_id: oid,
                value: v,
            } = m
            {
                if oid == op_id && !got_d[from] {
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
    transport: &T,
) -> bool {
    let v = reveal(share, share_id, None, me, transport);
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
