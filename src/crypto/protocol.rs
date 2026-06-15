//! Distributed MPC protocol helpers used by `TransportBackend`.
//!
//! All functions here are *per-party*: each party calls the same function with
//! its own `Transport` handle, and the parties communicate to jointly execute
//! the protocol.  The functions are synchronous (blocking) and use
//! `recv_blocking` with a generous timeout.

use std::thread::sleep;

use rand::Rng;

use crate::crypto::arith::gen_beaver_triple;
use crate::crypto::net::{
    deserialize_msg, recv_blocking, serialize_msg, MpcMsg, PartyId, Transport,
};
use crate::crypto::types::{Fr, Share3};

const N: usize = 3;
const TIMEOUT_MS: u64 = 5_000;

// ---------------------------------------------------------------------------
// Per-party share storage type
// ---------------------------------------------------------------------------

/// The two field elements that party `id` holds for a given shared value.
/// In ABY3, party i holds (s_i, s_{(i+1) mod 3}).
#[derive(Clone, Debug)]
pub struct PartyShare {
    pub left: Fr,  // s_i
    pub right: Fr, // s_{(i+1) mod 3}
}

// ---------------------------------------------------------------------------
// Input sharing
// ---------------------------------------------------------------------------

/// Distribute a secret input `value` to all three parties.
///
/// - Party `owner` calls this with the actual `value`.
/// - All other parties call this with `value = 0` (ignored).
///
/// Returns the `PartyShare` for the *calling* party.
/// After this call every party holds its two-component share.
pub fn share_input<T: Transport>(
    value: Fr,
    share_id: u64,
    owner: PartyId,
    me: PartyId,
    transport: &T,
) -> PartyShare {
    if me == owner {
        // sample two random components; third determined by secret
        let mut rng = rand::thread_rng();
        let s: [Fr; 3] = {
            let s0: Fr = rng.gen();
            let s1: Fr = rng.gen();
            let s2 = value.wrapping_sub(s0).wrapping_sub(s1);
            [s0, s1, s2]
        };
        // party i holds (s_i, s_{i+1 mod 3})
        // send each party its left component first, then right component
        for i in 0..N {
            let right_idx = (i + 1) % N;
            if i != me {
                // left component first
                transport.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[i],
                    }),
                );
                // then right component
                transport.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[right_idx],
                    }),
                );
            }
        }
        // owner's own share
        PartyShare {
            left: s[me],
            right: s[(me + 1) % N],
        }
    } else {
        // wait for two InputShare messages for this share_id
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
        // first received = left, second = right (owner sends left first then right for each peer)
        PartyShare {
            left: components[0],
            right: components[1],
        }
    }
}

// ---------------------------------------------------------------------------
// Public-value sharing (no communication needed)
// ---------------------------------------------------------------------------

/// Share a public constant locally (no communication).  Party 0 gets the
/// value in its left component; others get 0 so that reconstruction = value.
pub fn share_public(value: Fr, me: PartyId) -> PartyShare {
    // s0 = value, s1 = 0, s2 = 0  so s0+s1+s2 = value
    let left = if me == 0 { value } else { 0 };
    let right = 0;
    PartyShare { left, right }
}

// ---------------------------------------------------------------------------
// Local arithmetic (non-interactive)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Reveal
// ---------------------------------------------------------------------------

/// Reveal a shared value to `reveal_to`.
///
/// - If `reveal_to == Some(p)`, all parties send their left component to p.
/// - If `reveal_to == None`, all parties broadcast; everyone reconstructs.
///
/// Returns the reconstructed value for the caller if they are the reveal
/// target (or if reveal_to == None).  Returns 0 for non-target parties.
pub fn reveal<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    reveal_to: Option<PartyId>,
    me: PartyId,
    transport: &T,
) -> Fr {
    match reveal_to {
        Some(target) => {
            // send our left component to the target
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
            // we are the target: collect from the other two parties
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
            // broadcast to everyone
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

// ---------------------------------------------------------------------------
// Beaver triple dealer
// ---------------------------------------------------------------------------

/// Beaver triple from the dealer's perspective: distribute triple components
/// to each party.  Call this once from the *dealer* (e.g. party 0 during
/// preprocessing) and `receive_triple` on the other parties.
///
/// In a full implementation triples would be generated with a secure protocol;
/// here we use a trusted dealer for simplicity.
pub fn deal_triple<T: Transport>(
    triple_id: u64,
    me: PartyId,
    transport: &T,
) -> (PartyShare, PartyShare, PartyShare) {
    // Generate a complete triple centrally (dealer model)
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();

    let x_parts = party_pairs(&x_sh);
    let y_parts = party_pairs(&y_sh);
    let z_parts = party_pairs(&z_sh);

    // send each party its components
    for i in 0..N {
        println!("Dealer sending triple_id={} to party {}", triple_id, i);
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
        // encode y and z in subsequent messages; use triple_id+1 and triple_id+2
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

// ---------------------------------------------------------------------------
// Distributed multiplication (Beaver triple online phase)
// ---------------------------------------------------------------------------

/// ABY3 multiplication using Beaver triples over a Transport.
///
/// All three parties call this function concurrently.  Party 0 acts as
/// the triple dealer; others receive their triple share.
pub fn mul_party<T: Transport>(
    a: &PartyShare,
    b: &PartyShare,
    op_id: u64,
    me: PartyId,
    transport: &T,
) -> PartyShare {
    println!("mul_party: op_id={}, me={}", op_id, me);
    //let triple_id = op_id + 3; // reserve 3 ids for x/y/z
    let triple_id = op_id + 1; // get a fresh triple_id for this multiplication instance

    // obtain triple
    let (x, y, z) = if me == 0 {
        deal_triple(triple_id, me, transport)
    } else {
        sleep(std::time::Duration::from_millis(100)); // give the dealer a chance to send before we start recv'ing
        receive_triple(triple_id, transport)
    };

    // compute local d_i = a_i - x_i  and  e_i = b_i - y_i
    let d_i = a.left.wrapping_sub(x.left);
    let e_i = b.left.wrapping_sub(y.left);

    // broadcast d_i and e_i to all other parties
    transport.broadcast(serialize_msg(&MpcMsg::TaggedD { op_id, value: d_i }));
    println!("broadcast d_i for op_id={}", op_id);  
    transport.broadcast(serialize_msg(&MpcMsg::TaggedE { op_id, value: e_i }));
    println!("broadcast e_i for op_id={}", op_id);  

    // collect d and e from the other two parties
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
            println!("mul_party: op_id={}, me={}, received from {}", op_id, me, from);
            let m: MpcMsg = deserialize_msg(&msg);
            println!("m: {:?}", m);
            match m {
                MpcMsg::TaggedD {
                    op_id: oid,
                    value: v,
                } if (oid == op_id) => {
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

    // reconstruct public d and e
    let d: Fr = d_shares.iter().fold(0u128, |acc, &v| acc.wrapping_add(v));
    let e: Fr = e_shares.iter().fold(0u128, |acc, &v| acc.wrapping_add(v));

    // compute output share:  z + d*y + e*x + share_public(d*e)
    let de_pub = share_public(d.wrapping_mul(e), me);
    let dy = scalar_mul_party(&y, d);
    let ex = scalar_mul_party(&x, e);
    let t = add_party(&z, &dy);
    let t = add_party(&t, &ex);
    add_party(&t, &de_pub)
}

// ---------------------------------------------------------------------------
// Distributed a2b
// ---------------------------------------------------------------------------

/// Arithmetic-to-bit decomposition over a Transport.
///
/// Returns `nbits` `PartyShare` values (LSB first), each sharing one bit
/// of the arithmetic value.
pub fn a2b_party<T: Transport>(
    a: &PartyShare,
    nbits: usize,
    op_id: u64,
    me: PartyId,
    transport: &T,
) -> Vec<PartyShare> {
    println!("a2b_party: op_id={}, me={}", op_id, me);
    // Sample mask r (each party samples its own random r for its left component).
    // We use a shared-seed approach: all parties use op_id as seed so r is consistent.
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&op_id.to_le_bytes());
    use rand::SeedableRng;
    let mut seeded_rng = rand::rngs::StdRng::from_seed(rng_seed);
    let r: Fr = seeded_rng.gen::<u128>() & ((1u128 << nbits) - 1);

    // Build r_bits as PartyShare values using party-local bit shares
    let mut r_bits: Vec<PartyShare> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let bit = ((r >> i) & 1) != 0;
        let bv: Fr = if bit { 1 } else { 0 };
        r_bits.push(share_public(bv, me)); // each party knows r, so it's public
    }

    // assemble r_sh from r_bits
    let mut r_sh = share_public(0, me);
    for (i, r_bit) in r_bits.iter().enumerate() {
        let term = scalar_mul_party(r_bit, 1u128 << i);
        r_sh = add_party(&r_sh, &term);
    }

    // each party computes d_i = a_i - r_i and broadcasts
    let d_i = a.left.wrapping_sub(r_sh.left);
    transport.broadcast(serialize_msg(&MpcMsg::TaggedD { op_id, value: d_i }));

    let mut d_shares = [0u128; N];
    d_shares[me] = d_i;
    let mut got_d = [false; N];
    got_d[me] = true;
    let mut remaining = N - 1;

    while remaining > 0 {
        if let Some((from, msg)) = recv_blocking(transport, TIMEOUT_MS) {
            println!("a2b_party: op_id={}, me={}, received from {}", op_id, me, from);
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

    // compute shared bit decomposition using public d and shared r_bits
    // ripple-carry adder: a = d + r, where d is a public integer and r is secret-shared bit-by-bit
    //
    // For each bit position i:
    //   sum_i  = r_i XOR d_i XOR carry_i
    //   carry_{i+1} = majority(r_i, d_i, carry_i)
    //
    // Since r_i and d_i are both PUBLIC (r is the agreed mask, d = a - r is broadcast),
    // majority(r_i, d_i, carry_i) = r_i*d_i + (r_i XOR d_i)*carry_i
    // — a local computation: r_i*d_i is a scalar-scalar product, (r_i XOR d_i) is
    //   a public bit, so the carry update requires NO communication.
    let mut out: Vec<PartyShare> = Vec::with_capacity(nbits);
    let mut carry = share_public(0, me); // carry starts at 0

    for (i, r_bit) in r_bits.iter().enumerate() {
        let d_bit: Fr = (d >> i) & 1; // public
        let r_bit_val: Fr = (r >> i) & 1; // public (same r for all parties since seeded rng)

        // sum = r_i XOR d_i XOR carry  (local, no communication)
        let s1 = xor_public_party(r_bit, &share_public(d_bit, me), me);
        let sum = xor_public_party(&s1, &carry, me);
        out.push(sum);

        // majority(r_i, d_i, carry) = r_i*d_i + (r_i XOR d_i)*carry
        //   r_i*d_i = scalar (both public)
        //   (r_i XOR d_i) = public scalar
        //   (r_i XOR d_i)*carry = scalar_mul on a secret-shared carry
        let r_and_d: Fr = r_bit_val.wrapping_mul(d_bit); // scalar
        let r_xor_d: Fr = r_bit_val ^ d_bit; // scalar
        let xor_times_carry = scalar_mul_party(&carry, r_xor_d); // local scalar mul
                                                                 // carry_{i+1} = r_and_d_public + (r_xor_d)*carry
        let carry_base = share_public(r_and_d, me);
        carry = add_party(&carry_base, &xor_times_carry);
    }

    out
}

/// XOR two party shares where b may be a public value (d_bit).
/// XOR(a, b) = a + b - 2*a*b; for public b: a + b - 2*a*b = a*(1-2b) + b
fn xor_public_party(a: &PartyShare, b: &PartyShare, _me: PartyId) -> PartyShare {
    // a + b - 2*(a*b) — since b is a public share (only party 0 has nonzero)
    // treat as local: left_a + left_b - 2 * left_a * left_b ... but mod 2 for bits
    // For field arithmetic: result = a + b - 2ab (additive, not modular-2)
    let two_ab = PartyShare {
        left: a.left.wrapping_mul(2).wrapping_mul(b.left),
        right: a.right.wrapping_mul(2).wrapping_mul(b.right),
    };
    let ab = add_party(a, b);
    sub_party(&ab, &two_ab)
}

// ---------------------------------------------------------------------------
// Reveal protocol for comparison results (bit shares)
// ---------------------------------------------------------------------------

/// Reveal a bit share to the calling party.  All parties participate; me
/// receives the reconstructed bit.
pub fn reveal_bit_to_all<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    me: PartyId,
    transport: &T,
) -> bool {
    let v = reveal(share, share_id, None, me, transport);
    v % 2 != 0
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
