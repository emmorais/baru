use crate::crypto::arith::{add_shares, gen_beaver_triple, scalar_mul, share_secret};
use crate::crypto::bits;
use crate::crypto::bits::share_bool;
use crate::crypto::net::{deserialize_msg, serialize_msg, MpcMsg, Transport};
use crate::crypto::types::{Fr, Share3};
use rand::Rng;

// Simple helper: split a Share3 into per-party pairs (party i holds s_i, s_{i+1}).
fn split_to_party_pairs(s: &Share3) -> Vec<(Fr, Fr)> {
    vec![(s.0, s.1), (s.1, s.2), (s.2, s.0)]
}

/// Simulate per-party message passing for ABY3 multiplication protocol over a Transport.
pub fn mul_ab3_protocol<T: Transport + Clone>(a: &Share3, b: &Share3, transport: &T) -> Share3 {
    // Preprocessing: triple
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();

    // split into per-party local views
    let a_pairs = split_to_party_pairs(a);
    let b_pairs = split_to_party_pairs(b);
    let x_pairs = split_to_party_pairs(&x_sh);
    let y_pairs = split_to_party_pairs(&y_sh);

    // Each party computes local d_i and e_i and broadcasts them
    let mut d_shares = [0u128; 3];
    let mut e_shares = [0u128; 3];

    for i in 0..3 {
        if transport.id() == i {
            let di = a_pairs[i].0.wrapping_sub(x_pairs[i].0);
            let ei = b_pairs[i].0.wrapping_sub(y_pairs[i].0);
            // broadcast D and E values
            transport.broadcast(serialize_msg(&MpcMsg::D(di)));
            transport.broadcast(serialize_msg(&MpcMsg::E(ei)));
            // store own contributions
            d_shares[i] = di;
            e_shares[i] = ei;
        }
    }

    // collect broadcasts from others (non-blocking loop until we've got 3 contributions each)
    let mut got_d = vec![false; 3];
    let mut got_e = vec![false; 3];
    got_d[transport.id()] = true;
    got_e[transport.id()] = true;
    let mut pending = 2;
    while pending > 0 {
        if let Some((from, msg)) = transport.recv() {
            let m: MpcMsg = deserialize_msg(&msg);
            match m {
                MpcMsg::D(v) => {
                    if !got_d[from] {
                        d_shares[from] = v;
                        got_d[from] = true;
                        pending -= 1;
                    }
                }
                MpcMsg::E(v) => {
                    if !got_e[from] {
                        e_shares[from] = v;
                        got_e[from] = true;
                        pending -= 1;
                    }
                }
                _ => {}
            }
        }
    }

    // Collect broadcasts (network step) and reconstruct public d,e
    let mut d: Fr = 0;
    let mut e: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
        e = e.wrapping_add(e_shares[i]);
    }

    // Reshare cross term and compute final shares locally
    let cross = share_secret(d.wrapping_mul(e));
    let term_dy = scalar_mul(&y_sh, d);
    let term_ex = scalar_mul(&x_sh, e);
    let s1 = add_shares(&z_sh, &term_dy);
    let s2 = add_shares(&term_ex, &cross);
    add_shares(&s1, &s2)
}

/// Simulate ABY3-style arithmetic-to-bit decomposition using masked opening over Transport.
pub fn a2b_protocol<T: Transport + Clone>(a: &Share3, nbits: usize, transport: &T) -> Vec<Share3> {
    let mut rng = rand::thread_rng();
    let r: Fr = rng.gen();
    // construct r_sh from per-bit shares so r_sh and r_bits are consistent
    let mut r_bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let bit = ((r >> i) & 1) != 0;
        r_bits.push(share_bool(bit));
    }
    // assemble r_sh = sum_i r_bits[i] * (1<<i)
    let mut r_sh = share_secret(0);
    for i in 0..nbits {
        let term = scalar_mul(&r_bits[i], 1u128 << i);
        r_sh = add_shares(&r_sh, &term);
    }

    // split to per-party
    let a_pairs = split_to_party_pairs(a);
    let r_pairs = split_to_party_pairs(&r_sh);

    // each party computes local d_i and broadcasts
    let mut d_shares = [0u128; 3];
    for i in 0..3 {
        if transport.id() == i {
            let di = a_pairs[i].0.wrapping_sub(r_pairs[i].0);
            transport.broadcast(serialize_msg(&MpcMsg::D(di)));
            d_shares[i] = di;
        }
    }

    // collect contributions
    let mut got_d = vec![false; 3];
    got_d[transport.id()] = true;
    let mut pending = 2;
    while pending > 0 {
        if let Some((from, msg)) = transport.recv() {
            let m: MpcMsg = deserialize_msg(&msg);
            if let MpcMsg::D(v) = m {
                if !got_d[from] {
                    d_shares[from] = v;
                    got_d[from] = true;
                    pending -= 1;
                }
            }
        }
    }

    let mut d: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
    }

    // compute bits LSB-first using public d and r_bits
    let mut out: Vec<Share3> = Vec::with_capacity(nbits);
    let mut carry = share_bool(false);
    for i in 0..nbits {
        let d_bit = ((d >> i) & 1) != 0;
        let d_share = share_bool(d_bit);
        let r_bit = &r_bits[i];

        // sum bit = r_i XOR d_i XOR carry
        let s1 = bits::xor_shared(r_bit, &d_share);
        let sum = bits::xor_shared(&s1, &carry);
        out.push(sum);

        // carry_next = majority(r_i, d_i, carry)
        let a_and_b = bits::and(r_bit, &d_share);
        let a_and_c = bits::and(r_bit, &carry);
        let b_and_c = bits::and(&d_share, &carry);
        let t = bits::or(&a_and_c, &b_and_c);
        carry = bits::or(&a_and_b, &t);
    }

    out
}

/// Unsigned less-than implemented via a2b protocol on (a - b) and returning MSB.
pub fn lt_protocol<T: Transport + Clone>(
    a: &Share3,
    b: &Share3,
    nbits: usize,
    transport: &T,
) -> Share3 {
    let diff = crate::crypto::arith::sub_shares(a, b);
    let bits = a2b_protocol(&diff, nbits, transport);
    bits[nbits - 1].clone()
}

pub fn le_protocol<T: Transport + Clone>(
    a: &Share3,
    b: &Share3,
    nbits: usize,
    transport: &T,
) -> Share3 {
    let lt = lt_protocol(a, b, nbits, transport);
    let eq = crate::crypto::bits::eq_ab3(a, b, nbits);
    crate::crypto::bits::or(&lt, &eq)
}

pub fn eq_protocol<T: Transport + Clone>(
    a: &Share3,
    b: &Share3,
    nbits: usize,
    transport: &T,
) -> Share3 {
    let diff = crate::crypto::arith::sub_shares(a, b);
    let bits = a2b_protocol(&diff, nbits, transport);
    let mut acc = bits[0].clone();
    for i in 1..bits.len() {
        acc = crate::crypto::bits::or(&acc, &bits[i]);
    }
    crate::crypto::bits::not(&acc)
}

pub fn a2b_debug(a: &Share3, nbits: usize) -> String {
    use crate::crypto::arith::reconstruct;
    use crate::crypto::bits::share_bool;
    let mut rng = rand::thread_rng();
    let r: Fr = rng.gen::<u128>() & ((1u128 << nbits) - 1);
    // build r_bits and r_sh consistently
    let mut r_bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let bit = ((r >> i) & 1) != 0;
        r_bits.push(share_bool(bit));
    }
    let mut r_sh = share_secret(0);
    for i in 0..nbits {
        let term = scalar_mul(&r_bits[i], 1u128 << i);
        r_sh = add_shares(&r_sh, &term);
    }

    // split to per-party
    let a_pairs = split_to_party_pairs(a);
    let r_pairs = split_to_party_pairs(&r_sh);

    // each party computes local d_i and broadcasts
    let mut d_shares = [0u128; 3];
    for i in 0..3 {
        d_shares[i] = a_pairs[i].0.wrapping_sub(r_pairs[i].0);
    }
    let mut d: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
    }

    // prepare r_bits plain
    let mut r_bits_plain: Vec<bool> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        r_bits_plain.push(((r >> i) & 1) != 0);
    }

    let mut d_bits_plain: Vec<bool> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        d_bits_plain.push(((d >> i) & 1) != 0);
    }

    // compute bits LSB-first and collect reconstructed plains at each step
    let mut out_plain: Vec<bool> = Vec::with_capacity(nbits);
    let mut carry_plain: Vec<bool> = Vec::with_capacity(nbits);
    let mut carry = share_bool(false);
    for i in 0..nbits {
        let d_bit = d_bits_plain[i];
        let d_share = share_bool(d_bit);
        let r_bit_share = share_bool(r_bits_plain[i]);

        // sum bit = r_i XOR d_i XOR carry
        let s1 = crate::crypto::bits::xor_shared(&r_bit_share, &d_share);
        let sum = crate::crypto::bits::xor_shared(&s1, &carry);
        out_plain.push(reconstruct(&sum) % 2 != 0);

        // carry_next = majority(r_i, d_i, carry)
        let a_and_b = crate::crypto::bits::and(&r_bit_share, &d_share);
        let a_and_c = crate::crypto::bits::and(&r_bit_share, &carry);
        let b_and_c = crate::crypto::bits::and(&d_share, &carry);
        let t = crate::crypto::bits::or(&a_and_c, &b_and_c);
        carry = crate::crypto::bits::or(&a_and_b, &t);
        carry_plain.push(reconstruct(&carry) % 2 != 0);
    }

    let mut s = String::new();
    use std::fmt::Write;
    writeln!(s, "a (shared) = {:?}", a).ok();
    writeln!(s, "r = {}", r).ok();
    writeln!(s, "d = {}", d).ok();
    writeln!(s, "r_bits = {:?}", r_bits_plain).ok();
    writeln!(s, "d_bits = {:?}", d_bits_plain).ok();
    writeln!(s, "sum_bits = {:?}", out_plain).ok();
    writeln!(s, "carry_bits = {:?}", carry_plain).ok();
    s
}

/// Run the mul_ab3 protocol using an in-process transport (3 parties).
/// This function orchestrates the broadcast and collection phases using the
/// Transport API but runs sequentially (no threads): each party broadcasts
/// its contributions, then we collect messages from each party's queue, and
/// finally reconstruct the public values and compute the final Share3.
pub fn mul_ab3_with_inproc(a: &Share3, b: &Share3) -> Share3 {
    use crate::crypto::net::InprocHandle;
    // create 3 connected inproc handles
    let handles = InprocHandle::create(3);

    // Preprocessing triple (shared across parties)
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();
    let a_pairs = split_to_party_pairs(a);
    let b_pairs = split_to_party_pairs(b);
    let x_pairs = split_to_party_pairs(&x_sh);
    let y_pairs = split_to_party_pairs(&y_sh);

    // Phase 1: each party broadcasts its local d_i and e_i
    for i in 0..3 {
        let handle = &handles[i];
        let di = a_pairs[i].0.wrapping_sub(x_pairs[i].0);
        let ei = b_pairs[i].0.wrapping_sub(y_pairs[i].0);
        handle.broadcast(serialize_msg(&MpcMsg::D(di)));
        handle.broadcast(serialize_msg(&MpcMsg::E(ei)));
    }

    // Phase 2: collect contributions for each party and reconstruct d,e
    let mut d_shares = [0u128; 3];
    let mut e_shares = [0u128; 3];
    for i in 0..3 {
        let handle = &handles[i];
        // each party should see 2 incoming D and 2 incoming E messages from others
        let mut got_d = vec![false; 3];
        let mut got_e = vec![false; 3];
        let mut pending = 4; // expecting D and E from two other parties
                             // also include own contributions by recomputing them
        let own_d = a_pairs[i].0.wrapping_sub(x_pairs[i].0);
        let own_e = b_pairs[i].0.wrapping_sub(y_pairs[i].0);
        d_shares[i] = own_d;
        got_d[i] = true;
        pending -= 2; // counted own D and E
        e_shares[i] = own_e;
        while pending > 0 {
            if let Some((from, msg)) = handle.recv() {
                let m: MpcMsg = deserialize_msg(&msg);
                match m {
                    MpcMsg::D(v) => {
                        if !got_d[from] {
                            d_shares[from] = v;
                            got_d[from] = true;
                            pending -= 1;
                        }
                    }
                    MpcMsg::E(v) => {
                        if !got_e[from] {
                            e_shares[from] = v;
                            got_e[from] = true;
                            pending -= 1;
                        }
                    }
                    _ => {}
                }
            } else {
                // nothing available in this party's queue: break to allow others to be collected
                break;
            }
        }
    }

    // Reconstruct global d and e
    let mut d: Fr = 0;
    let mut e: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
        e = e.wrapping_add(e_shares[i]);
    }

    // Compute final shares as in mul_ab3_protocol
    let cross = share_secret(d.wrapping_mul(e));
    let term_dy = scalar_mul(&y_sh, d);
    let term_ex = scalar_mul(&x_sh, e);
    let s1 = add_shares(&z_sh, &term_dy);
    let s2 = add_shares(&term_ex, &cross);
    add_shares(&s1, &s2)
}

/// Run a2b over an in-process transport: create handles, have each party
/// broadcast its d = a_i - r_i, collect the broadcasts and compute the
/// bit-decomposition result. Sequential phased execution (no threads).
pub fn a2b_with_inproc(a: &Share3, nbits: usize) -> Vec<Share3> {
    use crate::crypto::net::InprocHandle;
    let handles = InprocHandle::create(3);

    // prepare mask r and r_bits centrally (so r_sh and r_bits consistent)
    let mut rng = rand::thread_rng();
    let r: Fr = rng.gen();
    let mut r_bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let bit = ((r >> i) & 1) != 0;
        r_bits.push(share_bool(bit));
    }
    let mut r_sh = share_secret(0);
    for i in 0..nbits {
        let term = scalar_mul(&r_bits[i], 1u128 << i);
        r_sh = add_shares(&r_sh, &term);
    }

    let a_pairs = split_to_party_pairs(a);
    let r_pairs = split_to_party_pairs(&r_sh);

    // Phase 1: each party broadcasts its local d_i
    for i in 0..3 {
        let handle = &handles[i];
        let di = a_pairs[i].0.wrapping_sub(r_pairs[i].0);
        handle.broadcast(serialize_msg(&MpcMsg::D(di)));
    }

    // Phase 2: collect contributions per party and reconstruct d
    let mut d_shares = [0u128; 3];
    for i in 0..3 {
        let handle = &handles[i];
        let mut got_d = vec![false; 3];
        let mut pending = 2; // expecting D from two other parties
        let own_d = a_pairs[i].0.wrapping_sub(r_pairs[i].0);
        d_shares[i] = own_d;
        got_d[i] = true; // accounted own
        while pending > 0 {
            if let Some((from, msg)) = handle.recv() {
                let m: MpcMsg = deserialize_msg(&msg);
                if let MpcMsg::D(v) = m {
                    if !got_d[from] {
                        d_shares[from] = v;
                        got_d[from] = true;
                        pending -= 1;
                    }
                }
            } else {
                break;
            }
        }
    }

    let mut d: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
    }

    // compute bits using public d and shared r_bits
    let mut out: Vec<Share3> = Vec::with_capacity(nbits);
    let mut carry = share_bool(false);
    for i in 0..nbits {
        let d_bit = ((d >> i) & 1) != 0;
        let d_share = share_bool(d_bit);
        let r_bit = &r_bits[i];
        let s1 = bits::xor_shared(r_bit, &d_share);
        let sum = bits::xor_shared(&s1, &carry);
        out.push(sum);
        let a_and_b = bits::and(r_bit, &d_share);
        let a_and_c = bits::and(r_bit, &carry);
        let b_and_c = bits::and(&d_share, &carry);
        let t = bits::or(&a_and_c, &b_and_c);
        carry = bits::or(&a_and_b, &t);
    }
    out
}

/// Run the mul_ab3 protocol concurrently: spawn 3 threads (one per party), each
/// with its own `InprocHandle` which performs the broadcast. The main thread
/// collects the broadcasts and computes the final shares.
pub fn mul_ab3_concurrent(a: &Share3, b: &Share3) -> Share3 {
    use crate::crypto::net::InprocHandle;
    // create 3 connected inproc handles
    let handles = InprocHandle::create(3);

    // Preprocessing triple (shared across parties)
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();
    let a_pairs = split_to_party_pairs(a);
    let b_pairs = split_to_party_pairs(b);
    let x_pairs = split_to_party_pairs(&x_sh);
    let y_pairs = split_to_party_pairs(&y_sh);

    // spawn threads: each thread broadcasts its local d_i and e_i
    let mut joins = Vec::with_capacity(3);
    for i in 0..3 {
        let handle = handles[i].clone();
        let ai = a_pairs[i].0;
        let bi = b_pairs[i].0;
        let xi = x_pairs[i].0;
        let yi = y_pairs[i].0;
        joins.push(std::thread::spawn(move || {
            let di = ai.wrapping_sub(xi);
            let ei = bi.wrapping_sub(yi);
            handle.broadcast(serialize_msg(&MpcMsg::D(di)));
            handle.broadcast(serialize_msg(&MpcMsg::E(ei)));
        }));
    }
    // wait for broadcasts to finish
    for j in joins {
        let _ = j.join();
    }

    // Phase: collect contributions for each party and reconstruct d,e
    let mut d_shares = [0u128; 3];
    let mut e_shares = [0u128; 3];
    for i in 0..3 {
        let handle = &handles[i];
        // include own contributions
        d_shares[i] = a_pairs[i].0.wrapping_sub(x_pairs[i].0);
        e_shares[i] = b_pairs[i].0.wrapping_sub(y_pairs[i].0);
        let mut got_d = vec![false; 3];
        let mut got_e = vec![false; 3];
        got_d[i] = true;
        got_e[i] = true;
        let mut pending = 4; // expecting D and E from two other parties
        while pending > 0 {
            if let Some((from, msg)) = handle.recv() {
                let m: MpcMsg = deserialize_msg(&msg);
                match m {
                    MpcMsg::D(v) => {
                        if !got_d[from] {
                            d_shares[from] = v;
                            got_d[from] = true;
                            pending -= 1;
                        }
                    }
                    MpcMsg::E(v) => {
                        if !got_e[from] {
                            e_shares[from] = v;
                            got_e[from] = true;
                            pending -= 1;
                        }
                    }
                    _ => {}
                }
            } else {
                // no message currently, yield to allow other threads to progress
                std::thread::yield_now();
            }
        }
    }

    // Reconstruct global d and e
    let mut d: Fr = 0;
    let mut e: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
        e = e.wrapping_add(e_shares[i]);
    }

    // Compute final shares
    let cross = share_secret(d.wrapping_mul(e));
    let term_dy = scalar_mul(&y_sh, d);
    let term_ex = scalar_mul(&x_sh, e);
    let s1 = add_shares(&z_sh, &term_dy);
    let s2 = add_shares(&term_ex, &cross);
    add_shares(&s1, &s2)
}

/// Run a2b concurrently: spawn 3 threads that broadcast their d_i; main thread
/// collects broadcasts and computes the bit decomposition.
pub fn a2b_concurrent(a: &Share3, nbits: usize) -> Vec<Share3> {
    use crate::crypto::net::InprocHandle;
    let handles = InprocHandle::create(3);

    // prepare mask r and r_bits centrally (so r_sh and r_bits consistent)
    let mut rng = rand::thread_rng();
    let r: Fr = rng.gen();
    let mut r_bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let bit = ((r >> i) & 1) != 0;
        r_bits.push(share_bool(bit));
    }
    let mut r_sh = share_secret(0);
    for i in 0..nbits {
        let term = scalar_mul(&r_bits[i], 1u128 << i);
        r_sh = add_shares(&r_sh, &term);
    }

    let a_pairs = split_to_party_pairs(a);
    let r_pairs = split_to_party_pairs(&r_sh);

    // spawn threads to broadcast d_i
    let mut joins = Vec::with_capacity(3);
    for i in 0..3 {
        let handle = handles[i].clone();
        let ai = a_pairs[i].0;
        let ri = r_pairs[i].0;
        joins.push(std::thread::spawn(move || {
            let di = ai.wrapping_sub(ri);
            handle.broadcast(serialize_msg(&MpcMsg::D(di)));
        }));
    }
    for j in joins {
        let _ = j.join();
    }

    // collect contributions per party and reconstruct d
    let mut d_shares = [0u128; 3];
    for i in 0..3 {
        let handle = &handles[i];
        let own_d = a_pairs[i].0.wrapping_sub(r_pairs[i].0);
        d_shares[i] = own_d;
        let mut got_d = vec![false; 3];
        got_d[i] = true;
        let mut pending = 2; // expecting D from two other parties
        while pending > 0 {
            if let Some((from, msg)) = handle.recv() {
                let m: MpcMsg = deserialize_msg(&msg);
                if let MpcMsg::D(v) = m {
                    if !got_d[from] {
                        d_shares[from] = v;
                        got_d[from] = true;
                        pending -= 1;
                    }
                }
            } else {
                std::thread::yield_now();
            }
        }
    }

    let mut d: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
    }

    // compute bits using public d and shared r_bits
    let mut out: Vec<Share3> = Vec::with_capacity(nbits);
    let mut carry = share_bool(false);
    for i in 0..nbits {
        let d_bit = ((d >> i) & 1) != 0;
        let d_share = share_bool(d_bit);
        let r_bit = &r_bits[i];
        let s1 = bits::xor_shared(r_bit, &d_share);
        let sum = bits::xor_shared(&s1, &carry);
        out.push(sum);
        let a_and_b = bits::and(r_bit, &d_share);
        let a_and_c = bits::and(r_bit, &carry);
        let b_and_c = bits::and(&d_share, &carry);
        let t = bits::or(&a_and_c, &b_and_c);
        carry = bits::or(&a_and_b, &t);
    }
    out
}
