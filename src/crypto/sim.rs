use crate::crypto::types::{Share3, Fr};
use crate::crypto::arith::{gen_beaver_triple, share_secret, scalar_mul, add_shares};
use crate::crypto::bits::{share_bool};
use crate::crypto::bits;
use rand::Rng;

// Simple helper: split a Share3 into per-party pairs (party i holds s_i, s_{i+1}).
fn split_to_party_pairs(s: &Share3) -> Vec<(Fr, Fr)> {
    vec![(s.0, s.1), (s.1, s.2), (s.2, s.0)]
}

/// Simulate per-party message passing for ABY3 multiplication protocol.
/// This is an explicit, step-by-step version of `mul_ab3_simulate` that
/// reflects the per-party broadcasts and resharing.
pub fn mul_ab3_protocol(a: &Share3, b: &Share3) -> Share3 {
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
        d_shares[i] = a_pairs[i].0.wrapping_sub(x_pairs[i].0);
        e_shares[i] = b_pairs[i].0.wrapping_sub(y_pairs[i].0);
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

/// Simulate ABY3-style arithmetic-to-bit decomposition using masked opening.
/// This explicitly models per-party broadcasts of d = x - r and computes bit
/// shares using r_bits shares.
pub fn a2b_protocol(a: &Share3, nbits: usize) -> Vec<Share3> {
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
        d_shares[i] = a_pairs[i].0.wrapping_sub(r_pairs[i].0);
    }
    let mut d: Fr = 0;
    for i in 0..3 { d = d.wrapping_add(d_shares[i]); }

    // r_bits already prepared above

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
pub fn lt_protocol(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let diff = crate::crypto::arith::sub_shares(a, b);
    let bits = a2b_protocol(&diff, nbits);
    bits[nbits - 1].clone()
}

pub fn le_protocol(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let lt = lt_protocol(a, b, nbits);
    let eq = crate::crypto::bits::eq_ab3(a, b, nbits);
    crate::crypto::bits::or(&lt, &eq)
}

pub fn eq_protocol(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let diff = crate::crypto::arith::sub_shares(a, b);
    let bits = a2b_protocol(&diff, nbits);
    let mut acc = bits[0].clone();
    for i in 1..bits.len() { acc = crate::crypto::bits::or(&acc, &bits[i]); }
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
    for i in 0..3 { d = d.wrapping_add(d_shares[i]); }

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
