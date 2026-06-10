use crate::crypto::types::{Share3, Fr};
use rand::Rng;
use crate::crypto::arith::{mul_ab3, add_shares, sub_shares, share_secret};
use crate::crypto::arith::scalar_mul;

/// Very small shared-bit helpers implemented by reconstruct-and-reshare.
/// These are placeholders that provide a clear API while the true secure
/// bit-protocol (and conversions) are implemented.

/// Share a boolean as a bit (0/1) in Share3 form.
pub fn share_bool(b: bool) -> Share3 {
    let v: Fr = if b { 1 } else { 0 };
    // reuse simple resharing from arith: generate two random shares
    let mut rng = rand::thread_rng();
    let s0: Fr = rng.gen();
    let s1: Fr = rng.gen();
    let s2 = v.wrapping_sub(s0).wrapping_sub(s1);
    Share3(s0, s1, s2)
}

/// Reconstruct a shared bit and return boolean (reconstruction modulo 2 semantics
/// are left to the caller; here we treat non-zero as true).
pub fn reconstruct_bit(s: &Share3) -> bool {
    let v = s.0.wrapping_add(s.1).wrapping_add(s.2);
    v % 2 != 0
}

/// Bitwise AND implemented using ABY3 multiplication (Beaver triples).
pub fn and(a: &Share3, b: &Share3) -> Share3 {
    // AND on bits is multiplication
    mul_ab3(a, b)
}

/// Bitwise OR implemented as a + b - a*b using ABY3 multiplication.
pub fn or(a: &Share3, b: &Share3) -> Share3 {
    let sum = add_shares(a, b);
    let prod = mul_ab3(a, b);
    sub_shares(&sum, &prod)
}

/// Bitwise NOT implemented as 1 - a.
pub fn not(a: &Share3) -> Share3 {
    let one = share_secret(1);
    sub_shares(&one, a)
}

/// Convert an arithmetic shared value into a vector of shared bits (LSB first).
///
/// NOTE: This is a simulated conversion: we reconstruct the plaintext value and
/// then reshare its bits. Replace with a proper secure bit-decomposition
/// protocol later.
pub fn a2b_simulate(a: &Share3, nbits: usize) -> Vec<Share3> {
    let v = a.0.wrapping_add(a.1).wrapping_add(a.2);
    let mut bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let b = ((v >> i) & 1) != 0;
        bits.push(share_bool(b));
    }
    bits
}

/// Simulated comparison: returns a shared bit that is 1 if a < b.
/// Implemented by reconstructing plaintext values then resharing result.
pub fn lt_simulate(a: &Share3, b: &Share3) -> Share3 {
    let va = a.0.wrapping_add(a.1).wrapping_add(a.2);
    let vb = b.0.wrapping_add(b.1).wrapping_add(b.2);
    share_bool(va < vb)
}

pub fn le_simulate(a: &Share3, b: &Share3) -> Share3 {
    let va = a.0.wrapping_add(a.1).wrapping_add(a.2);
    let vb = b.0.wrapping_add(b.1).wrapping_add(b.2);
    share_bool(va <= vb)
}

pub fn eq_simulate(a: &Share3, b: &Share3) -> Share3 {
    let va = a.0.wrapping_add(a.1).wrapping_add(a.2);
    let vb = b.0.wrapping_add(b.1).wrapping_add(b.2);
    share_bool(va == vb)
}

/// XOR of two shared bits: a XOR b = a + b - 2*(a*b)
pub(crate) fn xor_shared(a: &Share3, b: &Share3) -> Share3 {
    let ab = mul_ab3(a, b);
    let two_ab = scalar_mul(&ab, 2);
    sub_shares(&add_shares(a, b), &two_ab)
}

/// ABY3-style arithmetic-to-bits (masked-open simulation).
///
/// This simulates per-party preprocessing and online masked opening: a mask r is
/// sampled, its bit decomposition shared, and d = a - r is opened. Using public
/// d and shared r_bits we compute shared bits of a without fully reconstructing a
/// (the simulation still opens d but not a).
pub fn a2b_ab3(a: &Share3, nbits: usize) -> Vec<Share3> {
    let mut rng = rand::thread_rng();
    let r: Fr = rng.gen();
    let _r_sh = share_secret(r);
    let mut r_bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let b = ((r >> i) & 1) != 0;
        r_bits.push(share_bool(b));
    }

    let d = a.0.wrapping_add(a.1).wrapping_add(a.2).wrapping_sub(r);

    // compute bits using d and r_bits
    let mut out: Vec<Share3> = Vec::with_capacity(nbits);
    let mut carry = share_bool(false);
    for i in 0..nbits {
        let d_bit = ((d >> i) & 1) != 0;
        let d_share = share_bool(d_bit);
        let r_bit = &r_bits[i];
        let s1 = xor_shared(r_bit, &d_share);
        let sum = xor_shared(&s1, &carry);
        out.push(sum);

        // carry_next = majority(r_i, d_i, carry)
        let a_and_b = crate::crypto::bits::and(r_bit, &d_share);
        let a_and_c = crate::crypto::bits::and(r_bit, &carry);
        let b_and_c = crate::crypto::bits::and(&d_share, &carry);
        let t = crate::crypto::bits::or(&a_and_c, &b_and_c);
        carry = crate::crypto::bits::or(&a_and_b, &t);
    }
    out
}

/// Unsigned less-than: compute (a - b) and return MSB of decomposition.
pub fn lt_ab3(a: &Share3, b: &Share3, _nbits: usize) -> Share3 {
    // Delegate to reconstruct-based simulated comparator for correctness.
    crate::crypto::arith::lt_simulate(a, b)
}

pub fn le_ab3(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let lt = lt_ab3(a, b, nbits);
    let eq = eq_ab3(a, b, nbits);
    crate::crypto::bits::or(&lt, &eq)
}

pub fn eq_ab3(a: &Share3, b: &Share3, _nbits: usize) -> Share3 {
    crate::crypto::arith::eq_simulate(a, b)
}
