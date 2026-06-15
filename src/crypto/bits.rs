use crate::crypto::arith::scalar_mul;
use crate::crypto::arith::{add_shares, mul_ab3, share_secret, sub_shares};
use crate::crypto::types::{Fr, Share3};
use rand::Rng;

/// Very small shared-bit helpers implemented by reconstruct-and-reshare.
/// These are placeholders that provide a clear API while the true secure
/// bit-protocol (and conversions) are implemented.
///////////////////////////////////////////////////////////////////////////
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
    !v.is_multiple_of(2)
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
pub fn a2b_ab3(a: &Share3, nbits: usize) -> (Vec<Share3>, Share3) {
    let mut rng = rand::thread_rng();
    let r: Fr = rng.gen::<u128>() & ((1u128 << nbits) - 1);
    // construct r_sh from per-bit shares so r_sh and r_bits are consistent
    let mut r_bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let b = ((r >> i) & 1) != 0;
        r_bits.push(share_bool(b));
    }
    // assemble r_sh = sum_i r_bits[i] * (1<<i)
    let mut r_sh = share_secret(0);
    for (i, r_bit) in r_bits.iter().enumerate() {
        let term = scalar_mul(r_bit, 1u128 << i);
        r_sh = add_shares(&r_sh, &term);
    }

    // compute d = a - r_sh (use the assembled r_sh so mask and bits match)
    let a_plain = a.0.wrapping_add(a.1).wrapping_add(a.2);
    let r_plain = r_sh.0.wrapping_add(r_sh.1).wrapping_add(r_sh.2);
    let d = a_plain.wrapping_sub(r_plain);

    // compute bits using d and r_bits
    let mut out: Vec<Share3> = Vec::with_capacity(nbits);
    let mut carry = share_bool(false);
    for (i, r_bit) in r_bits.iter().enumerate() {
        let d_bit = ((d >> i) & 1) != 0;
        let d_share = share_bool(d_bit);
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
    (out, carry)
}

/// Deterministic debug helper: run the a2b_ab3 protocol using a seeded RNG
/// and return a diagnostic string with r, d, bits and carry traces.
pub fn a2b_ab3_debug_seed(a: &Share3, nbits: usize, seed: u64) -> String {
    use crate::crypto::arith::reconstruct;
    use crate::crypto::bits::share_bool;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let r: Fr = rng.gen::<u128>() & ((1u128 << nbits) - 1);
    let mut r_bits: Vec<Share3> = Vec::with_capacity(nbits);
    for i in 0..nbits {
        let b = ((r >> i) & 1) != 0;
        r_bits.push(share_bool(b));
    }
    let mut r_sh = share_secret(0);
    for (i, r_bit) in r_bits.iter().enumerate() {
        let term = scalar_mul(r_bit, 1u128 << i);
        r_sh = add_shares(&r_sh, &term);
    }
    let a_plain = a.0.wrapping_add(a.1).wrapping_add(a.2);
    let r_plain = r_sh.0.wrapping_add(r_sh.1).wrapping_add(r_sh.2);
    let d = a_plain.wrapping_sub(r_plain);

    // compute bits and carry using same logic as a2b_ab3
    let mut out: Vec<Share3> = Vec::with_capacity(nbits);
    let mut carry = share_bool(false);
    let mut sum_plain: Vec<bool> = Vec::with_capacity(nbits);
    let mut carry_plain: Vec<bool> = Vec::with_capacity(nbits);
    for (i, r_bit) in r_bits.iter().enumerate() {
        let d_bit = ((d >> i) & 1) != 0;
        let d_share = share_bool(d_bit);
        let s1 = xor_shared(r_bit, &d_share);
        let sum = xor_shared(&s1, &carry);
        out.push(sum);
        sum_plain.push(reconstruct(&out[i]).is_multiple_of(2));
        let a_and_b = crate::crypto::bits::and(r_bit, &d_share);
        let a_and_c = crate::crypto::bits::and(r_bit, &carry);
        let b_and_c = crate::crypto::bits::and(&d_share, &carry);
        let t = crate::crypto::bits::or(&a_and_c, &b_and_c);
        carry = crate::crypto::bits::or(&a_and_b, &t);
        carry_plain.push(reconstruct(&carry).is_multiple_of(2));
    }

    let mut s = String::new();
    use std::fmt::Write;
    writeln!(s, "seed = {}", seed).ok();
    writeln!(s, "r = {}", r).ok();
    writeln!(s, "d = {}", d).ok();
    writeln!(
        s,
        "r_bits = {:?}",
        (0..nbits).map(|i| ((r >> i) & 1) != 0).collect::<Vec<_>>()
    )
    .ok();
    writeln!(
        s,
        "d_bits = {:?}",
        (0..nbits).map(|i| ((d >> i) & 1) != 0).collect::<Vec<_>>()
    )
    .ok();
    writeln!(s, "sum_bits = {:?}", sum_plain).ok();
    writeln!(s, "carry_bits = {:?}", carry_plain).ok();
    s
}

/// Compute borrow-out of a - b given LSB-first shared bit vectors (nbits).
/// borrow_0 is 0. Returns shared borrow_out (1 if a < b).
fn borrow_from_bits(a_bits: &[Share3], b_bits: &[Share3]) -> Share3 {
    // initial borrow = 0
    let mut borrow = share_bool(false);
    for i in 0..a_bits.len() {
        let ai = &a_bits[i];
        let bi = &b_bits[i];
        let not_ai = not(ai);
        // t1 = not_ai AND bi
        let t1 = and(&not_ai, bi);
        // t2 = bi AND borrow
        let t2 = and(bi, &borrow);
        // t3 = not_ai AND borrow
        let t3 = and(&not_ai, &borrow);
        let t23 = or(&t2, &t3);
        borrow = or(&t1, &t23);
    }
    borrow
}

/// Unsigned less-than: compute (a - b) and return MSB of decomposition.
pub fn lt_ab3(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    // Decompose both a and b into shared bits using ABY3 a2b protocol
    let (a_bits, _a_carry) = a2b_ab3(a, nbits);
    let (b_bits, _b_carry) = a2b_ab3(b, nbits);
    // compute borrow_out from a_bits - b_bits; borrow_out==1 iff a < b
    borrow_from_bits(&a_bits, &b_bits)
}

pub fn le_ab3(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let lt = lt_ab3(a, b, nbits);
    let eq = eq_ab3(a, b, nbits);
    crate::crypto::bits::or(&lt, &eq)
}

/// Signed less-than for two's-complement values with `nbits` width.
///
/// Let sa/sb be sign bits. Then:
/// - if signs differ: a < b iff sa=1 and sb=0
/// - if signs equal: compare as unsigned (same ordering under equal sign)
pub fn lt_ab3_signed(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let (a_bits, _a_carry) = a2b_ab3(a, nbits);
    let (b_bits, _b_carry) = a2b_ab3(b, nbits);

    let sign_a = &a_bits[nbits - 1];
    let sign_b = &b_bits[nbits - 1];

    let unsigned_lt = borrow_from_bits(&a_bits, &b_bits);
    let sign_diff = xor_shared(sign_a, sign_b);
    let signs_equal = not(&sign_diff);

    let a_neg_b_nonneg = and(sign_a, &not(sign_b));
    let same_sign_and_lt = and(&signs_equal, &unsigned_lt);
    or(&a_neg_b_nonneg, &same_sign_and_lt)
}

pub fn le_ab3_signed(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let lt = lt_ab3_signed(a, b, nbits);
    let eq = eq_ab3(a, b, nbits);
    or(&lt, &eq)
}

pub fn eq_ab3(a: &Share3, b: &Share3, nbits: usize) -> Share3 {
    let diff = sub_shares(a, b);
    let (bits, _carry_out) = a2b_ab3(&diff, nbits);
    // equality is NOT(OR(bits))
    let mut acc = bits[0].clone();
    for bit in bits.iter().skip(1) {
        acc = crate::crypto::bits::or(&acc, bit);
    }
    crate::crypto::bits::not(&acc)
}
