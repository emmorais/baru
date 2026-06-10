use baru::crypto::{arith, bits, party};
use baru::crypto::types::Fr;

#[test]
fn test_share_add_mul_insecure() {
    let a: Fr = 42u128;
    let b: Fr = 7u128;
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);
    let sum = arith::add_shares(&sa, &sb);
    assert_eq!(arith::reconstruct(&sum), a.wrapping_add(b));
    let prod = arith::mul_insecure(&sa, &sb);
    assert_eq!(arith::reconstruct(&prod), a.wrapping_mul(b));
}

#[test]
fn test_mul_ab3_simulate() {
    let a: Fr = 5u128;
    let b: Fr = 11u128;
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);
    let prod = arith::mul_ab3(&sa, &sb);
    assert_eq!(arith::reconstruct(&prod), a.wrapping_mul(b));
    let prod2 = party::mul_ab3_simulate(&sa, &sb);
    assert_eq!(arith::reconstruct(&prod2), a.wrapping_mul(b));
}

#[test]
fn test_bits_ops() {
    let t = bits::share_bool(true);
    let f = bits::share_bool(false);
    let andv = bits::and(&t, &f);
    assert_eq!(bits::reconstruct_bit(&andv), true & false);
    let orv = bits::or(&t, &f);
    assert_eq!(bits::reconstruct_bit(&orv), true | false);
    let notv = bits::not(&t);
    assert_eq!(bits::reconstruct_bit(&notv), !true);
}

#[test]
fn test_mul_with_scheduler_honest_and_corrupt() {
    use baru::crypto::scheduler;
    let a: Fr = 13u128;
    let b: Fr = 17u128;
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);
    let (out, rounds) = scheduler::mul_with_scheduler(&sa, &sb, None);
    assert_eq!(arith::reconstruct(&out), a.wrapping_mul(b));
    // should have 6 broadcast rounds (3 for d, 3 for e)
    assert_eq!(rounds, 6);

    // corrupted party: output should differ
    let (out_corrupt, _rounds2) = scheduler::mul_with_scheduler(&sa, &sb, Some(1));
    assert_ne!(arith::reconstruct(&out_corrupt), a.wrapping_mul(b));
}

#[test]
fn test_a2b_and_comparisons() {
    // test decomposition and comparison helpers
    let a: Fr = 37u128; // 0b100101
    let b: Fr = 45u128; // 0b101101
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);

    // a2b decomposition
    let bits = bits::a2b_simulate(&sa, 8);
    let mut recovered: u128 = 0;
    for (i, bit_sh) in bits.iter().enumerate() {
        if bits::reconstruct_bit(bit_sh) {
            recovered |= 1u128 << i;
        }
    }
    assert_eq!(recovered, a);

    // comparisons
    let lt_sh = bits::lt_ab3(&sa, &sb, 8);
    assert_eq!(bits::reconstruct_bit(&lt_sh), a < b);
    let eq_sh = bits::eq_ab3(&sa, &sb, 8);
    assert_eq!(bits::reconstruct_bit(&eq_sh), a == b);
}
