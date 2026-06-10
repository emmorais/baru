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
