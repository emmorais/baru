use baru::interpreter::backend::{SimBackend, Backend};
use baru::crypto::types::Fr;

#[test]
fn test_backend_share_add_mul_roundtrip() {
    let mut be = SimBackend::new();
    let a: Fr = 42u128;
    let b: Fr = 7u128;
    let sa = be.share_secret(a, 128);
    let sb = be.share_secret(b, 128);
    let sum = be.add(&sa, &sb);
    assert_eq!(be.reconstruct(&sum), a.wrapping_add(b));
    let prod = be.mul(&sa, &sb);
    assert_eq!(be.reconstruct(&prod), a.wrapping_mul(b));
}

#[test]
fn test_backend_a2b_and_b2a() {
    let mut be = SimBackend::new();
    let a: Fr = 37u128; // 0b00100101
    let sa = be.share_secret(a, 8);
    let bits = be.a2b(&sa, 8);
    assert_eq!(bits.len(), 8);
    // reconstruct bitwise
    let mut rec: u128 = 0;
    for (i, b) in bits.iter().enumerate() {
        let bv = be.reconstruct(b) & 1u128;
        if bv == 1 { rec |= 1u128 << i; }
    }
    assert_eq!(rec & ((1u128 << 8) - 1), a & ((1u128 << 8) - 1));

    // test b2a roundtrip
    let sa2 = be.b2a(&bits);
    assert_eq!(be.reconstruct(&sa2) & ((1u128<<8)-1), a & ((1u128<<8)-1));
}

#[test]
fn test_mixed_local_and_shared() {
    let mut be = SimBackend::new();
    let local: Fr = 5u128;
    let shared_val: Fr = 11u128;
    let s_local = be.share_secret(local, 64);
    let s_shared = be.share_secret(shared_val, 64);
    let sum = be.add(&s_local, &s_shared);
    assert_eq!(be.reconstruct(&sum), local.wrapping_add(shared_val));
}
