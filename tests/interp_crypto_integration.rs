use baru::crypto::types::Fr;
use baru::interpreter::backend::{Aby3Backend, Backend};

#[test]
fn test_backend_share_add_mul_roundtrip() {
    let mut be = Aby3Backend::new();
    let a: Fr = 42u128;
    let b: Fr = 7u128;
    let sa = be.share_secret(a, 128, 0);
    let sb = be.share_secret(b, 128, 0);
    let sum = be.add(&sa, &sb);
    assert_eq!(be.reveal(&sum, None), a.wrapping_add(b));
    let prod = be.mul(&sa, &sb);
    assert_eq!(be.reveal(&prod, None), a.wrapping_mul(b));
}

#[test]
fn test_backend_a2b_and_b2a() {
    let mut be = Aby3Backend::new();
    let a: Fr = 37u128; // 0b00100101
    let sa = be.share_secret(a, 8, 0);
    let bits = be.a2b(&sa, 8);
    assert_eq!(bits.len(), 8);
    // reconstruct bitwise
    let mut rec: u128 = 0;
    for (i, b) in bits.iter().enumerate() {
        let bv = be.reveal(b, None) & 1u128;
        if bv == 1 {
            rec |= 1u128 << i;
        }
    }
    assert_eq!(rec & ((1u128 << 8) - 1), a & ((1u128 << 8) - 1));

    // test b2a roundtrip
    let sa2 = be.b2a(&bits);
    assert_eq!(
        be.reveal(&sa2, None) & ((1u128 << 8) - 1),
        a & ((1u128 << 8) - 1)
    );
}

#[test]
fn test_mixed_local_and_shared() {
    let mut be = Aby3Backend::new();
    let local: Fr = 5u128;
    let shared_val: Fr = 11u128;
    let s_local = be.share_secret(local, 64, 0);
    let s_shared = be.share_secret(shared_val, 64, 0);
    let sum = be.add(&s_local, &s_shared);
    assert_eq!(be.reveal(&sum, None), local.wrapping_add(shared_val));
}
