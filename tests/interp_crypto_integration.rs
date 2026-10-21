use baru::crypto::types::Fr;
use baru::interpreter::aby3_simulator::Aby3Simulator;
use baru::interpreter::backend::Backend;

#[test]
fn test_backend_share_add_mul_roundtrip() {
    let mut be = Aby3Simulator::new();
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
    let mut be = Aby3Simulator::new();
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
    let mut be = Aby3Simulator::new();
    let local: Fr = 5u128;
    let shared_val: Fr = 11u128;
    let s_local = be.share_secret(local, 64, 0);
    let s_shared = be.share_secret(shared_val, 64, 0);
    let sum = be.add(&s_local, &s_shared);
    assert_eq!(be.reveal(&sum, None), local.wrapping_add(shared_val));
}

/// A secret-conditioned `if` must return its branch, not the condition.
///
/// Regression for a silent truncation fixed 2026-10-21. `IfResultNormalize` took
/// its share width from the *condition*, and a comparison's share is one bit
/// wide, so a numeric branch result was reduced modulo 2: `(if (> x 0) 111 222)`
/// returned 1 and 0. It survived because no benchmark program emits an `if` —
/// ReLU is `(* p (> p 0))`, argmax is comparison bits times a product tree — and
/// the only secret-conditioned `if` coverage in the tree (`fhe_differential.rs`)
/// exercises the FHE path, not the share path.
///
/// Both polarities and both branch *kinds* are asserted: constant arms isolate
/// selection, computed arms (which arrive already shared and bypass the
/// normalize entirely) would pass even with the bug present.
#[test]
fn shared_if_returns_the_branch_not_the_condition() {
    let cases = [
        ("(let ((x (share 7 32))) (reconstruct (if (> x 0) 111 222)))", 111),
        ("(let ((x (share -7 32))) (reconstruct (if (> x 0) 111 222)))", 222),
        ("(let ((x (share 7 32))) (reconstruct (if (> x 0) (* x 2) 0)))", 14),
        ("(let ((x (share 5 32))) (reconstruct (if (> x 0) 1000 2000)))", 1000),
        ("(let ((x (share -5 32))) (reconstruct (if (> x 0) 1000 2000)))", 2000),
    ];
    for (src, want) in cases {
        let got = baru::interpreter::eval::eval_shared(src);
        assert_eq!(got, want, "{src}");
    }
}
