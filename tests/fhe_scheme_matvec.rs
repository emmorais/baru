//! Packed matvec through BFV and BGV, end to end: encrypt → multiply → decrypt.
//!
//! This is the level that catches wiring bugs the polynomial layer's own tests
//! cannot. `fhe::ntt` checks the transform against a schoolbook reference at
//! full ring dimension, which proves the *product* is exact — but a product that
//! is exact in isolation can still be plumbed in wrongly: a prepared-operand path
//! that skips a component, a centering convention that differs between encrypt
//! and decrypt, an output position computed for the wrong row. Only a
//! scheme-level test sees that.
//!
//! It was `tests/fhe_poly_backends.rs` until 2026-08-03, when the NTL polynomial
//! backend was removed and with it the runtime backend switch this used to sweep.
//! The sweep is gone; the exactness check it wrapped is the part worth keeping,
//! so it lives on here without the machinery.

use baru::fhe::bfv::{Bfv, BfvParams};
use baru::fhe::bgv::{Bgv, BgvParams};

/// A small signed matvec with negatives and a zero, so a sign or centering bug
/// cannot pass by accident.
fn problem() -> (Vec<i64>, Vec<Vec<i64>>, Vec<i64>) {
    let x = vec![1i64, -2, 3, 4];
    let w = vec![
        vec![2i64, 3, -1, 4],
        vec![-5i64, 1, 0, 2],
        vec![7i64, -7, 2, 2],
    ];
    let expected = w
        .iter()
        .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
        .collect();
    (x, w, expected)
}

fn bfv_matvec(params: BfvParams) -> Vec<i64> {
    let s = Bfv::new(params);
    let (ck, pk) = s.keygen();
    let (x, w, _) = problem();
    let cx = s.encrypt_poly(&pk, &s.pack_input(&x));
    let cw = s.encrypt_poly(&pk, &s.pack_weights(&w));
    let out = s.decrypt_poly(&ck, &s.mul(&cx, &cw));
    (0..w.len())
        .map(|r| s.to_signed(out[s.matvec_output_position(x.len(), r)]))
        .collect()
}

fn bgv_matvec(params: BgvParams) -> Vec<i64> {
    let s = Bgv::new(params);
    let (ck, pk) = s.keygen();
    let (x, w, _) = problem();
    let cx = s.encrypt_poly(&pk, &s.pack_input(&x));
    let cw = s.encrypt_poly(&pk, &s.pack_weights(&w));
    let out = s.decrypt_poly(&ck, &s.mul(&cx, &cw));
    (0..w.len())
        .map(|r| s.to_signed(out[s.matvec_output_position(x.len(), r)]))
        .collect()
}

#[test]
fn bfv_packed_matvec_is_exact() {
    let (_, _, expected) = problem();
    assert_eq!(bfv_matvec(BfvParams::TEST_SMALL), expected);
}

#[test]
fn bgv_packed_matvec_is_exact() {
    let (_, _, expected) = problem();
    assert_eq!(bgv_matvec(BgvParams::TEST_SMALL), expected);
}

/// The same at the **deployed** parameters — the ring dimension and coefficient
/// width the schemes actually run at. Slow in debug, ~ms in release.
#[test]
#[ignore]
fn deployed_parameters_are_exact() {
    let (_, _, expected) = problem();
    assert_eq!(bfv_matvec(BfvParams::DEFAULT), expected, "BFV n=4096");
    assert_eq!(bgv_matvec(BgvParams::DEFAULT), expected, "BGV n=4096");
}
