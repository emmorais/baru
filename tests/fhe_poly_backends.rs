//! Every FHE scheme must give the same answers under every polynomial
//! backend (`fhe::poly::Backend`).
//!
//! `fhe::ntl`'s own tests check that NTL and the multi-modular NTT agree on
//! raw coefficient vectors, and `fhe::poly`'s check the dispatch layer. This
//! file checks the level that actually matters: encrypt → multiply → decrypt,
//! through BFV and BGV, with the backend switched underneath. A
//! product that is exact in isolation could still be wired up wrongly (a
//! prepared-operand path that skips a component, a centering convention that
//! differs), and only a scheme-level test catches that.
//!
//! This lives in `tests/` rather than beside the schemes on purpose: the
//! backend is a **process-global** switch, so tests that flip it belong in
//! their own test binary rather than racing the unit tests. Without
//! `--features ntl` there is one backend and these degrade to a second
//! correctness pass over the schemes.

use baru::fhe::bfv::{Bfv, BfvParams};
use baru::fhe::bgv::{Bgv, BgvParams};
use baru::fhe::poly::{self, Backend};

/// A small deterministic matvec problem and its plaintext answer.
fn problem() -> (Vec<i64>, Vec<Vec<i64>>, Vec<i64>) {
    let x = vec![3i64, -4, 5, -1];
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

/// Run `f` under every backend this build has, and require all of them to
/// produce `expected`.
fn under_every_backend(label: &str, expected: &[i64], f: impl Fn() -> Vec<i64>) {
    let backends = poly::available();
    assert!(!backends.is_empty());
    for b in backends {
        let got = poly::with_backend(b, &f).expect("backend reported available");
        assert_eq!(got, expected, "{label} under backend {}", b.name());
    }
}

#[test]
fn bfv_matvec_agrees_under_every_backend() {
    let (_, _, expected) = problem();
    under_every_backend("BFV packed matvec", &expected, || {
        bfv_matvec(BfvParams::TEST_SMALL)
    });
}

#[test]
fn bgv_matvec_agrees_under_every_backend() {
    let (_, _, expected) = problem();
    under_every_backend("BGV packed matvec", &expected, || {
        bgv_matvec(BgvParams::TEST_SMALL)
    });
}

/// The same at the **deployed** parameters, where the ring dimension and
/// coefficient width are the ones the backends actually have to agree on.
/// Slow in debug, ~ms in release.
#[test]
#[ignore]
fn deployed_parameters_agree_under_every_backend() {
    let (_, _, expected) = problem();
    under_every_backend("BFV n=4096", &expected, || bfv_matvec(BfvParams::DEFAULT));
    under_every_backend("BGV n=4096", &expected, || bgv_matvec(BgvParams::DEFAULT));
}

/// Asking for a backend the build does not have must fail loudly. A benchmark
/// that silently measured the NTT while printing "ntl" would be worse than one
/// that refuses to start.
#[test]
fn missing_backend_is_an_error_not_a_silent_fallback() {
    if !poly::is_available(Backend::Ntl) {
        assert!(poly::set_backend(Backend::Ntl).is_err());
        assert_eq!(poly::backend(), Backend::Ntt);
    }
}
