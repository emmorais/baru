//! Soundness tests for the **real** Binius64 backend.
//!
//! `tests/zkp_soundness.rs` documents what the placeholder backend does: it
//! accepts a proof produced with no witness at all. These tests are its mirror
//! image — the same attacks, run against [`Binius64Real`], where they must now
//! be rejected.
//!
//! Read the two files together. The pair is the actual deliverable: one records
//! that the old backend proves nothing, the other that the new one does.

use std::collections::HashMap;

use baru::interpreter::parser::parse;
use baru::zkp::binius_real::{Binius64Real, OUTPUT_NAME};
use baru::zkp::ZKPBackend;

/// `(* x y)` with `x` private and `y` public.
fn setup() -> (Binius64Real, baru::zkp::Circuit) {
    let expr = parse("(* x y)").expect("parse");
    let be = Binius64Real::new();
    let ckt = be
        .compile_circuit(&expr, &["x".to_string()])
        .expect("compile");
    (be, ckt)
}

fn public(y: i64, out: i64) -> HashMap<String, i64> {
    let mut m = HashMap::new();
    m.insert("y".to_string(), y);
    m.insert(OUTPUT_NAME.to_string(), out);
    m
}

fn witness(x: i64) -> HashMap<String, i64> {
    let mut m = HashMap::new();
    m.insert("x".to_string(), x);
    m
}

#[test]
fn an_honest_proof_verifies() {
    let (be, ckt) = setup();
    let pubs = public(7, 42);
    let proof = be.prove(&ckt, &witness(6), &pubs).expect("prove 6*7=42");
    assert!(be.verify(&ckt, &proof, &pubs).expect("verify"));
}

/// The property the placeholder lacked entirely: a prover with no witness
/// cannot produce a proof at all.
#[test]
fn proving_without_a_witness_fails() {
    let (be, ckt) = setup();
    let err = be
        .prove(&ckt, &HashMap::new(), &public(7, 42))
        .expect_err("a missing witness must not silently become zero");
    assert!(err.contains("missing private value"), "{err}");
}

/// A witness that does not satisfy the claimed statement must be rejected —
/// the circuit constrains `x * y == out`, so `5 * 7 != 42` cannot be proven.
#[test]
fn a_witness_that_does_not_satisfy_the_statement_is_rejected() {
    let (be, ckt) = setup();
    let result = be.prove(&ckt, &witness(5), &public(7, 42));
    assert!(
        result.is_err(),
        "proving 5*7 = 42 must fail; the placeholder happily accepted this"
    );
}

/// A proof of one statement must not verify as a proof of another. This is
/// what binds the proof to its public inputs.
#[test]
fn a_proof_does_not_transfer_to_a_different_statement() {
    let (be, ckt) = setup();
    let proof = be.prove(&ckt, &witness(6), &public(7, 42)).expect("prove");
    // Same proof, different claimed output.
    assert!(
        be.verify(&ckt, &proof, &public(7, 43)).is_err(),
        "a proof of 6*7=42 must not verify as 6*7=43"
    );
    // Same proof, different public input.
    assert!(
        be.verify(&ckt, &proof, &public(8, 42)).is_err(),
        "a proof for y=7 must not verify for y=8"
    );
}

/// Tampering with the transcript must be detected.
#[test]
fn a_tampered_proof_is_rejected() {
    let (be, ckt) = setup();
    let pubs = public(7, 42);
    let mut proof = be.prove(&ckt, &witness(6), &pubs).expect("prove");
    let mid = proof.proof_data.len() / 2;
    proof.proof_data[mid] ^= 0xff;
    assert!(
        be.verify(&ckt, &proof, &pubs).is_err(),
        "a corrupted transcript must not verify"
    );
}

/// Different witnesses satisfying different statements both work, and neither
/// proof verifies against the other's statement.
#[test]
fn proofs_are_bound_to_their_own_statements() {
    let (be, ckt) = setup();
    let p1 = be.prove(&ckt, &witness(6), &public(7, 42)).expect("6*7");
    let p2 = be.prove(&ckt, &witness(3), &public(5, 15)).expect("3*5");
    assert!(be.verify(&ckt, &p1, &public(7, 42)).unwrap());
    assert!(be.verify(&ckt, &p2, &public(5, 15)).unwrap());
    assert!(be.verify(&ckt, &p1, &public(5, 15)).is_err());
    assert!(be.verify(&ckt, &p2, &public(7, 42)).is_err());
}

/// Operations the backend cannot compile soundly are refused, rather than
/// mis-compiled into a circuit that proves the wrong thing.
#[test]
fn unsupported_operations_are_refused_not_mis_compiled() {
    let be = Binius64Real::new();
    for src in ["(< x y)", "(/ x y)", "(if x y 1)"] {
        let expr = parse(src).expect("parse");
        assert!(
            be.compile_circuit(&expr, &["x".to_string()]).is_err(),
            "{src} must be refused until its circuitry exists"
        );
    }
}

/// Addition and subtraction compile and prove correctly, so the supported
/// fragment is genuinely usable rather than multiplication-only.
#[test]
fn the_supported_arithmetic_fragment_works() {
    let be = Binius64Real::new();
    let expr = parse("(+ (* x x) (- y 3))").expect("parse");
    let ckt = be
        .compile_circuit(&expr, &["x".to_string()])
        .expect("compile");
    // x = 5, y = 10  ->  25 + 7 = 32
    let mut pubs = HashMap::new();
    pubs.insert("y".to_string(), 10i64);
    pubs.insert(OUTPUT_NAME.to_string(), 32i64);
    let proof = be.prove(&ckt, &witness(5), &pubs).expect("prove");
    assert!(be.verify(&ckt, &proof, &pubs).expect("verify"));

    // And the wrong output cannot be proven.
    let mut wrong = pubs.clone();
    wrong.insert(OUTPUT_NAME.to_string(), 33i64);
    assert!(be.prove(&ckt, &witness(5), &wrong).is_err());
}
