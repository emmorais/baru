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
///
/// `<`, `if`, division and modulo were all on this list at various points and
/// now have real circuitry, covered by the tests below. What remains refused is
/// what has no circuit at all: lambdas and list operations.
#[test]
fn unsupported_operations_are_refused_not_mis_compiled() {
    let be = Binius64Real::new();
    for src in ["(lambda (a) a)", "(car (list 1 2))"] {
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

// ─── The extended fragment: comparisons, `if`, `let` ─────────────────────────

/// Prove `expr` on the given inputs, with every named input private, and return
/// whether the honest proof verified plus the value that was claimed.
fn prove_and_verify(expr: &str, private: &[(&str, i64)]) -> (bool, i64) {
    let be = Binius64Real::new();
    let parsed = parse(expr).unwrap_or_else(|e| panic!("parse {expr}: {e}"));
    let names: Vec<String> = private.iter().map(|(k, _)| k.to_string()).collect();
    let ckt = be
        .compile_circuit(&parsed, &names)
        .unwrap_or_else(|e| panic!("compile {expr}: {e}"));
    let priv_map: HashMap<String, i64> = private.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    let mut pubs = HashMap::new();
    let claimed = be
        .evaluate(&ckt, &priv_map, &pubs)
        .unwrap_or_else(|e| panic!("evaluate {expr}: {e}"));
    pubs.insert(OUTPUT_NAME.to_string(), claimed);
    let proof = be
        .prove(&ckt, &priv_map, &pubs)
        .unwrap_or_else(|e| panic!("prove {expr}: {e}"));
    (be.verify(&ckt, &proof, &pubs).unwrap_or(false), claimed)
}

/// Signed comparison is the detail most likely to be quietly wrong: the
/// underlying gate is *unsigned*, so without flipping the sign bit every
/// negative value compares as larger than every positive one.
#[test]
fn signed_comparisons_are_correct_including_negatives() {
    for (a, b) in [
        (3i64, 9i64),
        (9, 3),
        (-1, 1),
        (1, -1),
        (-5, -2),
        (0, 0),
        (-7, -7),
    ] {
        let (ok, lt) = prove_and_verify("(< x y)", &[("x", a), ("y", b)]);
        assert!(ok, "verify (< {a} {b})");
        assert_eq!(lt, (a < b) as i64, "signed {a} < {b}");

        let (ok, le) = prove_and_verify("(<= x y)", &[("x", a), ("y", b)]);
        assert!(ok, "verify (<= {a} {b})");
        assert_eq!(le, (a <= b) as i64, "signed {a} <= {b}");

        let (ok, gt) = prove_and_verify("(> x y)", &[("x", a), ("y", b)]);
        assert!(ok, "verify (> {a} {b})");
        assert_eq!(gt, (a > b) as i64, "signed {a} > {b}");

        let (ok, eq) = prove_and_verify("(== x y)", &[("x", a), ("y", b)]);
        assert!(ok, "verify (== {a} {b})");
        assert_eq!(eq, (a == b) as i64, "{a} == {b}");
    }
}

/// **ReLU** — the operation the whole extension exists for, and the one the
/// benchmark suite could not compile before it.
#[test]
fn relu_compiles_proves_and_verifies() {
    for v in [-100i64, -1, 0, 1, 42] {
        let (ok, out) = prove_and_verify("(if (> x 0) x 0)", &[("x", v)]);
        assert!(ok, "verify relu({v})");
        assert_eq!(out, v.max(0), "relu({v})");
    }
}

/// The multiplicative form of ReLU that Baru's own MPC path uses, so the two
/// backends agree on the same expression.
#[test]
fn multiplicative_relu_agrees() {
    for v in [-9i64, -1, 0, 5, 77] {
        let (ok, out) = prove_and_verify("(* x (> x 0))", &[("x", v)]);
        assert!(ok, "verify mul-relu({v})");
        assert_eq!(out, v.max(0), "mul-relu({v})");
    }
}

/// `let` with sequential bindings, and a shadowed name that must not leak.
#[test]
fn let_bindings_scope_correctly() {
    let (ok, out) = prove_and_verify("(let ((a (* x 2)) (b (+ a 1))) (* a b))", &[("x", 5)]);
    assert!(ok);
    assert_eq!(out, 10 * 11);

    // Inner shadowing must be restored afterwards.
    let (ok, out) = prove_and_verify("(+ (let ((x 100)) x) x)", &[("x", 7)]);
    assert!(ok);
    assert_eq!(out, 107);
}

/// A single neuron with a ReLU — the shape private inference is built from.
#[test]
fn a_relu_neuron_proves() {
    let src = "(let ((z (+ (* w x) bias))) (if (> z 0) z 0))";
    for (w, x, bias) in [(-1i64, 3i64, 0i64), (2, 3, 1), (1, 1, -5)] {
        let (ok, out) = prove_and_verify(src, &[("w", w), ("x", x), ("bias", bias)]);
        let expect = (w * x + bias).max(0);
        assert!(ok, "verify neuron({w},{x},{bias})");
        assert_eq!(out, expect, "neuron({w},{x},{bias})");
    }
}

/// Soundness must survive the extension: a false claim about a ReLU is still
/// unprovable.
#[test]
fn a_false_relu_claim_is_still_unprovable() {
    let be = Binius64Real::new();
    let parsed = parse("(if (> x 0) x 0)").expect("parse");
    let ckt = be
        .compile_circuit(&parsed, &["x".to_string()])
        .expect("compile");
    let mut priv_map = HashMap::new();
    priv_map.insert("x".to_string(), -5i64);
    // relu(-5) = 0, but claim 5.
    let mut pubs = HashMap::new();
    pubs.insert(OUTPUT_NAME.to_string(), 5i64);
    assert!(
        be.prove(&ckt, &priv_map, &pubs).is_err(),
        "claiming relu(-5) = 5 must fail"
    );
}

/// Fixed-point rescaling: Q16.16 multiplication needs a `>> 16` that preserves
/// sign. A *logical* shift here would turn every negative activation into a
/// huge positive one, so this sweeps negatives deliberately.
#[test]
fn arithmetic_shift_right_preserves_sign() {
    for v in [-65536i64, -1, -100000, 0, 65536, 123456] {
        let (ok, out) = prove_and_verify("(ash-right x 16)", &[("x", v)]);
        assert!(ok, "verify (ash-right {v} 16)");
        assert_eq!(out, v >> 16, "arithmetic shift of {v}");
    }
}

/// A Q16.16 multiply-then-rescale, which is the operation every fixed-point
/// layer in this repo performs.
#[test]
fn q16_16_multiply_rescales_correctly() {
    let scale = 65536i64;
    for (a, b) in [(1.5f64, 2.0f64), (-1.5, 2.0), (0.25, -4.0)] {
        let (av, bv) = ((a * scale as f64) as i64, (b * scale as f64) as i64);
        let (ok, out) = prove_and_verify("(ash-right (* x y) 16)", &[("x", av), ("y", bv)]);
        assert!(ok, "verify Q16.16 {a} * {b}");
        assert_eq!(out, (av * bv) >> 16, "Q16.16 {a} * {b}");
    }
}

#[test]
fn shift_left_works_and_secret_shift_amounts_are_refused() {
    let (ok, out) = prove_and_verify("(ash-left x 4)", &[("x", 3)]);
    assert!(ok);
    assert_eq!(out, 48);

    // A secret shift amount needs a multiplexer construction that does not
    // exist here, so it must be refused rather than mis-compiled.
    let be = Binius64Real::new();
    let expr = parse("(ash-right x y)").expect("parse");
    assert!(be
        .compile_circuit(&expr, &["x".to_string(), "y".to_string()])
        .is_err());
}

// ─── Division ────────────────────────────────────────────────────────────────

/// Truncating signed division and remainder, matching Rust's `/` and `%`
/// across every sign combination.
#[test]
fn division_matches_rust_semantics() {
    for (a, b) in [
        (7i64, 2i64),
        (-7, 2),
        (7, -2),
        (-7, -2),
        (100, 10),
        (0, 5),
        (1, 3),
        (-1, 3),
        (i64::MAX, 3),
    ] {
        let (ok, q) = prove_and_verify("(/ x y)", &[("x", a), ("y", b)]);
        assert!(ok, "verify ({a} / {b})");
        assert_eq!(q, a / b, "{a} / {b}");

        let (ok, r) = prove_and_verify("(mod x y)", &[("x", a), ("y", b)]);
        assert!(ok, "verify ({a} % {b})");
        assert_eq!(r, a % b, "{a} % {b}");
    }
}

/// **The attack the gadget exists to stop.**
///
/// Circuit multiplication wraps mod 2^64, so the naive check `a == b*q + r`
/// with `|r| < |b|` is satisfiable by a false quotient: for `a = 1, b = 3`, the
/// prover can set `r = 0` and `q = 3^-1 mod 2^64`, since 3 is invertible.
/// The no-overflow constraint on the product is what rejects it.
///
/// This drives the forgery through the real prover rather than reasoning about
/// it: `1 / 3` is claimed to be the modular inverse of 3.
#[test]
fn a_modular_inverse_quotient_is_rejected() {
    let be = Binius64Real::new();
    let expr = parse("(/ x y)").expect("parse");
    let ckt = be
        .compile_circuit(&expr, &["x".to_string(), "y".to_string()])
        .expect("compile");

    // 3^-1 mod 2^64: the value that satisfies 3*q == 1 with wrapping.
    let inv3 = 0xAAAA_AAAA_AAAA_AAABu64 as i64;
    assert_eq!((3i64).wrapping_mul(inv3), 1, "inv3 really is the inverse");

    let mut priv_map = HashMap::new();
    priv_map.insert("x".to_string(), 1i64);
    priv_map.insert("y".to_string(), 3i64);
    let mut pubs = HashMap::new();
    pubs.insert(OUTPUT_NAME.to_string(), inv3);

    assert!(
        be.prove(&ckt, &priv_map, &pubs).is_err(),
        "the modular-inverse forgery must be rejected: this is the whole point \
         of the no-overflow constraint on the product"
    );

    // And the honest answer is still provable.
    let mut honest = HashMap::new();
    honest.insert(OUTPUT_NAME.to_string(), 0i64);
    let proof = be.prove(&ckt, &priv_map, &honest).expect("1/3 == 0");
    assert!(be.verify(&ckt, &proof, &honest).unwrap());
}

/// Division by zero must be refused, not wrapped into some other value.
#[test]
fn division_by_zero_is_refused() {
    let be = Binius64Real::new();
    let expr = parse("(/ x y)").expect("parse");
    let ckt = be
        .compile_circuit(&expr, &["x".to_string(), "y".to_string()])
        .expect("compile");
    let mut priv_map = HashMap::new();
    priv_map.insert("x".to_string(), 5i64);
    priv_map.insert("y".to_string(), 0i64);
    let mut pubs = HashMap::new();
    pubs.insert(OUTPUT_NAME.to_string(), 0i64);
    assert!(be.prove(&ckt, &priv_map, &pubs).is_err());
}

/// A wrong-but-plausible quotient (off by one) must be rejected — this is what
/// the remainder-sign constraint rules out.
#[test]
fn an_off_by_one_quotient_is_rejected() {
    let be = Binius64Real::new();
    let expr = parse("(/ x y)").expect("parse");
    let ckt = be
        .compile_circuit(&expr, &["x".to_string(), "y".to_string()])
        .expect("compile");
    let mut priv_map = HashMap::new();
    priv_map.insert("x".to_string(), 7i64);
    priv_map.insert("y".to_string(), 2i64);
    for claimed in [2i64, 4, -3] {
        let mut pubs = HashMap::new();
        pubs.insert(OUTPUT_NAME.to_string(), claimed);
        assert!(
            be.prove(&ckt, &priv_map, &pubs).is_err(),
            "7/2 = {claimed} must not be provable"
        );
    }
}

/// **The derived forms reach the real prover with no compiler change.**
///
/// `cond`, `when`, `let*`, named `let` and `case` are desugared in the parser
/// into `If`/`Let`/`LetRec`/`Seq`, precisely so the circuit backends need no
/// knowledge of them. This test is the load-bearing evidence for that design
/// choice: a ReLU *written as a `cond`* compiles, proves and verifies through
/// Binius64, and a false claim about its output is still rejected.
///
/// If someone later gives one of these forms its own `Expr` variant, this test
/// fails and tells them the compilers now need teaching too.
#[test]
fn derived_forms_compile_prove_and_verify() {
    let be = Binius64Real::new();
    // relu(x), written three ways that must all lower to the same circuit.
    for src in [
        "(cond ((> x 0) x) (else 0))",
        "(when (> x 0) x)",
        "(let* ((positive (> x 0))) (* x positive))",
    ] {
        let expr = parse(src).unwrap_or_else(|e| panic!("parse {src}: {e}"));
        let ckt = be
            .compile_circuit(&expr, &["x".to_string()])
            .unwrap_or_else(|e| panic!("compile {src}: {e}"));

        for (x, out) in [(5i64, 5i64), (-5, 0), (7, 7)] {
            let mut pubs = HashMap::new();
            pubs.insert(OUTPUT_NAME.to_string(), out);
            let mut w = HashMap::new();
            w.insert("x".to_string(), x);

            let proof = be
                .prove(&ckt, &w, &pubs)
                .unwrap_or_else(|e| panic!("prove {src} at x={x}: {e}"));
            assert!(
                be.verify(&ckt, &proof, &pubs).expect("verify"),
                "{src} should verify at x={x} with relu={out}"
            );
        }

        // Soundness is not weakened by the desugaring: a wrong claimed output
        // must still fail, or the form would be proving nothing.
        let mut bad = HashMap::new();
        bad.insert(OUTPUT_NAME.to_string(), 99);
        let mut w = HashMap::new();
        w.insert("x".to_string(), -5);
        let rejected = match be.prove(&ckt, &w, &bad) {
            Err(_) => true,
            Ok(p) => !be.verify(&ckt, &p, &bad).unwrap_or(false),
        };
        assert!(rejected, "{src}: relu(-5)=99 must not be provable");
    }
}
