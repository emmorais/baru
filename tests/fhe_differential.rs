//! Differential tests for the FHE pipeline.
//!
//! The same expression (a function of a single variable `x`) is evaluated by
//!   1. the plain CEK interpreter, and
//!   2. the FHE circuit pipeline (compile_fhe → evaluate_fhe) with `x`
//!      supplied as an encrypted input.
//! Both must agree for every sample value.  A divergence is a bug in the
//! compiler lowering, the circuit emitter, or a backend gate.
//!
//! The corpus runs on the mock backend (instant).  A small subset also runs
//! on the real TFHE backend — slow in debug builds, so prefer
//! `cargo test --release --test fhe_differential`.

use std::collections::HashMap;

use baru::fhe::backend::FheCircuit;
use baru::fhe::circuit_emitter::evaluate_fhe;
use baru::fhe::{FheBackend, MockFheBackend, TfheBackend};
use baru::interpreter::eval::eval_plain;
use baru::interpreter::parse;
use baru::zkp::circuit_compiler::CircuitCompiler;

fn compile_with_encrypted_x(expr_src: &str) -> FheCircuit {
    let expr = parse(expr_src).expect("parse");
    let mut cc = CircuitCompiler::new();
    let mut circuit = cc.compile_fhe(&expr).expect("compile_fhe");
    circuit.encrypted_inputs = vec!["x".to_string()];
    circuit.public_inputs.retain(|v| v != "x");
    circuit
}

/// Evaluate `expr_src` homomorphically on the mock backend with `x` encrypted.
fn eval_fhe_mock(expr_src: &str, x: i64) -> i64 {
    let circuit = compile_with_encrypted_x(expr_src);
    let be = MockFheBackend;
    let (ck, _) = be.keygen();
    let mut enc = HashMap::new();
    enc.insert("x".to_string(), be.encrypt(&ck, x as u64));
    let ct = evaluate_fhe(&circuit, &HashMap::new(), &enc, &be);
    be.decrypt(&ck, &ct) as i64
}

fn assert_differential(expr: &str, xs: &[i64]) {
    for &x in xs {
        let plain = eval_plain(&format!("(let ((x {})) {})", x, expr));
        let fhe = eval_fhe_mock(expr, x);
        assert_eq!(
            plain, fhe,
            "divergence for `{}` at x={}: plain={} fhe={}",
            expr, x, plain, fhe
        );
    }
}

const SIGNED_SAMPLES: &[i64] = &[-7, -1, 0, 1, 5, 7, 12];

#[test]
fn diff_arithmetic() {
    assert_differential("(+ (* x 3) 7)", SIGNED_SAMPLES);
    assert_differential("(- x 10)", SIGNED_SAMPLES);
    assert_differential("(* (+ x 1) (- x 1))", SIGNED_SAMPLES);
}

#[test]
fn diff_relu() {
    assert_differential("(if (> x 0) x 0)", SIGNED_SAMPLES);
}

#[test]
fn diff_comparisons_in_if() {
    assert_differential("(if (< x 5) (* x 2) (+ x 1))", SIGNED_SAMPLES);
    assert_differential("(if (== x 7) 1 0)", SIGNED_SAMPLES);
}

#[test]
fn diff_le_ge_lowering() {
    // <= and >= lower to comparison + logical NOT (XOR 1) — a bitwise NOT
    // here would poison the mux condition.
    assert_differential("(if (<= x 5) 10 20)", SIGNED_SAMPLES);
    assert_differential("(if (>= x 5) 10 20)", SIGNED_SAMPLES);
}

#[test]
fn diff_logical_not() {
    assert_differential("(if (not (< x 5)) 1 0)", SIGNED_SAMPLES);
}

#[test]
fn diff_nested_if() {
    assert_differential("(if (> x 0) (if (< x 10) x 10) 0)", SIGNED_SAMPLES);
}

#[test]
fn diff_div_mod() {
    // Plain `/` demands exact division, so sample multiples of 3 only.
    assert_differential("(/ x 3)", &[0, 3, 6, 21]);
    assert_differential("(mod x 3)", &[0, 1, 5, 7, 12]);
}

// ---------------------------------------------------------------------------
// Real TFHE backend — small subset (keygen once, a few evaluations).
// ---------------------------------------------------------------------------

#[test]
fn diff_tfhe_subset() {
    let be = TfheBackend::new();
    let (ck, sk) = be.keygen();
    TfheBackend::install_server_key(&sk);

    let cases: &[(&str, &[i64])] = &[
        ("(if (< x 5) (* x 2) (+ x 1))", &[3, 9]),
        ("(if (>= x 0) x (- 0 x))", &[-6]), // abs(-6) = 6
        ("(/ x 3)", &[21]),
    ];

    for (expr, xs) in cases {
        let circuit = compile_with_encrypted_x(expr);
        for &x in *xs {
            let plain = eval_plain(&format!("(let ((x {})) {})", x, expr));
            let mut enc = HashMap::new();
            enc.insert("x".to_string(), be.encrypt(&ck, x as u64));
            let ct = evaluate_fhe(&circuit, &HashMap::new(), &enc, &be);
            let fhe = be.decrypt(&ck, &ct) as i64;
            assert_eq!(
                plain, fhe,
                "TFHE divergence for `{}` at x={}: plain={} fhe={}",
                expr, x, plain, fhe
            );
        }
    }
}
