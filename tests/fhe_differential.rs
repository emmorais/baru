//! Differential tests for the FHE pipeline.
//!
//! The same expression (a function of a single variable `x`) is evaluated by
//!   1. the plain CEK interpreter, and
//!   2. the FHE circuit pipeline (compile_fhe → evaluate_fhe) with `x`
//!      supplied as an encrypted input.
//!
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
use baru::interpreter::{parse, Participant, Types};
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
#[ignore]
fn diff_arithmetic() {
    assert_differential("(+ (* x 3) 7)", SIGNED_SAMPLES);
    assert_differential("(- x 10)", SIGNED_SAMPLES);
    assert_differential("(* (+ x 1) (- x 1))", SIGNED_SAMPLES);
}

#[test]
#[ignore]
fn diff_relu() {
    assert_differential("(if (> x 0) x 0)", SIGNED_SAMPLES);
}

#[test]
#[ignore]
fn diff_comparisons_in_if() {
    assert_differential("(if (< x 5) (* x 2) (+ x 1))", SIGNED_SAMPLES);
    assert_differential("(if (== x 7) 1 0)", SIGNED_SAMPLES);
}

#[test]
#[ignore]
fn diff_le_ge_lowering() {
    // <= and >= lower to comparison + logical NOT (XOR 1) — a bitwise NOT
    // here would poison the mux condition.
    assert_differential("(if (<= x 5) 10 20)", SIGNED_SAMPLES);
    assert_differential("(if (>= x 5) 10 20)", SIGNED_SAMPLES);
}

#[test]
#[ignore]
fn diff_logical_not() {
    assert_differential("(if (not (< x 5)) 1 0)", SIGNED_SAMPLES);
}

#[test]
#[ignore]
fn diff_nested_if() {
    assert_differential("(if (> x 0) (if (< x 10) x 10) 0)", SIGNED_SAMPLES);
}

#[test]
#[ignore]
fn diff_div_mod() {
    // Plain `/` demands exact division, so sample multiples of 3 only.
    assert_differential("(/ x 3)", &[0, 3, 6, 21]);
    assert_differential("(mod x 3)", &[0, 1, 5, 7, 12]);
}

const DOT_PRODUCT_SRC: &str = "(letrec ((dot (lambda (a b)
       (if (null? a)
           0
           (+ (* (car a) (car b)) (dot (cdr a) (cdr b)))))))
     (dot (list x y) '(2 3)))";

/// Compile the recursive dot product with x and y as encrypted inputs and
/// evaluate it homomorphically: dot([x, y], [2, 3]) = 2x + 3y.
fn eval_dot_product_fhe<B: baru::fhe::FheBackend>(be: &B, ck: &[u8], x: i64, y: i64) -> i64 {
    let expr = parse(DOT_PRODUCT_SRC).expect("parse");
    let mut cc = CircuitCompiler::new();
    let mut circuit = cc.compile_fhe(&expr).expect("compile_fhe");
    circuit.encrypted_inputs = vec!["x".to_string(), "y".to_string()];
    circuit.public_inputs = vec![];

    let mut enc = HashMap::new();
    enc.insert("x".to_string(), be.encrypt(ck, x as u64));
    enc.insert("y".to_string(), be.encrypt(ck, y as u64));
    let ct = evaluate_fhe(&circuit, &HashMap::new(), &enc, be);
    be.decrypt(ck, &ct) as i64
}

#[test]
#[ignore]
fn diff_dot_product_recursion_mock() {
    // Recursion over a compile-time list is unrolled by the circuit compiler;
    // the encrypted scalars flow through the unrolled multiply-add chain.
    let be = MockFheBackend;
    let (ck, _) = be.keygen();
    assert_eq!(eval_dot_product_fhe(&be, &ck, 4, 5), 23); // 2*4 + 3*5
    assert_eq!(eval_dot_product_fhe(&be, &ck, -1, 7), 19); // -2 + 21
}

/// Plain evaluation returning the raw wire value: Num as-is, Fixed as its
/// Q16.16 scaled integer (which is exactly what a fixed circuit outputs).
fn eval_plain_raw(expr_src: &str, x: i64) -> i64 {
    let src = format!("(let ((x {})) {})", x, expr_src);
    let expr = parse(&src).expect("parse");
    match Participant::new(0).eval_quiet(&expr) {
        Types::Num(n) => n,
        Types::Fixed(f) => f,
        other => panic!("expected Num or Fixed, got {:?}", other),
    }
}

#[test]
#[ignore]
fn diff_fixed_point_mock() {
    // Q16.16 arithmetic compiled to circuits must match the interpreter on
    // the raw scaled representation.  Division stays non-negative (IDiv is
    // unsigned), everything else exercises negatives too.
    let cases: &[(&str, &[i64])] = &[
        ("(* x 1.5)", &[4, -6, 0, 7]),
        ("(* 0.5 (* x 1.5))", &[4, -4]), // fixed×fixed rescale chain
        ("(+ 1.25 (* x 0.5))", &[3, -3]),
        ("(- (* x 0.5) 2.75)", &[9, -9]),
        ("(dot (list x x) '(0.5 2.5))", &[4, -4]),
        ("(if (> (* x 1.5) 0.0) (* x 1.5) 0.0)", &[5, -5]),
        ("(fixed->int (* x 2.5))", &[3, -3, 0]),
        ("(int->fixed x)", &[7, -7]),
        ("(/ (int->fixed x) 4.0)", &[10, 0]),
    ];
    for (expr, xs) in cases {
        let circuit = compile_with_encrypted_x(expr);
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        for &x in *xs {
            let plain = eval_plain_raw(expr, x);
            let mut enc = HashMap::new();
            enc.insert("x".to_string(), be.encrypt(&ck, x as u64));
            let ct = evaluate_fhe(&circuit, &HashMap::new(), &enc, &be);
            let fhe = be.decrypt(&ck, &ct) as i64;
            assert_eq!(
                plain, fhe,
                "fixed-point divergence for `{}` at x={}: plain={} fhe={}",
                expr, x, plain, fhe
            );
        }
    }
}

#[test]
#[ignore]
fn diff_fixed_output_flag() {
    // compile_fhe must mark fixed outputs (and only those).
    let expr = parse("(* x 1.5)").expect("parse");
    let mut cc = CircuitCompiler::new();
    assert!(cc.compile_fhe(&expr).expect("compile").output_fixed);

    let expr = parse("(fixed->int (* x 1.5))").expect("parse");
    let mut cc = CircuitCompiler::new();
    assert!(!cc.compile_fhe(&expr).expect("compile").output_fixed);

    let expr = parse("(+ (* x 3) 7)").expect("parse");
    let mut cc = CircuitCompiler::new();
    assert!(!cc.compile_fhe(&expr).expect("compile").output_fixed);
}

#[test]
#[ignore]
fn diff_dot_builtin_mock() {
    // The native `dot` builtin lowers to a flat multiply-add chain — no
    // letrec unrolling involved.  dot([x, y], [2, 3]) = 2x + 3y.
    let src = "(dot (list x y) '(2 3))";
    let expr = parse(src).expect("parse");
    let mut cc = CircuitCompiler::new();
    let mut circuit = cc.compile_fhe(&expr).expect("compile_fhe");
    circuit.encrypted_inputs = vec!["x".to_string(), "y".to_string()];
    circuit.public_inputs = vec![];

    let be = MockFheBackend;
    let (ck, _) = be.keygen();
    for (x, y) in [(4i64, 5i64), (-1, 7), (0, 0)] {
        let plain = eval_plain(&format!("(let ((x {}) (y {})) {})", x, y, src));
        let mut enc = HashMap::new();
        enc.insert("x".to_string(), be.encrypt(&ck, x as u64));
        enc.insert("y".to_string(), be.encrypt(&ck, y as u64));
        let ct = evaluate_fhe(&circuit, &HashMap::new(), &enc, &be);
        let fhe = be.decrypt(&ck, &ct) as i64;
        assert_eq!(plain, fhe, "dot builtin divergence at x={} y={}", x, y);
    }
}

#[test]
#[ignore]
fn diff_matvec_builtin_mock() {
    // (vsum (matvec [[2 -3] [1 2]] [x y])) — a two-neuron linear layer.
    let src = "(vsum (matvec (list '(2 -3) '(1 2)) (list x y)))";
    let expr = parse(src).expect("parse");
    let mut cc = CircuitCompiler::new();
    let mut circuit = cc.compile_fhe(&expr).expect("compile_fhe");
    circuit.encrypted_inputs = vec!["x".to_string(), "y".to_string()];
    circuit.public_inputs = vec![];

    let be = MockFheBackend;
    let (ck, _) = be.keygen();
    for (x, y) in [(4i64, 4i64), (-2, 3)] {
        let plain = eval_plain(&format!("(let ((x {}) (y {})) {})", x, y, src));
        let mut enc = HashMap::new();
        enc.insert("x".to_string(), be.encrypt(&ck, x as u64));
        enc.insert("y".to_string(), be.encrypt(&ck, y as u64));
        let ct = evaluate_fhe(&circuit, &HashMap::new(), &enc, &be);
        let fhe = be.decrypt(&ck, &ct) as i64;
        assert_eq!(plain, fhe, "matvec builtin divergence at x={} y={}", x, y);
    }
}

// ---------------------------------------------------------------------------
// Real TFHE backend — small subset (keygen once, a few evaluations).
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn diff_tfhe_subset() {
    let be = TfheBackend::new();
    let (ck, sk) = be.keygen();
    TfheBackend::install_server_key(&sk);

    let cases: &[(&str, &[i64])] = &[
        ("(if (< x 5) (* x 2) (+ x 1))", &[3, 9]),
        ("(if (>= x 0) x (- 0 x))", &[-6]), // abs(-6) = 6
        ("(/ x 3)", &[21]),
        ("(fixed->int (* x 2.5))", &[-3]), // Q16.16: rescale + trunc gadgets
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

    // Recursive dot product, unrolled at compile time: 2*4 + 3*5 = 23.
    assert_eq!(eval_dot_product_fhe(&be, &ck, 4, 5), 23);
}

// ---------------------------------------------------------------------------
// CKKS real-arithmetic pipeline: compile_fhe in new_real mode, evaluate with
// RealFheBackend.  The mock twin is exact f64; SEAL is approximate, so the
// comparison is tolerance-based.  Corpus values stay exactly representable
// in Q16.16 so the plain interpreter's truncating fixed multiply introduces
// no rounding of its own.
// ---------------------------------------------------------------------------

use baru::fhe::circuit_emitter::evaluate_fhe_real;
use baru::fhe::{CkksBackend, MockCkksBackend, RealFheBackend};

fn compile_real_with_encrypted_x(expr_src: &str) -> FheCircuit {
    let expr = parse(expr_src).expect("parse");
    let mut cc = CircuitCompiler::new_real();
    let mut circuit = cc.compile_fhe(&expr).expect("compile_fhe real");
    circuit.encrypted_inputs = vec!["x".to_string()];
    circuit.public_inputs.retain(|v| v != "x");
    circuit
}

/// Plain result as a true real: Num as-is, Fixed unscaled.
fn eval_plain_real(expr_src: &str, x: i64) -> f64 {
    let raw = eval_plain_raw(expr_src, x);
    let src = format!("(let ((x {})) {})", x, expr_src);
    let expr = parse(&src).expect("parse");
    match Participant::new(0).eval_quiet(&expr) {
        Types::Fixed(_) => raw as f64 / 65536.0,
        _ => raw as f64,
    }
}

fn ckks_corpus() -> Vec<(&'static str, Vec<i64>)> {
    vec![
        ("(* x 1.5)", vec![4, -6, 0]),
        ("(* 0.5 (* x 1.5))", vec![4, -4]),
        ("(+ 1.25 (* x 0.5))", vec![3, -3]),
        ("(- (* x 0.5) 2.75)", vec![9, -9]),
        ("(* x x)", vec![5, -3]),             // ct×ct + relinearization
        ("(+ (* x x) (* x 2))", vec![4, -2]), // depth-mismatched add (lift)
        ("(dot (list x x) '(0.5 2.5))", vec![4, -4]),
        ("(int->fixed x)", vec![7, -7]),
    ]
}

fn assert_ckks_differential<B: RealFheBackend>(be: &B, tol: f64) {
    let (ck, _) = be.keygen();
    for (expr, xs) in ckks_corpus() {
        let circuit = compile_real_with_encrypted_x(expr);
        for x in xs {
            let plain = eval_plain_real(expr, x);
            let mut enc = HashMap::new();
            enc.insert("x".to_string(), be.encrypt(&ck, x as f64));
            let ct = evaluate_fhe_real(&circuit, &HashMap::new(), &enc, be);
            let got = be.decrypt(&ck, &ct);
            assert!(
                (got - plain).abs() <= tol,
                "CKKS divergence for `{}` at x={}: plain={} ckks={} (tol {})",
                expr,
                x,
                plain,
                got,
                tol
            );
        }
    }
}

#[test]
#[ignore]
fn diff_ckks_mock() {
    assert_ckks_differential(&MockCkksBackend, 1e-9);
}

#[test]
#[ignore]
fn diff_ckks_seal() {
    // Fresh SEAL keygen + evaluation keys; approximate arithmetic, so the
    // tolerance covers CKKS encoding noise (~2^-40 relative per op).
    let (ck, sk) = CkksBackend::new().keygen();
    let be = CkksBackend::with_server_keys(&sk);
    let mut ran = 0;
    for (expr, xs) in ckks_corpus() {
        let circuit = compile_real_with_encrypted_x(expr);
        for x in xs {
            let plain = eval_plain_real(expr, x);
            let mut enc = HashMap::new();
            enc.insert("x".to_string(), be.encrypt(&ck, x as f64));
            let ct = evaluate_fhe_real(&circuit, &HashMap::new(), &enc, &be);
            let got = be.decrypt(&ck, &ct);
            assert!(
                (got - plain).abs() <= 1e-4,
                "SEAL CKKS divergence for `{}` at x={}: plain={} ckks={}",
                expr,
                x,
                plain,
                got
            );
            ran += 1;
        }
    }
    assert!(ran > 0);
}

#[test]
#[ignore]
fn diff_ckks_rejects_comparisons() {
    // Comparisons/mux cannot exist on approximate ciphertexts; the real-mode
    // emitter must fail loudly rather than compute nonsense.
    let circuit = compile_real_with_encrypted_x("(if (> x 0) x 0)");
    let be = MockCkksBackend;
    let (ck, _) = be.keygen();
    let mut enc = HashMap::new();
    enc.insert("x".to_string(), be.encrypt(&ck, 3.0));
    let result = std::panic::catch_unwind(|| {
        evaluate_fhe_real(&circuit, &HashMap::new(), &enc, &MockCkksBackend)
    });
    assert!(result.is_err(), "expected unsupported-gate panic");
}

#[test]
#[ignore]
fn diff_ckks_rejects_fixed_to_int() {
    // Truncation needs comparisons; real-mode compilation must refuse it.
    let expr = parse("(fixed->int (* x 1.5))").expect("parse");
    let mut cc = CircuitCompiler::new_real();
    assert!(cc.compile_fhe(&expr).is_err());
}

#[test]
#[ignore]
fn diff_ckks_batched_dot() {
    // The batched SIMD path used by the MNIST demo: one ciphertext holds the
    // whole vector; dot_plain sums via rotations.
    let be = CkksBackend::new();
    let (ck, sk) = be.keygen();
    let eval_be = CkksBackend::with_server_keys(&sk);

    let xs: Vec<f64> = (0..784).map(|i| (i % 256) as f64 / 255.0).collect();
    let ws: Vec<f64> = (0..784).map(|i| ((i as f64) * 0.37).sin()).collect();
    let expected: f64 = xs.iter().zip(&ws).map(|(a, b)| a * b).sum();

    let ct = be.encrypt_vec(&ck, &xs);
    let dot = eval_be.dot_plain(&ct, &ws);
    let got = eval_be.decrypt(&ck, &dot);
    assert!(
        (got - expected).abs() < 1e-4,
        "batched dot: expected {} got {}",
        expected,
        got
    );
}
