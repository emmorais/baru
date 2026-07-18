//! MNIST private-inference demo: a linear discriminator over encrypted pixels.
//!
//! A single neuron with Q16.16 fixed-point weights (integer weights / 128)
//! scores two flattened 28x28 MNIST digits.  The pixels are the private
//! input; the weights are public constants baked into the circuit:
//!
//!   score(img) = bias + dot(img, weights)        (Q16.16 fixed-point)
//!
//! The digit "0" image must score strictly higher than the "1" image.
//! Backends:
//!   plain      CEK interpreter — ground truth
//!   fhe-mock   FHE circuit pipeline on plaintext u64 (always run, instant)
//!   mpc        3-party ABY3, pixels secret-shared, integer weights (--mpc)
//!   fhe-tfhe   real TFHE — 784 encrypted pixels, minutes of compute (--tfhe)
//!
//! Run:  cargo run --release --example mnist [--mpc] [--tfhe]
//!
//! Exits non-zero if any backend misclassifies or diverges from plain.
//!
//! Results:
//!
//! ╰─ cargo run --release --example mnist -- --tfhe                                                                                                                                                               ─╯
//!    Finished `release` profile [optimized] target(s) in 0.16s
//!     Running `target/release/examples/mnist --tfhe`
//! MNIST linear discriminator — 784 private pixels, Q16.16 public weights
//!
//! plain      score("0")=3389.7422  score("1")=2853.0156  → predicts 0   (2.8ms)
//! fhe-mock   score("0")=3389.7422  score("1")=2853.0156  → predicts 0   (compile 2.8ms, 2365 gates, eval 842.1µs)
//!
//! fhe-tfhe: keygen + 784 encrypted pixels — this takes minutes...
//! fhe-tfhe   keygen 860.9ms
//! fhe-tfhe   score("0")=3389.7422  score("1")=2853.0156  → predicts 0   (eval 2829.1s)
//!
//! all backends agree — image "0" outscores image "1"

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;

use baru::fhe::{FheBackend, MockFheBackend, TfheBackend};
use baru::interpreter::{parse, Participant, Types};
use baru::zkp::circuit_compiler::CircuitCompiler;

include!("data/mnist_data.rs");

const FIXED_SCALE: f64 = 65536.0;
/// Weights are scaled to w/128 in Q16.16 — exactly representable (128 = 2^7).
const WEIGHT_DIV: f64 = 128.0;
const BIAS: i64 = 80;

/// `(+ bias (dot (list p0 .. p783) '(w0 .. w783)))` with fixed-point weights.
fn score_expr_fixed() -> String {
    let mut pixels = String::new();
    let mut weights = String::new();
    for (i, elem) in WEIGHTS.iter().enumerate() {
        let _ = write!(pixels, " p{}", i);
        let _ = write!(weights, " {}", *elem as f64 / WEIGHT_DIV);
    }
    format!(
        "(+ {} (dot (list{}) '({} )))",
        BIAS as f64 / WEIGHT_DIV,
        pixels,
        weights
    )
}

/// Same fixed-point score over secret-shared pixels.  Integer pixel shares
/// × fixed public weights are exact (no share truncation needed), so the
/// MPC score matches the plain Q16.16 score bit-for-bit.
fn score_expr_mpc(img: &[i64; 784]) -> String {
    let mut pixels = String::new();
    let mut weights = String::new();
    for i in 0..784 {
        let _ = write!(pixels, " (share {} 32)", img[i]);
        let _ = write!(weights, " {}", WEIGHTS[i] as f64 / WEIGHT_DIV);
    }
    format!(
        "(reconstruct (+ {} (dot (list{}) '({} ))))",
        BIAS as f64 / WEIGHT_DIV,
        pixels,
        weights
    )
}

fn eval_types(src: &str) -> Types {
    let expr = parse(src).expect("parse");
    Participant::new(0).eval_quiet(&expr)
}

/// Plain-interpreter fixed score (raw Q16.16).
fn plain_score(img: &[i64; 784]) -> i64 {
    let mut bindings = String::new();
    for (i, px) in img.iter().enumerate() {
        let _ = write!(bindings, " (p{} {})", i, px);
    }
    let src = format!("(let ({}) {})", bindings, score_expr_fixed());
    match eval_types(&src) {
        Types::Fixed(f) => f,
        other => panic!("expected Fixed score, got {:?}", other),
    }
}

/// FHE score: compile once, encrypt all 784 pixels, evaluate, decrypt.
fn fhe_score<B: FheBackend>(
    be: &B,
    ck: &[u8],
    circuit: &baru::fhe::backend::FheCircuit,
    img: &[i64; 784],
) -> i64 {
    let mut enc: HashMap<String, B::Ct> = HashMap::new();
    for (i, px) in img.iter().enumerate() {
        enc.insert(format!("p{}", i), be.encrypt(ck, *px as u64));
    }
    let ct = baru::fhe::circuit_emitter::evaluate_fhe(circuit, &HashMap::new(), &enc, be);
    be.decrypt(ck, &ct) as i64
}

fn compile_circuit() -> baru::fhe::backend::FheCircuit {
    let expr = parse(&score_expr_fixed()).expect("parse");
    let mut cc = CircuitCompiler::new();
    let mut circuit = cc.compile_fhe(&expr).expect("compile_fhe");
    circuit.encrypted_inputs = (0..784).map(|i| format!("p{}", i)).collect();
    circuit.public_inputs = vec![];
    assert!(circuit.output_fixed, "score circuit must be fixed-point");
    circuit
}

fn as_decimal(scaled: i64) -> f64 {
    scaled as f64 / FIXED_SCALE
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run_mpc = args.iter().any(|a| a == "--mpc");
    let run_tfhe = args.iter().any(|a| a == "--tfhe");
    let mut failures = 0;

    println!("MNIST linear discriminator — 784 private pixels, Q16.16 public weights\n");

    // ── Plain baseline ────────────────────────────────────────────────────
    let t0 = Instant::now();
    let plain0 = plain_score(&IMG_ZERO);
    let plain1 = plain_score(&IMG_ONE);
    println!(
        "plain      score(\"0\")={:.4}  score(\"1\")={:.4}  → predicts {}   ({:.1?})",
        as_decimal(plain0),
        as_decimal(plain1),
        if plain0 > plain1 { '0' } else { '1' },
        t0.elapsed()
    );
    if plain0 <= plain1 {
        eprintln!("plain: MISCLASSIFIED — \"0\" must outscore \"1\"");
        failures += 1;
    }

    // ── FHE circuit, mock backend ─────────────────────────────────────────
    let t0 = Instant::now();
    let circuit = compile_circuit();
    let compile_t = t0.elapsed();

    let be = MockFheBackend;
    let (ck, _) = be.keygen();
    let t0 = Instant::now();
    let mock0 = fhe_score(&be, &ck, &circuit, &IMG_ZERO);
    let mock1 = fhe_score(&be, &ck, &circuit, &IMG_ONE);
    println!(
        "fhe-mock   score(\"0\")={:.4}  score(\"1\")={:.4}  → predicts {}   (compile {:.1?}, {} gates, eval {:.1?})",
        as_decimal(mock0),
        as_decimal(mock1),
        if mock0 > mock1 { '0' } else { '1' },
        compile_t,
        circuit.num_gates,
        t0.elapsed()
    );
    if (mock0, mock1) != (plain0, plain1) {
        eprintln!("fhe-mock: DIVERGES from plain interpreter");
        failures += 1;
    }

    // ── MPC (optional) ────────────────────────────────────────────────────
    if run_mpc {
        let t0 = Instant::now();
        let scores: Vec<i64> = [&IMG_ZERO, &IMG_ONE]
            .iter()
            .map(|img| match eval_types(&score_expr_mpc(img)) {
                Types::Fixed(f) => f,
                other => panic!("expected Fixed from reconstruct, got {:?}", other),
            })
            .collect();
        println!(
            "mpc-aby3   score(\"0\")={:.4}  score(\"1\")={:.4}  → predicts {}   ({:.1?}, Q16.16 shares)",
            as_decimal(scores[0]),
            as_decimal(scores[1]),
            if scores[0] > scores[1] { '0' } else { '1' },
            t0.elapsed()
        );
        if (scores[0], scores[1]) != (plain0, plain1) {
            eprintln!("mpc-aby3: DIVERGES from plain interpreter");
            failures += 1;
        }
    }

    // ── Real TFHE (optional, slow) ────────────────────────────────────────
    if run_tfhe {
        println!("\nfhe-tfhe: keygen + 784 encrypted pixels — this takes minutes...");
        let be = TfheBackend::new();
        let t0 = Instant::now();
        let (ck, sk) = be.keygen();
        TfheBackend::install_server_key(&sk);
        println!("fhe-tfhe   keygen {:.1?}", t0.elapsed());

        let t0 = Instant::now();
        let tfhe0 = fhe_score(&be, &ck, &circuit, &IMG_ZERO);
        let tfhe1 = fhe_score(&be, &ck, &circuit, &IMG_ONE);
        println!(
            "fhe-tfhe   score(\"0\")={:.4}  score(\"1\")={:.4}  → predicts {}   (eval {:.1?})",
            as_decimal(tfhe0),
            as_decimal(tfhe1),
            if tfhe0 > tfhe1 { '0' } else { '1' },
            t0.elapsed()
        );
        if (tfhe0, tfhe1) != (plain0, plain1) {
            eprintln!("fhe-tfhe: DIVERGES from plain interpreter");
            failures += 1;
        }
    }

    if failures > 0 {
        std::process::exit(1);
    }
    println!("\nall backends agree — image \"0\" outscores image \"1\"");
}
