//! Private-inference demo: a linear discriminator over encrypted pixels, run
//! on every FHE scheme Baru supports plus MPC, on two datasets.
//!
//! A single neuron with Q16.16 fixed-point weights (integer weights / 128)
//! scores two flattened 28x28 images.  The pixels are the private input; the
//! weights are public constants baked into the circuit:
//!
//!   score(img) = bias + dot(img, weights)        (Q16.16 fixed-point)
//!
//! Datasets (both 28x28, 784 pixels, so the same circuit serves both):
//!   mnist        handwritten digits — the digit "0" must outscore the "1"
//!   pneumonia    PneumoniaMNIST chest X-rays (MedMNIST v2, CC BY 4.0) — the
//!                pneumonia scan must outscore the normal one (--pneumonia)
//!
//! Backends:
//!   plain      CEK interpreter — ground truth
//!   fhe-mock   FHE circuit pipeline on plaintext u64 (always run, instant)
//!   mpc        3-party ABY3, pixels secret-shared, integer weights (--mpc)
//!   fhe-tfhe   real TFHE — 784 encrypted pixels, minutes of compute (--tfhe)
//!   fhe-ckks   SEAL CKKS — same circuit on scalar ciphertexts, plus the
//!              batched SIMD form (whole image in one ciphertext) (--ckks)
//!
//! Run:  cargo run --release --example mnist [--pneumonia] [--mpc] [--tfhe] [--ckks]
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
//!
//! ╰─ cargo run --release --example mnist -- --ckks                                                                                                                                                               ─╯
//! fhe-ckks: SEAL CKKS (N=8192, 128-bit) — approximate real arithmetic
//! fhe-ckks   keygen 669.0ms   (client 786.6 KB, server 39.9 MB incl. rotation keys)
//! fhe-ckks   [circuit] score("0")=3389.7422  score("1")=2853.0156  → predicts 0   (compile 2.8ms, 2353 gates, encrypt 8.3s, eval 1.6s, 784 cts/img)
//! fhe-ckks   [batched] score("0")=3389.7422  score("1")=2853.0156  → predicts 0   (enc+eval+dec 91.6ms, 1 ct/img = 393.3 KB)
//! fhe-ckks   [circuit] max |error| vs plain = 8.15e-9
//! fhe-ckks   [batched] max |error| vs plain = 9.58e-8
//!
//! CKKS vs TFHE on the same 2365-gate linear circuit: eval 1.6s vs 2829s
//! (~1800x), and the batched SIMD form does both images in 92ms (~30000x)
//! with one 393 KB ciphertext per image instead of 784.  The price: results
//! are approximate (~1e-8 here), only +,-,* are computable (no comparisons,
//! so no encrypted argmax/ReLU), and the server needs 40 MB of rotation keys.
//!
//! ╰─ cargo run --release --example mnist -- --pneumonia --mpc --ckks                                                                                                                                             ─╯
//! PneumoniaMNIST linear discriminator — 784 private pixels, Q16.16 public weights
//! pediatric chest X-ray: pneumonia vs normal
//! this integer model scores 84.13% over the 624-image test set (majority-class baseline 62.50%)
//!
//! plain      score(pneumonia)=1870.8125  score(normal)=-1173.4219  → predicts pneumonia   (2.8ms)
//! fhe-mock   score(pneumonia)=1870.8125  score(normal)=-1173.4219  → predicts pneumonia   (compile 2.8ms, 2380 gates, eval 917.0µs)
//! mpc-aby3   score(pneumonia)=1870.8125  score(normal)=-1173.4219  → predicts pneumonia   (101.9ms, Q16.16 shares)
//! fhe-ckks   keygen 662.3ms   (client 786.6 KB, server 39.9 MB incl. rotation keys)
//! fhe-ckks   [circuit] score(pneumonia)=1870.8125  score(normal)=-1173.4219  → predicts pneumonia   (compile 2.5ms, 2353 gates, encrypt 8.4s, eval 1.9s, 784 cts/img)
//! fhe-ckks   [batched] score(pneumonia)=1870.8125  score(normal)=-1173.4219  → predicts pneumonia   (enc+eval+dec 91.2ms, 1 ct/img = 393.3 KB)
//! fhe-ckks   [circuit] max |error| vs plain = 4.47e-9
//! fhe-ckks   [batched] max |error| vs plain = 1.23e-7
//!
//! Every exact backend (plain, fhe-mock, mpc-aby3) returns the *same* Q16.16
//! score bit-for-bit, and CKKS agrees to ~1e-7.
//!
//! Same circuit, same cost, a medical decision instead of a digit — which is
//! the point: the cost of privacy here is a property of the *program shape*
//! (784 encrypted inputs, one dot product), not of what the pixels mean.
//! The gain on the weights is chosen so the Q16.16 accumulator stays inside
//! the 32-bit MPC shares (see tools/train_medmnist.py --linear); chest X-rays
//! are dense where digits are mostly zero, so that bound actually binds.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;

use baru::fhe::{CkksBackend, FheBackend, MockFheBackend, RealFheBackend, TfheBackend};
use baru::interpreter::{parse, Participant, Types};
use baru::zkp::circuit_compiler::CircuitCompiler;

mod digits {
    include!("data/mnist_data.rs");
}
mod pneumonia {
    include!("data/pneumonia_linear.rs");
}

const FIXED_SCALE: f64 = 65536.0;
/// Weights are scaled to w/128 in Q16.16 — exactly representable (128 = 2^7).
const WEIGHT_DIV: f64 = 128.0;

/// One linear discriminator: public Q16.16 weights, and two private images of
/// which `img_hi` must score strictly higher.
struct Dataset {
    name: &'static str,
    /// One-line description, printed for the medical datasets.
    task: Option<&'static str>,
    weights: &'static [i64],
    bias: i64,
    img_hi: &'static [i64],
    img_lo: &'static [i64],
    label_hi: &'static str,
    label_lo: &'static str,
    /// `(integer-model test accuracy, majority-class baseline, test-set size)`
    /// where a real trained model exists. The MNIST weights are hand-written,
    /// so they have no meaningful accuracy to quote.
    accuracy: Option<(f64, f64, usize)>,
}

fn mnist_dataset() -> Dataset {
    Dataset {
        name: "MNIST",
        task: None,
        weights: &digits::WEIGHTS,
        bias: 80,
        img_hi: &digits::IMG_ZERO,
        img_lo: &digits::IMG_ONE,
        label_hi: "0",
        label_lo: "1",
        accuracy: None,
    }
}

fn pneumonia_dataset() -> Dataset {
    Dataset {
        name: pneumonia::DATASET,
        task: Some(pneumonia::TASK),
        weights: &pneumonia::WEIGHTS,
        bias: pneumonia::BIAS,
        img_hi: &pneumonia::IMG_POS,
        img_lo: &pneumonia::IMG_NEG,
        label_hi: pneumonia::POS_LABEL,
        label_lo: pneumonia::NEG_LABEL,
        accuracy: Some((
            pneumonia::TEST_ACCURACY,
            pneumonia::MAJORITY_BASELINE,
            pneumonia::TEST_N,
        )),
    }
}

/// `(+ bias (dot (list p0 .. p783) '(w0 .. w783)))` with fixed-point weights.
fn score_expr_fixed(ds: &Dataset) -> String {
    let mut pixels = String::new();
    let mut weights = String::new();
    for (i, elem) in ds.weights.iter().enumerate() {
        let _ = write!(pixels, " p{}", i);
        let _ = write!(weights, " {}", *elem as f64 / WEIGHT_DIV);
    }
    format!(
        "(+ {} (dot (list{}) '({} )))",
        ds.bias as f64 / WEIGHT_DIV,
        pixels,
        weights
    )
}

/// Same fixed-point score over secret-shared pixels.  Integer pixel shares
/// × fixed public weights are exact (no share truncation needed), so the
/// MPC score matches the plain Q16.16 score bit-for-bit.
fn score_expr_mpc(ds: &Dataset, img: &[i64]) -> String {
    let mut pixels = String::new();
    let mut weights = String::new();
    for (px, w) in img.iter().zip(ds.weights) {
        let _ = write!(pixels, " (share {} 32)", px);
        let _ = write!(weights, " {}", *w as f64 / WEIGHT_DIV);
    }
    format!(
        "(reconstruct (+ {} (dot (list{}) '({} ))))",
        ds.bias as f64 / WEIGHT_DIV,
        pixels,
        weights
    )
}

fn eval_types(src: &str) -> Types {
    let expr = parse(src).expect("parse");
    Participant::new_simulated(0).eval_quiet(&expr)
}

/// Plain-interpreter fixed score (raw Q16.16).
fn plain_score(ds: &Dataset, img: &[i64]) -> i64 {
    let mut bindings = String::new();
    for (i, px) in img.iter().enumerate() {
        let _ = write!(bindings, " (p{} {})", i, px);
    }
    let src = format!("(let ({}) {})", bindings, score_expr_fixed(ds));
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
    img: &[i64],
) -> i64 {
    let mut enc: HashMap<String, B::Ct> = HashMap::new();
    for (i, px) in img.iter().enumerate() {
        enc.insert(format!("p{}", i), be.encrypt(ck, *px as u64));
    }
    let ct = baru::fhe::circuit_emitter::evaluate_fhe(circuit, &HashMap::new(), &enc, be);
    be.decrypt(ck, &ct) as i64
}

fn compile_circuit(ds: &Dataset) -> baru::fhe::backend::FheCircuit {
    let expr = parse(&score_expr_fixed(ds)).expect("parse");
    let mut cc = CircuitCompiler::new();
    let mut circuit = cc.compile_fhe(&expr).expect("compile_fhe");
    circuit.encrypted_inputs = (0..ds.weights.len()).map(|i| format!("p{}", i)).collect();
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
    let run_ckks = args.iter().any(|a| a == "--ckks");
    let ds = if args.iter().any(|a| a == "--pneumonia") {
        pneumonia_dataset()
    } else {
        mnist_dataset()
    };
    let (hi, lo) = (ds.label_hi, ds.label_lo);
    let mut failures = 0;

    println!(
        "{} linear discriminator — {} private pixels, Q16.16 public weights",
        ds.name,
        ds.weights.len()
    );
    if let Some(task) = ds.task {
        println!("{task}");
    }
    if let Some((acc, base, n)) = ds.accuracy {
        println!(
            "this integer model scores {:.2}% over the {}-image test set \
             (majority-class baseline {:.2}%)",
            acc * 100.0,
            n,
            base * 100.0
        );
    }
    println!();

    // ── Plain baseline ────────────────────────────────────────────────────
    let t0 = Instant::now();
    let plain_hi = plain_score(&ds, ds.img_hi);
    let plain_lo = plain_score(&ds, ds.img_lo);
    println!(
        "plain      score({})={:.4}  score({})={:.4}  → predicts {}   ({:.1?})",
        hi,
        as_decimal(plain_hi),
        lo,
        as_decimal(plain_lo),
        if plain_hi > plain_lo { hi } else { lo },
        t0.elapsed()
    );
    if plain_hi <= plain_lo {
        eprintln!("plain: MISCLASSIFIED — {hi:?} must outscore {lo:?}");
        failures += 1;
    }

    // ── FHE circuit, mock backend ─────────────────────────────────────────
    let t0 = Instant::now();
    let circuit = compile_circuit(&ds);
    let compile_t = t0.elapsed();

    let be = MockFheBackend;
    let (ck, _) = be.keygen();
    let t0 = Instant::now();
    let mock_hi = fhe_score(&be, &ck, &circuit, ds.img_hi);
    let mock_lo = fhe_score(&be, &ck, &circuit, ds.img_lo);
    println!(
        "fhe-mock   score({})={:.4}  score({})={:.4}  → predicts {}   (compile {:.1?}, {} gates, eval {:.1?})",
        hi,
        as_decimal(mock_hi),
        lo,
        as_decimal(mock_lo),
        if mock_hi > mock_lo { hi } else { lo },
        compile_t,
        circuit.num_gates,
        t0.elapsed()
    );
    if (mock_hi, mock_lo) != (plain_hi, plain_lo) {
        eprintln!("fhe-mock: DIVERGES from plain interpreter");
        failures += 1;
    }

    // ── MPC (optional) ────────────────────────────────────────────────────
    if run_mpc {
        let t0 = Instant::now();
        let scores: Vec<i64> = [ds.img_hi, ds.img_lo]
            .iter()
            .map(|img| match eval_types(&score_expr_mpc(&ds, img)) {
                Types::Fixed(f) => f,
                other => panic!("expected Fixed from reconstruct, got {:?}", other),
            })
            .collect();
        println!(
            "mpc-aby3   score({})={:.4}  score({})={:.4}  → predicts {}   ({:.1?}, Q16.16 shares)",
            hi,
            as_decimal(scores[0]),
            lo,
            as_decimal(scores[1]),
            if scores[0] > scores[1] { hi } else { lo },
            t0.elapsed()
        );
        if (scores[0], scores[1]) != (plain_hi, plain_lo) {
            eprintln!(
                "mpc-aby3: DIVERGES from plain interpreter — if the scores look \
                 wrapped, the Q16.16 accumulator has outgrown the 32-bit shares"
            );
            failures += 1;
        }
    }

    // ── Real TFHE (optional, slow) ────────────────────────────────────────
    if run_tfhe {
        println!(
            "\nfhe-tfhe: keygen + {} encrypted pixels — this takes minutes...",
            ds.weights.len()
        );
        let be = TfheBackend::new();
        let t0 = Instant::now();
        let (ck, sk) = be.keygen();
        TfheBackend::install_server_key(&sk);
        println!("fhe-tfhe   keygen {:.1?}", t0.elapsed());

        let t0 = Instant::now();
        let tfhe_hi = fhe_score(&be, &ck, &circuit, ds.img_hi);
        let tfhe_lo = fhe_score(&be, &ck, &circuit, ds.img_lo);
        println!(
            "fhe-tfhe   score({})={:.4}  score({})={:.4}  → predicts {}   (eval {:.1?})",
            hi,
            as_decimal(tfhe_hi),
            lo,
            as_decimal(tfhe_lo),
            if tfhe_hi > tfhe_lo { hi } else { lo },
            t0.elapsed()
        );
        if (tfhe_hi, tfhe_lo) != (plain_hi, plain_lo) {
            eprintln!("fhe-tfhe: DIVERGES from plain interpreter");
            failures += 1;
        }
    }

    // ── CKKS (optional) ───────────────────────────────────────────────────
    // Two forms:
    //   circuit — the same 2365-gate IR as TFHE, compiled in real-arithmetic
    //             mode, one scalar ciphertext per pixel (apples-to-apples).
    //   batched — the idiomatic CKKS encoding: all 784 pixels in one
    //             ciphertext's SIMD slots, dot product via rotations.
    // CKKS is approximate: scores are compared with a tolerance and the
    // prediction (argmax) must match plain.
    if run_ckks {
        const TOL: f64 = 1e-3;
        println!("\nfhe-ckks: SEAL CKKS (N=8192, 128-bit) — approximate real arithmetic");
        let t0 = Instant::now();
        let (ck, sk) = CkksBackend::new().keygen();
        let be = CkksBackend::with_server_keys(&sk);
        println!(
            "fhe-ckks   keygen {:.1?}   (client {:.1} KB, server {:.1} MB incl. rotation keys)",
            t0.elapsed(),
            ck.len() as f64 / 1e3,
            sk.len() as f64 / 1e6
        );

        let weights: Vec<f64> = ds.weights.iter().map(|w| *w as f64 / WEIGHT_DIV).collect();
        let bias = ds.bias as f64 / WEIGHT_DIV;

        // Circuit form: same IR as TFHE, scalar ciphertext per pixel.
        let t0 = Instant::now();
        let expr = parse(&score_expr_fixed(&ds)).expect("parse");
        let mut cc = CircuitCompiler::new_real();
        let mut real_circuit = cc.compile_fhe(&expr).expect("compile_fhe real");
        real_circuit.encrypted_inputs = (0..ds.weights.len()).map(|i| format!("p{}", i)).collect();
        real_circuit.public_inputs = vec![];
        let compile_t = t0.elapsed();

        let mut circuit_scores = [0.0f64; 2];
        let mut enc_t = std::time::Duration::ZERO;
        let mut eval_t = std::time::Duration::ZERO;
        for (s, img) in circuit_scores.iter_mut().zip([ds.img_hi, ds.img_lo]) {
            let t0 = Instant::now();
            let mut enc: HashMap<String, _> = HashMap::new();
            for (i, px) in img.iter().enumerate() {
                enc.insert(format!("p{}", i), be.encrypt(&ck, *px as f64));
            }
            enc_t += t0.elapsed();
            let t0 = Instant::now();
            let ct = baru::fhe::circuit_emitter::evaluate_fhe_real(
                &real_circuit,
                &HashMap::new(),
                &enc,
                &be,
            );
            eval_t += t0.elapsed();
            *s = be.decrypt(&ck, &ct);
        }
        println!(
            "fhe-ckks   [circuit] score({})={:.4}  score({})={:.4}  → predicts {}   (compile {:.1?}, {} gates, encrypt {:.1?}, eval {:.1?}, {} cts/img)",
            hi,
            circuit_scores[0],
            lo,
            circuit_scores[1],
            if circuit_scores[0] > circuit_scores[1] { hi } else { lo },
            compile_t,
            real_circuit.num_gates,
            enc_t,
            eval_t,
            ds.weights.len()
        );

        // Batched form: whole image in one ciphertext, rotations sum the dot.
        let mut batched_scores = [0.0f64; 2];
        let mut ct_bytes = 0usize;
        let t0 = Instant::now();
        for (s, img) in batched_scores.iter_mut().zip([ds.img_hi, ds.img_lo]) {
            let pixels: Vec<f64> = img.iter().map(|p| *p as f64).collect();
            let ct = be.encrypt_vec(&ck, &pixels);
            ct_bytes = be.serialize_ct(&ct).len();
            let dot = be.dot_plain(&ct, &weights);
            let score = be.add_plain(&dot, bias);
            *s = be.decrypt(&ck, &score);
        }
        println!(
            "fhe-ckks   [batched] score({})={:.4}  score({})={:.4}  → predicts {}   (enc+eval+dec {:.1?}, 1 ct/img = {:.1} KB)",
            hi,
            batched_scores[0],
            lo,
            batched_scores[1],
            if batched_scores[0] > batched_scores[1] { hi } else { lo },
            t0.elapsed(),
            ct_bytes as f64 / 1e3
        );

        let plain_real = [as_decimal(plain_hi), as_decimal(plain_lo)];
        for (label, scores) in [("circuit", circuit_scores), ("batched", batched_scores)] {
            let err = (scores[0] - plain_real[0])
                .abs()
                .max((scores[1] - plain_real[1]).abs());
            let agree = (scores[0] > scores[1]) == (plain_real[0] > plain_real[1]);
            println!("fhe-ckks   [{}] max |error| vs plain = {:.2e}", label, err);
            if err > TOL || !agree {
                eprintln!("fhe-ckks [{}]: DIVERGES from plain interpreter", label);
                failures += 1;
            }
        }
    }

    if failures > 0 {
        std::process::exit(1);
    }
    println!("\nall backends agree — image {hi:?} outscores image {lo:?}");
}
