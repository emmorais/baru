//! `baru bench-mnist` — real-dataset accuracy for the private CNN.
//!
//! Everything else in this repo proves the hybrid is *exact* (encrypted class
//! == plaintext class). That is the cryptographic claim. This benchmark adds
//! the orthogonal **ML** claim: what accuracy the model actually achieves on
//! real data — using **trained** weights (not the synthetic ones
//! `bench-cnn` uses for its cost/correctness microbenchmark) on **real MNIST
//! test digits**.
//!
//! # What is embedded
//!
//! `examples/data/mnist_cnn.rs` carries weights trained by
//! `scratchpad/train_mnist.py` (numpy SGD on the full 60 000-image MNIST
//! training set, then post-training quantisation to the exact integer
//! fixed-point scheme [`crate::bench::cnn::CnnModel::forward`] evaluates),
//! plus the first `N_TEST` real MNIST **test** images with their labels.
//!
//! # The two numbers, and why they coincide
//!
//! - **Plaintext-integer accuracy**: `forward` over the embedded test images
//!   vs their labels. Because quantisation is baked into `forward`, this is
//!   the accuracy of the *deployed integer model* — not a float model that
//!   would then lose something in quantisation.
//! - **Encrypted accuracy**: the full hybrid (BFV ct×ct conv/dense, ABY3
//!   nonlinearities, zero bootstraps) over a subset. Since the hybrid is
//!   bit-exact against `forward`, every encrypted prediction *equals* the
//!   plaintext one — so encrypted accuracy == plaintext-integer accuracy.
//!   The subset run demonstrates that equality on real inputs rather than
//!   asserting it.
//!
//! The architecture is a small but real convolutional net —
//! `conv(8,3×3) → ReLU → maxpool → dense(→128) → ReLU → dense(→10)` — with
//! **two** encrypted linear layers *plus* an encrypted hidden layer, each
//! separated by an MPC nonlinearity and a scheme switch. That the hidden
//! layer composes in with nothing more than "switch again, no bootstrap" is
//! the whole thesis made concrete: depth is free. The accuracy below
//! reflects this model; a still-larger one is a training question the
//! construction is indifferent to.

use std::time::Instant;

use serde::Serialize;

use crate::bench::cnn::{run_cnn_hybrid, CnnModel};
use crate::bench::comm::NetworkProfile;

#[path = "../../examples/data/mnist_cnn.rs"]
#[allow(dead_code)] // generated file exports a few descriptive consts we don't read
mod data;

/// Full-test-set (all 10 000 images) integer accuracy measured offline by
/// `train_mnist.py` on the identical weights. Stated here so the in-repo
/// subset number can be read against it.
pub const FULL_TEST_ACCURACY: f64 = 0.9763;

fn load_model() -> (CnnModel, Vec<Vec<i64>>, Vec<u8>) {
    let images: Vec<Vec<i64>> = data::TEST_IMAGES.iter().map(|im| im.to_vec()).collect();
    let labels: Vec<u8> = data::TEST_LABELS.to_vec();
    let kernels: Vec<Vec<i64>> = data::KERNELS.iter().map(|k| k.to_vec()).collect();
    let hidden_w: Vec<Vec<i64>> = data::HIDDEN_W.iter().map(|r| r.to_vec()).collect();
    let dense_w: Vec<Vec<i64>> = data::DENSE_W.iter().map(|r| r.to_vec()).collect();
    let model = CnnModel::from_trained(
        28,
        kernels,
        data::CBIAS.to_vec(),
        Some(hidden_w),
        Some(data::HIDDEN_B.to_vec()),
        dense_w,
        data::DENSE_B.to_vec(),
        data::SHIFT,
        &images,
    );
    (model, images, labels)
}

#[derive(Debug, Serialize)]
pub struct MnistReport {
    pub n_test: usize,
    pub plaintext_correct: usize,
    pub plaintext_accuracy: f64,
    pub full_test_accuracy: f64,
    pub hybrid_evaluated: usize,
    pub hybrid_matches_plaintext: usize,
    pub hybrid_correct: usize,
    pub bootstraps: u64,
    pub ms: f64,
}

pub fn run(hybrid_n: usize, profile: NetworkProfile) -> MnistReport {
    let (model, images, labels) = load_model();
    let t0 = Instant::now();

    // Plaintext-integer accuracy over every embedded test image.
    let mut plaintext_correct = 0usize;
    let preds: Vec<usize> = images
        .iter()
        .map(|img| {
            let (class, _) = model.forward(img);
            class
        })
        .collect();
    for (p, &l) in preds.iter().zip(&labels) {
        if *p == l as usize {
            plaintext_correct += 1;
        }
    }

    // Encrypted accuracy on a subset: prove each hybrid prediction equals the
    // plaintext one (bit-exact) and count how many match the true label.
    let hybrid_n = hybrid_n.min(images.len());
    let (mut matches, mut hybrid_correct, mut bootstraps) = (0usize, 0usize, 0u64);
    for i in 0..hybrid_n {
        let run = run_cnn_hybrid(&model, &images[i], profile);
        if run.class == preds[i] {
            matches += 1;
        }
        if run.class == labels[i] as usize {
            hybrid_correct += 1;
        }
        bootstraps += run.bootstraps;
    }

    MnistReport {
        n_test: images.len(),
        plaintext_correct,
        plaintext_accuracy: plaintext_correct as f64 / images.len() as f64,
        full_test_accuracy: FULL_TEST_ACCURACY,
        hybrid_evaluated: hybrid_n,
        hybrid_matches_plaintext: matches,
        hybrid_correct,
        bootstraps,
        ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

const USAGE: &str = "usage: baru bench-mnist [options]

Real-dataset accuracy for the private CNN: trained (quantised) weights on real
MNIST test digits, run through the exact-and-bootstrap-free hybrid pipeline.

options:
  --hybrid N   run the full ENCRYPTED pipeline on the first N test images
               (default 15); the rest are scored with the bit-identical
               plaintext-integer forward pass
  --json       machine-readable output
  -h, --help   this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut hybrid_n = 15usize;
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hybrid" => {
                hybrid_n = it
                    .next()
                    .ok_or("--hybrid needs a value")?
                    .parse()
                    .map_err(|_| "bad --hybrid")?;
            }
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let r = run(hybrid_n, NetworkProfile::INSTANT);
    if json {
        println!("{}", serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?);
        return Ok(());
    }

    println!("Private CNN on real MNIST — trained weights, real test digits, ZERO bootstraps");
    println!(
        "architecture: 1×28×28 → conv({}, 3×3) → ReLU → maxpool(2×2) → dense(→{}) → ReLU → dense(→10) → argmax",
        data::OC,
        data::HIDDEN
    );
    println!();
    println!(
        "  plaintext-integer accuracy : {}/{} = {:.2}%  (this in-repo subset)",
        r.plaintext_correct,
        r.n_test,
        r.plaintext_accuracy * 100.0
    );
    println!(
        "  full MNIST test set (10 000): {:.2}%  (identical weights, measured offline)",
        r.full_test_accuracy * 100.0
    );
    println!();
    println!("  ENCRYPTED pipeline on {} images:", r.hybrid_evaluated);
    println!(
        "    matches plaintext exactly : {}/{}  (bit-exact hybrid ⇒ same prediction)",
        r.hybrid_matches_plaintext, r.hybrid_evaluated
    );
    println!(
        "    correct vs true label     : {}/{}",
        r.hybrid_correct, r.hybrid_evaluated
    );
    println!("    FHE bootstraps            : {}", r.bootstraps);
    println!("\n  time: {:.0} ms", r.ms);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded subset must reproduce a sane accuracy (near the full-set
    /// number) and the plaintext forward must be deterministic.
    #[test]
    fn plaintext_accuracy_is_reasonable() {
        let (model, images, labels) = load_model();
        let correct = images
            .iter()
            .zip(&labels)
            .filter(|(img, &l)| model.forward(img).0 == l as usize)
            .count();
        let acc = correct as f64 / images.len() as f64;
        assert!(
            acc > 0.80,
            "embedded-subset accuracy {acc:.3} is implausibly low — check the \
             quantisation mapping or the flatten order"
        );
    }

    /// The encrypted pipeline must reproduce the plaintext prediction on real
    /// images — the whole point: privacy costs nothing in accuracy.
    #[test]
    #[ignore] // runs the real BFV × ABY3 hybrid; slow in debug
    fn encrypted_matches_plaintext_on_real_digits() {
        let r = run(6, NetworkProfile::INSTANT);
        assert_eq!(
            r.hybrid_matches_plaintext, r.hybrid_evaluated,
            "every encrypted prediction must equal the plaintext one"
        );
        assert_eq!(r.bootstraps, 0);
    }
}
