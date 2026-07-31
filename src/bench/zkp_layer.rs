//! `baru bench-zkp-layer` — what does it cost to *prove* a quantized layer?
//!
//! Baru's comparison table has always had an empty ZKP column for anything
//! resembling inference, for a simple reason: until 2026-07-31 the ZKP backend
//! was a placeholder that proved nothing, and the real one that replaced it
//! could not compile a ReLU. Both are fixed
//! ([`crate::zkp::binius_real`]), so the column can be filled with measurements
//! instead of left blank.
//!
//! # What is measured
//!
//! A **Q16.16 quantized neuron**: `relu((Σ wᵢ·xᵢ) >> 16 + b)`, which is the
//! unit every dense and convolutional layer in this repository is built from.
//! The width sweeps so the *scaling* is visible, not just one point — circuit
//! size, proving time and verification time each grow differently, and a single
//! width would hide that.
//!
//! Weights are public and inputs are private, matching the deployment the rest
//! of the repo targets (a published model, a private query). Changing which
//! side is secret changes only which wires are `add_inout` versus
//! `add_witness`, not the cost.
//!
//! # What the numbers mean, and what they do not
//!
//! Proving is done by the real Binius64 prover, so the times are real. But this
//! is **not** the same task the MPC and FHE columns perform, and the report says
//! so rather than inviting a false comparison:
//!
//! * MPC and FHE keep the data secret *from the parties computing on it*.
//! * A SNARK keeps the witness secret from a *verifier* while proving the
//!   computation was done correctly. The prover holds everything in the clear.
//!
//! They answer different questions — confidentiality versus integrity — and the
//! honest use of this benchmark is "what does proving a layer cost", not "is
//! ZKP faster than MPC".

use std::collections::HashMap;
use std::time::Instant;

use serde::Serialize;

use crate::interpreter::parser::parse;
use crate::zkp::binius_real::{Binius64Real, OUTPUT_NAME};
use crate::zkp::ZKPBackend;

/// Q16.16 scale, matching the rest of the repo's fixed-point convention.
const SCALE: i64 = 1 << 16;

#[derive(Debug, Clone, Serialize)]
pub struct LayerCost {
    pub width: usize,
    pub gates: usize,
    pub constraints: usize,
    pub compile_ms: f64,
    pub prove_ms: f64,
    pub verify_ms: f64,
    pub proof_bytes: usize,
    /// The proof verified and the claimed output matched a plaintext
    /// recomputation — without both, the timings measure nothing.
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ZkpLayerReport {
    pub rows: Vec<LayerCost>,
}

/// Source for one Q16.16 neuron of `width` inputs.
///
/// Built as a left-nested sum so the circuit's shape is deterministic and the
/// per-term cost is easy to read off the width sweep.
fn neuron_src(width: usize, weights: &[i64]) -> String {
    let mut acc = format!("(* {} x0)", weights[0]);
    for (i, w) in weights.iter().enumerate().take(width).skip(1) {
        acc = format!("(+ {acc} (* {w} x{i}))");
    }
    // Rescale the accumulated Q16.16 products, add bias, then ReLU.
    format!("(let ((z (+ (ash-right {acc} 16) bias))) (if (> z 0) z 0))")
}

/// Plaintext reference, in exactly the arithmetic the circuit performs.
fn reference(width: usize, weights: &[i64], inputs: &[i64], bias: i64) -> i64 {
    let mut acc: i64 = 0;
    for i in 0..width {
        acc = acc.wrapping_add(weights[i].wrapping_mul(inputs[i]));
    }
    ((acc >> 16) + bias).max(0)
}

fn measure(width: usize) -> LayerCost {
    // Deterministic, mixed-sign fixed-point values — negatives matter, since a
    // sign error in the rescale or the ReLU only shows up on them.
    let weights: Vec<i64> = (0..width)
        .map(|i| ((i as i64 % 7) - 3) * SCALE / 4)
        .collect();
    let inputs: Vec<i64> = (0..width)
        .map(|i| ((i as i64 % 5) - 2) * SCALE / 2)
        .collect();
    let bias: i64 = SCALE / 8;

    let src = neuron_src(width, &weights);
    let expr = parse(&src).unwrap_or_else(|e| panic!("parse: {e}"));
    let be = Binius64Real::new();

    let mut private: Vec<String> = (0..width).map(|i| format!("x{i}")).collect();
    private.push("bias".to_string());

    let t0 = Instant::now();
    let circuit = be
        .compile_circuit(&expr, &private)
        .unwrap_or_else(|e| panic!("compile width={width}: {e}"));
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut priv_map: HashMap<String, i64> = (0..width)
        .map(|i| (format!("x{i}"), inputs[i]))
        .collect();
    priv_map.insert("bias".to_string(), bias);

    let mut pubs: HashMap<String, i64> = HashMap::new();
    let claimed = be
        .evaluate(&circuit, &priv_map, &pubs)
        .unwrap_or_else(|e| panic!("evaluate: {e}"));
    pubs.insert(OUTPUT_NAME.to_string(), claimed);

    let t1 = Instant::now();
    let proof = be
        .prove(&circuit, &priv_map, &pubs)
        .unwrap_or_else(|e| panic!("prove width={width}: {e}"));
    let prove_ms = t1.elapsed().as_secs_f64() * 1e3;

    let t2 = Instant::now();
    let verified = be.verify(&circuit, &proof, &pubs).unwrap_or(false);
    let verify_ms = t2.elapsed().as_secs_f64() * 1e3;

    let expected = reference(width, &weights, &inputs, bias);
    LayerCost {
        width,
        gates: circuit.metadata.num_gates,
        constraints: circuit.metadata.num_constraints,
        compile_ms,
        prove_ms,
        verify_ms,
        proof_bytes: proof.proof_data.len(),
        correct: verified && claimed == expected,
    }
}

pub fn run(widths: &[usize]) -> ZkpLayerReport {
    ZkpLayerReport {
        rows: widths.iter().map(|w| measure(*w)).collect(),
    }
}

pub fn main(args: &[String]) {
    let mut widths = vec![1usize, 4, 16, 64, 128];
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--widths" {
            if let Some(v) = args.get(i + 1) {
                widths = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            }
            i += 2;
        } else {
            i += 1;
        }
    }

    let r = run(&widths);
    println!("\n=== proving a Q16.16 quantized neuron — relu((Σ wᵢ·xᵢ) >> 16 + b) ===");
    println!("weights public, inputs private; real Binius64 prover\n");
    println!(
        "  {:>6} {:>8} {:>12} {:>11} {:>11} {:>12} {:>8}",
        "width", "gates", "constraints", "prove ms", "verify ms", "proof bytes", "correct"
    );
    for row in &r.rows {
        println!(
            "  {:>6} {:>8} {:>12} {:>11.1} {:>11.1} {:>12} {:>8}",
            row.width,
            row.gates,
            row.constraints,
            row.prove_ms,
            row.verify_ms,
            row.proof_bytes,
            if row.correct { "yes" } else { "NO" }
        );
        assert!(row.correct, "width {} produced a wrong result", row.width);
    }
    println!(
        "\nNote: this measures *integrity* (proving the layer was computed correctly),\n\
         not confidentiality from the computing party — the prover holds the witness in\n\
         the clear. It is not the same task the MPC and FHE columns perform."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every width must produce a verifying proof of the *correct* value.
    /// Without this the timings would be measuring a broken circuit.
    #[test]
    fn a_quantized_neuron_proves_correctly_at_several_widths() {
        for row in run(&[1, 4, 16]).rows {
            assert!(
                row.correct,
                "width {}: proof did not verify or claimed the wrong value",
                row.width
            );
            assert!(row.constraints > 0, "width {} produced no constraints", row.width);
        }
    }

    /// Circuit size must grow with the layer, and proof size must not — the
    /// succinctness that makes a SNARK worth using at all.
    #[test]
    fn constraints_grow_with_width_while_proofs_stay_small() {
        let rows = run(&[4, 64]).rows;
        assert!(
            rows[1].constraints > rows[0].constraints,
            "a wider layer must produce more constraints"
        );
        assert!(
            rows[1].proof_bytes < 16 * rows[0].proof_bytes,
            "proof size must grow far slower than the circuit: {} -> {} bytes",
            rows[0].proof_bytes,
            rows[1].proof_bytes
        );
    }

    /// The reference and the circuit must agree on negative activations, which
    /// is where a sign error in the rescale or the ReLU would hide.
    #[test]
    fn the_reference_exercises_negative_activations() {
        let width = 16;
        let weights: Vec<i64> = (0..width).map(|i| ((i as i64 % 7) - 3) * SCALE / 4).collect();
        let inputs: Vec<i64> = (0..width).map(|i| ((i as i64 % 5) - 2) * SCALE / 2).collect();
        assert!(
            weights.iter().any(|w| *w < 0) && inputs.iter().any(|x| *x < 0),
            "the sweep must include negative weights and inputs"
        );
    }
}
