//! Model import & quantization: turn a trained network (plain floats, from
//! whatever framework produced it) into fixed-point data Baru can run
//! privately — replacing the ad hoc workflow in `examples/data/mnist_data.rs`
//! (weights hand-quantized once, pasted into a Rust literal array, never
//! reproducible) with a small, reusable pipeline:
//!
//! ```text
//! train elsewhere (numpy / sklearn / PyTorch / anything)
//!   -> export weights+biases to a plain JSON NetworkSpec (see below)
//!   -> `baru quantize` (this module): fixed-point + a measured error report
//!   -> quantized JSON
//!   -> Baru: (nn-load-model "path.json") + (nn-forward layers input)
//! ```
//!
//! # Format
//!
//! [`NetworkSpec`] is deliberately framework-agnostic: a flat JSON document
//! (`Vec<LayerSpec>`, each an `out_dim x in_dim` weight matrix, an `out_dim`
//! bias vector, and an activation tag). Any training stack can produce this
//! with a few lines — no ONNX/pickle/tensor-format parser is needed. Only
//! [`Activation::Relu`] and [`Activation::Identity`] are supported: Baru's
//! MPC path can only branch obliviously (mux, not exact nonlinearities), and
//! CKKS cannot do comparisons at all — richer activations (sigmoid/tanh via
//! polynomial approximation) are exactly the "crypto-friendly ML"
//! follow-up this module's docs point at, not solved here.
//!
//! # Quantization is measured, not assumed
//!
//! [`NetworkSpec::quantize`] doesn't just scale-and-round: it also runs both
//! the float and quantized forward passes on the same input(s) and reports
//! the actual error ([`QuantizationReport`]), so "a bit of precision lost,
//! but good enough for inference" is a number you check, not a hope.
//! `scale_bits` is a parameter specifically so this can be experimented
//! with — lower bits mean cheaper MPC comparisons/`a2b` and a smaller field
//! footprint, at the cost of more quantization error; this module makes that
//! trade-off visible instead of hiding it.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

const USAGE: &str = "\
Usage: baru quantize --in <model.json> --out <model.quant.json> [options]

Turn a floating-point NetworkSpec (a plain JSON export from whatever trained
it — see the schema in src/ml/mod.rs) into a fixed-point QuantizedNetwork
that Baru can load with (nn-load-model \"model.quant.json\").

Options:
  --in PATH        input NetworkSpec JSON (required)
  --out PATH       output QuantizedNetwork JSON (required)
  --scale-bits N   fixed-point scale, default 16 (Baru's Types::Fixed/
                   nn-load-model require exactly 16; pass a different value
                   only to study the accuracy/size trade-off with --validate)
  --validate PATH  JSON array of sample input vectors ([[f64, ...], ...]) to
                   measure quantization error against — the report is always
                   printed, never silently skipped, so \"good enough\" is a
                   number you see, not an assumption
  --help, -h       show this message
";

/// `baru quantize` CLI entry point (see `USAGE` for the contract).
pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut scale_bits: u32 = 16;
    let mut validate_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--in" => {
                i += 1;
                in_path = Some(args.get(i).ok_or("--in needs a path")?.clone());
            }
            "--out" => {
                i += 1;
                out_path = Some(args.get(i).ok_or("--out needs a path")?.clone());
            }
            "--scale-bits" => {
                i += 1;
                scale_bits = args
                    .get(i)
                    .ok_or("--scale-bits needs a value")?
                    .parse()
                    .map_err(|_| "--scale-bits needs a non-negative integer".to_string())?;
            }
            "--validate" => {
                i += 1;
                validate_path = Some(args.get(i).ok_or("--validate needs a path")?.clone());
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    let in_path = in_path.ok_or_else(|| format!("--in is required\n{USAGE}"))?;
    let out_path = out_path.ok_or_else(|| format!("--out is required\n{USAGE}"))?;

    let net = NetworkSpec::load_json(&in_path)?;
    net.validate()?;

    let samples: Vec<Vec<f64>> = match &validate_path {
        Some(p) => {
            let text = fs::read_to_string(p).map_err(|e| e.to_string())?;
            serde_json::from_str(&text).map_err(|e| e.to_string())?
        }
        None => Vec::new(),
    };

    let (quantized, report) = net.quantize(scale_bits, &samples);
    quantized.save_json(&out_path)?;

    println!(
        "wrote {out_path} ({} layer(s), {scale_bits} scale bits)",
        quantized.layers.len()
    );
    if samples.is_empty() {
        eprintln!(
            "warning: no --validate samples given — quantization error was NOT measured. \
             Pass --validate <samples.json> to check accuracy before trusting this model."
        );
    } else {
        println!(
            "quantization error over {} sample(s):",
            report.samples_checked
        );
        println!("  max abs error:  {:.6}", report.max_abs_error);
        println!("  mean abs error: {:.6}", report.mean_abs_error);
        match report.max_rel_error {
            Some(r) => println!("  max rel error:  {:.4}%", r * 100.0),
            None => println!("  max rel error:  n/a (all reference outputs were exactly zero)"),
        }
    }
    if scale_bits != 16 {
        println!(
            "note: --scale-bits {scale_bits} != 16 — this model can be inspected/measured but \
             cannot be loaded with (nn-load-model ...) yet, since Baru's Types::Fixed is a \
             fixed Q16.16 representation. Re-run with --scale-bits 16 to use it in Baru."
        );
    }
    Ok(())
}

/// A layer's activation. Deliberately small: only what every Baru backend
/// (MPC via mux, integer FHE via `cmpslt`+`mux`) can actually express
/// obliviously. CKKS cannot represent either — real-arithmetic layers must
/// stay `Identity` (see `bench-nn`'s CKKS runner, which skips
/// comparison-shaped workloads for exactly this reason).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Relu,
    Identity,
}

/// One dense layer in floating point, as exported from a training framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    /// `out_dim` rows of `in_dim` weights each.
    pub weights: Vec<Vec<f64>>,
    /// `out_dim` biases.
    pub bias: Vec<f64>,
    pub activation: Activation,
}

impl LayerSpec {
    pub fn in_dim(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }
    pub fn out_dim(&self) -> usize {
        self.weights.len()
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let pre: Vec<f64> = self
            .weights
            .iter()
            .zip(&self.bias)
            .map(|(row, b)| row.iter().zip(x).map(|(w, xi)| w * xi).sum::<f64>() + b)
            .collect();
        match self.activation {
            Activation::Identity => pre,
            Activation::Relu => pre.into_iter().map(|v| v.max(0.0)).collect(),
        }
    }
}

/// A full network: an ordered stack of layers, `layers[i].out_dim() ==
/// layers[i+1].in_dim()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpec {
    pub layers: Vec<LayerSpec>,
}

/// Error introduced by quantizing a [`NetworkSpec`], measured (not assumed)
/// by comparing the float and quantized forward passes on the same inputs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuantizationReport {
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    /// `None` when every reference output was exactly zero (relative error
    /// undefined) — reported as such rather than silently as `0.0`.
    pub max_rel_error: Option<f64>,
    pub scale_bits: u32,
    pub samples_checked: usize,
}

impl NetworkSpec {
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| e.to_string())
    }

    /// Validate that consecutive layers' dimensions actually chain.
    pub fn validate(&self) -> Result<(), String> {
        for (i, pair) in self.layers.windows(2).enumerate() {
            if pair[0].out_dim() != pair[1].in_dim() {
                return Err(format!(
                    "layer {i} out_dim ({}) != layer {} in_dim ({})",
                    pair[0].out_dim(),
                    i + 1,
                    pair[1].in_dim()
                ));
            }
        }
        Ok(())
    }

    /// Reference floating-point forward pass (the ground truth
    /// [`Self::quantize`]'s report is measured against).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut cur = input.to_vec();
        for layer in &self.layers {
            cur = layer.forward(&cur);
        }
        cur
    }

    /// Quantize every weight/bias to a `Q(scale_bits)` fixed-point integer
    /// (round-to-nearest; matches this crate's existing `FIXED_SCALE = 1 <<
    /// 16` convention when `scale_bits == 16`), and measure the resulting
    /// error on `sample_inputs` by comparing [`Self::forward`] against the
    /// quantized network's own fixed-point forward pass.
    pub fn quantize(
        &self,
        scale_bits: u32,
        sample_inputs: &[Vec<f64>],
    ) -> (QuantizedNetwork, QuantizationReport) {
        let scale = (1i64 << scale_bits) as f64;
        let q_layer = |l: &LayerSpec| QuantizedLayer {
            weights: l
                .weights
                .iter()
                .map(|row| row.iter().map(|&w| (w * scale).round() as i64).collect())
                .collect(),
            bias: l.bias.iter().map(|&b| (b * scale).round() as i64).collect(),
            activation: l.activation,
        };
        let quantized = QuantizedNetwork {
            scale_bits,
            layers: self.layers.iter().map(q_layer).collect(),
        };

        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut max_rel: Option<f64> = None;
        let mut n = 0usize;
        for input in sample_inputs {
            let expected = self.forward(input);
            let q_input: Vec<i64> = input.iter().map(|&v| (v * scale).round() as i64).collect();
            let got_fixed = quantized.forward(&q_input);
            let got: Vec<f64> = got_fixed.iter().map(|&v| v as f64 / scale).collect();
            for (e, g) in expected.iter().zip(&got) {
                let abs_err = (e - g).abs();
                max_abs = max_abs.max(abs_err);
                sum_abs += abs_err;
                n += 1;
                if *e != 0.0 {
                    let rel = abs_err / e.abs();
                    max_rel = Some(max_rel.map_or(rel, |m: f64| m.max(rel)));
                }
            }
        }
        let report = QuantizationReport {
            max_abs_error: max_abs,
            mean_abs_error: if n > 0 { sum_abs / n as f64 } else { 0.0 },
            max_rel_error: max_rel,
            scale_bits,
            samples_checked: sample_inputs.len(),
        };
        (quantized, report)
    }
}

/// One dense layer, quantized to `Q(scale_bits)` fixed-point integers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedLayer {
    pub weights: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
    pub activation: Activation,
}

impl QuantizedLayer {
    pub fn in_dim(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }
    pub fn out_dim(&self) -> usize {
        self.weights.len()
    }
}

/// A network quantized to fixed-point — what [`super::ml`] hands to Baru.
/// `forward` performs the *same* `(a*b) >> scale_bits` rescale-after-multiply
/// arithmetic as Baru's Q16.16 `Types::Fixed` path and the MPC/FHE backends,
/// so this is a true reference for what the cryptographic backends will
/// compute, not just an approximation of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedNetwork {
    pub scale_bits: u32,
    pub layers: Vec<QuantizedLayer>,
}

impl QuantizedNetwork {
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| e.to_string())
    }

    /// Fixed-point forward pass: `(a*b) >> scale_bits` per multiply-then-sum,
    /// then bias-add, then activation — the same rescaling convention as
    /// `Types::Fixed`, so behavior matches what a private (shared/encrypted)
    /// evaluation of the same network will produce.
    pub fn forward(&self, input: &[i64]) -> Vec<i64> {
        let mut cur = input.to_vec();
        for layer in &self.layers {
            let pre: Vec<i64> = layer
                .weights
                .iter()
                .zip(&layer.bias)
                .map(|(row, &b)| {
                    let acc: i128 = row
                        .iter()
                        .zip(&cur)
                        .map(|(&w, &x)| w as i128 * x as i128)
                        .sum();
                    ((acc >> self.scale_bits) as i64).wrapping_add(b)
                })
                .collect();
            cur = match layer.activation {
                Activation::Identity => pre,
                Activation::Relu => pre.into_iter().map(|v| v.max(0)).collect(),
            };
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny 2-layer network: 2 inputs -> 3 hidden (relu) -> 1 output.
    fn toy_net() -> NetworkSpec {
        NetworkSpec {
            layers: vec![
                LayerSpec {
                    weights: vec![vec![0.5, -0.25], vec![1.0, 1.0], vec![-0.75, 0.5]],
                    bias: vec![0.1, -0.2, 0.0],
                    activation: Activation::Relu,
                },
                LayerSpec {
                    weights: vec![vec![1.0, -1.0, 0.5]],
                    bias: vec![0.05],
                    activation: Activation::Identity,
                },
            ],
        }
    }

    #[test]
    fn validate_accepts_chained_dims_and_rejects_mismatch() {
        assert!(toy_net().validate().is_ok());
        let mut bad = toy_net();
        bad.layers[1].weights[0].push(1.0); // now expects in_dim=4, prev out_dim=3
        assert!(bad.validate().is_err());
    }

    #[test]
    fn quantization_report_is_small_for_a_well_scaled_network() {
        let net = toy_net();
        let samples = vec![vec![0.3, -0.6], vec![1.2, 0.4], vec![-0.5, -0.5]];
        let (_q, report) = net.quantize(16, &samples);
        // Q16.16 on small, well-scaled weights should be accurate to a tiny fraction.
        assert!(
            report.max_abs_error < 1e-3,
            "unexpectedly large error: {report:?}"
        );
        assert_eq!(report.samples_checked, 3);
    }

    #[test]
    fn lower_scale_bits_trade_accuracy_for_smaller_integers() {
        let net = toy_net();
        let samples = vec![vec![0.3, -0.6], vec![1.2, 0.4]];
        let (_q16, r16) = net.quantize(16, &samples);
        let (_q4, r4) = net.quantize(4, &samples);
        assert!(
            r4.max_abs_error > r16.max_abs_error,
            "coarser quantization (4 bits) should be less accurate than 16: {r4:?} vs {r16:?}"
        );
    }

    #[test]
    fn quantized_forward_matches_float_forward_within_report_bound() {
        let net = toy_net();
        let (q, report) = net.quantize(16, &[vec![0.3, -0.6]]);
        let scale = (1i64 << 16) as f64;
        let x_fixed: Vec<i64> = [0.3, -0.6]
            .iter()
            .map(|&v| (v * scale).round() as i64)
            .collect();
        let got: Vec<f64> = q
            .forward(&x_fixed)
            .iter()
            .map(|&v| v as f64 / scale)
            .collect();
        let expected = net.forward(&[0.3, -0.6]);
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() <= report.max_abs_error + 1e-9);
        }
    }

    #[test]
    fn json_roundtrip_preserves_structure() {
        let net = toy_net();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("baru_ml_test_{}.json", std::process::id()));
        net.save_json(&path).unwrap();
        let loaded = NetworkSpec::load_json(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.layers.len(), net.layers.len());
        assert_eq!(loaded.layers[0].out_dim(), net.layers[0].out_dim());
        assert_eq!(
            loaded.forward(&[0.3, -0.6]),
            net.forward(&[0.3, -0.6]),
            "roundtrip must be bit-identical for f64 JSON"
        );
    }

    #[test]
    fn quantized_json_roundtrip() {
        let net = toy_net();
        let (q, _report) = net.quantize(16, &[]);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("baru_ml_qtest_{}.json", std::process::id()));
        q.save_json(&path).unwrap();
        let loaded = QuantizedNetwork::load_json(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            loaded.forward(&[10_000, -5_000]),
            q.forward(&[10_000, -5_000])
        );
    }
}
