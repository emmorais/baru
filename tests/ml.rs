// End-to-end tests for the model import/quantization pipeline
// (`src/ml/mod.rs`): NetworkSpec -> quantize -> QuantizedNetwork JSON ->
// Baru's `nn-load-model`/`nn-forward` builtins, both for a fully public
// model+input and for a public model evaluated over a private (shared)
// input — mirroring the existing `matvec` builtin's dual-path design
// exercised in tests/nn.rs.

use baru::interpreter::eval::eval_plain as eval;
use baru::interpreter::{parse, Participant, Types};
use baru::ml::{Activation, LayerSpec, NetworkSpec, QuantizedNetwork};

fn temp_json_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("baru_ml_test_{tag}_{}.json", std::process::id()))
}

/// Escape a Windows path for embedding in a Baru string literal (the
/// parser supports `\\` -> `\`, matching Rust's own escaping convention).
fn lisp_path_literal(path: &std::path::Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn toy_net() -> NetworkSpec {
    // A single 2-in/2-out ReLU layer with weights/biases exactly
    // representable in Q16.16 (powers of two / halves), so quantization
    // introduces zero rounding error and outputs are checked exactly.
    NetworkSpec {
        layers: vec![LayerSpec {
            weights: vec![vec![1.0, -1.0], vec![0.5, 0.5]],
            bias: vec![0.0, 0.0],
            activation: Activation::Relu,
        }],
    }
}

#[test]
fn test_nn_load_model_and_forward_public_input() {
    let net = toy_net();
    let (quantized, _report) = net.quantize(16, &[]);
    let path = temp_json_path("public");
    quantized.save_json(&path).unwrap();

    // pre = [1*3 + -1*1, 0.5*3 + 0.5*1] = [2.0, 2.0]; relu keeps both; sum = 4.0
    let src = format!(
        "(let ((layers (nn-load-model \"{}\"))
               (input (list (int->fixed 3) (int->fixed 1))))
           (fixed->int (vsum (nn-forward layers input))))",
        lisp_path_literal(&path)
    );
    let result = eval(&src);
    let _ = std::fs::remove_file(&path);
    assert_eq!(result, 4);
}

#[test]
fn test_nn_load_model_and_forward_private_input() {
    // Same network, but the input is a plain-integer share (the pixel/
    // feature values are private; the model weights stay public Q16.16
    // fixed constants) — the mixed Fixed-weight x Share-input path that
    // `vec_mul`/`vec_add`/`vec_relu` already handle for `dot`/`matvec`.
    let net = toy_net();
    let (quantized, _report) = net.quantize(16, &[]);
    let path = temp_json_path("private");
    quantized.save_json(&path).unwrap();

    let src = format!(
        "(let ((layers (nn-load-model \"{}\"))
               (x (share 3 32))
               (y (share 1 32)))
           (fixed->int (reconstruct (vsum (nn-forward layers (list x y))))))",
        lisp_path_literal(&path)
    );
    let result = eval(&src);
    let _ = std::fs::remove_file(&path);
    assert_eq!(result, 4);
}

#[test]
fn test_nn_forward_two_layers_chain_and_relu_zeroes_negatives() {
    // Layer 1: identity, drives one output negative; layer 2: relu clamps it.
    let net = NetworkSpec {
        layers: vec![
            LayerSpec {
                weights: vec![vec![1.0, 0.0], vec![0.0, -1.0]],
                bias: vec![0.0, 0.0],
                activation: Activation::Identity,
            },
            LayerSpec {
                weights: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                bias: vec![0.0, 0.0],
                activation: Activation::Relu,
            },
        ],
    };
    let (quantized, _report) = net.quantize(16, &[]);
    let path = temp_json_path("chain");
    quantized.save_json(&path).unwrap();

    // input [2, 3] -> layer1 -> [2, -3] -> layer2 (relu) -> [2, 0]
    let src = format!(
        "(let ((layers (nn-load-model \"{}\"))
               (input (list (int->fixed 2) (int->fixed 3))))
           (fixed->int (vsum (nn-forward layers input))))",
        lisp_path_literal(&path)
    );
    let result = eval(&src);
    let _ = std::fs::remove_file(&path);
    assert_eq!(result, 2);
}

#[test]
fn test_nn_load_model_rejects_non_16_scale_bits() {
    let net = toy_net();
    let (quantized, _report) = net.quantize(8, &[]);
    let path = temp_json_path("wrong_scale");
    quantized.save_json(&path).unwrap();

    let src = format!("(nn-load-model \"{}\")", lisp_path_literal(&path));
    let expr = parse(&src).expect("parse failed");
    let p = Participant::new(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.eval(&expr)));
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_err(),
        "loading a non-16-scale-bit model must panic, not silently misinterpret it"
    );
}

#[test]
fn test_nn_load_model_returns_inspectable_layer_structure() {
    // The layer list is plain Baru data — a program should be able to poke
    // at weights/bias/activation directly, not just call nn-forward.
    let net = toy_net();
    let (quantized, _report) = net.quantize(16, &[]);
    let path = temp_json_path("inspect");
    quantized.save_json(&path).unwrap();

    let src = format!("(nn-load-model \"{}\")", lisp_path_literal(&path));
    let expr = parse(&src).expect("parse failed");
    let p = Participant::new(0);
    let result = p.eval(&expr);
    let _ = std::fs::remove_file(&path);

    match result {
        Types::List(layers) => {
            assert_eq!(layers.len(), 1);
            match &layers[0] {
                Types::List(parts) => {
                    assert_eq!(parts.len(), 3);
                    assert!(matches!(&parts[2], Types::Str(s) if s == "relu"));
                    match &parts[0] {
                        Types::List(rows) => assert_eq!(rows.len(), 2),
                        other => panic!("expected weights list, got {:?}", other),
                    }
                }
                other => panic!("expected a 3-element layer list, got {:?}", other),
            }
        }
        other => panic!("expected a list of layers, got {:?}", other),
    }
}

#[test]
fn test_quantize_cli_roundtrip_with_validation_report() {
    let dir = std::env::temp_dir();
    let in_path = dir.join(format!("baru_ml_cli_in_{}.json", std::process::id()));
    let out_path = dir.join(format!("baru_ml_cli_out_{}.json", std::process::id()));
    let samples_path = dir.join(format!("baru_ml_cli_samples_{}.json", std::process::id()));

    toy_net().save_json(&in_path).unwrap();
    std::fs::write(&samples_path, "[[3.0, 1.0], [-2.0, 4.0]]").unwrap();

    let args = vec![
        "--in".to_string(),
        in_path.display().to_string(),
        "--out".to_string(),
        out_path.display().to_string(),
        "--scale-bits".to_string(),
        "16".to_string(),
        "--validate".to_string(),
        samples_path.display().to_string(),
    ];
    let result = baru::ml::run_cli(&args);

    let quantized = QuantizedNetwork::load_json(&out_path).ok();
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_file(&samples_path);

    assert!(result.is_ok(), "run_cli failed: {:?}", result.err());
    let quantized = quantized.expect("CLI must have written a loadable QuantizedNetwork");
    assert_eq!(quantized.scale_bits, 16);
    assert_eq!(quantized.layers.len(), 1);
}
