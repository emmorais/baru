//! Export an embedded MedMNIST model and its test subset as JSON, so a
//! reference implementation in another language evaluates *the same* network on
//! *the same* images.
//!
//! `examples/medmnist.baru` reads the weights through the `medmnist-model`
//! builtin and `tools/medmnist_reference.py` reads them through this file. Both
//! therefore multiply the identical `i64`s, which is the only way a
//! "Baru vs Python" timing says anything about the language rather than about
//! two different networks.
//!
//! ```sh
//! cargo run --release --example export_medmnist -- pneumonia /tmp/pneumonia.json
//! ```

use baru::bench::medmnist::load;
use serde_json::{json, Value};

fn main() {
    let mut args = std::env::args().skip(1);
    let dataset = args
        .next()
        .expect("usage: export_medmnist <dataset> <out.json>");
    let out = args
        .next()
        .expect("usage: export_medmnist <dataset> <out.json>");
    let t = load(&dataset).unwrap_or_else(|e| panic!("{e}"));

    let extra: Vec<Value> = t
        .extra_conv
        .iter()
        .map(|(w, b, in_c, in_h, in_w)| {
            json!({
                "kernels": w, "bias": b,
                "in_c": in_c, "in_h": in_h, "in_w": in_w,
                // Square kernel side, recovered from the row length rather than
                // stored — the same derivation the Lisp view makes.
                "k": ((w[0].len() / in_c) as f64).sqrt().round() as usize,
            })
        })
        .collect();
    let hidden: Vec<Value> = t
        .hidden
        .iter()
        .map(|(w, b)| json!({ "w": w, "b": b }))
        .collect();

    let doc = json!({
        "dataset": t.dataset,
        "task": t.task,
        "d": t.d,
        "in_c": t.in_c(),
        "shift": t.shift,
        "nclasses": t.dense_b.len(),
        "conv1": {
            "kernels": t.kernels,
            "bias": t.cbias,
            "k": ((t.kernels[0].len() / t.in_c()) as f64).sqrt().round() as usize,
        },
        "extra_conv": extra,
        "hidden": hidden,
        "dense_w": t.dense_w,
        "dense_b": t.dense_b,
        "images": t.images,
        "labels": t.labels,
        "full_test_accuracy": t.full_test_accuracy,
    });
    std::fs::write(&out, serde_json::to_string(&doc).expect("serialise")).expect("write");
    println!(
        "wrote {out} — {} ({} images, {}x{}x{}, shift {})",
        t.dataset,
        t.images.len(),
        t.in_c(),
        t.d,
        t.d,
        t.shift
    );
}
