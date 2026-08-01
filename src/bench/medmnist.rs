//! `baru bench-medmnist` — private *diagnosis*: the same encrypted CNN
//! pipeline as [`crate::bench::mnist`], on real biomedical images.
//!
//! # Why this exists
//!
//! `bench-mnist` answers the ML question (what accuracy does the deployed
//! integer model reach?) on handwritten digits. Digits are not a medical
//! claim, and a reviewer will say so. **MedMNIST v2**
//! (<https://medmnist.com>, CC BY 4.0) distributes twelve 2D biomedical
//! datasets at 28×28 — *exactly* MNIST's shape — so the existing 784-pixel
//! pipeline runs on them unchanged and the artifact stops being about digits.
//!
//! Two datasets are embedded, chosen because they are grayscale (the conv
//! packing assumes one input channel) and because they stress different ends
//! of the classifier:
//!
//! | dataset | task | classes | test set |
//! |---|---|---|---|
//! | `pneumonia` | pediatric chest X-ray, pneumonia vs normal | 2 | 624 |
//! | `organa` | abdominal CT, which organ (axial views) | 11 | 17 778 |
//!
//! # What is measured
//!
//! Exactly what `bench-mnist` measures, and for the same reasons:
//!
//! - **Plaintext-integer accuracy** over the embedded test subset. Because
//!   quantisation is baked into [`CnnModel::forward`], this is the accuracy of
//!   the *deployed* integer model, not of a float model that would then lose
//!   something on the way down.
//! - **Encrypted accuracy** — the full hybrid (BFV ct×ct conv/dense, ABY3
//!   nonlinearities, zero bootstraps) on a subset. The hybrid is bit-exact
//!   against `forward`, so every encrypted prediction *equals* the plaintext
//!   one; the subset run demonstrates that on real medical images rather than
//!   asserting it.
//! - **Balanced accuracy, AUC, per-class recall and a Wilson interval**
//!   ([`crate::bench::metrics`]). Accuracy alone is not enough to judge a
//!   *medical* classifier and on these weights actively misleads: see the
//!   sensitivity/specificity note below. AUC also puts this model on the
//!   metric MedMNIST's own tables lead with, which accuracy cannot.
//!
//! The full-test-set number in each weight file was measured offline by
//! `tools/train_medmnist.py` on the identical integer weights, so the
//! in-repo subset can be read against it.
//!
//! # Results (2026-08-01, Apple M1 Pro / 8 cores, 20 images per config)
//!
//! | | pneumonia | organa |
//! |---|---|---|
//! | integer accuracy, 300 embedded images | **85.00%** | **84.00%** |
//! | integer accuracy, full test set | 84.78% (624) | 81.18% (17 778) |
//! | 95% CI on the full-test figure | [81.7, 87.4] | [80.6, 81.8] |
//! | balanced accuracy / AUC, **full test** | 80.64% / **0.9314** | 80.49% / **0.9590** |
//! | sensitivity / specificity, full test | 97.18% / **64.10%** | — |
//! | published ResNet-18 (28×28), ACC / AUC | 0.854 / 0.944 | 0.935 / 0.997 |
//! | majority-class baseline | 62.50% | 18.48% |
//! | **encrypted == plaintext diagnosis** | **300/300** | **300/300** |
//! | FHE bootstraps | 0 | 0 |
//! | plaintext modulus | 2^20 | 2^22 |
//! | latency/diagnosis, `a2b` | 1.63 s | 1.74 s |
//! | + `--gc` | 0.88 s | 0.90 s |
//! | + `--precompute`, single core | 0.435 s | 0.460 s |
//! | + **4 NTT threads** (online) | **0.268 s** | **0.291 s** |
//! | offline, **per query** (mask pool) | 0.14 s/image | 0.14 s/image |
//! | online at WAN (`--rtt-ms 20 --bandwidth-mbps 1000`) | 2.65 s | 2.81 s |
//!
//! Cost of one encrypted pneumonia diagnosis, `--gc --precompute`: **268 ms**
//! online on four cores (435 ms single-core — `BARU_NTT_THREADS=1`), of which
//! FHE is 99 ms (mul 44, decrypt 45, re-encrypt 10) and the MPC segment
//! **168 ms (63%)** (import 22, ReLU 38, pool 69, argmax 8), over 10.1 MB of
//! party-to-party traffic. At RTT 20 ms / 1 Gbit/s it is 2.65 s. Garbling is
//! worth 2.0× on the MPC segment at that latency — a constant-round protocol
//! pays where rounds cost.
//!
//! The threading lives in the multi-modular NTT (independent primes, plus a
//! pure-map CRT reconstruction), is bit-identical to the serial path, and is
//! switchable — so the core-count-independent figure stays quotable.
//!
//! Quote the online figure together with the per-query offline one: the mask
//! pool must be regenerated for every query and only moves off the critical
//! path, so 413 ms is the latency, not the work.
//!
//! The encrypted==plaintext row is now **every embedded image**, not a
//! 3-image sample: 600 diagnoses, 600 agreements, 0 bootstraps. It was run
//! twice — once before the ReLU/max-pool commutation and NTT threading, once
//! after — and both passes agree on all 600, with encrypted-correct equal to
//! plaintext-correct (255 and 252). That is the check that those two
//! optimisations are exact rather than merely close: a rounding-level
//! difference would show up as a changed prediction somewhere in 600.
//!
//! Two things to read off that table. **Quantisation is free**: the integer
//! net matches its float parent to within 0.2 points, agreeing with it on
//! 99.7% (pneumonia) / 98.7% (organa) of test predictions — so the
//! fixed-point scheme the FHE layer demands costs no accuracy, and neither
//! does the encryption, which reproduces the plaintext diagnosis bit-exactly.
//! The *cost* is entirely elsewhere — and as of 2026-07-28 it moved.
//!
//! **The cost centre is now the FHE arithmetic, not the MPC nonlinearities.**
//! A diagnosis used to take ~33 s, of which the scheme-switch *import* alone
//! was ~19 s: importing an opened layer ran three blocking input-sharing
//! rounds **per activation**, so a 5408-wide convolution cost 16 224 round
//! trips (`import_raw_batch` in [`crate::bench::cnn`] now does it in three).
//! Then the nonlinearities were garbled (1.7×), and then the *other* half of
//! the switch — encrypting its query-independent masks, 48% of all FHE time —
//! moved to an offline phase (1.7×). A diagnosis is now **2.90 s online**,
//! 73% of it FHE. Anyone reading this module for where to optimise next
//! should read that number and go to `fhe::bfv`, not here.
//!
//! **The model, not the crypto, is the accuracy ceiling.** Published
//! ResNet-18 at 28×28 reaches 0.854 / 0.935 ACC on these two datasets; this
//! net is a 2-conv, **53 k-parameter** classifier and it lands at 0.848 /
//! 0.812. On pneumonia that is *statistically indistinguishable* — 84.78% on
//! 624 images carries a 95% Wilson interval of [81.7, 87.4], which contains
//! 0.854 — so the honest claim is indistinguishability, not parity. On organa
//! the 12-point gap is real: [80.6, 81.8] on 17 778 images excludes 0.935.
//!
//! Three limits set that ceiling, and the noise budget is only the third.
//! *Spatially*, 3×3 valid convolution plus floor-pooling takes 28 → 13 → 5 → 1,
//! so at most three convs fit. *By packing*, `2·in_c·H·W ≲ n` means the first
//! conv may emit at most **12** channels before the second conv's 13×13 input
//! passes `n = 4096` — this, not the noise budget, is what stops the model
//! getting wider, and it relaxes at `n = 8192`. Only then does the depth-1
//! budget bind, on activation magnitude.
//!
//! **Accuracy alone misreads this classifier, and the benchmark now says so.**
//! PneumoniaMNIST's test set is ~60% positive, and over the full 624 images the
//! model scores **97.2% sensitivity against 64.1% specificity** — it earns its
//! 84.78% largely by over-calling disease. Balanced accuracy is **80.64%**, and
//! AUC **0.9314** against the published 0.944. Organa's macro one-vs-rest AUC
//! is **0.9590** against 0.997, with kidney-left and spleen the weakest classes
//! at 64% recall each. None of this was visible while only accuracy was
//! reported, and none of it is a cryptographic result — it is what a
//! 53 k-parameter model does — but quoting accuracy alone overstates it.
//!
//! Note also that organa's validation accuracy is 94.7% against 81.2% on test
//! — the official split separates patients, and that gap is a property of the
//! dataset, not of the quantisation (the gain was chosen on validation, never
//! on test).
//!
//! Honest scope, to be repeated in the paper: MedMNIST is pre-processed and
//! small, and clinical imaging is 512³ volumes. The claim here is about *cost
//! structure*, and scaling past 28×28 is an argument the cost model has to
//! make analytically.
//!
//! Weights are regenerated by `tools/train_medmnist.py <npz> <out.rs>`
//! (see that script for the quantisation derivation).

use std::time::Instant;

use serde::Serialize;

use crate::bench::cnn::{run_cnn_hybrid_in_session, CnnModel, CnnSession, ConvLayer, Nonlin};
use crate::bench::comm::NetworkProfile;
use crate::bench::metrics::ClassificationMetrics;

/// One conv layer past the first, exactly as the generated weight files hand
/// it over: `(kernels[out_c][in_c·K·K], bias[out_c], in_c, in_h, in_w)`.
pub type ExtraConv = (Vec<Vec<i64>>, Vec<i64>, usize, usize, usize);

/// A trained, quantised CNN plus the real test images it was measured on, in
/// the form the generated weight files hand over. One struct per dataset,
/// built by the `trained()` function each `examples/data/*_cnn.rs` emits.
pub struct TrainedCnn {
    pub dataset: &'static str,
    pub task: &'static str,
    pub class_names: Vec<&'static str>,
    /// Side length of the square, single-channel input the model expects.
    pub d: usize,
    pub kernels: Vec<Vec<i64>>,
    pub cbias: Vec<i64>,
    /// The conv layers *after* the first, in order — the first is kept apart
    /// as `kernels`/`cbias` because it is the single-channel one the Cheetah
    /// packing sees. Each is `(kernels[out_c][in_c·K·K], bias, in_c, in_h,
    /// in_w)`. Arbitrary length: the trainer's `--conv` decides how many, and
    /// every one is just another FHE conv + MPC nonlinearity + scheme switch.
    pub extra_conv: Vec<ExtraConv>,
    pub hidden: Vec<(Vec<Vec<i64>>, Vec<i64>)>,
    pub dense_w: Vec<Vec<i64>>,
    pub dense_b: Vec<i64>,
    pub shift: u32,
    pub images: Vec<Vec<i64>>,
    pub labels: Vec<u8>,
    pub full_test_accuracy: f64,
    /// Accuracy of the float net these weights were quantised from, over the
    /// same full test set — the baseline that says whether quantisation cost
    /// anything. `None` in weight files generated before the trainer recorded
    /// it; regenerate the file to populate it rather than filling it in by
    /// hand, which is exactly the drift these generated files exist to avoid.
    pub float_test_accuracy: Option<f64>,
    /// Balanced accuracy and AUC over the FULL test set, measured by the
    /// trainer. The benchmark can only compute these on the embedded subset,
    /// and these are the numbers that belong in a paper. `None` in weight
    /// files generated before the trainer measured them.
    pub full_balanced_accuracy: Option<f64>,
    pub full_auc: Option<f64>,
    pub majority_baseline: f64,
    pub full_test_n: usize,
}

impl TrainedCnn {
    /// Build the runnable model. `nbits` is calibrated from the observed
    /// activation range over the embedded images, exactly as the synthetic
    /// model does, so it fits the depth-1 noise budget.
    pub fn model(&self) -> CnnModel {
        CnnModel::from_trained(
            self.d,
            self.kernels.clone(),
            self.cbias.clone(),
            self.extra_conv
                .iter()
                .map(|(w, b, in_c, in_h, in_w)| ConvLayer {
                    kernels: w.clone(),
                    cbias: b.clone(),
                    in_c: *in_c,
                    in_h: *in_h,
                    in_w: *in_w,
                })
                .collect(),
            self.hidden.clone(),
            self.dense_w.clone(),
            self.dense_b.clone(),
            self.shift,
            &self.images,
        )
    }
}

#[path = "../../examples/data/pneumonia_cnn.rs"]
#[allow(dead_code)] // the generated files export descriptive consts we don't all read
mod pneumonia;

#[path = "../../examples/data/organa_cnn.rs"]
#[allow(dead_code)]
mod organa;

/// A dataset's `--dataset` name paired with the loader its generated weight
/// file exports.
pub type Embedded = (&'static str, fn() -> TrainedCnn);

/// The embedded datasets, by the name `--dataset` takes.
pub const DATASETS: &[Embedded] = &[
    ("pneumonia", pneumonia::trained),
    ("organa", organa::trained),
];

pub fn load(name: &str) -> Result<TrainedCnn, String> {
    DATASETS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, f)| f())
        .ok_or_else(|| {
            let names: Vec<&str> = DATASETS.iter().map(|(n, _)| *n).collect();
            format!("unknown dataset {name:?}; known: {}", names.join(", "))
        })
}

#[derive(Debug, Serialize)]
pub struct MedMnistReport {
    pub dataset: String,
    pub task: String,
    /// The layer stack, read off this dataset's trained weights. Per-report
    /// rather than a single banner: each dataset carries its own weight file,
    /// so with depth free they need not share an architecture.
    pub architecture: String,
    pub class_names: Vec<String>,
    pub nclasses: usize,
    pub n_test: usize,
    pub plaintext_correct: usize,
    pub plaintext_accuracy: f64,
    /// Per-class recall, confusion, balanced accuracy, AUC and a Wilson
    /// interval over the embedded subset. Accuracy alone cannot distinguish a
    /// balanced classifier from one that over-predicts the majority class,
    /// which on an imbalanced medical set is the difference that matters.
    pub metrics: ClassificationMetrics,
    pub full_test_accuracy: f64,
    /// Full-test balanced accuracy and AUC, when the weight file records them.
    pub full_balanced_accuracy: Option<f64>,
    pub full_auc: Option<f64>,
    /// The float parent's accuracy on the same test set, when the weight file
    /// records it — the baseline that says whether quantisation cost anything.
    pub float_test_accuracy: Option<f64>,
    pub full_test_n: usize,
    pub majority_baseline: f64,
    pub hybrid_evaluated: usize,
    pub hybrid_matches_plaintext: usize,
    pub hybrid_correct: usize,
    pub bootstraps: u64,
    pub log2_t: u32,
    /// Wall-clock for the whole command, plaintext scoring pass included.
    pub ms: f64,
    /// Which protocol evaluated the MPC nonlinearities.
    pub nonlin: String,
    /// Mean wall-clock of **one encrypted diagnosis**, and its split. This —
    /// not [`Self::ms`] — is the deployable latency: `ms` also contains the
    /// 300-image plaintext scoring pass, which no deployment runs.
    pub per_image_ms: f64,
    pub per_image_fhe_ms: f64,
    pub per_image_mpc_ms: f64,
    /// Per-image breakdown of the MPC segment, averaged over the encrypted runs.
    pub per_image_import_ms: f64,
    pub per_image_relu_ms: f64,
    pub per_image_pool_ms: f64,
    pub per_image_argmax_ms: f64,
    /// Party-to-party messages and bytes for one encrypted diagnosis.
    pub per_image_messages: f64,
    pub per_image_bytes: f64,
    /// Of `per_image_bytes`, the ABY3 ring resharing — encoded `Vec<u128>`, so
    /// 16 B per element whatever `nbits` is. See `CommStats::ring_bytes`.
    pub per_image_ring_bytes: f64,
    /// Of `per_image_bytes`, the garbled tables and labels — already compact.
    pub per_image_gc_bytes: f64,
    /// Query-independent work per image, excluded from `per_image_ms`. The
    /// sum of the two fields below.
    pub per_image_offline_ms: f64,
    /// The per-SESSION share: `(keygen + encrypted model) / images`. Paid once
    /// for the whole batch, so this shrinks as the batch grows — real
    /// amortization, not a latency shift.
    pub per_image_offline_session_ms: f64,
    /// The per-QUERY part: the mask pool, rebuilt for every image because masks
    /// must be fresh. Flat in batch size.
    pub per_image_offline_query_ms: f64,
    /// Per-image FHE sub-split — which of the FHE operations the online time
    /// is actually in. `weights` is query-independent and not yet preprocessed.
    pub per_image_weights_ms: f64,
    pub per_image_mul_ms: f64,
    pub per_image_maskenc_ms: f64,
    pub per_image_decrypt_ms: f64,
    pub per_image_reencrypt_ms: f64,
    pub precomputed_masks: bool,
}

pub fn run(
    name: &str,
    hybrid_n: usize,
    profile: NetworkProfile,
    nonlin: Nonlin,
    precompute: bool,
) -> Result<MedMnistReport, String> {
    let data = load(name)?;
    let model = data.model();
    let t0 = Instant::now();

    // Plaintext-integer scoring over every embedded test image. Logits rather
    // than the argmax alone, because AUC is a property of the ranking and is
    // invisible to a benchmark that only ever sees the decision.
    let logits: Vec<Vec<i64>> = data
        .images
        .iter()
        .map(|img| model.forward_logits(img).0)
        .collect();
    let preds: Vec<usize> = logits.iter().map(|l| crate::gadgets::argmax_ref(l)).collect();
    let metrics = ClassificationMetrics::compute(&logits, &data.labels, model.nclasses());
    let plaintext_correct = metrics.correct;

    // Encrypted accuracy on a subset: each hybrid prediction must equal the
    // plaintext one (bit-exact), and we count how many hit the true label.
    let hybrid_n = hybrid_n.min(data.images.len());
    let (mut matches, mut hybrid_correct, mut bootstraps) = (0usize, 0usize, 0u64);
    let mut tot = CnnTotals::default();
    // ONE session for the whole batch — this is the point of precomputing the
    // model rather than the masks. Keygen and the encrypted weights are paid
    // once here and reused by every image below, so their per-image share falls
    // as the batch grows. The mask pool is still rebuilt inside each run,
    // because masks must be fresh per query.
    let session = CnnSession::new(&model, precompute);
    for ((img, &plain), &label) in data
        .images
        .iter()
        .zip(&preds)
        .zip(&data.labels)
        .take(hybrid_n)
    {
        let run = run_cnn_hybrid_in_session(&session, &model, img, profile, nonlin, precompute);
        if run.class == plain {
            matches += 1;
        }
        if run.class == label as usize {
            hybrid_correct += 1;
        }
        bootstraps += run.bootstraps;
        tot.add(&run);
    }
    let per = tot.mean(hybrid_n);

    Ok(MedMnistReport {
        dataset: data.dataset.to_string(),
        task: data.task.to_string(),
        architecture: model.architecture(),
        class_names: data.class_names.iter().map(|s| s.to_string()).collect(),
        nclasses: model.nclasses(),
        n_test: data.images.len(),
        plaintext_correct,
        plaintext_accuracy: plaintext_correct as f64 / data.images.len() as f64,
        metrics,
        full_test_accuracy: data.full_test_accuracy,
        full_balanced_accuracy: data.full_balanced_accuracy,
        full_auc: data.full_auc,
        float_test_accuracy: data.float_test_accuracy,
        full_test_n: data.full_test_n,
        majority_baseline: data.majority_baseline,
        hybrid_evaluated: hybrid_n,
        hybrid_matches_plaintext: matches,
        hybrid_correct,
        bootstraps,
        log2_t: model.nbits as u32,
        ms: t0.elapsed().as_secs_f64() * 1000.0,
        nonlin: nonlin.label().to_string(),
        per_image_ms: per.total_ms,
        per_image_fhe_ms: per.fhe_ms,
        per_image_mpc_ms: per.mpc_ms,
        per_image_import_ms: per.import_ms,
        per_image_relu_ms: per.relu_ms,
        per_image_pool_ms: per.pool_ms,
        per_image_argmax_ms: per.argmax_ms,
        per_image_messages: per.messages,
        per_image_bytes: per.bytes,
        per_image_ring_bytes: per.ring_bytes,
        per_image_gc_bytes: per.gc_bytes,
        per_image_offline_ms: session.offline_ms() / hybrid_n as f64 + per.offline_query_ms,
        per_image_offline_session_ms: session.offline_ms() / hybrid_n as f64,
        per_image_offline_query_ms: per.offline_query_ms,
        per_image_weights_ms: per.weights_ms,
        per_image_mul_ms: per.mul_ms,
        per_image_maskenc_ms: per.maskenc_ms,
        per_image_decrypt_ms: per.decrypt_ms,
        per_image_reencrypt_ms: per.reencrypt_ms,
        precomputed_masks: precompute,
    })
}

/// One cell of the crossover sweep: a network condition, a nonlinearity, and
/// what that combination cost per encrypted diagnosis.
pub struct SweepCell {
    pub rtt_ms: u64,
    pub bandwidth_mbps: f64,
    pub nonlin: Nonlin,
    pub ms: f64,
    pub messages: f64,
    pub bytes: f64,
    /// Every cell recomputes the same diagnosis; if one disagrees with the
    /// plaintext argmax the whole row is meaningless, so it is carried here
    /// rather than assumed.
    pub matches_plaintext: bool,
}

/// Sweep network conditions across nonlinearity protocols to find where the
/// cheaper-in-bytes route overtakes the cheaper-in-time one.
///
/// This is the measurement that justifies Baru *choosing* a nonlinearity rather
/// than fixing one, so it takes care to be a fair comparison: **one** model and
/// **one** [`CnnSession`] for the entire grid, so keygen and the encrypted
/// weights are identical across cells and only the protocol and the network
/// vary. The plaintext reference is computed once and every cell is checked
/// against it — a latency comparison between routes computing different things
/// would be worthless.
pub fn sweep_crossover(
    name: &str,
    hybrid_n: usize,
    rtts: &[u64],
    bws: &[f64],
    protocols: &[Nonlin],
) -> Result<Vec<SweepCell>, String> {
    let data = load(name)?;
    let model = data.model();
    let hybrid_n = hybrid_n.min(data.images.len()).max(1);
    let plain: Vec<usize> = data
        .images
        .iter()
        .take(hybrid_n)
        .map(|img| crate::gadgets::argmax_ref(&model.forward_logits(img).0))
        .collect();
    let session = CnnSession::new(&model, true);

    let mut cells = Vec::new();
    for &rtt in rtts {
        for &bw in bws {
            let profile = NetworkProfile::fixed_latency(std::time::Duration::from_micros(
                rtt * 500,
            ))
            .with_bandwidth(bw * 1e6 / 8.0);
            for &p in protocols {
                let mut tot = CnnTotals::default();
                let mut ok = true;
                for (img, &want) in data.images.iter().take(hybrid_n).zip(&plain) {
                    let run = run_cnn_hybrid_in_session(&session, &model, img, profile, p, true);
                    ok &= run.class == want;
                    tot.add(&run);
                }
                let per = tot.mean(hybrid_n);
                cells.push(SweepCell {
                    rtt_ms: rtt,
                    bandwidth_mbps: bw,
                    nonlin: p,
                    ms: per.total_ms,
                    messages: per.messages,
                    bytes: per.bytes,
                    matches_plaintext: ok,
                });
            }
        }
    }
    Ok(cells)
}

/// Print the sweep as a latency grid, then name the bandwidth at which the
/// winner changes for each RTT — the number the whole experiment is for.
fn print_sweep(cells: &[SweepCell], rtts: &[u64], bws: &[f64], protocols: &[Nonlin]) {
    println!("crossover sweep — ms per encrypted diagnosis, * marks the winner\n");
    let mismatched = cells.iter().filter(|c| !c.matches_plaintext).count();
    if mismatched > 0 {
        println!("  ⚠️  {mismatched} cell(s) disagreed with the plaintext argmax — DO NOT quote\n");
    }
    let at = |r: u64, b: f64, p: Nonlin| {
        cells
            .iter()
            .find(|c| c.rtt_ms == r && c.bandwidth_mbps == b && c.nonlin == p)
            .map(|c| c.ms)
    };
    for &rtt in rtts {
        println!("  RTT {rtt} ms");
        print!("    {:<10}", "Mbps");
        for p in protocols {
            print!("{:>12}", p.short());
        }
        println!();
        for &bw in bws {
            print!("    {bw:<10.0}");
            let best = protocols
                .iter()
                .filter_map(|&p| at(rtt, bw, p))
                .fold(f64::INFINITY, f64::min);
            for &p in protocols {
                match at(rtt, bw, p) {
                    Some(ms) if ms == best => print!("{:>12}", format!("*{ms:.0}")),
                    Some(ms) => print!("{ms:>12.0}"),
                    None => print!("{:>12}", "-"),
                }
            }
            println!();
        }
        // The crossover itself: the bracket in which the winner changes.
        // Reported as the grid interval actually measured, not interpolated
        // into a false-precision single number.
        if protocols.len() == 2 {
            let winner = |bw: f64| match (at(rtt, bw, protocols[0]), at(rtt, bw, protocols[1])) {
                (Some(a), Some(b)) => Some(if a <= b { protocols[0] } else { protocols[1] }),
                _ => None,
            };
            let mut sorted: Vec<f64> = bws.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let flip = sorted
                .windows(2)
                .find(|w| winner(w[0]) != winner(w[1]) && winner(w[1]).is_some());
            match flip {
                Some(w) => println!(
                    "    → crossover between {:.0} and {:.0} Mbps: {} wins above, {} below\n",
                    w[1],
                    w[0],
                    winner(w[0]).map(|p| p.short()).unwrap_or("?"),
                    winner(w[1]).map(|p| p.short()).unwrap_or("?"),
                ),
                None => println!(
                    "    → no crossover in this range: {} wins throughout\n",
                    winner(sorted[0]).map(|p| p.short()).unwrap_or("?")
                ),
            }
        } else {
            println!();
        }
    }
}

/// Running totals over the encrypted runs, so the report can quote a mean
/// per-image cost rather than a sum that silently depends on `--hybrid N`.
#[derive(Default)]
struct CnnTotals {
    total_ms: f64,
    fhe_ms: f64,
    mpc_ms: f64,
    import_ms: f64,
    relu_ms: f64,
    pool_ms: f64,
    argmax_ms: f64,
    messages: f64,
    bytes: f64,
    ring_bytes: f64,
    gc_bytes: f64,
    offline_query_ms: f64,
    weights_ms: f64,
    mul_ms: f64,
    maskenc_ms: f64,
    decrypt_ms: f64,
    reencrypt_ms: f64,
}

impl CnnTotals {
    fn add(&mut self, r: &crate::bench::cnn::CnnRun) {
        self.total_ms += r.total_ms;
        self.fhe_ms += r.fhe_ms;
        self.mpc_ms += r.mpc_ms;
        self.import_ms += r.phases.import_ms;
        self.relu_ms += r.phases.relu_ms;
        self.pool_ms += r.phases.pool_ms;
        self.argmax_ms += r.phases.argmax_ms;
        self.messages += r.comm.messages as f64;
        self.ring_bytes += r.comm.ring_bytes as f64;
        self.gc_bytes += r.comm.gc_bytes as f64;
        self.bytes += r.comm.bytes as f64;
        self.offline_query_ms += r.offline_query_ms;
        self.weights_ms += r.fhe_phases.weights_ms;
        self.mul_ms += r.fhe_phases.mul_ms;
        self.maskenc_ms += r.fhe_phases.mask_ms;
        self.decrypt_ms += r.fhe_phases.decrypt_ms;
        self.reencrypt_ms += r.fhe_phases.reencrypt_ms;
    }

    fn mean(self, n: usize) -> Self {
        if n == 0 {
            return Self::default();
        }
        let d = n as f64;
        Self {
            total_ms: self.total_ms / d,
            fhe_ms: self.fhe_ms / d,
            mpc_ms: self.mpc_ms / d,
            import_ms: self.import_ms / d,
            relu_ms: self.relu_ms / d,
            pool_ms: self.pool_ms / d,
            argmax_ms: self.argmax_ms / d,
            messages: self.messages / d,
            bytes: self.bytes / d,
            ring_bytes: self.ring_bytes / d,
            gc_bytes: self.gc_bytes / d,
            offline_query_ms: self.offline_query_ms / d,
            weights_ms: self.weights_ms / d,
            mul_ms: self.mul_ms / d,
            maskenc_ms: self.maskenc_ms / d,
            decrypt_ms: self.decrypt_ms / d,
            reencrypt_ms: self.reencrypt_ms / d,
        }
    }
}

const USAGE: &str = "usage: baru bench-medmnist [options]

Private diagnosis: the encrypted CNN pipeline on real biomedical images
(MedMNIST v2, 28x28, CC BY 4.0) instead of handwritten digits.

options:
  --dataset NAME  pneumonia (chest X-ray, 2 classes) | organa (abdominal CT,
                  11 classes) | all   (default pneumonia)
  --hybrid N      run the full ENCRYPTED pipeline on the first N test images
                  (default 5); the rest are scored with the bit-identical
                  plaintext-integer forward pass
  --gc            evaluate the MPC nonlinearities with 3-party garbled
                  circuits instead of batched bit decomposition (same
                  function, bit-exact; constant-round)
  --both          report both nonlinearity protocols
  --precompute    generate the query-independent scheme-switch masks in an
                  offline phase, so the reported latency is ONLINE latency
  --rtt-ms N            round-trip latency between parties (default 0)
  --bandwidth-mbps X    link bandwidth in Mbit/s (default: unlimited)
  --sweep         measure the CROSSOVER: run every (rtt, bandwidth) pair over
                  the gc and falcon nonlinearities from one shared session and
                  report which wins where. Add --both to include a2b.
  --rtt-list A,B        sweep grid, ms          (default 0,20)
  --bw-list A,B         sweep grid, Mbit/s      (default 1000,100,60,40,30,20,15,10)
  --json          machine-readable output
  -h, --help      this message
";

fn print_report(r: &MedMnistReport) {
    println!(
        "{} — {}  ({} classes)",
        r.dataset.to_uppercase(),
        r.task,
        r.nclasses
    );
    println!("  architecture               : {}", r.architecture);
    let m = &r.metrics;
    println!(
        "  plaintext-integer accuracy : {}/{} = {:.2}%  95% CI [{:.1}, {:.1}]  (in-repo subset)",
        r.plaintext_correct,
        r.n_test,
        r.plaintext_accuracy * 100.0,
        m.accuracy_ci.0 * 100.0,
        m.accuracy_ci.1 * 100.0
    );
    // Balanced accuracy is printed next to accuracy, never instead of it: the
    // gap between the two IS the imbalance, and hiding either one hides it.
    println!(
        "  balanced accuracy          : {:.2}%  (mean per-class recall, subset)",
        m.balanced_accuracy * 100.0
    );
    match m.auc {
        Some(a) if r.nclasses == 2 => {
            println!("  AUC                        : {a:.4}  (subset)")
        }
        Some(a) => println!("  AUC (macro one-vs-rest)    : {a:.4}  (subset)"),
        None => println!("  AUC                        : undefined (a class has no support)"),
    }
    if let (Some(sens), Some(spec)) = (m.sensitivity, m.specificity) {
        let tp = m.confusion[1][1];
        let fn_ = m.confusion[1][0];
        let tn = m.confusion[0][0];
        let fp = m.confusion[0][1];
        println!(
            "  sensitivity / specificity  : {:.2}% / {:.2}%   (TP {tp} FN {fn_} TN {tn} FP {fp})",
            sens * 100.0,
            spec * 100.0
        );
        // The asymmetry is the finding, so it is stated rather than left for
        // the reader to derive from four counts.
        if (sens - spec).abs() > 0.15 {
            let (weak, n_missed) = if sens < spec {
                ("misses disease", fn_)
            } else {
                ("over-calls disease", fp)
            };
            println!(
                "    ^ SKEWED by {:.0} points — the model {weak} ({n_missed} of {} in that class); \
                 read balanced accuracy, not accuracy",
                (sens - spec).abs() * 100.0,
                if sens < spec { m.per_class_n[1] } else { m.per_class_n[0] }
            );
        }
    }
    // Per-class recall, worst first: on an 11-way task a single collapsed
    // class is invisible in the mean but is exactly what a clinician would
    // care about.
    if r.nclasses > 2 {
        let mut by_recall: Vec<usize> = (0..r.nclasses).filter(|&c| m.per_class_n[c] > 0).collect();
        by_recall.sort_by(|&a, &b| m.per_class_recall[a].total_cmp(&m.per_class_recall[b]));
        let worst: Vec<String> = by_recall
            .iter()
            .take(3)
            .map(|&c| {
                format!(
                    "{} {:.0}% (n={})",
                    r.class_names.get(c).map(|s| s.as_str()).unwrap_or("?"),
                    m.per_class_recall[c] * 100.0,
                    m.per_class_n[c]
                )
            })
            .collect();
        println!("  weakest classes (recall)   : {}", worst.join(", "));
    }
    println!(
        "  full test set ({:>6})     : {:.2}%  (identical weights, measured offline)",
        r.full_test_n,
        r.full_test_accuracy * 100.0
    );
    // Full-test balanced accuracy / AUC come from the trainer, which is the
    // only side that sees the whole test set. Absent until a file is
    // regenerated — printed as a gap, never inferred from the subset.
    match (r.full_balanced_accuracy, r.full_auc) {
        (None, None) => println!(
            "    (full-test balanced accuracy and AUC not recorded — regenerate \
             the weight file to measure them)"
        ),
        (bal, a) => {
            let bal = bal.map_or("—".to_string(), |b| format!("{:.2}%", b * 100.0));
            let a = a.map_or("—".to_string(), |v| format!("{v:.4}"));
            println!("    full-test balanced accuracy {bal}, AUC {a}");
        }
    }
    // Only printed when the weight file recorded it: the point of the line is
    // that quantisation is free, and an absent number must read as "not
    // measured" rather than be quietly filled in from somewhere else.
    if let Some(f) = r.float_test_accuracy {
        println!(
            "  float parent, same net     : {:.2}%  (quantisation cost: {:+.2} points)",
            f * 100.0,
            (r.full_test_accuracy - f) * 100.0
        );
    }
    println!(
        "  majority-class baseline    : {:.2}%",
        r.majority_baseline * 100.0
    );
    println!("  ENCRYPTED pipeline on {} images:", r.hybrid_evaluated);
    println!(
        "    matches plaintext exactly : {}/{}  (bit-exact hybrid ⇒ same diagnosis)",
        r.hybrid_matches_plaintext, r.hybrid_evaluated
    );
    println!(
        "    correct vs true label     : {}/{}",
        r.hybrid_correct, r.hybrid_evaluated
    );
    println!("    FHE bootstraps            : {}", r.bootstraps);
    println!("    plaintext modulus         : 2^{}", r.log2_t);
    println!(
        "  cost of ONE encrypted diagnosis ({}{}):",
        r.nonlin,
        if r.precomputed_masks {
            ", masks precomputed"
        } else {
            ""
        }
    );
    println!(
        "    latency  : {:.0} ms   (fhe {:.0} ms, mpc {:.0} ms)",
        r.per_image_ms, r.per_image_fhe_ms, r.per_image_mpc_ms
    );
    println!(
        "    fhe split: weights {:.0} ms, mul {:.0} ms, mask-enc {:.0} ms, \
         decrypt {:.0} ms, re-encrypt {:.0} ms",
        r.per_image_weights_ms,
        r.per_image_mul_ms,
        r.per_image_maskenc_ms,
        r.per_image_decrypt_ms,
        r.per_image_reencrypt_ms
    );
    println!(
        "    mpc split: import {:.0} ms, relu {:.0} ms, pool {:.0} ms, argmax {:.0} ms",
        r.per_image_import_ms, r.per_image_relu_ms, r.per_image_pool_ms, r.per_image_argmax_ms
    );
    println!(
        "    offline  : {:.0} ms/image query-independent, NOT in the latency above",
        r.per_image_offline_ms
    );
    if r.precomputed_masks {
        println!(
            "               {:.0} ms/image per-SESSION (keygen + encrypted model, paid once for \
             the batch — amortizes)",
            r.per_image_offline_session_ms
        );
        println!(
            "               {:.0} ms/image per-QUERY (mask pool, fresh every image — shifts \
             latency only)",
            r.per_image_offline_query_ms
        );
    }
    println!(
        "    traffic  : {:.0} messages, {:.1} MB party-to-party",
        r.per_image_messages,
        r.per_image_bytes / 1e6
    );
    // The split matters because the two halves are encoded very differently:
    // ring shares go out as `Vec<u128>` (16 B per element regardless of
    // `nbits`), garbled material as packed bytes. A traffic figure that does not
    // separate them cannot be compared against a bit-packed system.
    println!(
        "      of which : ring(ABY3) {:.1} MB [16 B/elem, ~{:.0}% padding], \
         garbled {:.1} MB, other {:.1} MB",
        r.per_image_ring_bytes / 1e6,
        100.0 * (1.0 - r.log2_t as f64 / 128.0),
        r.per_image_gc_bytes / 1e6,
        (r.per_image_bytes - r.per_image_ring_bytes - r.per_image_gc_bytes) / 1e6
    );
    println!(
        "  wall clock incl. plaintext scoring pass: {:.0} ms\n",
        r.ms
    );
}

/// Parse a comma-separated numeric list for the sweep grid flags.
fn parse_list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>, String> {
    s.split(',')
        .map(|p| p.trim().parse().map_err(|_| format!("bad list entry {p:?}")))
        .collect()
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut dataset = "pneumonia".to_string();
    let mut hybrid_n = 5usize;
    let mut json = false;
    let mut nonlin = Nonlin::A2b;
    let mut both = false;
    let mut precompute = false;
    let mut rtt_ms = 0u64;
    let mut bandwidth_mbps: Option<f64> = None;
    let mut sweep = false;
    let mut rtt_list: Vec<u64> = vec![0, 20];
    let mut bw_list: Vec<f64> = vec![1000.0, 100.0, 60.0, 40.0, 30.0, 20.0, 15.0, 10.0];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dataset" => dataset = it.next().ok_or("--dataset needs a value")?.clone(),
            "--sweep" => sweep = true,
            "--rtt-list" => {
                rtt_list = parse_list(it.next().ok_or("--rtt-list needs a value")?)?;
            }
            "--bw-list" => {
                bw_list = parse_list(it.next().ok_or("--bw-list needs a value")?)?;
            }
            "--hybrid" => {
                hybrid_n = it
                    .next()
                    .ok_or("--hybrid needs a value")?
                    .parse()
                    .map_err(|_| "bad --hybrid")?;
            }
            "--json" => json = true,
            "--gc" => nonlin = Nonlin::Gc,
            "--falcon" => nonlin = Nonlin::Falcon,
            "--both" => both = true,
            "--precompute" => precompute = true,
            "--rtt-ms" => {
                rtt_ms = it
                    .next()
                    .ok_or("--rtt-ms needs a value")?
                    .parse()
                    .map_err(|_| "bad --rtt-ms")?;
            }
            "--bandwidth-mbps" => {
                bandwidth_mbps = Some(
                    it.next()
                        .ok_or("--bandwidth-mbps needs a value")?
                        .parse()
                        .map_err(|_| "bad --bandwidth-mbps")?,
                );
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let names: Vec<String> = if dataset == "all" {
        DATASETS.iter().map(|(n, _)| n.to_string()).collect()
    } else {
        vec![dataset]
    };

    if sweep {
        // Only the two routes that are ever competitive: a2b is dominated at
        // every point measured, so including it would pad the grid by half
        // with cells nobody would choose.
        let protocols = if both {
            vec![Nonlin::A2b, Nonlin::Gc, Nonlin::Falcon]
        } else {
            vec![Nonlin::Gc, Nonlin::Falcon]
        };
        for n in &names {
            println!("=== {n} ===");
            let cells = sweep_crossover(n, hybrid_n, &rtt_list, &bw_list, &protocols)?;
            print_sweep(&cells, &rtt_list, &bw_list, &protocols);
        }
        return Ok(());
    }

    let protocols: Vec<Nonlin> = if both {
        vec![Nonlin::A2b, Nonlin::Gc]
    } else {
        vec![nonlin]
    };
    let mut profile = NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt_ms * 500));
    if let Some(mbps) = bandwidth_mbps {
        profile = profile.with_bandwidth(mbps * 1e6 / 8.0);
    }
    let mut reports = Vec::new();
    for n in &names {
        for p in &protocols {
            reports.push(run(n, hybrid_n, profile, *p, precompute)?);
        }
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Private diagnosis on MedMNIST v2 — trained weights, real biomedical");
    println!("images, exact-and-bootstrap-free hybrid pipeline\n");
    for r in &reports {
        print_report(r);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every embedded dataset must reproduce close to the offline full-test
    /// accuracy, and must beat its majority-class baseline — the check that
    /// catches a quantisation or flatten-order regression, which shows up as
    /// a classifier collapsed onto one class.
    #[test]
    fn embedded_accuracy_is_reasonable() {
        for (name, _) in DATASETS {
            let data = load(name).unwrap();
            let model = data.model();
            let correct = data
                .images
                .iter()
                .zip(&data.labels)
                .filter(|(img, &l)| model.forward(img).0 == l as usize)
                .count();
            let acc = correct as f64 / data.images.len() as f64;
            assert!(
                acc > data.majority_baseline + 0.05,
                "{name}: subset accuracy {acc:.3} barely beats the majority baseline \
                 {:.3} — check the quantisation mapping or the flatten order",
                data.majority_baseline
            );
            assert!(
                (acc - data.full_test_accuracy).abs() < 0.10,
                "{name}: subset accuracy {acc:.3} is far from the offline full-test \
                 {:.3} — the embedded images or weights are out of sync",
                data.full_test_accuracy
            );
        }
    }

    /// The encrypted pipeline must reproduce the plaintext diagnosis on real
    /// medical images — privacy costs nothing in accuracy. Both nonlinearity
    /// protocols, because the reported latencies are only comparable if both
    /// produce the same diagnosis.
    #[test]
    #[ignore] // runs the real BFV × ABY3 hybrid; slow in debug
    fn encrypted_matches_plaintext_on_real_scans() {
        for (name, _) in DATASETS {
            for (nonlin, precompute) in [
                (Nonlin::A2b, false),
                (Nonlin::Gc, false),
                (Nonlin::Gc, true),
                (Nonlin::Falcon, false),
                (Nonlin::Falcon, true),
            ] {
                let r = run(name, 2, NetworkProfile::INSTANT, nonlin, precompute).unwrap();
                assert_eq!(
                    r.hybrid_matches_plaintext, r.hybrid_evaluated,
                    "{name}/{nonlin:?}: every encrypted prediction must equal the plaintext one"
                );
                assert_eq!(r.bootstraps, 0, "{name}: the hybrid must not bootstrap");
                assert!(
                    r.per_image_ms > 0.0,
                    "{name}: per-image latency must be measured, not defaulted"
                );
            }
        }
    }

    /// The crossover sweep is only meaningful if every cell computes the *same*
    /// diagnosis — a latency comparison between protocols that disagree is
    /// worthless. This asserts that, and that the network profile is actually
    /// being applied (a lower bandwidth must cost more time), which is the way
    /// a sweep would silently degenerate into measuring nothing.
    #[test]
    #[ignore] // runs the real BFV × ABY3 hybrid; slow in debug
    fn sweep_cells_agree_and_respond_to_bandwidth() {
        let protocols = [Nonlin::Gc, Nonlin::Falcon];
        let cells = sweep_crossover("pneumonia", 1, &[0], &[1000.0, 10.0], &protocols).unwrap();
        assert_eq!(cells.len(), 4);
        for c in &cells {
            assert!(
                c.matches_plaintext,
                "{:?} at {} Mbps disagreed with the plaintext argmax",
                c.nonlin, c.bandwidth_mbps
            );
        }
        for p in protocols {
            let fast = cells
                .iter()
                .find(|c| c.nonlin == p && c.bandwidth_mbps == 1000.0)
                .unwrap();
            let slow = cells
                .iter()
                .find(|c| c.nonlin == p && c.bandwidth_mbps == 10.0)
                .unwrap();
            assert!(
                slow.ms > fast.ms,
                "{p:?}: 10 Mbps ({:.0} ms) must cost more than 1000 Mbps ({:.0} ms) — \
                 the bandwidth profile is not being applied",
                slow.ms,
                fast.ms
            );
        }
    }
}
