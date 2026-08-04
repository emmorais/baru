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
//! Two datasets are embedded, chosen because they stress different ends of the
//! classifier:
//!
//! Both happen to be grayscale, and this module used to say that was *required*
//! because "the conv packing assumes one input channel". **That was wrong.** The
//! reversed channel offset in [`crate::fhe::conv_pack`] has always been
//! multi-channel — channels summing on a shared coefficient is the point of it,
//! and it carries `in_c = 2` and `in_c = 3` unit tests. What was single-channel
//! was the plumbing: `CnnModel` hardcoded `in_c = 1` at three `ConvShape` sites,
//! so no RGB image ever reached the hybrid. It does now
//! (`cnn::tests::rgb_input_classifies_exactly_under_encryption`), at one
//! ciphertext multiplication per output channel, because an RGB 28×28 conv needs
//! degree 3977 against `n = 4096`.
//!
//! So an RGB MedMNIST set (PathMNIST, DermaMNIST) is a *training and embedding*
//! task now, not a packing one. The wide **intermediate** layer used to be the limit
//! that still bit — 16 channels at 13×13 needs degree 5266, which is what capped the
//! first conv at 12 channels — and **channel splitting removed it** on 2026-08-04
//! ([`crate::fhe::conv_pack::channel_groups`], wired through `hybrid_conv_open`).
//! A layer too wide for one ciphertext is evaluated as equal channel groups whose
//! products are added, which is free homomorphically: it costs one multiplication per
//! group per output channel and still no evaluation key and no bootstrap. Both limits
//! are now training problems rather than packing ones.
//!
//! | dataset | task | classes | channels | test set |
//! |---|---|---|---|---|
//! | `pneumonia` | pediatric chest X-ray, pneumonia vs normal | 2 | 1 | 624 |
//! | `organa` | abdominal CT, which organ (axial views) | 11 | 1 | 17 778 |
//! | `derma` | dermatoscopy, 7-class skin lesion | 7 | **3 (RGB)** | 2 005 |
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
//! Two changes landed 2026-08-04: organa was retrained wider (`conv(16,32)`, its
//! second conv split across two channel groups) and `derma` was added as the first
//! **RGB** set. Pneumonia is unchanged at `conv(8,16)`.
//!
//! | | pneumonia | organa | derma (RGB) |
//! |---|---|---|---|
//! | architecture | conv(8,16) | **conv(16,32)**, conv2 split 2×12 | conv(8,16), 3 channels |
//! | integer accuracy, 300 embedded images | **85.00%** | **85.00%** | 69.33% |
//! | integer accuracy, full test set | 84.78% (624) | **82.93%** (17 778) | **72.12%** (2 005) |
//! | 95% CI on the full-test figure | [81.7, 87.4] | **[82.4, 83.5]** | [70.1, 74.0] |
//! | balanced accuracy / AUC, **full test** | 80.64% / **0.9314** | **82.21%** / **0.9605** | **30.14%** / **0.8519** |
//! | majority-class baseline | 62.50% | 18.48% | 66.88% |
//!
//! **Read derma on balanced accuracy and AUC, not accuracy.** Its largest class is
//! 66.88% of the test set, and published ResNet-18 reaches 0.735 — beating the
//! majority by under seven points. Accuracy is nearly uninformative at that skew,
//! which is why `embedded_accuracy_is_reasonable` checks balanced accuracy against
//! chance rather than accuracy against the majority.
//!
//! **Colour is worth having, and costs nothing in ciphertext multiplications.**
//! Against a grayscale-folded control trained identically, RGB gives AUC **0.8519 vs
//! 0.7667** and balanced accuracy **30.14% vs 23.30%** — dermatoscopy is where colour
//! should carry signal, and it does. An RGB 28×28 convolution needs degree 3977
//! against `n = 4096`, so it fits one ciphertext at **one multiplication per output
//! channel**: three channels do not triple the work, they sum on shared coefficients.
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
//! **The 10.1 MB is unchanged by the 2026-08-03 encoding change and the other byte
//! figures are not.** Ring shares used to go on the wire as `u128` whatever their
//! width; they now carry the model's own width
//! ([`crate::bench::cnn::wire_bits_for`]). That leaves this configuration alone
//! because it is garbled-dominated — ring traffic is 5% of it — but `a2b` fell
//! **29.9 → 5.3 MB** and Falcon **10.1 → 3.5 MB**, both of which were mostly
//! padding. Any byte figure elsewhere in the repo dated before that change is at the
//! old encoding; `BARU_WIRE_BITS=128` reproduces it.
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
    /// Side length of the square input the model expects.
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
    /// Input channels, **derived from the embedded images** rather than stored.
    ///
    /// An image is a flat `in_c · d · d`, so the channel count falls out of its
    /// length. Deriving it avoids adding a field to the generated files, which are
    /// marked "do not edit" and would otherwise all need regenerating — and
    /// regenerating them would churn their weights and every accuracy figure quoted
    /// against them, as a side effect of a format change. It is also impossible to
    /// get out of step with the data, which a stored field would not be.
    pub fn in_c(&self) -> usize {
        let per_channel = self.d * self.d;
        self.images
            .first()
            .map(|img| {
                assert_eq!(
                    img.len() % per_channel,
                    0,
                    "embedded image of {} values is not a whole number of {}x{} channels",
                    img.len(),
                    self.d,
                    self.d
                );
                img.len() / per_channel
            })
            .unwrap_or(1)
    }

    /// Build the runnable model. `nbits` is calibrated from the observed
    /// activation range over the embedded images, exactly as the synthetic
    /// model does, so it fits the depth-1 noise budget.
    pub fn model(&self) -> CnnModel {
        CnnModel::from_trained(
            self.d,
            self.in_c(),
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
                    // The shipped weights are 3×3; the trainer would have to emit a
                    // kernel size for anything else, and `check_packing` would need it too.
                    k: crate::bench::cnn::DEFAULT_K,
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

/// The RGB set: three input channels, which the conv packing takes at one
/// ciphertext multiplication per output channel — colour does *not* triple the
/// work. `TrainedCnn::in_c` derives the channel count from the image length.
#[path = "../../examples/data/derma_cnn.rs"]
#[allow(dead_code)]
mod derma;

/// A dataset's `--dataset` name paired with the loader its generated weight
/// file exports.
pub type Embedded = (&'static str, fn() -> TrainedCnn);

/// The embedded datasets, by the name `--dataset` takes.
pub const DATASETS: &[Embedded] = &[
    ("pneumonia", pneumonia::trained),
    ("organa", organa::trained),
    ("derma", derma::trained),
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
    /// Falcon comparison elements one diagnosis requests — the per-diagnosis pool size,
    /// counted by the backend rather than derived from the layer shapes.
    pub per_image_falcon_comparisons: usize,
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
    let preds: Vec<usize> = logits
        .iter()
        .map(|l| crate::gadgets::argmax_ref(l))
        .collect();
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
    let mut last_falcon_comparisons = 0usize;
    for ((img, &plain), &label) in data
        .images
        .iter()
        .zip(&preds)
        .zip(&data.labels)
        .take(hybrid_n)
    {
        let run = run_cnn_hybrid_in_session(&session, &model, img, profile, nonlin, precompute);
        // Identical for every image of a given model, so the last one is the figure.
        last_falcon_comparisons = run.falcon_comparisons;
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
        per_image_falcon_comparisons: last_falcon_comparisons,
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
    /// Of `bytes`, the ABY3 ring resharing — 16 B per element whatever the ring
    /// width is. Carried because it decides whether the byte comparison between
    /// two protocols means anything: a route whose traffic is mostly padding is
    /// being charged for this crate's serialisation, not for its protocol, and
    /// this repo has already drawn one wrong conclusion that way.
    pub ring_bytes: f64,
    /// Every cell recomputes the same diagnosis; if one disagrees with the
    /// plaintext argmax the whole row is meaningless, so it is carried here
    /// rather than assumed.
    pub matches_plaintext: bool,
    /// Fastest and slowest of the `repeats` observations behind [`SweepCell::ms`]
    /// (which is their **median**). Carried because a crossover is a claim that one
    /// route is faster than another, and two cells whose ranges overlap do not
    /// support it however far apart their point estimates sit.
    ///
    /// The need is measured, not hypothetical: re-running an *unchanged* `gc`
    /// configuration moved one 60 Mbps cell by 13%, which is larger than the gap
    /// across a whole grid step near the boundary. Four of this experiment's
    /// withdrawn crossover claims were single-run numbers.
    ///
    /// **This range is within-session and so understates the real spread** — see
    /// [`separable`]. It bounds jitter between back-to-back repeats sharing one
    /// `CnnSession`, not reproducibility across invocations.
    pub ms_lo: f64,
    pub ms_hi: f64,
    pub repeats: usize,
    /// Whether this cell's Falcon preprocessing was pre-generated, making `ms` an online
    /// cost rather than a total. Carried on the cell so a table can never be printed
    /// without its label.
    pub warm: bool,
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
/// `repeats` re-runs each cell and reports the **median**, with the observed range
/// kept in [`SweepCell::ms_lo`]/[`SweepCell::ms_hi`] so [`print_sweep`] can refuse to
/// call a winner where the ranges overlap. One repeat is allowed and is the default,
/// because most uses of this grid are checking correctness or orders of magnitude
/// rather than resolving a boundary — but a boundary quoted from `repeats == 1` is a
/// point estimate with no error bar, and this experiment has already published four
/// wrong ones.
/// `warm` pre-generates each MPC segment's Falcon preprocessing from a plan measured by
/// one cold run, and reports the **online** cost (total minus that offline phase).
///
/// This is the second of two honest boundaries, not a replacement for the first. A cold
/// server pays the total; a warm one, which has generated query-independent preprocessing
/// in advance, pays only the online phase. Both are real deployments, and quoting one
/// protocol's online cost against another's total is precisely how this experiment's first
/// crossover went wrong — so the flag is explicit and the column is labelled.
///
/// `gc` is unaffected: garbled circuits have no query-independent phase to hoist here, so
/// its column is identical warm and cold. That asymmetry is the point of the comparison
/// rather than a flaw in it, but it is why the warm boundary must never be presented as
/// *the* boundary.
pub fn sweep_crossover(
    name: &str,
    hybrid_n: usize,
    rtts: &[u64],
    bws: &[f64],
    protocols: &[Nonlin],
    repeats: usize,
    warm: bool,
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

    // The plan depends on the model's shapes, not on the network, so one cold run at an
    // instant profile suffices for the whole grid. Measured rather than derived — see
    // `cnn::FalconPoolPlan`.
    let plan = if warm {
        let probe = run_cnn_hybrid_in_session(
            &session,
            &model,
            &data.images[0],
            NetworkProfile::INSTANT,
            Nonlin::Falcon,
            true,
        );
        probe.falcon_plan
    } else {
        crate::bench::cnn::FalconPoolPlan::new()
    };

    let mut cells = Vec::new();
    for &rtt in rtts {
        for &bw in bws {
            let profile =
                NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt * 500))
                    .with_bandwidth(bw * 1e6 / 8.0);
            for &p in protocols {
                let reps = repeats.max(1);
                let mut ok = true;
                let mut observed: Vec<f64> = Vec::with_capacity(reps);
                // The last repeat's traffic stands for all of them: bytes and
                // messages are determined by the protocol and the shapes, so they
                // are identical across repeats. Only the timing varies.
                let mut last = CnnTotals::default().mean(1);
                for _ in 0..reps {
                    let mut tot = CnnTotals::default();
                    for (img, &want) in data.images.iter().take(hybrid_n).zip(&plain) {
                        let run = crate::bench::cnn::run_cnn_hybrid_in_session_pooled(
                            &session, &model, img, profile, p, true, &plan,
                        );
                        ok &= run.class == want;
                        tot.add(&run);
                    }
                    let per = tot.mean(hybrid_n);
                    // The online cost: the pool fill happens inside the timed region, so a
                    // warm cell must have it subtracted or it is a total wearing an online
                    // label. Cold runs have no offline phase and are unaffected.
                    observed.push(per.total_ms - per.falcon_offline_ms);
                    last = per;
                }
                observed.sort_by(f64::total_cmp);
                cells.push(SweepCell {
                    rtt_ms: rtt,
                    bandwidth_mbps: bw,
                    nonlin: p,
                    ms: observed[observed.len() / 2],
                    messages: last.messages,
                    bytes: last.bytes,
                    ring_bytes: last.ring_bytes,
                    matches_plaintext: ok,
                    ms_lo: observed[0],
                    ms_hi: observed[observed.len() - 1],
                    repeats: reps,
                    warm,
                });
            }
        }
    }
    Ok(cells)
}

/// Print the sweep as a latency grid, then name the bandwidth at which the
/// winner changes for each RTT — the number the whole experiment is for.
/// Bandwidth label that survives sub-1 Mbps.
///
/// `{:.0}` printed the 0.5 Mbps row as `0`, which matters because that is exactly
/// where the crossover turned out to be at the default wire encoding.
fn fmt_bw(mbps: f64) -> String {
    if mbps.fract() == 0.0 {
        format!("{mbps:.0}")
    } else {
        format!("{mbps:.2}")
    }
}

/// Whether the faster cell is *distinguishably* faster **within one session**: its
/// slowest observation still beat the other's fastest.
///
/// # This captures the smaller of the two variance components
///
/// Measured on pneumonia at RTT 0: repeats *inside* one `sweep_crossover` call spread
/// under 2%, while an unchanged `gc` configuration re-run as a separate **process** moved
/// one 60 Mbps cell by **13%**. The repeats share a `CnnSession`, run back to back, and
/// hit warm caches, so they are not independent draws — they bound jitter, not run-to-run
/// reproducibility.
///
/// So a `*` here means "separable given this session", which is weaker than "separable".
/// Cells within ~13% of each other should be read as unresolved however many repeats say
/// otherwise, and a boundary read off the grid is resolved to about ±1 step. Capturing the
/// larger component would mean re-running the whole binary and comparing, which the CLI
/// does not do because each cell already costs seconds to minutes at low bandwidth.
///
/// The comparison is a strict range test rather than a t-test on the medians because with
/// three observations a range is what the data supports, and the failure mode guarded
/// against — quoting a boundary the next run moves — comes from over-claiming, so erring
/// toward "tie" is right. With `repeats == 1` both ranges are points and this reduces to
/// `<`, which is why the printer warns rather than pretending to have an error bar.
fn separable(win: &SweepCell, lose: &SweepCell) -> bool {
    win.ms_hi < lose.ms_lo
}

fn print_sweep(cells: &[SweepCell], rtts: &[u64], bws: &[f64], protocols: &[Nonlin]) {
    let reps = cells.first().map(|c| c.repeats).unwrap_or(1);
    let warm = cells.iter().any(|c| c.warm);
    if warm {
        println!(
            "  ⚙  WARM: falcon's per-segment preprocessing was generated offline; the \
             latency column is an ONLINE cost.\n     gc has no query-independent phase to \
             hoist, so its column is unchanged — quote alongside the cold boundary, never \
             instead.\n     NOTE the traffic table below is still a TOTAL (offline + \
             online): the transport counts bytes for the whole run, so the two tables are \
             not on the same footing."
        );
    }
    if reps > 1 {
        println!(
            "crossover sweep — median ms per encrypted diagnosis over {reps} repeats, \
             * marks a winner, ~ a tie (ranges overlap)\n  \
             (repeats share one session, so the spread bounds jitter, NOT run-to-run \
             reproducibility — an unchanged config has moved 13% between invocations, \
             so read cells within ~13% as unresolved and the boundary as ±1 step)\n"
        );
    } else {
        println!(
            "crossover sweep — ms per encrypted diagnosis, * marks the winner\n  \
             (single run: no error bar, so a boundary read off this grid is \
             unresolved to ±1 step — pass --repeat 3 to resolve it)\n"
        );
    }
    let mismatched = cells.iter().filter(|c| !c.matches_plaintext).count();
    if mismatched > 0 {
        println!("  ⚠️  {mismatched} cell(s) disagreed with the plaintext argmax — DO NOT quote\n");
    }
    let cell = |r: u64, b: f64, p: Nonlin| {
        cells
            .iter()
            .find(|c| c.rtt_ms == r && c.bandwidth_mbps == b && c.nonlin == p)
    };
    let at = |r: u64, b: f64, p: Nonlin| cell(r, b, p).map(|c| c.ms);
    // A winner only where the ranges are disjoint. With one repeat the range is a
    // point, so this degrades to the old strict comparison — which is exactly the
    // situation the header warns about rather than hides.
    let decisive = |r: u64, b: f64| -> Option<(Nonlin, bool)> {
        if protocols.len() != 2 {
            return None;
        }
        let (a, b_) = (cell(r, b, protocols[0])?, cell(r, b, protocols[1])?);
        let (win, lose) = if a.ms <= b_.ms { (a, b_) } else { (b_, a) };
        Some((win.nonlin, separable(win, lose)))
    };
    for &rtt in rtts {
        println!("  RTT {rtt} ms");
        print!("    {:<10}", "Mbps");
        for p in protocols {
            print!("{:>12}", p.short());
        }
        println!();
        for &bw in bws {
            print!("    {:<10}", fmt_bw(bw));
            let best = protocols
                .iter()
                .filter_map(|&p| at(rtt, bw, p))
                .fold(f64::INFINITY, f64::min);
            let mark = match decisive(rtt, bw) {
                Some((_, true)) | None => '*',
                Some((_, false)) => '~',
            };
            for &p in protocols {
                match at(rtt, bw, p) {
                    Some(ms) if ms == best => print!("{:>12}", format!("{mark}{ms:.0}")),
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
            let winner = |bw: f64| decisive(rtt, bw).map(|(p, _)| p);
            let mut sorted: Vec<f64> = bws.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let flip = sorted
                .windows(2)
                .find(|w| winner(w[0]) != winner(w[1]) && winner(w[1]).is_some());
            match flip {
                Some(w) => println!(
                    "    → crossover between {} and {} Mbps: {} wins above, {} below",
                    fmt_bw(w[1]),
                    fmt_bw(w[0]),
                    winner(w[0]).map(|p| p.short()).unwrap_or("?"),
                    winner(w[1]).map(|p| p.short()).unwrap_or("?"),
                ),
                None => println!(
                    "    → no crossover in this range: {} wins throughout",
                    winner(sorted[0]).map(|p| p.short()).unwrap_or("?")
                ),
            }
            // The tie band is the honest width of the boundary: every bandwidth at
            // which the two routes are not separable. Quoting a crossover narrower
            // than this band claims resolution the measurement does not have.
            let ties: Vec<f64> = sorted
                .iter()
                .copied()
                .filter(|&b| matches!(decisive(rtt, b), Some((_, false))))
                .collect();
            if reps > 1 && !ties.is_empty() {
                println!(
                    "      not separable at {} Mbps — the boundary is a band, not a point",
                    ties.iter().map(|&b| fmt_bw(b)).collect::<Vec<_>>().join(", "),
                );
            }
            println!();
        } else {
            println!();
        }
    }

    // ── The two terms behind every cell above ────────────────────────────────
    //
    // Printed because a crossover is a claim about *trading* one term against the
    // other, and the latencies alone do not show whether a trade is even on offer.
    // A protocol can only win somewhere if it is cheaper in one term and dearer in
    // the other; when one route is cheaper in **both**, no (rtt, bandwidth) pair
    // favours the other and extending the grid cannot find one. That is the check
    // the withdrawn crossover needed and did not have.
    //
    // These are counts, not the latency model's inputs: `ms` is measured wall-clock
    // under a simulator with concurrent links, so a byte ratio does *not* predict
    // the low-bandwidth limit of the columns — messages spread over links overlap.
    // Read them as the two terms' relative size, not as a formula for the table.
    //
    // Query-independent for a given protocol, so one row each rather than per cell.
    println!("  traffic per diagnosis (identical across the grid — only the link varies)");
    println!(
        "    {:<10}{:>10}{:>10}{:>15}",
        "protocol", "messages", "MB wire", "of which ring"
    );
    for &p in protocols {
        if let Some(c) = cells.iter().find(|c| c.nonlin == p) {
            println!(
                "    {:<10}{:>10.0}{:>10.1}{:>15.1}",
                p.short(),
                c.messages,
                c.bytes / 1e6,
                c.ring_bytes / 1e6,
            );
        }
    }
    if protocols.len() == 2 {
        let get = |p: Nonlin| cells.iter().find(|c| c.nonlin == p);
        if let (Some(x), Some(y)) = (get(protocols[0]), get(protocols[1])) {
            let (a, b) = (protocols[0], protocols[1]);
            // The ring term is broken out because it is where a protocol
            // comparison can quietly stop being about protocols. Ring shares are
            // packed at whatever width the caller declares, so a route that asks
            // for 128 bits to carry 20-bit values pays 6.4x for nothing — and
            // since the two comparison protocols differ in how much of their
            // traffic is ring shares, that overhead does not cancel between them.
            // It once made this very comparison read 3.2x the wrong way; see
            // `docs/falcon-reproduction.md`. A large `of which ring` next to a
            // large total is the signal to go and check the declared widths.
            let cheaper_bytes = if x.bytes <= y.bytes { a } else { b };
            let cheaper_msgs = if x.messages <= y.messages { a } else { b };
            let bratio = x.bytes.max(y.bytes) / x.bytes.min(y.bytes).max(1.0);
            let mratio = x.messages.max(y.messages) / x.messages.min(y.messages).max(1.0);
            // A 1.01x byte ratio is a tie, not an advantage, and calling it a
            // trade would license predicting a crossover from what is really
            // noise. Below this margin the byte axis carries no signal and the
            // boundary — if there is one — is set by the compute intercept
            // instead, which is far outside any bandwidth worth quoting.
            const TIE: f64 = 1.05;
            if bratio < TIE {
                println!(
                    "    → the two routes send the SAME bytes to within {:.0}% ({bratio:.2}x), so \n\
                     \x20     the byte axis cannot decide between them; {} wins on messages \n\
                     \x20     ({mratio:.1}x) and any boundary is set by the compute intercept",
                    (TIE - 1.0) * 100.0,
                    cheaper_msgs.short(),
                );
            } else if cheaper_bytes == cheaper_msgs {
                println!(
                    "    → {} is cheaper in BOTH terms ({mratio:.1}x messages, {bratio:.2}x \n\
                     \x20     bytes) — no (rtt, bandwidth) pair favours the other, so no crossover \n\
                     \x20     exists however far the grid is widened",
                    cheaper_bytes.short(),
                );
            } else {
                println!(
                    "    → the terms trade: {} is cheaper in bytes ({bratio:.2}x) and {} in \n\
                     \x20     messages ({mratio:.1}x), so a crossover exists — low bandwidth \n\
                     \x20     favours the byte-cheaper route, high RTT the message-cheaper one",
                    cheaper_bytes.short(),
                    cheaper_msgs.short(),
                );
            }
        }
    }
    println!();
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
    /// Offline Falcon preprocessing, tracked so the online figure can exclude it. A warm
    /// run's `total_ms` contains its own pool fill, because the fill happens inside the
    /// `run_parties` call that is timed.
    falcon_offline_ms: f64,
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
        self.falcon_offline_ms += r.phases.falcon_offline_ms;
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
            falcon_offline_ms: self.falcon_offline_ms / d,
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
  --warm          pre-generate each MPC segment's Falcon preprocessing from a plan
                  measured by one cold run, and report the ONLINE cost. gc is
                  unaffected (it has no query-independent phase to hoist), so the
                  warm boundary must be quoted alongside the cold one, never
                  instead of it.
  --repeat N      re-run every sweep cell N times and report the MEDIAN, marking
                  cells whose ranges overlap as ties rather than wins (default 1).
                  Needed to resolve a crossover: an unchanged configuration has
                  been seen to vary 13% between runs, which is wider than one
                  grid step near the boundary.
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
    // ring shares are packed at whatever width the backend declares, garbled
    // material as compact bytes. A traffic figure that does not separate them
    // cannot be compared against a bit-packed system — and the padding line below
    // is why: at the 128-bit default a 20-bit value travels in 128 bits, so 84% of
    // the ring half is nothing. `BARU_WIRE_BITS` narrows it (see `cnn`), which is
    // worth 5.6x on this route, so the figure is computed from the width actually
    // in force rather than hardcoded.
    let wire = crate::bench::cnn::wire_bits_for(r.log2_t as usize).unwrap_or(128);
    println!(
        "      of which : ring(ABY3) {:.1} MB [{} b/elem on the wire, ~{:.0}% padding], \
         garbled {:.1} MB, other {:.1} MB",
        r.per_image_ring_bytes / 1e6,
        wire,
        100.0 * (1.0 - (r.log2_t as f64 / wire as f64)).max(0.0),
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
        .map(|p| {
            p.trim()
                .parse()
                .map_err(|_| format!("bad list entry {p:?}"))
        })
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
    let mut repeats = 1usize;
    let mut warm = false;
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
            "--warm" => warm = true,
            "--repeat" => {
                repeats = it
                    .next()
                    .ok_or("--repeat needs a value")?
                    .parse()
                    .map_err(|_| "bad --repeat")?;
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
            let cells =
                sweep_crossover(n, hybrid_n, &rtt_list, &bw_list, &protocols, repeats, warm)?;
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
    /// **A boundary is only as sharp as the runs behind it.**
    ///
    /// `separable` is what stops the sweep naming a winner it cannot distinguish. The
    /// cases below are the ones that matter: a clear win, a clear loss, an overlap
    /// where the medians differ (the dangerous case — a point estimate would call it),
    /// and the degenerate single-run case where the ranges are points.
    #[test]
    fn a_sweep_winner_needs_disjoint_ranges() {
        let cell = |ms: f64, lo: f64, hi: f64, reps: usize| SweepCell {
            rtt_ms: 0,
            bandwidth_mbps: 30.0,
            nonlin: Nonlin::Gc,
            ms,
            messages: 0.0,
            bytes: 0.0,
            ring_bytes: 0.0,
            matches_plaintext: true,
            ms_lo: lo,
            ms_hi: hi,
            repeats: reps,
            warm: false,
        };

        // Disjoint: 100..110 entirely below 200..210.
        assert!(separable(&cell(105.0, 100.0, 110.0, 3), &cell(205.0, 200.0, 210.0, 3)));

        // Overlapping, and the medians are 5% apart — exactly the situation that
        // produced this repo's withdrawn crossover claims.
        assert!(
            !separable(&cell(100.0, 90.0, 115.0, 3), &cell(105.0, 95.0, 120.0, 3)),
            "overlapping ranges must not yield a winner however far apart the medians"
        );

        // Touching but not overlapping is still not separable: `<` is strict on
        // purpose, so a shared endpoint reads as a tie.
        assert!(!separable(&cell(100.0, 90.0, 110.0, 3), &cell(120.0, 110.0, 130.0, 3)));

        // Single run: ranges are points, so this degrades to a bare comparison.
        // Pinned so the degradation is a decision on record, not an accident.
        assert!(separable(&cell(100.0, 100.0, 100.0, 1), &cell(101.0, 101.0, 101.0, 1)));
    }

    /// **The Falcon pool sizes a real diagnosis needs**, measured rather than derived.
    ///
    /// This records the **total**, and pins that the counter reports something rather than
    /// silently zero — which is how a pool sized from it would end up serving nothing while
    /// a benchmark reported an "online" latency that was really a total.
    ///
    /// A pool cannot use a total, though: the pipeline runs one `run_parties` call per MPC
    /// segment, each with a fresh backend, so the sizes have to be *per segment*. That
    /// wiring exists (`--warm`, `cnn::FalconPoolPlan`) and
    /// `the_per_segment_pool_sizes_and_what_pooling_buys` is where the split and the
    /// warm/cold gap are measured. This test remains the cheap guard on the counter itself.
    ///
    /// The count is deliberately *not* asserted against a fixed number, and 2026-08-04
    /// is why: wiring `Π_ReLU` into `fused_relu_rescale` routed every ReLU gate through
    /// the same `falcon_drelu_ring`, so the counter grew from the max-pool comparisons
    /// alone to those plus the activations. A hard-coded expectation would have failed
    /// and read as a regression, when the number changing *was* the change.
    #[test]
    #[ignore] // full encrypted pipeline
    fn the_falcon_pool_size_a_diagnosis_needs_is_counted() {
        for name in ["pneumonia", "organa", "derma"] {
            let r = run(name, 1, NetworkProfile::INSTANT, Nonlin::Falcon, false)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            println!(
                "  {name:<10} falcon comparisons per diagnosis: {}",
                r.per_image_falcon_comparisons
            );
            assert!(
                r.per_image_falcon_comparisons > 0,
                "{name}: the Falcon path must report the comparisons it requested — a \
                 zero here would size a pool at nothing and quietly fall back to \
                 generating preprocessing on the critical path"
            );
        }

        // The other modes must report zero, or the counter is measuring the wrong thing.
        let gc = run("pneumonia", 1, NetworkProfile::INSTANT, Nonlin::Gc, false).unwrap();
        assert_eq!(
            gc.per_image_falcon_comparisons, 0,
            "the garbled route must not report Falcon comparisons"
        );
    }

    /// **Warming Falcon must not change what `gc` costs.**
    ///
    /// `gc` is the control: it has no query-independent phase to hoist, so a `--warm` sweep
    /// must leave its traffic bit-for-bit identical to a cold one. This is the test for
    /// that, and it exists because the first warm sweep failed it — the plan was applied to
    /// every route, so `gc` filled Falcon pools it never read and its traffic moved from
    /// **61 messages / 10.1 MB to 154 / 12.2**. The latency table looked entirely plausible;
    /// what gave it away was an unchanged control moving.
    ///
    /// Traffic rather than time, deliberately: bytes and messages are deterministic, so
    /// this can assert equality instead of a tolerance, and it fails loudly rather than
    /// blending into timing noise.
    #[test]
    #[ignore] // two full sweeps
    fn warming_falcon_leaves_the_gc_control_untouched() {
        let protocols = [Nonlin::Gc, Nonlin::Falcon];
        let grid = (&[0u64][..], &[1000.0f64][..]);
        let cold = sweep_crossover("pneumonia", 1, grid.0, grid.1, &protocols, 1, false).unwrap();
        let warm = sweep_crossover("pneumonia", 1, grid.0, grid.1, &protocols, 1, true).unwrap();

        let find = |cs: &[SweepCell], p: Nonlin| {
            cs.iter()
                .find(|c| c.nonlin == p)
                .map(|c| (c.messages, c.bytes, c.ring_bytes, c.matches_plaintext))
                .expect("cell present")
        };

        let (cm, cb, cr, cok) = find(&cold, Nonlin::Gc);
        let (wm, wb, wr, wok) = find(&warm, Nonlin::Gc);
        println!("  gc cold: {cm} msgs / {cb:.0} B ({cr:.0} ring)");
        println!("  gc warm: {wm} msgs / {wb:.0} B ({wr:.0} ring)");
        assert!(cok && wok, "both gc cells must match the plaintext argmax");
        assert_eq!(cm, wm, "warming Falcon changed gc's message count");
        assert_eq!(cb, wb, "warming Falcon changed gc's byte count");
        assert_eq!(cr, wr, "warming Falcon changed gc's ring bytes");

        // And the warm flag must actually be set, or this passes because nothing warmed.
        assert!(
            warm.iter().any(|c| c.warm) && !cold.iter().any(|c| c.warm),
            "the warm sweep must be marked warm and the cold one not"
        );
        // Falcon *should* differ — otherwise the plan reached nothing.
        let (fcm, _, _, _) = find(&cold, Nonlin::Falcon);
        let (fwm, _, _, _) = find(&warm, Nonlin::Falcon);
        assert_ne!(
            fcm, fwm,
            "falcon's traffic should change when its preprocessing is pooled ({fcm} vs {fwm})"
        );
    }

    /// **The per-segment Falcon pool sizes a real diagnosis needs, and what pooling buys.**
    ///
    /// `the_falcon_pool_size_a_diagnosis_needs_is_counted` records the *total*; a pool
    /// cannot use a total, because the CNN runs one `run_parties` call per MPC segment with
    /// a fresh backend. This prints the split and measures the warm/cold gap it unlocks.
    ///
    /// The online figure is `online_mpc_ms()`, not `mpc_ms`: the pool fill happens inside
    /// the call `mpc_ms` brackets, so a warm run's `mpc_ms` contains its own offline phase.
    #[test]
    #[ignore] // full encrypted pipeline, per dataset
    fn the_per_segment_pool_sizes_and_what_pooling_buys() {
        use crate::bench::cnn::run_cnn_hybrid_in_session_pooled;
        for name in ["pneumonia", "organa", "derma"] {
            let data = load(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            let model = data.model();
            let img = &data.images[0];
            let session = CnnSession::new(&model, true);
            let profile = NetworkProfile::INSTANT;

            let cold = run_cnn_hybrid_in_session(
                &session, &model, img, profile, Nonlin::Falcon, true);
            let warm = run_cnn_hybrid_in_session_pooled(
                &session, &model, img, profile, Nonlin::Falcon, true, &cold.falcon_plan);

            let total: usize = cold.falcon_plan.values().sum();
            println!("\n  {name}: {} comparisons over {} segments", total,
                     cold.falcon_plan.len());
            for (seg, n) in &cold.falcon_plan {
                println!("    {seg:?}: {n}");
            }
            println!(
                "    online MPC {:.1} → {:.1} ms ({:.2}x), offline {:.1} ms",
                cold.online_mpc_ms(),
                warm.online_mpc_ms(),
                cold.online_mpc_ms() / warm.online_mpc_ms().max(1e-9),
                warm.phases.falcon_offline_ms,
            );

            assert_eq!(warm.class, cold.class, "{name}: pooling changed the diagnosis");
            assert_eq!(
                warm.phases.falcon_pool_missed, 0,
                "{name}: {} misses — the online figure would include preprocessing",
                warm.phases.falcon_pool_missed
            );
            assert_eq!(
                total, cold.falcon_comparisons,
                "{name}: the per-segment sizes must sum to the counted total"
            );
            assert!(
                warm.online_mpc_ms() < cold.online_mpc_ms(),
                "{name}: pooling must shorten the online segment"
            );
        }
    }

    /// **The shipped derma model is genuinely RGB**, and this is the fast guard for it.
    ///
    /// The slow guard — `encrypted_matches_plaintext_on_real_scans` — takes twenty
    /// minutes across three datasets and five protocol configurations, so it is not
    /// the thing that should be relied on to notice a three-channel model quietly
    /// becoming a one-channel one. That failure has a specific shape worth naming: the
    /// trainer folds RGB to luminance unless `--rgb` is passed, and a regenerated
    /// weight file missing that flag would still load, still train, still score
    /// plausibly, and silently be a different experiment. A `--rgb` flag that fails to
    /// reach the trainer is not hypothetical — it happened, via shell word-splitting,
    /// and produced two "RGB vs grayscale" arms that were both grayscale.
    ///
    /// So this checks the property that distinguishes them: the channel count, derived
    /// from the embedded image length rather than declared, and the packing cost that
    /// follows from it.
    #[test]
    fn the_shipped_derma_model_is_rgb_and_costs_one_multiply_per_channel() {
        let data = load("derma").expect("derma must be embedded");
        assert_eq!(
            data.in_c(),
            3,
            "derma must be RGB — a grayscale fold means --rgb never reached the trainer"
        );
        assert_eq!(
            data.images[0].len(),
            3 * data.d * data.d,
            "an RGB image is three channels of d×d, laid out channel-major"
        );

        let model = data.model();
        let n = crate::fhe::bfv::BfvParams::DEFAULT.n;
        let plan = crate::bench::cnn::packing_plan(&model, n);
        let (c1, _, ng1) = plan[0];
        assert_eq!(c1.in_c, 3, "the first conv sees three channels");
        assert_eq!(
            ng1, 1,
            "RGB 28×28 must fit ONE ciphertext (degree {} against n={n}) — three \
             channels sum on shared coefficients rather than costing three multiplies",
            c1.required_degree()
        );
        assert_eq!(c1.required_degree(), 3977, "the degree this rests on");
    }

    /// **The shipped organa model exercises channel splitting**, and this pins that
    /// the capability is actually in use rather than merely available.
    ///
    /// Retrained 2026-08-04 at `--conv 16,32`, which the trainer used to refuse: its
    /// second convolution takes 16 channels at 13×13, degree 5266 against `n = 4096`,
    /// so it runs as two groups of 12 whose products are added. That widening is what
    /// took organa from 81.64% to **82.93%** on 17 778 test images — disjoint
    /// confidence intervals, so a real gain rather than seed noise.
    ///
    /// The cost is visible and worth stating: conv2 goes from 32 ciphertext
    /// multiplications to 64, because each output channel now multiplies two input
    /// groups. Still no evaluation key, still no bootstrap — group results combine by
    /// homomorphic addition.
    #[test]
    fn the_shipped_organa_model_uses_channel_splitting() {
        let model = super::DATASETS
            .iter()
            .find(|(n, _)| *n == "organa")
            .expect("organa must be embedded")
            .1()
            .model();
        let plan = crate::bench::cnn::packing_plan(&model, crate::fhe::bfv::BfvParams::DEFAULT.n);
        assert_eq!(plan.len(), 2, "organa is a two-conv model");

        let (c1, gs1, ng1) = plan[0];
        assert_eq!((gs1, ng1), (c1.in_c, 1), "conv1 fits one ciphertext");

        let (c2, gs2, ng2) = plan[1];
        assert_eq!(c2.in_c, 16, "conv2 must take the widened 16 channels");
        assert_eq!(
            (gs2, ng2),
            (12, 2),
            "conv2 at {}x{} must split into 2 groups of 12 (degree {} vs n={})",
            c2.h,
            c2.w,
            c2.required_degree(),
            crate::fhe::bfv::BfvParams::DEFAULT.n
        );
        assert!(
            c2.required_degree() > crate::fhe::bfv::BfvParams::DEFAULT.n,
            "the point of this test is that the layer does NOT fit unsplit"
        );
    }

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

            // **Balanced accuracy against chance, not accuracy against the majority
            // class.** This assertion exists to catch a wiring error — a wrong
            // quantisation mapping or flatten order — and plain accuracy cannot do
            // that on a severely imbalanced multi-class set. DermaMNIST's largest
            // class is 67% of it, and published ResNet-18 beats that by only ~6.6
            // points, so "majority + 5" is unreachable there for reasons that have
            // nothing to do with wiring. Balanced accuracy against `1/nclasses` is
            // the check that works for every shape of dataset: a misflattened input
            // scores at chance whatever the class balance is.
            let nclasses = data.class_names.len();
            let logits: Vec<Vec<i64>> = data
                .images
                .iter()
                .map(|img| model.forward_logits(img).0)
                .collect();
            let m = ClassificationMetrics::compute(&logits, &data.labels, nclasses);
            let chance = 1.0 / nclasses as f64;
            assert!(
                m.balanced_accuracy > chance * 1.5,
                "{name}: balanced accuracy {:.3} is near chance {chance:.3} — check the \
                 quantisation mapping or the flatten order",
                m.balanced_accuracy
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
        let cells =
            sweep_crossover("pneumonia", 1, &[0], &[1000.0, 10.0], &protocols, 1, false).unwrap();
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
