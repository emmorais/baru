//! `baru bench-train` — **private training**, not just private inference.
//!
//! A least-squares linear classifier is trained by SGD entirely under the
//! real 3-party ABY3 protocol: the training images/labels stay secret-
//! shared, every forward pass, gradient, and weight update is an oblivious
//! computation composed from `crate::gadgets`, and the resulting model is
//! then used for private inference on held-out examples. This closes the
//! loop the rest of the crate opened — "receive encrypted data, train,
//! later do encrypted inference" — end to end.
//!
//! # Why MSE (and not softmax)
//!
//! Training a classifier the usual way needs softmax + cross-entropy, whose
//! gradient is `exp`/`log`/division — the same transcendental wall that
//! blocks private transformers, and one we deliberately do not climb here.
//! Regressing to one-hot targets with a **squared-error** loss trains the
//! same linear model with a gradient that is purely `sub`/`mul`/shift — no
//! nonlinearity in the loss at all:
//!
//!   forward   z   = W·x + b
//!   error     δ   = z − y            (y = one-hot target, scaled)
//!   gradient  ∂W  = δ ⊗ x,  ∂b = δ
//!   update    W  -= ∂W ≫ lr,  b -= ∂b ≫ lr   (lr = 2^-lr; `gadgets::arith_shr`)
//!
//! Every step is exact integer fixed-point, so the encrypted run matches a
//! plaintext integer-SGD reference **bit-for-bit** (verified below) — the
//! loss curve and the final weights are identical, not approximate. This is
//! the same discipline as the inference benchmarks: correctness is exact;
//! *accuracy* is an ML property of the (here synthetic, linearly separable)
//! data, reported honestly and separately.
//!
//! # Where this sits relative to the no-bootstrap result
//!
//! Training is naturally an MPC computation (as in SecureML / ABY3's own
//! training experiments) — there is no FHE ciphertext to bootstrap here, so
//! "0 bootstraps" is trivially true on this side. The paper's no-bootstrap
//! *FHE* contribution is the **inference** deployment; this module completes
//! the private-ML lifecycle around it and shows the same UC-gadget library
//! composes a full SGD trainer, not just a forward pass. (A deep MLP's
//! forward+backward *step* is itself a deep hybrid chain that the
//! no-bootstrap property would carry — the ReLU-derivative backward gadget
//! is `lt`, already here — but that larger build is future work; this
//! demonstrator is the linear case, which converges provably.)

use serde::Serialize;

use crate::bench::comm::{CommStats, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, MpcTransport};
use crate::gadgets::{arith_shr, argmax};
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

const NBITS: usize = 32;
/// The one-hot target's "hot" value doubles as the fixed-point scale: it
/// must be large enough that the optimal (fractional-real) weights round to
/// comfortable integers — otherwise integer SGD oscillates around a value
/// it cannot represent and diverges.
const TARGET: i64 = 256;

fn residue(v: i64, nbits: usize) -> u128 {
    (v as i128).rem_euclid(1i128 << nbits) as u128
}
fn to_signed(v: u128) -> i64 {
    (v as u32) as i32 as i64
}

// ─── Synthetic dataset + plaintext integer-SGD reference ──────────────────────

pub struct Dataset {
    pub x: Vec<Vec<i64>>, // [n_samples][n_features]
    pub label: Vec<usize>,
    pub n_features: usize,
    pub n_classes: usize,
    /// The ground-truth model the labels were drawn from — kept so a
    /// held-out test example can be drawn from the *same* distribution.
    w_true: Vec<Vec<i64>>,
}

/// A deterministic, linearly separable dataset: pick a random "true"
/// weight matrix, draw feature vectors, and label each by the argmax of the
/// true scores — so a linear classifier can (nearly) fit it.
pub fn make_dataset(n: usize, n_features: usize, n_classes: usize, seed: u64) -> Dataset {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let w_true: Vec<Vec<i64>> = (0..n_classes)
        .map(|_| (0..n_features).map(|_| rng.gen_range(-4i64..=4)).collect())
        .collect();
    let mut x = Vec::with_capacity(n);
    let mut label = Vec::with_capacity(n);
    for _ in 0..n {
        // Binary features (bounds ‖x‖² ≤ n_features, which keeps the SGD
        // stability condition lr·‖x‖² < 2 easy to satisfy) — also a natural
        // stand-in for binarized image pixels.
        let xi: Vec<i64> = (0..n_features).map(|_| rng.gen_range(0i64..2)).collect();
        let scores: Vec<i64> = w_true
            .iter()
            .map(|row| row.iter().zip(&xi).map(|(&w, &v)| w * v).sum())
            .collect();
        let lab = scores
            .iter()
            .enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i)
            .unwrap();
        x.push(xi);
        label.push(lab);
    }
    Dataset { x, label, n_features, n_classes, w_true }
}

impl Dataset {
    fn onehot(&self, i: usize) -> Vec<i64> {
        (0..self.n_classes)
            .map(|c| if c == self.label[i] { TARGET } else { 0 })
            .collect()
    }

    /// Draw one held-out example from the *same* ground-truth model as the
    /// training set (so a correct prediction is actually meaningful).
    fn held_out(&self, seed: u64) -> (Vec<i64>, usize) {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i64> = (0..self.n_features).map(|_| rng.gen_range(0i64..2)).collect();
        let scores: Vec<i64> = self
            .w_true
            .iter()
            .map(|row| row.iter().zip(&x).map(|(&w, &v)| w * v).sum())
            .collect();
        let lab = scores.iter().enumerate().max_by_key(|(_, &s)| s).map(|(i, _)| i).unwrap();
        (x, lab)
    }
}

/// Plaintext integer-SGD reference — the exact ground truth the encrypted
/// run must match. `w`/`b` are updated in place; returns nothing.
fn plain_step(
    w: &mut [Vec<i64>],
    b: &mut [i64],
    x: &[i64],
    y: &[i64],
    lr_shift: usize,
) {
    let n_classes = w.len();
    let z: Vec<i64> = (0..n_classes)
        .map(|c| w[c].iter().zip(x).map(|(&wi, &xi)| wi * xi).sum::<i64>() + b[c])
        .collect();
    for c in 0..n_classes {
        let delta = z[c] - y[c];
        for i in 0..x.len() {
            w[c][i] -= (delta * x[i]) >> lr_shift;
        }
        b[c] -= delta >> lr_shift;
    }
}

fn plain_predict(w: &[Vec<i64>], b: &[i64], x: &[i64]) -> usize {
    let z: Vec<i64> = (0..w.len())
        .map(|c| w[c].iter().zip(x).map(|(&wi, &xi)| wi * xi).sum::<i64>() + b[c])
        .collect();
    z.iter().enumerate().max_by(|a, bb| a.1.cmp(bb.1)).map(|(i, _)| i).unwrap()
}

fn loss(w: &[Vec<i64>], b: &[i64], ds: &Dataset) -> i64 {
    let mut total = 0i64;
    for s in 0..ds.x.len() {
        let y = ds.onehot(s);
        for c in 0..ds.n_classes {
            let z: i64 = w[c].iter().zip(&ds.x[s]).map(|(&wi, &xi)| wi * xi).sum::<i64>() + b[c];
            let d = z - y[c];
            total += d * d;
        }
    }
    total / ds.x.len() as i64
}

fn accuracy(w: &[Vec<i64>], b: &[i64], ds: &Dataset) -> f64 {
    let correct = (0..ds.x.len())
        .filter(|&s| plain_predict(w, b, &ds.x[s]) == ds.label[s])
        .count();
    correct as f64 / ds.x.len() as f64
}

// ─── The private SGD step (real 3-party protocol, gadget-composed) ────────────

/// Run `epochs` full passes of SGD over `ds` under the real ABY3 protocol,
/// starting from zero weights. Returns the final revealed weights/bias and
/// party-0 wire stats. Structurally identical to `plain_step`, so the
/// result is bit-exact with the plaintext reference.
#[allow(clippy::type_complexity)]
fn private_train(
    ds: &Dataset,
    epochs: usize,
    lr_shift: usize,
    profile: NetworkProfile,
) -> (Vec<Vec<i64>>, Vec<i64>, CommStats) {
    let c = ds.n_classes;
    let d = ds.n_features;
    let x_data: Vec<Vec<i64>> = ds.x.clone();
    let onehots: Vec<Vec<i64>> = (0..ds.x.len()).map(|s| ds.onehot(s)).collect();

    let (mut per_party, comm) = run_parties(profile, move |me| {
        let x_data = x_data.clone();
        let onehots = onehots.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            let share = |be: &mut TransportBackend<MpcTransport>, v: i64| {
                be.share_secret(if me == 0 { residue(v, NBITS) } else { 0 }, NBITS, 0)
            };
            // Weights/bias start at public zero.
            let mut w: Vec<Vec<ShareRef>> = (0..c)
                .map(|_| (0..d).map(|_| be.share_public(0, NBITS)).collect())
                .collect();
            let mut b: Vec<ShareRef> = (0..c).map(|_| be.share_public(0, NBITS)).collect();

            for _ in 0..epochs {
                for s in 0..x_data.len() {
                    let x: Vec<ShareRef> = x_data[s].iter().map(|&v| share(be, v)).collect();
                    let y: Vec<ShareRef> = onehots[s].iter().map(|&v| share(be, v)).collect();
                    // forward z = W·x + b
                    let z: Vec<ShareRef> = (0..c)
                        .map(|cc| {
                            let mut acc = b[cc].clone();
                            for i in 0..d {
                                let p = be.mul(&w[cc][i], &x[i]);
                                acc = be.add(&acc, &p);
                            }
                            acc
                        })
                        .collect();
                    // update W -= (δ·x) ≫ lr, b -= δ ≫ lr
                    for cc in 0..c {
                        let delta = be.sub(&z[cc], &y[cc]);
                        for i in 0..d {
                            let g = be.mul(&delta, &x[i]);
                            let upd = arith_shr(be, &g, lr_shift, NBITS);
                            w[cc][i] = be.sub(&w[cc][i], &upd);
                        }
                        let gb = arith_shr(be, &delta, lr_shift, NBITS);
                        b[cc] = be.sub(&b[cc], &gb);
                    }
                }
            }
            // Reveal the trained model.
            let wr: Vec<Vec<i64>> = w
                .iter()
                .map(|row| row.iter().map(|s| to_signed(be.reveal(s, None))).collect())
                .collect();
            let br: Vec<i64> = b.iter().map(|s| to_signed(be.reveal(s, None))).collect();
            (wr, br)
        }
    });
    let (w, b) = per_party.swap_remove(0);
    (w, b, comm)
}

/// Private inference: classify `x` under the trained (secret-shared) model,
/// revealing only the class index (`gadgets::argmax`).
fn private_predict(w: &[Vec<i64>], b: &[i64], x: &[i64], profile: NetworkProfile) -> usize {
    let w = w.to_vec();
    let b = b.to_vec();
    let x = x.to_vec();
    let (classes, _) = run_parties(profile, move |me| {
        let w = w.clone();
        let b = b.clone();
        let x = x.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            let share = |be: &mut TransportBackend<MpcTransport>, v: i64| {
                be.share_secret(if me == 0 { residue(v, NBITS) } else { 0 }, NBITS, 0)
            };
            let xs: Vec<ShareRef> = x.iter().map(|&v| share(be, v)).collect();
            let logits: Vec<ShareRef> = (0..w.len())
                .map(|cc| {
                    let mut acc = share(be, b[cc]);
                    for i in 0..x.len() {
                        let wi = share(be, w[cc][i]);
                        let p = be.mul(&wi, &xs[i]);
                        acc = be.add(&acc, &p);
                    }
                    acc
                })
                .collect();
            let idx = argmax(be, &logits, NBITS);
            (be.reveal(&idx, None) as u64 & ((1u64 << NBITS) - 1)) as usize
        }
    });
    classes[0]
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct TrainReport {
    pub epochs: usize,
    pub loss_start: i64,
    pub loss_end: i64,
    pub train_accuracy: f64,
    pub exact_match: bool,
    pub messages: u64,
    pub bytes: u64,
    pub bootstraps: u64,
}

const USAGE: &str = "usage: baru bench-train [options]

Private training: an SGD least-squares classifier trained end to end under
the real 3-party ABY3 protocol (data secret-shared, gradients and updates
oblivious, composed from the gadget library), then used for private
inference. Verified bit-exact against a plaintext integer-SGD reference.

options:
  --samples N          training set size (default 12)
  --features D         input features (default 6)
  --classes C          number of classes (default 3)
  --epochs E           passes over the data (default 8)
  --lr-shift K         learning rate = 2^-K (default 8)
  --rtt-ms N           round-trip latency (default 0)
  --seed N             RNG seed (default 1)
  --json               emit JSON
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut n = 12usize;
    let mut d = 6usize;
    let mut c = 3usize;
    let mut epochs = 20usize;
    let mut lr_shift = 4usize;
    let mut rtt_ms = 0u64;
    let mut seed = 1u64;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--samples" => { i += 1; n = args.get(i).ok_or("--samples needs a value")?.parse().map_err(|_| "bad samples".to_string())?; }
            "--features" => { i += 1; d = args.get(i).ok_or("--features needs a value")?.parse().map_err(|_| "bad features".to_string())?; }
            "--classes" => { i += 1; c = args.get(i).ok_or("--classes needs a value")?.parse().map_err(|_| "bad classes".to_string())?; }
            "--epochs" => { i += 1; epochs = args.get(i).ok_or("--epochs needs a value")?.parse().map_err(|_| "bad epochs".to_string())?; }
            "--lr-shift" => { i += 1; lr_shift = args.get(i).ok_or("--lr-shift needs a value")?.parse().map_err(|_| "bad lr-shift".to_string())?; }
            "--rtt-ms" => { i += 1; rtt_ms = args.get(i).ok_or("--rtt-ms needs a value")?.parse().map_err(|_| "bad rtt-ms".to_string())?; }
            "--seed" => { i += 1; seed = args.get(i).ok_or("--seed needs a value")?.parse().map_err(|_| "bad seed".to_string())?; }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    let profile = NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt_ms * 500));
    let ds = make_dataset(n, d, c, seed);

    // Plaintext reference (the exact ground truth + loss curve).
    let mut pw = vec![vec![0i64; d]; c];
    let mut pb = vec![0i64; c];
    let loss_start = loss(&pw, &pb, &ds);
    let mut curve = vec![loss_start];
    for _ in 0..epochs {
        for s in 0..ds.x.len() {
            let y = ds.onehot(s);
            plain_step(&mut pw, &mut pb, &ds.x[s], &y, lr_shift);
        }
        curve.push(loss(&pw, &pb, &ds));
    }
    let loss_end = *curve.last().unwrap();
    let train_acc = accuracy(&pw, &pb, &ds);

    // Private training on the real protocol; verify bit-exact.
    let (w, b, comm) = private_train(&ds, epochs, lr_shift, profile);
    let exact = w == pw && b == pb;

    let report = TrainReport {
        epochs,
        loss_start,
        loss_end,
        train_accuracy: train_acc,
        exact_match: exact,
        messages: comm.messages,
        bytes: comm.bytes,
        bootstraps: 0,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
        return Ok(());
    }

    println!("Private training — SGD least-squares classifier under real 3-party ABY3");
    println!("data secret-shared; forward/gradient/update all oblivious (gadget-composed)");
    println!(
        "dataset: {n} samples, {d} features, {c} classes | epochs={epochs}, lr=2^-{lr_shift}\n"
    );
    println!("  loss (MSE) curve : {curve:?}");
    println!("  loss {} -> {}   ({:.0}% reduction)", loss_start, loss_end,
        100.0 * (loss_start - loss_end) as f64 / loss_start.max(1) as f64);
    println!("  train accuracy   : {:.0}%", train_acc * 100.0);
    println!(
        "  encrypted == plaintext model : {}",
        if exact { "EXACT (bit-for-bit)" } else { "MISMATCH" }
    );
    println!("  FHE bootstraps   : 0  (MPC training; model then deployable for no-bootstrap inference)");
    println!("  MPC wire traffic : {} messages, {} bytes", comm.messages, comm.bytes);

    // The lifecycle: classify a fresh example (drawn from the same
    // ground-truth model) under the trained model, revealing only the class.
    let (tx, tlabel) = ds.held_out(seed ^ 0xABCD);
    let pred = private_predict(&w, &b, &tx, profile);
    println!(
        "\n  private inference on a held-out example -> class {pred}  (true {tlabel}, {})",
        if pred == tlabel { "correct" } else { "misclassified" }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plaintext_sgd_reduces_loss_on_separable_data() {
        let ds = make_dataset(16, 6, 3, 2);
        let mut w = vec![vec![0i64; 6]; 3];
        let mut b = vec![0i64; 3];
        let l0 = loss(&w, &b, &ds);
        for _ in 0..20 {
            for s in 0..ds.x.len() {
                let y = ds.onehot(s);
                plain_step(&mut w, &mut b, &ds.x[s], &y, 4);
            }
        }
        let l1 = loss(&w, &b, &ds);
        assert!(l1 < l0, "loss must decrease: {l0} -> {l1}");
        assert!(accuracy(&w, &b, &ds) > 0.6, "should fit separable data reasonably");
    }

    #[test]
    #[ignore] // real 3-party protocol — slow in debug, fine in release
    fn private_training_matches_plaintext_exactly() {
        let ds = make_dataset(8, 4, 2, 5);
        let (mut pw, mut pb) = (vec![vec![0i64; 4]; 2], vec![0i64; 2]);
        for _ in 0..4 {
            for s in 0..ds.x.len() {
                let y = ds.onehot(s);
                plain_step(&mut pw, &mut pb, &ds.x[s], &y, 4);
            }
        }
        let (w, b, comm) = private_train(&ds, 4, 4, NetworkProfile::INSTANT);
        assert_eq!(w, pw, "encrypted weights must match plaintext exactly");
        assert_eq!(b, pb, "encrypted bias must match plaintext exactly");
        assert!(comm.messages > 0, "must use the real MPC protocol");
    }
}
