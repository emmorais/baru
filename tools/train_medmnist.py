#!/usr/bin/env python3
"""Train + quantise a 28x28 classifier for Baru's private-CNN pipeline.

Replaces the lost scratchpad/train_modeA.py (kept in tools/, not scratchpad,
because that is how the original was lost), and generalised: it handles any
grayscale 28x28 MedMNIST v2 dataset *and* plain MNIST, so every embedded
weight file in the repo can be regenerated from this one script.  Emits
examples/data/<name>_cnn.rs in exactly the integer fixed-point scheme
`CnnModel::forward` (src/bench/cnn.rs) evaluates.

Architecture (fixed by the Rust side: K=3, POOL=2, one extra conv, one hidden FC):
    1x28x28 -> conv(OC,3x3) -> ReLU -> pool2 -> conv(C2,3x3) -> ReLU -> pool2
            -> dense(->HN) -> ReLU -> dense(->NCLASSES) -> argmax

Quantisation.  The deployed integer net keeps every activation at the *input*
scale (pixels are raw 0..255), which fixes the mapping from float weights
trained on x/255:

    conv/dense weights   w_q = round(W * 2^shift)
    conv bias            cbias = round(B * 255)        (forward adds cbias<<shift)
    dense bias           b_q   = round(B * 255 * 2^shift)

Derivation: with A_l = 255*a_l, forward's `relu(sum w_q A_l + bias) >> shift`
equals 255*relu(W a_l + B) exactly when the rules above hold.  ReLU is
positively homogeneous, so before quantising we rescale each layer's weights
by a per-layer gain (and divide the next layer's by it) -- this leaves the
network's function untouched but lifts small weights away from the rounding
floor, which is where quantisation error would otherwise come from.

Get the data:

  MedMNIST  curl -sLo pneumoniamnist.npz \
    "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
  MNIST     for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
                     t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
              curl -sLo mnist/$f.gz \
                "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"; done

Usage: python3 tools/train_medmnist.py <npz|mnist-dir> <rust-out>
         [--epochs N] [--n-test N] [--test-order first|random] [--linear]
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OC, C2, HN, K, POOL, SHIFT = 8, 16, 128, 3, 2, 8

# ─── data loading: MedMNIST .npz, or a directory of MNIST idx.gz files ───────


def _idx(path):
    """Read an IDX file (MNIST's format), gzipped or not."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        blob = f.read()
    ndim = blob[3]
    dims = [int.from_bytes(blob[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(blob[4 + 4 * ndim :], dtype=np.uint8).reshape(dims)


def load_dataset(path, val_frac=1 / 12):
    """Return the six arrays the trainer works from, in MedMNIST's layout.

    MNIST ships no validation split, so we carve the *last* `val_frac` of the
    training set off for one -- needed here because the quantisation gain has
    to be chosen without ever touching test."""
    if path.endswith(".npz"):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    names = {
        "train_images": "train-images-idx3-ubyte",
        "train_labels": "train-labels-idx1-ubyte",
        "test_images": "t10k-images-idx3-ubyte",
        "test_labels": "t10k-labels-idx1-ubyte",
    }
    raw = {}
    for key, stem in names.items():
        for cand in (f"{stem}.gz", stem):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                raw[key] = _idx(full)
                break
        else:
            sys.exit(f"{path}: missing {stem}[.gz]")
    n_val = int(len(raw["train_images"]) * val_frac)
    return {
        "train_images": raw["train_images"][:-n_val],
        "train_labels": raw["train_labels"][:-n_val].reshape(-1, 1),
        "val_images": raw["train_images"][-n_val:],
        "val_labels": raw["train_labels"][-n_val:].reshape(-1, 1),
        "test_images": raw["test_images"],
        "test_labels": raw["test_labels"].reshape(-1, 1),
    }


class Net(nn.Module):
    """Exactly the layer stack CnnModel::forward implements (bias everywhere,
    valid padding, floor pooling)."""

    def __init__(self, nclasses):
        super().__init__()
        self.c1 = nn.Conv2d(1, OC, K)
        self.c2 = nn.Conv2d(OC, C2, K)
        self.fc1 = nn.Linear(C2 * 5 * 5, HN)
        self.fc2 = nn.Linear(HN, nclasses)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), POOL, POOL)  # 26 -> 13
        x = F.max_pool2d(F.relu(self.c2(x)), POOL, POOL)  # 11 -> 5 (floor)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


# ─── integer reference: a line-for-line mirror of CnnModel::forward ──────────


def int_forward(img, q):
    """img: (28,28) int64 raw pixels. Returns (class, max_abs)."""
    x = img.astype(np.int64)
    max_abs = int(np.abs(x).max())

    def conv(inp, w, b, in_c, in_h, in_w, out_c):
        nonlocal max_abs
        oh, ow = in_h - K + 1, in_w - K + 1
        # im2col over (in_c, ky, kx) -- the same index order the Rust uses.
        cols = np.empty((in_c * K * K, oh * ow), dtype=np.int64)
        for ic in range(in_c):
            for ky in range(K):
                for kx in range(K):
                    cols[(ic * K + ky) * K + kx] = inp[
                        ic, ky : ky + oh, kx : kx + ow
                    ].reshape(-1)
        v = w @ cols + (b.reshape(-1, 1) << SHIFT)
        max_abs = max(max_abs, int(np.abs(v).max()))
        return np.where(v >= 0, v >> SHIFT, 0).reshape(out_c, oh, ow)

    def pool(a):
        c, h, w = a.shape
        oh, ow = (h - POOL) // POOL + 1, (w - POOL) // POOL + 1
        out = a[:, : oh * POOL, : ow * POOL].reshape(c, oh, POOL, ow, POOL)
        return out.max(axis=(2, 4))

    a = pool(conv(x[None, :, :], q["k1"], q["b1"], 1, 28, 28, OC))
    a = pool(conv(a, q["k2"], q["b2"], OC, a.shape[1], a.shape[2], C2))
    h = q["w3"] @ a.reshape(-1) + q["b3"]
    max_abs = max(max_abs, int(np.abs(h).max()))
    h = np.where(h >= 0, h >> SHIFT, 0)
    logits = q["w4"] @ h + q["b4"]
    max_abs = max(max_abs, int(np.abs(logits).max()))
    return int(np.argmax(logits)), max_abs


def quantise(net, gain_target=None):
    """Float weights -> the integer scheme, after a function-preserving
    per-layer rescale.  Returns the quantised tensors as int64 arrays.

    `gain_target` (if given) is the max |w_q| each layer is scaled towards.
    Two opposing pressures fix the useful range: larger weights mean a smaller
    relative rounding error, but every layer's gain also multiplies that
    layer's activations, and the product across layers has to stay inside the
    2^28 the depth-1 FHE plaintext modulus allows.  `pick_gain` sweeps it on
    the validation split.

    ReLU is positively homogeneous, so scaling layer l by g and layer l+1 by
    1/g leaves the float function untouched.  The *output* layer is the
    exception in the other direction: argmax is scale-invariant, so its gain
    needs no compensation at all -- which is also why it must get one.
    Forgetting that starves the logit weights to zero and the classifier
    collapses to its majority class."""
    W = [p.detach().numpy().astype(np.float64) for p in
         (net.c1.weight, net.c2.weight, net.fc1.weight, net.fc2.weight)]
    B = [p.detach().numpy().astype(np.float64) for p in
         (net.c1.bias, net.c2.bias, net.fc1.bias, net.fc2.bias)]

    if gain_target is not None:
        for l in range(len(W)):
            peak = np.abs(W[l]).max() * (1 << SHIFT)
            g = gain_target / peak if peak > 0 else 1.0
            W[l] *= g
            B[l] *= g
            if l + 1 < len(W):  # last layer: argmax absorbs the gain
                W[l + 1] /= g

    s = 1 << SHIFT
    q = {
        "k1": np.rint(W[0].reshape(OC, K * K) * s).astype(np.int64),
        "b1": np.rint(B[0] * 255.0).astype(np.int64),
        "k2": np.rint(W[1].reshape(C2, OC * K * K) * s).astype(np.int64),
        "b2": np.rint(B[1] * 255.0).astype(np.int64),
        "w3": np.rint(W[2] * s).astype(np.int64),
        "b3": np.rint(B[2] * 255.0 * s).astype(np.int64),
        "w4": np.rint(W[3] * s).astype(np.int64),
        "b4": np.rint(B[3] * 255.0 * s).astype(np.int64),
    }
    return q


def score_int(q, imgs_u8, labels, progress_every=0):
    """Integer-net accuracy over raw-uint8 images. Returns (acc, max_abs, preds)."""
    preds, max_abs = [], 0
    for i, im in enumerate(imgs_u8):
        c, m = int_forward(im, q)
        preds.append(c)
        max_abs = max(max_abs, m)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    integer-scored {i + 1}/{len(imgs_u8)}", flush=True)
    preds = np.array(preds)
    return float((preds == labels).mean()), max_abs, preds


def pick_gain(net, xva_u8, yva, targets=(None, 64.0, 128.0, 256.0, 512.0)):
    """Choose the quantisation gain on the VALIDATION split (never on test),
    rejecting any candidate whose activations pass the 2^28 the depth-1 noise
    budget allows."""
    best = None
    for t in targets:
        q = quantise(net, t)
        acc, max_abs, _ = score_int(q, xva_u8, yva)
        needed = int(max_abs).bit_length() + 2
        flag = "" if needed <= 30 else "  <- OVERFLOWS, rejected"
        print(f"  gain {str(t):>6}: val-int {acc * 100:6.2f}%  "
              f"max|act| = 2^{needed - 2}{flag}", flush=True)
        if needed <= 30 and (best is None or acc > best[1]):
            best = (t, acc, q)
    if best is None:
        sys.exit("every candidate gain overflows the FHE plaintext modulus")
    print(f"chosen gain {best[0]} (val-int {best[1] * 100:.2f}%)", flush=True)
    return best[2], best[0]


# ─── Rust emission ───────────────────────────────────────────────────────────


def rust_array(name, rows, elem_ty, static_=True):
    kw = "pub static" if static_ else "pub const"
    if isinstance(rows[0], (list, np.ndarray)):
        inner = len(rows[0])
        body = ",\n".join(
            "    [" + ", ".join(str(int(v)) for v in r) + "]" for r in rows
        )
        return f"{kw} {name}: [[{elem_ty}; {inner}]; {len(rows)}] = [\n{body},\n];\n"
    body = ", ".join(str(int(v)) for v in rows)
    return f"{kw} {name}: [{elem_ty}; {len(rows)}] = [{body}];\n"


def emit(path, q, images, labels, meta):
    o = []
    o.append(f"""// GENERATED - do not edit. Trained + quantised two-conv CNN on
// {meta['dataset']} ({meta['provenance']}), plus a real test subset.
// See tools/train_medmnist.py.
//
// Task: {meta['task']}
// Architecture: 1x28x28 -> conv({OC},3x3) -> ReLU>>{SHIFT} -> pool
//   -> conv({C2},3x3) -> ReLU>>{SHIFT} -> pool -> dense(400->{HN})
//   -> ReLU>>{SHIFT} -> dense({HN}->{meta['nclasses']}) -> argmax
// Full {meta['n_full']}-image integer test accuracy: {meta['full_acc'] * 100:.2f}%
// (majority-class baseline {meta['baseline'] * 100:.2f}%)

pub const DATASET: &str = "{meta['dataset']}";
pub const TASK: &str = "{meta['task']}";
pub const FULL_TEST_ACCURACY: f64 = {meta['full_acc']:.6f};
pub const MAJORITY_BASELINE: f64 = {meta['baseline']:.6f};
pub const FULL_TEST_N: usize = {meta['n_full']};
pub const OC: usize = {OC};
pub const K: usize = {K};
pub const NCLASSES: usize = {meta['nclasses']};
pub const HN: usize = {HN};
pub const SHIFT: u32 = {SHIFT};
pub const CONV2_IN_C: usize = {OC};
pub const CONV2_IN_H: usize = 13;
pub const CONV2_IN_W: usize = 13;
pub const CLASS_NAMES: [&str; {meta['nclasses']}] = [{
    ", ".join('"' + c + '"' for c in meta['class_names'])
}];
""")
    o.append(rust_array("KERNELS", q["k1"], "i64"))
    o.append(rust_array("CBIAS", q["b1"], "i64", static_=False))
    o.append(rust_array("CONV2_W", q["k2"], "i64"))
    o.append(rust_array("CONV2_B", q["b2"], "i64", static_=False))
    o.append(rust_array("FC1_W", q["w3"], "i64"))
    o.append(rust_array("FC1_B", q["b3"], "i64"))
    o.append(rust_array("DENSE_W", q["w4"], "i64"))
    o.append(rust_array("DENSE_B", q["b4"], "i64"))
    o.append(f"""
pub const N_TEST: usize = {len(labels)};
""")
    o.append(rust_array("TEST_LABELS", labels, "u8"))
    o.append(rust_array("TEST_IMAGES", [im.reshape(-1) for im in images], "i64"))
    o.append("""
/// Hidden dense layers in order, each `(weights[out][in], bias[out])`.
pub fn hidden_layers() -> Vec<(Vec<Vec<i64>>, Vec<i64>)> {
    vec![(FC1_W.iter().map(|r| r.to_vec()).collect(), FC1_B.to_vec())]
}

/// Everything `bench-medmnist` needs, in the crate's normalised form.
pub fn trained() -> crate::bench::medmnist::TrainedCnn {
    crate::bench::medmnist::TrainedCnn {
        dataset: DATASET,
        task: TASK,
        class_names: CLASS_NAMES.to_vec(),
        kernels: KERNELS.iter().map(|k| k.to_vec()).collect(),
        cbias: CBIAS.to_vec(),
        conv2: (
            CONV2_W.iter().map(|r| r.to_vec()).collect(),
            CONV2_B.to_vec(),
            CONV2_IN_C,
            CONV2_IN_H,
            CONV2_IN_W,
        ),
        hidden: hidden_layers(),
        dense_w: DENSE_W.iter().map(|r| r.to_vec()).collect(),
        dense_b: DENSE_B.to_vec(),
        shift: SHIFT,
        images: TEST_IMAGES.iter().map(|im| im.to_vec()).collect(),
        labels: TEST_LABELS.to_vec(),
        full_test_accuracy: FULL_TEST_ACCURACY,
        majority_baseline: MAJORITY_BASELINE,
        full_test_n: FULL_TEST_N,
    }
}
""")
    with open(path, "w") as f:
        f.write("".join(o))


# ─── linear discriminator (for examples/mnist.rs's scheme comparison) ────────
#
# A different, much smaller model: one neuron over the 784 raw pixels with
# Q16.16 public weights, which is what examples/mnist.rs compiles to a
# 2365-gate circuit and runs on TFHE and CKKS.  Only 2-class datasets.
#
# Fixed-point mapping.  The demo's source reads `(+ b/128 (dot pixels
# '(w0/128 ...)))` over raw pixels, so a Q16.16 weight is `w_int/128` (128 =
# 2^7, exactly representable) and every term lands at raw `w_int * 512 * px`.
# Training on x/255 gives float W, so `w_int = W/255 * 128 * g` and
# `BIAS = B * g * 128` for a free gain `g` -- free because comparing two
# scores, and thresholding at zero, are both scale-invariant.
#
# `g` is chosen by measuring the accumulator: the MPC path uses 32-bit shares
# (the REPL convention the demo follows), so the running Q16.16 partial sum
# has to stay inside 2^30.  Chest X-rays are dense where digits are mostly
# zero, so this bound actually binds here and a gain picked by eye would
# silently wrap the shares.

MPC_ACC_BUDGET = 1 << 30
LIN_WEIGHT_DIV = 128.0


def lin_quantise(W, B, g):
    """Float logistic weights -> (w_int[784], BIAS) at gain g."""
    w_int = np.rint(W / 255.0 * LIN_WEIGHT_DIV * g).astype(np.int64)
    bias = int(round(B * g * LIN_WEIGHT_DIV))
    return w_int, bias


def lin_score_raw(w_int, bias, imgs_u8):
    """Raw Q16.16 scores, and the largest running partial sum any image
    produces -- the quantity the 32-bit MPC shares have to hold."""
    terms = imgs_u8.reshape(len(imgs_u8), -1) * (w_int * 512)
    partial = np.cumsum(terms, axis=1) + bias * 512
    return partial[:, -1], int(np.abs(partial).max())


def train_linear(args, d, nclasses):
    if nclasses != 2:
        sys.exit(f"--linear needs a 2-class dataset; {args.npz} has {nclasses}")
    xtr = torch.from_numpy(d["train_images"].astype(np.float32) / 255.0).flatten(1)
    ytr = torch.from_numpy(d["train_labels"].ravel().astype(np.float32))
    xte_u8 = d["test_images"].astype(np.int64)
    yte = d["test_labels"].ravel().astype(np.int64)

    lin = nn.Linear(784, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                lin(xtr[idx]).squeeze(1), ytr[idx]
            ).backward()
            opt.step()
    W = lin.weight.detach().numpy().ravel().astype(np.float64)
    B = float(lin.bias.detach().numpy()[0])
    with torch.no_grad():
        xte = torch.from_numpy(xte_u8.astype(np.float32) / 255.0).flatten(1)
        float_acc = float(((lin(xte).squeeze(1) > 0).numpy().astype(np.int64) == yte).mean())

    # Sweep the gain on VALIDATION (never on test), rejecting any gain whose
    # accumulator would wrap the demo's 32-bit MPC shares.  The peak is
    # measured over validation *and* test images, since the bound has to hold
    # for whatever the demo is handed.
    xva_u8 = d["val_images"].astype(np.int64)
    yva = d["val_labels"].ravel().astype(np.int64)
    print("linear gain sweep (accuracy on validation, bound over all images):",
          flush=True)
    best = None
    for g in (1.0, 4.0, 16.0, 64.0, 128.0, 256.0, 512.0, 1024.0):
        w_int, bias = lin_quantise(W, B, g)
        va_scores, va_peak = lin_score_raw(w_int, bias, xva_u8)
        _, te_peak = lin_score_raw(w_int, bias, xte_u8)
        peak = max(va_peak, te_peak)
        acc = float(((va_scores > 0).astype(np.int64) == yva).mean())
        fits = peak < MPC_ACC_BUDGET
        print(f"  gain {g:6.1f}: val acc {acc * 100:6.2f}%  max|partial| = 2^"
              f"{int(peak).bit_length()}  max|w| = {int(np.abs(w_int).max()):5d}"
              f"{'' if fits else '  <- exceeds 32-bit shares, rejected'}", flush=True)
        if fits and (best is None or acc > best[0]):
            best = (acc, g, w_int, bias, peak)
    if best is None:
        sys.exit("no gain keeps the Q16.16 accumulator inside 32-bit MPC shares")
    va_acc, g, w_int, bias, peak = best
    scores, _ = lin_score_raw(w_int, bias, xte_u8)
    acc = float(((scores > 0).astype(np.int64) == yte).mean())
    print(f"chosen gain {g} (val {va_acc * 100:.2f}%) -> test: int {acc * 100:.2f}%, "
          f"float {float_acc * 100:.2f}%")

    # Two demo images: the most confidently scored example of each class, so
    # the backend-agreement check is not decided by a near-tie.
    scores, _ = lin_score_raw(w_int, bias, xte_u8)
    pos = int(np.argmax(np.where(yte == 1, scores, -np.inf)))
    neg = int(np.argmin(np.where(yte == 0, scores, np.inf)))
    names = args.classes.split(",") if args.classes else ["class0", "class1"]
    baseline = float(np.bincount(yte).max() / len(yte))

    out = [f"""// GENERATED - do not edit. Linear discriminator over 784 raw pixels of
// {args.dataset or args.npz} (MedMNIST v2, CC BY 4.0), plus one sample image
// per class. See tools/train_medmnist.py --linear.
//
// Task: {args.task}
// score(img) = BIAS/{LIN_WEIGHT_DIV:.0f} + dot(img, WEIGHTS/{LIN_WEIGHT_DIV:.0f})   (Q16.16)
// predicts "{names[1]}" iff score > 0.
// Test-set accuracy of this integer model: {acc * 100:.2f}% over {len(yte)} images
// (float logistic {float_acc * 100:.2f}%, majority-class baseline {baseline * 100:.2f}%).
// Largest Q16.16 partial sum over the test set: 2^{int(peak).bit_length()}
// (must stay under 2^30 for the demo's 32-bit MPC shares).

pub const DATASET: &str = "{args.dataset or args.npz}";
pub const TASK: &str = "{args.task}";
pub const NEG_LABEL: &str = "{names[0]}";
pub const POS_LABEL: &str = "{names[1]}";
pub const BIAS: i64 = {bias};
pub const TEST_ACCURACY: f64 = {acc:.6f};
pub const MAJORITY_BASELINE: f64 = {baseline:.6f};
pub const TEST_N: usize = {len(yte)};
"""]
    out.append(rust_array("WEIGHTS", w_int, "i64", static_=False))
    out.append(rust_array("IMG_POS", xte_u8[pos].reshape(-1), "i64", static_=False))
    out.append(rust_array("IMG_NEG", xte_u8[neg].reshape(-1), "i64", static_=False))
    with open(args.rust_out, "w") as f:
        f.write("".join(out))
    print(f"wrote {args.rust_out}: score({names[1]})={scores[pos] / 65536.0:.4f} "
          f"> score({names[0]})={scores[neg] / 65536.0:.4f}")
    print(json.dumps({"dataset": args.dataset, "int_acc": acc, "float_acc": float_acc,
                      "gain": g, "baseline": baseline, "peak": peak}))


# ─── main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", metavar="npz-or-mnist-dir")
    ap.add_argument("rust_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--test-order", choices=("first", "random"), default="random",
                    help="embed the FIRST n-test images of the test set, or a "
                         "random sample of them (default random; the embedded "
                         "MNIST subset uses 'first')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", default="", help="comma-separated class names")
    ap.add_argument("--dataset", default="", help="human-readable dataset name")
    ap.add_argument("--task", default="", help="one-line description of the task")
    ap.add_argument("--provenance", default="",
                    help="source/license line for the generated file's header "
                         "(defaults by input type; do not mis-attribute MNIST "
                         "to MedMNIST)")
    ap.add_argument("--linear", action="store_true",
                    help="train the single-neuron Q16.16 discriminator that "
                         "examples/mnist.rs runs on TFHE/CKKS (2-class only)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_dataset(args.npz)
    xtr = d["train_images"].astype(np.float32) / 255.0
    ytr = d["train_labels"].ravel().astype(np.int64)
    xva = d["val_images"].astype(np.float32) / 255.0
    yva = d["val_labels"].ravel().astype(np.int64)
    xte_u8 = d["test_images"].astype(np.int64)
    yte = d["test_labels"].ravel().astype(np.int64)
    nclasses = int(ytr.max()) + 1
    print(f"{args.npz}: train {xtr.shape} val {xva.shape} test {xte_u8.shape}, "
          f"{nclasses} classes", flush=True)

    if args.linear:
        return train_linear(args, d, nclasses)

    net = Net(nclasses)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr = torch.from_numpy(xtr).unsqueeze(1)
    Ytr = torch.from_numpy(ytr)
    Xva = torch.from_numpy(xva).unsqueeze(1)
    Yva = torch.from_numpy(yva)

    best, best_state = -1.0, None
    bs = 128
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            va = (net(Xva).argmax(1) == Yva).float().mean().item()
        if va > best:
            best = va
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  epoch {ep + 1:3d}  loss {loss.item():.4f}  val {va * 100:.2f}%",
              flush=True)

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(xte_u8.astype(np.float32) / 255.0).unsqueeze(1)
        float_acc = (net(Xte).argmax(1) == torch.from_numpy(yte)).float().mean().item()
    print(f"float test accuracy: {float_acc * 100:.2f}% (best val {best * 100:.2f}%)",
          flush=True)

    # Pick the quantisation gain on validation, then score on the FULL test set.
    print("quantisation gain sweep (validation split):", flush=True)
    q, gain = pick_gain(net, d["val_images"].astype(np.int64), yva)
    int_acc, max_abs, preds = score_int(q, xte_u8, yte, progress_every=2000)
    needed = int(max_abs).bit_length() + 2
    baseline = float(np.bincount(yte, minlength=nclasses).max() / len(yte))
    with torch.no_grad():
        float_preds = net(Xte).argmax(1).numpy()
    agree = float((preds == float_preds).mean())
    print(f"integer test accuracy: {int_acc * 100:.2f}%  "
          f"(float {float_acc * 100:.2f}%, majority {baseline * 100:.2f}%)")
    print(f"integer/float prediction agreement: {agree * 100:.2f}%")
    print(f"max |activation| = {max_abs} -> needs 2^{needed} "
          f"({'OK' if needed <= 30 else 'TOO WIDE for the depth-1 noise budget'})")
    if needed > 30:
        sys.exit("quantisation overflows the FHE plaintext modulus; lower the gain")

    # Embed a slice of the real test set: either its first `n_test` images
    # (what the original MNIST weight file did) or a random sample of them.
    n_test = min(args.n_test, len(xte_u8))
    if args.test_order == "first":
        sel = np.arange(n_test)
    else:
        rng = np.random.default_rng(args.seed)
        sel = rng.permutation(len(xte_u8))[:n_test]
        sel.sort()
    names = (args.classes.split(",") if args.classes
             else [str(i) for i in range(nclasses)])
    assert len(names) == nclasses, f"{len(names)} names for {nclasses} classes"
    meta = {
        "dataset": args.dataset or args.npz,
        "provenance": args.provenance or (
            "MedMNIST v2, CC BY 4.0" if args.npz.endswith(".npz")
            else "LeCun, Cortes & Burges — the original handwritten-digit benchmark"
        ),
        "task": args.task or "classification",
        "nclasses": nclasses,
        "full_acc": int_acc,
        "baseline": baseline,
        "n_full": len(xte_u8),
        "class_names": names,
    }
    emit(args.rust_out, q, [xte_u8[i] for i in sel], yte[sel].astype(np.uint8), meta)
    subset_acc = float((preds[sel] == yte[sel]).mean())
    print(f"wrote {args.rust_out}: {n_test} embedded test images, "
          f"subset integer accuracy {subset_acc * 100:.2f}%")
    print(json.dumps({"dataset": meta["dataset"], "int_acc": int_acc,
                      "float_acc": float_acc, "subset_acc": subset_acc,
                      "baseline": baseline, "max_abs": max_abs,
                      "gain": gain, "agreement": agree}))


if __name__ == "__main__":
    main()
