#!/usr/bin/env python3
"""Train + quantise a 28x28 classifier for Baru's private-CNN pipeline.

Replaces the lost scratchpad/train_modeA.py (kept in tools/, not scratchpad,
because that is how the original was lost), and generalised: it handles any
grayscale 28x28 MedMNIST v2 dataset *and* plain MNIST, so every embedded
weight file in the repo can be regenerated from this one script.  Emits
examples/data/<name>_cnn.rs in exactly the integer fixed-point scheme
`CnnModel::forward` (src/bench/cnn.rs) evaluates.

Architecture.  The Rust side fixes K=3 and POOL=2 and nothing else: `CnnModel`
carries `extra_conv: Vec<ConvLayer>` and `hidden: Vec<(W, b)>`, so the conv
stack and the dense stack are both free.  `--conv` and `--fc` set them:

    1x28x28 -> [conv(c,3x3) -> ReLU -> pool2]*   for c in --conv
            -> [dense(->h)  -> ReLU]*            for h in --fc
            -> dense(->NCLASSES) -> argmax

`--conv 8,16 --fc 128` (the default) is the shallow net the first round of
`bench-medmnist` numbers were measured on, kept as the default so every weight
file already in the repo regenerates unchanged.  Deeper/wider stacks are what
`--conv 32,64 --fc 256,128` and friends are for.

Three limits bind, and they bind for different reasons.

*Spatially*, 3x3 valid convolution and floor-pooling take 28 -> 13 -> 5 -> 1,
so three convs is the most a 28x28 input admits and the third leaves no spatial
extent at all.  `conv_shapes` refuses the rest.

*By packing*, the Cheetah-style conv packing encodes a whole layer into one
polynomial, so the product degree must stay under the ring dimension (4096).
That binds on input channels times spatial extent and bites hardest at the
SECOND conv, where the channels have grown but the image has been pooled only
once: at 13x13 it admits at most 12 input channels, so `--conv 16,32` is
already too wide even though it trains perfectly well.  `check_packing`
refuses it up front, since otherwise the failure surfaces twenty minutes later
from inside the FHE code.

*Numerically*, every activation has to stay inside the depth-1 FHE plaintext
modulus (2^30 at n=4096, q=2^100), and a wider layer sums more terms into each
accumulator -- so capacity is capped by the noise budget, not by memory.
`pick_gain` rejects any quantisation that crosses that line.

The last two mean a net can train fine in float and still be refused here.
That refusal is a result, not a bug: report it rather than lowering `shift`
until it passes.

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

RGB datasets.  Four of the twelve MedMNIST v2 2D sets ship (N,28,28,3):
Path, Derma, Blood, Retina.  `--rgb` keeps the three channels; the default folds
them to luminance (ITU-R BT.601: 0.299R+0.587G+0.114B, rounded to uint8).

The packing does NOT require the fold -- see `fold_channels` for the claim that
was wrong here for a long time.  An RGB 28x28 convolution fits one ciphertext at
one multiplication per output channel.  The fold stays the default only because
every other weight file in the repo was measured with it.

Folding is a real change to the task -- colour carries diagnostic signal in
dermatoscopy and histology, measured at AUC 0.8519 against 0.7667 on DermaMNIST
-- so the generated file records which was used, and a folded accuracy must be
read against a grayscale baseline rather than the published RGB ResNet-18.

Usage: python3 tools/train_medmnist.py <npz|mnist-dir> <rust-out>
         [--epochs N] [--n-test N] [--test-order first|random] [--linear]
         [--conv 8,16] [--fc 128] [--gains none,64,128] [--device auto|cpu|mps]
"""
import argparse
import gzip
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K, POOL, SHIFT = 3, 2, 8
# The default stack, kept so the weight files already in the repo regenerate
# byte-identically.  `--conv`/`--fc` override both.
DEFAULT_CONV, DEFAULT_FC = [8, 16], [128]
# The gain ladder swept on validation; `None` is "no rescale at all".
DEFAULT_GAINS = (None, 64.0, 128.0, 256.0, 512.0)
# Bits a candidate gain must leave spare below the 2^30 plaintext modulus, measured
# on validation. See `pick_gain` for the DermaMNIST run that calibrated it: the
# validation maximum underestimated the test maximum by three bits.
HEADROOM_BITS = 4
# BT.601 luminance, the fold applied to the four RGB MedMNIST sets.
LUMA = np.array([0.299, 0.587, 0.114])
# The BFV ring dimension the hybrid pipeline runs at (n=4096, q=2^100).
RING_N = 4096


def packing_degree(in_c, h, w):
    """Highest coefficient the packed convolution product can touch.

    A line-for-line mirror of `ConvShape::required_degree`
    (src/fhe/conv_pack.rs).  The Cheetah-style packing puts a whole conv layer
    into one polynomial, so this has to stay under `RING_N` or negacyclic
    wraparound folds garbage onto the coefficients the output is read from."""
    hw = h * w
    max_x = in_c * hw - 1
    max_w = (in_c - 1) * hw + (K - 1) * w + (K - 1)
    return max_x + max_w


def check_packing(shapes):
    """Report how each conv will be packed, and refuse only the impossible.

    A third limit, independent of the other two: it binds on input channels times
    spatial extent, so it bites hardest at the second conv, where the channel count
    has grown but the image has only been pooled once.  Checked up front rather than
    after training, because otherwise the refusal arrives twenty minutes late and
    from the middle of the FHE code.

    Since 2026-08-04 the Rust side can **split channel groups** (`conv_pack::
    channel_groups`): convolution is linear in the input channels, so a layer too
    wide for one ciphertext is evaluated as equal groups whose products are added,
    which is free homomorphically.  So a wide layer is no longer refused -- it costs
    one ciphertext multiplication per group per output channel, and this prints that
    cost so the choice is informed.  Only a layer whose *single* channel overruns the
    ring is still impossible, and there the remedy is a bigger ring, not splitting.
    """
    plan = []
    for i, (in_c, h, w) in enumerate(shapes):
        deg = packing_degree(in_c, h, w)
        group = max((c for c in range(1, in_c + 1)
                     if packing_degree(c, h, w) < RING_N), default=0)
        if group == 0:
            sys.exit(
                f"conv {i} cannot be run under FHE at any grouping: a single "
                f"{h}x{w} channel packs to degree {packing_degree(1, h, w)} against "
                f"ring dimension {RING_N}. Channel splitting cannot help -- only a "
                f"larger ring can. (This is the packing limit, not the noise budget "
                f"-- the model would train and score fine in plaintext.)"
            )
        groups = -(-in_c // group)
        plan.append((i, in_c, h, w, deg, group, groups))

    if any(g > 1 for *_, g in plan):
        print("  conv packing plan (channel splitting is free: homomorphic adds)")
        for i, in_c, h, w, deg, group, groups in plan:
            note = "one ciphertext" if groups == 1 else (
                f"{groups} groups of {group} -> {groups}x the multiplications")
            print(f"    conv {i}: {in_c}x{h}x{w}, degree {deg} vs n={RING_N} -- {note}")


def conv_shapes(d, convs, in_c=1):
    """Spatial shape entering each conv, and the flattened feature width.

    Returns `([(in_c, in_h, in_w), ...], flat)`.  Mirrors `CnnModel::forward`
    exactly: 3x3 valid convolution then floor max-pool by 2, per layer.
    `in_c` is 3 for an `--rgb` run and 1 otherwise."""
    shapes, c, h, w = [], in_c, d, d
    for oc in convs:
        if h < K or w < K:
            sys.exit(
                f"conv stack {convs} does not fit a {d}x{d} input: layer with "
                f"{oc} channels sees {h}x{w}, smaller than the {K}x{K} kernel"
            )
        shapes.append((c, h, w))
        h, w = (h - K + 1) // POOL, (w - K + 1) // POOL
        c = oc
    return shapes, c * h * w

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


def fold_channels(d, keep_rgb):
    """Normalise every split to `(N,h,w)` or, with `keep_rgb`, `(N,h,w,3)`.

    Four of the twelve MedMNIST v2 2D sets ship RGB (Path, Derma, Blood, Retina).

    **This used to say the FHE conv packing "assumes a single input channel", and
    that was wrong.**  `fhe::conv_pack` has always been multi-channel -- channels
    summing on a shared coefficient is the whole point of its reversed offset -- and
    what was single-channel was the Rust plumbing, which hardcoded `in_c = 1`.  Since
    that was fixed an RGB 28x28 convolution runs directly, needing degree 3977 against
    a ring dimension of 4096, at one ciphertext multiplication per output channel: the
    three channels do **not** cost three times the work.

    So folding to luminance is now a choice rather than a requirement.  It is still the
    default, because every embedded weight file and every accuracy figure in the repo
    was measured that way and flipping the default would silently redefine them.
    `--rgb` keeps the colour.  `folded` propagates into the generated file: an accuracy
    measured after the fold is not comparable to a published RGB number, and the file
    has to say which it is."""
    folded = False
    for k in ("train_images", "val_images", "test_images"):
        a = d[k]
        if a.ndim == 4 and a.shape[-1] == 3:
            if keep_rgb:
                continue  # left as (N,h,w,3); `flatten` lays it out channel-major
            d[k] = np.rint(a.astype(np.float64) @ LUMA).clip(0, 255).astype(np.uint8)
            folded = True
        elif a.ndim == 4 and a.shape[-1] == 1:
            d[k] = a[..., 0]
        elif a.ndim != 3:
            sys.exit(f"{k}: expected (N,h,w), (N,h,w,1) or (N,h,w,3), got {a.shape}")
    if keep_rgb and not any(d[k].ndim == 4 for k in ("train_images",)):
        sys.exit("--rgb given but the dataset is not RGB: images are already single-channel")
    return d, folded


def parse_widths(s, flag, allow_empty):
    """Parse a `--conv`/`--fc` comma-separated width list."""
    s = s.strip()
    if not s:
        if allow_empty:
            return []
        sys.exit(f"{flag}: needs at least one layer")
    try:
        vals = [int(tok) for tok in s.split(",")]
    except ValueError:
        sys.exit(f"{flag}: expected comma-separated integers, got {s!r}")
    if any(v <= 0 for v in vals):
        sys.exit(f"{flag}: widths must be positive, got {vals}")
    return vals


def parse_gains(s, default):
    """Parse the `--gains` ladder.  `none` is the no-rescale candidate."""
    if not s.strip():
        return default
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.lower() in ("none", "raw"):
            out.append(None)
            continue
        try:
            g = float(tok)
        except ValueError:
            sys.exit(f"--gains: expected numbers or 'none', got {tok!r}")
        if g <= 0:
            sys.exit(f"--gains: gains must be positive, got {g}")
        out.append(g)
    return tuple(out)


def pick_device(choice):
    """Resolve `--device`.  Only the float training runs here; the integer
    reference is numpy on the CPU either way, and it is the integer model that
    is deployed, so the device can never change the reported result."""
    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "mps" and not torch.backends.mps.is_available():
        sys.exit("--device mps: no MPS backend available on this machine")
    return torch.device(choice)


class Net(nn.Module):
    """Exactly the layer stack CnnModel::forward implements (bias everywhere,
    valid padding, floor pooling), at whatever depth/width `convs` and `fcs`
    ask for."""

    def __init__(self, nclasses, convs=None, fcs=None, d=28, in_c=1):
        super().__init__()
        convs = list(convs or DEFAULT_CONV)
        fcs = list(fcs or DEFAULT_FC)
        shapes, flat = conv_shapes(d, convs, in_c)
        self.convs = nn.ModuleList(
            nn.Conv2d(in_c, oc, K) for (in_c, _, _), oc in zip(shapes, convs)
        )
        widths = [flat] + fcs
        self.fcs = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:])
        )
        self.out = nn.Linear(widths[-1], nclasses)

    def forward(self, x):
        for c in self.convs:
            x = F.max_pool2d(F.relu(c(x)), POOL, POOL)
        x = x.flatten(1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return self.out(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─── integer reference: a line-for-line mirror of CnnModel::forward ──────────


def int_forward(img, q):
    """img: (d,d) int64 raw pixels. Returns (class, max_abs)."""
    logits, max_abs = int_logits(img, q, want_max=True)
    return int(np.argmax(logits)), max_abs


def int_logits(img, q, want_max=False):
    """img: (d,d) int64 raw pixels. Returns the output-layer logits.

    A line-for-line mirror of `CnnModel::forward_logits` over `q`'s conv stack
    (`q["conv"]`, a list of `(w, b)`) and dense stack (`q["fc"]`, the hidden
    layers, then `q["out"]`).  With `want_max`, also returns the largest
    intermediate magnitude, which is what sizes the plaintext modulus."""
    x = img.astype(np.int64)
    max_abs = int(np.abs(x).max())

    def conv(inp, w, b):
        nonlocal max_abs
        in_c, in_h, in_w = inp.shape
        out_c = w.shape[0]
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

    # `(d,d)` for a grayscale image, `(C,d,d)` for an RGB one. `conv` below is
    # already multi-channel -- it has to be, since every conv after the first takes
    # many channels -- so RGB needs no other change here.
    a = x[None, :, :] if x.ndim == 2 else x
    for w, b in q["conv"]:
        a = pool(conv(a, w, b))
    h = a.reshape(-1)
    for w, b in q["fc"]:
        h = w @ h + b
        max_abs = max(max_abs, int(np.abs(h).max()))
        h = np.where(h >= 0, h >> SHIFT, 0)
    w, b = q["out"]
    logits = w @ h + b
    max_abs = max(max_abs, int(np.abs(logits).max()))
    return (logits, max_abs) if want_max else logits


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
    layers = list(net.convs) + list(net.fcs) + [net.out]
    n_conv = len(net.convs)
    W = [p.weight.detach().cpu().numpy().astype(np.float64) for p in layers]
    B = [p.bias.detach().cpu().numpy().astype(np.float64) for p in layers]

    if gain_target is not None:
        for l in range(len(W)):
            peak = np.abs(W[l]).max() * (1 << SHIFT)
            g = gain_target / peak if peak > 0 else 1.0
            W[l] *= g
            B[l] *= g
            if l + 1 < len(W):  # last layer: argmax absorbs the gain
                W[l + 1] /= g

    s = 1 << SHIFT
    # Conv bias is `round(B*255)` because `forward` adds `cbias << shift`;
    # dense bias is `round(B*255*2^shift)` because `forward` adds it raw.
    # Same quantity, applied at the two places the Rust shifts differently.
    conv = [
        (np.rint(W[l].reshape(W[l].shape[0], -1) * s).astype(np.int64),
         np.rint(B[l] * 255.0).astype(np.int64))
        for l in range(n_conv)
    ]
    dense = [
        (np.rint(W[l] * s).astype(np.int64),
         np.rint(B[l] * 255.0 * s).astype(np.int64))
        for l in range(n_conv, len(W))
    ]
    return {"conv": conv, "fc": dense[:-1], "out": dense[-1]}


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


def score_logits(q, imgs_u8, progress_every=0):
    """Raw output-layer logits per image -- what ranking metrics need.

    Kept separate from `score_int` because AUC is a property of the *ordering*
    the logits induce, not of the argmax, and the argmax is all the deployed
    encrypted pipeline ever reveals."""
    out = []
    for i, im in enumerate(imgs_u8):
        out.append(int_logits(im, q))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    logit-scored {i + 1}/{len(imgs_u8)}", flush=True)
    return np.array(out, dtype=np.int64)


def auc(scores, positive):
    """AUC via Mann-Whitney U over average ranks.

    Rank-based with tie-averaging because these logits are integers and
    genuinely tie; counting a tie as a win would inflate the number.  Returns
    None when one class is absent, where AUC is undefined."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    s = np.asarray(scores)[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), dtype=np.float64)
    r[order] = ranks
    u = r[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def wilson(k, n, z=1.96):
    """95% Wilson score interval -- correct near p=1 where the normal
    approximation escapes [0,1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def full_metrics(logits, labels, nclasses):
    """The metrics the benchmark reports, over the FULL test set.

    Mirrors `src/bench/metrics.rs` so the two agree; the Rust side can only see
    the embedded subset, so these are the numbers that belong in a paper."""
    preds = logits.argmax(axis=1)
    conf = np.zeros((nclasses, nclasses), dtype=np.int64)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    support = conf.sum(axis=1)
    recall = np.divide(np.diag(conf), support,
                       out=np.full(nclasses, np.nan), where=support > 0)
    present = recall[~np.isnan(recall)]
    correct = int(np.diag(conf).sum())
    if nclasses == 2:
        a = auc(logits[:, 1] - logits[:, 0], labels == 1)
    else:
        per = [auc(logits[:, c], labels == c) for c in range(nclasses)]
        per = [x for x in per if x is not None]
        a = float(np.mean(per)) if per else None
    return {
        "n": int(len(labels)),
        "accuracy": correct / len(labels),
        "accuracy_ci": wilson(correct, len(labels)),
        "balanced_accuracy": float(present.mean()) if len(present) else float("nan"),
        "auc": a,
        "per_class_recall": recall.tolist(),
        "per_class_n": support.tolist(),
        "sensitivity": float(recall[1]) if nclasses == 2 else None,
        "specificity": float(recall[0]) if nclasses == 2 else None,
        "confusion": conf.tolist(),
    }


def report_metrics(m, names):
    """Print the full-test metrics, calling out an imbalance rather than
    leaving it for the reader to derive from the confusion matrix."""
    lo, hi = m["accuracy_ci"]
    print(f"  accuracy          : {m['accuracy'] * 100:.2f}%  "
          f"95% CI [{lo * 100:.1f}, {hi * 100:.1f}]  (n={m['n']})")
    print(f"  balanced accuracy : {m['balanced_accuracy'] * 100:.2f}%")
    print(f"  AUC               : "
          + ("undefined" if m["auc"] is None else f"{m['auc']:.4f}")
          + ("" if len(names) == 2 else "  (macro one-vs-rest)"))
    if m["sensitivity"] is not None:
        sens, spec = m["sensitivity"], m["specificity"]
        print(f"  sensitivity/spec  : {sens * 100:.2f}% / {spec * 100:.2f}%")
        if abs(sens - spec) > 0.15:
            side = "misses disease" if sens < spec else "over-calls disease"
            print(f"    ^ SKEWED by {abs(sens - spec) * 100:.0f} points -- the model "
                  f"{side}; quote balanced accuracy, not accuracy")
    else:
        order = np.argsort([r if not math.isnan(r) else 2 for r in m["per_class_recall"]])
        worst = ", ".join(
            f"{names[c]} {m['per_class_recall'][c] * 100:.0f}% (n={m['per_class_n'][c]})"
            for c in order[:3]
        )
        print(f"  weakest classes   : {worst}")


def pick_gain(net, xva_u8, yva, targets=DEFAULT_GAINS):
    """Choose the quantisation gain on the VALIDATION split (never on test),
    rejecting any candidate whose activations come within `HEADROOM_BITS` of the
    2^30 the depth-1 plaintext modulus allows.

    Wider nets sum more terms per accumulator and so overflow at gains the
    default sweep never goes below; `--gains` extends the ladder downwards for
    them.

    # Why the headroom, and it is not conservatism for its own sake

    The validation split's largest activation is an *estimate* of the largest any
    input will produce, and it underestimates.  Measured on DermaMNIST in RGB: gain
    512 showed 2^28 on validation, was accepted, and then needed **2^31** on the test
    split -- a three-bit gap that made the chosen model unusable and only surfaced in
    the final check.  A model that overflows for some inputs and not others is worse
    than one that is slightly less precise, because the failure is data-dependent.

    So a candidate must leave `HEADROOM_BITS` spare on validation.  Four, from that
    three-bit observation plus one.  This is a *safety* criterion and deliberately not
    informed by the test split, which would be selection on test data; the final check
    against the test split remains as the backstop it already was."""
    best = None
    for t in targets:
        q = quantise(net, t)
        acc, max_abs, _ = score_int(q, xva_u8, yva)
        needed = int(max_abs).bit_length() + 2
        if needed > 30:
            flag = "  <- OVERFLOWS, rejected"
        elif needed > 30 - HEADROOM_BITS:
            flag = f"  <- within {HEADROOM_BITS}b of the ceiling, rejected"
        else:
            flag = ""
        print(f"  gain {str(t):>6}: val-int {acc * 100:6.2f}%  "
              f"max|act| = 2^{needed - 2}{flag}", flush=True)
        if needed <= 30 - HEADROOM_BITS and (best is None or acc > best[1]):
            best = (t, acc, q)
    if best is None:
        sys.exit(
            f"every candidate gain comes within {HEADROOM_BITS} bits of the FHE "
            f"plaintext modulus; extend the ladder downwards with --gains"
        )
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
    """Write the generated weight file for a conv stack of any depth/width.

    The first conv is emitted as `KERNELS`/`CBIAS` -- `TrainedCnn` keeps it
    separate because it is the single-channel one the Cheetah packing sees --
    and convs 1.. as `CONV1_W`/`CONV1_B`/..., hidden dense layers as
    `FC0_W`/`FC0_B`/..., the output layer as `DENSE_W`/`DENSE_B`.  The
    `trained()` constructor at the bottom assembles them into the shape
    `TrainedCnn` takes.  Each extra conv's input shape is baked in as a
    constant rather than recomputed in Rust, so the two sides cannot silently
    disagree about where pooling floors."""
    convs, fcs, (out_w, out_b) = q["conv"], q["fc"], q["out"]
    shapes = meta["conv_shapes"]  # [(in_c, in_h, in_w)] per conv layer
    # Wrapped by hand rather than with textwrap: the stages are the natural
    # break points, and a deep stack on one line is unreadable in the header.
    stages = (
        [f"1x{meta['d']}x{meta['d']}"]
        + [f"conv({w.shape[0]},{K}x{K}) -> ReLU>>{SHIFT} -> pool" for w, _ in convs]
        + [f"dense(->{w.shape[0]}) -> ReLU>>{SHIFT}" for w, _ in fcs]
        + [f"dense(->{meta['nclasses']}) -> argmax"]
    )
    lines, cur = [], "// Architecture: "
    for i, st in enumerate(stages):
        piece = st if i == 0 else f" -> {st}"
        if len(cur) + len(piece) > 76 and cur.strip() != "// Architecture:":
            lines.append(cur)
            cur = f"//   -> {st}"
        else:
            cur += piece
    lines.append(cur)
    arch = "\n".join(lines)
    fm = meta["full_metrics"]
    auc_lit = "None" if fm["auc"] is None else f"Some({fm['auc']:.6f})"
    auc_txt = "undefined" if fm["auc"] is None else f"{fm['auc']:.4f}"
    o = []
    o.append(f"""// GENERATED - do not edit. Trained + quantised {len(convs)}-conv CNN on
// {meta['dataset']} ({meta['provenance']}), plus a real test subset.
// See tools/train_medmnist.py.
//
// Task: {meta['task']}
{arch}
// Parameters: {meta['n_params']}
// Quantisation gain: {meta['gain']}; the largest activation over the full test
// set needs 2^{meta['act_bits']} of the 2^30 the depth-1 FHE plaintext modulus allows
// (max |activation| = {meta['max_abs']})
// Full {meta['n_full']}-image integer test accuracy: {meta['full_acc'] * 100:.2f}%
// 95% CI [{fm['accuracy_ci'][0] * 100:.2f}, {fm['accuracy_ci'][1] * 100:.2f}]
// (float parent {meta['float_acc'] * 100:.2f}%, majority-class baseline {meta['baseline'] * 100:.2f}%)
// Balanced accuracy {fm['balanced_accuracy'] * 100:.2f}%, AUC {auc_txt} — read these
// rather than accuracy when the test set is imbalanced.{meta['gray_note']}

pub const DATASET: &str = "{meta['dataset']}";
pub const TASK: &str = "{meta['task']}";
pub const FULL_TEST_ACCURACY: f64 = {meta['full_acc']:.6f};
/// Accuracy of the float net these weights were quantised from, on the same
/// full test set. `None` in weight files generated before the trainer
/// recorded it; regenerate to populate.
pub const FLOAT_TEST_ACCURACY: Option<f64> = Some({meta['float_acc']:.6f});
/// Metrics over the FULL test set, which only the trainer can see — the
/// benchmark itself is limited to the embedded subset. Accuracy alone cannot
/// distinguish a balanced classifier from one riding the majority class, so
/// balanced accuracy and AUC are recorded alongside it. `None` in weight files
/// generated before the trainer measured them.
pub const FULL_BALANCED_ACCURACY: Option<f64> = Some({meta['full_metrics']['balanced_accuracy']:.6f});
pub const FULL_AUC: Option<f64> = {auc_lit};
pub const MAJORITY_BASELINE: f64 = {meta['baseline']:.6f};
pub const FULL_TEST_N: usize = {meta['n_full']};
pub const D: usize = {meta['d']};
pub const K: usize = {K};
pub const NCLASSES: usize = {meta['nclasses']};
pub const SHIFT: u32 = {SHIFT};
/// Output channels of the first conv (the one the Cheetah packing sees as
/// single-channel input); the rest are read off each layer's weight array.
pub const OC: usize = {convs[0][0].shape[0]};
pub const CLASS_NAMES: [&str; {meta['nclasses']}] = [{
    ", ".join('"' + c + '"' for c in meta['class_names'])
}];
""")
    # Conv stack. Layer 0 is the single-channel first conv `TrainedCnn` keeps
    # separate (KERNELS/CBIAS); layers 1.. are the multi-channel extras.
    o.append(rust_array("KERNELS", convs[0][0], "i64"))
    o.append(rust_array("CBIAS", convs[0][1], "i64", static_=False))
    for i, (w, b) in enumerate(convs[1:], start=1):
        in_c, in_h, in_w = shapes[i]
        o.append(f"""
pub const CONV{i}_IN_C: usize = {in_c};
pub const CONV{i}_IN_H: usize = {in_h};
pub const CONV{i}_IN_W: usize = {in_w};
""")
        o.append(rust_array(f"CONV{i}_W", w, "i64"))
        o.append(rust_array(f"CONV{i}_B", b, "i64", static_=False))
    for i, (w, b) in enumerate(fcs):
        o.append(rust_array(f"FC{i}_W", w, "i64"))
        o.append(rust_array(f"FC{i}_B", b, "i64"))
    o.append(rust_array("DENSE_W", out_w, "i64"))
    o.append(rust_array("DENSE_B", out_b, "i64"))
    o.append(f"""
pub const N_TEST: usize = {len(labels)};
""")
    o.append(rust_array("TEST_LABELS", labels, "u8"))
    o.append(rust_array("TEST_IMAGES", [im.reshape(-1) for im in images], "i64"))

    extra = ",\n".join(
        f"""        (
            CONV{i}_W.iter().map(|r| r.to_vec()).collect(),
            CONV{i}_B.to_vec(),
            CONV{i}_IN_C,
            CONV{i}_IN_H,
            CONV{i}_IN_W,
        )"""
        for i in range(1, len(convs))
    )
    hidden = ",\n".join(
        f"        (FC{i}_W.iter().map(|r| r.to_vec()).collect(), FC{i}_B.to_vec())"
        for i in range(len(fcs))
    )
    o.append(f"""
/// One extra conv layer: `(kernels[out_c][in_c*K*K], bias[out_c], in_c, in_h,
/// in_w)`. Named so the signature below stays readable.
pub type ExtraConvLayer = (Vec<Vec<i64>>, Vec<i64>, usize, usize, usize);

/// The conv layers after the first, in order. Each is
/// `(kernels[out_c][in_c*K*K], bias[out_c], in_c, in_h, in_w)`; every one is
/// another FHE conv + MPC nonlinearity + scheme switch, and no bootstrap.
pub fn extra_conv_layers() -> Vec<ExtraConvLayer> {{
    vec![
{extra}
    ]
}}

/// Hidden dense layers in order, each `(weights[out][in], bias[out])`.
pub fn hidden_layers() -> Vec<(Vec<Vec<i64>>, Vec<i64>)> {{
    vec![
{hidden}
    ]
}}

/// Everything `bench-medmnist` needs, in the crate's normalised form.
pub fn trained() -> crate::bench::medmnist::TrainedCnn {{
    crate::bench::medmnist::TrainedCnn {{
        dataset: DATASET,
        task: TASK,
        class_names: CLASS_NAMES.to_vec(),
        d: D,
        kernels: KERNELS.iter().map(|k| k.to_vec()).collect(),
        cbias: CBIAS.to_vec(),
        extra_conv: extra_conv_layers(),
        hidden: hidden_layers(),
        dense_w: DENSE_W.iter().map(|r| r.to_vec()).collect(),
        dense_b: DENSE_B.to_vec(),
        shift: SHIFT,
        images: TEST_IMAGES.iter().map(|im| im.to_vec()).collect(),
        labels: TEST_LABELS.to_vec(),
        full_test_accuracy: FULL_TEST_ACCURACY,
        float_test_accuracy: FLOAT_TEST_ACCURACY,
        full_balanced_accuracy: FULL_BALANCED_ACCURACY,
        full_auc: FULL_AUC,
        majority_baseline: MAJORITY_BASELINE,
        full_test_n: FULL_TEST_N,
    }}
}}
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
    ap.add_argument("--conv", default=",".join(str(c) for c in DEFAULT_CONV),
                    help="comma-separated conv output channels, one per layer "
                         f"(default {','.join(str(c) for c in DEFAULT_CONV)}). "
                         "Each layer is 3x3 valid + ReLU + pool2, so a 28x28 "
                         "input admits at most three.")
    ap.add_argument("--fc", default=",".join(str(h) for h in DEFAULT_FC),
                    help="comma-separated hidden dense widths, one per layer "
                         f"(default {','.join(str(h) for h in DEFAULT_FC)}); "
                         "empty string for none")
    ap.add_argument("--gains", default="",
                    help="comma-separated quantisation-gain ladder to sweep on "
                         "validation (default 64,128,256,512 plus the "
                         "no-rescale case). Wider nets overflow the depth-1 "
                         "plaintext modulus at those, and want smaller ones.")
    ap.add_argument(
        "--rgb",
        action="store_true",
        help="keep the three colour channels instead of folding to luminance. "
        "Only meaningful for the RGB MedMNIST sets (Path, Derma, Blood, Retina). "
        "The FHE packing handles this at one ciphertext multiply per output "
        "channel -- three channels do NOT cost three times the work -- but every "
        "figure in the repo was measured on the grayscale fold, so the fold "
        "stays the default and an --rgb result is a different experiment.",
    )
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"),
                    help="training device; scoring the integer net is numpy on "
                         "CPU either way")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_dataset(args.npz)
    d, folded = fold_channels(d, args.rgb)
    if args.rgb:
        # MedMNIST ships RGB as (N,h,w,3); torch wants NCHW and the Rust side reads
        # an image as `[in_c][h][w]` row-major, so one transpose here makes both
        # layouts right and every later reshape channel-major by construction.
        for k in ("train_images", "val_images", "test_images"):
            d[k] = np.ascontiguousarray(d[k].transpose(0, 3, 1, 2))
    xtr = d["train_images"].astype(np.float32) / 255.0
    ytr = d["train_labels"].ravel().astype(np.int64)
    xva = d["val_images"].astype(np.float32) / 255.0
    yva = d["val_labels"].ravel().astype(np.int64)
    xte_u8 = d["test_images"].astype(np.int64)
    yte = d["test_labels"].ravel().astype(np.int64)
    nclasses = int(ytr.max()) + 1
    in_c = xtr.shape[1] if args.rgb else 1
    side = xtr.shape[-1]
    if xtr.shape[-2] != xtr.shape[-1]:
        sys.exit(f"expected square images, got {xtr.shape[-2]}x{xtr.shape[-1]}")
    print(f"{args.npz}: train {xtr.shape} val {xva.shape} test {xte_u8.shape}, "
          f"{nclasses} classes", flush=True)
    if folded:
        print("RGB input folded to BT.601 luminance: accuracy below is a "
              "GRAYSCALE result, not comparable to a published RGB number",
              flush=True)

    if args.linear:
        if side != 28:
            sys.exit(f"--linear is fixed to 784 raw pixels; this input is {side}x{side}")
        return train_linear(args, d, nclasses)

    convs = parse_widths(args.conv, "--conv", allow_empty=False)
    fcs = parse_widths(args.fc, "--fc", allow_empty=True)
    gains = parse_gains(args.gains, DEFAULT_GAINS)
    device = pick_device(args.device)
    shapes, flat = conv_shapes(side, convs, in_c)
    check_packing(shapes)
    net = Net(nclasses, convs, fcs, side, in_c).to(device)
    print(f"model: conv {convs} -> flat {flat} -> fc {fcs} -> {nclasses}  "
          f"({net.n_params()} parameters, training on {device.type})", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    chan = (lambda a: torch.from_numpy(a)) if args.rgb else (
        lambda a: torch.from_numpy(a).unsqueeze(1))
    Xtr = chan(xtr).to(device)
    Ytr = torch.from_numpy(ytr).to(device)
    Xva = chan(xva).to(device)
    Yva = torch.from_numpy(yva).to(device)

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
        Xte = chan(xte_u8.astype(np.float32) / 255.0).to(device)
        float_acc = (
            (net(Xte).argmax(1).cpu() == torch.from_numpy(yte)).float().mean().item()
        )
    print(f"float test accuracy: {float_acc * 100:.2f}% (best val {best * 100:.2f}%)",
          flush=True)

    # Pick the quantisation gain on validation, then score on the FULL test set.
    print("quantisation gain sweep (validation split):", flush=True)
    q, gain = pick_gain(net, d["val_images"].astype(np.int64), yva, gains)
    int_acc, max_abs, preds = score_int(q, xte_u8, yte, progress_every=2000)
    needed = int(max_abs).bit_length() + 2
    baseline = float(np.bincount(yte, minlength=nclasses).max() / len(yte))
    with torch.no_grad():
        float_preds = net(Xte).argmax(1).cpu().numpy()
    agree = float((preds == float_preds).mean())
    print(f"integer test accuracy: {int_acc * 100:.2f}%  "
          f"(float {float_acc * 100:.2f}%, majority {baseline * 100:.2f}%)")
    print(f"integer/float prediction agreement: {agree * 100:.2f}%")

    # Full-test metrics beyond accuracy. These are the numbers that belong in a
    # paper: the Rust benchmark can only ever see the embedded subset, and
    # accuracy alone cannot tell a balanced classifier from one that rides the
    # majority class -- which on an imbalanced medical set is the whole story.
    names_for_metrics = (args.classes.split(",") if args.classes
                         else [str(i) for i in range(nclasses)])
    print("full-test metrics (integer model):", flush=True)
    fm = full_metrics(score_logits(q, xte_u8, progress_every=5000), yte, nclasses)
    report_metrics(fm, names_for_metrics)
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
        "d": side,
        "conv_shapes": shapes,
        "n_params": net.n_params(),
        "gain": gain,
        "act_bits": needed,
        "max_abs": max_abs,
        "full_acc": int_acc,
        "float_acc": float_acc,
        "full_metrics": fm,
        "baseline": baseline,
        "n_full": len(xte_u8),
        "class_names": names,
        # An accuracy measured after the luminance fold is a grayscale result;
        # the file has to carry that or the number invites a comparison to the
        # published RGB model that it does not support.
        "gray_note": (
            "\n// NOTE: this dataset ships RGB; the input was folded to BT.601 "
            "luminance\n// before training, so the accuracy above is a GRAYSCALE "
            "result and must not\n// be read against the published RGB ResNet-18."
        ) if folded else "",
    }
    emit(args.rust_out, q, [xte_u8[i] for i in sel], yte[sel].astype(np.uint8), meta)
    subset_acc = float((preds[sel] == yte[sel]).mean())
    print(f"wrote {args.rust_out}: {n_test} embedded test images, "
          f"subset integer accuracy {subset_acc * 100:.2f}%")
    print(json.dumps({"dataset": meta["dataset"], "int_acc": int_acc,
                      "float_acc": float_acc, "subset_acc": subset_acc,
                      "baseline": baseline, "max_abs": max_abs,
                      "gain": gain, "agreement": agree, "conv": convs,
                      "fc": fcs, "n_params": net.n_params(),
                      "grayscale_fold": folded}))


if __name__ == "__main__":
    main()
