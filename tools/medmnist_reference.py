#!/usr/bin/env python3
"""Reference implementations of the MedMNIST quantised CNN, for timing only.

`examples/medmnist.baru` is the Baru implementation; this is the same network in
Python, so "what does privacy cost?" can be read against "what does the
computation cost in a language nobody would call slow on purpose".

Three variants, all integer, all bit-identical to `CnnModel::forward_logits`:

* ``pure``  — plain Python, no third-party libraries. The closest analogue to the
  Baru interpreter: a tree-walking evaluator against a bytecode one, both doing
  the arithmetic one scalar at a time.
* ``numpy`` — the same network vectorised. What a practitioner would actually
  write, and the honest reference for "how fast is this computation".
* ``torch``  — float32 PyTorch, which is what a practitioner would *really* run.
  It is NOT bit-identical (it is float, and the quantised net is not), so it is
  reported as a separate row and its predictions are compared, not asserted.

Usage:
    python3 tools/medmnist_reference.py model.json [--images N] [--impl pure|numpy|torch|all]

The JSON comes from `cargo run --release --example export_medmnist`, so the
weights and the images are the identical i64s every other implementation uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


# ── pure Python ──────────────────────────────────────────────────────────────
# Deliberately written the way the Baru program is written: a window is
# flattened in (ic, ky, kx) order and multiplied against the kernel row, the
# ReLU is fused with the rescale, and the pool is four comparisons. No numpy, so
# this measures the interpreter, which is the point of the comparison.


def relu_rescale(v: int, s: int) -> int:
    return (v >> s) if v >= 0 else 0


def conv2d(fmap, in_c, ih, iw, kernels, bias, k, s):
    oh, ow = ih - k + 1, iw - k + 1
    hw = ih * iw
    out = [0] * (len(kernels) * oh * ow)
    for c, kern in enumerate(kernels):
        b = bias[c] << s
        base = c * oh * ow
        for oy in range(oh):
            for ox in range(ow):
                acc = 0
                i = 0
                for ic in range(in_c):
                    off = ic * hw
                    for ky in range(k):
                        row = off + (oy + ky) * iw + ox
                        for kx in range(k):
                            acc += kern[i] * fmap[row + kx]
                            i += 1
                out[base + oy * ow + ox] = relu_rescale(acc + b, s)
    return out, len(kernels), oh, ow


def maxpool(fmap, c, ih, iw):
    # Floor geometry, matching gadgets::maxpool2d_ref: (in - 2) // 2 + 1.
    oh, ow = (ih - 2) // 2 + 1, (iw - 2) // 2 + 1
    hw = ih * iw
    out = [0] * (c * oh * ow)
    for ch in range(c):
        for oy in range(oh):
            for ox in range(ow):
                o = ch * hw + (oy * 2) * iw + ox * 2
                out[ch * oh * ow + oy * ow + ox] = max(
                    fmap[o], fmap[o + 1], fmap[o + iw], fmap[o + iw + 1]
                )
    return out, c, oh, ow


def forward_pure(m, img):
    s = m["shift"]
    c1 = m["conv1"]
    f, c, h, w = conv2d(
        img, m["in_c"], m["d"], m["d"], c1["kernels"], c1["bias"], c1["k"], s
    )
    f, c, h, w = maxpool(f, c, h, w)
    for cl in m["extra_conv"]:
        f, c, h, w = conv2d(
            f, cl["in_c"], cl["in_h"], cl["in_w"], cl["kernels"], cl["bias"], cl["k"], s
        )
        f, c, h, w = maxpool(f, c, h, w)
    x = f
    for lay in m["hidden"]:
        x = [
            relu_rescale(sum(wi * xi for wi, xi in zip(row, x)) + b, s)
            for row, b in zip(lay["w"], lay["b"])
        ]
    logits = [
        sum(wi * xi for wi, xi in zip(row, x)) + b
        for row, b in zip(m["dense_w"], m["dense_b"])
    ]
    # First maximum wins, as in gadgets::argmax_ref.
    best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


# ── numpy ────────────────────────────────────────────────────────────────────


def forward_numpy(m, img, np):
    s = m["shift"]

    def conv(x, in_c, ih, iw, kernels, bias, k):
        x = x.reshape(in_c, ih, iw)
        oh, ow = ih - k + 1, iw - k + 1
        # im2col: (in_c*k*k, oh*ow), rows in (ic, ky, kx) order so the flat
        # kernel rows apply unchanged.
        cols = np.empty((in_c * k * k, oh * ow), dtype=np.int64)
        r = 0
        for ic in range(in_c):
            for ky in range(k):
                for kx in range(k):
                    cols[r] = x[ic, ky : ky + oh, kx : kx + ow].reshape(-1)
                    r += 1
        w = np.asarray(kernels, dtype=np.int64)
        v = w @ cols + (np.asarray(bias, dtype=np.int64) << s)[:, None]
        return np.where(v >= 0, v >> s, 0).reshape(-1), len(kernels), oh, ow

    def pool(x, c, ih, iw):
        oh, ow = (ih - 2) // 2 + 1, (iw - 2) // 2 + 1
        x = x.reshape(c, ih, iw)
        # Trim any odd trailing row/column the floor geometry drops.
        x = x[:, : oh * 2, : ow * 2].reshape(c, oh, 2, ow, 2)
        return x.max(axis=(2, 4)).reshape(-1), c, oh, ow

    c1 = m["conv1"]
    f, c, h, w = conv(
        np.asarray(img, dtype=np.int64),
        m["in_c"],
        m["d"],
        m["d"],
        c1["kernels"],
        c1["bias"],
        c1["k"],
    )
    f, c, h, w = pool(f, c, h, w)
    for cl in m["extra_conv"]:
        f, c, h, w = conv(
            f, cl["in_c"], cl["in_h"], cl["in_w"], cl["kernels"], cl["bias"], cl["k"]
        )
        f, c, h, w = pool(f, c, h, w)
    x = f
    for lay in m["hidden"]:
        v = np.asarray(lay["w"], dtype=np.int64) @ x + np.asarray(
            lay["b"], dtype=np.int64
        )
        x = np.where(v >= 0, v >> s, 0)
    logits = np.asarray(m["dense_w"], dtype=np.int64) @ x + np.asarray(
        m["dense_b"], dtype=np.int64
    )
    return int(np.argmax(logits))


# ── torch (float32) ──────────────────────────────────────────────────────────


def build_torch(m, torch):
    """The same weights as float32 convolutions.

    Not bit-identical: the deployed model is integer with a fused shift, and this
    is float. It answers a different question — "what does this cost in the
    framework everyone actually uses" — so its accuracy is reported next to the
    integer one rather than asserted equal to it.
    """
    import torch.nn as nn

    s = m["shift"]
    scale = float(1 << s)
    layers = []
    c1 = m["conv1"]
    k = c1["k"]
    conv = nn.Conv2d(m["in_c"], len(c1["kernels"]), k, bias=True)
    with torch.no_grad():
        conv.weight.copy_(
            torch.tensor(c1["kernels"], dtype=torch.float32).reshape(
                len(c1["kernels"]), m["in_c"], k, k
            )
            / scale
        )
        conv.bias.copy_(torch.tensor(c1["bias"], dtype=torch.float32))
    layers += [conv, nn.ReLU(), nn.MaxPool2d(2, 2)]
    for cl in m["extra_conv"]:
        k = cl["k"]
        c = nn.Conv2d(cl["in_c"], len(cl["kernels"]), k, bias=True)
        with torch.no_grad():
            c.weight.copy_(
                torch.tensor(cl["kernels"], dtype=torch.float32).reshape(
                    len(cl["kernels"]), cl["in_c"], k, k
                )
                / scale
            )
            c.bias.copy_(torch.tensor(cl["bias"], dtype=torch.float32))
        layers += [c, nn.ReLU(), nn.MaxPool2d(2, 2)]
    layers += [nn.Flatten()]
    for lay in m["hidden"]:
        lin = nn.Linear(len(lay["w"][0]), len(lay["w"]))
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(lay["w"], dtype=torch.float32) / scale)
            lin.bias.copy_(torch.tensor(lay["b"], dtype=torch.float32) / scale)
        layers += [lin, nn.ReLU()]
    out = nn.Linear(len(m["dense_w"][0]), len(m["dense_w"]))
    with torch.no_grad():
        out.weight.copy_(torch.tensor(m["dense_w"], dtype=torch.float32) / scale)
        out.bias.copy_(torch.tensor(m["dense_b"], dtype=torch.float32) / scale)
    layers += [out]
    net = nn.Sequential(*layers).eval()
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="JSON from `cargo run --example export_medmnist`")
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--impl", default="all", choices=["pure", "numpy", "torch", "all"])
    args = ap.parse_args()

    with open(args.model) as fh:
        m = json.load(fh)
    n = min(args.images, len(m["images"]))
    imgs, labels = m["images"][:n], m["labels"][:n]
    print(f"{m['dataset']} — {n} images, {m['in_c']}x{m['d']}x{m['d']}, shift {m['shift']}")

    want = args.impl
    results: dict[str, tuple[int, float]] = {}

    if want in ("pure", "all"):
        t0 = time.perf_counter()
        preds = [forward_pure(m, img) for img in imgs]
        dt = time.perf_counter() - t0
        results["pure python"] = (sum(p == l for p, l in zip(preds, labels)), dt)
        ref = preds

    if want in ("numpy", "all"):
        try:
            import numpy as np
        except ImportError:
            print("  numpy not installed — skipping", file=sys.stderr)
        else:
            t0 = time.perf_counter()
            preds = [forward_numpy(m, img, np) for img in imgs]
            dt = time.perf_counter() - t0
            results["numpy"] = (sum(p == l for p, l in zip(preds, labels)), dt)
            if want == "all" and preds != ref:
                print("  ⚠️  numpy disagrees with pure python — DO NOT quote", file=sys.stderr)
                return 1

    if want in ("torch", "all"):
        try:
            import torch
        except ImportError:
            print("  torch not installed — skipping", file=sys.stderr)
        else:
            net = build_torch(m, torch)
            x = torch.tensor(imgs, dtype=torch.float32).reshape(
                n, m["in_c"], m["d"], m["d"]
            )
            with torch.no_grad():
                net(x[:1])  # warm up: first call pays lazy init
                t0 = time.perf_counter()
                out = net(x)
                dt = time.perf_counter() - t0
            preds = out.argmax(dim=1).tolist()
            results["torch float32"] = (
                sum(p == l for p, l in zip(preds, labels)),
                dt,
            )

    print()
    for name, (correct, dt) in results.items():
        print(
            f"  {name:<16} {correct}/{n} correct   "
            f"{dt * 1000:8.1f} ms total   {dt * 1000 / n:7.2f} ms/image"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
