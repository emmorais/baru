#!/usr/bin/env python3
"""Drive HEIR (github.com/google/heir) as an FHE baseline for `baru bench-heir`.

Reads one request JSON on stdin, writes one report JSON on stdout. Nothing else
goes to stdout — diagnostics go to stderr — so the Rust side can parse the whole
of stdout as a document.

Why a baseline generator and not a target to beat
-------------------------------------------------
HEIR is compiler *infrastructure*. Its paper (arXiv:2508.11095) claims generality
-- "support all mainstream techniques ... integrate with all major software
libraries and hardware accelerators" -- and reports no private-ML performance
evaluation. Beating its numbers would not be a result. What it is good for is
supplying an FHE arm that runs on tuned C++ (OpenFHE / Lattigo) rather than on
Baru's own hand-written NTT, which removes the "research Rust versus optimised
C++" objection from the comparison in the one direction that matters.

Two things this script refuses to paper over
--------------------------------------------
1. **Security parameters.** google/heir#1232 reports an MNIST MLP at ~30 s with
   ``HEStd_NotSet`` -- i.e. no security level at all -- and ~6 min at the secure
   ring dimension. A 12x difference sits between an insecure and a secure
   baseline, so the level actually used is reported in every case and the Rust
   side refuses to quote a figure that is not secure. An insecure baseline is
   worse than no baseline.
2. **The FHE arm's ReLU is approximate.** CKKS cannot compare, so ReLU becomes a
   polynomial of some degree, and the degree trades accuracy against depth
   (Orion, ASPLOS'25, measures 2.1% accuracy for 1.77x speed on that axis). The
   MPC arm computes ReLU *exactly*. The degree is therefore reported, and the
   Rust side requires the predicted classes to match before any latency is
   usable -- which is what stops a cheap, too-coarse approximation from looking
   fast.

Usage
-----
    baru bench-heir ...                 # normal path; Baru invokes this
    heir_baseline.py < request.json     # by hand
    heir_baseline.py --self-test < request.json

``--self-test`` evaluates the network in **plaintext Python** and reports
``security_level: "HEStd_NotSet"``. It exists so the request/report contract can
be exercised without HEIR installed -- and because a run that reports no security
level must be rejected, it doubles as the fixture that proves the Rust side's
security gate fires. It deliberately does not pretend to be a measurement:
``simulated`` is true and every timing is zero.

Requirements for the real path
------------------------------
    python >= 3.10  (heir-py publishes cp310-abi3 wheels only)
    pip install "heir-py[python,openfhe]"

This file itself runs on 3.9 so that ``--self-test`` works anywhere.
"""

import json
import statistics
import sys
import time

# The only security levels that count as secure. `HEStd_NotSet` is OpenFHE's
# "do not check" mode and is the trap google/heir#1232 fell into.
SECURE_LEVELS = ("HEStd_128_classic", "HEStd_192_classic", "HEStd_256_classic")


def relu(v):
    return v if v > 0.0 else 0.0


def forward_plain(layers, x):
    """Float reference forward pass — the same arithmetic CKKS approximates."""
    cur = list(x)
    for layer in layers:
        pre = [
            sum(w * v for w, v in zip(row, cur)) + b
            for row, b in zip(layer["weights"], layer["bias"])
        ]
        if layer["activation"] == "relu":
            cur = [relu(v) for v in pre]
        elif layer["activation"] == "identity":
            cur = pre
        else:
            raise ValueError("unknown activation %r" % (layer["activation"],))
    return cur


def argmax(vals):
    """First-max-wins, matching Baru's `argmax_ref` tie-break exactly."""
    best = 0
    for i in range(1, len(vals)):
        if vals[i] > vals[best]:
            best = i
    return best


def fail(msg, **extra):
    out = {"ok": False, "error": msg}
    out.update(extra)
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 1


def self_test(req):
    """Plaintext evaluation, honestly labelled as not a measurement."""
    classes = [argmax(forward_plain(req["layers"], x)) for x in req["inputs"]]
    json.dump(
        {
            "ok": True,
            "simulated": True,
            "heir_version": "none (--self-test)",
            "scheme": req.get("scheme", "ckks"),
            "backend": "plaintext",
            "ring_dim": 0,
            # Honest, and therefore rejected by the caller's security gate. That
            # rejection is the point of this mode.
            "security_level": "HEStd_NotSet",
            # `forward_plain` computes an exact ReLU, so this is what the run
            # actually evaluated — echoed back so the caller's activation check
            # has something real to compare and does not silently skip.
            "activation": "relu",
            "activation_degree": 0,
            "keygen_ms": 0.0,
            "encrypt_ms": 0.0,
            "eval_ms": 0.0,
            "decrypt_ms": 0.0,
            "ciphertext_bytes": 0,
            "classes": classes,
            "notes": "plaintext self-test: exercises the contract, measures nothing",
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


def build_source(layers, in_dim, activation):
    """Emit the network as a `Secret`-annotated Python function, as text.

    HEIR's Python frontend compiles a *type-annotated function* — the documented
    entry point is a decorator over `Secret[...]` arguments, not a Torch module.
    (The Torch path in HEIR's ML pipeline goes torch-mlir -> linalg MLIR text and
    enters through `compile(mlir_str=...)`, a different door.) Weights are
    plaintext, so they are inlined as literals and only the input is secret.
    """
    lines = ["def _net(x: Secret[Tensor[[%d], F32]]):" % in_dim]
    cur = ["x[%d]" % i for i in range(in_dim)]
    for li, layer in enumerate(layers):
        for oi, (row, b) in enumerate(zip(layer["weights"], layer["bias"])):
            terms = [
                "%.17g * %s" % (w, v) for w, v in zip(row, cur) if w != 0.0
            ]
            body = " + ".join(terms) if terms else "0.0"
            lines.append("    p%d_%d = %s + %.17g" % (li, oi, body, b))
        names = []
        for oi in range(len(layer["weights"])):
            pre = "p%d_%d" % (li, oi)
            if layer["activation"] == "identity":
                names.append(pre)
            elif activation == "square":
                lines.append("    h%d_%d = %s * %s" % (li, oi, pre, pre))
                names.append("h%d_%d" % (li, oi))
            else:
                raise ValueError("unsupported activation %r" % (activation,))
        cur = names
    # One output: the logit vector, returned as a tuple so the caller can argmax.
    lines.append("    return (%s,)" % ", ".join(cur))
    return "\n".join(lines)


def run_heir(req):
    try:
        import heir
        from heir import compile as heir_compile
        from heir.mlir import F32, Secret, Tensor
    except ImportError as e:
        return fail(
            "heir is not importable (%s). Install it with:\n"
            '    pip install "heir-py[python,openfhe]"\n'
            "It publishes cp310-abi3 wheels only, so Python >= 3.10 is required "
            "(this interpreter is %d.%d). Run with --self-test to exercise the "
            "request/report contract without it."
            % (e, sys.version_info[0], sys.version_info[1])
        )

    scheme = req.get("scheme", "ckks")
    activation = req.get("activation", "square")
    layers = req["layers"]

    # CKKS cannot compare, so an *exact* ReLU is not expressible on this arm. The
    # options are a polynomial approximation (whose degree trades accuracy for
    # depth) or an FHE-friendly activation that both arms can compute exactly --
    # Orion (ASPLOS'25) uses x^2 for its MNIST networks for exactly this reason.
    # Guessing between them would silently change the network being compared, so
    # this refuses instead.
    if any(l["activation"] == "relu" for l in layers) and activation != "relu_poly":
        if activation != "square":
            return fail(
                "this network has ReLU layers and the FHE arm cannot compute ReLU "
                "exactly. Choose deliberately: `activation: \"square\"` makes BOTH "
                "arms compute x^2 (exact on each, and what Orion does for MNIST), "
                "while `activation: \"relu_poly\"` approximates ReLU on this arm "
                "only and needs the degree justified. Refusing to pick for you, "
                "because either choice changes what is being compared."
            )
    if activation == "relu_poly":
        return fail(
            "activation `relu_poly` is not implemented: it needs a minimax "
            "polynomial over a justified input range, which is a modelling "
            "decision and not a default. `square` is implemented and exact on "
            "both arms."
        )

    in_dim = len(layers[0]["weights"][0])
    try:
        src = build_source(layers, in_dim, activation)
    except ValueError as e:
        return fail(str(e))

    # Security level and any approximation options are *not* parameters of
    # `heir.compile` -- its signature is
    # `compile(mlir_str=None, scheme="bgv", backend=None, config=None,
    #          debug=False, heir_opt_options=None)`
    # -- so they go through `heir_opt_options` as compiler-pass flags. The exact
    # flag spelling is version-dependent and is therefore *reported* rather than
    # trusted: whatever level HEIR ends up using is read back below, and the
    # caller refuses to quote a latency if it is not a real one. An unverified
    # configuration cannot silently produce a quotable number.
    opts = list(req.get("heir_opt_options") or [])

    backend = None
    try:
        from heir.backends.openfhe import OpenFHEBackend, resolve_config

        if req.get("backend", "openfhe") == "openfhe":
            backend = OpenFHEBackend(resolve_config())
    except Exception as e:  # noqa: BLE001 - optional, and the message is useful
        sys.stderr.write("could not construct the OpenFHE backend: %r\n" % (e,))

    ns = {"Secret": Secret, "Tensor": Tensor, "F32": F32}
    try:
        exec(compile(src, "<heir-net>", "exec"), ns)  # noqa: S102
    except Exception as e:  # noqa: BLE001
        return fail("the generated network does not compile as Python: %r" % (e,))

    t0 = time.perf_counter()
    try:
        kwargs = {"scheme": scheme, "heir_opt_options": opts} if opts else {"scheme": scheme}
        if backend is not None:
            kwargs["backend"] = backend
        fn = heir_compile(**kwargs)(ns["_net"])
    except Exception as e:  # noqa: BLE001 - the message is the useful artifact
        return fail(
            "heir.compile failed: %r\n"
            "The generated function was:\n%s" % (e, src)
        )
    compile_ms = (time.perf_counter() - t0) * 1e3

    try:
        t0 = time.perf_counter()
        fn.setup()
        keygen_ms = (time.perf_counter() - t0) * 1e3
    except Exception as e:  # noqa: BLE001
        return fail("key setup failed: %r" % (e,))

    repeats = max(1, int(req.get("repeats", 1)))
    enc_ms, eval_ms, dec_ms, classes, ct_bytes = [], [], [], [], 0
    for x in req["inputs"]:
        try:
            t0 = time.perf_counter()
            ct = fn.encrypt_x(x)
            enc_ms.append((time.perf_counter() - t0) * 1e3)

            best = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = fn.eval(ct)
                dt = (time.perf_counter() - t0) * 1e3
                best = dt if best is None else min(best, dt)
            eval_ms.append(best)

            t0 = time.perf_counter()
            logits = fn.decrypt_result(out)
            dec_ms.append((time.perf_counter() - t0) * 1e3)
        except Exception as e:  # noqa: BLE001
            return fail("homomorphic evaluation failed: %r" % (e,))

        classes.append(argmax(list(logits)))
        try:
            ct_bytes = max(ct_bytes, len(bytes(ct)))
        except Exception:  # noqa: BLE001 - size is nice to have, not load-bearing
            pass

    # Whatever HEIR actually used, not what was asked for. If it does not report
    # a security level the caller refuses the number, which is the correct
    # outcome for an unverified configuration.
    params = getattr(fn, "parameters", None)
    level = getattr(params, "security_level", None)
    ring = getattr(params, "ring_dim", None) or getattr(params, "ring_dimension", 0) or 0

    json.dump(
        {
            "ok": True,
            "simulated": False,
            "heir_version": getattr(heir, "__version__", "unknown"),
            "scheme": scheme,
            "backend": req.get("backend", "openfhe"),
            "ring_dim": int(ring),
            "security_level": str(level) if level else "HEStd_NotSet",
            "activation": activation,
            "activation_degree": 2 if activation == "square" else 0,
            "heir_opt_options": opts,
            "compile_ms": compile_ms,
            "keygen_ms": keygen_ms,
            "encrypt_ms": statistics.median(enc_ms),
            "eval_ms": statistics.median(eval_ms),
            "decrypt_ms": statistics.median(dec_ms),
            "ciphertext_bytes": ct_bytes,
            "classes": classes,
            "notes": "eval_ms is the min over %d repeats then the median over "
            "inputs; keygen excluded. activation=%s" % (repeats, activation),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


def main(argv):
    simulate = "--self-test" in argv
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001
        return fail("could not parse the request on stdin: %r" % (e,))
    for key in ("layers", "inputs"):
        if key not in req:
            return fail("request is missing %r" % (key,))
    if not req["layers"]:
        return fail("request has no layers")
    return self_test(req) if simulate else run_heir(req)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
