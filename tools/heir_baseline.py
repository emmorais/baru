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

import io
import json
import os
import re
import statistics
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

# The only security levels that count as secure. `HEStd_NotSet` is OpenFHE's
# "do not check" mode and is the trap google/heir#1232 fell into.
SECURE_LEVELS = ("HEStd_128_classic", "HEStd_192_classic", "HEStd_256_classic")


def relu(v):
    return v if v > 0.0 else 0.0


def forward_plain(layers, x, activation="relu"):
    """Float reference forward pass — the same arithmetic CKKS approximates.

    `activation` overrides the per-layer tag for hidden layers, so `--self-test`
    evaluates the *same* function the FHE arm would and can therefore be used to
    tell a genuine FHE error apart from a float-versus-fixed-point divergence.
    """
    cur = list(x)
    for layer in layers:
        pre = [
            sum(w * v for w, v in zip(row, cur)) + b
            for row, b in zip(layer["weights"], layer["bias"])
        ]
        if layer["activation"] == "identity":
            cur = pre
        elif activation == "square":
            cur = [v * v for v in pre]
        elif activation == "relu":
            cur = [relu(v) for v in pre]
        else:
            raise ValueError("unknown activation %r" % (activation,))
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
    act = req.get("activation", "relu")
    classes = [argmax(forward_plain(req["layers"], x, act)) for x in req["inputs"]]
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
            # What the run actually evaluated — echoed back so the caller's
            # activation check has something real to compare against and does not
            # silently skip.
            "activation": act,
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


# OpenFHE's own default when `SetSecurityLevel` is not called. Verified in the
# installed headers (`pke/scheme/gen-cryptocontext-params-defaults.h`:
# `constexpr SecurityLevel securityLevel = HEStd_128_classic;`) rather than
# assumed -- and it matters, because HEIR's emitted C++ sets only the
# multiplicative depth and the plaintext modulus, never the security level.
OPENFHE_DEFAULT_LEVEL = "HEStd_128_classic"


def configure_openfhe_env():
    """Derive the OPENFHE_* variables HEIR needs from a single prefix.

    HEIR resolves OpenFHE by scanning `/usr/local` and `/usr`, or by reading
    `OPENFHE_INCLUDE_DIR` / `OPENFHE_LIB_DIR` / `OPENFHE_LINK_LIBS`. A build
    installed anywhere else needs those three set, which is fiddly enough to be
    worth doing here from `BARU_OPENFHE_PREFIX` instead of asking every caller to
    export four variables in the right shape.
    """
    prefix = os.environ.get("BARU_OPENFHE_PREFIX")
    if not prefix or os.environ.get("OPENFHE_INCLUDE_DIR"):
        return
    inc = os.path.join(prefix, "include", "openfhe")
    lib = os.path.join(prefix, "lib")
    if not os.path.isdir(inc):
        return
    os.environ["OPENFHE_INCLUDE_DIR"] = ":".join([
        inc,
        os.path.join(inc, "core"),
        os.path.join(inc, "pke"),
        os.path.join(inc, "binfhe"),
        os.path.join(prefix, "include"),
    ])
    os.environ["OPENFHE_LIB_DIR"] = lib
    os.environ.setdefault(
        "OPENFHE_LINK_LIBS", "OPENFHEpke:OPENFHEcore:OPENFHEbinfhe:omp"
    )
    # The emitted C++ is linked with `-L$OPENFHE_LIB_DIR` only, so libomp has to
    # be reachable from there. A symlink is the least surprising way to do that.
    omp = os.environ.get(
        "BARU_LIBOMP_DYLIB", "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    )
    link = os.path.join(lib, "libomp.dylib")
    if os.path.exists(omp) and not os.path.exists(link):
        try:
            os.symlink(omp, link)
        except OSError as e:
            sys.stderr.write(
                "could not link libomp into the OpenFHE lib dir: %r\n" % (e,)
            )
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = ":".join(
        [q for q in [lib, os.path.dirname(omp), existing] if q]
    )


def emitted_params(debug_output):
    """Read the parameters HEIR actually compiled in, out of its emitted C++.

    There is no Python-side accessor: the compiled object exposes only
    `compilation_result`, `eval`, `setup`, `decrypt_result` and `original`, and
    `compilation_result` carries no crypto parameters. So the ground truth is the
    generated `.cpp`, which `debug=True` leaves on disk and names on stderr.

    Returns `(security_level, source, depth, plaintext_modulus)`.
    """
    level, source, depth, ptmod = None, "unknown", None, None
    for path in re.findall(r"(\S+\.cpp)", debug_output):
        if not os.path.exists(path) or path.endswith("_bindings.cpp"):
            continue
        with open(path) as fh:
            cpp = fh.read()
        m = re.search(r"SetSecurityLevel\(\s*([A-Za-z0-9_]+)\s*\)", cpp)
        if m:
            level, source = m.group(1), "set by HEIR"
        else:
            # HEIR does not call SetSecurityLevel, so OpenFHE's own default
            # applies. Reported as such rather than silently as a measurement.
            level, source = OPENFHE_DEFAULT_LEVEL, "OpenFHE default (HEIR sets none)"
        m = re.search(r"SetMultiplicativeDepth\(\s*(\d+)\s*\)", cpp)
        depth = int(m.group(1)) if m else None
        m = re.search(r"SetPlaintextModulus\(\s*(\d+)\s*\)", cpp)
        ptmod = int(m.group(1)) if m else None
        break
    return level, source, depth, ptmod


def _dense_1d(vals):
    return "[" + ", ".join("%.9e" % v for v in vals) + "]"


def _dense_2d(rows):
    return "[" + ", ".join(_dense_1d(r) for r in rows) + "]"


def build_mlir(layers, activation, name="net"):
    """Emit the network as `linalg` MLIR on tensors with a secret input.

    **Why MLIR and not the decorator.** HEIR's Python frontend runs the function
    through numba, and its supported surface does not include what a dense layer
    needs: measured on 2026.8.1, tensor subscripting (`x[0]`) raises
    "Unsupported IR Element: getitem", `@` raises "Extension handling for
    non-integer (e.g., floats, tensors) types is not yet supported", and
    `np.sum` / `np.matmul` are unsupported. Elementwise tensor ops (`x * x`) do
    work. So a matvec is not expressible there at all.

    `linalg` on tensors is the level HEIR's ML pipeline actually targets -- the
    Torch route reaches it via torch-mlir -- and `compile(mlir_str=...)` is the
    door to it. Writing the MLIR directly skips the torch-mlir dependency
    entirely.

    Conventions that took measurement to pin down: the argument attribute is
    `{secret.secret}`, the generated encryptor is named after the *argument*
    (`encrypt_arg_0`, with an underscore), and `arith.addf`/`arith.mulf` are
    elementwise on shaped types so the bias add and the squaring need no loop.
    """
    lines = ["module {"]
    in_dim = len(layers[0]["weights"][0])
    out_dim = len(layers[-1]["weights"])
    lines.append(
        "  func.func @%s(%%arg0: tensor<%dxf32> {secret.secret}) -> tensor<%dxf32> {"
        % (name, in_dim, out_dim)
    )
    lines.append("    %zero = arith.constant 0.0 : f32")
    cur, cur_dim = "%arg0", in_dim
    for li, layer in enumerate(layers):
        n_out = len(layer["weights"])
        lines.append(
            "    %%w%d = arith.constant dense<%s> : tensor<%dx%dxf32>"
            % (li, _dense_2d(layer["weights"]), n_out, cur_dim)
        )
        lines.append(
            "    %%b%d = arith.constant dense<%s> : tensor<%dxf32>"
            % (li, _dense_1d(layer["bias"]), n_out)
        )
        lines.append("    %%e%d = tensor.empty() : tensor<%dxf32>" % (li, n_out))
        lines.append(
            "    %%f%d = linalg.fill ins(%%zero : f32) outs(%%e%d : tensor<%dxf32>)"
            " -> tensor<%dxf32>" % (li, li, n_out, n_out)
        )
        lines.append(
            "    %%m%d = linalg.matvec ins(%%w%d, %s : tensor<%dx%dxf32>,"
            " tensor<%dxf32>) outs(%%f%d : tensor<%dxf32>) -> tensor<%dxf32>"
            % (li, li, cur, n_out, cur_dim, cur_dim, li, n_out, n_out)
        )
        lines.append(
            "    %%p%d = arith.addf %%m%d, %%b%d : tensor<%dxf32>" % (li, li, li, n_out)
        )
        if layer["activation"] == "identity":
            cur = "%%p%d" % li
        elif activation == "square":
            lines.append(
                "    %%h%d = arith.mulf %%p%d, %%p%d : tensor<%dxf32>"
                % (li, li, li, n_out)
            )
            cur = "%%h%d" % li
        else:
            raise ValueError("unsupported activation %r" % (activation,))
        cur_dim = n_out
    lines.append("    return %s : tensor<%dxf32>" % (cur, cur_dim))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _ct_arity(main_func):
    """How many ciphertext sequences the emitted entry point actually takes.

    The pybind docstring is the only place this is exposed; `compilation_result`
    does not carry it.
    """
    doc = getattr(main_func, "__doc__", "") or ""
    return doc.split("->")[0].count("Sequence[")


class _HighLevelRunner:
    """HEIR's supported path: `setup` / `encrypt_arg_0` / `eval` / `decrypt_result`."""

    path = "high-level"

    def __init__(self, fn, _cr):
        self.fn = fn

    def setup(self):
        self.fn.setup()

    def encrypt(self, x):
        return self.fn.encrypt_arg_0(x)

    def evaluate(self, ct):
        return self.fn.eval(ct)

    def decrypt(self, out):
        return self.fn.decrypt_result(out)


class _RawRunner:
    """Direct pybind path, for networks whose `eval()` cannot be called at all.

    **Why this exists** (measured against google/heir 2026.8.1 on 2026-10-20).
    For some networks HEIR's layout passes promote the `linalg.fill`
    zero-initializer into a *second* ciphertext argument of the emitted C++ --

        std::vector<CiphertextT> net(CryptoContextT cc,
                                     std::vector<CiphertextT> v0,
                                     const std::vector<CiphertextT>& v1);

    -- and emit a companion `net__encrypt__zeros` to produce it. But the Python
    layer still registers a single argument (`compilation_result.arg_names ==
    ['arg_0']`), so `eval()` passes one sequence to a two-sequence binding and
    pybind raises `incompatible function arguments`. The network is perfectly
    compiled; only the Python wrapper's view of its arity is wrong. Whether it
    triggers depends on the weights: a 16x8x4 net with uniform weights compiles
    to one argument and the same shape with the fixture's weights to two.

    The emitted module exports every entry point we need, so we drive them
    directly. Argument order was established against the plaintext reference
    (`(arg0, zeros)` reproduces it exactly; `(zeros, arg0)` does not), which is
    also what the report's class check re-verifies on every run.

    The zero ciphertext does not depend on the input, so it is produced once in
    `setup` and excluded from the per-inference encrypt time.
    """

    path = "raw-pybind"

    def __init__(self, _fn, cr):
        mod, name = cr.module, cr.func_name
        self.main = getattr(mod, name)
        self._gen = getattr(mod, "%s__generate_crypto_context" % name)
        self._conf = getattr(mod, "%s__configure_crypto_context" % name)
        self._enc = getattr(mod, "%s__encrypt__arg0" % name)
        self._zero_fn = getattr(mod, "%s__encrypt__zeros" % name)
        self._dec = getattr(mod, "%s__decrypt__result0" % name)

    def setup(self):
        cc = self._gen()
        self._kp = cc.KeyGen()
        self._cc = self._conf(cc, self._kp.secretKey)
        self._zeros = self._zero_fn(self._cc, self._kp.publicKey)

    def encrypt(self, x):
        return self._enc(self._cc, [float(v) for v in x], self._kp.publicKey)

    def evaluate(self, ct):
        return self.main(self._cc, ct, self._zeros)

    def decrypt(self, out):
        return self._dec(self._cc, out, self._kp.secretKey)


def run_heir(req):
    configure_openfhe_env()
    try:
        import heir
        from heir import compile as heir_compile
    except ImportError as e:
        return fail(
            "heir is not importable (%s). Install it with:\n"
            '    pip install "heir-py[python,openfhe]"\n'
            "It publishes cp310-abi3 wheels only, so Python >= 3.10 is required "
            "(this interpreter is %d.%d). Run with --self-test to exercise the "
            "request/report contract without it."
            % (e, sys.version_info[0], sys.version_info[1])
        )

    try:
        import numpy as np
    except ImportError as e:
        return fail("numpy is required for the FHE arm: %r" % (e,))

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

    try:
        src = build_mlir(layers, activation)
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

    # `debug=True` is what leaves the emitted C++ on disk and names it, which is
    # the only way to read back the crypto parameters HEIR actually compiled in
    # (see `emitted_params`). Its chatter is captured rather than printed, because
    # stdout is the report channel.
    kwargs = {"mlir_str": src, "scheme": scheme, "debug": True}
    if opts:
        kwargs["heir_opt_options"] = opts
    if backend is not None:
        kwargs["backend"] = backend

    chatter = io.StringIO()
    t0 = time.perf_counter()
    try:
        with redirect_stdout(chatter), redirect_stderr(chatter):
            fn = heir_compile(**kwargs)
    except Exception as e:  # noqa: BLE001 - the message is the useful artifact
        return fail(
            "heir.compile failed: %r\nGenerated MLIR:\n%s\nHEIR output:\n%s"
            % (e, src, chatter.getvalue()[-3000:])
        )
    compile_ms = (time.perf_counter() - t0) * 1e3
    level, level_source, depth, ptmod = emitted_params(chatter.getvalue())

    cr = fn.compilation_result
    arity = _ct_arity(cr.main_func)
    runner_cls = _HighLevelRunner if arity <= 1 else _RawRunner
    try:
        runner = runner_cls(fn, cr)
    except AttributeError as e:
        return fail(
            "the emitted module takes %d ciphertext arguments and does not "
            "export the entry points needed to drive it directly (%r). "
            "See _RawRunner's docstring." % (arity, e)
        )

    try:
        t0 = time.perf_counter()
        with redirect_stdout(chatter), redirect_stderr(chatter):
            runner.setup()
        keygen_ms = (time.perf_counter() - t0) * 1e3
    except Exception as e:  # noqa: BLE001
        return fail("key setup failed: %r" % (e,))

    repeats = max(1, int(req.get("repeats", 1)))
    enc_ms, eval_ms, dec_ms, classes, ct_bytes = [], [], [], [], 0
    all_logits, ref_logits = [], []
    for x in req["inputs"]:
        try:
            t0 = time.perf_counter()
            ct = runner.encrypt(np.asarray(x, dtype=np.float32))
            enc_ms.append((time.perf_counter() - t0) * 1e3)

            best = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = runner.evaluate(ct)
                dt = (time.perf_counter() - t0) * 1e3
                best = dt if best is None else min(best, dt)
            eval_ms.append(best)

            t0 = time.perf_counter()
            logits = runner.decrypt(out)
            dec_ms.append((time.perf_counter() - t0) * 1e3)
        except Exception as e:  # noqa: BLE001
            return fail("homomorphic evaluation failed: %r" % (e,))

        vec = [float(v) for v in np.asarray(logits).reshape(-1)]
        classes.append(argmax(vec))
        all_logits.append(vec)
        # The float reference for the same input, so a disagreement can be read as
        # a margin rather than guessed at: a near-tie flipped by CKKS noise is a
        # different finding from a wrong computation.
        ref_logits.append(forward_plain(layers, x, activation))
        try:
            ct_bytes = max(ct_bytes, len(bytes(ct)))
        except Exception:  # noqa: BLE001 - size is nice to have, not load-bearing
            pass


    json.dump(
        {
            "ok": True,
            "simulated": False,
            "heir_version": getattr(heir, "__version__", "unknown"),
            "scheme": scheme,
            "backend": req.get("backend", "openfhe"),
            "ring_dim": 0,
            "security_level": str(level) if level else "HEStd_NotSet",
            "security_level_source": level_source,
            "multiplicative_depth": depth,
            "plaintext_modulus": ptmod,
            "activation": activation,
            "activation_degree": 2 if activation == "square" else 0,
            "heir_opt_options": opts,
            "driver_path": runner.path,
            "ciphertext_arity": arity,
            "compile_ms": compile_ms,
            "keygen_ms": keygen_ms,
            "encrypt_ms": statistics.median(enc_ms),
            "eval_ms": statistics.median(eval_ms),
            "decrypt_ms": statistics.median(dec_ms),
            "ciphertext_bytes": ct_bytes,
            "classes": classes,
            "logits": all_logits,
            "reference_logits": ref_logits,
            "notes": "eval_ms is the min over %d repeats then the median over "
            "inputs; keygen excluded. activation=%s; driver=%s"
            % (repeats, activation, runner.path),
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
