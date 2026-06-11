# baru

[![CI](https://github.com/emmorais/baru/actions/workflows/ci.yml/badge.svg)](https://github.com/emmorais/baru/actions/workflows/ci.yml)

Private computation

## Overview

This repository contains two main components:

- Interpreter: a minimal Lisp-like evaluator implemented in Rust (parser, AST, trampolined evaluator, REPL). It is exercised by the feature tests in `tests/features.rs`.

- Crypto primitives: a small three-party additive-sharing (ABY3-like) library under `src/crypto/` implementing `Share3` types, Beaver triple generation, secure multiplication, arithmetic-to-bit (`a2b`) masked-open conversion, bitwise ops and comparison helpers, plus simulator and transport abstractions for inproc and TCP testing.

See the `src/` tree and `tests/` for usage examples and tests.

## Quickstart

Build and run the tests:

```
cargo test
```

Run the REPL:

```
cargo run --release
```

The REPL exposes a small Lisp-like language with a few builtins useful to exercise MPC (for example: `share`, `reconstruct`, `a2b`, `b2a`).

## Example: private ReLU (single-neuron)

This example shows how to express a simple neuron with ReLU activation in the REPL / language and how MPC is used to compute it without revealing intermediate private inputs.

Lisp-like expression (REPL input):

```
(let ((w (share -1 8))   ; weight (private signed 8-bit)
      (x (share  3 8))   ; input (private signed 8-bit)
      (b (share  0 8)))  ; bias (private signed 8-bit)
  (let ((z (+ (* w x) b)))
    (reconstruct z)      ; simulated reveal of z to the caller (owner)
    (let ((relu (if (> z 0) z 0)))
      (reconstruct relu))))
```

What happens under the hood (high level):

- `share` converts a local immediate (signed integer) into a two's-complement field element and creates a `ShareRef` owned by the calling participant. The simulated `Backend` splits values into shares and stores them as `ShareRef`.

- Arithmetic on `ShareRef` values (`+`, `*`, `-`) is carried out by the `Backend` without reconstructing the secret: simulated MPC primitives (Beaver triples, masked opens, etc.) are used to produce new `ShareRef` results.

- Comparison operators (`>`, `<`, `==`, ...) that operate on shared values use a masked/reconstruct flow implemented by the backend: the simulator reconstructs the compared value to the owner, interprets it as a signed integer (two's complement using the provided bit-width), then re-shares the boolean result as a 1-bit `ShareRef`. This keeps the boolean as a shared value for downstream computation.

- When an `if` condition is a shared boolean, the evaluator reveals the condition to the owner to decide the branch. To preserve the invariant that callers expect a shared result when the condition was a shared boolean, the evaluator normalizes the chosen branch's numeric/boolean result into a share (converting NUM or BOOL branch results into a `ShareRef` owned by the original condition owner with the same bit-width). This ensures downstream code continues to operate on shares and prevents accidental leaks.

Notes and limitations

- The repository currently ships with a single-process simulator (`SimBackend`) that mimics the MPC protocol locally. It is useful for development and testing but does not provide real networked security guarantees.

- The `reconstruct` builtin uses the backend `reveal` API; in the simulator this returns the reconstructed plain value (visible to the owner). In a deployed multi-party backend the reveal implementation would perform the proper secure reveal protocol.

- Bit-widths (`nbits`) are important: immediates are converted to two's-complement representations according to the requested `nbits` when shared. Comparison and signed interpretation use the same `nbits` to decode two's-complement values correctly.

Further reading

- See `tests/features.rs` for concrete examples exercised by the test-suite.
- Check `src/crypto/` for the simulated ABY3-like primitives and the implementation of `Share3` operations.

Contributions and issues

Suggestions, bug reports and pull requests are welcome.