# baru

[![CI](https://github.com/emmorais/baru/actions/workflows/ci.yml/badge.svg)](https://github.com/emmorais/baru/actions/workflows/ci.yml)

Private computation

## Overview

This repository contains two main components:

- Interpreter: a minimal Lisp-like evaluator implemented in Rust (parser, AST, trampolined evaluator, REPL). It is exercised by the feature tests in `tests/features.rs`.

- Crypto primitives: a small three-party additive-sharing (ABY3-like) library under `src/crypto/` implementing `Share3` types, Beaver triple generation, secure multiplication, arithmetic-to-bit (`a2b`) masked-open conversion, bitwise ops and comparison helpers, plus simulator and transport abstractions for inproc and TCP testing.

See the `src/` tree and `tests/` for usage examples and tests.