# The Baru Language

Baru is a small Lisp-like language for **privacy-preserving computation**. A single
program can be evaluated on four different backends:

| Backend | What it provides | Where |
|---|---|---|
| **Plain interpreter** | Ordinary evaluation (CEK-style trampolined evaluator) — the correctness baseline | `src/interpreter/` |
| **MPC** | 3-party secure computation with ABY3-style replicated secret sharing (plus an experimental 2-party garbled-circuits backend) | `src/crypto/`, `src/interpreter/*_backend.rs` |
| **FHE** | Fully homomorphic evaluation of compiled circuits — mock, TFHE-rs (exact integers) and SEAL CKKS (approximate reals) | `src/fhe/` |
| **ZKP** | Zero-knowledge proofs of circuit satisfaction using the Binius64 SNARK | `src/zkp/` |

This document describes the language surface: syntax, data types, special forms,
every builtin, the REPL, the CLI tools, and worked examples for each backend.

---

## 1. Getting started

```sh
cargo build                 # build everything
cargo test                  # unit + integration tests
cargo run --bin baru        # single-process REPL (SimBackend, no networking)
cargo run --bin baru bench  # benchmark harness (see §10)
cargo run --release --example mnist   # MNIST private-inference demo (see §11)
```

Multi-party TCP mode uses the `tcp_party` binary (see §6.5).

---

## 2. Syntax

Baru programs are s-expressions:

```lisp
(+ (* 6 7) (* 3 14))        ; => 84
```

- `;` starts a line comment.
- Whitespace separates tokens; parentheses delimit forms.

### Literals

| Literal | Example | Type |
|---|---|---|
| Integer | `42`, `-7` | `Num` (i64) |
| Fixed-point decimal | `3.5`, `-1.25` | `Fixed` — Q16.16 (stored as `value × 65536`) |
| String | `"hello\n"` | `Str` (escapes: `\n` `\t` `\\` `\"`) |
| Character | `'a'` | `Char` |
| Boolean | `#t`, `#f` | `Bool` |
| Quoted form | `'expr` (sugar for `(quote expr)`) | `Quoted` |

The tokenizer distinguishes the char literal `'a'` from quote shorthand `'abc` /
`'(...)` by lookahead: a single character followed by a closing quote is a char.

### Fixed-point (Q16.16)

A decimal literal like `42.5` is stored as the scaled integer `42.5 × 65536 =
2785280`. Addition/subtraction are direct; multiplication truncates (arithmetic
right-shift by 16). Mixed `Num`/`Fixed` arithmetic coerces the integer to fixed.
Fixed values print with 4 decimal digits (`3.5000`). Conversions are explicit via
`int->fixed`, `fixed->int`, `fixed->str` (§5.3).

---

## 3. Special forms and operators

### Binding and functions

```lisp
(let ((x 3) (y 4)) (+ x y))              ; local bindings; body may contain
                                          ; several expressions, evaluated in
                                          ; order — the last is the result
(letrec ((f (lambda (n) ...))) (f 10))   ; recursive (and mutually recursive) bindings
(lambda (a b) (+ a b))                    ; first-class closures, lexical capture
((lambda (x) (* x x)) 7)                  ; direct call => 49
```

Currying, higher-order functions, closures capturing their environment, and
mutual recursion (`even?`/`odd?` style) all work; the evaluator is trampolined so
deep recursion (Ackermann, Collatz, …) does not overflow the Rust stack.

### `:sintN` sugar — auto-sharing in `let` bindings

Inside a `let` binding, the annotation `:sintN` wraps the bound expression in
`(share expr N)`; `:sintN-from-P` wraps it in `(share-from P expr N)`:

```lisp
(let ((z :sint32 -5)) (if (> z 0) z 0))          ; z is a shared signed 32-bit value
(let ((a :sint32-from-1 30)) ...)                 ; party 1 provides the input
```

### Control flow

```lisp
(if cond then else)
```

Any nonzero number / `#t` is truthy. When `cond` is a **shared** value the
condition bit is revealed first — see §6.3.

### Operators

| Form | Meaning |
|---|---|
| `(+ a b)` `(- a b)` `(* a b)` | arithmetic (wrapping on i64; works on `Num`, `Fixed`, and shares) |
| `(/ a b)` | exact integer division |
| `(mod a b)` | modulo |
| `(- a)` | unary negation |
| `(== a b)` `(< a b)` `(> a b)` `(<= a b)` `(>= a b)` | comparisons — return `1`/`0` (a 1-bit share when operands are shared) |
| `(and a b)` `(or a b)` | logical on booleans, **bitwise** on numbers (`(and 6 3)` ⇒ `2`) |
| `(not x)` | logical/bitwise not |
| `(ash-left x n)` `(ash-right x n)` | arithmetic shifts |
| `(bitlen x)` | number of significant bits |
| `(atom x)` | 1 if `x` is an atom |
| `(null? lst)` | 1 if the list is empty |

### Lists

```lisp
(cons 1 (cons 2 '()))     ; build
(car lst)                  ; head
(cdr lst)                  ; tail
(list 1 2 3)               ; variadic constructor (builtin, §5.2)
```

### Metaprogramming

```lisp
(quote expr)   ; or 'expr — return the expression unevaluated
(eval q)       ; evaluate a quoted expression
(private e)    ; annotation marking e as a ZKP witness (see §8)
```

---

## 4. Data types

Runtime values (`Types` in `src/interpreter/ast.rs`):

| Type | Description |
|---|---|
| `Num` | 64-bit signed integer |
| `Fixed` | Q16.16 fixed-point (scaled i64) |
| `Bool`, `Str`, `Char` | as expected |
| `List` | heterogeneous list |
| `Quoted` | unevaluated expression |
| `Closure` | lambda + captured environment |
| `Share` | reference to an MPC secret share (`ShareRef`: id, bit-width, owner, fixed flag) |
| `FheCiphertext` | opaque FHE ciphertext handle (backend id + serialized bytes + fixed flag) |
| `NativeFunc` | builtin implemented in Rust |

---

## 5. Builtin reference

Builtins are ordinary bindings in the global store; a user `letrec` with the same
name shadows them.

### 5.1 MPC builtins

| Builtin | Signature | Description |
|---|---|---|
| `share` | `(share value nbits)` | Secret-share a `Num` or `Fixed` as a signed `nbits`-bit value (two's-complement). Owner is party 0. Sharing a `Fixed` sets the share's fixed flag so reconstruction and multiplication stay in Q16.16. |
| `share-from` | `(share-from owner value nbits)` | Like `share` but the input owner is explicit, so each party can contribute its own private input. |
| `reconstruct` | `(reconstruct s)` | Reveal a shared value to all parties. Returns `Num`, or `Fixed` for fixed-point shares (sign-extended). |
| `a2b` | `(a2b s nbits)` | Arithmetic→binary decomposition: returns a list of `nbits` 1-bit shares, LSB first. |
| `b2a` | `(b2a bits)` | Binary→arithmetic: recompose a list of bit shares into one arithmetic share. |

### 5.2 Vector builtins

Native replacements for the recursive `car`/`cdr` patterns; elements may be
`Num`, `Fixed`, or shares (mixed is fine — MPC multiplications are dispatched
through the backend):

| Builtin | Signature | Description |
|---|---|---|
| `list` | `(list v1 v2 …)` | Build a list from evaluated arguments (also used for `fhe-eval` ciphertext lists). |
| `dot` | `(dot vec-a vec-b)` | Inner product. |
| `vsum` | `(vsum vec)` | Sum of the elements. |
| `matvec` | `(matvec matrix vec)` | Matrix (list of row lists) × vector. |

### 5.3 Fixed-point conversions

| Builtin | Description |
|---|---|
| `(int->fixed n)` | integer → Q16.16 |
| `(fixed->int f)` | truncate toward zero → integer |
| `(fixed->str f)` | decimal string, e.g. `"3.1416"` |

### 5.4 ZKP builtins (Binius64 backend)

Circuits and proofs travel as JSON strings so they can be bound with `let`,
printed, and persisted.

| Builtin | Signature | Description |
|---|---|---|
| `zkp-compile` | `(zkp-compile 'expr '("x" "y"))` | Compile an expression to a circuit; second arg lists the private (witness) variables. Returns circuit JSON. |
| `zkp-prove` | `(zkp-prove circuit '(priv-keys) '(priv-vals) '(pub-keys) '(pub-vals))` | Produce a proof of knowledge of the private values consistent with the public ones. Returns proof JSON. |
| `zkp-verify` | `(zkp-verify circuit proof '(pub-keys) '(pub-vals))` | Verify. Returns `1` (valid) or `0`. |
| `proof-save` | `(proof-save proof filename)` | Write a proof to a JSON file. |
| `proof-load` | `(proof-load filename)` | Read a proof back; returns proof JSON. |

### 5.5 FHE builtins

| Builtin | Signature | Description |
|---|---|---|
| `fhe-keygen` | `(fhe-keygen)` | Key pair for the **mock** backend (plaintext u64 "ciphertexts" — instant, for development). Returns key JSON. |
| `fhe-keygen-tfhe` | `(fhe-keygen-tfhe)` | Key pair for the real **TFHE-rs** backend (exact 64-bit integers; keygen ≈ seconds). Installs the server key globally. |
| `fhe-keygen-ckks` | `(fhe-keygen-ckks)` | Key pair for the **SEAL CKKS** backend (approximate real arithmetic; vendored, patched `sealy` in `third_party/`). |
| `fhe-keygen-ckks-mock` | `(fhe-keygen-ckks-mock)` | Exact `f64` twin of the CKKS backend, for fast differential testing. |
| `fhe-enc` | `(fhe-enc keys value)` | Encrypt a `Num`, `Fixed`, or `Bool` under the client key. On CKKS backends a `Fixed` crosses the boundary as its true real value (`v / 2¹⁶`). |
| `fhe-dec` | `(fhe-dec keys ct)` | Decrypt; returns `Num`, or `Fixed` if the ciphertext is flagged fixed-point. |
| `fhe-compile` | `(fhe-compile 'expr '("enc-var"))` | Compile to an FHE circuit (integer mode, Q16.16 gadgets for fixed-point). Second arg names variables supplied as ciphertexts. Returns circuit JSON. |
| `fhe-compile-ckks` | `(fhe-compile-ckks 'expr '("enc-var"))` | Same, in **real-arithmetic mode**: fixed literals keep their true value, no Q16.16 gadgets. Required for CKKS backends. |
| `fhe-eval` | `(fhe-eval circuit keys '(pub-keys) '(pub-vals) '(enc-keys) '(enc-cts))` | Homomorphically evaluate the circuit. Public inputs are plaintext; encrypted inputs are ciphertexts (build the list with `(list ct1 ct2 …)`, or `'()` for none). Returns the encrypted result. |

The key JSON's `backend_id` selects the backend at `fhe-enc`/`fhe-dec`/`fhe-eval`
time, so the same program text runs on mock, TFHE, or CKKS by swapping the keygen
call.

---

## 6. MPC layer

### 6.1 Secret sharing

Every secret `x` is split into three additive shares over a 128-bit ring:
`s₀ + s₁ + s₂ = x (mod 2¹²⁸)`, with party *i* holding the replicated pair
`(sᵢ, sᵢ₊₁)` (ABY3 style). No single party learns anything from its own pair.
Signed values use two's-complement in the declared bit-width: `(share -3 8)`
shares `253`.

### 6.2 Cost of each operation

| Operation | Protocol | Communication |
|---|---|---|
| `+`, `-`, scalar × | local on shares | none |
| `*` | Beaver triple (dealer: party 0) | 1 broadcast round |
| `a2b` | masked open + ripple-carry adder | 1 broadcast round |
| `<`, `>`, `<=`, `>=`, `==` | `a2b(a−b)`, sign bit / zero test | 1 broadcast round |
| `reconstruct` | each party sends its share | 1 round |

Comparisons are **signed** (two's-complement MSB). Fixed-point share
multiplication truncates by 16 bits to stay in Q16.16.

### 6.3 Branching on secrets

`(if (> z 0) z 0)` on shares reveals the condition bit (one round), then takes
the branch — this **leaks which branch ran**. For fully oblivious code, write a
mux instead:

```lisp
(* z (> z 0))                      ; oblivious ReLU
(+ (* c a) (* (- 1 c) b))          ; oblivious select(c, a, b)
```

### 6.4 Interpreter backends

The evaluator dispatches all share operations through the `Backend` trait
(`src/interpreter/backend.rs`):

| Backend | Description |
|---|---|
| `SimBackend` | Single-process simulation — the REPL default. |
| `Aby3Backend` | In-process replicated 3-party ABY3 (`src/interpreter/aby3_backend.rs`). |
| `TransportBackend<T>` | Real distributed protocol over a `Transport` (in-process queues or TCP). |
| `GCBackend` | Experimental 2-party garbled-circuits backend (`src/interpreter/gc_backend.rs`); library-level, not wired into a binary. |

### 6.5 Multi-party binary

```sh
cargo build --bin tcp_party

./target/debug/tcp_party        # no args: in-process demo, 3 threaded parties

# TCP mode — three terminals; party 0 reads expressions from stdin
./target/debug/tcp_party 0 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
./target/debug/tcp_party 1 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
./target/debug/tcp_party 2 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
```

All parties evaluate the same expression in lockstep; `share-from` /
`:sintN-from-P` let each party contribute its own private input.

---

## 7. FHE layer

Pipeline (mirrors ZKP): Lisp AST → `compile_fhe` (shared FlatIR pipeline, SSA,
optimised) → `FheCircuit` → `evaluate_fhe` per-gate on an `FheBackend`.

| Backend id | Kind | Arithmetic | Notes |
|---|---|---|---|
| mock | `FheBackend` | exact u64 | plaintext stand-in; instant; use for development and differential tests |
| `tfhe-rs-v1.6` | `FheBackend` | exact `FheUint64` | real FHE; supports comparisons (so ReLU works); slow (minutes for large circuits) |
| `seal-ckks` | `RealFheBackend` | approximate `f64` (N=8192, 128-bit security) | only `+ − ×` — **no comparisons**, so no encrypted ReLU/argmax; errors ~1e-8; supports SIMD batching and plaintext-weight dot products with rotation keys (Rust API: `encrypt_vec`, `dot_plain`) |
| `ckks-mock` | `RealFheBackend` | exact `f64` | drop-in twin of `seal-ckks` for fast testing |

Compile with `fhe-compile` for integer backends and `fhe-compile-ckks` for the
CKKS pair. Fixed-point values are Q16.16 gadgets on integer backends and true
reals on CKKS.

MNIST benchmark data point (784-input linear layer, 2365 gates): TFHE eval
≈ 2800 s; CKKS circuit form ≈ 1.6 s; CKKS batched SIMD form ≈ 92 ms with one
393 KB ciphertext per image (see `examples/mnist.rs`).

### 7.1 Delegated computation (the FHE job market)

FHE's headline use is **delegation**: a data owner hands a heavy computation to
an untrusted worker who runs it *on ciphertexts* and never sees the plaintext.
`src/fhe/market.rs` models this as a verifiable job, and `src/fhe/lifecycle.rs`
wraps it in economics.

Three roles, none of which requires trusting the others:

- **Requester** holds the client key. They compile the function, encrypt their
  private inputs, and post an [`FheJob`] carrying only the circuit, the public
  inputs, and the ciphertext bytes — **the client key never leaves them**.
- **Worker** runs `evaluate_job`, producing the result ciphertext plus a
  succinct `WorkReceipt` (a commitment to the output). It needs only the
  evaluation/server key and never decrypts.
- **Verifier** (anyone) calls `verify_receipt`, which **re-executes the circuit
  on the ciphertexts** and compares the output commitment. Homomorphic
  evaluation is a deterministic function of `(circuit, inputs, keys)`, so a
  wrong result yields a different commitment and is *provable* fraud — the
  verifier never decrypts either.

```rust
use std::collections::HashMap;
use baru::fhe::market::{evaluate_job, verify_receipt, FheJob, Verdict};
use baru::fhe::{FheBackend, MockFheBackend};
use baru::interpreter::parse;
use baru::zkp::circuit_compiler::CircuitCompiler;

// ── Requester: compile f(x,y) = x + y, encrypt private inputs, post a job ──
let backend = MockFheBackend;              // swap for the TFHE backend in production
let (client_key, _server_key) = backend.keygen();

let mut cc = CircuitCompiler::new();
let mut circuit = cc.compile_fhe(&parse("(+ x y)").unwrap()).unwrap();
circuit.public_inputs.retain(|v| v != "x" && v != "y");
circuit.encrypted_inputs = vec!["x".into(), "y".into()];

let mut encrypted_inputs = HashMap::new();
encrypted_inputs.insert("x".into(), backend.serialize_ct(&backend.encrypt(&client_key, 10)));
encrypted_inputs.insert("y".into(), backend.serialize_ct(&backend.encrypt(&client_key, 32)));

let job = FheJob {
    circuit,
    public_inputs: HashMap::new(),
    encrypted_inputs,                      // ciphertext bytes only — no client key
    backend_id: backend.backend_id().to_string(),
};

// ── Worker: evaluate homomorphically; never decrypts ──
let (output, receipt) = evaluate_job(&job, &backend).unwrap();

// ── Anyone: verify by deterministic re-execution (no key, no decryption) ──
assert_eq!(verify_receipt(&job, &receipt, &backend), Verdict::Valid);

// ── Requester: decrypt the delivered result with the client key ──
let out_ct = backend.deserialize_ct(&output.bytes);
assert_eq!(backend.decrypt(&client_key, &out_ct), 42);
```

**Economic wrapper.** `fhe::lifecycle` turns that objective correctness check
into an incentive-compatible market. A job moves
`Open → Claimed → Delivered → {Settled | Challenged → Settled}`: the reward is
escrowed, the worker posts a claim bond, and a challenge window lets anyone
dispute a delivered result. A dispute is adjudicated by the same re-execution —
`verdict_to_adjudication` maps a `Verdict` to the state machine's `Adjudicated`
event — and settlement emits `Effect`s (stake releases/slashes and reputation
deltas) that a consensus ledger folds at each epoch boundary. Those reputation
deltas are exactly what `tcp_party`'s `BARU_POUW` consensus weights leader
election by, so honest delegated work compounds into consensus influence.

---

## 8. ZKP layer

The `zkp-compile`/`zkp-prove`/`zkp-verify` builtins target the **Binius64**
SNARK (`binius-core`/`binius-field`). The circuit compiler
(`src/zkp/circuit_compiler.rs`) supports `+ − ×`, exact `/` and `mod` (via
non-deterministic quotient witnesses), all comparisons, `and`/`or`/`not`,
shifts, `let`/`letrec`, lambdas, and compile-time lists (`car`/`cdr`/`cons` on
constant lists — loops are unrolled at compile time). `(private e)` marks a
subexpression as a witness. Proofs serialize to JSON (`proof-save` /
`proof-load`).

---

## 9. REPL

`cargo run --bin baru`. Expressions are evaluated on three simulated
participants. Colon-commands:

| Command | Description |
|---|---|
| `:help` | language reference |
| `:examples [category]`, `:example <n>` | browse / run the built-in example library (arithmetic, comparisons, booleans, let-lambda, higher-order, lists, recursion, MPC, ZKP, FHE, …) |
| `:env` | print the participant environment |
| `:verbose on\|off` | trace evaluation |
| `:last`, `:results` | inspect previous results |
| `:reveal <pid>` | reveal the last shared result to a participant |
| `:evalfor <pid> EXPR` | evaluate as a specific participant |
| `:inspect <pid> <var>` | inspect a binding |
| `:save <pid> [file]`, `:load <file>` | persist / restore an environment (JSON) |
| `:reset` | clear user bindings, re-register builtins |
| `quit` | exit |

---

## 10. Benchmark harness

```sh
cargo run --bin baru bench -- [--list] [--backends plain,mpc-aby3,fhe-mock,fhe-tfhe,zkp-binius64]
                              [--iters N] [--expr EXPR] [--mpc-expr EXPR] [--input name=val ...]
```

Runs each case of a built-in suite (`add-mul`, `relu`, dot products, …) on every
selected backend, checks the decrypted/reconstructed result against the plain
interpreter, and reports per-phase timings (compile / keygen / encrypt / eval /
prove / verify). Cases that branch on secrets carry a mux-style `mpc_expr`
rewrite.

---

## 11. Worked examples

### Private ReLU under MPC

```lisp
(let ((w :sint8 -1) (x :sint8 3) (b :sint8 0))
  (let ((z (+ (* w x) b)))
    (reconstruct (if (> z 0) z 0))))      ; => 0
```

### Millionaires' problem — inputs from different parties

```lisp
(let ((a :sint32-from-0 1000000)
      (b :sint32-from-1 2000000))
  (reconstruct (> a b)))                  ; => 0  (party 1 is richer)
```

### Fixed-point dot product on shares

```lisp
(let ((w (list (share 0.5 64) (share -1.25 64)))
      (x (list (share 2.0 64) (share 4.0 64))))
  (fixed->str (reconstruct (dot w x))))   ; => "-4.0000"
```

Fixed×fixed share products need headroom before the Q16.16 rescale: use 64-bit
share widths (a 32-bit share wraps — `0.5 × 2.0` scales to `2³²` ≡ 0). Mixed
integer-share × fixed-constant products are exact at any width.

### Zero-knowledge proof: "I know x with x·(x+1) = 12"

```lisp
(let ((c (zkp-compile '(* x (+ x 1)) '("x"))))
  (let ((p (zkp-prove c '("x") '(3) '() '())))
    (zkp-verify c p '() '())))            ; => 1
```

### FHE: encrypted input, public weight

```lisp
(let ((keys (fhe-keygen)))                            ; swap for fhe-keygen-tfhe
  (let ((ct (fhe-enc keys 5))
        (circ (fhe-compile '(+ (* y 3) 4) '("y"))))
    (fhe-dec keys
      (fhe-eval circ keys '() '() '("y") (list ct))))) ; => 19
```

### CKKS: approximate reals

```lisp
(let ((keys (fhe-keygen-ckks)))                        ; or fhe-keygen-ckks-mock
  (let ((ct (fhe-enc keys 1.5))
        (circ (fhe-compile-ckks '(* y 2.0) '("y"))))
    (fhe-dec keys
      (fhe-eval circ keys '() '() '("y") (list ct))))) ; => 3.0000
```

### MNIST private inference

`cargo run --release --example mnist [--mpc] [--tfhe] [--ckks]` — a linear
discriminator over 784 encrypted pixels with Q16.16 public weights, run on
plain / fhe-mock / MPC / TFHE / CKKS (scalar and batched SIMD), with all
backends checked against the plain result.

---

## 12. Architecture map

```
src/
  interpreter/
    parser.rs            tokenizer + parser; quote shorthand, :sintN sugar
    ast.rs               Expr / Types, Q16.16 constants
    eval.rs              trampolined evaluator, Participant, all builtins
    backend.rs           Backend trait, ShareRef, SimBackend
    aby3_backend.rs      in-process replicated ABY3
    gc_backend.rs        2-party garbled circuits (experimental)
    transport_backend.rs distributed backend over a Transport
  crypto/                field arithmetic, Beaver triples, bit protocols,
                         InprocHandle/TcpTransport, scheduler
  fhe/                   FheBackend/RealFheBackend traits, mock, TFHE-rs,
                         SEAL CKKS (+ mock twin), circuit emitter
  zkp/                   CircuitCompiler (FlatIR), Binius64 backend, proof
                         serialization
  bench.rs               `baru bench` harness
  repl.rs                REPL commands + example library
  bin/tcp_party.rs       multi-party binary
third_party/sealy/       vendored, patched SEAL bindings (see BARU_PATCHES.md)
```

## 13. Known limitations

- **Branch leakage**: `if` on a shared condition reveals the branch taken; use
  mux-style expressions for oblivious code.
- **Trusted dealer, semi-honest model**: Beaver triples come from party 0; no
  malicious-security authentication.
- **CKKS**: approximate results, `+ − ×` only (no encrypted comparisons).
- **Ring size**: 128-bit ring wraps on overflow.
- **Single-evaluator model**: all parties run the same expression text.
