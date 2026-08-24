# The Baru Language

Baru is a small Lisp-like language for **privacy-preserving computation**. A single
program can be evaluated on four different backends — and on **combinations** of
them, since values can cross between MPC shares and FHE ciphertexts
mid-computation without being exposed:

| Backend | What it provides | Where |
|---|---|---|
| **Plain interpreter** | Ordinary evaluation (CEK-style trampolined evaluator) — the correctness baseline | `src/interpreter/` |
| **MPC** | 3-party secure computation with ABY3-style replicated secret sharing; triple-free multiplication, batched bit protocols, garbled-circuit nonlinearities | `src/crypto/`, `src/interpreter/*_backend.rs` |
| **FHE** | Fully homomorphic evaluation of compiled circuits — mock, TFHE-rs (exact integers), and an exact real-arithmetic backend | `src/fhe/` |
| **ZKP** | Zero-knowledge proofs of circuit satisfaction using the Binius64 SNARK | `src/zkp/` |
| **Hybrid** | Scheme switching in both directions (`share->fhe`, `fhe->share`), and re-encryption as an alternative to bootstrapping | `src/hybrid/` |

This document describes the language surface: syntax, data types, special forms,
every builtin, the REPL, the CLI tools, and worked examples for each backend. For
what the layers *cost*, measured, see
[`docs/experiments.md`](docs/experiments.md); for the repository overview see
[`README.md`](README.md).

---

## 1. Getting started

```sh
cargo build                 # build everything
cargo test                  # unit + integration tests
cargo run --bin baru        # single-process REPL (in-process ABY3, no networking)
cargo run --bin baru bench  # benchmark harness (see §10)
cargo run --release --example mnist   # MNIST private-inference demo (see §11)
```

Multi-party TCP mode uses the `tcp_party` binary (see §6.5). One optional cargo
feature: `pq-tls` (post-quantum TLS transports). Everything else is in the
default build, which is pure Rust and needs no C++ toolchain.

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

```lisp
(let* ((x 2) (y (* x 3))) (+ x y))       ; => 8; each init sees the previous
(let loop ((i 0) (acc 0))                ; named let — the standard Scheme loop
  (if (== i 5) acc (loop (+ i 1) (+ acc i))))   ; => 10

(define x 5)                             ; bind in the current environment
(define (sq n) (* n n))                  ; sugar for (define sq (lambda (n) ...))
(set! x 42)                              ; assign to an EXISTING binding

(lambda (a . rest) (length rest))        ; variadic: rest collects the remainder
(lambda (. args) (length args))          ; every argument in one list
```

With a rest parameter the fixed parameters become a **minimum** rather than an
exact arity, and the rest list is empty (not absent) when only the fixed
arguments are supplied. Scheme spells the take-everything form `(lambda args …)`;
here a bare identifier in that position already means *one required parameter*,
so the dotted `(lambda (. args) …)` is the supported spelling — see §13.

`define` introduces a binding; `set!` assigns to one that already exists and
errors if it does not. Both return the value (this language has no "unspecified").

**`set!` writes through a shared cell**, so a closure sees assignments made after
it was created — the usual counter works:

```lisp
(let ((n 0))
  (let ((bump (lambda () (set! n (+ n 1)))))
    (begin (bump) (bump) (bump) n)))     ; => 3
```

Neither form can be compiled to a circuit: an FHE or ZKP circuit has no
environment to extend and is single-assignment. Both are refused with an error
naming the reason rather than mis-compiled.

Currying, higher-order functions, closures capturing their environment, and
mutual recursion (`even?`/`odd?` style) all work; the evaluator is trampolined so
deep recursion (Ackermann, Collatz, …) does not overflow the Rust stack.

**Tail calls are proper.** A closure body is evaluated with the *caller's*
continuation rather than a new frame, so a named-`let` loop runs in constant
continuation space. The practical ceiling is `MAX_FRAMES` (a step budget on the
trampoline, currently 1 000 000 ≈ 50 000 loop iterations), not the recursion
depth.

> **`let` is sequential, unlike Scheme's.** In Baru, `(let ((a 1) (b (+ a 10))) b)`
> is `11`: a later init sees an earlier binding. Scheme's `let` evaluates inits in
> the *outer* scope and would reject that (or see an outer `a`); the sequential
> behaviour is Scheme's `let*`. **`let*` is therefore semantically redundant
> here** and is provided so Scheme source ports unchanged and so intent is
> explicit. See §13.

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
(begin e1 e2 ... en)                     ; evaluate in order, return the last
(when   cond body...)                    ; body if truthy, else 0
(unless cond body...)                    ; body if falsy,  else 0
(cond (test1 body...)                    ; first matching clause wins
      (test2 body...)
      (else  body...))                   ; optional; must come last
(case key ((1 2) body...)                ; key evaluated once, compared with ==
          ((3)   body...)
          (else  body...))
(do ((i 0 (+ i 1))                       ; iteration: (var init [step])
     (acc 0 (+ acc i)))
    ((== i 5) acc)                       ; (test result...) — result may be empty
  body...)                               ; body runs each pass, for effect
```

Any nonzero number / `#t` is truthy. When `cond` is a **shared** value the
condition bit is revealed first — see §6.3.

### Conditions and `guard`

```lisp
(guard (e handler...)                    ; e is bound to the condition
  body...)                               ; if body raises, handler runs instead
(error "message" irritant...)            ; signal a condition
(raise obj)                              ; raise any object
```

```lisp
(guard (e (error-message e)) (error "boom"))   ; => "boom"
(guard (e 0) (/ 1 0))                          ; => 0
(guard (e 99) some-unbound-name)               ; => 99
```

**Errors propagate through an error continuation.** A raised condition never
reaches the continuation it was handed to: the machine truncates its task stack
back to the height recorded when the innermost `guard` was installed, then runs
that handler. Unwinding is therefore a single truncate, not a walk over pending
frames, and everything the guarded expression had left to do is discarded.

Three consequences worth knowing:

- **An uncaught condition is the value of the evaluation**, not a crash. A REPL
  session survives a bad expression and carries on to the next one.
- **The condition is bound inert** inside the handler, so `e` can be inspected
  and passed around. A handler may `raise` it again, and it then propagates to
  the *enclosing* guard.
- **A handler is uninstalled when its body exits normally**, so it cannot catch
  something raised later in the program.

| Builtin | Description |
|---|---|
| `error` | `(error "msg" irritant...)` — build and signal a condition. |
| `raise` | Re-raise a caught condition, or wrap any other object. |
| `error?` | Is this value a condition? |
| `error-message` / `error-irritants` | Accessors on a caught condition. |

**Coverage, stated precisely.** Catchable by `guard`:

- explicit `error` / `raise`;
- unbound variables, and `set!` on an unbound name;
- arity mismatches (including the zero-argument case), and calling a
  non-procedure or a share as one;
- division and modulo by zero, non-exact division;
- type errors in `car`, `cdr`, `cons`, `if`, `not`, `null?`, `bitlen`,
  negation and the arithmetic shifts — each carrying the offending value as an
  irritant;
- every failure in the procedures of §5.2–§5.2b;
- **every MPC, FHE and ZKP builtin failure** — bad arguments, malformed JSON, a
  missing model or proof file, a wrong quantisation scale.

- length and type errors in the vector builtins `dot`, `vsum`, `matvec` and
  `nn-forward`;
- secret-dependent failures: a branch under a shared condition that cannot
  produce a share (`(if <share> "yes" "no")`), and operations with no
  fixed-point-share implementation such as `(mod <share> 2.5)`.

Still a panic, so **not** catchable — 17 sites, each deliberate:

- the `MAX_FRAMES` step budget, and one internal invariant that would indicate a
  bug in the evaluator rather than in the program;
- six type invariants guarded by an outer match that has already established the
  type, so no program can reach them (verified: `and`/`or` on shares, which
  looked like the likely route, dispatch correctly);
- the `eval_plain` / `eval_shared` / `eval_fhe` test harness, where a panic is
  the point: a test should fail loudly rather than return a condition a
  careless assertion might ignore.

`guard` cannot be compiled to a circuit: a circuit has no control flow to unwind,
so there is nothing for a handler to intercept.

A `cond` or `case` with no matching clause and no `else` evaluates to `0`, the
same "no value" an untaken `when` produces.

> **These are derived forms.** `begin`, `when`, `unless`, `cond`, `case`, `do`,
> `let*` and named `let` are desugared **in the parser** into `if` / `let` /
> `letrec` / sequences. Nothing in the evaluator or the FHE/ZKP circuit compilers knows they
> exist, which is why they work on every backend the core forms work on — a ReLU
> written `(cond ((> x 0) x) (else 0))` compiles, proves and verifies under
> Binius64 exactly like the `if` it lowers to
> (`tests/zkp_real.rs::derived_forms_compile_prove_and_verify`).

### Operators

| Form | Meaning |
|---|---|
| `(+ a b)` `(- a b)` `(* a b)` | arithmetic (wrapping on i64; works on `Num`, `Fixed`, and shares) |
| `(+ a b c …)` `(* a b c …)` `(- a)` `(- a b c)` `(/ a b c)` | **non-binary arities** route to the operator's *procedure* binding: variadic, but **public values only** |
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

**Operators are also values.** `+`, `-`, `*`, `/`, `mod` and the comparisons are
bound as ordinary procedures, so they can be applied or passed around:

```lisp
(- 5)                     ; => -5   (unary negation)
(+ 1 2 3)                 ; => 6
(+)                       ; => 0    (Scheme's additive identity)
(apply * (list 2 3 4))    ; => 24
```

Two operands take the **special-form** path, which is the only one that
dispatches secret shares and FHE ciphertexts through the backend. Any other
arity — and any use of an operator as a *value* — takes the procedure path,
which handles `Num` and `Fixed` but **refuses secrets** with an error naming the
two-operand form. Duplicating the MPC dispatch into a second code path is the
kind of drift this repository is built to measure, so it is not done.

`apply` is likewise a **binding, not syntax**, so `(let ((apply …)) …)` shadows
it exactly as in Scheme.

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
| `length` | `(length lst)` | Number of elements. |
| `append` | `(append lst …)` | Concatenate lists. |
| `reverse` | `(reverse lst)` | Reversed copy. |
| `list-ref` | `(list-ref lst i)` | Element at **public** index `i`. A *secret* index is refused with an error rather than silently scanned — see the note below. |

### 5.2a Vectors

Fixed-length, mutable, with **reference semantics** — the counterpart to this
language's immutable lists. Indices are public, as for `list-ref`.

| Builtin | Description |
|---|---|
| `vector` | `(vector v1 v2 …)` — build from arguments. |
| `make-vector` | `(make-vector n [fill])` — `n` slots, `fill` defaults to `0`. |
| `vector-length` | Number of slots. |
| `vector-ref` | `(vector-ref v i)` at a **public** index; bounds-checked. |
| `vector-set!` | `(vector-set! v i x)` — mutates in place, returns `x`. |
| `vector->list` / `list->vector` | Convert between the two. |
| `vector?` | Type test. |

`vector-set!` is the reason vectors exist: without mutation a vector would be a
list with O(1) indexing, which `list-ref` already provides. Because the storage
is shared, a mutation is visible through every alias and through any procedure
the vector was passed to:

```lisp
(let ((v (vector 1 2)))
  (let ((f (lambda (x) (vector-set! x 1 77))))
    (begin (f v) (vector-ref v 1))))     ; => 77
```

### 5.2b Predicates and equivalence

| Builtin | Description |
|---|---|
| `number?` `integer?` `boolean?` `string?` `char?` `vector?` `error?` | Type tests on public values. |
| `list?` `pair?` | `pair?` is a *non-empty* list; there are no improper pairs (§13). |
| `procedure?` | True for closures and builtins. |
| `share?` `ciphertext?` | Is this value a secret share / an FHE ciphertext? |
| `zero?` | `(zero? x)` on a `Num` or `Fixed`. |
| `eq?` `eqv?` | Identity/atom equality. Returns `1`/`0`. |
| `equal?` | Structural equality, recursing into lists. |

All predicates return `1`/`0` rather than `#t`/`#f`, so they compose with the
existing numeric comparisons and with `and`/`or`'s bitwise behaviour.

Three deliberate boundaries, because each is a place where a convenience could
have quietly cost privacy:

- **`share?` is safe and `eq?` on shares is cheap.** A value's *type* is public —
  only its contents are secret — so asking whether something is a share leaks
  nothing. `eq?` on two shares compares **handle identity**, never contents:
  comparing secrets by value needs a protocol round, and spending one inside a
  predicate would hide a real cost. Use `(== a b)`, which is visibly a protocol
  operation.
- **`list-ref` requires a public index.** A secret index needs a linear scan of
  muxes over the whole list (or ORAM). That is a protocol decision with a cost,
  not something an accessor should pick silently, so it is an error.
- **These builtins do not exist on the circuit backends.** They are
  `NativeFunc`s, which have no gate-level meaning; a program compiled to FHE or
  ZKP has already resolved its public structure.

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
| `fhe-keygen-ckks-mock` | `(fhe-keygen-ckks-mock)` | Key pair for the **exact real-arithmetic** backend (plain `f64`, no keys, no noise). Named for the SEAL CKKS backend it used to mirror, removed 2026-08-03. |
| `fhe-enc` | `(fhe-enc keys value)` | Encrypt a `Num`, `Fixed`, or `Bool` under the client key. On CKKS backends a `Fixed` crosses the boundary as its true real value (`v / 2¹⁶`). |
| `fhe-dec` | `(fhe-dec keys ct)` | Decrypt; returns `Num`, or `Fixed` if the ciphertext is flagged fixed-point. |
| `fhe-compile` | `(fhe-compile 'expr '("enc-var"))` | Compile to an FHE circuit (integer mode, Q16.16 gadgets for fixed-point). Second arg names variables supplied as ciphertexts. Returns circuit JSON. |
| `fhe-compile-ckks` | `(fhe-compile-ckks 'expr '("enc-var"))` | Same, in **real-arithmetic mode**: fixed literals keep their true value, no Q16.16 gadgets. Required for CKKS backends. |
| `fhe-eval` | `(fhe-eval circuit keys '(pub-keys) '(pub-vals) '(enc-keys) '(enc-cts))` | Homomorphically evaluate the circuit. Public inputs are plaintext; encrypted inputs are ciphertexts (build the list with `(list ct1 ct2 …)`, or `'()` for none). Returns the encrypted result. |

The key JSON's `backend_id` selects the backend at `fhe-enc`/`fhe-dec`/`fhe-eval`
time, so the same program text runs on mock, TFHE, or the real-arithmetic backend by swapping the keygen
call.

### 5.6 Scheme-switching builtins

The two directions of `src/hybrid/` (see §7.1), exposed as builtins so a single
program can move a value between representations without revealing it:

| Builtin | Signature | Description |
|---|---|---|
| `share->fhe` | `(share->fhe keys s)` | Convert an MPC share into an FHE ciphertext. Each party encrypts its own additive component under the public key and the evaluator sums them homomorphically — the value is never reconstructed. Returns a ciphertext. |
| `fhe->share` | `(fhe->share keys ct nbits)` | Convert an FHE ciphertext into an MPC share by mask-and-open: only a uniformly-masked value is ever decrypted. Returns a `Share` of the given bit-width. |

A converted ciphertext holds the value's *residue class* mod `2^nbits` (share-sum
carries may sit above it), so `+`/`−`/`×` keep working but an FHE-side comparison
on a converted value would see the carry. Convert back to MPC for comparisons —
which is the point of switching in the first place.

### 5.7 Model builtins

| Builtin | Signature | Description |
|---|---|---|
| `nn-load-model` | `(nn-load-model "model.quant.json")` | Load a quantised `NetworkSpec` (produced by `baru quantize`, §10.2) as ordinary Lisp data: a list of `(weights bias activation)` triples, inspectable like any other value rather than an opaque handle. Requires `--scale-bits 16`. |
| `nn-forward` | `(nn-forward layers input)` | Full multi-layer forward pass: matvec + bias-add + activation, chained layer to layer, reusing the `matvec` precompile. Works identically whether `input` is public (`Num`/`Fixed`) or a private `Share` vector, and whether the model is public or shared — each helper dispatches per element on the runtime type, so there is no separate "private mode". Only `relu` and `identity` activations are supported. |

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
| `*` | ABY3 resharing — **no Beaver triples, no dealer**. Each party forms three of the nine cross terms locally, re-randomises with zero-sum PRF randomness, and sends the result to one neighbour. | 1 ring element to 1 peer, 1 round |
| `a2b` | binary-share the three additive components (free), carry-save reduce 3→2, then a Kogge–Stone parallel-prefix adder | `2 + 1 + ⌈log₂ nbits⌉ + 1` ≈ **9 rounds** at 20 bits |
| `<`, `>`, `<=`, `>=`, `==` | `a2b(a−b)`, sign bit / zero test | as `a2b` |
| `reconstruct` | each party sends its component | 1 round |

Comparisons are **signed** (two's-complement MSB). Fixed-point share
multiplication truncates by 16 bits to stay in Q16.16.

Two properties are worth knowing before you tune anything:

- **There is no offline phase to optimise.** Multiplication needs no precomputed
  triples, because the correlated randomness is regenerated locally per operation
  from pairwise PRF keys. The only preprocessing is a 32-byte key exchange per
  party pair per session.
- **Cost is rounds, not operations.** The backend exposes batched forms
  (`mul_batch`, `a2b_batch`, `lt_batch`, `share_secret_batch`) that run *any*
  number of independent lanes in the round count of one, so a wide layer costs
  what a single element costs. Rust code that calls the per-element form in a
  loop pays `O(n)` rounds for nothing — measured at 9–45× wall-clock on a 64-wide
  layer. `src/gadgets/` and `src/precompile/` are written over the batched
  primitives; the `matvec` builtin dispatches to a batched precompile. An earlier
  `a2b` that appeared to cost 1 round was leaking its input — see
  [`docs/experiments.md` §I.2](docs/experiments.md#i2-a2b-rebuilt-after-a-verified-privacy-break).

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
| `Aby3Simulator` | In-process *simulation* of replicated 3-party ABY3 (`src/interpreter/aby3_simulator.rs`) — what `Participant::new_simulated` and the REPL use. Correct on values, but **not the distributed protocol**: all three parties' shares live in one process, so its share multiplication reconstructs and reshares locally — free, and revealing everything. Never quote its timings as protocol costs; it understated one measured layer by 2.7×. (It was named `Aby3Backend` until 2026-07-30, which read as a peer of `TransportBackend` rather than a simulation of it.) |
| `TransportBackend<T>` | The real distributed protocol over a `Transport` (in-process queues or TCP). **The only backend whose costs mean anything**, and the only one the benchmarks use. It needs all three parties running concurrently — a lone party has no peers and blocks — so it is always constructed inside a three-thread or three-process spawn. |

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
| `ckks-mock` | `RealFheBackend` | exact `f64` | the real-arithmetic backend; only `+ − ×`, since comparison has no meaning on reals here |

Compile with `fhe-compile` for integer backends and `fhe-compile-ckks` for the
real-arithmetic one. Fixed-point values are Q16.16 gadgets on integer backends
and true reals under real-arithmetic mode.

MNIST benchmark data point (784-input linear layer): TFHE eval ≈ 2800 s. The
SEAL CKKS rows that sat here — ≈ 1.6 s in circuit form, ≈ 92 ms batched with one
393 KB ciphertext — went with that backend on 2026-08-03 and are no longer
reproducible; `docs/experiments.md` §II.4 keeps them
per image (see `examples/mnist.rs`). Those were measured at **2365 gates**;
common-subexpression elimination inside `CircuitCompiler` has since brought the
same circuit to **1764 gates (−25%)** with bit-identical scores, so the integer
backends should be proportionally faster — at TFHE's ~0.2–0.5 s/gate those 601
gates are minutes per run.

### 7.1 Scheme switching: MPC <-> FHE

MPC and FHE have complementary weaknesses: MPC does comparisons in milliseconds
but pays communication for every multiplication, while FHE does a whole linear
layer in one ciphertext operation but needs *seconds* for an approximate
comparison. `src/hybrid/` converts values between the two representations
mid-computation, so each phase runs where it is cheapest. From Baru that is two
builtins (§5.6):

```lisp
; a secret computed under MPC crosses into FHE without being revealed
(let ((keys (fhe-keygen)) (x (share 6 32)) (y (share -7 32)))
  (fhe-dec keys (share->fhe keys (* x y))))          ; => -42

; a value under FHE needs a ReLU (FHE's weak spot): switch to MPC for it
(let ((keys (fhe-keygen)))
  (let ((z (fhe->share keys (fhe-enc keys -4) 32)))
    (reconstruct (* z (> z 0)))))                    ; => 0
```

Composing the two directions gives **re-encryption instead of bootstrapping**
(`hybrid::mpc_reencrypt`): the round trip resets FHE noise and, for levelled
a levelled scheme, restores the full depth budget, for the price of one
interactive round.
That is the mechanism behind the repository's headline result — a network of
arbitrary depth evaluated with zero FHE bootstraps, since the scheme switch that
computes each ReLU also refreshes the ciphertext. See
[`docs/experiments.md` Part III](docs/experiments.md#part-iii--the-hybrid-construction).

Scope, stated in the module too: the FHE→MPC direction stands a single client key
in for a threshold decryption (the data flow is already the right one; a real
deployment needs distributed-key threshold FHE), and everything is semi-honest.

The Rust-level FHE schemes that make this cheap are not exposed as builtins,
because they implement one specialized primitive rather than a general circuit
evaluator: `fhe::bfv` and `fhe::bgv` do a **depth-1 packed matvec** — one
ciphertext×ciphertext multiplication, no relinearization keys, no bootstrapping,
decrypting the degree-2 product under `s²`. Drive them from Rust (`src/bench/`,
`src/gadgets/`) or through the `baru bench-*` commands.

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
participants. Colon-commands (`:help` prints the live list, which is shorter in
distributed mode — see §6.5):

| Command | Description |
|---|---|
| `:help` | show the command list |
| `:examples [category]`, `:example <n>` | browse / run the built-in example library (arithmetic, comparisons, booleans, let-lambda, higher-order, lists, recursion, MPC, ZKP, FHE, …) |
| `:verbose [on\|off]` | toggle per-participant eval traces |
| `:last [pid]`, `:results` | last result for one participant / for all of them |
| `:reveal <pid>` | reveal the last shared result for a participant |
| `:evalfor <pid> EXPR` | evaluate `EXPR` as a specific participant |
| `:inspect <pid> <var>` | reveal a variable for a participant |
| `:save <pid> [file]`, `:load <pid> [file]` | persist / restore a participant store (JSON) |
| `:reset [pid]` | clear participant store(s) |
| `quit` / `exit` | exit the REPL |

In distributed mode party 0 has `:help`, `:examples`, `:example`, `:verbose` and
`:quit` (which signals every party to exit), plus the node commands in
[README](README.md#running-a-multi-party-node) (`:commit`, `:chat`, `:post`).

---

## 10. Command-line tools

Everything below is a subcommand of the `baru` binary. `--help` on any of them
prints its own options.

### 10.1 `baru bench` — the cross-backend suite

```sh
cargo run --bin baru bench -- [--list] [--backends plain,mpc-aby3,fhe-mock,fhe-tfhe,zkp-binius64]
                              [--iters N] [--expr EXPR] [--mpc-expr EXPR] [--input name=val ...]
```

Runs each case of a built-in suite (`add-mul`, `relu`, dot products, …) on every
selected backend, checks the decrypted/reconstructed result against the plain
interpreter, and reports per-phase timings (compile / keygen / encrypt / eval /
prove / verify). Cases that branch on secrets carry a mux-style `mpc_expr`
rewrite.

The size-swept and application-level benchmarks are separate subcommands
(`bench-nn`, `bench-hybrid-nn`, `bench-deep-nn`, `bench-cnn`, `bench-mnist`,
`bench-medmnist`, `bench-train`, `bench-fhe-innerprod`, `bench-polymul`,
`bench-triples`, `bench-ot`, `bench-psi`, `bench-layer`,
`bench-net`, `bench-circuit`, `bench-backend-aware`, `bench-compiled-nn`,
`bench-abstraction`, `bench-gc-compare`, `bench-mixed-layer`, `bench-cohort`,
`bench-cohort-eval`). They drive the Rust APIs directly
rather than the interpreter, for cost attribution and to reach realistic sizes;
most accept `--json`, `--rtt-ms` and `--bandwidth-mbps`. What each one measures is
listed in the [README](README.md#command-line-surface) and what it *found* is in
[`docs/experiments.md`](docs/experiments.md).

### 10.2 `baru quantize` — model import

```sh
baru quantize --in model.json --out model.quant.json --scale-bits 16 --validate samples.json
```

Train a model in whatever framework you like, export weights/biases to a plain
JSON `NetworkSpec`, then quantize to Q16.16 fixed-point. `--validate` compares the
float and quantized forward passes on real samples and reports max/mean absolute
and max relative error — "good enough for inference" is a number you check, not
an assumption. It is intentionally not optional-and-silent: skipping it prints a
warning rather than implying the model is fine. `--scale-bits` defaults to 16,
which is what `nn-load-model` (§5.7) requires; lower values are for studying the
accuracy/footprint trade-off in isolation.

### 10.3 `baru mpcc` — the MPC compiler

```sh
baru mpcc protocols                      # the registry: which protocols exist, and their status
baru mpcc dump    --expr EXPR            # the lowered circuit IR
baru mpcc analyse --expr EXPR            # rounds, interactive ops, depth
baru mpcc compare --expr EXPR            # cost every protocol under a deployment
baru mpcc run     --expr EXPR --protocol dealer|shamir|semi2k|lowgear --input k=v
baru mpcc calibrate --expr EXPR          # predicted vs measured rounds and bytes
baru mpcc optimize  --expr EXPR          # apply rewrite rules for a target backend
baru mpcc assign    --expr EXPR          # per-region protocol assignment (mixed protocols)
```

Lowering options (`--domain ringN|field128`, `--bits N`, `--public a,b,c`) and
deployment options (`--parties N`, `--security semi-honest|malicious`,
`--lan`/`--wan`, `--dishonest`, `--dealer`, `--only ids`, `--no-offline`,
`--json`) apply across the subcommands. `calibrate` is the important one: it
checks each protocol's *cost model* against the VM actually running it, and has
already caught a real accounting bug. Adding a rewrite rule is documented in
`.claude/skills/mpcc-rewrite-rule/`.

### 10.4 `baru db` — the experiment store

```sh
baru db put   --suite S --case C --backend B --metric name=value[unit] --tag T --ok true
baru db query [--suite S] [--backend B] [--tag T] [--format table|csv|latex|json]
baru db get   --id ID
baru db stats
```

Records benchmark results so a claim in a paper can be traced to the run that
produced it. `--store local:PATH` (a JSONL file, the default) or `barudb:PATH`
(the encrypted store — see [`docs/barudb-api.md`](docs/barudb-api.md)).

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
  (let ((p (zkp-prove c '("x") '(3) '("__out") '(12))))
    (zkp-verify c p '("__out") '(12))))   ; => 1
```

The claimed output must be supplied as the public input `__out`. Without it the
statement would be "some circuit was evaluated", which proves nothing useful, so
the prover refuses. (This example passed `'() '()` before 2026-08-03 and had not
been updated when the real Binius64 backend replaced the placeholder.)

### FHE: encrypted input, public weight

```lisp
(let ((keys (fhe-keygen)))                            ; swap for fhe-keygen-tfhe
  (let ((ct (fhe-enc keys 5))
        (circ (fhe-compile '(+ (* y 3) 4) '("y"))))
    (fhe-dec keys
      (fhe-eval circ keys '() '() '("y") (list ct))))) ; => 19
```

### Real arithmetic: exact `f64`

```lisp
(let ((keys (fhe-keygen-ckks-mock)))                   ; exact real arithmetic
  (let ((ct (fhe-enc keys 1.5))
        (circ (fhe-compile-ckks '(* y 2.0) '("y"))))
    (fhe-dec keys
      (fhe-eval circ keys '() '() '("y") (list ct))))) ; => 3.0000
```

This used to run on SEAL CKKS too, with a tolerance instead of an exact
comparison. That backend was removed on 2026-08-03 — see §13.

### MNIST private inference

`cargo run --release --example mnist [--mpc] [--tfhe]` — a linear
discriminator over 784 encrypted pixels with Q16.16 public weights, run on
plain / fhe-mock / MPC / TFHE, with all
backends checked against the plain result.

---

## 12. Architecture map

```
src/
  interpreter/
    parser.rs            tokenizer + parser; quote shorthand, :sintN sugar
    ast.rs               Expr / Types, Q16.16 constants
    eval.rs              trampolined evaluator, Participant, all builtins
    backend.rs           Backend trait, ShareRef, batched-primitive defaults
    aby3_simulator.rs    in-process ABY3 simulation (values only, not costs)
    transport_backend.rs distributed backend over a Transport; batched primitives
    hashcons.rs merkle.rs structural sharing; commitments
  crypto/                Fr/Share3, ABY3 protocols (mul, a2b, batched forms),
                         garbled circuits, Transport (InprocHandle/TcpTransport),
                         PQ-TLS, identity, sim (in-process wrappers)
  fhe/                   FheBackend/RealFheBackend traits and the mock, TFHE-rs
                         and exact_real backends; bfv/bgv/bgv_wide (depth-1
                         packed matvec, no relin); leveled_relu_cost (the CKKS
                         experiment's surviving level model); ntt
                         (multi-modular, exact over power-of-two q);
                         slots (CRT packing);
                         conv_pack (Cheetah-style); circuit_emitter; market; lifecycle
  hybrid/                MPC <-> FHE scheme switching, mpc_reencrypt
  gadgets/               relu, max2/max_reduce/maxpool2d, argmax, dense, conv2d
  mpcc/                  circuit IR, rewrite rules, cost models, protocol registry
                         (dealer/shamir/semi2k/lowgear), mixed assignment, VM
  zkp/                   CircuitCompiler (FlatIR), Binius64 backend, proof
                         serialization
  ot/ psi/               Chou-Orlandi + IKNP OT; DH-PSI
  ml/                    NetworkSpec/QuantizedNetwork, `baru quantize`
  precompile/            Gadget/Registry/champion dispatch, batched matvec
  db/                    the experiment store (local JSONL and BaruDB backends)
  bench/                 one module per `baru bench-*` command
  demo/                  `baru demo-private-mnist`
  repl.rs                REPL commands + example library
  bin/tcp_party.rs       multi-party node (inproc demo + TCP mode)
```

## 13. Known limitations

- **Branch leakage**: `if` on a shared condition reveals the branch taken; use
  mux-style expressions for oblivious code.
- **Semi-honest only**: no MACs, no authentication, no abort handling — a
  malicious party can deviate undetected. This applies to every layer: the ABY3
  protocols, IKNP (an inconsistent choice vector leaks the sender's bits) and
  LowGear (no zero-knowledge proofs of plaintext knowledge). What malicious
  security would cost is sketched in
  [`docs/malicious-hybrid.md`](docs/malicious-hybrid.md).
- **Scheme switching simulates threshold decryption** with a single client key
  (§7.1). The data flow is the right one; a real deployment needs
  distributed-key threshold FHE.
- **No approximate-real FHE.** The SEAL CKKS backends were removed on
  2026-08-03: approximate comparison cost seconds per operation with real error,
  and they carried 3.4 MB of vendored C++ plus the whole Windows build story. The
  `ckks-mock` backend that remains is *exact* `f64` and offers `+ − ×` only. The
  measurement lives in [`docs/experiments.md`](docs/experiments.md) §II.4, and the
  level arithmetic the flagship comparison needs lives in
  `fhe::leveled_relu_cost`.
- **Ring size**: the 128-bit ring wraps on overflow; there is no prime-order
  field.
- **Single-evaluator model**: all parties evaluate the same expression text.
- **`nn-forward` activations**: only `relu` and `identity`. Richer
  nonlinearities need a polynomial approximation to stay oblivious.

### Deviations from Scheme

Baru is Scheme-*like*, not a Scheme. The gaps are of three different kinds, and
the distinction matters because only the first is a matter of effort:

**Simply not implemented yet** (ordinary interpreter work):
`values`/`call-with-values`, `delay`/`force`, `call/cc`, hygienic macros
(`define-syntax`/`syntax-rules`), string and char procedures beyond equality, and
the numeric tower (only `i64` and Q16.16 exist).

Errors *are* now catchable via `guard` (§3): the evaluator, the Scheme
procedures, the MPC/FHE/ZKP builtins and the vector builtins all raise. The 17
remaining panics are the `MAX_FRAMES` budget, one evaluator invariant, six
unreachable type invariants and the test harness. The exact list is in §3.

Also: `define` binds in the *current* environment rather than being hoisted like
Scheme's internal `define` (which behaves as `letrec*` at the head of a body), so
mutual recursion between two internal `define`s needs `letrec`.

**Representational differences** that would need a refactor:

- **`let` is sequential** (Scheme's `let*`); see §3. `let*` is provided but
  redundant.
- **No improper pairs.** A list is `List(Vec<Types>)`, not cons cells, so
  `(cons 1 2)` is an error rather than a dotted pair, and `set-car!` has nothing
  to mutate.
- **`quote` yields an AST wrapper, not data.** `'sym` evaluates to a quoted
  *expression*, and `(car '(a b))` fails, because quoted forms are inputs to
  `eval` / `zkp-compile` rather than list-and-symbol data. Scheme's code-is-data
  property is therefore absent.
- **`and` / `or` are two-argument and bitwise**, not variadic and
  short-circuiting: `(and 6 3)` is `2`. Short-circuiting on a *secret* condition
  would leak, so this is not purely an oversight.
- **Take-everything lambdas are written `(lambda (. args) …)`**, not Scheme's
  `(lambda args …)`. A bare identifier in that position already means one
  required parameter here, and changing it would silently alter existing
  programs.

**In conflict with the cryptography**, and so unlikely ever to close:

- **Secret-dependent recursion is not expressible obliviously.** A loop whose
  trip count depends on a secret cannot reveal the condition (a leak) and cannot
  evaluate both branches (it diverges). Any oblivious language must bound loops
  publicly. The same argument applies to `call/cc` under a secret, which makes
  the *round count* data-dependent.
- **The circuit backends are static by construction.** `zkp::circuit_compiler`
  resolves lists at compile time and unrolls recursion, terminating via static
  `if` resolution; a circuit has no runtime control flow at all. Runtime-shaped
  data and dynamic dispatch are unrepresentable there, not merely unimplemented.

A fuller treatment, including why the public/secret split is the natural stopping
point, is in [`docs/survey.md`](docs/survey.md) §3.
