# Private inference performance: state, method, and what to do next

**Self-contained handoff. Start here if you are picking this up cold.**

Working notes for the **switch-as-bootstrap** paper (`paper/main.tex`,
outline in [`paper-outline.md`](paper-outline.md)) — *not* the cost-model
paper, whose agenda is [`research-agenda.md`](research-agenda.md).

Last measured **2026-07-28** on an Apple-silicon laptop, `--release`,
single-threaded. Ratios transfer between machines; absolute seconds do not.

---

> **If you are here to work on the *circuit-optimization paper*** (Baru as a
> language for private AI), read
> [`paper-circuit-optimization.md`](paper-circuit-optimization.md) instead —
> this document is the engineering state of the switch-as-bootstrap pipeline,
> which that paper uses as its expert baseline.

## 0. Orientation

**The project.** Baru evaluates a neural network with the *image and the
weights both encrypted*: linear layers under BFV (FHE), nonlinearities under
3-party ABY3 (MPC), switching representation between them. The paper's claim
is that the switch is also a **noise reset**, so a network of any depth runs
at **zero FHE bootstraps** with a depth-1 FHE engine. Everything is verified
**bit-exact** against a plaintext integer reference — that invariant is what
makes it safe to optimize aggressively.

**What happened on 2026-07-28.** One encrypted medical diagnosis went from
**32.7 s to 2.90 s online (11×)** without touching the cryptography. Every
win was round structure or work placement, and every one was invisible to the
measurement granularity in use at the time. In order:

| # | change | effect | where |
|---|---|---|---|
| 1 | batched the scheme-switch import | 32.7 s → 5.71 s | `aby3_protocol::share_input_batch` |
| 2 | integrated 3-party garbled nonlinearities | → 4.94 s | `bench::cnn::Nonlin::Gc` |
| 3 | tournament `argmax` | 2.4× on that phase at RTT 20 ms | `gadgets::argmax` |
| 4 | preprocessed the switch masks | → **2.90 s** online + 2.03 s offline | `bench::cnn::MaskPool` |
| 5 | batched the whole gadget library | 9–45× on the composable API | `src/gadgets/` |
| 6 | CSE into the compile pipeline | MNIST circuit 2365 → 1764 gates | `zkp::circuit_compiler::cse` |

Items 1–4 are §1–§2; item 5 is §2.1; item 6 is §2.2. **§2 is the most
important section in this document** — it is the method that found all six,
and the reason the earlier conclusions in the paper were wrong.

**Git state.** Items 1–4 were committed by the user as `e487bb8 Optimize` and
`0b36019 Some optimizations`. **Items 5–6 and all documentation updates were
left uncommitted** — check `git status` before assuming anything is saved.
New untracked files: `src/bench/abstraction.rs`, `src/bench/circuit.rs`,
`docs/private-inference-performance.md` (this file).

---

## 1. Where it stands

One encrypted diagnosis, PneumoniaMNIST, trained two-conv CNN, image **and**
weights encrypted, zero bootstraps, bit-exact against the plaintext integer
reference:

```sh
baru bench-medmnist --dataset pneumonia --hybrid 2 --gc --precompute
```

```
latency  : 508 ms   (fhe 278 ms, mpc 227 ms)
fhe split: weights 0 ms, mul 130 ms, mask-enc 0 ms, decrypt 118 ms, re-encrypt 37 ms
mpc split: import 23 ms, relu 124 ms, pool 63 ms, argmax 10 ms
offline  : 558 ms/image  —  ~103 per-SESSION (amortizes), ~456 per-QUERY (shifts only)
```

It was **32.7 s** when this document was started. The construction never
changed — only round batching, the protocol chosen for comparisons, what sits
on the critical path, and the speed of the ring arithmetic underneath.

| step | pneumonia latency | what changed |
|---|---|---|
| start | 32.7 s | — |
| batched scheme-switch import | 5.71 s | 3 rounds per *layer*, not per activation |
| garbled nonlinearities (`--gc`) | 4.94 s | constant-round ReLU + max-pool |
| tournament `argmax` | 4.94 s | (invisible at rtt=0; 2.4× at RTT 20 ms) |
| preprocessed masks (`--precompute`) | 2.90 s | + 2.03 s offline |
| Shoup butterfly in `ntt.rs` | 1.75 s | 4.65× on every transform |
| precomputed model (§3.1) | 1.15 s | weights encrypted once per *session* |
| pre-transformed weights (§3.2) | 1.09 s | `mul_pre_pre`, both operands prepared |
| batched garbled `y2a` (§3.3b) | 1.00 s | 379 725 → 61 messages |
| fixed-key AES gate hash (§3.3) | 542 ms | 178 ns → 5 ns per hash |
| prepared decryption key (§3.3d) | **508 ms** | key transformed once per session |

**The composition inverted twice in one day.** It was 26% MPC; after §3.1–§3.2
made FHE ~3× cheaper it became 69% MPC; then §3.3/§3.3b took the garbled ReLU
from 526 ms to 124 ms and it is **55% FHE again** (278 ms of 508). The lesson is
not any one of those numbers — it is that *"which side dominates"* has a
half-life of about one optimisation, so any plan resting on it should be
re-derived rather than inherited.

WAN (RTT 20 ms, 1 Gbps): the pipeline is **not round-bound** — that is what the
constant-round nonlinearities bought.

### The synthetic control (`bench-cnn --img 28`)

Smaller model (one conv, no hidden dense), so proportions differ — it has two
scheme switches to the medical model's four. Useful because it runs in ~2 s
and prints everything:

```sh
baru bench-cnn --img 28 --gc --precompute
# total 272ms (fhe 91ms, mpc 178ms) | offline 217ms
# fhe: keygen 5 / input 4 / weights 0 (0x) / mul 43 / mask 0 (0x) / decrypt 40 / re-encrypt 11 (3x)
# mpc: import 8 / relu 82 / pool 44 / argmax 25
```

Note this control builds a fresh session per call, so its `offline` includes the
whole per-session cost. `bench-medmnist` builds one session for the batch, which
is where the amortization shows.

---

## 2. The method — read this before optimising anything

Three wins today, and **all three were invisible at the granularity we were
measuring**. Each time, the component-level number (`fhe_ms` vs `mpc_ms`) was
correct and the inference drawn from it was wrong.

1. "The MPC segment dominates" was true. "Therefore the MPC protocols are the
   bottleneck" did not follow — 94% of that segment was an unbatched
   scheme-switch import. The paper and README had both already committed to
   the wrong conclusion in print, naming garbled circuits as "the
   highest-value remaining optimization".
2. "Garbled circuits are 2.7–3.4× faster" was true of the garbling
   *primitive*, measured in isolation. The deployable three-party protocol
   including the Y2A return leg is 1.4–1.7×.
3. "FHE is now 84%" was true, and 48% of *that* was encrypting masks — work
   that need not be on the critical path at all.

The rule that falls out: **measure phases, not components, and when a bench
reports both an isolated primitive and a full protocol, quote the protocol.**
`MpcPhases` and `FhePhases` in `src/bench/cnn.rs` exist to make this cheap.
Add a phase before optimising a phase.

Corollary that keeps biting: a **cross-stage ratio** (e.g. `bench-cohort-eval`'s
"selection is N× cheaper than inference") is invalidated whenever either stage
improves. It went 1900× → 388× today. Quote the command's output, never a
remembered number.

### 2.1 The same defect lived in the composable API (fixed 2026-07-28)

`bench-cnn` and `bench-medmnist` are hand-written Rust calling the backend's
*batched* entry points. The [`gadgets`](../src/gadgets/mod.rs) library — the
API the paper's composability section is about, and what a user actually
writes — was round-serial almost everywhere:

| gadget | was | now |
|---|---|---|
| `relu` | `lt` + `mul` per element | one `lt_batch` + one `mul_batch` |
| `dense` | one `mul` **per weight** | one `mul_batch` for the matrix |
| `conv2d` | one `mul` per multiply-accumulate | one `mul_batch` for the layer |
| `max_reduce` | left fold of `max2` (dependent) | tournament over `max2_batch` |
| `maxpool2d`, `argmax`, `max2_batch` | already batched | unchanged |

Measured by `baru bench-abstraction` (n=64, dense 8×64, real transport):

| workload | naive messages | batched messages | naive @ RTT 20 ms | batched |
|---|---|---|---|---|
| relu | 771 | 141 | 9454 ms | 1049 ms |
| max-reduce | 635 | 65 | 8418 ms | 865 ms |
| dense | 549 | 38 | 6880 ms | 154 ms |

**Bytes differ by <5% across routes.** Same protocol, same data, same values
bit-for-bit — only the number of round trips differs. After the fix the
`gadget` route matches the hand-batched route *message for message*, and a
test asserts it so the library cannot regress. The `naive` route is kept in
the benchmark permanently, because deleting it would make the size of the
defect unmeasurable.

This is worth stating in the paper: composing from the gadget library is
secure by construction **and now costs nothing over hand-writing the batched
calls**. Before the fix, the second half of that sentence was false, and the
benchmarked pipeline avoided the gadgets for exactly the operations where it
mattered.

**Not yet measured:** the Lisp front end (`Participant::new_transport`, which
runs `Expr` on a real `TransportBackend`). It sits on the same gadgets and
primitives, so it inherits the fix, but whether the interpreter or
`CircuitCompiler` introduces its *own* serialization — e.g. lowering a vector
builtin to a scalar chain — is an open question. Add a `language` route to
`bench-abstraction`; the harness is already shaped for it.

### 2.2 The compiled path is a different story — and supports the thesis

`bench-abstraction` measures the *eager* APIs. The **compiled** path
(`Expr` → `CircuitCompiler` → `FlatOp` → `mpcc`) is where Baru can beat
hand-written code rather than merely match it, and `baru bench-circuit`
measures it:

```
  program          n      ops ops/no-cse   interactive   rounds
  relu-layer       1        3          3             3        3
  relu-layer      64      192   255 (1x)           129        3
  dot             64      127        127            65        2
  max-fold         8       28 4372 (156x)            15       15
```

**Independent work batches automatically.** ReLU-layer rounds are *flat in
width* — 3 at n=1 and 3 at n=64, while interactive ops go 3 → 129. ASAP
scheduling over the dependence DAG puts every independent operation at one
level, with no vectorized IR and no user effort. That is exactly what the
hand-written route needs explicit `lt_batch`/`mul_batch` calls to achieve, and
what the gadget library had to be patched three times to get right. **On this
axis the compiler is already better than hand-writing**, which is the thesis.

**Dependent reductions needed a rewrite, and now have one (2026-07-28).** A
fold is a chain; ASAP is optimal for a *fixed* lowering but cannot
reassociate. `Rule::ReassociateReduction` rebalances a fold whose combining
operator is a **compound subgraph** into a tournament — `2n−1` rounds become
`2⌈log₂ n⌉+1`:

| max-fold | as written | rebalanced |
|---|---|---|
| n=4 | 7 rounds | 5 |
| n=8 | 15 | **7** |
| n=10 | 19 | **9** |

End to end through `mpcc optimize --protocol shamir` on the n=8 fold:
**65 → 29 rounds (2.21×)**, instruction count unchanged, differentially
verified. This closes the last axis on which hand-written code beat the
compiler, and it generalises: `gadgets::argmax` and `max_reduce` had to be
rewritten as tournaments *by hand*, whereas the rule derives the same
transformation for any associative operator, including compound ones it has
never seen.

**CSE was missing from the compile path entirely (fixed 2026-07-28).**
`mpcc::opt` had a `Cse` rule, but it was reachable only from
`baru mpcc optimize` — the FHE and ZKP pipelines ran `fold_constants` +
dead-wire elimination and nothing else. So every circuit handed to a backend
carried whatever redundancy the source expression had. A fold written the
obvious way repeats its accumulator subtree at each step and flattens
**exponentially**: 4372 ops at n=8.

The pass now lives in `zkp::circuit_compiler` beside `fold_constants` (it had
to move *down*: `mpcc::opt` imports `FlatOp` from `zkp`, so the dependency
only runs one way) and `mpcc::opt::cse` delegates to it. Measured effect:

| circuit | before | after |
|---|---|---|
| max-fold n=8 | 4372 ops | 28 ops (156×) |
| **MNIST demo (real)** | **2365 gates** | **1764 gates (−25%)** |

MNIST scores are bit-identical (3389.7422 / 2853.0156), and all 18 FHE
differential tests pass including real TFHE and SEAL CKKS. At TFHE's
~0.2–0.5 s/gate those 601 gates are minutes per run.

`CircuitCompiler::without_cse()` exists for A/B measurement only.

---

## 3. Next steps, in priority order

### 3.0 ~~Reassociate dependent reductions~~ — DONE 2026-07-28

Landed as `Rule::ReassociateReduction` (`src/mpcc/opt.rs`). It recognises an
associative reduction whose operator is a repeated *subgraph shape* and
rebalances the chain into a tournament. Max-pooling, argmax and max-reduction
are the AI idioms it covers, and they are everywhere in a CNN.

**Verified by** `compound_reduction_rebalances_and_stays_exact` (equivalence
via `mpcc::opt::equivalent`, plus depth strictly falls),
`a_non_associative_reduction_is_left_alone` (a subtraction fold must not
move), and `compound_reductions_become_logarithmic_in_depth` (the gap widens
with width). `optimize` additionally runs `equivalent` over the whole result.

**Soundness.** `optimize` accepts candidates on **cost**, not on semantics, so
a rule cannot rely on the search to reject an unsound rewrite.
`step_is_associative` therefore tests the operator by evaluation — edge values
plus random triples, `f(f(a,b),c)` against `f(a,f(b,c))` — before the rewrite
is offered at all.

**Three things that bit, all worth knowing before touching this code:**

1. **"Used once" is the wrong privacy test for the accumulator.** A maximum
   reads `acc` three times *inside its own step* (compare, subtract, add). The
   correct condition is that every use of `acc` lies inside the step that
   consumes it. Getting this wrong made the rule silently never fire.
2. **Prefer the longest chain.** A fold's inner roots are valid chains too, so
   taking the first match rebalanced a 4-leaf prefix and left the tail serial
   — 65 → 47 rounds instead of 65 → 29.
3. **A rebalanced tree still contains chains.** Every root-to-leaf path is one,
   so a fixpoint pass cheerfully rebuilt a balanced path into a deeper one and
   depth came out `n+1` instead of `2⌈log₂n⌉+1`. The rule now refuses any
   rewrite that does not strictly reduce depth.

The tie-breaking caution from `gadgets::argmax` still applies if this is ever
extended to reductions returning an *index*: only the value is
order-insensitive.


### 3.1 ~~Precompute the encrypted **model**~~ — DONE 2026-07-30

Landed as `CnnSession` (`src/bench/cnn.rs`): the FHE context, the key and the
encrypted weights are built once per model and reused by every query.
`bench-medmnist` now builds one session for the whole batch.

**Measured** (pneumonia, `--gc`, this machine — the control is the same command
without `--precompute`):

| | control | precomputed |
|---|---|---|
| online latency | 1752 ms | **1134 ms** |
| of which FHE | 971 ms | **358 ms** |
| weight encryption, online | 152 ms | **0 ms** |
| mask encryption, online | 457 ms | **0 ms** |

**The predicted 670 ms was stale, and the shape of the result is what matters.**
That figure predated the Shoup butterfly (§II.2 of `experiments.md`), which made
`encrypt_poly` several times cheaper; on current code the model's encryption is
**152 ms**, not 670. The saving is real but the headline 618 ms of offline work
is mostly the *masks*, not the weights.

**Which is why the two are now reported separately**, and this is the part worth
carrying into the paper. `CnnRun` splits `offline_ms` into
`offline_session_ms` and `offline_query_ms`, and `bench-medmnist` prints both
per image:

```
offline  : 613 ms/image        (1 image)      511 ms/image      (3 images)
           156 ms/image per-SESSION            52 ms/image per-SESSION
           457 ms/image per-QUERY             459 ms/image per-QUERY
```

The per-session line falls exactly as `1/batch` while the per-query line stays
flat. That is the distinction stated as a measurement rather than a claim:

- **Masks are per-query.** Pooling them shifts latency off the critical path
  and buys **no throughput** — a saturated server does the same total work.
- **Weights are per-session.** Encrypting them once is genuine **amortization**
  — total work per query drops permanently after the first.

**Verified by** `precomputed_masks_change_nothing_but_the_timing` (extended: the
online path must do zero weight encryptions and the offline column must grow)
and `the_encrypted_model_is_reused_across_queries_in_a_session`.

**Two structural notes for whoever continues:**

- `weight_polys` is now the single source of truth for the pipeline's shape.
  It yields one packed polynomial per ct×ct multiply, and since every multiply
  is followed by exactly one scheme switch, it *also* sizes the mask pool. The
  hand-written count in `MaskPool::for_model` is gone — the two can no longer
  drift, rather than drifting and being caught.
- The keygen trap held: `CnnSession::new` does keygen first, and everything else
  in the session is encrypted under that key.

### 3.2 ~~Store the weight ciphertexts pre-transformed (NTT domain)~~ — DONE 2026-07-30

`EncryptedModel` now stores `BfvCtNtt`, not `BfvCt`: the session's weights are
forward-transformed offline as well as encrypted, and `Bfv::mul_pre_pre` takes
both operands prepared, leaving only the 3 inverse transforms per prime online.
New path: `ntt::exact_bfv_tensor_pre_pre` → `poly::rlwe_tensor_pre_pre` →
`Bfv::mul_pre_pre`.

**Measured** (pneumonia, `--gc --precompute`):

| | before 3.2 | after |
|---|---|---|
| `mul` phase | 172 ms | **127 ms** (−26%) |
| online latency | 1152 ms | **1086 ms** |
| per-session offline | 78 ms/image | 101 ms/image |

The synthetic control moves the same way: `mul` 57 → 42 ms, also −26%.

**The estimate was 40%; the truth is 26%** — worth recording, because the
arithmetic explains it. Removing both forward transforms takes 2 of the 5
transforms per prime, but the tensor also does 3 pointwise `mulmod` passes and a
Garner CRT reconstruction, and neither moved. The transforms were never the
whole phase. Cost moves rather than vanishing: the per-session offline grew
78 → 101 ms/image, which is the transforms now being paid there.

**Verified by** `mul_pre_pre_agrees_with_mul_and_mul_pre` — a differential test
requiring bit-identical output against both existing paths, not approximate
agreement. This is the one place in the change where a wrong answer would be
silent: a bad product decrypts to plausible garbage and surfaces as a
misclassification several layers away. The end-to-end `2/2 matches plaintext
exactly` is the second check.

One asymmetry: under the NTL backend `PreparedPair` holds raw coefficients, so
`rlwe_tensor_pre_pre` degrades to the unprepared path and the saving does not
exist there. Mixing a raw operand with a transformed one now panics rather than
returning a quiet wrong answer.

**Where this leaves the pipeline.** FHE is 311 ms of 1086 (29%); MPC is 771 ms
(71%), of which ReLU alone is 510 ms. Further FHE micro-optimisation has little
left to reach — see the struck entry in §3.5 and read §3.3 next.

### 3.3 ~~Replace the garbled-circuit label hash with fixed-key AES~~ — DONE 2026-07-30

`crypto::garbled::hash` is now **TMMO over a fixed-key AES permutation**,
`H(x, i) = π(π(x) ⊕ i) ⊕ π(x)`, replacing truncated SHA-256.

| | SHA-256 | TMMO/AES |
|---|---|---|
| per hash | 178 ns | **5 ns** |
| garbling one conv layer | 169 ms | **10 ms** |
| ReLU phase (`bench-cnn --img 28 --gc`) | 323 ms | **88 ms** |
| synthetic total | 628 ms | **270 ms** |
| medical diagnosis | 1000 ms | **542 ms** |

Bit-exact throughout; `gc_and_a2b_nonlinearities_agree` still holds, so the
garbled and arithmetic routes remain interchangeable.

**The construction is not interchangeable with a plausible-looking one.**
Half-gates with free-XOR needs the gate hash to be **tweakable circular
correlation robust** — stronger than collision resistance, because the garbler
hashes labels differing by the global offset `R` and reuses `R` across every
gate. Guo–Katz–Wang–Yu (S&P 2020) prove TMMO achieves it; `emp-toolkit` uses the
same. Plain MMO `π(x) ⊕ x` is not tweakable, and `π(x ⊕ i) ⊕ x` leaks under
free-XOR's correlated labels. The two permutation calls are load-bearing.

**⚠️ The first attempt made it 3× SLOWER, and the reason generalises.** With a
default `cargo build` on aarch64 the RustCrypto `aes` crate falls back to a
constant-time *bitsliced software* implementation: the gate hash measured
**525 ns**, three times worse than the SHA-256 it replaced — because `sha2` *is*
hardware-accelerated on Apple Silicon and `aes` was not. The fix is
`.cargo/config.toml`, scoped to `aarch64-apple-darwin`:

```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-feature=+aes", "--cfg", "aes_armv8"]
```

Scoped deliberately — every Apple Silicon machine has the crypto extensions, so
`+aes` cannot SIGILL there, while generic aarch64 keeps the software fallback.
x86-64 needs nothing; the crate detects AES-NI at run time.
`where_the_garbled_relu_spends_its_time` now **warns** if it measures >50 ns per
hash, so a build that has silently lost the hardware path is visible rather than
just slow. *"Replace the hash with the faster primitive" was not sufficient
advice; which build gets the hardware was the whole result.*

**What is verified and what is not.** Verified: the construction computes what
the citation specifies, the key is derived rather than chosen
(`pi_key_is_nothing_up_my_sleeve`), the hash is deterministic and
tweak-separated, and every garbling test including the bit-exact end-to-end CNN
passes. **Not verified by any test here: that the implementation is *secure*.** A
wrong gate hash produces correct outputs while breaking the security argument, so
these tests are evidence of correctness only. **This warrants review against the
paper by someone who has it open** — it is the one change in this sequence whose
failure mode is silent.

### 3.3b ~~The garbled `y2a` conversion~~ — DONE 2026-07-30

`gc_protocol::y2a` converts the garbled boolean output back to arithmetic. It
called `share_input` **per (item, bit)** — `2 × 5408 × 12 = 129 792` calls for
one D=28 ReLU — while `aby3_protocol::share_input_batch` had existed since the
scheme-switch import was fixed. Two batched calls replace both loops.

| `bench-cnn --img 28 --gc --precompute` | before | after |
|---|---|---|
| ReLU phase | 409 ms | **323 ms** |
| total | 736 ms | **628 ms** |
| **MPC messages** | **319 139** | **47** |
| MPC bytes | 21.8 MB | 18.0 MB |

Medical model: 1086 → 1000 ms, traffic 379 725 messages → 61. Bit-exact
throughout.

**This is the fifth instance of the same defect** (import, `gadgets::relu`,
`bench-hybrid-nn`, `bench-deep-nn`, now `y2a`), and in every case the batched
primitive already existed. §2 is the method that finds them; the thing worth
internalising is that *four* previous fixes did not stop the fifth, because each
was fixed where it was found rather than searched for elsewhere. **A useful next
task is not an optimisation at all: grep every call of `share_input`,
`share_secret`, `mul`, `lt` and `a2b` that sits inside a loop over a layer, and
check each against its `_batch` form.**

Pinned by `garbled_nonlinearity_does_not_scale_with_layer_width`, asserting both
a ratio (constant in layer width) and an absolute ceiling — a ratio test alone
passes if both sides regress together.

### 3.3c ~~That search~~ — DONE 2026-07-30, and it found a sixth instance

Running the grep above immediately turned up **`bench-train`**: both trainers
issued one `mul` per weight and one `arith_shr` (hence one `a2b`) per gradient.
Batched with `gadgets::dense` for the matvecs, one `mul_batch` per outer product,
and a new `gadgets::arith_shr_batch` per learning rate:

| `baru bench-train` | messages before | after |
|---|---|---|
| linear | 62 683 | **3 643** |
| MLP (`--hidden 8`) | 199 655 | **7 271** |

Models stay bit-for-bit identical to the plaintext integer-SGD reference, with
identical loss curves. The MLP forward pass also stopped recomputing its own
comparison — the backward pass needs `step(z1)`, which *is* the ReLU's mask, so
it is computed once and reused.

Most other hits are deliberate per-element **controls** — `NaiveMatVec`,
`bench-nn`'s naive path, `bench-abstraction` — which exist to keep the gap
reproducible and were left alone. Distinguishing a control from a defect is the
only judgement in this search; everything else is mechanical.

Pinned by `training_round_structure_is_constant_in_model_size`. **Repeat this
search whenever an interactive primitive gains a `_batch` form** — that is the
cheap habit six instances have now argued for.

### 3.3d ~~Prepare the decryption key per session~~ — DONE 2026-07-30

With the model precomputed and the gate hash fixed, `decrypt` became the FHE
side's largest item. `Bfv::decrypt_poly` was doing two avoidable things on
**every call**: bincode-deserializing the whole key bundle (three
`n`-coefficient polynomials, ~196 KB at n=4096) and forward-transforming the
secret key — while an inference decrypts once per ct×ct multiply and the key
does not change within a session.

`Bfv::decrypt_key` deserializes and transforms once; `decrypt_poly_pre` uses it.
`CnnSession` holds it, alongside the encrypted model.

| | before | after |
|---|---|---|
| decrypt, medical | 149 ms | **118 ms** |
| decrypt, synthetic | 49 ms | **40 ms** |
| medical latency | 542 ms | **509 ms** |

21%, not the 33% the transform count suggests — the centering, the CRT
reconstruction and the final rounding did not move, so the hoisted transform was
never the whole phase. Same shape as §3.2's 26%-against-40%: **in this codebase
the transforms are consistently about two thirds of what a transform-count
estimate predicts.** Worth assuming for the next such estimate.

Verified by `decrypt_matches_the_prepared_path`, requiring bit-identical output
on both a fresh ciphertext and a degree-2 product (different numbers of key
powers).

### 3.3e ~~Prepare the encryption key per session~~ — DONE 2026-07-30

The same hoist as §3.3d, on the busier side. `Bfv::encrypt_poly` deserialized
the key bundle and forward-transformed the *public* key on every call, while
encryption multiplies a **fresh** `u` by a **fixed** `(b, a)` — and the mask pool
alone encrypts three polynomials per switch, ~150 per inference.

`Bfv::encrypt_key` / `encrypt_poly_pre`, held by `CnnSession` next to the
decryption key. This one pays mostly on the *offline* column, where the mask pool
lives:

| | before | after |
|---|---|---|
| mask pool (per-query offline) | 456 ms/image | **321 ms/image** |
| re-encrypt (online) | 37 ms | **24 ms** |
| medical latency | 508 ms | **488 ms** |
| synthetic offline | 217 ms | **160 ms** |

**This closed a gap that had just opened** — see §4's offline-vs-online entry.

Verified by `encrypt_prepared_roundtrips`. Note it cannot be a
ciphertext-for-ciphertext differential like §3.2's and §3.3d's — both paths draw
fresh `u`, `e₁`, `e₂` — so it checks the property that matters instead: the
ciphertext decrypts to the message, survives a ct×ct multiply, and interoperates
with the unprepared path.

**Three hoists, one shape.** §3.2 (weights), §3.3d (secret key) and §3.3e
(public key) are the same observation applied three times: *an operand that is
fixed for the session was being transformed per ciphertext*. Any remaining
`poly_mul_mod` with a session-lifetime operand is a candidate; there are no
obvious ones left in `bfv.rs`.

### 3.4 Parallelism

Everything is single-threaded. The per-output-channel FHE work (encrypt, mul,
decrypt) is embarrassingly parallel — conv2 has 16 independent channels — and
garbling parallelises across gates.

**Do this last of the three, and keep a single-thread control.** The paper
currently says "single-thread on commodity hardware ... read as ratios";
threading changes what every number means, so it needs its own flag
(`--threads N`, default 1) and its own row, not a silent global change.

**Expected:** near-linear in cores on the FHE phases. This is the largest
absolute win available and the least interesting scientifically.

### 3.5 Things measured and deliberately *not* worth doing

- **Faster decryption by computing only the coefficients we read.** We decrypt
  a full polynomial and then read `oh·ow` positions. Selective evaluation of a
  polynomial product is not obviously cheaper than the NTT that computes all of
  it. Price it analytically before writing code.
- **Optimising `import`.** It is 21 ms. It was 19 100 ms this morning; that is
  finished.
- **Optimising `argmax` further.** 7 ms on 2 classes, 30 ms on 11.
- ~~**More MPC protocol work generally.**~~ **This entry flipped twice in one
  day; do not inherit either state.** Written when MPC was 756 ms of 2896 ms
  (26%). §3.1–§3.2 made FHE ~3× cheaper and it became 69%, with the garbled
  ReLU the single biggest item. Then §3.3 (fixed-key AES) and §3.3b (batched
  `y2a`) took that ReLU from 526 ms to 124 ms, and MPC is back to **227 ms of
  508 (45%)**. Re-derive before acting. Re-derive before acting: this
  is exactly the kind of entry §2 warns goes stale when the other side moves.

---

## 4. Open paper gaps (independent of performance work)

- **No latency comparison against prior systems.** `tab:sota` compares only
  *what is hidden*. We now claim seconds per inference; Gazelle / Delphi /
  Cheetah / CrypTFlow2 all report latencies. A reviewer will ask. The honest
  framing is a two-axis table — they are faster *and* hide less (plaintext
  weights, 2-party) — but the numbers have to be in it. **Do not** put their
  numbers and ours in one column without the hardware and threat-model caveat.
- **`tab:arch` mixes training budgets.** The winning row is a longer retrain
  (98.89%); the other rows are the original sweep. Flagged in the caption, but
  re-running the sweep under one recipe would remove the asterisk.
- ~~**The throughput caveat for preprocessing**~~ — resolved 2026-07-30. The
  two kinds are now measured separately (`offline_session_ms` vs
  `offline_query_ms`, printed per image by `bench-medmnist`), and §3.1 has the
  numbers showing the per-session line falling as `1/batch` while the per-query
  line stays flat. The paper should use that split rather than one offline
  total.
- **`--precompute` offline vs online — the margin nearly closed, and is now the
  thing to watch.** Only the per-*query* part matters for a continuously-loaded
  server; the per-session part amortizes away. After §3.3 cut the online path to
  508 ms the mask pool stood at **456 ms/image — 90% of it**, about to cross the
  line where the framing must change from "latency" to "latency, given spare
  offline capacity". §3.3e took the pool to **321 ms** against **488 ms** online,
  so it is comfortably under again. But masks are per-query and never amortize,
  so **every further online win narrows this margin**. Current: synthetic 160 ms
  offline / 260 ms online; medical 321 ms/image per-query offline / 488 ms
  online.
- **MedMNIST scope.** Already stated: downsampled, pre-processed, no clinical
  claim. Keep it stated.

---

## 5. Reproduction

```sh
cargo build --release

# headline: private diagnosis
./target/release/baru bench-medmnist --dataset all --hybrid 3 --gc --precompute
./target/release/baru bench-medmnist --dataset all --hybrid 2 --gc --precompute \
    --rtt-ms 20 --bandwidth-mbps 1000        # WAN

# the fast control, prints every phase
./target/release/baru bench-cnn --img 28 --gc --precompute
./target/release/baru bench-cnn --img 28 --both     # a2b vs gc, refuses to
                                                    # print a ratio unless both
                                                    # agree bit-exactly
# isolated protocol comparison
./target/release/baru bench-gc-compare

# what the abstraction costs: naive vs gadget library vs hand-batched
./target/release/baru bench-abstraction --n 64 --out-dim 8

# circuit generation quality: ops, CSE contribution, ASAP round count
./target/release/baru bench-circuit

# circuit size on a real AI workload (prints "N gates")
cargo run --release --example mnist

# correctness — all three must be green
cargo test --release --lib                                   # fast suite (~380)
cargo test --release --lib -- --ignored \
    --skip bench::cohort --skip bench::net                   # slow suite (~41)
cargo test --release --test fhe_differential -- --ignored    # 18, real TFHE + SEAL
```

The gadget and circuit-quality tests are in the **fast** suite (they use the
in-process simulator); the pipeline exactness tests are `#[ignore]`d because
they run real BFV at `n=4096`. To run just the slow pipeline ones:
`cargo test --release --lib -- --ignored bench::cnn`.

`bench::cohort` needs `BARUDB_BIN=~/github/barudb/target/release/barudb` and
`BLINDSEARCH_PASSWORD` (not `BARUDB_PASSWORD`). The two `bench::net` tests
measure a real socket and are load-sensitive — they fail under a loaded
machine and pass in isolation; that is not a regression.

Key tests to keep green when touching any of this:

| test | pins |
|---|---|
| `two_conv_hybrid_matches_plaintext` | the flagship exactness claim |
| `gc_and_a2b_nonlinearities_agree` | the two nonlinearity routes are the same function |
| `precomputed_masks_change_nothing_but_the_timing` | preprocessing is not cheating |
| `scheme_switch_import_does_not_scale_with_layer_width` | import stays batched (message count, not wall-clock) |
| `argmax_breaks_ties_to_first_over_every_small_input` | tournament tie-breaking, exhaustive |
| `gadget_route_matches_the_hand_batched_route_and_the_reference` | the gadget library stays batched (round count, not results) |
| `layer_round_count_is_flat_in_width` | ASAP scheduling still batches independent ops |
| `cse_collapses_the_exponential_fold_and_is_on_by_default` | CSE is part of compilation, not opt-in |
| `compiled_circuits_arrive_already_cse_d` | backends never see duplicate subexpressions |
| `compiled_and_hand_written_agree_on_the_whole_network` | the compiled route computes the network |
| `compound_reduction_rebalances_and_stays_exact` | the reduction rewrite is exact and cuts depth |
| `a_non_associative_reduction_is_left_alone` | it refuses non-associative operators |
| `each_backend_prefers_its_own_circuit` | the cross-application diagonal wins |
| `a_mixed_assignment_computes_the_same_value_as_pure_mpc` | the MPC/FHE bridge is exact |
| `a_second_multiplication_is_demoted_rather_than_attempted` | depth-1 is enforced, not assumed |
| `the_models_decision_is_executed_and_exact` | the assignment reaches the executor and is exact |
| `packing_the_crossing_matches_the_scalar_route` | packed switch is exact, constant in width |
| `packed_dot_refuses_a_layer_its_modulus_cannot_hold` | the plaintext-modulus bound fails loudly |
| `sinking_an_operation_across_the_boundary_saves_a_crossing` | the boundary rule saves a switch, value unchanged |
| `some_workload_produces_genuinely_different_circuits` | divergence is real, not nominal |
| `batched_input_sharing_matches_the_scalar_protocol` | batched == scalar sharing |

---

## 6. Where the code is

| thing | location |
|---|---|
| pipeline | `bench::cnn::run_cnn_hybrid_full` |
| phase structs | `bench::cnn::{MpcPhases, FhePhases}` |
| nonlinearity switch | `bench::cnn::Nonlin`, `TransportBackend::set_gc_nonlinearities` |
| batched import | `aby3_protocol::share_input_batch`, `bench::cnn::import_raw_batch` |
| mask preprocessing | `bench::cnn::{MaskPool, Masks}` |
| tournament argmax | `gadgets::argmax` |
| abstraction-cost bench | `bench::abstraction` (`baru bench-abstraction`) |
| circuit-quality bench | `bench::circuit` (`baru bench-circuit`) |
| compiled-vs-handwritten NN | `bench::compiled_nn` (`baru bench-compiled-nn`) |
| backend-aware divergence | `bench::backend_aware` (`baru bench-backend-aware`) |
| level-batched circuit executor | `mpcc::exec::run_flat` |
| mixed MPC+FHE executor | `mpcc::mixed::run_mixed`, `mpcc::mixed::FheArith` |
| model decision -> execution | `mpcc::mixed::from_assignment`, `bench::mixed_layer` |
| packed MPC->FHE crossing | `mpcc::mixed::packed_dot` |
| boundary-placement rule | `mpcc::mixed::Placement::sink_to_fhe` |
| interactive-call counter | `TransportBackend::interactive_calls` |
| shared FlatOp passes | `zkp::circuit_compiler::{cse, rename}`, `CircuitCompiler::fold_constants` |
| BFV / NTT | `fhe::bfv`, `fhe::ntt` |
| conv packing | `fhe::conv_pack` |

Docstrings in `bench::cnn` and `bench::medmnist` carry the current measured
tables. **If you change a number, change them in the same commit** — a stale
docstring citing a measurement is worse than no docstring
(see the reproducibility note in `tools/README.md`).

---

## 7. Working conventions in this repo

Not style preferences — each of these came from something going wrong.

- **Never write a measured number you have not just run.** A results table
  written ahead of the run survives into a commit and reads as a measurement.
  This document's own §2.2 table was drafted from an estimate and had to be
  corrected against the real output before it was committed.
- **Update docstrings in the same change as the number.** `bench::cnn` and
  `bench::medmnist` module docs carry the current tables. A stale docstring
  citing a measurement is worse than no docstring, because it looks
  authoritative.
- **Anything that produces a committed artifact goes in `tools/`,** never in a
  scratch directory — see `tools/README.md`. The MNIST weights once outlived
  their trainer and the result became unreproducible.
- **A cross-stage ratio is invalidated by improving either stage.**
  `bench-cohort-eval`'s "cohort selection is N× cheaper than inference" went
  1900× → 388× in one day. Quote the command's output, never a remembered
  number.
- **When a bench prints both an isolated primitive and a full protocol, quote
  the protocol.** The paper claimed 2.7–3.4× for garbled ReLU; that was the
  2-party garbling primitive. The deployable 3-party protocol with its Y2A
  return leg is 1.4–1.7×.
- **`cargo fmt` reformats ~35 pre-existing files.** Run `rustfmt` on specific
  files. `src/bench/medmnist.rs` is rustfmt-clean and should stay that way;
  `cnn.rs`, `gadgets/mod.rs` and `aby3_protocol.rs` are not and never were.
- **Preprocessing: say which kind it is.** Per-*query* work (masks) moved
  offline shifts latency but buys **no throughput** — a saturated server does
  the same total work. Per-*session* work (encrypted weights, keys) is genuine
  amortization. §3.1 depends on this distinction.
- **Two environment-dependent test groups.** `bench::cohort` needs
  `BARUDB_BIN` + `BLINDSEARCH_PASSWORD`; the two `bench::net` tests measure a
  real socket and fail under machine load but pass in isolation. Both
  reproduce on a clean tree — neither is a regression you introduced.

### Traps that already bit once

- **The mask pool must use the same key as the pipeline.** The first
  `--precompute` draft called `s.keygen()` a second time to build the pool;
  the masks then did not decrypt. Keygen is hoisted above the preamble.
- **Tie-breaking survives reassociation only by accident of ordering.**
  `gadgets::argmax` returns the *first* maximum. The tournament preserves that
  only because the bracket pairs adjacent elements in index order. Random
  logits never produce a tie split across sibling subtrees, so it is pinned by
  an exhaustive test over small alphabets, not by sampling. `max_reduce`
  returns a value, not an index, so it is exempt.
- **CSE had to move *down*, not up.** `mpcc::opt` imports `FlatOp` from
  `zkp::circuit_compiler`, so the dependency runs one way; the pass lives in
  `zkp` and `mpcc::opt::cse` delegates to it.
- **Message count, not wall-clock, is the right regression metric for round
  structure.** Nothing about the *results* changes when a gadget regresses to
  a per-element loop, and at zero latency the wall-clock barely moves either.
  Two tests assert message counts for exactly this reason.
