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
latency  : 2896 ms   (fhe 2123 ms, mpc 756 ms)
fhe split: weights 670 ms, mul 634 ms, mask-enc 0 ms, decrypt 668 ms, re-encrypt 151 ms
mpc split: import 21 ms, relu 505 ms, pool 216 ms, argmax 7 ms
offline  : 2028 ms query-independent (keygen + mask pool), NOT in the latency above
```

It was **32.7 s** the same morning. The construction never changed — only
round batching, the protocol chosen for comparisons, and what sits on the
critical path.

| step | pneumonia latency | what changed |
|---|---|---|
| start of day | 32.7 s | — |
| batched scheme-switch import | 5.71 s | 3 rounds per *layer*, not per activation |
| garbled nonlinearities (`--gc`) | 4.94 s | constant-round ReLU + max-pool |
| tournament `argmax` | 4.94 s | (invisible at rtt=0; 2.4× at RTT 20 ms) |
| preprocessed masks (`--precompute`) | **2.90 s** | + 2.03 s offline |

WAN (RTT 20 ms, 1 Gbps): 5.00 s online. The pipeline is **not round-bound** —
that is what the constant-round nonlinearities bought.

### The synthetic control (`bench-cnn --img 28`)

Smaller model (one conv, no hidden dense), so proportions differ — it has two
scheme switches to the medical model's four. Useful because it runs in ~2 s
and prints everything:

```sh
baru bench-cnn --img 28 --gc --precompute
# total 1312ms (fhe 695ms, mpc 600ms) | offline 673ms
# fhe: keygen 10 / input 17 / weights 218 (13x) / mul 209 / mask 0 (0x) / decrypt 216 / re-encrypt 51 (3x)
# mpc: import 11 / relu 412 / pool 158 / argmax 28
```

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


### 3.1 Precompute the encrypted **model** — best remaining win, low risk

**670 ms, 23% of the online diagnosis**, and it is query-independent for a
strictly better reason than the masks were.

`hybrid_conv_open` and `hybrid_matvec_open` call `s.encrypt_poly(ck, &kpoly)`
inside their per-channel / per-chunk loop, so every inference re-encrypts the
whole model. But the model does not change between patients. Under a
per-session key the encrypted weights are uploaded **once** and reused for
every query in that session.

Note this is *better than the mask result*, and the distinction belongs in the
paper:

- **Masks** are per-query — precomputing shifts latency but buys **no
  throughput**. A saturated server does the same total work.
- **Weights** are per-*session* — precomputing is genuine **amortization**.
  Total work drops by 670 ms per query after the first.

**How:** generalise `MaskPool` into a session preamble holding both the mask
pool and the encrypted model. Follow the existing shape exactly: build it
before `start`, report it in `offline_ms`, assert exact consumption so the
size prediction cannot drift from the pipeline
(`MaskPool::for_model` + `rows_per_chunk` is the pattern).

**Trap:** the model ciphertexts must be encrypted under the *same* key as
everything else. This already bit once — the first `--precompute` draft called
`s.keygen()` a second time for the pool, which produces masks that do not
decrypt. Hoist keygen above the preamble.

**Verify:** extend `precomputed_masks_change_nothing_but_the_timing` — same
class, both nonlinearity routes, and the two-conv model (multi-layer chunking
is where a size prediction goes wrong).

**Expected:** 2.90 s → ~2.23 s online.

### 3.2 Store the weight ciphertexts **pre-transformed** (NTT domain)

Follows directly from 3.1 and needs it first.

`mul_pre(a: &BfvCtNtt, b: &BfvCt)` takes the input pre-transformed and
transforms `b` (the weights) inside `exact_bfv_tensor_pre`, once per output
channel. With one input and *N* weight ciphertexts per layer, almost all the
forward-NTT work is on the weight side — and if the weights are already
offline, their transforms can be too.

**How:** add a both-operands-transformed entry point
(`exact_bfv_tensor_pre_pre` / `Bfv::mul_pre_pre`) and store `BfvCtNtt` in the
preamble instead of `BfvCt`. `BfvCtNtt` is currently not `Clone`/serialisable
— it will need to be.

**Expected:** roughly 2 of the ~5 NTTs per channel, so ~40% of the 634 ms
`mul` phase. Estimate only — **measure before quoting**. Cumulative with 3.1:
~2.0 s online.

### 3.3 Parallelism

Everything is single-threaded. The per-output-channel FHE work (encrypt, mul,
decrypt) is embarrassingly parallel — conv2 has 16 independent channels — and
garbling parallelises across gates.

**Do this last of the three, and keep a single-thread control.** The paper
currently says "single-thread on commodity hardware ... read as ratios";
threading changes what every number means, so it needs its own flag
(`--threads N`, default 1) and its own row, not a silent global change.

**Expected:** near-linear in cores on the FHE phases. This is the largest
absolute win available and the least interesting scientifically.

### 3.4 Things measured and deliberately *not* worth doing

- **Faster decryption by computing only the coefficients we read.** We decrypt
  a full polynomial and then read `oh·ow` positions. Selective evaluation of a
  polynomial product is not obviously cheaper than the NTT that computes all of
  it. Price it analytically before writing code.
- **Optimising `import`.** It is 21 ms. It was 19 100 ms this morning; that is
  finished.
- **Optimising `argmax` further.** 7 ms on 2 classes, 30 ms on 11.
- **More MPC protocol work generally.** The MPC segment is 756 ms of 2896 ms
  and its largest phase (ReLU, 505 ms) is already constant-round garbled.

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
- **The throughput caveat for preprocessing** (see 3.1) is stated for masks in
  §7.5 but should be sharpened once weights are preprocessed too, because the
  two differ in kind.
- **`--precompute` offline cost exceeds online cost** for the synthetic model
  (673 ms vs 1312 ms is fine; on the medical model 2028 vs 2896 is also fine —
  but check this stays true as online shrinks). If offline ever exceeds online
  by a lot, a continuously-loaded server cannot keep up and the framing has to
  change from "latency" to "latency, given spare offline capacity".
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
