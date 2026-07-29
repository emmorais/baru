# Paper plan: Baru as a language for private AI

**Status: plan, not results.** Written 2026-07-28 after auditing what the repo
actually measures. Numbers below were re-verified against the current tree
unless marked otherwise. Companion to
[`private-inference-performance.md`](private-inference-performance.md) (the
engineering state) and [`research-agenda.md`](research-agenda.md) (Thesis S).

---

## 1. The claim, and a caveat about the premise

The intended claim: **Baru is a language for private AI, because it optimizes
circuits automatically.** The working premise was *"if Baru has no overhead
versus hand-written circuits, that is enough to publish."*

That premise is too weak, and it should be stated here before it costs a
submission. "Our DSL matches hand-written code" invites exactly one reviewer
question — *then why compile?* — and the answer has to be more than
convenience, because the audience for a privacy-preserving-ML paper is people
who are willing to hand-write.

The good news is that the stronger claim is the true one, and the repo already
has most of the evidence for it. **Automation does not merely tie here; it
wins, and it wins for a reason that generalises**: a compiler batches
*everything* that is independent, whereas a human batches what they remembered
to look at. We have direct evidence of the difference on our own expert code
(§4).

So: parity is the *floor* you report, not the claim you sell.

---

## 2. What the evidence already supports

All re-verified against the current tree (`baru bench-circuit`,
`baru bench-abstraction`, `baru mpcc assign`).

### 2.1 Round-optimality of independent work is automatic

`mpcc::schedule` is ASAP over the dependence DAG, so independent operations
land at one level and cost one round:

| program | n=1 | n=64 | n=512 | rounds |
|---|---|---|---|---|
| relu-layer | 3 ops | 192 ops | 1536 ops | **3, flat** |
| dot | 1 op | 127 ops | 1023 ops | **2, flat** |

Interactive ops grow 3 → 1025 while rounds stay at 3. No vectorized IR, no
user annotation. This is what the hand-written route needs explicit
`lt_batch`/`mul_batch` calls to achieve.

### 2.2 The abstraction has no residual overhead

`bench-abstraction` runs one layer three ways over the identical transport:

| workload | naive | gadget library | hand-batched |
|---|---|---|---|
| relu (n=64) | 771 msgs | **141** | 141 |
| max-reduce (n=64) | 635 msgs | **65** | 65 |
| dense (8×64) | 549 msgs | **38** | 38 |

The library now matches hand-written batched code *message for message*, with
byte counts within 5% of the naive route and bit-identical outputs. A test
pins the round count so it cannot regress.

### 2.3 The compiler removes redundancy a human would ship

CSE was missing from the compile pipeline until 2026-07-28 — it existed only
behind `baru mpcc optimize`, so every circuit handed to FHE or ZKP carried the
source's redundancy. Now in `CircuitCompiler`:

| circuit | before | after |
|---|---|---|
| max-fold n=8 | 4372 ops | 28 (156×) |
| **MNIST demo (real AI circuit)** | **2365 gates** | **1764 (−25%)** |

Bit-identical scores; all 18 FHE differential tests pass (real TFHE, SEAL
CKKS).

### 2.4 Protocol assignment derives expert designs from a cost model

`baru mpcc assign` on a 32-wide dense + ReLU layer:

```
assignment       rounds      online     offline  conversions  predicted
all-arithmetic       12    10.5 KiB     1.0 MiB            0    1.45 ms
all-binary           26    17.0 KiB   104.1 MiB            0   88.83 ms
mixed (best)         11     1.1 KiB   199.1 KiB            3   714.3 µs
```

**2.03× over the better pure protocol**, moving `cmp-lt → binary` — the
classic hybrid-MPC split, *derived* rather than assumed. Elsewhere the same
machinery reproduces the Gazelle/Delphi FHE-linear/MPC-nonlinear split from
measured inputs alone.

---

## 3. What the evidence does *not* yet support

Three gaps, in order of how badly a reviewer will punish them.

### ~~Gap A — a dependent reduction is *slower* than hand-written~~ CLOSED 2026-07-28

`Rule::ReassociateReduction` rebalances a fold whose combining operator is a
**compound subgraph** (a maximum is `acc + (x>acc)·(x−acc)`, four
instructions) into a tournament: `2n−1` → `2⌈log₂ n⌉+1` rounds. Measured
15 → 7 at n=8, and 65 → 29 rounds (2.21×) end to end under Shamir,
instruction count unchanged, differentially verified.

**This is now the paper's cleanest single result**, because it is the
transformation we performed *by hand* on `gadgets::argmax` and
`max_reduce` — the compiler derives it for any associative operator, including
compound ones it has never seen, and refuses it for non-associative ones
(a subtraction fold is left alone). Automation is now at parity or better on
every axis measured.

### ~~Gap B — no end-to-end compiled AI workload~~ CLOSED 2026-07-28 (scope (a))

`baru bench-compiled-nn` evaluates a whole network —
`x(n) → dense(h) → ReLU → dense(c) → max`, **secret weights throughout** — two
ways over the **same** 3-party ABY3 transport, both checked against a
plaintext integer reference:

- `gadget` — Rust calling `gadgets::{dense, relu, max_reduce}`, the expert
  baseline (batched; matches hand-tuned code message for message).
- `compiled` — the same network as a **Baru expression**, through
  `CircuitCompiler` → `FlatOp` → `mpcc::exec`, a level-batched executor that
  issues one `mul_batch`/`lt_batch` per ASAP level.

The source says *nothing* about batching: it is scalar arithmetic, a mux-style
ReLU and a left-fold `max`. Every round saved comes from the schedule.

n=16, h=8, c=8, RTT 20 ms:

| route | ops | interactive calls | messages | wall-clock |
|---|---|---|---|---|
| `gadget` (expert) | — | 31 | 79 | 664 ms |
| `compiled` | 413 | 27 | 87 | 1224 ms |
| `compiled+reassoc` | 413 | **15** | **47** | 665 ms |

n=32, h=16, c=8: gadget 39 calls / 95 msgs; compiled+reassoc **15 / 47** —
**0.38× the interactive calls, 0.49× the messages, 0.98× the wall-clock.**

Three things to read off this, and the middle one is the honest part:

1. **The advantage grows with width.** The compiled route stays at 15
   interactive calls while the expert route grows 31 → 39, because ASAP
   batches every independent operation regardless of how many there are.
2. **Without `ReassociateReduction` the compiled route *loses*** (1.88×
   wall-clock) — the expert's `max_reduce` is a hand-written tournament. This
   is the ablation that shows which pass buys what, and it should be in the
   paper exactly as it is.
3. **Wall-clock is parity, communication is a win.** Both routes perform the
   same number of *comparison* layers (ReLU + tournament), and those dominate
   latency; the compiler's gain is in multiplies and sharing. Do not claim a
   latency win that is not there.

**Metric note.** `interactive calls` counts batched protocol invocations at
the `Backend` level (`TransportBackend::interactive_calls`), so it is exactly
comparable between routes however they were written. It is a proxy for rounds,
not a round count: one `lt_batch` runs several a2b rounds internally.

**Remaining scope limit (deliberate, per §6a).** This is the *MPC* path. The
FHE-linear/MPC-nonlinear hybrid is still Rust orchestration; the language does
not express scheme switching. Named as future work, not hidden.

### Gap C — the baseline is a straw man — PARTLY CLOSED

`bench-abstraction`'s `naive` route is *not* a publishable baseline, and it is
now clearly labelled as a control rather than a comparator.

`bench-compiled-nn` uses the honest baseline: the **batched** gadget library,
which matches hand-tuned code message for message. That is a competent
implementer's code, and the compiled route beats it.

Still open: the strongest possible baseline is `bench::cnn`, the hybrid
pipeline that absorbed a full day of expert optimization. Comparing against it
requires scope (b) (the language expressing scheme switching), so under scope
(a) the gadget library is the right baseline — say so explicitly in the paper
rather than leaving the reader to wonder.

---

## 4. The argument that makes the paper

The strongest evidence for automation is not a microbenchmark; it is what
happened to this repo's own experts.

The hand-written pipeline contained a scheme-switch import that issued **three
blocking rounds per activation** — 16 224 sequential round trips for one
convolution. It was **88% of end-to-end runtime**, it survived review and
benchmarking, and the published paper text had already concluded, in print,
that the bottleneck was somewhere else entirely. It was found only by
per-phase instrumentation.

An ASAP scheduler would never have emitted it. That is the paper's motivating
story, and it is backed by measurements rather than assertion:

> Experts serialize layers by accident. Compilers do not, because batching
> independent work is a property of the schedule rather than of the author's
> attention.

Three independent instances in one codebase (the import, `gadgets::relu` and
friends, `argmax`) make it a pattern, not an anecdote.

---

## 5. Phased plan

### Phase 0 — close Gap A (compiler reassociates dependent reductions)

Recognise an associative reduction whose operator is a repeated *subgraph
shape* and rebalance the chain into a tournament: `O(n)` → `⌈log₂ n⌉` rounds.
`Rule::Reassociate` exists but only handles chains of a single associative
primitive; a max-fold's operator is a compound (`acc + (x>acc)·(x−acc)`), so
it never fires.

- **Verify with** `mpcc::opt::equivalent` (the differential oracle that already
  accepts/rejects rewrites) and `bench-circuit`, whose max-fold row must go
  linear → logarithmic.
- **Trap:** tie-breaking. Reductions returning a *value* are order-insensitive;
  reductions returning an *index* (argmax) are not — `gadgets::argmax`
  preserves first-maximum only because the bracket pairs adjacent elements in
  index order. Pin exhaustively over small alphabets, not by sampling.
- **Deliverable:** the compiler matches the hand-written tournament, so §3 Gap A
  disappears and automation is at parity or better *everywhere*.

### ~~Phase 1 — close Gap B~~ DONE 2026-07-28 — see Gap B above

Delivered as `src/mpcc/exec.rs` (level-batched `FlatOp` → `Backend` executor)
and `baru bench-compiled-nn`. Original notes kept below.

### Phase 1 (original notes)

Decide the scope question in §6 first, then:

- Express the MedMNIST CNN (2 conv + 2 dense + ReLU/pool/argmax) in Baru source.
- Compile → `FlatOp` → `mpcc` → execute under an executable protocol
  (`mpcc run --protocol shamir|dealer` already works).
- Check bit-exactness against the same plaintext integer reference the
  hand-written pipeline uses. **This invariant is what makes the rest safe.**
- **Risk to watch:** circuit size. A 512-wide layer compiles fine (1536 ops),
  but a 5408-activation conv layer is a different scale. Measure before
  committing to the full model; fall back to one conv block if needed and say so.

### ~~Phase 2 — the ablation~~ DONE 2026-07-28

`baru bench-compiled-nn` reports a ladder in which each rung adds exactly one
compiler pass, against the expert gadget baseline. n=8, h=4, c=4, RTT 0:

| route | ops | interactive | messages | bytes | wall-clock |
|---|---|---|---|---|---|
| `gadget` (expert) | — | 20 | 53 | 56.0 k | 44 ms |
| `compiled: serial` | 7419 | 4280 | 5859 | 1798 k | 6839 ms |
| `  + ASAP batching` | 7419 | 15 | 47 | 1682 k | 111 ms |
| `  + CSE` | 109 | 15 | 47 | 55.7 k | 65 ms |
| `  + reassociation` | 109 | **12** | **37** | 55.5 k | 49 ms |

Contribution of each pass:

| pass | ops | interactive | bytes |
|---|---|---|---|
| ASAP batching | 1.00× | **285.33×** | 1.07× |
| CSE | **68.06×** | 1.00× | **30.18×** |
| reassociation | 1.00× | 1.25× | 1.00× |

**The passes act on different axes and none substitutes for another.**
Batching collapses rounds and leaves circuit size untouched; CSE collapses
circuit size and bandwidth and changes no round count; reassociation collapses
depth. Report the ladder, never a single number — a one-column summary makes
CSE look worthless at 1.00× when it is a 30× bandwidth win.

The `serial` row is the honest floor: an executor written the obvious way,
walking the circuit and calling the backend per operation. It is 285× the
interactive calls of the same circuit under ASAP levelling, on identical
bytes — the round-structure finding again, now inside the compiler's own
executor.

### Phase 2 (original notes)

The headline table. Same network, same backends, same transport, three
implementations:

| route | what it is |
|---|---|
| `expert` | `bench::cnn`, hand-tuned over a full day |
| `compiled` | Baru source → `CircuitCompiler` → `mpcc` |
| `compiled+opt` | the same with backend-aware rewrites enabled |

Metrics: **rounds, bytes, wall-clock at several RTTs, gate count**, plus
bit-exactness of every route against the plaintext reference. Report parity
honestly where it is parity.

**Report the ablation**, because it is what makes the mechanism legible: what
does auto-batching contribute, what does CSE contribute, what does
reassociation contribute, what does assignment contribute.

### ~~Phase 3 — the novel claim~~ DONE 2026-07-28

`baru bench-backend-aware` builds the **cross-application matrix**: optimize
one program once per backend, then price *every* resulting circuit under
*every* backend. Divergence alone would be unconvincing (circuits could differ
in ways nobody pays for); the matrix shows the diagonal winning its own
column, and the command **fails** if it ever does not.

WAN, 3 parties, n=8, every optimized circuit differentially verified:

```
max-fold        as shamir  as semi-bin     as bmr    rules kept
  shamir       1453.010ms*  3761.043ms   281.564ms   reassociate-reduction
  semi-bin     1453.010ms   3761.043ms*  281.564ms   reassociate-reduction
  bmr          3253.010ms   8761.043ms   281.564ms*  -
```

**This single row is the paper's figure.** The *same* rewrite —
`reassociate-reduction` — is a **2.24× win under Shamir and worth exactly
nothing under BMR**, so the optimizer keeps it for one and declines it for the
other. Running BMR's circuit under Shamir costs 2.24×. Depth is the round
count for an interactive protocol and free for a constant-round garbled one;
the search does not need to be told this, `objective` prices it.

Across the three workloads: **2, 2 and 3 distinct circuits from one program**,
with worst wrong-circuit penalties of 1.66×, 2.33× and 1.81×. `dot-const`
diverges fully — three backends, three different circuits — because strength
reduction pays per AND gate for binary/garbled backends and is a loss for an
arithmetic one where multiplying by a constant is already local.

Two details worth keeping in the paper:

- **Declining is a result.** Where a row keeps no rules, that backend wanted
  the original circuit. An optimizer that always rewrites is not being
  selective, it is being busy.
- **Op count is a bad divergence metric** — reassociation changes shape while
  preserving count — so distinct circuits are counted by content fingerprint.

### Phase 3 (original notes)

Thesis S's last clause is the part nobody else does: *the same program is
optimized differently depending on which backend it targets.* MP-SPDZ
optimizes rounds and is blind to FHE; FHE compilers optimize multiplicative
depth and are blind to MPC.

Baru already has the ingredients — `mpcc::opt` accepts or rejects each rewrite
*per protocol*, and there is prior evidence that a rewrite worth ~32× under
Yao is worthless under ABY3, with `Reassociate` inverting by protocol family.

**Deliverable:** for one AI workload, show the *emitted circuits differ per
backend* and that using the wrong backend's circuit costs measurably. That is
the figure the paper is built around.

### Phase 4 — assembly

- Positioning against HyCC → Silph (ILP) → SING (GNN) for assignment, and CirC
  for one-IR-many-backends. See [`related-work.md`](related-work.md).
- **Baru's defensible difference:** the others assign *among MPC protocols*.
  Baru's cost model spans **MPC, FHE and ZKP**, so it can ask "should this be
  an FHE program at all?" — a question the others cannot pose.
- Honest scope: semi-honest, honest-majority; MedMNIST is downsampled;
  absolute times are single-threaded and unoptimized NTT.

---

## 6. The scope decision — **(a) chosen 2026-07-28**

To claim "a language for private AI", the language must compile the thing.
Today it compiles *expressions* to circuits; the FHE/MPC hybrid orchestration
lives in Rust. Two options:

**(a) Scope to the compiled MPC path.** Compile the whole network to MPC,
compare against a hand-written MPC implementation. Achievable now, honest,
and Phase 1 is small. The paper is then about *circuit optimization*, and the
switch-as-bootstrap hybrid stays in the other paper.

**(b) Extend the language to express scheme switching.** The full "language
for private AI" claim, and it makes the backend-aware story much stronger
(the compiler chooses where to switch). Substantially more work, and it
overlaps the other paper's subject matter.

**Chosen: (a), with (b) named as future work.** Phase 1 is done under this
scope (`bench-compiled-nn`). The paper is about *circuit optimization for
private AI*; the switch-as-bootstrap hybrid stays in the other paper, and the
language does not claim to express scheme switching.

---

## 7. Recommended framing

Three framings compete for the same material:

| framing | strength | risk |
|---|---|---|
| A. "zero abstraction overhead" | easy to prove | *"then why compile?"* |
| B. "automatic optimization beats hand-tuning" | compelling, we have the story | needs Phase 1+2 to land |
| C. "backend-aware circuit optimization" | genuinely novel; nobody spans MPC+FHE+ZKP | needs Phase 3 |

**Recommendation: C is the thesis, B is the headline experiment, A is the
sanity floor reported first.** That is one coherent paper: *the same private-AI
program should become a different circuit depending on the backend, a compiler
can decide that automatically, and doing so beats expert hand-tuning — which we
demonstrate against our own experts, who serialized a layer without noticing.*

---

## 8. What to do next, concretely

1. **Phase 0** — reassociation rule. Self-contained, has an oracle, closes the
   one case where we lose.
2. **Decide §6** — (a) or (b). Everything downstream depends on it.
3. **Phase 1** — one real compiled network, bit-exact.
4. **Phase 2** — the expert-vs-compiled table with ablation.
5. **Phase 3** — per-backend circuit divergence.

Do **not** start writing prose before Phase 2 produces a number. The
switch-as-bootstrap paper twice committed conclusions to print that later
measurement contradicted (see `private-inference-performance.md` §2); the
discipline that fixed it was measure-then-write, and it applies here.
