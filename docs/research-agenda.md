# Research agenda: private AI for healthcare, and what Baru has to become

This document exists to stop a good instinct from becoming a bad repository.
The instinct is right: healthcare is the application where privacy-preserving
computation stops being a demo, and Baru already has an unusual amount of the
stack. The risk is that "criptolandia" — a repo with every primitive in it —
reads to a program committee as a library, not a result.

So: one thesis, one workflow, and a rule for admitting new primitives.

---

## 1. Two papers, not one

Baru already has a sharp cryptographic thesis in
[paper-outline.md](paper-outline.md): **switch-as-bootstrap** (the MPC round
trip a nonlinearity already needs also refreshes the FHE ciphertext, so depth is
free). That is a crypto paper. It should stay a crypto paper, and healthcare is
its evaluation domain, not its subject.

The work in `src/mpcc/` is a *different* claim, and it wants a systems venue:

> **Thesis (S).** Which privacy technology is cheapest is a property of the
> *program*, not of the technology — and it can be predicted from the program's
> compiled form, before running it. Baru compiles one source language to MPC
> (many protocols), FHE (several schemes) and ZKP, models each backend's cost
> analytically, validates the model against measured runs, and then *optimizes
> the circuit differently depending on which backend it is headed for*.

The last clause is the part nobody else does. MP-SPDZ optimizes rounds and is
blind to FHE. FHE compilers optimize multiplicative depth and are blind to MPC.
Neither asks "should this even be an FHE program?" — and neither can, because
neither has a cost model spanning the alternatives.

**Healthcare's role in (S)** is to supply a workload where the answer differs
per stage. That is the whole argument: if one backend won everywhere, there
would be no paper.

---

## 2. The workflow: one realistic healthcare pipeline

A single end-to-end scenario, chosen so each stage stresses a different layer.
Two hospitals and a model owner; nobody trusts the others with raw data.

| # | Stage | The privacy problem | Layer in Baru | Status |
|---|---|---|---|---|
| 1 | *"How many patients aged 40–60 with HbA1c > 7?"* | cohort selection over records at rest | BaruDB (DET + ORE) | ✅ measured — `baru bench-cohort` |
| 2 | *Fetch record `r` for review* | the **access pattern** leaks who is interesting | **PIR** | ⚠️ built + measured (`baru bench-pir`), and it **loses to downloading the database** — §3a |
| 3 | *Which patients appear at both hospitals?* | linkage without disclosure | **PSI** (from OT) | 🔨 OT done + measured (`baru bench-ot`, §3b); PSI still to build |
| 4 | *Classify this scan* | image + model both private | MPC / FHE / hybrid | ✅ measured on real biomedical images — `baru bench-medmnist` (§3) |
| 5 | *Train on the union of both cohorts* | cross-institution learning | MPC (Shamir / SPDZ2k) | partial (`bench/train`) |
| 6 | *Iterate over a private index during training* | data-dependent addressing | **ORAM** | 🔨 |
| 7 | *Prove the approved model was the one applied* | regulatory audit | ZKP (Binius64) | ✅-ish (prover is simulated) |

Every stage is a real requirement in a hospital deployment, and every stage has
a *different* cheapest backend. Stage 4 is where the existing MNIST work lives;
stages 1–3 are cheap to build and rarely benchmarked together with 4–7, which is
exactly the gap.

### The admission rule for new primitives

**A primitive enters Baru when it is the cheapest way to serve a stage of the
workflow, and it enters with a cost model entry and a benchmark case.** PIR,
ORAM, OT and PSI all pass on that test. A primitive that does not serve a stage
waits, however interesting it is. This is the difference between criptolandia
being an asset (a stack whose parts are compared under one cost model) and a
liability (a museum).

### First cross-layer result (2026-07-27)

`baru bench-cohort` runs identical cohort queries against a plaintext and an
encrypted BaruDB database, checks the counts agree, and times both. Over 200 /
1000 / 5000 synthetic patients and four predicate shapes:

| | encrypted / plaintext |
|---|---|
| **query** (equality, range, conjunction, empty) | **≈ 1.0×** |
| **ingest** (6200 rows) | **5.05×** |

ORE-indexed selection costs about the same encrypted as plaintext, because the
comparison runs against tokens either way; encryption's price at this stage is
in ingest and in one-time Argon2id key derivation (~17 ms per process).

Two measurement corrections were needed to get there, and both matter more than
the numbers. Key derivation is ~99% of BaruDB's reported crypto time and is
per-*process*, not per-query. And driving BaruDB over a pipe costs ~14 ms of
`fork`/link/open per invocation — larger than the query itself at these sizes,
so a naive ratio of wall clocks would have been a ratio of `fork()` calls and
would have read ≈1.04× for entirely the wrong reason. The harness reports
`io_us + marginal_crypto_us` and states both exclusions in its output.

Set that ≈1.0× against stage 4's 10³–10⁶× and the agenda's thesis is visible in
a single table: **the cost of privacy is wildly non-uniform across a pipeline**,
so choosing one technology for all of it wastes most of the budget. That is the
argument the compiler exists to make quantitative.

The plaintext mode is also the strongest correctness test BaruDB's encrypted
query path has — every run is a differential check, and a disagreement fails the
command.

---

## 3. MNIST is not healthcare — done 2026-07-28

The image work used MNIST, and a reviewer would say, correctly, that handwritten
digits are not a medical claim.

**Fix: MedMNIST v2** (<https://medmnist.com>, CC BY 4.0). Twelve 2D biomedical
datasets — PathMNIST (colon pathology), ChestMNIST (chest X-ray, multi-label),
DermaMNIST (skin lesions), OCTMNIST, PneumoniaMNIST, BloodMNIST… — distributed
at **28×28, exactly MNIST's shape**, so the existing 784-pixel pipeline runs on
them, and there is a 64×64/128×128/224×224 variant to scale into later.

**Done**, on both models the pipeline has:

* **The CNN** — `baru bench-medmnist [--dataset pneumonia|organa|all]`
  (`src/bench/medmnist.rs`), the same two-conv hybrid `bench-mnist` runs:
  BFV ct×ct conv/dense, ABY3 nonlinearities, zero bootstraps.
* **The linear discriminator** — `cargo run --release --example mnist --
  --pneumonia [--mpc] [--tfhe] [--ckks]`, the 2380-gate circuit across every
  FHE scheme. Digits stay available as the default, unchanged.
* Weights are trained and quantised by `tools/train_medmnist.py` — kept in
  `tools/`, not `scratchpad/`, because that is how the original MNIST trainer
  was lost.

| | pneumonia (X-ray, 2 classes) | organa (CT, 11 classes) |
|---|---|---|
| CNN integer accuracy, full test set | **84.78%** (624) | **81.18%** (17 778) |
| float accuracy, same net | 84.78% | 81.03% |
| majority-class baseline | 62.50% | 18.48% |
| published ResNet-18 (28×28) | 0.854 | 0.935 |
| encrypted == plaintext diagnosis | 3/3 | 3/3 |
| FHE bootstraps | 0 | 0 |
| linear model, full test set | 84.13% | — (2-class only) |

Three things worth carrying into the paper.

**Quantisation is free, and so is encryption.** The integer nets match their
float parents to within 0.2 points, and the encrypted pipeline reproduces the
plaintext diagnosis bit-exactly — so neither the fixed-point scheme the FHE
layer demands nor the encryption itself costs accuracy. The price is entirely
wall-clock (~33 s per encrypted image).

**The model, not the crypto, is the accuracy ceiling.** This is a 2-conv
classifier sized to fit the depth-1 noise budget; it matches published
ResNet-18 on pneumonia and gives up 12 points on organa. Saying so is stronger
than quietly reporting only the dataset where a small model wins.

**Two methodological traps, both hit and both fixed.** The quantisation gain is
chosen on the *validation* split, never on test. And the gain is not free in
either direction: it multiplies activations layer over layer, so too large
overflows the plaintext modulus, while too small starves the output layer's
weights to zero and the classifier silently collapses onto its majority class
(62.50% — which is exactly what the first run produced). The linear model has
the mirror-image constraint: chest X-rays are dense where digits are mostly
zero, so the Q16.16 accumulator genuinely threatens the demo's 32-bit MPC
shares, and the trainer picks the gain against a *measured* accumulator bound
rather than by eye.

Honest caveat to state in the paper: MedMNIST is pre-processed and small;
clinical-grade imaging is 512³ volumes. The claim is about *cost structure*, and
the scaling argument has to be made analytically — which is what the cost model
is for.

---

## 3a. Stage 2: PIR is built, measured, and not yet worth using

Done 2026-07-28: `src/pir/` implements the textbook single-server scheme over
the BFV already in the tree, and `baru bench-pir` measures it. The database is
one plaintext polynomial per shard, the query is an encrypted monomial, and
because `xⁿ = −1` a single plaintext-ciphertext multiply drops the wanted record
into the low coefficients. One round, `O(1)` query, no server key, and **zero
multiplicative depth** — so it composes with the depth-1 pipeline without a
bootstrap.

And it is the wrong tool, which is worth reporting rather than hiding. Measured
at n=4096, q=2^100, t=2^20, 256-byte records:

| records | shards | query | answer | whole-db download | PIR ÷ whole-db |
|---|---|---|---|---|---|
| 32 | 1 | 104 KB | 104 KB | 8 KB | 0.04× |
| 256 | 8 | 104 KB | 832 KB | 64 KB | 0.07× |
| 2048 | 64 | 104 KB | 6.5 MB | 512 KB | 0.08× |

Retrieval is correct at every size (checked against the plaintext database), and
the ratio is converging to `1/13`, not improving with scale. The reason is
structural:

```text
expansion = 2·⌈log₂q/8⌉ / ⌊log₂t/8⌋ = 13×
```

One response ciphertext costs 13× the *entire plaintext content* of the shard
that produced it, and the scheme returns one response per shard. So its traffic
is ~13× a whole-database download **regardless of N** — there is no crossover to
find, and no parameter choice inside this construction finds one, because the
expansion factor is a property of the ciphertext, not of the database.

**By the admission rule in this document, PIR has therefore not earned stage 2.**
A primitive enters when it is the cheapest way to serve a stage; here the
cheapest way to hide the access pattern is still to download the shard. Keeping
the code is justified by what it *does* establish — a correct implementation, a
measured baseline, and a specific next step — but the workflow table says ⚠️,
not ✅, and the paper should say so too.

The specific next step is **recursion with server-side query expansion**
(SealPIR): fold the shard dimension on the server so the response is `O(1)`
ciphertexts while `N` grows. That needs Galois automorphisms and a substitution
operation `src/fhe/bfv.rs` does not have, and it is real work rather than the
"few hundred lines" this document previously estimated for the whole item.

There is a second, cheaper lever worth measuring first: the expansion factor is
`2·log₂q / log₂t`, so raising `t` shrinks it directly. At `t = 2^40` it would be
5×, still above 1. That confirms expansion alone cannot rescue the scheme and
recursion is doing the essential work — which is the sort of thing the cost model
should be used to check *before* writing the automorphisms.

---

## 3b. Stage 3, part one: oblivious transfer is done and measured

Done 2026-07-28: `src/ot/` implements Chou–Orlandi base OT over Ristretto255 and
the IKNP extension on top, with `baru bench-ot` measuring both. OT was put ahead
of the other criptolandia items in §6 because it pays twice — stage 3's PSI is
built on it, *and* Semi2k's and MASCOT's triples come from it, so it is also
roadmap P1/P3 infrastructure.

| protocol | OTs | µs/OT | bytes/OT | rounds |
|---|---|---|---|---|
| base | 128 | 126.35 | 64.2 | 2 |
| extension | 128 | 127.99 | 48.2 | 3 |
| base | 8192 | 123.92 | 64.0 | 2 |
| **extension** | **8192** | **4.67** | **16.5** | 3 |

Base OT is flat in the batch size, as it must be — each transfer is its own pair
of scalar multiplications. The extension starts level with it at 128 OTs (that
batch *is* the `κ = 128` base OTs it must pay for first) and is **27× cheaper per
OT and 4× smaller by 8192**. Bytes per OT converge to `κ/8 = 16` from above
(48 → 20 → 16.5), which is the protocol's analytic online cost of one column bit
per output OT — the measured curve landing exactly on that floor is what says the
byte accounting is the protocol's rather than an artifact of the harness.

Two honest limits. **Both layers are semi-honest**, and for IKNP that is a
concrete gap rather than a caveat: nothing forces the receiver to use the same
choice vector in every column, and an inconsistent one leaks the sender's `s`
bit by bit. KOS15's correlation check is what closes it, and until that exists
this module cannot instantiate anything maliciously secure. Second, the per-OT
*time* has a floor because row extraction from the column-major matrix is `O(κ)`
bit lookups rather than a blocked transpose — the shape of the curve is the
claim here, not the absolute throughput.

**The "pays twice" half is now cashed in (same day).** Semi2k's, SPDZ2k's and
MASCOT's offline parameters were literature constants; `mpcc::ot_cost` now
derives their offline volume from the measured OT cost above times a
protocol-level OT count, and `baru mpcc compare` labels those rows "derived from
MEASURED OT cost" rather than disclaiming an estimate. It moved a ranking:
LowGear's BFV preprocessing moves ~384 B per triple against 9.2 / 24.4 / 48.8 KiB
for Semi2k / SPDZ2k / MASCOT, a gap the old constants understated by an order of
magnitude and exactly the bandwidth-for-CPU trade LowGear exists to make. Treat
it as an upper bound on the OT side (vector OLE and batched sacrifice both beat
naive Gilboa), and note that the CPU half of LowGear's trade is still missing, so
its total is still flattering.

**PSI is still to build**, and the interesting question there is which
construction — a DH-based PSI needs no OT at all and is the practical baseline at
moderate set sizes, so the cost model should be asked which wins before assuming
the OT-based one does.

---

## 4. Backend-aware circuit optimization — the novel contribution

Implemented as a first slice in [`src/mpcc/opt.rs`](../src/mpcc/opt.rs).

The observation that makes it a result rather than an engineering detail:
**the same rewrite can be a large win under one backend and a loss under
another.** Two examples the implementation demonstrates:

* **Strength reduction** (`x * 5` → `(x << 2) + x`). Under Yao, a `k×k`
  schoolbook multiplier is `k²` AND gates and an adder is `k` — a ~64× saving at
  `k = 64`. Under ABY3, multiplying by a *public* constant is already a local
  scalar multiply costing nothing, so the rewrite adds an instruction and buys
  nothing. A backend-blind optimizer must choose one behaviour and be wrong half
  the time.
* **Reassociation** (done 2026-07-28) rebalances a chain of one associative
  operator into a tree, cutting dependence depth from `n-1` to `⌈log₂ n⌉`. Under
  ASAP scheduling depth *is* the online round count, so this is the sharpest
  form of the claim: the instruction count and the AND-gate count come out
  **identical**, so every cost difference it produces is purely rounds.

  What makes it more than one more rule is that *which operator pays depends on
  the protocol*, and it inverts. Measured on WAN, 8-element chains:

  | | arithmetic sharing (ABY3, Shamir, SPDZ2k, …) | BMR (constant-round) | Semi-Bin / Tinier (binary, not constant-round) |
  |---|---|---|---|
  | `*` chain | **1.8–2.0× — kept** | 1.00× — dropped | **2.3× — kept** |
  | `+` chain | 1.00× — dropped | 1.00× — dropped | **2.3× — kept** |

  Multiplication costs a round under secret sharing, so rebalancing pays;
  addition is *local* there, so it cannot. In a binary protocol the adder is a
  real AND-depth, so the addition chain is exactly where the win moves to. And
  a constant-round protocol is indifferent to depth by construction, so it
  declines both. Three different answers to one rewrite, on two programs that
  differ only in an operator — which is the thing a backend-blind optimizer has
  no way to express.

**Per-site search** (also 2026-07-28) removed what had been the pass's main
weakness. The two rules whose sign depends on their operand are now priced site
by site instead of as one bundle. It mattered: on `(+ (* x 4) (* y 100))` a
whole-circuit search has to take both rewrites or neither, and an arithmetic
backend took neither — losing the free `x * 4 → x << 2` for the sake of
declining `y * 100`. Now ABY3 and Shamir take the first site and decline the
second, while the binary protocols take both.

It also sharpens the claim, because backends now diverge *within* a rule and not
merely over which rules they accept. On a 64-element constant dot product,
Shamir accepts strength reduction at **5 sites** (the exact powers of two) and
Semi-Bin at all **42** — one program, one rule, two different circuits, and the
subset relation is the cost model's doing rather than a heuristic's. The price
is search time (every site is a full lower → schedule → estimate;
`sites_considered` is reported so coverage is visible, and a 2351-instruction
circuit across all 11 backends takes ~1 minute). Still greedy and still
site-independent, so a pair of rewrites that only pays *together* is missed.

  Getting the reassociation result right required **fixing the cost model**: it had been charging
  chained *local* arithmetic the maximum AND-depth over the program rather than
  the longest path, so a chain of `n` adders was priced as one. That
  under-priced deep local arithmetic — fixed-point rescales and adder trees,
  i.e. precisely what decides MPC-vs-GC for the NN workloads — by up to `n×`
  (7× on the chain above). The model now accumulates depth along the dependence
  DAG.

So the pass takes the target protocol as a *parameter*, uses
[`mpcc::cost`](../src/mpcc/cost.rs) as the objective function, and keeps a
rewrite only when the predicted cost for that backend improves. Correctness is
not argued, it is *checked*: every candidate is differentially tested against
the original circuit on random inputs using the existing `FlatOp` evaluator, so
a rewrite that changes semantics cannot survive.

That gives the paper a clean table: one program, N backends, and the observation
that the optimized circuits are *different circuits*.

---

## 5. On RAG vs. a Claude skill for circuit optimization

Recommendation: **build the skill, skip the RAG for now.**

Circuit optimization is a search-and-verify problem, not a recall problem.
Retrieval over crypto papers answers "what is the best known AND-gate count for
a comparator?" — useful, occasionally, and bounded. It cannot tell you whether a
rewrite is correct, and it cannot search the space.

What actually pays:

1. **A rewrite engine with a cost-guided search and a differential oracle** —
   §4. This is where the wins are, and it is mechanical.
2. **A Claude skill that knows Baru's IR, gadget catalogue and cost model**, used
   as a *generator* of candidate gadgets and lowerings. The skill proposes; the
   differential test corpus disposes. That division of labour is what makes it
   trustworthy — the model never has to be right, only productive, because
   nothing it proposes lands without passing the oracle.
3. Literature retrieval as a *thin* part of (2), if at all: a curated file of
   known-optimal gadget costs (comparator, adder, multiplier, argmax) beats a
   RAG pipeline at this scale, and can be checked into the repo.

The skill is worth writing after `opt.rs` has enough rules that "add a rule" is
a well-defined shape to imitate. Writing it first would mean asking a model to
generate into a target that does not exist yet.

---

## 6. Sequence

Ordered by value per unit of work, with the dependency structure respected.

1. ~~**MedMNIST swap** (§3)~~ — **done 2026-07-28.** The artifact is now a
   healthcare artifact: private pneumonia and organ diagnosis, plaintext-exact
   under encryption, no new crypto. Next dataset in this direction would be a
   *multi-channel* one (PathMNIST, DermaMNIST), which needs the conv packing to
   split RGB channels across ciphertexts — a packing question, not a
   construction one.
2. **Backend-aware optimizer** (§4). Started. Extend the rule set; add
   reassociation-for-depth; report per-backend optimized circuits.
3. **P2 of the MPC roadmap** — random bits and edaBits, so comparisons *run* and
   not merely cost. Everything in stages 4–6 is comparison-heavy; this is the
   binding constraint on the whole agenda.
4. **Cohort queries in BaruDB** (stage 1). Mostly a schema and a query-shape
   question; BaruDB already has ORE.
5. ~~**PIR** (stage 2), textbook single-server over the existing BFV~~ — **done
   2026-07-28, with a negative result** (§3a): correct, but ~13× *worse* than
   downloading the database at every size, because ciphertext expansion exceeds
   the database-to-response ratio. The follow-on is SealPIR-style recursion with
   server-side query expansion (Galois automorphisms + substitution in
   `src/fhe/bfv.rs`) — and the estimate for *that* is not a few hundred lines.
   Worth doing only if stage 2 stays in the story; the cost model now says the
   dull answer wins today.
6. **OT + PSI** (stage 3). **OT done 2026-07-28** (§3b) — base + IKNP, measured,
   semi-honest. Remaining: (a) feed the measured OT cost into the Semi2k/MASCOT
   offline parameters, which is the "pays twice" payoff and roadmap P1; (b) PSI,
   deciding by cost model rather than by assumption whether the OT-based
   construction or a DH-based one wins at the set sizes this workflow has;
   (c) KOS-style consistency checks if malicious security is wanted.
7. **ORAM** (stage 6). Path ORAM over the storage layer. Largest of these; do it
   last, and only if stage 6 is still in the story by then.
8. **Circuit-optimization skill** (§5).

Items 5–7 are the criptolandia items. Note that each one is *also* infrastructure
for the compiler: PIR reuses BFV, OT unlocks two MPC protocols, ORAM gives the
cost model a data-dependent-addressing entry. That is the test they pass.

## 7. What to cut, or at least defer

* **Private RAG / encrypted LLM inference.** Fashionable, enormous, and a
  different paper. The non-private RAG for circuit construction (§5) is a *tool*,
  not a contribution — do not let it become a claimed one.
* **Federated learning.** Adjacent, crowded, and orthogonal to the cost-model
  thesis.
* **A real 512³ imaging pipeline.** State the scaling analytically instead.
* **Differential privacy.** Composes with all of this and belongs in future work;
  adding it now doubles the evaluation matrix for no new claim.
