# Research agenda: private AI for healthcare, and what Baru has to become

> Performance state, method and next steps for the private-inference
> pipeline: [`private-inference-performance.md`](private-inference-performance.md).

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
| 3 | *Which patients appear at both hospitals?* | linkage without disclosure | **PSI** (DH today, OT next) | ✅ measured — `baru bench-psi`; OT also done (§3b) |
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
  --pneumonia [--mpc] [--tfhe] [--ckks]`, the 1669-gate circuit across every
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
wall-clock — **2.90 s per encrypted diagnosis online** (plus ~2.0 s of
query-independent preprocessing) as of 2026-07-28; it was ~33 s when this
section was written. See
[private-inference-performance.md](private-inference-performance.md) for how,
and for what to optimise next.

**The model, not the crypto, is the accuracy ceiling.** This is a 2-conv
classifier sized to the envelope the pipeline allows: at most three convs
before 3×3 valid convolution exhausts a 28×28 input, at most 12 channels out of
the first conv before the Cheetah packing passes `n = 4096` at the second
conv's 13×13 input, and activations inside the depth-1 noise budget. It matches
published ResNet-18 on pneumonia and gives up 12 points on organa. Saying so is
stronger than quietly reporting only the dataset where a small model wins — and
worth separating: the width limit is a *packing* question that relaxes at
`n = 8192`, not a limit of the no-bootstrap construction.

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
LowGear's SHE preprocessing moves ~384 B per triple against 9.2 / 24.4 / 48.8 KiB
for Semi2k / SPDZ2k / MASCOT, a gap the old constants understated by an order of
magnitude. Treat it as an upper bound on the OT side (vector OLE and batched
sacrifice both beat naive Gilboa).

**Both halves are now measured** (`baru bench-triples`, `src/mpcc/lowgear.rs`),
and one of them came out the other way round. Running the protocol — BGV over
CRT slots, one ciphertext×plaintext multiply per cross term — gives **192 B per
triple** at its native ~61-bit field,
2× better than the 384 B estimate. But LowGear is also **~28× faster per
triple than the OT route**, not slower. The "bandwidth-for-CPU trade LowGear exists to
make" was never measured and does not appear at these parameters: SHE wins on
bytes *and* on CPU, so the OT route's remaining advantage is neither, it is
that it works over `Z_2^k` directly and needs no noise-flooding budget.

**And it changes an answer.** §4's crossover — a wide encrypted layer moves to
FHE under a dishonest majority — was measured against Semi2k, i.e. against **OT**
preprocessing. LowGear now runs the same layer itself (its field widened to
~2^61 on 2026-07-30, so it can bit-decompose the ReLU), and **the hybrid does
not beat it at any width measured**: 606 ms vs 692 at width 64, 678 vs 734 at
4096, against a trusted-dealer lower bound of 601–611. The mechanism survives —
preprocessing volume still decides it — but the boundary belongs to the
*baseline's preprocessing*, not to MPC in general, and against the best
preprocessing there is no boundary in range. The claim has to name its baseline.
See `paper-outline` E25.

Two caveats bound the claim. The implemented engine is **semi-honest** — no
MACs, no ZKPoPK, so it is the LowGear analogue of Semi2k rather than of SPDZ.
And the ~17-bit native field is forced by the flooding budget at this crate's
`u128`-capped `q ≤ 2^127`, where Overdrive runs 250–350-bit `q`; the 64-bit
figure comes from CRT over 4 such primes.

### PSI: built the baseline, then asked the cost model which construction to keep

This document wrote stage 3 as "PSI (**from OT**)". That was an assumption, and
the thesis of this whole repo is that the cost model decides. So `src/psi/`
implements the construction that needs *no* OT — DH-PSI over Ristretto255,
semi-honest, one-sided — and `baru bench-psi` prices the OT route from the
measured OT cost above rather than from intuition.

| | hides non-shared items? | bytes/item | µs/item |
|---|---|---|---|
| naive hash exchange | **no** | 32 | **0.43** |
| DH-PSI (measured) | yes | 96 | 165 |
| OT-based (projected) | yes | ~58 | ~1 |

DH-PSI is flat in both, as it must be: three flights of compressed points and two
scalar multiplications per item. The insecure baseline is **~380× faster**, which
is worth stating because it explains why hash-exchange keeps getting shipped —
and what the 380× buys is an offline dictionary attack that recovers the sender's
*entire set*, not just the overlap. Patient identifiers are exactly the
low-entropy inputs that attack needs.

**The assumption was right, and now for a stated reason.** An OT-based PSI should
win on bytes by only ~1.7×, but on *time* by two orders of magnitude, because
DH-PSI's cost is two scalar multiplications per item and no engineering removes
them. The deciding axis is **compute, not bandwidth** — which is not what the
"from OT" shorthand suggested, and is the kind of thing worth knowing before
building the harder protocol.

Two honest qualifications. The ~58 B/item is a *projection* (measured per-OT cost
× a literature 448-bit OPRF code width), not a measured PSI, so it belongs in a
"what to build next" argument and not a results table. And DH-PSI stays the right
choice for small sets, where 165 µs × n is irrelevant and a construction needing
no OT infrastructure is worth more than a constant factor.

---

## 3c. Next: the database makes a *composed* image-recognition result possible

Agreed 2026-07-28. Every stage measured so far is measured *alone*: stage 1 is a
cohort query, stage 4 is an encrypted classification, and the striking table in §2
sets ≈1.0× against 10³–10⁶×. That is a good argument and it is also the easy one.
The next contribution is to **compose two stages**, which is what a reviewer asks
for and what no single-technology system can do:

> **Cohort-conditioned private evaluation.** "What is this model's accuracy on
> patients aged 40–60 with HbA1c > 7?" — answered without decrypting the cohort,
> the images, or the per-patient predictions.

### Why this is worth doing rather than more single-stage benchmarks

* **It is a real requirement, not a demo.** Regulators and clinicians do not accept
  one aggregate accuracy; they require performance broken down by subgroup, and
  subgroup membership is exactly the sensitive attribute. Today that analysis is
  done by putting demographics, labels and predictions in the clear together.
* **It needs the whole stack at once**: BaruDB's ORE/DET cohort selection (stage 1)
  to pick the subgroup, the encrypted CNN (stage 4) to produce predictions, and
  the result store to accumulate runs. Each already works; nothing new has to be
  invented, which is why this is the *next* step and not a research project.
* **It produces a number nobody else can produce**: per-cohort accuracy with cost
  attributed per stage, under one cost model. That is Thesis (S) applied to a
  pipeline rather than to a single program.

### Done 2026-07-28 — `baru bench-cohort-eval`

```
predicate            : age>=40 and age<=60 and hba1c>7.00  (over ENCRYPTED attributes)
records / cohort     : 300 / 66  (22.0% selected)
cohorts agree        : yes  (encrypted == plaintext BaruDB == local filter)

accuracy on cohort   : 52/66 = 78.79%
accuracy overall     : 85.00%
cohort std. error    : ±5.0 points on 66 patients — the gap is 1.2 SE, i.e. noise

cost, by stage:
  stage 1 ingest     : 125.5 ms for 300 records
  stage 1 selection  : 14.9 ms
  stage 4 inference  : 5772 ms per image, encrypted  (2/2 matched plaintext, 0 bootstraps)
```

**Selecting the entire cohort costs ~390× less than classifying one member of
it.** The §2 headline — that the price of privacy is wildly non-uniform across a
pipeline — is now visible *inside a single query* rather than only across separate
benchmarks. That is the composed result the agenda was missing.

> Re-measured 2026-07-28 after the scheme-switch import was batched. The ratio
> was ~1900× when one encrypted inference cost 32.6 s; inference is now 5.8 s,
> so the *spread narrowed by 5×* and the qualitative claim survived it. Worth
> noting as a hazard: this ratio is a claim about two stages, and optimising
> either one silently invalidates it. The command recomputes it, so quote the
> command's output, never a remembered number.

The cohort is computed **three ways** — encrypted BaruDB, plaintext BaruDB, and a
local filter — and all three must agree or the command fails, so this is a
differential test of the encrypted query path as well as a measurement. The
encrypted classification matched the plaintext prediction with zero bootstraps.

Two deliberate design points:

* It uses `cohort_ids`, not `cohort_count`, and the threat model is what licenses
  that: the **hospital** selects out of its own records (learning which of its own
  patients matched discloses nothing), the **storage provider** sees only
  ciphertext and ORE tokens, and the **model owner** learns only the cohort size.
  `cohort_ids` is documented as inappropriate wherever the querier is not the data
  owner.
* The accuracy gap is reported **with its standard error**. ±5.0 points on 66
  patients makes the 6-point gap 1.2 SE — noise. Printing the SE is what turns
  "this is not a clinical finding" from an assertion into something a reader can
  check.

### The honest problem, stated before the work starts

**MedMNIST has no demographics** — confirmed while building this, and handled as
below. Cohort-conditioned evaluation therefore needs synthetic patient attributes
attached to real images. That is defensible for a
*cost-structure* claim — the encrypted selection does the same work whether the
ages are real or generated — and it is **not** defensible for any clinical or
fairness claim, because a subgroup gap measured on synthetic demographics is an
artifact of the generator. So:

* the benchmark must label the attributes synthetic in its own output, not just in
  a doc;
* the claim is "this composition is possible and this is what it costs", never
  "the model underperforms on subgroup X";
* a real clinical claim needs a dataset carrying genuine demographics (CheXpert
  and MIMIC-CXR do; both need credentialed access), and that is the upgrade path
  to note rather than to fake.

### Also on the list, in priority order

1. **Cross-institution private benchmarking.** Two hospitals evaluate the same
   model on their own patients and learn only the *pooled* metric — PSI for
   deduplicating shared patients (stage 3, built), then an aggregate. This is the
   federated-evaluation story and it reuses everything.
2. **Longitudinal accumulation.** `baru bench --store barudb:<path>` already
   persists runs; the unrealised half is generating the paper's tables *from
   queries* rather than transcription. Cheap, and it removes a whole class of
   copy-paste error this session has already made more than once.
3. **Multi-channel MedMNIST** (PathMNIST, DermaMNIST) — see the wish list; a
   packing question.

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

### The wait budget, and the first assignment measured end to end (2026-08-05)

`sync_waits` — the number of times a party blocks for a peer — is the counter
that predicts wall clock; `interactive_calls` and `messages` both contradicted
it (E33). With it in place the compiled network's 33 blocks can be *attributed*
rather than guessed at, and the attribution is lopsided:

| levels | what they hold | blocks each | share of budget |
|---|---|---|---|
| 3 of 8 | a comparison | 9 | **72%** |
| 5 of 8 | multiplies / muxes only | 2 | 28% |

So the comparison is the primitive worth changing, and Baru has three. Measured
at 64 comparisons, ℓ=32, preprocessing pre-filled
(`the_wait_cost_of_each_comparison_backend`):

| backend | waits | messages | kB |
|---|---|---|---|
| `a2b` | 10 | 16 | 127.1 |
| `falcon` | 11 | 44 | **49.1** |
| `gc` | **3** | 11 | 109.9 |

A **3.7× spread in blocking**, which is the precondition for an assignment pass
being worth deriving at all — if the three agreed there would be nothing to
choose. Two readings matter:

- **`gc` strictly dominates `a2b`** on both axes, and `a2b` was the compiled
  route's default. A free improvement, invisible to every correctness test and
  to the byte count.
- **`falcon` is byte-cheapest by 2.2× and block-dearest.** The same trade the
  network-level crossover reports, arriving at the granularity of one
  primitive. An assignment cannot be read off one column, which is why the
  registry key has to carry the network regime.

Assigning comparison levels to `gc` cuts the compiled route **33 → 12 blocks
(2.75×)** — the largest lever after levelling — and 486 → 270 ms at RTT 20.

**The control is the result.** The same assignment takes the *hand-written*
route 33 → 12 blocks and 493 → 279 ms. The gc win is a **protocol** result,
available to whoever writes the circuit; measuring the assigned compiler against
the unassigned expert reports 0.36× waits of which the compiler contributed
nothing. Like for like the two are at **parity (12 vs 12)**. `bench-compiled-nn`
now carries a `gadget + gc` control row, the summary compares at the matched
backend, and the parity test asserts *that the mismatched comparison is the
flattering one*.

*(All figures in this section were re-measured on the 16/8/4 network over four
repetitions after an earlier pass quoted some of them from the 8/4/4 CLI default;
see the jitter note below and E37.)*

Per-pass ladder in the binding metric: **serial 5854 → levelling 43 → CSE 43
(nothing) → reassociation 32 → gc comparisons 12.**

**A prediction measurement killed.** The cost model charges a level the `max`
over its primitives; the executor issues one batched call per *kind* and each
blocks, so a mixed level pays the `sum`. I had written this into the paper as
the most valuable change outstanding. Counting the mixed levels: **there are
none** — 8 levels, 8 batched calls — because `z·(z>0)` and a mux-style `max`
place comparison and multiply at consecutive levels *by construction*. The
definitions do differ; this class of program does not pay for it.
`ExecStats::mixed_levels` is kept so a schedule that does pay is visible, and
the optimisation is not claimed.

### The registry keyed on the link, and the third cost term measurement forced (2026-08-05)

The assignment above was applied by hand to one operation kind. Making it a pass
means keying the `precompile/` registry on the deployment, so
`Registry::champion_for(profile)` now exists alongside `champion()`:
`PrecompileMeta` carries an optional `MeasuredCost`, selection ranks only entries
that have been measured, and an uncosted registry (`matvec`) behaves exactly as
before. Registration stays the human gate — the costs are transcribed by hand
from a named test, and nothing is auto-promoted. `precompile/mpc_compare.rs` is
the first regime-keyed gadget, with the three comparison backends as entries.

**The premise it was built on turned out to be false, and finding that out was
the useful part.** From `falcon` being 2.2× cheaper on the wire and 3.7× dearer
in blocking, I predicted a champion that flips — byte-cheap on a thin pipe,
block-cheap on a slow one — and wrote a test asserting the flip. Measured:

| link | a2b | falcon | gc |
|---|---|---|---|
| RTT 0, unlimited | 33 ms | 79 ms | **13 ms** |
| RTT 0, 10 Mbps | 206 | 130 | **105** |
| RTT 40, 100 Mbps | 334 | 1056 | **201** |

`gc` wins **everywhere**. With the network free, `falcon` costs 6× what `gc`
does, and that is **local cryptographic work** — Z_p arithmetic and
preprocessing — which a model of the form `waits × RTT + bytes / bandwidth`
charges every candidate *zero* for. Two consequences:

1. The network-only model could not rank the candidates at all on a fast link
   (all three predict 0 ms), and the tie-break returned the *dominated* entry —
   a wrong answer, not an arbitrary one.
2. On a thin pipe it actively mispicked: `falcon` at 130 ms over `gc` at 105 ms.
   The 78 kB `falcon` saves is worth 62 ms at 10 Mbps; it pays 66 ms more in
   local work to save it.

`MeasuredCost` therefore carries `local_secs`, measured on
`NetworkProfile::INSTANT`. With it the selection matches the wall clock at
**1.00×** at all three points.

What survives: the machinery (a link-keyed champion is the right shape, and it is
exercised on synthetic entries that do trade places), and the value of the
assignment itself (`gc` over `a2b` is 2.75× in blocking). What does not survive:
the claim that *this* gadget is where the champion flips. The test that asserted
a flip now asserts the measured fact, so a future change that produces one fails
and gets looked at.

*Next*: extend the key from `(gadget, link)` to
`(gadget, shape, backend, link)` and populate from the accumulated corpus — the
DSL vision — with the UC opening-audit lemma as the per-backend admission
criterion.

### P1 done: `local_ns_per_lane` fitted, and it was 214× low (2026-08-05)

`mpcc::cost` turned out **not** to have the structural omission the section above
predicted for it — it already carries a `local_ns` term and even reports a
`compute` bottleneck. The problem was narrower and worse: the constant feeding it
had carried the comment *"an estimate; calibrated in P1"* since it was written,
and P1 never happened. `baru mpcc calibrate` validates predicted **rounds** and
**online bytes** against a measured run and reports 0% error on both; it never
touched compute.

Fitted in `the_local_work_per_multiplication_is_what_the_spec_says`. The method is
the part worth keeping: wall clock at `NetworkProfile::INSTANT` is per-lane work
*plus* a fixed cost (thread spawn, share setup), so dividing one measurement by
its width charges the fixed part to every lane. It measures four widths
(1024…65536) and fits the **slope**, reporting the intercept separately.

| quantity | measured |
|---|---|
| `mul_batch`, real transport, network free | **1072 ns/lane** |
| local-only `add`, no communication at all | 145 ns/lane |
| the old estimate | 5 ns/lane |

So 5 ns was **214× low** for a multiplication lane, and **29× low even for the
pure arithmetic floor** — an `add` that communicates nothing already costs 145 ns
in share bookkeeping and a ring addition. A multiply is 7.4× an add; the rest is
serialization and the channel round. Neither reading rescues the constant, so
`aby3`'s is now 1072, documented as fitted, with every other protocol's left at
5–80 ns and explicitly marked as still an estimate.

**It changes a conclusion, in exactly the regime the NN work lives in.** On a
3-instruction program compute is 3 µs against 878 µs of latency and nothing moves.
On a *batched* program — 513 interactive ops, 2 dependence levels, batching
256.5×, i.e. the shape of a dense layer:

| | aby3 total | bottleneck | rank |
|---|---|---|---|
| old (5 ns, estimated) | 169.0 µs | latency | 1st |
| new (1072 ns, fitted) | 716.3 µs | **compute** | 2nd, behind lowgear |

4.2× on the total, and the bottleneck flips from latency to compute.

**And the resulting table must not be read as a ranking.** `aby3` is now the only
protocol whose compute term is measured; the others are unfitted estimates that
are presumably low by a similar factor, so "lowgear beats aby3" would be the same
incommensurable comparison this project keeps catching. `format_comparison` now
emits a note whenever any row is compute-bound, saying which constant is fitted
and that a cross-protocol ordering must not be read off a compute-bound table.

*Also found*: `Aby3Simulator` is **not** usable as a performance instrument. It
has no batched multiply, and `sim::mul_ab3_concurrent` spawns three OS threads and
three inproc handles *per multiplication* — 58 µs/lane, 54× the real backend. It
is a correctness aid, and the calibration test now says so where someone would
otherwise reach for it.

*Next on this thread*: fit the remaining constants, which needs an implementation
of each protocol to time — so it is gated on the same roadmap as running them.
Until then the fitted/estimated split is a caveat, not a comparison.

### The byte column was not the traffic: an unaccounted link (2026-08-05)

Having fitted the compute constant, the obvious question was whether the NN
pipeline is compute-bound — which would mean every byte/round/wait optimisation
in this repo has been tuning the wrong term. It is not, and the investigation
turned up something more specific.

Fit gc's per-diagnosis time against `1/bandwidth` at RTT 0 (from the existing
sweep):

| 1/bw (s/MB) | 0.08 | 0.133 | 0.2 | 0.267 | 0.4 | 0.8 |
|---|---|---|---|---|---|---|
| measured (s) | 3.17 | 4.70 | 6.49 | 8.45 | 12.45 | 23.81 |

Strikingly linear — consecutive slopes 28.9, 29.8, 28.7 MB — so the model's
*form* is right and compute is a floor rather than the dominant term. But the
fitted slope is **~29 MB against the 10.1 MB the traffic table reported**.

**Correction to the intercept.** That fit put the compute floor at 0.85 s, and I
quoted it. Measured *directly*, with `NetworkProfile::INSTANT` so there is no
bandwidth or latency term at all, one gc diagnosis costs **465 ms** — the fit
over-estimated by 1.8×, because a six-point least-squares fit is dominated by its
10 Mbps end and any slight nonlinearity there inflates the intercept (the sweep
data also predates the Π_PC fix). A floor that can be measured directly should
never be quoted from an extrapolation. Fully attributed by the existing
breakdown: fhe 292 ms (weights 43, mul 56, **mask-enc 131**, decrypt 45,
re-encrypt 10) and mpc 172 ms (import 22, relu 53, **pool 78**, argmax 7).

Two candidate explanations, one killed cheaply:

- **Per-party asymmetry — refuted.** `run_parties` kept only party 0's counters,
  and in an asymmetric protocol the busiest link need not be party 0's. Added a
  `BARU_PER_PARTY_COMM` diagnostic reporting all three. Party 0 **is** the
  busiest in every route (1.00×); in the gc routes it sends 30 191 B against
  6 415 and 1 571. The reported figures were the right party's.
- **Unaccounted evaluator-link traffic — confirmed.** The CNN pipeline calls
  `sleep_link` with FHE ciphertext and scheme-switch bytes that never cross
  `CountingTransport`. It *really sleeps* for `bytes / bandwidth`, so the latency
  columns have always paid for that traffic while `comm.bytes` never saw it.

**Fix**: `FhePhases::link_bytes` accumulates at every `sleep_link` site in the CNN
path and threads through `CnnTotals` → `SweepCell`. The traffic table now reads
`MB party | of which ring | MB eval link | MB total`, with a note that the old
`MB wire` heading covered the party transport only.

**What this does and does not invalidate.** The eval-link term is *common to both
nonlinearity protocols* — it is the same hybrid pipeline either way — so the
**gc-vs-falcon crossover location is unaffected**; that comparison was always a
common term plus a difference. What is affected is any **absolute** traffic
claim: a deployment moves roughly 3× what the party column alone says. That bears
directly on the paper's traffic comparisons against Falcon's published figures,
which must now be stated against the total or explicitly scoped to MPC traffic.

*Also found*: `CountingTransport::broadcast` charges `len × fanout` while
`LatencyTransport::broadcast` charges `deliver_at(len)` **once** — the counting
layer and the pipe layer disagree on what a broadcast costs. Not yet reconciled;
it means messages and bytes are counted per recipient while the simulated pipe
serializes a broadcast as a single transmission.

*Method note*: an earlier attempt to confirm this ran the sweep in the background
while `cargo test` and clippy ran, which contends for CPU and inflates the
millisecond columns. Byte columns are unaffected by contention; timing columns
from a contended run are not usable. Do not measure and build at the same time.

### What the compute floor is made of, and what is already hoisted (2026-08-05)

With bandwidth accounted (above), the remaining term is local compute, measured
directly at **465 ms** per gc diagnosis. Mask encryption is its largest single
item — 131 ms, 45% of the FHE side and 28% of the whole floor — and it is
query-independent in principle. `--precompute` already exploits that:

| | online latency | fhe | mpc |
|---|---|---|---|
| cold | 465 ms | 292 ms | 172 ms |
| `--precompute` | **276 ms** | 102 ms | 173 ms |

A **1.68×** reduction in online latency, with `mask-enc` and `weights` going to
zero. The tool's own caveat is the important part and it is already there: of the
207 ms/image hoisted, **137 ms is per-QUERY** (the mask pool is fresh for every
image, as it must be — reusing a mask would break the scheme switch's blinding),
so it *shifts work off the critical path without reducing total work*. Only the
~70 ms of weight encryption is genuinely amortisable across queries, because the
weights do not change. Total work is 276 + 207 = 483 ms against 465 ms cold,
i.e. unchanged within noise, which is the check that the split is a split and not
a saving.

So the online compute floor is **~276 ms and MPC-dominated** (173 of 276), with
max-pooling its largest single item at 84 ms — larger than ReLU at 50 ms.

*Next on this thread*: pooling and ReLU are both comparison work, and the
comparison backend changes local cost by 6× at the primitive level
(`precompile/mpc_compare`: gc 13 ms, a2b 33, falcon 79 per 64 comparisons). The
registry's `local_secs` was measured on raw comparisons; measuring it at the
*gadget* level — a max-pool layer, a ReLU layer — would let the assignment pass
choose on the term that now binds. That connects the two open threads: the
backend assignment and the compute floor are the same question at high bandwidth.

### DKG for threshold FHE: the dealer removed, and why it was there (2026-08-05)

The single named gap in the UC argument was that threshold FHE key generation was
dealer-based. It now is not — and the interesting part is *why the dealer was
there*, which was not oversight.

`deal_key_shares` makes components `s₀`, `s₁` **uniform mod q** while their sum is
**ternary**, by setting `s₂ = s − s₀ − s₁`. Those two properties together are what
a dealer buys. **No protocol in which each party samples its own component
independently can have all three of independent sampling, a small sum, and
uniform components.** Sampling `sᵢ` uniform makes `s` uniform mod q, and BFV
decryption computes `c₀ + c₁·s = Δm + (e₁ + e₂·s − e·u)` — with `‖s‖ ≈ q` the
`e₂·s` term exceeds `q/2t` and decryption fails outright, not gracefully.

**The design takes the other horn.** Each party samples `sᵢ` ternary and sends it
to its predecessor, so the replicated layout `(sᵢ, sᵢ₊₁)` is preserved — kept
deliberately, because replication is what makes a partial decryption checkable by
recompute-and-compare, and a DKG producing only additive shares would remove the
dealer at the cost of the malicious-security argument.

*Security of the non-uniform components.* One corruption learns two components,
hence `k = sᵢ + sᵢ₊₁`, and that `s − k` is ternary. That reduces to the standard
assumption rather than breaking it: `b + a·k = −(a·(s−k) + e)` is an RLWE sample
with a ternary secret. The algebraic identity is a test — the residual sup-norm is
exactly 3, the sum of three ternary errors — so if the identity ever failed the
argument would be visibly vacuous rather than quietly so.

*Cost.* `‖s‖∞ ≤ 3` instead of 1, about 1.6 bits of noise budget for a fresh
ciphertext and ~3.2 for a product; `dkg_key_norm_headroom_bits` reports it, since
it comes off the same budget as smudging.

**The part that falls out for free.** A product decrypts under `s²` because Baru
relinearizes nothing, and with replicated shares each party can form its own `s²`
share **with no communication**:

```
s² = s₀² + s₁² + s₂² + 2(s₀s₁ + s₁s₂ + s₂s₀)
```

Party `i`, holding `(sᵢ, sᵢ₊₁)`, computes `sᵢ² + 2·sᵢ·sᵢ₊₁`; the three sum to
exactly `s²` — the same identity ABY3 multiplication uses, because it is the same
sharing. So the expensive part of threshold FHE, distributed generation of a
relinearization key, is not merely deferred but **not needed**: the depth-1 design
avoids requiring one. `partial_decrypt_any` handles degree 1 and 2 and *refuses*
anything higher, where the old `partial_decrypt` read only `c[1]` and would have
returned a wrong answer silently.

**What is not closed, stated plainly.** The CNN hybrid still decrypts with
`decrypt_poly_pre` and a whole `DecryptKey`, so the deployed pipeline retains a
party that can decrypt. Key generation is what this closes; wiring the threshold
path into the pipeline is a separate change with its own cost — three partial
decryptions and a combine per scheme switch against one local decryption today.
`DKG_WIRING_GAP` names it in the code so "DKG landed" cannot be read as "no party
can decrypt".

### Convolution has a backend crossover too (2026-08-06)

Adopting CHIP's fast convolution (Winograd F(2x2,3x3), `gadgets::winograd`) left
a correct gadget with no consumer: `gadgets::conv2d` had no callers outside
tests, since the CNN convolves in FHE. `bench-conv` is the consumer, and it
turned a dormant optimisation into a measured decision.

Three strategies for one 3x3 layer, all keeping the model secret, all ending
with **secret shares** (the boundary the following ReLU needs), all checked
against one plaintext reference. Timing is the **per-layer online cost**: for the
shared routes the convolution itself, for the homomorphic route the convolution
plus its scheme switch. See the timing note below — getting this boundary right
changed the answer by 20x.

| shape | direct kB | winograd kB | fhe kB | wino ms | fhe ms | byte win pays below |
|---|---|---|---|---|---|---|
| 2x8x8 -> 3 | 49 | **32** | 1769 | **11** | 31 | shared dominates |
| 4x12x12 -> 4 | 300 | **172** | 2359 | **21** | 42 | shared dominates |
| 8x16x16 -> 8 | 2063 | **1060** | 4719 | 90 | **69** | 1424 Mbps |
| 16x20x20 -> 8 | 6602 | **3284** | 4719 | 231 | **130** | 114 Mbps |
| 16x28x28 -> 16 | 26478 | 12634 | **9437** | 1022 | **332** | fhe dominates |

**Winograd moves the assignment.** At `16x20x20->8` the direct MPC convolution
(6602 kB) loses to FHE on bytes while the Winograd one (3284 kB) beats it. And in
wall clock Winograd is 2.2x the direct route (231 vs 519 ms at that shape),
matching its 2.25x multiplication ratio almost exactly.

**The conclusion favours secret sharing more than first measured.** The shared
route dominates outright at small layers, and where neither dominates the
break-even bandwidths are 1424 and 114 Mbps — above almost any real link. Only
the largest layer goes to FHE on both axes.

**Three wrong narratives, and the fix.** I wrote this summary three times:
"bytes decide", then "compute decides and the byte win is irrelevant", then "fhe
is faster at every shape". Each was true of the data in front of me and falsified
by the next measurement. The benchmark now **derives** its summary sentence from
the measured rows — counting which route wins on each axis — so a stale narrative
cannot survive a changed table.

**The timing bug that caused most of it.** The first published figures had the
shared route at 5023 ms against FHE's 135 ms, a 37x gap. Almost all of it was my
own benchmark: it opened all 2592 outputs **one at a time** as verification
scaffolding, while the homomorphic route decrypts a whole channel per call. In a
deployment the shared route's output is never opened — it feeds the next layer's
ReLU as shares. Excluding the scaffolding, and symmetrically excluding FHE's
per-session `keygen` from its side, moved the shared route 5023 -> 236 ms and the
comparison from 37x to 1.8x. **95% of what I had reported as a protocol cost was
scaffolding**, and the asymmetry was invisible because each route's opening looked
natural in isolation.

### Per-operation protocol assignment (2026-07-28, from SING)

Reading SING (USENIX Sec'26) exposed a real gap: Baru ranked **whole-program**
protocol choices, while the state of the art (HyCC → Silph → SING) assigns
**per operation** with conversion costs — and the `mpcc::select` recommender both
this document and the roadmap referred to *did not exist*. `src/mpcc/assign.rs`
and `baru mpcc assign` now do the per-operation-kind version, reproducing the
classic result (multiplications arithmetic, comparisons binary) from Baru's own
cost model, with a measured 1.25× on a mixed program and a 9.5 KiB → 41 B drop in
online traffic.

The argument worth carrying into the paper: **Baru needs neither Silph's ILP nor
SING's GNN**, because its cost model is non-separable in the assignment variables
(a level costs the max over its members; conversions batch per level) *and* cheap
to evaluate exactly. ILP cannot express that objective and a learned surrogate is
not needed to approximate it. Full analysis, including what was deliberately not
adopted, in [related-work.md](related-work.md).

### FHE in the assignment — the result the two papers share (2026-07-28)

`src/mpcc/fhe_cost.rs` puts FHE into the assignment, which HyCC, Silph and SING
structurally cannot do. The model gets **only measured inputs**: BFV ct×ct
multiply 20.7 ms and ct 128 KB (`bench-fhe-innerprod`), TFHE 0.2–0.5 s/gate
(`baru bench`), Beaver-triple cost derived from measured OT (§3b), plus two
*structural* facts — BFV is arithmetic-only (no comparison at any price) and has a
depth-1 budget.

Given that, on one NN layer — 256 secret×secret products (encrypted weights *and*
encrypted input) followed by a comparison, WAN, 2 parties:

```
assignment       rounds      online     offline  conversions  predicted
all-arithmetic       12     8.8 KiB     1.7 MiB            0  740.96 ms
all-binary           17    66.0 KiB   808.6 MiB            0    68.69 s
mixed (best)         11   128.0 KiB   101.1 KiB          771  589.47 ms

mixed wins 1.26x, by moving:  cmp-lt -> binary,  mul -> fhe
```

**The model derives the hybrid architecture** — for a *dishonest majority*.
Linear layer on FHE, nonlinearity on MPC: Gazelle/Delphi/Cheetah's design, and the
one `src/hybrid/` implements by hand. Nothing told it to; it followed measured
costs. A test pins it, so a change to any cost input that would overturn the claim
fails the suite instead of quietly rewriting the paper.

**The threat model is part of the answer, and this is the sharper claim.** The run
above uses Semi2k, which needs a Beaver triple per multiplication (~3.1 KB, priced
from measured OT). Re-run it against **ABY3** — honest majority, multiplication
from correlated randomness, `triples_per_mul = 0` — and the layer **stays on MPC**:

```
assignment       rounds      online     offline  conversions  predicted
all-arithmetic       12     4.4 KiB         0 B            0  600.36 ms
mixed (best)         11     2.0 KiB         0 B            3  550.17 ms
mixed wins 1.09x, by moving:  cmp-lt -> binary        (mul stays arithmetic)
```

So the same program flips technology on the *deployment's threat model*, not just
on its shape. And that explains something about the literature: **the hybrid
FHE+MPC systems are all 2-party** — Gazelle, Delphi, Cheetah — which is exactly the
dishonest-majority regime where the model says the hybrid pays. Under an honest
majority the hybrid is not merely unnecessary, it is a loss.

#### Checked against measurement, and it needed a correction

`bench-hybrid-nn` measures one dense layer end to end under six strategies, so the
prediction is testable. At WAN settings matching the model (RTT 100 ms, 100 Mbit/s),
sweeping layer width with `--out 1`:

| multiplies | measured pure MPC (ABY3) | measured hybrid-BFV |
|---|---|---|
| 8 | **775 ms** | 1338 ms |
| 32 | **775 ms** | 1335 ms |
| 128 | **766 ms** | 1343 ms |
| 512 | **778 ms** | 1331 ms |

MPC wins at every width and is **flat in width** — the layer's cost is rounds, and
rounds do not grow with width. The model predicts the same decision (mul stays
arithmetic) and the same order of magnitude (600 ms predicted vs 775 ms measured).

My first reading of this was that the model was wrong, because I had compared a
Semi2k *prediction* against an ABY3 *measurement* — two different threat models.
`bench-hybrid-nn` implements ABY3 only. Once matched, model and measurement agree
on the decision. The lesson is worth keeping: a prediction and a measurement must
name the same protocol before they can disagree.

#### The gap closed: Semi2k now runs, and the triple cost is measured

`src/mpcc/semi2k.rs` implements the missing dishonest-majority baseline —
additive sharing mod `2^k`, semi-honest, two parties, with Beaver triples from
**real oblivious transfer** (Gilboa OLE over the IKNP correlated OT added in
`src/ot/iknp.rs`). It is in `mpcc::RUNNABLE`, so `baru mpcc calibrate` runs it.

Two results:

* **Online phase: 0% prediction error.** Rounds, bytes, multiplications and
  triples all match on a comparison-heavy program (303 triples), same as Dealer.
* **Offline bytes per triple, measured: 3104 B against a predicted 3136 B — 1%.**
  This is the number the whole crossover claim rests on, and it was previously
  derived rather than measured. It now runs through real OT and lands within 1% of
  `mpcc::ot_cost`'s Gilboa derivation (2k correlated OTs × (16.5 B measured + k/8
  payload)). A test pins the ratio.

So the chain behind "FHE takes the wide layer under a dishonest majority" is now:
measured OT cost → analytic Gilboa count → **measured triple cost** → the
crossover. Only the last step is still model rather than measurement.

Two measurement traps worth recording, both hit:

1. **Offline cost amortizes over the generation batch, not the consumption count.**
   Dividing offline bytes by triples *consumed* reported 68 288 B/triple on a
   program that used 3 of a 64-triple batch — a 21× apparent miss that was purely
   a wrong denominator. `Counters::triples_generated` now exists so the ratio is
   taken against the right thing.
2. **Correlated OT, not chosen-message OT.** Gilboa's two messages differ by a
   known correlation, so only one `k`-bit correction is sent — `κ + k` bits per OT,
   not `κ + 2κ`. Measuring with the chosen-message interface would have inflated
   triple cost by ~2× and made the model look wrong. `iknp::correlated` implements
   the right interface and a test pins it just above its `24 B/OT` analytic floor.

#### The crossover, measured (`baru bench-layer`)

One private layer, both operands secret, plus a ReLU. MPC costs are **measured by
running the protocols** on `mpcc`'s VM; the FHE side uses measured per-operation
costs; wall-clock is computed from those counts and the WAN profile.

| width | `dealer` (lower bound) | `semi2k` (deployable) | hybrid/bfv | semi2k offline |
|---|---|---|---|---|
| 8 | 600.4 ms | **682.4 ms** | 692.1 ms | 1000 KiB |
| 64 | 600.6 ms | 715.4 ms | **692.1 ms** | 1.4 MiB |
| 256 | 601.4 ms | 814.4 ms | **692.1 ms** | 2.5 MiB |
| 1024 | 604.3 ms | 1.21 s | **692.1 ms** | 7.2 MiB |

**The hybrid overtakes the deployable dishonest-majority protocol from width 64**,
and by 1024 it is 1.75× ahead. The model predicted the crossover between 64 and 256;
measured, it is at 64 — the model was slightly conservative, in the direction that
does not flatter it.

The mechanism is visible in the last column and it is measured, not modelled:
Semi2k's real OT preprocessing grows 1000 KiB → 7.2 MiB across the sweep while the
hybrid's packed ciphertext stays at 256 KiB. Online bytes and rounds barely move for
anyone.

**`dealer` is the control, and it carries the argument.** It assumes a trusted party
hands out triples for free — not a threat model, a lower bound — and it stays flat
and cheapest at every width. So the entire case for FHE here is the cost of
*preprocessing*, not of computation. If triples were free, MPC would win outright.

Two comparison errors were caught building this, both of the same kind — comparing
things that are not the same computation:

1. The first version priced FHE's *linear* part against MPC's *whole layer*,
   omitting the ReLU from one side, and FHE appeared to win at every width. BFV
   cannot evaluate a comparison at any price, so the nonlinearity stays on MPC
   either way and the honest comparison is against the **hybrid**.
2. The crossover was first computed against *every* MPC protocol including
   `dealer`, i.e. asking whether FHE beats free preprocessing. It does not, and the
   question is meaningless. Deployable protocols only.

#### The formula itself is now validated (`baru bench-net`)

Every wall-clock number above is a measured *count* put through
`rounds × latency + bytes / bandwidth`. That formula had never been checked. It is
now, against a real loopback TCP socket, measuring the link's own constants and
sweeping each term independently:

| | measured | predicted | error |
|---|---|---|---|
| 500 rounds, 8 B | 12.27 ms | 11.53 ms | −6.1% |
| 1000 rounds, 8 B | 22.94 ms | 23.06 ms | **+0.5%** |
| 4000 rounds, 8 B | 92.01 ms | 92.22 ms | **+0.2%** |
| 16 rounds, 64 KiB | 0.78 ms | 0.74 ms | −5.2% |
| 16 rounds, 256 KiB | 1.86 ms | 1.84 ms | **−1.0%** |
| 16 rounds, 1 MiB | 7.24 ms | 6.25 ms | −13.6% |

**Worst case 13.6%, under 1% through the middle of both ranges.** Cost is linear in
rounds and linear in bytes with no missing cross-term — which is what justifies
computing wall-clock from measured counts everywhere else.

Two methodology errors, both mine, both worth recording because the first run
reported 36% error and blamed the model:

1. **Probing cold and sweeping warm.** The first socket in a process is slower than
   the tenth, which inflated the measured RTT by 36%. There is now a warm-up phase.
2. **Measuring a constant in one regime and applying it in another.** Throughput
   from a single 4 MiB transfer beats what the sweep's 16×256 KiB shape achieves,
   biasing the bandwidth constant high and predictions low. The residual −13.6% at
   1 MiB is that same effect not fully gone — throughput degrades with payload size
   — and is left as the honest error rather than tuned away.

**Still open, and narrower now:** the constants are loopback's (RTT 23 µs), not a
WAN's. The formula's *shape* is validated and transfers; a WAN deployment needs its
own two constants measured on that link. Also, this measures a socket rather than a
multi-party protocol — whether a protocol's *rounds* are counted correctly is
`baru mpcc calibrate`, which reports 0% error.

The mechanism is specific and worth stating: **MPC's Beaver-triple preprocessing
grows linearly with layer width while FHE's packed ciphertext is flat and needs
none.** Offline drops 1.7 MiB → 101 KiB. And there is a genuine crossover — the
model keeps a *narrow* layer on MPC, because 8 multiplications batch into one
round that FHE cannot amortize a switch against:

| layer width | assignment | speed-up |
|---|---|---|
| 8 muls | stays arithmetic | — |
| 64 muls | stays arithmetic | — |
| 256 muls | **mul → fhe** | 1.18× |
| 512 muls | **mul → fhe** | 1.54× |

This is where the two papers meet. Thesis (S) says the cheapest technology is a
property of the program; here the *same* program shape switches technology at a
width the model predicts. And `paper-outline.md`'s switch-as-bootstrap claim is
what makes the FHE side legal at all: the depth-1 budget means a second
multiplication would be infeasible, and the switch to MPC for the nonlinearity
*is* the refresh. Depth being free is not an add-on to this result — it is the
precondition.

#### Two modelling errors found by disagreeing with a measurement

Worth recording, because the process is the point. The first version of the FHE
cost model concluded **"never use FHE"** — which contradicts `bench-cnn`, where a
working hybrid puts FHE at ~5% of runtime. Two terms were missing:

1. **SIMD packing.** FHE was charged per scalar operation. But `src/fhe/conv_pack.rs`
   packs a whole convolution channel into *one* ct×ct multiply, so per-lane
   charging is the wrong unit. Fixed by tallying operations per (level, kind) and
   charging `ceil(count / slots)` — the same batching shape MPC's rounds have, and
   a second non-linearity an ILP could not express.
2. **Preprocessing.** The objective ignored offline cost entirely, so MPC's triples
   were free. They are not, and pricing them from measured OT is what creates the
   crossover above.

A third error was subtler: the switch was charged **one ciphertext per crossing
edge**, re-encrypting the same value once per use and making FHE look 16× worse on
a wide layer. Crossings pack too.

Still not done: assignment is per operation *kind*, and CKKS is excluded because
its SEAL implementation is not timing-comparable with Baru's schoolbook-NTT BFV —
so the FHE side is priced at BFV's speed, which is conservative *against* FHE.

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

**Done 2026-07-28**, once that precondition was met (five rules, per-site search,
a differential oracle):

* `.claude/skills/mpcc-rewrite-rule/SKILL.md` — the generator. It documents the
  `FlatOp` invariants a rewrite must preserve, the per-site-vs-whole-circuit
  decision, the oracle, the protocol-family table that predicts *who will take a
  rule*, and the repo conventions (including "never write a results table you
  have not run"). Every symbol, `FlatOp` variant and protocol id it names was
  checked to resolve against the tree — a skill describing an API wrongly is
  worse than no skill.
* `docs/gadget-costs.md` — point 3, the curated table, deliberately a small
  checked-in file rather than retrieval. Each row carries **provenance**:
  *standard* (textbook), *Baru's model* (what `cost.rs` charges), or *measured*
  (with the command). Writing it surfaced a modelling choice worth knowing:
  Baru charges a **ripple** adder's gate count at a **prefix** adder's depth, so
  binary-protocol latency estimates are optimistic — which matters for
  `semi-bin`/`tinier` and not at all for the constant-round protocols.

---

## 6. Sequence

Ordered by value per unit of work, with the dependency structure respected.

1. ~~**MedMNIST swap** (§3)~~ — **done 2026-07-28.** The artifact is now a
   healthcare artifact: private pneumonia and organ diagnosis, plaintext-exact
   under encryption, no new crypto. Next dataset in this direction would be a
   *multi-channel* one (PathMNIST, DermaMNIST). **Corrected 2026-08-03: this does
   not need channel splitting.** The claim that it did was wrong — `conv_pack` was
   always multi-channel, and an RGB 28×28 conv needs degree 3977 against
   `n = 4096`. The blocker was three hardcoded `in_c = 1` sites in `CnnModel`,
   now a real field, with an RGB image classifying exactly under encryption at one
   ciphertext multiplication per output channel
   (`cnn::tests::rgb_input_classifies_exactly_under_encryption`). What remains is
   ordinary: train on an RGB set and embed the weights.
2. ~~**Backend-aware optimizer** (§4)~~ — **done 2026-07-28**: reassociation-for-
   depth added (the sharpest form of the claim — it moves *only* depth), plus
   **per-site search**, which replaced the whole-circuit bundling that had been
   the pass's main weakness. Backends now diverge within a rule and not merely
   over which rules they accept. Getting reassociation right also required fixing
   a real cost-model bug in binary-protocol depth accounting (§4). Still open and
   worth doing: the search is greedy with a fixed rule order and evaluates sites
   independently, so a pair of rewrites that only pays *together* is missed.
3. ~~**P2 of the MPC roadmap** — random bits and edaBits, so comparisons *run*~~ —
   **done 2026-07-27**: `vm::bit_decompose_batch` (masked opening + Kogge–Stone
   prefix borrow, log depth); `relu`/`max` execute under Dealer and Shamir and
   agree with the plain interpreter, at 0% model error. Secret division is still
   refused, by design, with an error naming its phase.
4. ~~**Cohort queries in BaruDB** (stage 1)~~ — **done 2026-07-27**: `baru
   bench-cohort`, query overhead ≈1.0×, ingest 5.05× (§2).
5. ~~**PIR** (stage 2), textbook single-server over the existing BFV~~ — **done
   2026-07-28, with a negative result** (§3a): correct, but ~13× *worse* than
   downloading the database at every size, because ciphertext expansion exceeds
   the database-to-response ratio. The follow-on is SealPIR-style recursion with
   server-side query expansion (Galois automorphisms + substitution in
   `src/fhe/bfv.rs`) — and the estimate for *that* is not a few hundred lines.
   Worth doing only if stage 2 stays in the story; the cost model now says the
   dull answer wins today.
6. ~~**OT + PSI** (stage 3)~~ — **done 2026-07-28** (§3b): base OT + IKNP
   measured, their cost wired into the Semi2k/SPDZ2k/MASCOT offline parameters,
   and DH-PSI built and measured with the OT route priced against it. Remaining
   and now well-specified: a KKRT-style OT-based PSI (the projection says ~100×
   on time, and it is the compute axis that decides), and KOS-style consistency
   checks if malicious security is wanted. Both are optional — stage 3 is served
   today.
7. **ORAM** (stage 6) — **wish list, not scheduled** (2026-07-28 decision). Path
   ORAM over the storage layer. Deliberately parked: see the wish list at the end
   of this document for the condition under which it comes back.
8. ~~**Circuit-optimization skill** (§5)~~ — **done 2026-07-28**, together with the
   curated gadget-cost table it depends on.

Items 5–7 are the criptolandia items. Note that each one is *also* infrastructure
for the compiler: PIR reuses BFV, OT unlocks two MPC protocols, ORAM gives the
cost model a data-dependent-addressing entry. That is the test they pass.

**Status as of 2026-07-28: items 1–6 and 8 are done; item 7 (ORAM) is parked.**

---

## 7. Wish list — wanted, not scheduled

Things that are genuinely interesting and deliberately not being built. Parking
them here rather than leaving them as open to-dos is the point: an agenda where
everything is "planned" cannot express priority.

* **A whole-layer dishonest-majority measurement** — *what is left of the item
  above, now that Semi2k itself is done (§4).* Semi2k runs on `mpcc`'s VM and
  calibrates; `bench-hybrid-nn` measures layers on the ABY3 transport backend. To
  compare Semi2k end-to-end against hybrid-BFV, one of the two has to move — most
  likely re-expressing the ABY3 transport backend as a `ShareEngine` (already on the
  roadmap's P1 list). That would turn the crossover *point* from a prediction resting
  on measured inputs into a measured crossover.

* **GPU offload for the NTT — wanted, but Amdahl says wait.** The multi-modular
  NTT is the classic GPU target and the per-prime structure that made threading
  free (§3.5 of [private-inference-performance.md](private-inference-performance.md))
  maps to a GPU just as cleanly. The reason it is parked: after threading, FHE is
  **37% of the online latency** (99 ms of 268), so a GPU that made FHE *free*
  would buy 1.59× and no more, while the MPC nonlinearities that now dominate
  (63%) are garbled circuits — the thing GPUs are worst at. Two conditions would
  flip this, and both should be checked before any CUDA/Metal is written:
  (a) moving past 28×28, where `n` grows to 8192+ and the transforms dominate
  again; (b) *throughput* rather than latency — batched serving amortises kernel
  launch and PCIe transfer, which at `n = 4096` for a single image would likely
  eat the win outright. Prior art to price against first: CryptGPU and Piranha
  put 3-party MPC itself on the GPU, which is a different and possibly better
  bet than accelerating our FHE half.

* **Secret-index array access, priced** — *ahead of ORAM, and cheap.* CirC
  (`docs/related-work.md` §2) implements exactly the two cases either side of the
  ORAM line: a public index compiles away, a secret index becomes an `n`-way mux
  tree. `FlatOp` can already express that with `ICmpEq` + `Mux`. Doing it gives
  `mpcc::cost` the data-dependent-addressing entry that ORAM was supposed to
  provide, for a fraction of the work — and then the cost model, not a guess, says
  whether ORAM is ever needed.

* **ORAM (stage 6).** Path ORAM over the storage layer, to hide the access
  pattern of a training loop over a private index.

  *Why parked:* the admission rule says a primitive enters when it is the
  cheapest way to serve a stage, and §3a is a cautionary precedent — PIR turned
  out ~13× worse than reading the whole database, for a structural reason. The
  prior for ORAM is similar: it only pays when the dataset is large enough that
  `O(log² n)` bandwidth per access beats a linear scan, *and* the data is too big
  to stream once per epoch. For MedMNIST-scale training (tens of thousands of
  28×28 images, ~40 MB) streaming the whole set per epoch is trivially cheap, so
  ORAM would lose on exactly the axis PIR lost on.

  *Also:* CirC does not implement ORAM either — it implements the linear scan and
  stops there, which is a stronger endorsement of this decision than the reasoning
  alone was.

  *The condition that would revive it:* a stage-6 workload where the private
  index is over data that genuinely cannot be streamed — a hospital-scale imaging
  archive, or a training loop whose access pattern is itself the sensitive thing
  (e.g. active learning selecting which patients to label). Price it with the cost
  model before writing a line of it.

* **KKRT-style OT-based PSI.** The projection in §3b says ~100× on time over
  DH-PSI. Wanted; not urgent, because stage 3 is served today.

* **SealPIR-style recursion + query expansion.** What would make PIR competitive
  (§3a). Needs Galois automorphisms and a substitution op `src/fhe/bfv.rs` lacks.

* **KOS-style consistency checks** for the OT extension, if any claim needs
  malicious security.

* **Distributed key generation for the threshold FHE layer.** This is now the single named
  gap in the composability argument. The MPC layer UC-realizes `F_ABB` under stated hypotheses
  (private channels, a PRF, per-session key setup), the openings are enumerated and each is
  simulatable, and the abort is deferred to one input-independent point. The hybrid pipeline is
  covered except for key generation, which is dealer-keyed, so the honest statement is that it
  realizes its functionality in a model with a key-generation setup. Closing it means DKG,
  following Damgård–Nielsen (CRYPTO'03) for an arithmetic black box from threshold homomorphic
  encryption and Asharov et al. (EUROCRYPT'12) for the FHE case. The input-boundary proof that
  the malicious case also needs is already implemented.

* **A LAN/WAN measurement harness** so `mpcc`'s latency and bandwidth terms are
  calibrated against a real network rather than a profile (roadmap P1).

* ~~**An offline/online split for the MPC backend.**~~ ✅ **Done 2026-08-04.** It was the
  highest-value item because its absence had cost a headline result: `lt_falcon_batch` ran its
  preprocessing inline, so every latency was a *total* — fine when protocols share a phase
  structure, silently unfair when they do not. Falcon's preprocessing is now pooled per MPC
  segment (`cnn::FalconPoolPlan`, `--warm`), sized from counts a cold run *measures* rather
  than shapes it derives, and `falcon_pool_missed != 0` is a hard failure so an "online"
  latency cannot quietly contain preprocessing.

  It changed the answer: a **warm** Falcon wins to ≥100 Mbps at RTT 0, where a cold one loses
  to garbled circuits above ~25 Mbps. And it made Falcon's own claim measurable — at ℓ = 32
  its online phase is 0.25 MB/party against a 2.87 MB total, i.e. **0.33–0.57× Falcon's
  reported 0.74 MB** while the total is 3.9–6.7× it.

  The generalisation is still open: every preprocessing-based protocol in the repo has the
  same exposure, and only the Falcon path has the phase structure so far. `bench-malicious`
  is the next candidate, since its MAC-check preprocessing is query-independent too.

* **Multi-channel MedMNIST** (PathMNIST, DermaMNIST). No longer blocked on
  packing — see §6.1's correction. The crypto path takes RGB today; what is left
  is training on an RGB set and embedding the weights, plus teaching
  `tools/train_medmnist.py` not to call `to_grayscale`.

* ~~**Channel splitting for wide intermediate layers.**~~ **Built 2026-08-04**
  (`fhe::conv_pack::{channel_groups, pack_input_group, pack_kernel_group}`). The
  limit was real — 16 channels at 13×13 needs degree 5266 against `n = 4096`, and
  `max_group_channels` returns exactly **12** for that shape, which is precisely the
  cap the first conv has been living under. Splitting `in_c` into groups and summing
  the partial products is free: homomorphic addition needs no evaluation key and
  consumes no depth, so a wide layer costs `n_groups` multiplications and still zero
  bootstraps. That is what keeps the no-evaluation-keys property SIMD packing would
  forfeit.

  One subtlety worth carrying forward, because getting it wrong is silent:
  `output_position` is `(in_c − 1)·hw + …`, so **the answer's coefficient depends on
  the group's channel count.** Groups must be equal-sized with the final one
  zero-padded, or a short last group writes to different coefficients and the sum is
  nonsense that still decrypts. Both tests check against the *unsplit* plaintext
  convolution, which is what catches it — including under real encryption
  (`channel_splitting_stays_exact_under_encryption`, 4 channels of 5×5 at `n = 64`).

  **The cap is now a computed property, not folklore.** `bench::cnn::packing_plan`
  reports, per convolution, the group size and the multiplications per output channel;
  `the_channel_cap_is_where_the_packing_says_it_is` pins that `max_group_channels` at
  13×13 returns exactly **12**, that 13 does not fit, and that a 16-channel first conv
  is evaluable at 2 groups. So the constraint is checkable and a future widening cannot
  step over it silently.

  **Wired onto the hybrid path 2026-08-04.** All five sites are done —
  `weight_polys` emits one poly per (output channel, group), `hybrid_conv_open` takes
  one input ciphertext per group and sums the products, conv1 encrypts per group,
  conv2's `pack_shares_to_fhe_groups` slices the shares before re-encryption, and
  `n_weight_cts` follows from `polys.len()` automatically. A single `groups_for` is the
  only place the grouping is computed, because the weight cursor is **positional**: a
  disagreement between the packing site and the consuming site multiplies the wrong
  kernel by the wrong input group and yields a clean decryption of a wrong number.

  `a_conv_too_wide_to_pack_is_split_and_still_exact` runs a real two-conv model whose
  second convolution takes 16 channels at 13×13 — degree 5266 against `n = 4096` — as
  two groups: bit-exact against the plaintext reference, **0 bootstraps**, at the
  grouped multiplication count. `conv_mults` now counts per (output channel, group)
  rather than per output channel, so a split layer reports its real cost.

  **Retrained, and it paid off on organa** (2026-08-04). `conv(16,32)` takes organa
  from 81.64% [81.1, 82.2] to **82.93% [82.4, 83.5]** — disjoint intervals on 17 778
  test images — bit-exact under encryption with zero bootstraps, conv2 running as two
  groups of 12. Pneumonia gained nothing from the same widening (84.78% → 83.97%,
  overlapping intervals, validation up and test down): 4 708 training images overfit
  the extra capacity where organa's 34 561 use it. **So the packing limit was the
  ceiling only where there was data to exploit it**, which is a sharper claim than the
  docs previously made and needed both experiments to state.

  One latent coupling surfaced doing this and is worth remembering: the scheme-switch
  mask pool was sized from the weight-ciphertext count, on the invariant that "every
  ct×ct multiply is followed by exactly one switch". Splitting breaks it — a split conv
  multiplies per (output channel, group) and opens one summed result per output channel
  — so weights now outnumber masks. `MaskPool::assert_fully_consumed` caught it as an
  over-count rather than letting it become wasted offline work, and both counts now
  come from one traversal of the model, which `weight_polys`' own doc comment had
  already prescribed.

## 8. What to cut, or at least defer

* **Private RAG / encrypted LLM inference.** Fashionable, enormous, and a
  different paper. The non-private RAG for circuit construction (§5) is a *tool*,
  not a contribution — do not let it become a claimed one.
* **Federated learning.** Adjacent, crowded, and orthogonal to the cost-model
  thesis.
* **A real 512³ imaging pipeline.** State the scaling analytically instead.
* **Differential privacy.** Composes with all of this and belongs in future work;
  adding it now doubles the evaluation matrix for no new claim.
