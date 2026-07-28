# Related work, and what Baru took from it

Read 2026-07-28. Two sources, both squarely on Baru's path, with opposite lessons:
one says Baru is *behind* the state of the art, the other says a thing Baru just
parked was parked correctly.

---

## 1. SING — hybrid MPC protocol assignment with GNNs

**Blüml, Huppert, Khayata, Schmidt & Schneider (TU Darmstadt), USENIX Security
2026.** <https://eprint.iacr.org/2026/1237>

### What it says

The problem is **protocol assignment**: given a program, decide which MPC
protocol (sharing type) evaluates each *part* of it, and pay for the conversions
between parts. The lineage is HyCC (CCS'18) → **Silph** (Chen et al., S&P'23) →
SING.

* Silph solves the assignment with **Integer Linear Programming**, plus
  deployment-specific cost models the authors describe as "cumbersome to retune".
* ILP forces a **linear** cost model — cost must be a sum of per-operation terms.
* SING replaces the solver with a graph neural network. Two modes: *imitation*
  (reproduce Silph's assignments, reported up to **76 697×** faster at comparable
  quality) and *cost-driven* (train a GNN cost predictor on measured runtime and
  communication, then a second GNN to minimise it). The stated advantage is
  expressive **non-linear** cost models and cheap retargeting — re-fit the
  predictor rather than reformulate the ILP.
* They also release 704 MPC circuits with assignments as a benchmark.

### Where this left Baru: behind

Baru's comparison answered *"which single protocol for this whole program?"*
(`baru mpcc compare`). The literature answers the strictly more general
per-operation question. Worse, the roadmap and the agenda both referred to a
`mpcc::select` recommender **that did not exist** — the ranking was whole-program
only, and per-operation assignment had never been attempted.

Two prerequisites turned out to already be in place: `RegInfo` carries a
**per-register domain**, so the IR can represent a mixed program; and `Op::Convert`
is **costed** (one open round plus daBits). It was simply never emitted.

### What Baru took: `src/mpcc/assign.rs` + `baru mpcc assign`

Per-operation-kind assignment between an arithmetic and a binary protocol, with
conversions charged at the crossings, searched greedily against the exact cost
model. It reproduces the classic hybrid-MPC finding *from Baru's own model*
rather than assuming it (WAN, 2 parties, semi2k / semi-bin):

| program | all-arith | all-binary | mixed | moved |
|---|---|---|---|---|
| `(* (+ a b) (> c d))` | 550 ms | 800 ms | **500 ms** | `cmp-lt → binary` |
| `(+ (* a b) (+ (> c d) (> e f)))` | 501 ms | 550 ms | **400 ms** (1.25×) | `cmp-lt → binary` |
| `(if (> x 0) x 0)` | 550 ms | **350 ms** | 350 ms | — pure binary wins |
| `(dot (list a b c d) '(3 5 9 17))` | 100 ms | **50 ms** | 50 ms | — pure binary wins |

Multiplications stay arithmetic, comparisons go binary, and on the two programs
where one sharing suits everything it correctly declines to mix. The online
traffic drop is the sharper signal: 9.5 KiB → 41 B on the second row, because the
arithmetic comparison's masked-opening bit decomposition is what generated that
traffic in the first place.

### The interesting part: Baru needs neither the ILP nor the GNN

Baru is in exactly the regime that breaks Silph's formulation, and it does not
need SING's surrogate either, because it *has* the exact model and evaluating it
is cheap. Two non-linearities make the objective **non-separable** in the
assignment variables:

1. A dependence level costs the **maximum** over the instructions in it, not their
   sum — ten comparisons at one level cost one comparison's rounds. Moving one
   operation to the other side may change a level's cost by nothing or by the
   whole difference, depending on what else is at that level.
2. Conversions **batch per level** for the same reason, so a crossing's cost
   depends on whether another crossing already happens there. The measured table
   above shows this directly: six conversions landing at *one* level.

So the approach is the one `mpcc::opt` already uses for rewrites — propose,
evaluate the real objective, keep it if it improves. What Baru gives up is any
optimality guarantee; optimal assignment is NP-hard, which is why Silph reached
for ILP. SING's contribution is orthogonal and would still apply: if Baru ever
needs assignment over a space too large to evaluate exactly, a learned predictor
over `mpcc::cost` is the documented way to do it.

### Not taken, deliberately

* **Per-instruction / per-block granularity.** Baru assigns per operation *kind*.
  Coarser than Silph and SING, but the space is a few dozen kinds instead of `2ⁿ`,
  and the answer is *interpretable* — "comparisons went binary" is a statement
  about the cost model, which is what this repo is for. Per-instruction is the
  obvious next step.
* **The GNN.** No training data, no need for a surrogate while the exact model is
  cheap to evaluate, and a learned cost model would undermine the provenance
  discipline the rest of the repo runs on (`docs/gadget-costs.md`).
* **Their benchmark suite.** 704 circuits would be a good calibration target and
  is not a dependency worth taking on today.

### The extension nobody else can make

HyCC, Silph and SING are **MPC-only** — arithmetic, boolean and Yao sharings.
Baru's cost model spans MPC *and* FHE. **Done, same day:** `src/mpcc/fhe_cost.rs`
adds FHE as a third assignment target, and on a 256-wide encrypted layer with a
comparison the search returns `mul → fhe`, `cmp-lt → binary` — the hybrid
architecture, derived from measured costs rather than annotated. Full result and
the three modelling errors it took to get there in `research-agenda.md` §4.

The crossover is the part worth showing a reviewer: narrow layers (8, 64 muls)
stay on MPC because batched multiplies cost one round; wide ones (256, 512) move to
FHE because MPC's triple preprocessing grows linearly while a packed ciphertext is
flat. Neither Silph nor SING can express that comparison at all, because FHE is not
in their space.

---

## 2. CirC — compiler infrastructure for circuits

**Ozdemir, Brown, Wahby et al., S&P 2022.** <https://github.com/circify/circ>,
paper at <https://cs.stanford.edu/~aozdemir/docs/circ.pdf>

### What it says

A shared IR (`src/ir`: bit-vector and prime-field terms, typed) with multiple
frontends (ZoKrates, C) and multiple backends (R1CS/bellman, SMT via CVC4/cvc5,
ILP, MPC). Passes include constant folding, **n-ary flattening**, function
inlining, tuple elimination, and two memory passes:

* `opt/mem/obliv.rs` — **oblivious array elimination**: when every index into an
  array is independent of secret data, the array compiles away entirely.
* `opt/mem/lin.rs` — **linear-scan array elimination**: when an index *is* secret,
  the access becomes an `n`-way multiplexer selected by the index.

### What this validates

* **The architecture.** CirC's one-IR-many-backends design is Baru's `FlatOp` +
  `mpcc::ir` design. Independent arrival at the same structure is reassuring, and
  CirC's frontend/backend split is a good model for where Baru's own boundary
  should sit.
* **Parking ORAM was right.** `research-agenda.md` §7 parked Path ORAM on the
  argument that a linear scan beats `O(log² n)` per access until the data is too
  big to stream. CirC does not implement ORAM at all — it implements exactly the
  two cases either side of that line, and reaches for a mux tree when the index is
  secret. That is a stronger endorsement than the reasoning alone.
* **`n-ary flattening`** is the shape of the `reassociate` rule in `mpcc::opt`:
  flatten an associative chain into an n-ary node, then rebuild it balanced.
  Already done, arrived at independently.

### What Baru should take next (not done yet)

**Secret-index array access, priced.** Baru's lists are resolved at compile time
(`CVal`, static-if), so a *secret* index is currently not expressible at all. The
CirC recipe is the right one and is small:

1. Public index → direct addressing, free.
2. Secret index → linear scan, `n` equality tests and `n` muxes, which `FlatOp`
   can already express with `ICmpEq` and `Mux`.
3. Let `mpcc::cost` price it, and only consider ORAM when the linear scan is the
   bottleneck — with the cost model saying so, not a guess.

This is what would give the cost model the data-dependent-addressing entry the
agenda said ORAM was supposed to provide, at a fraction of the cost. It belongs
ahead of ORAM in any future sequence.

### Not taken

* CirC's frontends (ZoKrates, C). Baru has its own language; adopting a frontend
  is a different project.
* Its SMT and ILP backends. Verification of the compiler is interesting but is not
  on either paper's path.
