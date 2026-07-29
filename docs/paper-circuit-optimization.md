# Paper plan: Baru as a language for private AI

**Status: Phases 0–4 done; draft paper in `../paper-circuits/`.** Written
2026-07-28 after auditing what the repo actually measures, then updated as each
phase landed. Numbers below were re-verified against the current tree
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

### ~~Phase 4 — assembly~~ DRAFTED 2026-07-28

`paper-circuits/main.tex` — *One Program, Many Circuits: Backend-Aware Circuit
Optimization for Private AI*. 8 pages, builds clean, citations resolve.
`paper-circuits/README.md` maps every claim to the command that produces it.

Structure: the motivating story is our own experts serializing a layer (§1);
parity → advantage on a whole network (§4.2); the separable-axes ablation
(§4.3); the cross-application matrix as the central figure (§4.5); related
work positioned against HyCC/Silph/SING (MPC-only assignment) and CirC
(one-IR-many-backends, but no per-backend divergence under a shared cost
model).

**Two `\TODO` markers are left in the text deliberately** — see the
non-reproduction below and the workload-scale limitation. They are in
§Limitations where a reader will find them, not hidden.

#### The `mul -> fhe` claim: RESOLVED — it reproduces, on WAN

Recorded in `research-agenda.md` §4 and `related-work.md`: protocol assignment
places **`mul -> fhe`** for a 256-wide encrypted layer, reproducing the
hand-designed hybrid-inference architecture from measured costs.

I first reported this as **not reproducing**. That was wrong: I re-ran it on
the CLI's **default LAN** profile against a claim derived on **WAN**. On the
identical expression:

```sh
E='(if (> <256-wide dot, both operands secret> 0) 1 0)'
baru mpcc assign --expr "$E" --parties 2 --dishonest --lan
#   cmp-lt -> binary                      1.57x   (offline 885 KiB)
baru mpcc assign --expr "$E" --parties 2 --dishonest --wan
#   cmp-lt -> binary,  mul -> fhe         1.26x   (offline 101 KiB)
```

`the_model_derives_the_hybrid_split_for_a_neural_layer` in `src/mpcc/assign.rs`
pins the WAN case and was passing the whole time — checking the test first
would have cost one grep and saved the false alarm.

**This is a better result than the original claim, and it is now §4.6 of the
paper.** The assignment is *deployment-dependent*: FHE needs no preprocessing,
so moving the layer to FHE cuts the triple bill 1.7 MiB -> 101 KiB, and that
saving buys wall-clock only when bandwidth is scarce. Same program, same
widths, same threat model, different link -> different answer. That is the
paper's thesis on a second axis: §4.5 shows the right *circuit* depends on the
backend; §4.6 shows the right *backend* depends on the deployment.

**The lesson, which is already in this repo's history in another form:** a
prediction and a measurement must name the same protocol *and the same
deployment* before they can be said to disagree. The failure mode is quiet —
the run succeeded, produced a plausible assignment, and answered a different
question. When a recorded claim appears not to reproduce, **find its test
before doubting it**.

### Phase 4 (original notes)

- Positioning against HyCC → Silph (ILP) → SING (GNN) for assignment, and CirC
  for one-IR-many-backends. See [`related-work.md`](related-work.md).
- **Baru's defensible difference:** the others assign *among MPC protocols*.
  Baru's cost model spans **MPC, FHE and ZKP**, so it can ask "should this be
  an FHE program at all?" — a question the others cannot pose.
- Honest scope: semi-honest, honest-majority; MedMNIST is downsampled;
  absolute times are single-threaded and unoptimized NTT.

---

## 5b. The FHE bridge — scope (b), first step landed 2026-07-28

`src/mpcc/mixed.rs` executes **one circuit across two backends**, routing
operations by an assignment and inserting the scheme switch at every crossing.
This turns `mpcc::assign` from a costing result into an executable one: the
model says `mul -> fhe`, and now that can be run and checked rather than
believed.

**What works.** A dot product with both operands secret — the shape the cost
model puts on FHE — runs its multiplications under BFV and its result back
through mask-and-open into MPC shares. Verified: mixed == pure-MPC ==
plaintext, bit for bit, with every multiply on FHE, **zero demotions**, and
exactly one switch back.

**Depth-1 is enforced, not assumed.** BFV here has no relinearization, so a
product is a degree-2 ciphertext that cannot be multiplied again. When a
multiplication's operand is already a product the executor **demotes** that
operation to MPC and counts it (`MixedStats::demoted_muls`), because a silent
fallback would make the FHE assignment look better than it is. A test pins the
demotion on `(* (* a b) (* c d))`.

**Honest scope, and it is a real limit.** The MPC→FHE direction uses
`Backend::export_additive`, which only an in-process simulator satisfies — a
real party can produce its own share component and nobody else's. The
cryptography is real (BFV at the configured parameters; a genuine
mask-and-open in the other direction) but the **party separation of the bridge
is simulated**. Conversion counts and crossing bytes are meaningful; switch
wall-clock is not a deployment number. `bench::cnn` does the deployable
version, with each party encrypting its own component.

**Why this is the interesting direction.** With execution mixed, the boundary
becomes an optimization target rather than a fixed architectural choice: the
compiler can be asked *where* to switch, not just *what* to run where. That is
the natural next rewrite rule, and it is not expressible while the hybrid
lives in hand-written orchestration.

**(i) and (ii) done 2026-07-28.** `mixed::from_assignment` translates the cost
model's decision into the executor's routing, and `baru bench-mixed-layer`
runs it under both deployments. At n=256, both exact against plaintext:

| network | model's decision | mpc ops | fhe ops | switches out/back |
|---|---|---|---|---|
| LAN | (all arithmetic) | 511 | 0 | 0 / 0 |
| WAN | `mul -> fhe` (1.18×) | 0 | 511 | 512 / 1 |

The loop is closed: the model decides, the executor carries the decision out,
and the harness fails if the routing is not honoured (`mul -> fhe` with zero
FHE operations, or FHE operations under an all-arithmetic decision). Below
about n=64 both deployments keep the layer on MPC, which is the documented
crossover — a batched multiply is one round, so narrow layers do not pay for
the switch.

**Two gaps this exposed, both worth having.**

1. ~~**The executor converts scalar-wise; the cost model charges crossings
   *packed*.**~~ **Closed 2026-07-28** by `mixed::packed_dot`: one vector is
   packed forward and the other reversed, so a single polynomial
   multiplication puts the whole dot product in coefficient `d-1`. Cost is
   **two crossings out, one multiplication, one masked opening back —
   independent of layer width**, against `2n` crossings and `2n-1` operations
   scalar-wise. Verified equal to the plaintext dot product.

   This surfaced a constraint worth stating in the paper, because it is easy
   to miss and fails *silently*: **the plaintext modulus must hold un-reduced
   share products.** Each party's share is uniform mod `2^nbits`, so the FHE
   side does not multiply `a` and `b` but `a + k·2^nbits` and `b + m·2^nbits`.
   The product is still congruent to `a·b` mod `2^nbits` — which is why the
   construction is correct — but as an integer it is far larger, and summing a
   whole layer of them overflows a modulus sized for the *values*.
   `packed_dot` computes the bound and refuses rather than wrapping; a test
   pins the refusal. This is a real cost of switching secret-shared data into
   FHE and it does not appear when the operands are plaintext weights.
2. **Seconds are still not validated.** The MPC side is an in-process
   simulator and the bridge is party-simulated, so the model's predicted
   *crossover in time* remains unmeasured. Counts and bytes are real.

**(iii) done 2026-07-28 — `Placement::sink_to_fhe`.** The kind-level
assignment is coarse in a way that costs crossings: an operation the FHE side
*could* host, sitting between two FHE results, drags both back across the
boundary purely because its *kind* was assigned to MPC. The rule refines the
decision per operation:

> An operation runs on FHE if the FHE side can host it, its depth budget
> allows it, and **every operand is already produced on the FHE side.**

The last clause is what makes it safe to apply greedily — it never forces a
fresh value across, it only avoids dragging values back. On two dot products
feeding one addition, the addition sinks and one masked opening disappears; a
test pins that the value is unchanged and the crossing count falls. On the
256-wide layer, `bench-mixed-layer` reports **255 operations sunk** and one
switch back instead of one per partial sum.

This subsumed an earlier heuristic in `from_assignment` ("if `mul` went to
FHE, put `add` there too"), which has been **removed** — a hack sitting
alongside the rule would mask which one did the work.

**Why this is the move MPC-only compilers cannot make.** Silph and SING assign
per instruction, but over MPC sharings only; the boundary this rule moves is
the MPC/FHE representation boundary, and it is only expressible once execution
is genuinely mixed. While the hybrid lives in hand-written orchestration the
boundary is wherever the author drew it.

#### Wall-clock, and a comparison that does not test what it looks like it tests

`bench-mixed-layer --deployed` measures the same layer on the **real 3-party
transport**, with the crossing done per party (`my_additive_component`), three
encryptions per operand, and the ciphertexts charged latency and bandwidth —
the accounting `bench::cnn` uses. At n=256, W=10:

| network | route | wall-clock | mpc msgs | link bytes |
|---|---|---|---|---|
| rtt 0 | all-MPC (aby3) | **7.9 ms** | 8 | 0 |
| rtt 0 | layer on FHE | 201.0 ms | 4 | 983 k |
| rtt 20 ms | all-MPC (aby3) | **75.3 ms** | 8 | 0 |
| rtt 20 ms | layer on FHE | 234.7 ms | 4 | 983 k |

**This neither validates nor refutes the model's `mul -> fhe` prediction, and
it would be easy to report as if it did.** The measured MPC route is ABY3:
3-party, honest majority, **triple-free** (PRF resharing). The prediction was
computed for semi2k: 2-party, dishonest majority, one Beaver triple per
multiplication. FHE's entire case there is the offline bill it avoids — and
ABY3 has no offline bill, so there is nothing to amortise against. Applied to
ABY3 the model says *stay on MPC*, which is exactly what is measured.

This is the second time this trap has been walked into in this repo; the first
is recorded in the notes as *"a prediction and a measurement must name the
same protocol before they can disagree."* It is worth stating in the paper as
a methodological point rather than buried, because the failure mode is that
both numbers look reasonable.

**What the measurement does establish**, and it is worth having: against an
honest-majority triple-free MPC baseline, hosting a layer on FHE **loses** —
25× at zero latency, 3.1× at RTT 20 ms. That is the same reason every deployed
FHE+MPC hybrid is 2-party.

#### semi2k wired in — and it exposes a probable modelling bug

`bench-mixed-layer --deployed` now runs four routes on the same layer
(n=256, W=10, RTT 20 ms):

| route | wall-clock | triples | online B | offline / crossing B |
|---|---|---|---|---|
| all-MPC (aby3, 3pc honest) | **54 ms** | 0 | 37 088 | 0 |
| layer on FHE (3pc crossing) | 240 ms | — | 32 880 | 983 040 |
| layer on FHE (2pc crossing) | 169 ms | — | 0 | 720 896 |
| all-MPC (semi2k, 2pc dishonest) | **142 ms** | 256 | 2 050 | 125 184 |

The last two are a **controlled** pair — same party count, same width, same
layer. semi2k is faster *and* moves **5.8× fewer bytes** (125 KB of triples
against 721 KB of ciphertext). The model predicted FHE wins by 1.18×.

**Probable cause, checked by reading the code rather than guessed.**
`assign.rs:479` charges an MPC→FHE crossing as
`ct_bytes * ceil(registers / slots)` per level — *one packed ciphertext* —
with no party-count factor. But switching a **secret-shared** value into FHE
needs one ciphertext **per party**, because each party encrypts its own share
component and the evaluator sums them. At two parties and two operands that is
four ciphertexts, not one.

This is the complement of a bug the notes already record: crossings were once
charged *per edge* and the fix was to pack them per level. The fix appears to
have packed away the party dimension too.

**Do not act on this without closing the caveat.** The model's decision was
computed at 32-bit and this measurement runs at 10-bit, and semi2k's triple
cost scales with the width (measured ≈489 B/triple at W=10 against the
recorded ≈3104 B/triple at 64-bit — consistent, but not the same point). A
like-for-like test needs the measurement at the model's width.

**This matters for the paper: §4.6's `mul -> fhe` result depends on the
crossing charge.** If the party factor is missing, the FHE side is being
under-charged and the WAN decision may not survive the correction. Verify
before the claim is defended, and note it in Limitations meanwhile.

#### (i) attempted: the 32-bit configuration cannot be executed at all

Re-running the controlled pair at the model's 32-bit width is **not possible
with this BFV parameter set**, and the reason is cryptographic rather than an
implementation shortcut.

Switching a *secret-shared* value into FHE means the FHE side multiplies
un-reduced share sums: not `a·b` but `(a + k·2^W)(b + m·2^W)`. Correct modulo
`2^W` — which is why the construction works — but as an integer it is
`(parties·2^W)²` per term, and a layer sums `n` of them. Required plaintext
modulus:

| W | n=64 | n=256 |
|---|---|---|
| 10 | 2^29 | 2^31 |
| 16 | 2^41 | 2^43 |
| 24 | 2^57 | 2^59 |
| 32 | 2^73 | 2^75 |

Past about `t = 2^32` the depth-1 noise budget at `n=4096, q=2^100` is
exhausted and the route returns a *wrong value* rather than failing. Growing
the ring would buy headroom, but `fhe::ntt` caps at `MAX_N = 4096`. Measured
envelope: **W=10 works at n=256; W=12 already fails at n=64.** The benchmark
now refuses with that explanation instead of panicking or returning a wrong
number.

**Why this is a real finding and not a parameter-tuning gripe.** The blow-up
comes specifically from **both operands being secret**. The classic hybrids
(Gazelle, Delphi, Cheetah) keep the weights in *plaintext*, so only
activations cross and the product is ciphertext×plaintext — no squaring of
share magnitudes, and the modulus stays small. Encrypting the model too, which
is the distinguishing feature of the companion switch-as-bootstrap paper, is
exactly what makes this crossing expensive.

**Where that leaves §4.6.** The `mul -> fhe` decision is computed at 32-bit,
for a crossing this parameter set cannot perform on secret-shared operands. At
the one width where it *is* implementable (W=10), semi2k beats FHE on both
wall-clock and bytes. So the claim is now questionable on two independent
grounds — the possibly-missing party factor in the crossing charge, and
feasibility at the claimed width. **It should not be defended until both are
resolved.**

#### Resolved: the model now has a plaintext-modulus budget

`FheSpec::log2_plaintext` (measured ceiling 2^32 for `BFV_DEPTH1`) plus a
feasibility gate in `fhe_op_cost`: a ct×ct product of secret-shared operands
needs `2·(bits + log2 parties)` bits of plaintext modulus, and the assignment
**refuses FHE** when the scheme cannot provide it.

| network | share width | assignment | speed-up |
|---|---|---|---|
| LAN | 12-bit | `cmp-lt -> binary` | 1.16× |
| LAN | 32-bit | `cmp-lt -> binary` | 1.57× |
| WAN | 12-bit | `cmp-lt -> binary`, `mul -> fhe` | 1.10× |
| WAN | 32-bit | `cmp-lt -> binary` | 1.19× |

The gate's threshold (`W ≤ 15` at two parties) agrees with the measured
envelope (W=10 works at 256-wide, W=12 fails at 64-wide) **without having been
fitted to it** — the model was given the parameter ceiling, not the answer.

**Two tests had to change, and they were wrong before.**
`the_model_derives_the_hybrid_split_for_a_neural_layer` and
`the_threat_model_decides_whether_the_layer_goes_to_fhe` both pinned
`mul -> fhe` at 32-bit — an assignment that silently computes the wrong value.
They now run at 12-bit (the point of the second is the *threat model*, not the
width), and a new test
`a_realistic_share_width_puts_the_layer_out_of_fhes_reach` pins the refusal.

**The lesson worth carrying:** the model had a *depth* budget but no
*modulus* budget, so it could rank an assignment it could not perform. A cost
model spanning several backends needs feasibility gates per backend, not only
prices — otherwise the cheapest option is sometimes the impossible one.

#### FINAL: `mul -> fhe` is retracted; the model and execution now agree

Adding **ring width** to the model's vocabulary (`--domain ringN`, 2..=128)
was the unlock. The one configuration the FHE route could actually be executed
in was one the model could not name — which is what produced four apparent
model errors that were all configuration mismatches.

Pointed at the same configuration at last:

| ring | model (semi2k offline) | measured | agreement |
|---|---|---|---|
| 10-bit | 112.7 KiB | 122.3 KiB | **8%** |

And the assignment answers consistently everywhere:

| ring | ct×ct on FHE? | why |
|---|---|---|
| 10-bit | feasible, **not chosen** | priced and loses — matches execution, where semi2k beat FHE |
| 32-bit | **refused** | needs 2^66 plaintext modulus; parameters give 2^32 |
| 64-bit | **refused** | needs 2^130 |

**`mul -> fhe` is retracted.** It only ever appeared at configurations that
were infeasible (the gate now catches them) or mismatched. Two tests pinning
it are rewritten to assert the refusal and its reason; they had passed for
months because the model had no modulus budget and could rank an assignment it
was unable to perform.

**A gate bug of mine, caught the same way.** My first version gated on the
*value* width, so `--bits 10 --domain ring64` looked feasible. Share
components are uniform over the **ring**, so the gate must use ring width. The
corrected gate refuses at ring32 and ring64.

**What survives:** `cmp-lt -> binary` — the nonlinearity half of the classical
hybrid split — still derived, still deployment-dependent.

**Why this does not contradict Gazelle.** Those systems keep weights in
plaintext (ct×plaintext squares no share magnitude) *and* work in a two-party
client/server setting. In the multiparty setting this model represents, a
public-constant multiply is already local, so FHE has nothing to win there
either — verified directly.

**§4.6 is now a negative result, and it is stronger than the positive one it
replaces:** a cost model spanning MPC and FHE can say not only which side is
cheaper but whether the crossing is *possible at all*, and for
both-operands-secret layers on depth-1 BFV the answer is no.
