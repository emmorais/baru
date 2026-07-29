# Baru MPC Compiler (`mpcc`) — long-term roadmap

**Goal.** Turn Baru from "a language with three privacy backends bolted on" into a
*protocol-agnostic secure-computation compiler* in the class of
[MP-SPDZ](https://github.com/data61/MP-SPDZ): one source program, many MPC
protocols, and a principled answer to the question **"which backend should I use
for *this* program?"** — for arbitrary programs, not just the ones in a fixed
benchmark suite.

Baru's differentiator over MP-SPDZ is that the same front-end already targets
**FHE (TFHE, BFV, CKKS)** and **ZKP (Binius64)** as well as MPC. So the
recommender is not just "MASCOT vs Semi2k", it is "MPC vs FHE vs hybrid" —
a question MP-SPDZ structurally cannot ask.

---

## 1. Where we are today

```
Lisp source
   │  parser.rs
   ▼
  Expr (AST)
   ├──────────────► CEK interpreter (eval.rs) ──► Backend trait ──► Aby3Backend
   │                                                             ├─ TransportBackend (TCP, real 3-party)
   │                                                             └─ GcBackend (Yao)
   └──► CircuitCompiler ──► FlatOp SSA ──► ZKP (Binius64)
                                        └─► FHE (TFHE / BFV / mock / CKKS-real)
```

Two separate lowering paths, and the MPC path never sees a circuit at all — it
interprets the AST and calls `Backend` methods one gate at a time. That is why
`src/bench/nn.rs` measured *flat* MPC throughput regardless of workload shape:
nothing batches, because there is no program-level view to batch over.

### What MP-SPDZ has that we do not

| MP-SPDZ | Baru today | Gap |
|---|---|---|
| High-level Python → **bytecode** → per-protocol VM | AST → per-backend interpretation | No protocol-neutral program representation |
| Vectorized instructions (SIMD width on every op) | scalar gates | No batching / vectorization |
| Explicit `startopen`/`stopopen`, round-merging optimizer | implicit, one round per op | Rounds are the dominant MPC cost and we do not optimize them |
| 3 domains: mod p, mod 2^k, GF(2); daBits/edaBits to mix | mod 2^k only (ABY3), GF(2) only (Yao), no conversions | Cannot express mixed-circuit protocols |
| ~30 protocols across {honest, dishonest} × {semi-honest, covert, malicious} | 1.5 (ABY3 + Yao) | Nothing to compare |
| Split offline/online measurement | one wall-clock number | Cannot attribute cost |
| `-v` cost breakdown, `--papers` | ad-hoc bench reports | — |

---

## 2. Target architecture

```
                     Lisp source (Expr)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      CircuitCompiler                 CEK interpreter
      (FlatOp SSA)                    (interactive REPL, unchanged)
              │
              ▼
   ┌────────────────────────┐
   │   mpcc::lower          │  domain assignment, SIMD packing,
   │   FlatOp ──► Bytecode  │  mixed-domain conversion insertion
   └────────────────────────┘
              │
              ▼
   ┌────────────────────────┐
   │   mpcc::schedule       │  dependence levels ⇒ ROUNDS
   │   round minimization   │  merge independent opens/muls
   └────────────────────────┘
              │
      ┌───────┴────────┬──────────────┬─────────────────┐
      ▼                ▼              ▼                 ▼
 mpcc::cost      VM: ring 2^k    VM: field p       VM: binary
 (analytic       (ABY3, Semi2k,  (Shamir, MASCOT,  (Yao, BMR,
  prediction)     SPDZ2k, Rep4)   LowGear, ATLAS)   Tinier)
      │                │              │                 │
      └────────────────┴──────┬───────┴─────────────────┘
                              ▼
                    mpcc::select — recommender
             (analytic cost + measured calibration ⇒ ranking)
```

Three pieces are new: **the bytecode**, **the round scheduler**, and **the cost
model**. The protocol VMs then slot in behind a single interface, which is what
makes "add a protocol" a bounded task instead of a rewrite.

### Why a new IR rather than reusing `FlatOp`

`FlatOp` is a scalar, untyped-domain, boolean-flavoured SSA designed for ZKP
constraint emission. It is a fine *input* to MPC lowering, but it cannot express
what MPC costing needs:

* **vectors** — `IMul` on 784 elements is one instruction and one round, not 784.
* **domain** — the same `IMul` costs 1 triple mod 2^k and ~64 AND gates in GF(2).
* **secrecy** — clear×secret is free; secret×secret costs a round.
* **openings** — the unit of communication is the *open*, and `FlatOp` has none.
* **preprocessing** — triples/bits/daBits/edaBits consumption is the offline cost.

So `mpcc::ir` is a new bytecode, and `mpcc::lower` is a faithful FlatOp → bytecode
translation. FlatOp stays the shared ZKP/FHE IR, unchanged.

---

## 3. Bytecode sketch (`mpcc::ir`)

Registers are *vectors* with a domain and a secrecy tag:

```rust
Domain::Ring { k: 64 } | Domain::Field { modulus_bits: 128 } | Domain::Binary { k: 1 }
Secrecy::Clear | Secrecy::Secret
```

Instructions split into **local** (no communication) and **interactive**
(consumes a round and preprocessing material):

| local | interactive |
|---|---|
| `LdI`, `Add`, `Sub`, `Neg`, `MulC`, `AddC`, `Xor`, `Not`, `ShlC`, `Copy` | `Mul`, `Open`, `And`, `BitDecomp`, `BitComp`, `LtS`, `EqZ`, `TruncPr`, `Convert` |

`Mux(c,t,e)` lowers to `t + c*(e-t)` — one `Mul`. `LtS`/`EqZ`/`BitDecomp` are kept
as *primitives* rather than expanded, because their cost is wildly
protocol-dependent (a garbled-circuit comparison is ~k AND gates in one round; a
ring comparison is an edaBit plus a carry chain; a field comparison needs random
bits and a statistical mask). Expanding them at lowering time would bake one
protocol's strategy into every protocol's cost.

Every instruction carries a `SimdWidth`, so a 784-element dot product is
`Mul(width=784)` + a log-depth `Add` tree — **one** communication round, matching
how a real protocol would run it.

---

## 4. Cost model (`mpcc::cost`)

For a scheduled program the model reports, per protocol:

* **rounds** — scheduler depth over interactive instructions
* **online communication** — bytes sent per party, per round
* **preprocessing** — Beaver triples, random bits, daBits, edaBits, OT counts
* **local work** — field/ring ops, AES calls (garbling), FFT/HE ops (LowGear)
* **derived wall-clock estimate** — under a `NetworkProfile`
  (`src/bench/comm.rs` already has latency/bandwidth simulation, reused here)

The point is that this is *analytic*: it predicts cost for a program without
running it, so we can answer "which backend for this program" cheaply and then
**calibrate** the constants against measured runs of the protocols we have
actually implemented. Publishing the calibration error is itself a result.

---

## 5. Protocol matrix (target)

Marked ✅ = implemented in Baru today, 🔨 = planned, in roughly the order below.

| Protocol | Domain | Parties | Majority | Security | Status |
|---|---|---|---|---|---|
| ABY3 / Replicated | ring 2^k | 3 | honest | semi-honest | ✅ `crypto/aby3_protocol.rs` |
| Yao GC | binary | 2 | dishonest | semi-honest | ✅ `interpreter/gc_backend.rs` |
| Dealer | ring 2^k | n≥2 | — (trusted dealer) | semi-honest | ✅ `mpcc/sim.rs` — the online-cost lower bound |
| Semi2k | ring 2^k | n≥2 | dishonest | semi-honest | ✅ `mpcc/semi2k.rs` — Dealer's online phase, triples from real OT (2-party) |
| Shamir | field p | n≥3 | honest | semi-honest | ✅ `mpcc/sim.rs` (GRR degree reduction) |

| Replicated (field) | field p | 3 | honest | semi-honest | 🔨 P2 |
| ATLAS | field p | n≥3 | honest | semi-honest | 🔨 P2 |
| SPDZ2k | ring 2^k | n≥2 | dishonest | malicious | 🔨 P2 |
| MASCOT | field p | n≥2 | dishonest | malicious | 🔨 P3 (OT-based triples) |
| LowGear | field p | n≥2 | dishonest | **semi-honest** | ✅ `mpcc/lowgear.rs` — triples from real BGV over CRT slots (2-party, no ZKPoPK) |
| HighGear | field p | n≥2 | dishonest | malicious | 🔨 P3 — shared threshold key + ZKPoPK |
| Rep4 | ring 2^k | 4 | honest | malicious | 🔨 P3 |
| Tinier / SemiBin | binary | n≥2 | dishonest | mal / semi | 🔨 P4 |
| BMR | binary | n≥2 | dishonest | semi-honest | 🔨 P4 |

Two of these were near-free for us: **Dealer** (no crypto, pure lower bound — and
the single most useful baseline for a cost-model paper) and **LowGear**, whose
somewhat-homomorphic preprocessing is what the FHE stack already does. LowGear
turned out to need one genuinely new piece — **CRT slot packing**
(`src/fhe/slots.rs`), because an OLE needs `n` *independent* products and every
other packing in the repo is coefficient-based, i.e. a convolution. It also
prefers BGV over BFV: with the message in the low bits there is no `Δ = q/t`, so
the plaintext modulus is free to be the prime that slots require.

---

## 6. Phasing

**P0 — foundation (this change).**
`mpcc::ir` bytecode, `mpcc::lower` from FlatOp, `mpcc::schedule` round
minimization, `mpcc::protocols` registry with capability + cost parameters,
`mpcc::cost` analytic model, `mpcc::select` recommender, `baru mpcc` CLI.
No new protocol implementations — the point is the framework, validated against
the two backends we already have.

**P1 — first new protocols.** *In progress.*

Done: the bytecode VM (`mpcc/vm.rs`) with a `ShareEngine` trait — a protocol
supplies sharing, opening, multiplication and correlated randomness, and the
driver composes the bytecode from those, so adding a protocol is one trait
implementation rather than one VM. Two engines behind it (`mpcc/sim.rs`):
**Dealer** (additive mod 2^k, Beaver) and **Shamir** (prime field, GRR degree
reduction) — structurally different in exactly the way that matters for costing,
one sending two elements per peer per multiplication and consuming a triple, the
other sending one and consuming none. `baru mpcc calibrate` runs a program under
both and tables measured against predicted cost, with the answer checked against
the plain interpreter.

Current status: online rounds, bytes and triple counts match the model exactly
on the multiplicative core. That validates the *bookkeeping* — round scheduling,
peer scaling, Beaver-vs-GRR volume — not the wall-clock constants, which need a
real network (P1 remainder).

**Offline volume is now measured for the OT-based protocols (2026-07-28).**
`src/ot/` implements base OT + IKNP and `baru bench-ot` measures it, so Semi2k,
SPDZ2k and MASCOT no longer carry hand-entered `offline_elems_per_triple`
constants: `mpcc::ot_cost` derives their offline bytes from measured OT cost
(16.5 B/OT at large batch) times a protocol-level OT count (Gilboa: `2k`
correlated OTs per triple, doubled where a triple is sacrificed, widened by
SPDZ2k's `s` bits). `estimate` now says "derived from MEASURED OT cost" for those
rows instead of disclaiming a literature estimate.

That changed a **ranking**, in the direction the literature predicts: on a
3-multiply program LowGear's SHE preprocessing moves ~384 B per triple against
9.2 KiB (Semi2k), 24.4 KiB (SPDZ2k) and 48.8 KiB (MASCOT) — a gap the old
constants understated by an order of magnitude. A test pins it. Note this is the
naive Gilboa route on the OT side, so it is an upper bound (vector OLE and
batched sacrifice both do better).

**LowGear landed 2026-07-29** (`src/mpcc/lowgear.rs`, `baru bench-triples`), and
the CPU half that "remains flattering" above was measured and came out the *other
way*. Running the protocol — BGV over CRT slots, one ciphertext×plaintext
multiply per cross term — gives **416 B/triple at a 64-bit field** (104 B at its
native 17-bit field × 4 CRT limbs), within 8% of the 384 B estimate, **and ~4.5×
faster per triple than the OT route**. `mpcc::cost` used to assert "LowGear trades
bandwidth for CPU: its preprocessing is compute-bound"; that note was never
measured and is wrong at these parameters. It has been replaced by the
measurement. Both `semi2k` and `lowgear` are now in `RUNNABLE` and calibrate at
**0% on rounds and online bytes**.

Two limits bound the LowGear claim. The engine is **semi-honest** — no MACs, no
sacrifice, no ZKPoPK — so it is the LowGear analogue of Semi2k, not of SPDZ. And
its field is ~17 bits, forced by the noise-flooding budget at this crate's
`u128`-capped `q ≤ 2^127` (Overdrive uses 250–350-bit `q` at `n = 16384–32768`);
that is too small for masked bit-decomposition, so comparison-shaped programs are
refused rather than run wrong (`mpcc::engine_can_run_program`).

**Semi2k landed 2026-07-28** (`src/mpcc/semi2k.rs`): Dealer's online phase with
triples from real OT (Gilboa OLE over `ot::iknp::correlated`). It is in `RUNNABLE`
and calibrates at 0% error online; more importantly its **offline bytes are
measured** — 3104 B/triple against `ot_cost`'s predicted 3136, a 1% agreement that
validates the Gilboa derivation. Note `engine_supports`: the protocol is n-party
but this implementation is 2-party, because Gilboa OLE is pairwise, and the registry
describes the protocol while `engine_supports` describes what is built.

**The measurement harness landed 2026-07-28** (`baru bench-net`): the cost formula
`rounds × latency + bytes / bandwidth` validated against a real loopback TCP socket
to within 13.6% worst case and under 1% mid-range, with the link's own constants
measured rather than assumed. So the *shape* of the model is no longer an
assumption. What remains is that the constants are loopback's, not a WAN's.

Remaining: the existing ABY3 transport backend re-expressed as a `ShareEngine`,
WAN-link constants measured on a real WAN,
a CPU term for preprocessing (`ot_cost::US_PER_OT` is measured and recorded but
unused), and an n-party Semi2k (n(n−1) cross terms).

**P2 — comparisons, malicious security & mixed circuits.** *Comparisons done.*

Done: shared random bits are a `ShareEngine` primitive, and
`vm::bit_decompose_batch` implements masked-opening bit decomposition with a
**Kogge–Stone parallel-prefix borrow chain** — `log₂(nbits)` rounds instead of
the `nbits` a ripple-borrow chain would need. Two details make it cheap: the
minuend is public after the opening, so generate/propagate are *local*; and
`gᵢ·pᵢ = 0` always, so the `OR` in the prefix combine is a plain addition and
each combine is two independent multiplications. `LtS`, `LtU`, `EqZ` and
`TruncPr` all run on top of it, batched — ten comparisons at one dependence
level cost one comparison's rounds.

`relu` and `max` now execute under Dealer and Shamir and agree with the plain
interpreter. The cost model was corrected to match the implemented algorithm
(`cost::bit_decompose_muls` mirrors the VM instruction for instruction), and
`baru mpcc calibrate` reports 0% error on rounds, multiplications and bytes for
comparison-heavy programs. Note what that does and does not show: it shows the
model and the implementation agree about the *algorithm*, not that either
matches a real network — that still needs P1's remaining measurement harness.

A second cost-model correction landed 2026-07-28, on the *binary* side this
time. `Program::is_interactive` is domain-agnostic — an `Add` is local whatever
the protocol — so in a binary protocol the adders are costed outside the round
schedule, and their AND-depth was being charged as the **maximum over the
program** rather than the longest path through it. A chain of `n` adders was
therefore priced as one adder. That under-priced deep *local* arithmetic
specifically: fixed-point rescales and adder trees, which is the term §7's
MPC-vs-GC comparison turns on. `estimate` now accumulates that depth along the
dependence DAG; on an 8-element addition chain the Semi-Bin estimate goes from
251 ms to 1.75 s (7×). Arithmetic protocols are untouched, which is why the
`calibrate` tests (Dealer/Shamir) still report 0% error.

Remaining: secret division (restoring long division on top of the comparison
machinery — still refused by the VM, with an error naming this phase), SPDZ2k
MACs, replicated field, ATLAS, and daBit-based `Convert` for arithmetic↔binary
mixed circuits.

**P3 — preprocessing back-ends.** *LowGear landed 2026-07-29* (`mpcc/lowgear.rs`,
`baru bench-triples`) — see P1 below for the measurement, including the modelling
assumption it falsified. Remaining here: MASCOT (OT-based, malicious), HighGear,
and the malicious layer LowGear deliberately omits (MACs, sacrifice, ZKPoPK).
Split offline/online measurement in the harness (MP-SPDZ's `-v`) is still open.

**P4 — binary protocols and the full matrix.** Tinier, SemiBin, BMR; Yao folded
into the same VM interface.

**P5 — the recommender as a result.** *Per-operation assignment landed
2026-07-28* — `src/mpcc/assign.rs`, prompted by SING (USENIX Sec'26); see
`docs/related-work.md`. Note that `mpcc::select` never existed; the whole-program
ranking was `mpcc compare`. Assignment is per operation *kind* and MPC-only so
far.
 Given an arbitrary Baru program and a
deployment description (party count, threat model, LAN/WAN profile), output a
ranked backend list with predicted cost and a confidence interval from the
calibration data. Validate on programs *outside* the training suite — that is the
publishable claim, and it is a claim MP-SPDZ does not make.

---

## 7. Interaction with the rest of Baru

* **FHE/ZKP stay in the comparison.** `mpcc::select` ranks FHE backends
  alongside MPC ones using the measurements in `src/bench/`. A program with no
  comparisons and low depth should come back "CKKS"; a comparison-heavy one
  should come back "ABY3 on LAN, Yao on WAN".
* **`src/hybrid/`** already switches MPC↔FHE mid-computation. Once the bytecode
  exists, that becomes a *compiler pass* (split the program at the cheapest cut)
  rather than a manual annotation.
* **`src/db/` (BaruDB)** stores every run, so calibration data accumulates across
  months of experiments instead of living in scrollback. See
  [barudb-api.md](barudb-api.md).

## 8. Non-goals

* Bit-for-bit MP-SPDZ compatibility (neither bytecode nor `.mpc` source).
* Beating MP-SPDZ on raw protocol throughput. We are after *comparability and
  prediction*, not micro-optimized implementations.
* Reimplementing every MP-SPDZ protocol before the framework is useful — the
  registry deliberately carries unimplemented protocols with cost parameters
  only, so the analytic comparison covers the whole matrix from day one and each
  implementation *validates* one row of it.
