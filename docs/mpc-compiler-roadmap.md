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
| Dealer | ring 2^k | n≥2 | — (trusted dealer) | semi-honest | 🔨 P1 — the online-cost lower bound |
| Semi2k | ring 2^k | n≥2 | dishonest | semi-honest | 🔨 P1 |
| Shamir | field p | n≥3 | honest | semi-honest | 🔨 P1 (already on the roadmap) |
| Replicated (field) | field p | 3 | honest | semi-honest | 🔨 P2 |
| ATLAS | field p | n≥3 | honest | semi-honest | 🔨 P2 |
| SPDZ2k | ring 2^k | n≥2 | dishonest | malicious | 🔨 P2 |
| MASCOT | field p | n≥2 | dishonest | malicious | 🔨 P3 (OT-based triples) |
| LowGear / HighGear | field p | n≥2 | dishonest | malicious | 🔨 P3 — **reuses our BFV** (`src/fhe/bfv.rs`) |
| Rep4 | ring 2^k | 4 | honest | malicious | 🔨 P3 |
| Tinier / SemiBin | binary | n≥2 | dishonest | mal / semi | 🔨 P4 |
| BMR | binary | n≥2 | dishonest | semi-honest | 🔨 P4 |

Two of these are near-free for us: **Dealer** (no crypto, pure lower bound — and
the single most useful baseline for a cost-model paper) and **LowGear**, whose
somewhat-homomorphic preprocessing is exactly what `src/fhe/bfv.rs` already does.

---

## 6. Phasing

**P0 — foundation (this change).**
`mpcc::ir` bytecode, `mpcc::lower` from FlatOp, `mpcc::schedule` round
minimization, `mpcc::protocols` registry with capability + cost parameters,
`mpcc::cost` analytic model, `mpcc::select` recommender, `baru mpcc` CLI.
No new protocol implementations — the point is the framework, validated against
the two backends we already have.

**P1 — first new protocols.** Dealer, Semi2k, Shamir behind a single
`MpcProtocol` VM interface; the existing ABY3 transport backend re-expressed
through it. Calibrate the cost model against all four; report predicted vs
measured. Deliverable: a table with prediction error per protocol.

**P2 — malicious security & mixed circuits.** SPDZ2k MACs, replicated field,
ATLAS. daBit/edaBit generation, so `Convert` becomes real and arithmetic↔binary
mixed circuits work. This is what unlocks honest comparisons of comparison-heavy
workloads (ReLU, argmax, decision trees).

**P3 — preprocessing back-ends.** MASCOT (OT), LowGear (BFV). Split offline/online
measurement in the harness (MP-SPDZ's `-v`). This is where Baru's existing FHE
stack pays for itself.

**P4 — binary protocols and the full matrix.** Tinier, SemiBin, BMR; Yao folded
into the same VM interface.

**P5 — the recommender as a result.** Given an arbitrary Baru program and a
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
