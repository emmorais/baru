# Gadget costs — the curated table

Research agenda §5 argued for this file instead of a retrieval pipeline: at this
scale a small table you can diff beats a RAG you cannot audit. It exists to be
checked *before* claiming a rewrite or lowering is a win.

Every row carries its **provenance**, because the three kinds are not
interchangeable and conflating them is how a cost model starts lying:

- **standard** — a textbook/folklore result for the construction named.
- **Baru's model** — what `src/mpcc/cost.rs` actually charges. If this disagrees
  with *standard*, the disagreement is a modelling choice and is called out.
- **measured** — produced by a benchmark in this repo, with the command to
  reproduce it.

`k` is the bit width. Garbled-circuit costs count **AND gates**; XOR is free
(free-XOR), and under half-gates an AND gate is 2 ciphertexts.

---

## Binary / garbled circuits

| gadget | AND gates | AND depth | provenance |
|---|---|---|---|
| XOR, NOT, negate | 0 | 0 | standard (free-XOR; NOT flips the zero-label) |
| shift / truncate by a constant | 0 | 0 | standard (wire relabelling) — and the asymmetry that decides MPC-vs-GC for fixed-point work |
| ripple-carry adder | `k−1` | `k−1` | standard |
| parallel-prefix adder (Kogge–Stone) | `O(k log k)` | `⌈log₂ k⌉` | standard |
| adder | `k` | `⌈log₂ k⌉` | **Baru's model** — gate count of a ripple adder with the depth of a prefix adder. Optimistic on depth; see note below |
| `k`-bit mux | `k` | 1 | standard |
| comparator (`<`, `=`) | `k` | `⌈log₂ k⌉` | Baru's model; standard ripple comparator is `k−1` gates at depth `k−1` |
| schoolbook multiplier | `k²` | `2⌈log₂ k⌉` | standard for the gate count; depth is Baru's model |
| multiply by a public constant | `k²/2` | `2⌈log₂ k⌉` | Baru's model (half the partial products on average) |
| divider | `k²` | `k⌈log₂ k⌉` | Baru's model — restoring long division |
| fused ReLU∘rescale | `(k−1) + (k−shift)` | — | **measured** construction, `crypto::garbled` (adder ANDs + mux ANDs); `baru bench-gc-compare` |

**The adder-depth caveat.** Baru charges a ripple adder's gate count at a prefix
adder's depth. That is the favourable half of each, so *binary* protocol
estimates are optimistic on latency. It is a deliberate simplification —
`binary_op_cost` in `cost.rs` — and it matters for `semi-bin`/`tinier`, the two
binary protocols that are not constant-round. `bmr` and `yao` are constant-round,
so depth does not reach their cost at all.

## Arithmetic secret sharing

| gadget | rounds | communication | provenance |
|---|---|---|---|
| add, subtract, multiply by a public constant | 0 | 0 | standard — local |
| secret × secret | 1 | 2 elements/party (Beaver), 1 (Shamir/GRR) | standard; Baru's model matches, `mpcc/protocols.rs` |
| open a batch | 1 | 1 element/party | standard |
| bit-decompose `k` bits | 1 open + `(⌈log₂(k+1)⌉ + 1)` multiply rounds | masked opening + Kogge–Stone prefix borrow | **measured** — `vm::bit_decompose_batch`, 0% model error per `baru mpcc calibrate`; `cost::bit_decompose_muls` mirrors the VM instruction for instruction and must stay in lockstep |
| `<`, `=`, truncate | as bit-decompose, batched (`=` costs `2⌈log₂(k+1)⌉+1` — an extra AND tree) | — | same; ten comparisons at one dependence level cost one comparison's rounds |
| Beaver triple, OT-based (Gilboa) | offline | `2k` correlated OTs ⇒ ~3.1 KB at `k=64` | **measured** OT cost × protocol count, `mpcc::ot_cost` |
| Beaver triple, BFV-based (LowGear) | offline | ~384 B | Baru's model — literature estimate, and its CPU cost is *not* modelled |

## Primitives, measured in this repo

| thing | cost | reproduce |
|---|---|---|
| base OT (Chou–Orlandi) | ~124 µs, 64 B per OT | `baru bench-ot` |
| IKNP-extended OT | 4.67 µs, 16.5 B per OT at batch 8192 | `baru bench-ot` |
| OT-extension online floor | `κ/8 = 16` B per OT | analytic; the measured curve converges to it |
| DH-PSI | 96 B, 165 µs per item | `baru bench-psi` |
| naive hash-exchange PSI (**insecure**) | 32 B, 0.43 µs per item | `baru bench-psi` |
| BFV PIR response | 13× the shard's whole plaintext content | `baru bench-pir` |
| TFHE gate | ~0.2–0.5 s | `baru bench` |
| CKKS vs TFHE, same circuit | 1.6 s vs 2829 s | `cargo run --release --example mnist -- --ckks` |
| *(measured on the pre-CSE 2365-gate circuit; CSE now emits 1764 gates, so both sides should be ~25% lower — **not re-measured**)* | | |

## How to use this file

1. Before claiming a lowering is cheaper, find the gadget here and check you are
   comparing against the *best known* construction, not the naive one.
2. If your proposal beats a **standard** row, that is a claim worth checking
   carefully — the differential oracle proves correctness, not optimality.
3. If your proposal beats a **Baru's model** row, you may have found a modelling
   error rather than an optimization. Both are worth fixing; they are different
   fixes.
4. When you add a row, add its provenance. A row without provenance is worse than
   no row, because the next reader cannot tell whether to trust it.
