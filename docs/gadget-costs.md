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
| Beaver triple, SHE-based (LowGear) | offline | **192 B** at its native ~61-bit field (no CRT limbs needed) | **measured** — `baru bench-triples`; the literature estimate was 384 B, so 2× better |
| *(and it is **~28× faster** per triple than the OT route (73 400 vs 2 620 triples/s at a field wide enough for both), not slower. `mpcc::cost` used to carry "LowGear trades bandwidth for CPU: its preprocessing is compute-bound" — that note was never measured and is **wrong**: SHE wins on both axes, by 16.3× on bytes and ~28× on CPU.)* | | |
| Semi2k and LowGear online phase (Beaver) | rounds and online bytes both **0% model error** | `baru mpcc calibrate --expr '(+ (* a b) (* c d))' --parties 2` — both are now executable protocols (`impl` in `baru mpcc protocols`), not analysis-only rows |
| *(LowGear refuses comparison-shaped programs: its ~17-bit field cannot supply `KAPPA = 26` bits of masking headroom for bit-decomposition. `mpcc::engine_can_run_program` makes that a skip-with-reason, not a wrong answer.)* | | |

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
| exact negacyclic product, n=4096, ±2⁹⁹ coeffs | schoolbook 71.9 ms / our multi-modular NTT **1.77 ms** / NTL+GMP 3.15 ms | `baru bench-polymul --schoolbook-max 4096` (NTL column needs `--features ntl`) |
| *(the NTT is 40.6× schoolbook at n=4096 and **1.8× faster than NTL** at every size. It was 8.23 ms and lost to NTL by 2.6× until the butterfly moved to precomputed twiddles + Shoup-preconditioned multiplication — 4.65×. NTL is now a differential oracle, not a performance option.)* | | |
| depth-1 ct×ct packed matvec, secure params | BFV 5.4 ms / BGV 5.3 ms (n=4096) — **indistinguishable**; SI-NTRU 1.0 ms (n=2048, INSECURE) | `baru bench-fhe-innerprod --poly all --reps 15` |
| same, on the NTL polynomial backend | BFV 11.2 ms / BGV 11.1 ms / SI-NTRU 1.7 ms — **slower**, since the Shoup rewrite | as above, `--features ntl` |
| private layer + ReLU, WAN, 2 parties — **which preprocessing MPC is priced with decides the crossover** | hybrid beats OT-preprocessed MPC from **width 64**; **never** beats SHE-preprocessed MPC at any width measured (64→4096). On LAN, neither. | `baru bench-layer --widths 64,256,1024,4096 --parties 2 --wan` |
| *(all rows **measured** — LowGear runs the layer itself, ReLU included, since its field widened to ~2^61. At width 64 it is 606 ms against the trusted-dealer lower bound's 601 and the hybrid's 692. Qualifies `paper-outline` E25, which states the boundary against OT only.)* | | |
| *(the ~5× RLWE-vs-SI-NTRU gap is structural: 2 ring elements per ct at n=4096 vs 1 at n=2048. Changing the RLWE encoding does not move it; faster ring arithmetic does.)* | | |
| one private layer 256→8, secure hybrid (BFV+ABY3) | **71.5 ms** on the NTT backend (100.1 ms on NTL); pure ABY3 does the same layer in **34.1 ms** (medians of 8, separate processes, alternating order) | `baru bench-hybrid-nn --strategies mpc,hybrid-ntru,hybrid-bfv --in 256 --out 8 --poly all` |
| *(the hybrid is **2.09× pure MPC**, and FHE is **64%** of it — down from 87% before the Shoup butterfly. FHE's case here is the comm profile — **15 messages vs 9276** — not speed.)* | | |
| batched vs per-element ReLU, ABY3 over the real transport | out=8: 95.7 → 14.3 ms; out=32: 451.8 → 19.0 ms; out=64: 802.4 → **23.3 ms (34.4×)** — bit-identical output | `baru bench-hybrid-nn --strategies mpc --in 256 --out 64` |
| *(the batched form is ~flat in width — independent comparisons are constant-round — while the per-element form is linear. `gadgets::relu` had the batched form all along; `bench-hybrid-nn` had not adopted it.)* | | |
| batched vs per-element fused ReLU∘rescale + import (`a2b_batch`, `mul_batch`, `share_secret_batch`) | depth 4: width 16 MPC **1273 → 102 ms**, width 32 **2372 → 100 ms (23.7×)**; messages 2820 → **76**. Bit-exact, 0 bootstraps unchanged | `baru bench-deep-nn --depths 4 --width 32` |
| FHE→MPC import, batched per owner | `bench-hybrid-nn` MPC segment out=64 **1186 → 30 ms**; messages 1153 → **15** | `baru bench-hybrid-nn --strategies hybrid-bfv --in 256 --out 64` |
| *(the number that matters is that **messages are now constant in width** — 76 at both deep-nn widths, 15 at every hybrid-nn width — not the speedup. `share_secret_batch` already existed on `TransportBackend` and is used by `cnn.rs`, `compiled_nn.rs`, `abstraction.rs`, `mixed_layer.rs`; these two benchmarks never adopted it. Fourth instance of one defect.)* | | |
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
