# Reproducing Falcon inside Baru

**Status: planned, stage 1 in progress. Started 2026-08-01.**

## Why do this at all

The [comparison in experiments.md](experiments.md) puts Baru's private diagnosis
at 268 ms against Falcon's 47 ms for a comparable network (LeNet, ~60 k
parameters vs our 53 k), in the *same* stated threat model — semi-honest,
honest-majority, 3 party. That is 5.7× slower with 14× the communication, and no
amount of optimising the FHE hybrid closes it: **our FHE segment alone is 99 ms**,
more than Falcon's entire inference. Two sessions of measurement have made that
concrete, and two candidate optimisations died on it:

- **per-layer `nbits`** — every layer is within 2 bits of the global width, so
  ~10% of `a2b` work. Not built.
- **fused garbled max-of-4** — measured cost is proportional to AND-gate count
  (2.28 ms per gate-column at N=5256; the Y2A conversion is free and the Beaver
  select is 11%), so fusing would take a 2×2 window from 88 to 148 gates. A
  1.68× *loss*. Not built. Pinned by `profile_lt_versus_mul_under_gc`.

So the honest position is that the hybrid loses in this setting — which is what
Baru's own cost model already predicts (`paper-outline.md` E25: under an honest
majority the layer stays on MPC at *every* width).

**The question worth answering is therefore not "can the hybrid beat Falcon".**
It is:

> Reproduce Falcon *inside Baru*, using the same components and the same
> security model, and then ask whether the **compiled** circuit beats the
> **hand-constructed** protocol.

That is a fair fight, it plays to the contribution the repo actually has
evidence for — `bench-compiled-nn` already shows the compiler beating an expert
hand-written baseline at **0.60× interactive calls and 0.70× messages** — and it
turns a losing latency comparison into a claim about compilation. It also gives
the paper a baseline it currently lacks, which is the thing a reviewer will
demand first.

## What Falcon actually is

3-party replicated secret sharing (RSS) over `Z_L`, `L = 2^ℓ` (ℓ = 32 or 64),
honest majority, semi-honest **and** malicious. All parties run the *same*
protocol — the symmetry is itself one of Falcon's stated advantages over
SecureNN's asymmetric roles.

| | protocol | computes | over |
|---|---|---|---|
| Alg 1 | `Π_PC` private compare | shares of the bit `(x > r)`, `r` public | **`Z_p`, p = 67** |
| Alg 2 | `Π_WA` wrap3 | shares of `θ = wrap3(a₁,a₂,a₃,L)` | `Z_L` + `Z_p` |
| Alg 3 | `Π_ReLU` / DReLU | `ReLU(a)`, via `Π_WA` on `(2a₁,2a₂,2a₃,L)` | `Z_L` |
| Alg 4 | `Π_Maxpool` | `a_k` **and** the one-hot `e_k` | `Z_L` |
| Alg 5 | `Π_Pow` | bounding power of two | `Z_L` |
| Alg 6 | `Π_Div` | `a/b` by Newton iteration | `Z_L` |
| Alg 7 | `Π_BN` | batch norm | `Z_L` |

`wrap2` is computed **locally** and needs no protocol. The exact-integer
identities the construction rests on:

```
r   = a + x − θ·L            (5)
r   = r₁ + r₂ + r₃ − α·L     (6)
rᵢ  = aᵢ + xᵢ − βᵢ·L         (7)
x   = x₁ + x₂ + x₃ − δ·L     (8)
```

DReLU is then the MSB, obtained from `wrap3(2a₁,2a₂,2a₃,L)` — doubling moves the
sign question into the wrap. This is the "deeper mathematical insight" Falcon
claims over SecureNN, worth >2×.

For inference we need Alg 1–4 only. Alg 5–7 are training/regression machinery.

## Gap analysis against Baru

Baru has the substrate: `crypto::aby3_protocol` is exactly RSS over `Z_{2^k}` —
`share_input`, `mul_party`, `a2b_party`, `reveal`, plus batched forms — and
`bench::malicious` already carries an honest-majority malicious variant.

**The blocker is algebraic.** `crypto::types::Fr = u128` and every operation is
`wrapping_add` / `wrapping_mul`, i.e. the ring is hardwired to `Z_{2^128}` masked
to `2^nbits`. Falcon's `Π_PC` needs a **prime field**: it forms `c[i]` terms and
tests whether any is zero, and that test is only meaningful without zero
divisors. `Z_{2^k}` has them. **There is no field arithmetic anywhere in the
crate.**

| Falcon needs | Baru has | gap |
|---|---|---|
| RSS over `Z_L` | `aby3_protocol` | — |
| local `wrap2` | trivial | — |
| shared random bit in two rings | PRF/`alpha` machinery exists | plumbing |
| **arithmetic mod a small prime `p`** | **nothing** | **stage 1** |
| `Π_PC` | — | stage 2 |
| `Π_WA` wrap3 | — | stage 3 |
| `Π_DReLU` / `Π_ReLU` | garbled + `a2b` routes only | stage 4 |
| `Π_Maxpool` returning one-hot `e_k` | value-only `maxpool2d` | stage 5 |

## On Galois rings (eprint [2024/705](https://eprint.iacr.org/2024/705))

That paper — *Large-Scale MPC: Scaling Private Iris Code Uniqueness Checks to
Millions of Users*, Bloemen–Gillespie–Kales–Sippl–Walch — is 3-party, reports
690 k comparisons/s on one core and 4.29 G/s across 8×H100, and its revision
notes list **Galois Shamir sharing**. Directly relevant to this repo, which
already references an iris `hom-ham` pipeline.

**Be careful about what Galois rings buy, because it is easy to overstate.**
`GR(2^k, d)` is a degree-`d` extension of `Z_{2^k}`. For `k > 1` it **still has
zero divisors** (2 is one), so it is *not* a drop-in replacement for `Z_p` in
`Π_PC`. What it provides is a large *exceptional set* — pairwise differences
invertible — which is what Shamir sharing needs over a characteristic-2^k ring.
Its real uses are therefore:

1. **Shamir/packed sharing over characteristic 2^k**, hence more than 3 parties
   and better packing — which is how the iris paper uses it, and where its
   throughput numbers come from.
2. The residue field `GR(2,d) = F_{2^d}` *is* a field, and `F_{2^7}` (128
   elements) exceeds Falcon's `p = 67`. Characteristic 2 is friendlier to
   carry-less multiply than `Z_67`.

**Open question, to be settled by derivation before any code:** does `Π_PC`'s
algebra survive in characteristic 2? Falcon's construction adds small integers
and relies on them *not* wrapping in `Z_p`; over `F_{2^d}` doubling annihilates.
If it does not survive, `F_{2^d}` cannot host `Π_PC` and the honest move is a
plain `Z_p`, with Galois rings evaluated separately for the multi-party/packing
axis. **Do not assume the substitution works.**

## Stages

1. **Prime-field arithmetic + RSS over `Z_p`.** Self-contained, testable with no
   protocol changes. Montgomery or plain `u64` reduction for `p = 67`; the field
   is tiny, so correctness and constant-time behaviour matter more than speed.
2. **`Π_PC`.** ✅ **Done 2026-08-01** (`crypto::field::private_compare`), with the
   `Z_p` multiplication protocol it needs. Exhaustive over every `x`, `r` at
   ℓ = 5 and both blinding bits.

   **Finding, and it constrains stage 3.** The functionality is `x ≥ r`, and the
   blinding bit `β` works by selecting between complementary predicates:
   `β' = (x ≥ r)` at `β = 0`, `β' = (r > x)` at `β = 1`, so `β' ⊕ β = (x ≥ r)`
   either way. **That equivalence breaks at `x = r`.** There every `w[k]` and
   every `u[i]` is zero, so every `c[i]` is 1, the product is non-zero and
   `β' = 1` *whatever `β` is* — the output collapses to `1 ⊕ β`, a function of
   the blinding bit rather than of the inputs.

   So `Π_PC` is correct only for `x ≠ r`. That is fine in `Π_WA`, where `x` is a
   fresh random mask over `Z_L` and equality has probability `2^-ℓ`, negligible
   at ℓ = 32 or 64 — **but stage 3 must actually establish that**, rather than
   inherit it by assumption. Both behaviours are pinned by tests
   (`private_compare_computes_ge_for_distinct_inputs`,
   `private_compare_degenerates_when_x_equals_r`) so neither can drift.

   Found by an exhaustive test whose *expectation* was wrong — it asserted
   `x > r` and failed on exactly the 32 equality cases. Worth noting as
   methodology: at `P = 67` exhaustive testing is affordable, and it located a
   protocol edge case that spot-checks would have missed.
3. **`Π_WA` wrap3.** Test against the exact integer identities (5)–(8) directly:
   they are checkable in the clear.
4. **`Π_DReLU` / `Π_ReLU`.** Must agree bit-for-bit with `CnnModel::forward`'s
   `relu∘rescale`, the same standard every other route in this repo is held to.
5. **`Π_Maxpool`** with the one-hot `e_k` output.
6. **`bench-falcon`**: the same LeNet-shaped network three ways — Falcon
   hand-written, Baru gadgets, Baru compiled — over the same transport, with
   message and byte counts, exactly as `bench-compiled-nn` does. All three
   checked against one plaintext reference; a cost comparison between routes
   computing different things is meaningless.

## What must be true for the result to mean anything

- **Same security model on both sides.** Falcon's numbers are semi-honest LAN
  for the 47 ms figure; the malicious variant is 120 ms. Compare like with like
  and say which.
- **Same network.** Falcon's LeNet is ~282 k MACs against our current 240 k.
  Reproduce *their* architecture, not ours.
- **Same accounting.** Our `NetworkProfile` charges rounds and bytes; their LAN
  is a real network. `bench-net` validated the model's form to 13.6% worst-case,
  so quote the model *and* say it is a model.
- **The comparison that matters is internal**: hand-written Falcon vs compiled
  Falcon, both inside Baru, over the same transport. That is immune to
  machine differences and is the actual claim.
