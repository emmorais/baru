# Reproducing Falcon inside Baru

**Status: stages 1–5 done (the whole inference path, as a local simulation).
Remaining: truncation, then the transport wiring and the three-way benchmark.
Started 2026-08-01.**

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
3. **`Π_WA` wrap3.** ✅ **Done 2026-08-01** (`crypto::falcon::wrap3_protocol`).
   Exhaustive over all 4096 summand triples at ℓ = 4.

   **The stage-2 precondition is now discharged, and the answer is sharper than
   "negligible".** `Π_PC` misbehaves at `x = r+1`; in `Π_WA` we have
   `r ≡ a + x (mod L)`, so `x ≡ r+1` reduces to `a ≡ −1 (mod L)` — **the mask
   cancels entirely**. The degenerate set is exactly the single input
   `a = L − 1`, a property of the *data*, not of the randomness. A caller cannot
   dismiss it with a union bound.

   Confirmed empirically before it was believed: running `Π_WA` on every triple
   failed on 135 of 4096, and every failure had `a = 15 = L−1`. Roughly half the
   256 such triples pass by luck, because the degenerate `Π_PC` output is
   `1 ⊕ β` for a random `β` — which is itself the signature of the input being
   unreliable rather than merely wrong.

   **Falcon discharges it structurally, and DReLU depends on that.** DReLU calls
   `Π_WA` on the *doubled* summands `(2a₁, 2a₂, 2a₃)`, so the wrapped value is
   even while `L − 1 = 2^ℓ − 1` is odd; the bad input is unreachable. Stage 4
   must therefore keep the doubling — it is a correctness requirement, not an
   optimisation. Pinned by `the_degenerate_set_is_exactly_a_equals_l_minus_1`
   and `the_degenerate_input_is_unreachable_after_doubling`.
4. **`Π_DReLU` / `Π_ReLU`.** ✅ **Done 2026-08-01** (`crypto::falcon::drelu_protocol`,
   `relu_protocol`). Equation (11) is

   ```text
   DReLU(a) = MSB(a₁) ⊕ MSB(a₂) ⊕ MSB(a₃) ⊕ wrap3(2a₁,2a₂,2a₃,L) ⊕ 1
   ```

   — the MSB of a sum differs from the XOR of the summands' MSBs by the carry
   into the top position, and doubling is what moves that carry to where `Π_WA`
   can see it. Verified in the clear exhaustively at ℓ = 5 (32 768 triples)
   *before* any protocol was wrapped around it, then end-to-end at ℓ = 4.

   **`Π_DReLU` needs no excluded inputs**, unlike `Π_WA` — which is stage 3's
   analysis paying off. A negative control
   (`without_the_doubling_the_degenerate_input_becomes_reachable`) feeds the
   summands in undoubled and confirms `a = L−1` goes unreliable, so the test
   suite now *demonstrates* that the doubling is load-bearing rather than
   asserting it.

   Still to do before this can be compared against the repo's other routes:
   `relu_protocol` selects between `a` and `0` directly, where the paper uses
   `Π_SS`; and Falcon's ReLU has no rescale, so composing it with Baru's
   `relu∘rescale` convention needs the truncation protocol.
5. **`Π_Maxpool`.** ✅ **Done 2026-08-01** (`crypto::falcon::maxpool_protocol`),
   together with `Π_SS`. Folds left over the window: `n−1` DReLUs and `2(n−1)`
   selects, returning both the maximum and the one-hot `e_k`.

   **Two things that decide whether the eventual comparison is fair, both now
   encoded in the API.**

   *The one-hot is not free, and inference does not need it.* Falcon carries
   `ind` because it also **trains** — the indicator is what routes gradients
   back through a max-pool — and half the selects exist only for that. Timing
   Falcon's max-pool against Baru's `maxpool2d`, which returns a value only,
   would therefore charge Falcon for work our pipeline never does.
   `maxpool_value_only` is the like-for-like variant: `n−1` DReLUs, `n−1`
   selects. A test asserts the two agree on the value.

   *Inputs must be bounded.* `Π_DReLU` is applied to `max − aᵢ`, so `|aᵢ| < L/4`
   is needed to keep the difference inside the signed window. This is the same
   class of constraint as the two bits of headroom Baru's `nbits` calibration
   carries — a ring comparison always costs range somewhere — and it is worth
   noting that both systems pay it.

6. **`Π_Truncate`.** ✅ **Done 2026-08-01** (`crypto::falcon::truncate_exact`),
   with the probabilistic variant kept alongside as the baseline it replaces.

   Falcon delegates pair generation to ABY3's `Π_trunc2`: a random `r` with its
   exact shift `r' = r/2^d`, both precomputed, so that cost is offline. Online,
   open `c = a + r` and derive

   ```text
   a>>d = (c>>d) − (r>>d) − borrow,   borrow = 1 ⟺ c_lo < r_lo
   ```

   **`r_lo` is secret, and that is the whole story of "exact yet expensive".**
   The borrow is a secure comparison, one per truncation, and *every fixed-point
   multiply needs a truncation*. Falcon pays that and buys `ℓ = 32` instead of
   64, which is where its ≥2× communication advantage over prior work comes
   from. Measured here: dropping the borrow is wrong on **73%** of inputs, always
   by exactly one unit in the last place — so the guard bits an approximate
   scheme needs are real, not hypothetical.

   Getting there took a corrected derivation: masking with `c = a − r` wraps
   whenever `a < r` and produces a `L/2^d` error. The pair protocol requires
   `c = a + r` with `a + r < L`. Pinned by `exact_truncation_is_exact` and
   `probabilistic_truncation_is_off_by_one_about_half_the_time`.

### Correction, 2026-08-01: the private-compare modulus

An earlier revision of this document and of `crypto::field` stated that Falcon
uses `p = 67`. **It does not** — §3.1 says "we use `ℓ = 2^5`, `p = 37`", with 13
bits of fixed-point precision. `p = 67` is *SecureNN's* choice, sized for
`ℓ = 64`.

The binding constraint is `p > ℓ + 2`, so the modulus is a function of the ring
width rather than a constant. `crypto::field::P` stays at 67 because it serves
every `ℓ ≤ 64` and is small enough to test exhaustively, with `P_FALCON = 37`
recorded alongside it. **The benchmark must use Falcon's `ℓ = 32`**: their
communication advantage is largely the smaller ring, so running them at `ℓ = 64`
would understate them.
7. **`Z_p` on the real transport.** ✅ **Done 2026-08-01**
   (`crypto::fp_protocol`) — the mirror of `aby3_protocol`, so Falcon's field
   traffic is counted by the same transport as everything else. Verified over
   real 3-party messages on all 4489 products.

   Only two things differ from the ring protocol, and both matter:

   - **Zero-sum randomness.** `PrfKeys::alpha` sums to zero mod `2^128`, which
     says nothing mod `p`. The *construction* carries over — `F(kᵢ) − F(kᵢ₋₁)`
     telescopes in any group — so `alpha_mod` does the subtraction in the field.
   - **Wire width.** A field element is **one byte**, not sixteen. Falcon's
     headline claim is communication volume; carrying its field traffic in
     `u128` slots would overstate it 16× and make the comparison worthless.
     Hence `MpcMsg::TaggedFpBatch { values: Vec<u8> }`.

   *Bug caught here, worth recording because it is a trap:* the free random
   sharing must use the **raw** PRF outputs, not `alpha_mod`. The zero-sum form
   is one word away at the call site and yields a perfectly consistent
   replicated sharing **of zero** — which every structural assertion passes.
   `random_shares_are_consistent_and_uniform` checks reconstruction is non-zero,
   not just consistent.

8. **`Π_PC` on the transport, and the first comparable number.** ✅ **Done
   2026-08-01** (`crypto::falcon_mpc`). Verified against the local simulation on
   every `x ≠ r` at ℓ = 5, then measured at Falcon's own ℓ = 32:

   | per party, marginal | |
   |---|---|
   | rounds | **8** |
   | messages | **8** |
   | bytes | **225** |
   | field elements | 65 |
   | bytes/element | 3.46 |

   **The batching is the whole story, and it must be done by hand here.** Written
   literally from Algorithm 1, `Π_PC` is `2ℓ = 64` sequential rounds. But the `ℓ`
   products `u[i]` are mutually independent (one batch), and `Π c[i]` is a
   reduction (a tree of depth `⌈log₂ ℓ⌉`), giving `1 + 5 + 1 + 1 = 8`. An **8×**
   difference in round count.

   That transformation is precisely what `mpcc`'s ASAP scheduler and
   `ReassociateReduction` do automatically — which is why the hand-written side
   has to have it too. Benchmarking a compiled circuit against a 64-round
   hand-written baseline would be a straw man, and the resulting "compiler wins
   8×" claim would be worthless.

   *Measurement note:* the figure quoted is the **marginal** cost of one more
   comparison (two calls minus one). A single run also carries the one-time PRF
   key handshake — 9 messages, 261 bytes for the first call — and quoting that
   as the per-comparison cost would overstate the protocol.

9. **Batched `Π_PC`, and the first check against Falcon's published cost.**
   ✅ **Done 2026-08-01** (`private_compare_batch`).

   A deployment does not compare one element at a time: a conv layer has
   thousands of mutually independent comparisons, so they run in lockstep and
   the round count stays at 8 **for the entire layer**. Measured per party at
   ℓ = 32:

   | | unbatched | batched (n = 1000) |
   |---|---|---|
   | rounds | 8 | 8 |
   | marginal bytes / comparison | 225 | **65.0** |

   65.0 is exactly the 65 field elements at one byte each — framing amortises to
   nothing, and the measurement matches the analytic formula to the digit, which
   is a strong internal consistency check on both.

   **Extrapolated to Falcon's own LeNet** (4 684 ReLUs + 3 comparisons per 2×2
   window over 1 120 windows = 8 044 DReLUs):

   | | |
   |---|---|
   | this reproduction, per party | **0.52 MB** |
   | this reproduction, all parties | **1.57 MB** |
   | Falcon, reported | **0.74 MB** (LeNet, semi-honest LAN) |

   **Read this as encouraging but not yet validating.** Same order of magnitude,
   within ~2×, which is what you would hope for from an independent
   reimplementation — a reproduction that was structurally wrong would be out by
   10× or more. But the residual gap is unexplained, and there are at least four
   candidates, which must be settled before any of these numbers is quoted:

   - **`p = 37` vs `p = 67`.** Falcon's field is 6 bits; ours is 7, and we send a
     whole byte. Packing at Falcon's width is worth up to 1.33×.
   - **Online/offline split.** Falcon's preprocessing figure generates the `Π_PC`
     masks *and* the wrap3 mask bits offline (bit injection into `Z_p`). Some of
     what is counted here as online may be offline there.
   - **Is 0.74 MB per party or across all three?** If per party, this
     reproduction is *below* Falcon rather than above, and the comparison flips.
   - **DReLU count.** Our 3-comparisons-per-window assumption for max-pool may
     not match theirs.

10. **`bench-falcon`**: the same LeNet-shaped network three ways — Falcon
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
