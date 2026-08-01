# Reproducing Falcon inside Baru

**Status: stages 1–13 done. Falcon's inference nonlinearity is reproduced,
running on the real transport, and benchmarked four ways against Baru's own
routes. Remaining: whole-ReLU (needs transport truncation) and an end-to-end run
rather than an extrapolation. Started 2026-08-01.**

**Headline, after two corrections:** the compiled route reproduces the
hand-written `a2b` route *to the digit*; `wrap3` and `a2b` are within ~6% on
information content; garbled circuits are the real outlier at ~30×. The
apparent 34–136× advantage for `wrap3` was this crate's `u128` serialisation of
1-bit shares, not protocol.

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

   *(Superseded by the validation below — this extrapolation counted only
   `Π_PC`, not the full DReLU, and so understated by ~2×.)*

## Validated against Falcon's published complexity (2026-08-01)

The paper states per-protocol complexities, so the reproduction can be checked
against something far sharper than a whole-network megabyte figure. Semi-honest,
`n` the input width, `k` the bits of a `Z_p` element:

| protocol | rounds | comm |
|---|---|---|
| Private Compare | `2 + log₂ ℓ` | `2kn` |
| wrap3 | `3 + log₂ ℓ` | `3kn` |
| ReLU / DReLU | `5 + log₂ ℓ` | `4kn` |

At ℓ = 32 that is **7 rounds and `2ℓ = 64` field elements** for `Π_PC`. This
implementation measures **8 rounds and 65 elements** — one more of each, and the
*same* one: the final opening. Falcon's table counts the interactive
multiplication rounds of a protocol whose output is shares; ours additionally
reveals `d`, one round carrying one element. Net of that, the two agree
**exactly**: `32 (u) + 31 (tree) + 1 (mask) = 64` elements over
`1 + log₂ ℓ + 1 = 7` rounds. Asserted for ℓ ∈ {8,16,32,64} by
`the_cost_matches_falcons_published_complexity`, so it cannot drift.

### This also settles the accounting question

Taking `ReLU/DReLU = 4kn` at Falcon's `k = 6` bits (`p = 37`) gives 96 B/party
per DReLU, and LeNet's 8 044 DReLUs give **0.77 MB per party** — against the
**0.74 MB** the paper reports. So:

- **0.74 MB is per party**, not summed over the three.
- **LeNet's communication is essentially all DReLU traffic**; the linear layers
  are negligible beside it.
- This reproduction, at 8 bits per element, costs **1.03 MB/party — 1.33×
  Falcon**, and that factor is *exactly* `8/6`, the element packing. There is no
  unexplained residue.

The earlier "within ~2×, four candidate explanations" reading is superseded: of
those four, the packing was the whole story, the per-party question is answered,
and the online/offline and DReLU-count concerns turn out not to bite.

### Fidelity fixed, and the reproduction now lands on Falcon's number

**Done 2026-08-01.** Two changes, both required rather than optional:

- **`P` is now 37, Falcon's own modulus** (was 67, SecureNN's, sized for
  `ℓ = 64`). `P_SECURENN` is kept named so the distinction does not become folk
  knowledge. The bound `p > ℓ+2` caps the ring at 34, above Falcon's 32.
- **Field elements are bit-packed at `ELEM_BITS = 6`**, not one per byte.

Measured after both, per party at ℓ = 32:

| | |
|---|---|
| marginal bytes / comparison | **48.8** (measured) |
| Falcon's `2kn` | **48.0** |
| difference | 1 element — our revealed `d` |

and the LeNet extrapolation, using Falcon's own `DReLU = 4kn`:

| | |
|---|---|
| this reproduction | **0.77 MB/party** |
| Falcon, reported | **0.74 MB/party** |

**4% apart, with the residue accounted for.** The reproduction is faithful.

Three traps were passed on the way here, each of which would have silently
flattered our side by understating the baseline: the `Π_PC` batching (8× in
rounds), the max-pool one-hot (2× in selects), and the element packing (1.33× in
bytes). Any benchmark built on top of this must keep all three.

### ⚠️ And a fourth, pointing the other way: Baru's own byte counts are encoding-limited

Making the Falcon path wire-exact exposed an asymmetry that would now bias the
comparison **against Baru**.

`aby3_protocol` reshares through `MpcMsg::TaggedZBatch { values: Vec<u128> }`,
and bincode encodes `u128` as a fixed 16 bytes regardless of `nbits`. Measured
directly (`ring_traffic_is_dominated_by_the_u128_encoding`, n = 4096,
nbits = 20):

| | |
|---|---|
| bytes per ring element | **16.01** |
| meaningful content | 2.50 (20 bits) |
| **encoding overhead** | **6.4×** |

So every ring-share byte this repo reports is ~6.4× its information content,
while the `Z_p` path is now packed to the bit. **Benchmarking the two against
each other as they stand would handicap Baru by that factor** — precisely the
mirror image of the three traps above, and worth as much care.

#### The split, measured

`CommStats` now separates traffic by bincode variant tag, so every benchmark
reports it. For one pneumonia diagnosis:

| config | total | ring (ABY3) | garbled | other |
|---|---|---|---|---|
| `--gc --precompute` | 10.1 MB | **0.5 MB (5%)** | 7.3 MB | 2.3 MB |
| `a2b` | 29.9 MB | **29.5 MB (99%)** | 0.0 MB | 0.5 MB |

So the worry was right in principle and **mostly wrong about the headline
config**:

- **`--gc --precompute` — the configuration every headline number uses — is
  garbled-dominated, and garbled material is already packed bytes.** Ring shares
  are 5% of it, so the `u128` padding costs ~0.4 MB of 10.1. The figure stands,
  and a communication comparison against Falcon on this path is **not**
  materially biased.
- **The `a2b` figure is inflated, and substantially.** At 99% ring and 84%
  padding, its true information content is ≈ 29.5 × 20/128 + 0.5 ≈ **5.1 MB, not
  29.9 MB** — a 5.9× overstatement of a number the docs quote.

Two consequences:

1. **`a2b`'s 29.9 MB should be reported as encoding-limited** wherever it
   appears, or the encoding fixed. It makes Baru's least favourable
   configuration look ~6× worse than it is.
2. **Stage 10 does not need a width-aware ring encoding after all**, because it
   will compare on the `--gc` path where ring traffic is 5%. Worth fixing on its
   own merits; not a blocker.

The measurement was worth making precisely because the conclusion flipped: the
effect is real, and it lands almost entirely on a configuration the headline
does not use.

10. **Batched ring reveal.** ✅ **Done 2026-08-01**
    (`aby3_protocol::reveal_batch_party`). `Π_WA` opens one masked value per
    DReLU, and the only reveal the semi-honest path had was per-share — `N−1`
    messages *per value*, so a convolution layer would have spent tens of
    thousands of round trips on what is structurally one exchange, and the wrap
    protocol's measured cost would have been an artefact of the API rather than
    of the protocol. Now one directed message per party per batch, asserted.

    (`aby3_verify::reveal_batch_checked` already existed but broadcasts and
    cross-checks, which is the malicious variant's job.)

11. **`Π_WA` and `Π_DReLU` on the transport.** ✅ **Done 2026-08-01**
    (`falcon_mpc::wrap3_batch`, `drelu_batch`). Correctness carried over from the
    local simulation — every value in the ring at ℓ = 8, over real messages —
    and now measured rather than derived:

    | per party, ℓ = 32, batched | |
    |---|---|
    | messages | **10** (1 ring reveal + 8 `Π_PC` + 1) |
    | marginal bytes / DReLU | **52.8** |
    | Falcon `wrap3 = 3kn` | 72.0 → 0.73× |
    | Falcon `ReLU = 4kn` | 96.0 |

    (64.8 B before the ring reveal was packed at ℓ rather than padded to `u128`;
    `reveal_components_packed_party` fixes that, and it was our own number the
    padding was inflating.)

    **Compare against the `3kn` line, not `4kn`.** Falcon's table puts "ReLU
    **and** Derivative of ReLU" on one row at `4kn`, while `wrap3` is `3kn`. A
    DReLU is wrap3 plus a local XOR, so `3kn` is the like-for-like figure; the
    extra `kn` is `Π_SS`, the select that turns the bit into a ReLU, and **that
    is not built here yet**. The first version of this measurement compared
    against `4kn` and reported a flattering 0.67× — crediting us for work not
    done.

    ### ⚠️ How much weight the byte ratios can carry

    **The round comparison is solid; the byte comparison is not, and the reason
    is `k`.**

    Rounds match Falcon's `2 + log₂ ℓ` *exactly*, checked for ℓ ∈ {8,16,32,64}.
    That is unambiguous and is the strongest evidence the construction is theirs.

    The byte rows are a different matter. Reading `k` as the `Z_p` element width
    (6 bits) makes `Π_PC = 2kn = 48 B` agree with our measured 48.8 to within one
    element — convincing. But the same reading makes `wrap3 = 3kn` charge `kn`
    = 24 B more than `Π_PC`, where our wrap3 adds only the one opened ring value
    (4 B packed). Reading `k` as the ring width instead makes `Π_PC = 8 B`, which
    is impossible for a protocol performing 32 field multiplications. **Neither
    reading is consistent across both rows**, so `k` either changes meaning
    between them or absorbs costs the table does not itemise — plausibly the
    preprocessing (`Π_PC` masks and wrap3's bit-injected mask bits) that their
    Figure 6 generates offline.

    So: **do not quote 0.73× as though it meant we are 27% leaner than Falcon.**
    It means our wrap3 sends 52.8 B/party where their table's wrap3 row implies
    72, under an interpretation of `k` that does not survive contact with the
    other rows. The defensible statements are that the round structure is
    identical and that end-to-end LeNet lands at the right order of magnitude.
    Settling it needs their source (`snwagh/falcon-public`), not more arithmetic
    from the paper.

    Also needed on the way: `reveal_components_batch_party`. `Π_WA` step 3
    computes `α = wrap3(r₁,r₂,r₃)` in the clear, a function of the individual
    summands that cannot be recovered from `r = Σrⱼ`, so the reveal has to return
    components. Same single message — every party already holds all three after
    the exchange.

12. **`Π_SS` and `Π_ReLU` on the transport.** ✅ **Done 2026-08-01**
    (`falcon_mpc::select_batch`, `relu_batch`). **Falcon's whole inference
    nonlinearity now runs on real messages**, verified against `max(0, a)` for
    every value in the ring at ℓ = 8.

    `Π_SS`'s trick is worth naming: `b` lives in `Z_2` but the select needs it as
    a ring value to multiply by. Preprocessing supplies one random bit `c` shared
    in **both** rings; opening `b ⊕ c` reveals nothing, yet lets every party pick
    locally between `⟦c⟧_L` and `⟦1−c⟧_L` — which *is* `⟦b⟧_L`. That avoids a
    share-conversion protocol, and it costs one bit opening plus one ring
    multiplication.

13. **`bench-falcon`.** ✅ **First results 2026-08-01.** Sign extraction — the
    operation LeNet's cost consists of — four ways on one transport, all checked
    to agree bit for bit before anything is reported.

    ```
    n = 128, ring width 32
      route                    messages       bytes      ring B   garbled B      ms
      falcon (wrap3)                 10        6972           0           0    23.6
      baru-a2b                       10      919768      919732           0    50.1
      baru-gc                         5      202892        2068      192540    10.7
      baru-compiled (256 ops)        10      919768      919732           0    41.1
    ```

    ### The result the exercise was built to get

    **The compiled route reproduces the hand-written `a2b` route exactly** — same
    message count, same bytes, to the digit. From a scalar source
    (`(+ (> x0 −1) (+ (> x1 −1) …))`) the ASAP scheduler recovers the batching an
    expert would write by hand. That is the compiler claim, now demonstrated
    against a *published protocol's* workload rather than a self-chosen one.

    **And it cannot do more than that.** Falcon's route is far leaner because
    `wrap3` is a *different protocol* for the same predicate — bit-decomposition
    replaced by a carry recovered over `Z_p`. No amount of scheduling or CSE
    turns one into the other. So the honest claim is bounded, and the bound is
    the interesting part:

    > The compiler matches expert *composition* — batching, CSE, reassociation —
    > and cannot substitute for a better cryptographic *primitive*.

    That is a far more defensible thing to write than "the compiler wins", and it
    tells a reader exactly where compilation stops paying.

    ### ⚠️ How large the protocol advantage is — corrected twice

    The raw byte column says Falcon is ~136× leaner than `a2b`. **It is not, and
    two successive corrections were needed to see why.**

    *First correction:* Baru's ring shares serialise as `u128`, so some of that
    gap is padding rather than protocol. Discounting at the **ring** width
    (ℓ = 32, hence 4×) gave ~34×.

    *Second correction — the important one:* **that discount was wrong.** `a2b`
    does not send ring elements. It bit-decomposes and then operates on **1-bit**
    shares (`share_component((known >> k) & 1, …)`), each still serialised as a
    `u128`. Its padding is therefore **128×, not 4×**. Measured at n = 2048,
    `a2b` sends 7 184 B/sign as **449 elements carrying one bit each** — an
    information content of ~56 B.

    | route | wire B/sign | information content | why |
    |---|---|---|---|
    | falcon (wrap3) | 52.9 | **52.9 B** | `Z_p`, already packed at 6 bits |
    | baru-a2b | 7 184 | **~56 B** | 449 × 1-bit shares in `u128` slots |
    | baru-gc | 1 584 | **~1 584 B** | garbled tables, already `Vec<u8>` |
    | baru-compiled | 7 184 | **~56 B** | identical to `a2b` |

    **So on information content, `wrap3` and `a2b` are within ~6% of each other.**
    The apparent 34×–136× advantage was an artefact of this crate's
    serialisation. The genuine gap is against **garbled circuits**, ~30×, and
    that one is real — garbled tables are large by construction.

    That does not diminish Falcon's engineering (they *did* pack, and packing is
    most of why their published LeNet figure is 0.74 MB), but it is not a
    protocol-level advantage over bit-decomposition, and this document previously
    said it was.

    The compiled route matching `a2b` **to the digit at every n** is unaffected,
    and remains the strongest form of the compiler claim available here — not
    "close to", identical.

    ### Also visible

    Baru's garbled route is half the rounds and ~5× faster in wall-clock at zero
    latency (21 ms vs 88 ms at n = 2048) while sending ~30× more information —
    the classic garbling trade. Message counts are flat in `n` for every route,
    so all four are properly batched.

    ### Two harness bugs caught, both of the same kind

    - **The first run charged Baru 2n extra messages** — its route revealed each
      bit inside the measured region while Falcon's returned shares. Both now
      return shares.
    - **The byte classifier was silently wrong.** Adding `RevealPackedBatch`
      shifted the bincode discriminants, so hardcoded tags attributed a reveal to
      "garbled" and garbled traffic to nothing — with every *total* still correct
      and no test failing. Tags are now derived by serialising a sample variant.

    Both were found because a number looked wrong for a nameable reason, and
    both would have biased the comparison. The agreement check caught a third:
    `(> x 0)` is strict where DReLU is `≥ 0`, differing on exactly `a = 0`.

    ### Does a per-op difference survive to a network?

    Only if the operation dominates — so this had to be checked rather than
    assumed. **Falcon's MatMul and Baru's `dense` are the same RSS protocol**
    (one ring element per output, per party), so a layer differs *only* in its
    nonlinearity. Extrapolating the measured per-sign costs over LeNet's op
    counts (8 044 DReLUs, 4 694 linear outputs):

    Using **information content** per sign (not wire bytes — see the correction
    above) over LeNet's 8 044 DReLUs and 4 694 linear outputs:

    | route | nonlinear | total | nonlinear % |
    |---|---|---|---|
    | falcon (wrap3) | 0.425 MB | **0.444 MB** | 95.8% |
    | baru-a2b / compiled | 0.451 MB | 0.470 MB | 96.0% |
    | baru-gc | 12.7 MB | 12.7 MB | 99.9% |

    The nonlinearity is 96–100% of the traffic either way, so whatever the
    per-op gap is, it carries to the network undiluted. But the gap between
    `wrap3` and `a2b` is ~6%, not orders of magnitude — **the two share-based
    routes are near-equivalent, and garbling is the outlier.**

    Falcon reports **0.74 MB/party**; this reproduction extrapolates to
    **0.444 MB**, consistent given `Π_SS` is not counted here and their ReLU line
    (`4kn`) exceeds our DReLU.

14. **`Π_Maxpool` on the transport.** ✅ **Done 2026-08-01**
    (`falcon_mpc::maxpool_batch`). Folds left over each window with one
    `Π_DReLU` and one `Π_SS` per step, **every window's step `k` batched
    together** — so a pooling layer costs `arity−1` DReLU rounds regardless of
    how many windows it has. Value-only, matching what inference needs; the
    one-hot is training machinery and charging inference for it would overstate
    the baseline.

15. **Exact truncation on the transport.** ✅ **Done 2026-08-01**
    (`falcon_mpc::truncate_batch`). Open `c = a + r` from a truncation pair, then
    `a>>d = (c>>d) − (r>>d) − borrow`. Exact on every input, verified against a
    plain shift.

    **`Π_PC` cannot serve the borrow, and that is a finding.** It looks like it
    should: `borrow ⟺ r_lo ≥ c_lo + 1` is exactly `Π_PC`'s shape, secret against
    public. But `Π_PC` is undefined when its secret equals its public argument,
    and that case is `r_lo = c_lo + 1` — **precisely the boundary where the
    borrow flips**. The degenerate input is not rare and cannot be masked away:
    it *is* the answer. Caught at the first value where it bites (`a = 15`,
    `d = 4`, `r_lo = 10`, `c_lo + 1 = 10`).

    So the borrow is the sign of `c_lo − r_lo` by bit decomposition, which is
    exact at equality. Worth recording for the reproduction: **Falcon's own stack
    needs a comparison primitive `Π_PC` does not provide.** Their paper cites
    ABY3 for the truncation pair and does not spell this out.

16. **The layer, composed.** ✅ **Done 2026-08-01** (`falcon_mpc::dense_batch`,
    `layer_batch`). A whole fixed-point block — `ReLU(truncate(W·x, d))` with
    **secret weights** — runs on the transport and matches integer arithmetic
    exactly.

    `dense_batch` is Falcon's `Π_MatMul` and Baru's `gadgets::dense`: the *same*
    RSS protocol, one batched round of `out·in` secret×secret products followed
    by local sums. That equality is why the linear layers were never the
    difference between the two systems, and it is why the comparison has always
    reduced to the nonlinearity.

    Note where the truncation sits — **after the multiplication, not fused into
    the nonlinearity**. That is Falcon's placement, and it is why a like-for-like
    comparison against Baru's fused `relu∘rescale` has to be made at the layer
    rather than the operation.

    **Falcon's inference path is now reproduced in full**, every piece on the
    real transport and checked against a plaintext reference.

## Applying it to MedMNIST — and the inefficiency that surfaced

MedMNIST is the same shape Falcon targets (28×28, conv→pool→conv→pool→dense),
so the reproduced stack applies with no new cryptography. Working out the
comparison exposed a real defect in Baru's own dense layer.

**`gadgets::dense` was resharing per *product*, not per *output*.** In RSS a dot
product needs one resharing: each party accumulates its cross-terms locally,
adds one α, and sends a single element. Falcon's table says exactly this —
`MatMul (x×y)(y×z) = k·x·z`, where `x·z` counts *outputs*. Baru batched all
`out×in` products into one round (correct, and already a large win over the
per-element form) but still reshared each product separately.

On the MedMNIST CNN that is **239 520 products against 7 474 outputs — 32×**.

Fixed 2026-08-01 (`aby3_protocol::dot_batch_party`, `Backend::dot_batch`,
`gadgets::dense`). The trait default composes `mul_batch` with local addition, so
every other backend is unchanged; `TransportBackend` overrides it. Measured on
`bench-abstraction`'s 8×64 dense: **46 636 → 38 572 bytes**, the saving exactly
`512 − 8 = 504` elements × 16 B, with the round count untouched. The gadget
library now *beats* the hand-batched baseline it previously only matched.

### The comparison, per party

| | linear | nonlinear | total |
|---|---|---|---|
| Baru RSS + wrap3, **before** the fix | 0.958 MB | 0.377 MB | 1.335 MB |
| Baru RSS + wrap3, **after** | **0.030 MB** | 0.377 MB | **0.407 MB** |
| Falcon | 0.030 MB | 0.377 MB | 0.407 MB |
| Baru today (FHE + GC), *measured* | — | — | **10.1 MB** |

**The fix closes the linear-layer gap entirely — the reproduced stack is now at
communication parity with Falcon on MedMNIST**, and both are ~25× leaner than
Baru's current FHE+GC pipeline.

Latency is closer than communication: ~7 136 DReLUs at ~0.043 ms each is ≈307 ms
in this Rust simulation against Baru's measured 268 ms — roughly a wash *here*,
though Falcon's optimised C++ would be well ahead of both. Accuracy is identical;
both are exact on the same integer model.

Note the fix does **not** move `bench-medmnist`'s current numbers: that pipeline
uses FHE for its linear layers, so `gadgets::dense` is not on its path. It pays
off for the all-RSS route, which is the one now worth building.

### ⚠️ Measured end to end, and the estimate above was wrong by 33×

`baru bench-falcon --net` runs the MedMNIST shape entirely in RSS —
`conv(8) → conv(16) → dense(128) → dense(2)`, secret weights, truncation and
ReLU after every linear layer. Measured, per party:

| | | |
|---|---|---|
| linear (7 474 outputs × 16 B) | 0.120 MB | 0.9% |
| DReLU (7 136 × 52.8 B) | 0.377 MB | 2.8% |
| **truncation** (7 474 × ~1 717 B) | **12.832 MB** | **96.3%** |
| **total** | **13.328 MB** | |

**The 0.407 MB figure predicted above is wrong.** It counted linear and DReLU and
**omitted truncation entirely**, and it assumed 4-byte ring elements where the
wire carries 16. Both errors were mine; the measurement is what caught them.

**Truncation is 96% of an all-RSS fixed-point network.** Not the nonlinearity —
truncation, because our exact borrow is a 9-bit `a2b`, and every linear output
needs one. That reframes the whole comparison: the operation worth optimising in
this design is the one nobody was looking at.

It also sharpens a question about Falcon. They report 0.74 MB for LeNet while
insisting on *exact* truncation after every multiplication. If their truncation
cost anything like ours, that budget would be gone many times over — so their
construction must be far cheaper than an `a2b` borrow. The paper cites ABY3's
Fig. 3 rather than spelling it out, and **reproducing their cost now depends on
reading that construction**, not on anything left in their own text.

### Fixed: 13.328 → 1.378 MB/party (9.7×)

The diagnosis pointed at one thing — the borrow's 9-bit `a2b` was ~108 ring
elements at `u128`, so **1 728 B carrying 54 B of content**.

The earlier attempt to narrow `a2b` failed because its bit-shares are
*arithmetic*-encoded and feed wide multiplications on the `lt`/`max2` path. But
that is a property of the **consumer**, not the protocol, and truncation's
consumer is different: the borrow is read purely as a bit (`& 1`, then a `Z_2`
input to `Π_SS`) and never multiplied by a wide operand. So the narrowing is
safe *here* precisely where it was unsafe *there* — the same shape of argument
as the doubling that discharges `Π_PC`'s degenerate input.

`a2b_batch_party_at` carries the bit-shares at a caller-chosen width, with the
precondition stated on the function and `wire_bits ≥ 2` asserted (the XOR
encoding evaluates `a+b−2ab`, so one bit is not enough room even though the
value is). Truncation asks for 4.

The same reasoning then applies to everything else in `falcon_mpc`, which is
`l_bits`-consistent end to end: `dot_batch_party_at` and `mul_batch_party_at`
carry the linear layers and the selects at 32 bits rather than 128.

| | MB/party | |
|---|---|---|
| start (everything at `u128`) | 13.328 | |
| narrow the borrow's `a2b` to 4 bits | 2.326 | 5.7× |
| narrow linear + selects to `l_bits` | **1.378** | 1.7× |
| | | **9.7× total** |

All 63 exactness tests still pass, including `transport_truncation_is_exact` —
the narrowing changes what goes on the wire, not what is computed.

**Remaining gap to Falcon: 1.378 MB against 0.740 MB**, and theirs is for the
*larger* LeNet. So there is roughly another 2× to find, and truncation is still
the largest term. Whether their construction avoids the borrow comparison
entirely is the open question, and it is answerable only from ABY3's Fig. 3.

## Falcon as a selectable backend (2026-08-01)

Falcon is now a **nonlinearity mode**, not a parallel system:
`NonlinMode::{A2b, Gc, Falcon}` on `TransportBackend`, selectable as
`--falcon` wherever `--gc` already was.

That shape is forced by the finding this whole exercise produced: Falcon and
Baru share the replicated sharing *and the linear layer* unchanged —
`Π_MatMul` is `gadgets::dense`. The **only** thing that ever differed is the
comparison, so the only thing worth swapping is the comparison. Everything that
reduces to `lt_batch` — ReLU, max-pool, argmax — changes protocol for free.

`every_nonlinearity_mode_computes_the_same_comparison` asserts the three are
interchangeable before any cost is quoted.

### Measured on MedMNIST, one encrypted diagnosis

| mode | messages | traffic | latency | matches plaintext |
|---|---|---|---|---|
| `a2b` | 114 | 29.9 MB | 2367 ms | ✅ |
| `--gc` | 61 | 10.1 MB | 489 ms | ✅ |
| `--falcon` | 128 | **8.5 MB** | 1112 ms | ✅ |

**Falcon is the lowest-bandwidth route and `gc` the lowest-latency one** — 1.2×
fewer bytes against 2.3× more time. Neither dominates, which is exactly the
crossover the cost model exists to decide, now backed by three interchangeable
implementations rather than an argument.

### Two things it needed, both worth naming

**Output encoding.** `drelu_batch` produces a `Z_2` bit; every consumer of
`lt_batch` expects an *arithmetic* share it can multiply by a wide operand.
Lifting the bit is `Π_SS`, so that select is a real cost of interchangeability,
not an artefact.

**⚠️ Preprocessing is dealt, not generated.** `Π_WA` needs a mask with its bits
shared over `Z_p` and its own `wrap3`, and `δ = wrap3(x₁,x₂,x₃)` is derivable by
no party. Falcon generates these interactively (their Fig. 6); here they come
from a common seed, so **all three parties can reconstruct the mask**. The
online protocol and its costs are real; the offline phase is not yet. This is
the same disclosed caveat the repo already carries for dealer-based BFV keygen,
and it is the one thing between `NonlinMode::Falcon` and a deployable protocol.

## The measured crossover (2026-08-01)

This is the contribution *on top of* the reproduction, and it is now measured
rather than argued. Both routes compute the same MedMNIST diagnosis bit-exactly;
only the nonlinearity protocol differs.

```
baru bench-medmnist --dataset pneumonia --hybrid 1 --sweep
```

`--sweep` builds **one** model and **one** `CnnSession` for the entire grid, so
keygen and the encrypted weights are identical across cells and only the
protocol and the network vary; every cell is checked against the plaintext
argmax and the table refuses to print clean if any disagreed. Latency per
encrypted diagnosis, ms, `--precompute` on (so these are *online* figures):

| RTT | 1000 | 100 | 40 | 30 | 20 | 15 | 10 | 7 | 5 Mbps |
|---|---|---|---|---|---|---|---|---|---|
| **0 ms** gc | **638** | **3227** | **6554** | **8511** | 12357 | 16215 | 23915 | 33751 | 46869 |
| **0 ms** falcon | 1279 | 3528 | 6750 | 8583 | **12167** | **15740** | **22828** | **31804** | **43794** |
| **20 ms** gc | **2411** | **4839** | **8395** | **10292** | **14174** | **18046** | 25594 | 35618 | 48659 |
| **20 ms** falcon | 3873 | 6276 | 9585 | 11393 | 14998 | 18431 | **25543** | **34588** | **46850** |

**The boundary is in bandwidth, and it moves with RTT.** Falcon wins below
**20–30 Mbps at RTT 0** and below **10–15 Mbps at RTT 20 ms**; garbled circuits
win above. Higher RTT shifts the crossover *down* — it shrinks the region where
Falcon wins. Brackets, not interpolated points: those are the grid spacings
actually measured.

The 30 Mbps cell at RTT 0 is the boundary itself and behaves like it — across
repeat runs the winner flips (8511 vs 8583 one run, 8621 vs 8614 the next, a
0.1% spread). Neither ordering there should be quoted as a result; the finding
is that the two protocols are indistinguishable at that point, which is what a
crossover *is*.

### Why, and the check that it is not a coincidence

At 1 Gbit/s and RTT 0 the two differ almost entirely in compute: 638 ms for `gc`
against 1279 ms for `falcon`. So Falcon starts **~641 ms behind** and pays a
further **67 messages** of round latency, against which it saves **1.6 MB**.
Break-even is where the byte saving pays both back:

| | predicted | measured |
|---|---|---|
| RTT 0 ms | 20.0 Mbps | 20–30 Mbps |
| RTT 20 ms | 6.5 Mbps | 10–15 Mbps |

Right ballpark, right direction, right RTT-dependence. Falcon does consistently
a little better than the linear prediction, and the reason is structural rather
than noise: `NetworkProfile` charges per *round*, not per message, and Falcon's
67 extra messages batch into fewer than 67 extra rounds — so the true RTT
penalty is smaller than `67 × RTT`, which is exactly the direction of the
residual, and it grows with RTT (as the 6.5-vs-12 row shows) precisely because
that is the term being over-charged.

### What this licenses saying

> Given a deployment's RTT and bandwidth, **pick** the nonlinearity protocol
> rather than fixing it — with the crossover measured, not modelled.

Falcon always uses `wrap3`. Baru has three interchangeable implementations of
the same comparison and a cost model whose purpose is choosing between them, so
this is a claim the reproduction makes possible and Falcon cannot make.

Two honest limits on it. The compute term is Rust-in-simulation against what
optimised C++ would do, and it is the term that sets the RTT-0 intercept — a
faster `Π_PC` moves the crossover right. And `a2b` is dominated everywhere here,
so the practical choice is two-way, not three-way.

## What remains

- **An end-to-end LeNet run** rather than a per-op extrapolation. Everything it
  needs exists; what is missing is the convolution *packing* (expressing conv as
  a matmul over patches) and the harness, neither of which is new protocol.
- **A real offline phase** for the Falcon backend (see the caveat above) — the
  one thing between `NonlinMode::Falcon` and something deployable.

## Strategy from here (2026-08-01)

FHE does not win in this setting and garbled circuits do not either, so the plan
is to **match Falcon's approach first, then build on it**. What that needs, and
what it does not:

| piece | state |
|---|---|
| `Π_PC`, `Π_WA`, `Π_DReLU`, `Π_SS`, `Π_ReLU` | ✅ on transport, validated |
| `Π_Maxpool` | ✅ on transport |
| MatMul | ✅ already — `gadgets::dense` is the same RSS protocol |
| truncation | local simulation only |
| network runner | not started |

**Where parity actually stands.** On communication we are already there:
~0.44 MB/party extrapolated against Falcon's reported 0.74 MB. On *compute* we
are not — sign extraction runs at ~0.043 ms each, so LeNet's 8 044 DReLUs come
to ~350 ms against Falcon's 47 ms total. That gap is Rust-in-simulation versus
optimised C++ on a LAN, not protocol, but it is real and should not be papered
over when the reproduction is written up.

**And the "something new" is already visible in the measurements.** The four-way
benchmark shows `wrap3` and garbled circuits winning on *different axes* —
garbling is ~5× faster in wall-clock while sending ~30× more. Falcon always uses
`wrap3`. Baru has a cost model (`mpcc::assign`) whose entire purpose is choosing
per operation, and now has measured costs for all three nonlinearity routes.

That is the crossover, and it is measured — see [The measured
crossover](#the-measured-crossover-2026-08-01) above.

The remaining internal comparison is the three construction routes — Falcon
hand-written, Baru gadgets, Baru compiled — over the same transport, with
message and byte counts, exactly as `bench-compiled-nn` does. All three checked
against one plaintext reference; a cost comparison between routes computing
different things is meaningless.

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
