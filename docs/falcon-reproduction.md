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
| `Π_DReLU` / `Π_ReLU` | ✅ both, on the transport (was: garbled + `a2b` only) | stage 4 |
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

   Composing this with Baru's `relu∘rescale` convention needed the truncation
   protocol, since Falcon's ReLU has no rescale. ✅ **Closed 2026-08-04**:
   `random_trunc_pairs` supplies secret `Π_Trunc` preprocessing and
   `transport_backend::relu_falcon_batch` runs gate-then-truncate, so
   `Nonlin::Falcon` is a pure Falcon route end to end. The order is forced —
   see *What a pure Falcon ReLU actually needs* below.
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

#### The extrapolation, measured at scale and with both phases (2026-08-04)

Two tests replace the two things this section was extrapolating.

**The scaling assumption holds exactly.** `the_lenet_extrapolation_holds_at_lenet_scale`
measures the online DReLU at LeNet's own batch size instead of multiplying up from
n ≤ 1000:

| batch step | marginal B/DReLU | rounds |
|---|---|---|
| 500 → 1000 | 52.8 | 10 |
| 1000 → 2000 | 52.8 | 10 |
| 2000 → 4000 | 52.8 | 10 |
| **4000 → 8044** | **52.8** | 10 |

0% spread over a 16× range with a constant round count, so multiplying a small-batch
measurement up to 8 044 is arithmetic rather than optimism. That was the untested link
and it is now measured.

**And the online figure is 9% of the cost.** `the_online_drelu_figure_is_not_the_whole_cost`
measures both phases with the real offline phase:

| | B/DReLU | LeNet, MB/party |
|---|---|---|
| offline | **558.8** | 4.50 |
| online | 52.8 | 0.42 |
| **total** | **611.5** | **4.92** |
| Falcon reported | — | **0.74** |

So this reproduction costs **6.6× Falcon** once preprocessing is charged — and the
online-only figure of 0.42 MB, which is *below* Falcon's 0.74, is an artefact of
counting nothing offline. **The flattering reading was wrong, and it was wrong in the
way this document's four withdrawn claims were wrong**: comparing one protocol's
partial cost against another's total.

#### Two accounting cautions this exposed

**`4kn` is a bit-complexity line; the measurements above are wire bytes.** `ELEM_BITS`
is **6** (derived from `p = 37`, pinned by a test "as Falcon counts it"), so the claim
above that this reproduction runs "at 8 bits per element" with a factor "exactly 8/6"
is about **serialisation** — a 6-bit field element occupying a whole byte on the wire —
not about the field. Comparing serialised bytes against an analytic bit count is the
same category error as the ring-share padding in §7.3, and neither the 1.03 MB nor the
1.33× should be read as a protocol comparison.

**What is safe to say.** The round counts and the linearity are measured and match.
The absolute byte comparison against Falcon's 0.74 MB is not settled, because it needs
both sides charged for the same phases and counted in the same units, and the analytic
line supplies neither. What *is* settled is the direction: with preprocessing counted,
this implementation is several times more expensive, not cheaper.

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

### The crossover, measured properly (2026-08-03)

The boundary this document reported at **15–20 Mbps** is withdrawn: it was measured
while `Π_WA`'s masks were dealt from a shared seed — free precisely because they were
not secret — so it compared Falcon's *online* cost against garbled circuits' *total*
cost. With a real offline phase (bit composition, above) and the ring encoding
narrowed to the model's own width:

| RTT | 1000 | 100 | 60 | 40 | 30 | 20 | 15 | 10 Mbps |
|---|---|---|---|---|---|---|---|---|
| **0 ms** gc | **566** | **2928** | **5405** | **6672** | 8228 | 12294 | 15978 | 23525 |
| **0 ms** falcon | 1950 | 5459 | 5805 | 6824 | **7898** | **10714** | **13533** | **19297** |
| **20 ms** gc | **2273** | **4450** | **5931** | **7811** | **9713** | **13515** | **17364** | 24996 |
| **20 ms** falcon | 6865 | 8087 | 9339 | 10645 | 12153 | 15029 | 17822 | **23386** |

> ⚠️ **Superseded — this table is the *partial* Falcon route.** When it was measured,
> a pneumonia diagnosis sent **5 256** comparison elements through `lt_falcon_batch`
> (`MpcPhases::falcon_comparisons`), exactly its two max-pool layers
> (`8·13²·3 + 16·5²·3`), while its **ReLUs did not use Falcon at all**: roughly 7 500 of
> the ~12 700 nonlinearities took `a2b`, because `fused_relu_rescale` had no Falcon arm.
> So this table compares **(a2b ReLU + Falcon max-pool)** against **(garbled
> everything)**, not one comparison protocol against another.
>
> The code had always said so (`cnn::fused_relu_rescale`); the table had not. It did not
> invalidate the boundary — two runnable configurations were really measured — but the
> label overstated what varied between the columns, the same failure mode as the four
> withdrawn claims below.
>
> **Resolved 2026-08-04**: `fused_relu_rescale`'s Falcon arm now calls
> `relu_falcon_batch`, so the route is pure. The re-measured boundary is in *The pure
> route, measured* below; this table is kept because the comparison between the two is
> itself the finding.

#### What a pure Falcon ReLU actually needs (2026-08-04)

Not a wiring change, and the current arrangement turns out to be the sensible one
rather than an oversight. `fused_relu_rescale_batch` takes **both** the shifted value
and the sign bit from a *single* `a2b`: `shifted = b2a(bits[shift..])` and
`keep = 1 − bits[nbits−1]`. So routing only the sign bit through Falcon would pay for
the `a2b` *and* the DReLU and be strictly worse than either alone. There is no useful
partial step — either the whole ReLU is Falcon or none of it is.

Doing it all in Falcon means using Falcon's own exact truncation for the shift, and
that dependency is now **closed**: `random_trunc_pairs` generates `TruncPreproc`
(`r`, `r >> d`, `r mod 2^d`, plus the `Π_SS` selector) by bit composition, with no
party learning `r`. Until it landed, the only construction in the tree was the
locally-computed split inside `transport_truncation_is_exact` — dealt, and therefore
not secret. That was the last dealt piece in the Falcon stack after `random_wrap_masks`
closed `Π_WA`.

**It is far cheaper than the `Π_WA` masks, and for a structural reason.** Both build
their mask from shared random bits, but wrap3 then needs `δ` — a *carry* out of a
three-summand sum, which costs an `a2b`. Truncation needs no carry, because **shifting
is re-weighting**:

```text
r         = Σ_k     2^k     · b_k
r >> d    = Σ_{k≥d} 2^{k−d} · b_k
r mod 2^d = Σ_{k<d} 2^k     · b_k
```

Three *local* weighted sums of one set of bit-shares. The invariant `truncate_batch`
depends on — `r = (r>>d)·2^d + r_lo`, exactly, over the integers — therefore holds by
construction rather than being computed, so no wrong carry can break it. The whole
protocol is the daBits — the mask's `ℓ` bits plus one selector for each of the two bits
`truncate_batch` lifts — and only their *ring* images, since a truncation pair has no use
for the `Z_p` side.

Three tests carry it: `composed_truncation_pairs_are_mutually_consistent` reconstructs
all three parts from the shares and checks them against each other (by-construction is a
claim about code, so it is verified from the data),
`no_party_view_determines_a_truncation_mask` checks the secrecy, and
`truncation_is_exact_over_securely_generated_pairs` runs `truncate_batch` over generated
pairs across every value in range. The dealt-preprocessing test is kept alongside: one
isolates the protocol, the other says its preprocessing needs no dealer.

**A pure Falcon ReLU now exists** (`transport_backend::relu_falcon_batch`, 2026-08-04),
gating before shifting:

```text
ReLU(v)            = v · [v ≥ 0]        Π_DReLU, then one batched multiply
ReLU(v) >> shift   = truncate(ReLU(v))  Π_Trunc
```

Gating first keeps the truncation's input non-negative, which is where the protocol's
unsigned floor and the plaintext's arithmetic shift coincide — so
`truncate(ReLU(v))` equals `(v >= 0) ? (v >> shift) : 0` *exactly*, not approximately.
`falcon_and_a2b_relu_agree` checks that over inputs straddling zero and sitting on the
truncation's borrow boundary, and `encrypted_matches_plaintext_on_real_scans` confirms it
across three datasets × five configurations.

> **This ordering was originally described here as forced.** It was, under the
> precondition the truncation then carried (`a + r` must not wrap, which only a bounded
> `a` guarantees). That precondition is gone — see below — so the ordering is now a
> *convenience* rather than a requirement, and the paragraph above no longer claims
> otherwise.

### The truncation leaked, and correctness testing could not see it (2026-08-04)

`truncate_batch` **opens** `c = a + r`. Its documented precondition was that `a + r` must
not wrap, which is a *ceiling* on the mask — and `relu_falcon_batch` satisfied it the
cheapest way available, with `r_bits = nbits − 1`. That is the tightest correct width and
the worst private one, because a mask no wider than the value hides almost nothing:

```text
posterior support for a given c   =  min(c, A) − max(0, c − A)
                                  =  1   at c = 0        (a is determined)
I(a ; c) for a uniform on [0,A)   ≈  1 / (2 ln 2) ≈ 0.72 bits per truncation
```

Two honest limits on that figure. It assumes `a` **uniform**, which real ReLU outputs are
not — they are skewed toward zero, and since `I(a;c) ≤ H(a)`, a lower-entropy `a` is
leaked *less* in the mutual-information sense, so 0.72 is not a lower bound on the real
distribution. And the per-activation figures do not simply add: activations within an image
are strongly correlated, so ~12 700 truncations per diagnosis does not give
`12 700 × 0.72` bits about the image. The aggregate leakage was never measured.

What makes it disqualifying is not the size of the number but its sign. The worst case is
total (`c = 0` determines `a`), the leak is *structural* rather than a parameter choice, and
the `a2b` route it replaced leaks **nothing at all** — being a circuit that opens no
function of a secret. So wiring `Π_ReLU` in traded away a security property outright, and
no amount of arguing about bit counts changes that a strictly-private route was replaced by
a less-private one.

**Nothing in the test suite could have caught it.** Every correctness test passed, before
and after, because leakage is not a wrong answer. Worse, the three tests that exercise
truncation preprocessing all passed `r_bits = l_bits − 1` — chosen to satisfy the no-wrap
bound — so they never ran a wrapping mask and would have passed just as happily on the
leaking protocol *forever*. The property was invisible to the entire apparatus built to
check this stack.

That makes it a different failure from the four withdrawn crossover claims above. Those
were wrong *numbers*, caught by decomposing a total into the terms the cost model charges.
This was a wrong *protocol*, and the thing that caught it was reading what
`truncate_batch` opens and asking what the opened value is distributed like — an argument,
not a measurement.

#### Two ways to fix it, and why the obvious one is the wrong one

**Statistical slack (`κ`).** Widen `r` to `val_bits + κ` so the distributions of `c` for
different `a` are within `2^−κ`, and widen the ring to hold `a + r`. Correct, secure at
`2^−40`, implemented and measured — and **incompatible with the narrowed wire encoding**,
because narrowing *spends* the slack between the value width and the ring width and
statistical hiding *needs* that slack. The two optimisations want the same bits. Measured
at `nbits = 20`: a 61-bit ring, and Falcon's ReLU **1.9× dearer in bytes than `a2b`'s** —
a secure protocol whose cost erased the reason to prefer it.

**A wrap bit (what ships).** Let `r` be uniform over the *whole* ring. Then `c` is uniform
and **perfectly** independent of `a` — better than statistical — but `a + r` now wraps
about half the time, so the wrap is corrected instead of avoided:

```text
a>>d = (c>>d) − (r>>d) − borrow + wrap·2^(ℓ−d)
  borrow = 1 ⟺ c_lo < r_lo      (already computed)
  wrap   = 1 ⟺ c   < r          (provably the overflow indicator)
```

`wrap` has the same shape as the borrow — public `c`, secret `r` — so it costs one more
narrowed `a2b` and one more `Π_SS` lift, and **the ring stays narrow**. Both bits are read
only as bits, so both `a2b`s narrow to 4 columns.

The identity is checked exhaustively rather than sampled
(`the_wrap_correction_is_exact_over_every_mask`, all 524 288 `(a, r)` pairs at `ℓ = 10`),
because the corners are where it would fail: the wrap fires constantly once `r` covers the
ring, but the case where the borrow *and* the wrap both apply is rare enough for a sampled
test to miss. The test also pins the non-obvious half — that `[c < r]` really is
`[a + r ≥ 2^ℓ]`, which is what makes the wrap computable from an opened `c` at all.

#### The leaking configuration is now unrepresentable

The lesson taken was not "document the width". `random_trunc_pairs` calls
`assert_full_ring_mask(r_bits, l_bits)`, which refuses anything narrower than the ring, and
that refusal has its own test (plus a companion asserting the full-ring case is *accepted*,
so a check that rejected everything would not pass for the wrong reason). The
narrow-mask request that leaked cannot now be made.

`falcon_keeps_the_narrowed_wire` pins the other half: all three routes share one wire
width, so nothing silently reintroduces a route-specific ring.

One deliberate structural choice: `falcon_drelu_ring` was extracted from
`lt_falcon_batch` first, so `lt = 1 − drelu` and `ReLU = v · drelu` share one offline
phase, one id layout and one `Π_SS` lift. Two places computing what has to be one thing
is how both the weight-cursor and the mask-pool bugs on this page happened.

#### The pure route, measured (`--repeat 3`, medians, pneumonia, 2026-08-04)

The first grid on this page that is (a) a pure Falcon route, (b) a *private* one, and
(c) repeated rather than single-shot.

| Mbps | 40 | 30 | 20 | 15 | 10 | 8 |
|---|---|---|---|---|---|---|
| **RTT 0** gc | **6552** | **8453** | 12337 | 16639 | 24269 | 29982 |
| **RTT 0** falcon | 7255 | 8607 | **11859** | **15109** | **20979** | **24717** |
| **RTT 20** gc | **8157** | **10098** | **13925** | **17770** | **25744** | **31504** |
| **RTT 20** falcon | 13572 | 14969 | 17860 | 20784 | 27476 | 31836 |

> **At RTT 0 the crossover sits between 20 and 30 Mbps.** At RTT 20 ms there is **no
> crossover down to 8 Mbps** — garbled circuits win throughout, and the boundary is at or
> below the bottom of the grid. Higher RTT moves the boundary *down*, shrinking Falcon's
> region, because Falcon sends 7.4× the messages.

The mechanism, from the same run:

| protocol | messages | MB wire | of which ring |
|---|---|---|---|
| gc | 61 | 10.1 | 0.5 |
| falcon | 450 | **3.4** | 2.3 |

Falcon is **2.94× cheaper in bytes and 7.4× dearer in messages**, so the two terms
genuinely trade and the boundary is where the trade balances. Both protocols are charged
for their preprocessing and neither pays for `u128` padding.

**What making it private cost:** +39 messages and +0.3 MB per diagnosis (413 → 450,
3.1 → 3.4 MB) against the leaking version — the wrap bit's extra `a2b` and `Π_SS` lift,
once per truncation. Cheap enough that the crossover survives, which is the whole reason
the wrap bit is preferable to `κ` slack; under slack the byte ratio inverted and there was
no boundary left to find.

##### The two boundaries: a warm server's is a decade of bandwidth higher

Falcon's preprocessing is query-independent, so a deployment that generates it before a
query arrives pays only the online phase. `--warm` measures that, and the difference is not
marginal (pneumonia, single run):

|  | RTT 0 ms | RTT 20 ms |
|---|---|---|
| **cold** — server pays the total | crossover 20–30 Mbps | none down to 8 Mbps |
| **warm** — preprocessing hoisted | **falcon wins to ≥100 Mbps** | crossover 20–30 Mbps |

So the choice is not "garbled circuits unless the link is very slow". It is: **a cold server
should prefer garbled circuits above ~25 Mbps; a warm one should prefer Falcon almost
everywhere at low RTT, and above ~25 Mbps only when RTT is high.** Both rows are real
deployments, which is why both are quoted.

Warm falcon also sends **fewer** messages than cold falcon — 310 against 450 — because one
pool fill per segment batches better than several inline draws. Bytes are unchanged at
3.4 MB. That is a side benefit of pooling and not the reason for it.

> ⚠️ **`gc` gains nothing here, and that asymmetry is the comparison, not a flaw in it.**
> Garbled circuits have no query-independent phase to hoist in this pipeline, so the `gc`
> column is identical warm and cold. Presenting the warm boundary *instead of* the cold one
> would therefore be the same error as this page's first withdrawn crossover — one
> protocol's online cost against another's total. The banner, `SweepCell::warm`, and this
> table all say so.

**The first warm sweep was wrong, and an unchanged control is what said so.** `gc`'s traffic
came back as 154 messages / 12.2 MB against its usual 61 / 10.1, on a run where nothing in
the garbled path had changed: the pool plan was being applied to *every* route, so `gc` was
generating Falcon preprocessing it never reads. The latency table looked completely
plausible. `warming_falcon_leaves_the_gc_control_untouched` now asserts `gc`'s messages,
bytes and ring bytes are **equal** cold-to-warm — deterministic quantities, so equality
rather than a tolerance — and separately that Falcon's traffic *does* move, since otherwise
the test would pass by warming nothing.

One asymmetry remains, labelled rather than fixed: in a warm sweep the **latency** column is
an online cost but the **traffic** table is still a total, because the transport counts bytes
for the whole run. The two tables are not on the same footing and the banner says so.

##### How well this grid resolves the boundary — less well than three repeats suggest

Every cell above is marked a winner and none a tie, and that is **not** strong evidence.
`--repeat` re-runs each cell inside one `sweep_crossover` call, sharing one `CnnSession`,
back to back, on warm caches. Those repeats spread under 2%. But re-running an *unchanged*
`gc` configuration as a separate **process** moved one 60 Mbps cell by **13%** — and that
is the variance that matters for reproducibility. The tie test therefore measures jitter,
not run-to-run agreement, and is over-confident by construction.

Read it as: cells within ~13% of each other are unresolved, and the boundary is good to
about **±1 grid step**. At RTT 0 that makes 20–30 Mbps a genuine bracket; the 30 Mbps cell
itself (8453 vs 8607, 1.8% apart) is a coin flip. At RTT 20 the 8 Mbps cell (31504 vs
31836, 1.1%) is likewise a tie in substance, so "no crossover above 8 Mbps" is the honest
reading rather than "gc wins at 8".

The same applies to the warm table above, and it is worth being explicit about which of its
cells carry the claim. At RTT 0 the margins run 8% (100 Mbps), then 21/26/30/36/41% below —
so "falcon wins to ≥100 Mbps" rests on the lower cells, and the 100 Mbps cell alone would
not support it. At RTT 20 the 30 Mbps cell is 1% apart (a tie) and 40 and 20 Mbps are 8%
and 9%, so the warm crossover is a **20–40 Mbps band** of which "20–30" is the midpoint,
not a resolved point. More repeats would not fix this: they measure jitter, not the
run-to-run component that dominates.

This is recorded because the previous four withdrawn boundaries on this page were all
single-run numbers that looked exactly this convincing, and adding repeats without saying
what they *cover* would have reproduced the same error with more decimal places.

#### Read it as a *total-cost* boundary, because 89% of Falcon's cost is offline

Measured per comparison and per party at ℓ = 20 (`falcons_cost_is_mostly_offline`):

| phase | ms | messages | bytes/op |
|---|---|---|---|
| offline | 1048 | 22 | **289** |
| online | 118 | 9 | **34** |

**Preprocessing is 89% of the bytes, 71% of the messages and 8.9× the time.** So a
comparison that charges totals — which the sweep above does, because
`lt_falcon_batch` runs preprocessing inline — charges Falcon roughly **nine times its
online cost**. An online-only boundary would sit far above 30–40 Mbps.

That is not a reason to quote the online figure instead. Falcon's preprocessing is
query-independent and can be generated before a query arrives, so both numbers
describe real deployments — a cold one pays the total, a warm one pays the online
cost. The error to avoid is quoting one protocol's online cost against another's
total, which is precisely how this document's first crossover went wrong. Both
figures, labelled, or neither.

It also corrects a claim made earlier on this page: that Falcon is not cheaper than
`a2b` once both phases are counted. At the current encoding it is — 289 + 34 = 323 B
per comparison against `a2b`'s 954 B — because the padding that used to hide this
fell hardest on the preprocessing-heavy protocol.

**The withdrawn number was wrong in two directions at once.** It put the boundary at
15–20 Mbps by comparing Falcon's online cost against gc's total — which flattered
Falcon — while simultaneously inflating Falcon's bytes 2.9× with padding, which
penalised it. Correcting both moves the boundary *up*, to 30–40: Falcon wins over a
**wider** region than the original claim, for entirely different reasons. Two errors
had partly cancelled.

#### The three answers before this one

Each was reproducible, computed the correct diagnosis, and agreed with the cost
model — the first even matched a break-even prediction of 17.1 Mbps against 15–20
measured. None of that detected the problem.

| reported | why it was wrong |
|---|---|
| 15–20 Mbps | Falcon's *online* cost against gc's *total* |
| no crossover at any bandwidth | Falcon inflated 3.2× by serialisation padding |
| a tie, no boundary worth quoting | the grid stopped above the boundary |

What caught them was decomposing each total into the terms the cost model charges —
phases, messages, bytes, and the padded share of bytes — and asking whether both
protocols were being charged for the same things. The sweep now prints those terms
next to every table for that reason.

A methodological rule that came out of it: **a two-image spot check is not evidence
about a wire format.** Narrowing to a fixed 21 bits was bit-exact for the `2^20`
dataset and returned the wrong diagnosis for *every* image of the `2^22` one; only
`encrypted_matches_plaintext_on_real_scans`, over every embedded image of both
datasets, caught it.

### ⚠️ Withdrawn: the cheaper δ is not cheaper (2026-08-04)

This section derived an alternative to the `a2b` that computes δ. Writing
`w₀₁ = wrap2(x₀,x₁)` and `y = (x₀+x₁) mod L`:

```
δ = w₀₁ ⊕ [y + x₂ ≥ L]
```

verified exhaustively over all 2^18 triples at ℓ = 6. Party 0 holds both `x₀` and `x₁`
so it computes `w₀₁` and `y` for free, leaving one comparison between a value it knows
and a value parties 1 and 2 know — a millionaires' problem rather than another `Π_WA`,
and `gc_protocol`'s no-OT three-party comparator settles it in one round.

The construction is correct and it fits the code better than expected: `Shape::Comparator`
is "the sign of x + y", so at width ℓ+1 over garbler 0's *unreduced* `x₀+x₁` and
garbler 1's `x₂` it yields bit ℓ of the sum, which **is** δ — `w₀₁` falls out of the
unreduced summand and needs no separate handling.

**It should still not be built, because it is four times more expensive.** Measured at
ℓ = 32, per element:

| | bytes | rounds |
|---|---|---|
| δ via `a2b` at `DELTA_WIRE_BITS = 4` (+ conversion) | **255** | ~7 |
| δ via garbled comparator (ℓ AND gates at 32 B) | **1024** | 1 |

It trades bytes for rounds at 4:1 against, and the offline phase is byte-dominated.

**Why the original estimate was wrong, which is the part worth keeping.** When this was
derived, ring bit-shares still went on the wire as `u128` and the `a2b` cost ~3700 B
per element — the garbled route genuinely would have been ~3.6× better. Narrowing the
wire (§7.3) made the thing this was meant to improve nine times cheaper, and the
improvement obsolete. The estimate also compared garbled **bytes** against `a2b`
**rounds** and called the former cheaper, which is the same incommensurability error
this document records four withdrawn claims for.

**Where the offline cost actually sits**, measured by
`the_offline_bytes_split_between_dabits_and_the_delta_computation`:

| | B/element | share |
|---|---|---|
| `ℓ` daBits (the mask's bits over `Z_p`) | 304 | 54% |
| δ: `a2b` + ring→`Z₂` conversion | 255 | 46% |
| **total offline** | **559** | |

A near-even split, so neither half is a single target — and an arithmetic guess put it
at 79/21 before it was measured. The daBits are `ℓ` draws per element and are the
larger half; a construction that needs fewer of them would be the thing to look for.

### Why the ratio widens instead of narrowing

The earlier analysis had Falcon starting ~703 ms behind on compute and paying 112
extra messages against a 1.5 MB byte saving, which predicted break-even at 17.1 and
4.0 Mbps — and matched the measured 15–20 and 5–10. That model was right about the
protocol it described. It no longer describes this one.

Bit composition adds `ℓ` daBits **per element** plus one `a2b` over the batch. At
ℓ = 21 and a conv layer's worth of activations that is ~170 000 daBits per
comparison batch, so the offline phase now dominates both the compute term and the
byte term it was supposed to be trading against. The gap grows with bandwidth
because at 1 Gbit/s the byte saving buys almost nothing while the extra compute and
rounds are paid in full: **552 vs 3363** at RTT 0, a 6.1× loss, against 1.85× at
10 Mbps. A bandwidth-advantaged protocol should show the reverse.

### What this licenses saying

The *architectural* claim survives and is worth separating from the withdrawn one:

> Baru has three interchangeable implementations of the same comparison, verified
> bit-exact against each other, behind one interface — so which protocol backs
> every ReLU, max-pool and argmax is a one-line choice rather than a rewrite.

That is what made this measurement possible at all, and it is what let the
withdrawal be *detected* rather than shipped. What is no longer licensed is the
"pick per deployment" claim, because on this workload there is currently nothing to
pick: `gc` dominates both alternatives everywhere measured.

Two limits that still apply, and now cut the other way. The compute term is
Rust-in-simulation rather than optimised C++, and it sets the RTT-0 intercept where
Falcon's loss is worst — so a faster `Π_PC` and a cheaper offline phase both move
the boundary back toward Falcon. Neither is measured, so neither is claimed.

## What remains

- **Carry the sharing modulus in `ShareRef`, then narrow the general wire path.**
  Ring traffic is packed at whatever width the caller declares, and the general
  `mul_batch`/`a2b_batch` path declares **128** — so a 20-bit value travels in 128
  bits and 84% of it is padding. Two attempts to fix that by masking have broken
  correctness (`max2(9,6)` returned 22 020 105; the CNN returned class 5 for 7),
  because `nbits` records the width of the *value* while the sharing must stay
  valid modulo whatever a downstream consumer multiplies it by. `TransportBackend::store`
  documents the real fix: `ShareRef` has to carry the sharing modulus separately.

  **What it is worth, and why it may reorder the whole comparison** (upper bounds,
  assuming every 128-bit ring share carries a 20-bit value):

  | protocol | wire MB | bit-packed |
  |---|---|---|
  | `a2b` | 30.0 | **5.1** |
  | `gc` | 10.1 | 9.7 |
  | `falcon` | 10.1 | 2.4 |

  `a2b` is the protocol this document has been dismissing as never winning, and it
  is the one with almost nothing *but* ring traffic — so packing would make it the
  **byte-cheapest** of the three, ahead of garbled circuits. It stays round-heavy
  (9 rounds against 1), so this is a claim about the byte axis only; but the byte
  axis is exactly where the crossover lives. Until the refactor lands, the
  three-way ranking in this repo is partly a ranking of how much each protocol
  happens to use the padded encoding.

  The Falcon row already reflects the local narrowing described above, so it
  overstates its own remaining saving; `a2b` and `gc` are still at the default.

- ~~**An offline/online split for the Falcon backend.**~~ **Built 2026-08-04.**
  `TransportBackend::fill_falcon_pool` generates the correlated randomness ahead of a
  query; `lt_falcon_batch` draws from the pool or falls back to generating inline, and
  `falcon_pool_stats` reports which happened. A **cold** server pays the total, a
  **warm** one pays only the online phase, and both are now separately measurable —
  which is what the withdrawn crossover needed and did not have.

  Three properties are pinned by `a_pooled_falcon_run_matches_an_inline_one`: the
  answers are identical either way, every item is served from the pool, and **nothing is
  generated on the critical path**. That last assertion is the load-bearing one — a
  benchmark quoting an online latency while quietly generating preprocessing inline
  would be reporting a total and calling it online, which is precisely the error the
  crossover was withdrawn for. A `served`/`missed` counter makes it visible rather than
  plausible. The pool also refuses a **width mismatch** instead of serving it, since
  randomness drawn at one `ℓ` returns wrong comparisons at another.

  **Measured at the protocol level** by `the_warm_falcon_cost_is_measured_against_the_cold_one`,
  at ℓ = 20 over 256 comparisons:

  | | critical path | offline |
  |---|---|---|
  | cold | **1664.6 ms** | — |
  | warm | **570.9 ms** | 1069.2 ms |

  **2.9× off the critical path, with total traffic byte-identical** (99 064 both). The
  byte equality is the assertion that matters: a warm run that sent materially less
  would mean the pool had skipped work rather than pre-done it, so the test pins traffic
  to within 10%.

  2.9× rather than the ~11× the 91%-of-bytes figure suggests, because this is wall clock
  at *zero latency*, where compute dominates and the online phase still runs `Π_PC`'s
  tree and the `Π_SS` lift.

  **The gap does widen as bandwidth binds, and by less than predicted**
  (`the_warm_gap_widens_as_bandwidth_binds`, ℓ = 20, n = 128):

  | bandwidth | cold | warm | ratio |
  |---|---|---|---|
  | unlimited | 898 ms | 333 ms | 2.70× |
  | 5 Mbps | 1123 ms | 474 ms | 2.37× |
  | 1 Mbps | 1484 ms | 515 ms | 2.88× |
  | 0.2 Mbps | 3086 ms | 790 ms | **3.91×** |

  Three honest qualifications. The trend is **not monotonic** — 5 Mbps dips below the
  unlimited case — so this is a direction, not a curve. 3.91× is well short of the
  byte-share asymptote, because the warm run still pays the *online* bytes: the limit is
  total/online, not 1/(1−offline share), and at ℓ = 20 there are fewer daBits per element
  than the ℓ = 32 measurement that gave 91%.

  And these are **single-run wall-clock figures, sensitive to machine load** — every cell
  moves ~10–20% between runs, and the unlimited row moves most because it is pure compute.
  Read the direction, not the digits. The earlier version of this table reported 2.94× and
  4.07× for the first and last rows; the difference from the values above is partly the
  wrap-bit truncation and mostly run-to-run variance, which is why only the *shape* is
  claimed.

  **A first attempt at this measured 100 and 20 Mbps and found the ratio flat.** That was
  not a refutation: at this batch size those pipes contribute ~140 ms against ~840 ms of
  compute, so the experiment never entered the regime the hypothesis is about. Worth
  recording because a flat result from outside the relevant regime looks exactly like a
  falsified prediction.

  **The sweep is now done too** (`--warm`, 2026-08-04). It was not the one-line call this
  note first claimed: the CNN runs **one `run_parties` call per MPC segment**, each with a
  fresh backend, so a single pool cannot span a diagnosis and each segment needs its own.
  Threading sizes in call order would have reintroduced the positional coupling that broke
  the mask pool when channel splitting landed, so they travel under a `Segment` enum key
  (`Conv(i)`, `Hidden(i)`, `Argmax`) in a `FalconPoolPlan` — which also extends to any
  depth, as a fixed-field struct would not.

  Sizes are **counted, not derived**: a cold run measures the plan (`CnnRun::falcon_plan`)
  and a warm run consumes it, because deriving them would mean mirroring the batching
  inside `maxpool2d`, `fused_relu_rescale` and `argmax`. An under-size is safe but visible —
  `lt_falcon_batch` falls back to generating inline, and `falcon_pool_missed` goes non-zero,
  which `a_pooled_diagnosis_matches_an_inline_one` treats as a hard failure rather than a
  warning: a miss means the reported "online" latency is secretly a partial total.

  | model | comparisons | Conv(0) | Conv(1) | Hidden(0) | Argmax | online MPC |
  |---|---|---|---|---|---|---|
  | pneumonia | 7 137 | 5 408 | 1 600 | 128 | 1 | 2185 → 760 ms (**2.88×**) |
  | organa | 14 154 | 10 816 | 3 200 | 128 | 10 | 3920 → 1224 ms (**3.20×**) |
  | derma | 7 142 | 5 408 | 1 600 | 128 | 6 | 2299 → 756 ms (**3.04×**) |

  These supersede the 5 256 / 10 512 / 5 256 recorded earlier on this page, which counted
  **max-pool comparisons only** — before `Π_ReLU` routed activations through the same path.

  Two things the measurement corrected, neither of which a test would have caught:

  **The first figure was 1.18×, and it was wrong.** `mpc_ms` brackets the whole
  `run_parties` call and the pool fill happens *inside* it, so a warm run's `mpc_ms` still
  contained its own offline phase — the comparison was (fill + online) against
  (inline + online). `CnnRun::online_mpc_ms()` exists so there is one place to be right
  about this; it turned 1.18× into 3.43× on the same data. This is the page's oldest error
  in miniature: a total quoted as an online cost.

  **`argmax` does reach the Falcon path.** The code comment said the protocol flag does not
  reach `argmax` because it reduces over scalar `lt` — true of the *garbled* flag, false of
  Falcon's. The counter reports exactly `nclasses − 1` for each model (1, 10, 6 above), so
  that segment gets its own small pool. Two protocols diverging at one call site is exactly
  what a comment gets wrong and a counter does not.
- ~~**Check whether the offline phase can be cheaper.**~~ **Checked, and the answer was
  no** — see the withdrawn cheaper-δ section above. The offline bytes split 54/46
  between the `ℓ` daBits and the δ computation, and the garbled alternative to the δ
  `a2b` costs 1024 B against the 255 B it would replace. The daBits are the larger half,
  so a construction needing fewer of them is where to look next.
- ~~**An end-to-end LeNet run** rather than a per-op extrapolation.~~ ✅ **Done
  2026-08-04** (`lenet_end_to_end_is_exact_and_costed`). Kernel size was a module constant
  in `bench::cnn` — `fhe::conv_pack` was always general in `kh`/`kw` — so this needed a
  per-layer `k` field, not new protocol. Measured (release, random weights, `NetworkProfile::INSTANT`):

  | | |
  |---|---|
  | architecture | `1×28×28 → conv(6,5×5) → pool → conv(16,5×5) → pool → dense(120) → dense(84) → dense(10)` |
  | latency | **2021 ms** (266 FHE + 1753 MPC) |
  | traffic | **2.24 MB**, of which 1.54 MB ring (69%), 613 messages |
  | comparisons | **4 693** — Conv0 3 456, Conv1 1 024, Hidden0 120, Hidden1 84, argmax 9 |
  | bootstraps | **0** |
  | exact vs plaintext | ✅ |

  **The DReLU count is 4 693, not the 8 044 this page has been extrapolating from**, and the
  difference is a scheduling choice rather than an error: Baru max-pools *before* the
  nonlinearity (they commute, since `relu∘rescale` is monotonic non-decreasing), so ReLU runs
  on 1 120 pooled values instead of 4 480 unpooled ones. 1 324 ReLU gates + 3 360 pool
  comparisons + 9 argmax = 4 693, a **1.71× reduction** — now confirmed at the architecture
  the comparison targets rather than argued from a smaller one.

  At **329 B per comparison** against the 323 B measured at the primitive level, the
  extrapolation *method* holds at full-network scale to 2%. What it needed was the right
  count.

  > ⚠️ Random weights, so **accuracy is chance and is not reported**. Cost follows from the
  > shapes, and exactness against the plaintext reference is asserted, so the numbers describe
  > a pipeline computing this architecture correctly. And `ℓ = 20` here against Falcon's 32,
  > so 1.54 MB is **not** directly comparable to their reported 0.74 MB without matching
  > widths — a like-for-like byte comparison still needs that, and this page should not claim
  > one until it has it.

  **The first attempt at this measured the wrong network.** It built Caffe's LeNet
  (20/50/500) rather than LeNet-5 (6/16/120/84) — 2 293 000 MACs against 281 640, 8.1× apart —
  and reported "the extrapolation understates the DReLU count by 1.89×". That was not a finding
  about the extrapolation; it was two architectures being compared. The doc's own "~282 k MACs"
  line identifies which one is meant, and the MAC count derived from LeNet-5's shapes matches
  it to the digit. Both nets are now reachable under names that say which they are
  (`CnnModel::lenet5`, `CnnModel::lenet_caffe`), `lenet_end_to_end_is_exact_and_costed` asserts
  the architecture layer by layer, and `the_two_lenets_are_not_the_same_network` — cheap, not
  ignored — pins that they differ by >5× in MACs and that **only Caffe's** second conv needs
  channel splitting at `n = 4096` (2 groups of 14; LeNet-5's 6 channels at 12×12 fit one
  ciphertext).

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
