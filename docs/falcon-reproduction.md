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

13. **`bench-falcon`**: the same LeNet-shaped network three ways — Falcon
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
