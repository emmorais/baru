# Baru — the experimental record

Every measured result in this repository, organized by what it is about rather
than by when it was found. The README keeps only the headline numbers; this is
the long form, including the corrections, the negative results, and the
benchmark bugs that produced confident wrong answers before they were fixed.

**How to read it.** Each experiment names the command that produces it. Numbers
were measured on the maintainer's machine (Apple silicon, release build) unless
stated otherwise; they are meant to be reproduced by re-running the command, not
quoted as absolutes. Where a table predates a later change to the code, that is
said in the text — a stale number is worse than no number.

**Contents**

- [Part I — MPC protocol engineering](#part-i--mpc-protocol-engineering)
- [Part II — FHE engines](#part-ii--fhe-engines)
- [Part III — The hybrid construction](#part-iii--the-hybrid-construction)
- [Part IV — Applications](#part-iv--applications)
- [Part V — The circuit compiler](#part-v--the-circuit-compiler)
- [Part VI — Building blocks, measured](#part-vi--building-blocks-measured)
- [Part VII — Negative, superseded and withdrawn results](#part-vii--negative-superseded-and-withdrawn-results)

---

## Part I — MPC protocol engineering

The recurring lesson of this part, stated once so the individual experiments
read faster: **the arithmetic was never the bottleneck; the round structure
was.** Four separate times, a segment-level measurement said "MPC is slow" and
a per-phase breakdown said "one unbatched loop inside it is slow."

### I.1 Multiplication without Beaver triples

`*` on shared values is **one ring element sent to one neighbour**, with no
dealer and nothing precomputed. With replicated sharing `x = x₀+x₁+x₂` (party
`i` holds `xᵢ` and `xᵢ₊₁`), each party forms three of the nine cross terms of
`x·y` locally:

```
zᵢ = xᵢyᵢ + xᵢyᵢ₊₁ + xᵢ₊₁yᵢ + αᵢ
```

Summed over the three parties these cover all nine terms exactly once, so
`Σzᵢ = (Σxᵢ)(Σyᵢ) = xy`. The `αᵢ` are **zero-sum correlated randomness**
`αᵢ = F(kᵢ, ctr) − F(kᵢ₋₁, ctr)` from pairwise PRF keys established once at
startup: each PRF output appears once with each sign, so the α cancel, while no
party can predict another's. They re-randomise each party's `zᵢ` so it leaks
nothing. Then party `i` sends `zᵢ` to party `i−1` (a directed `TaggedZBatch`,
not a broadcast), leaving each party holding `(zᵢ, zᵢ₊₁)` — a valid replicated
sharing of `a·b`. See `crypto/aby3_protocol.rs`.

The previous implementation used **Beaver triples dealt online by party 0** — 6
ring elements distributed plus 2 opened per multiplication, with party 0 a
dealer bottleneck. That was not moved offline but **eliminated**: there is no
triple to precompute, because the correlated randomness is regenerated locally
per operation from the PRF keys. This is why "why not just precompute the
triples?" has no work left in it — precomputation would optimise a phase that
no longer exists. The remaining preprocessing is one 32-byte key exchange per
party pair per session.

**Measured** (`baru bench-cnn`, results bit-exact throughout — only privacy and
cost differ across rows):

| D | MPC, Beaver triples | MPC, ABY3 reshare | wire bytes, Beaver | ABY3 reshare |
|---|---|---|---|---|
| 8 | 4538 ms | 2254 ms | 9.1 MB | 0.59 MB |
| 12 | 11789 ms | 6175 ms | 24.1 MB | 1.58 MB |
| 28 | 76892 ms | 40759 ms | 159.4 MB | **10.4 MB** |

At D=28 that is **15× less bandwidth** and ~1.9× faster MPC. ~117 MB of the old
159 MB was triple *distribution* alone; removing it is most of the win.

### I.2 `a2b`, rebuilt after a verified privacy break

> **This protocol was rebuilt after a verified privacy break.** The previous
> `a2b` masked its input with a value derived from `op_id` — a public lockstep
> counter — and then opened `a − r`. That mask carried no secret entropy, so the
> input was recoverable by any party or wire observer, and everything built on
> `a2b` (`lt`, `eq`, `le`, and hence ReLU, max-pool and argmax) leaked its
> activations. The "no extra communication per bit" property the old version
> advertised was a *symptom*: the carry chain was free precisely because every
> operand was already public. A correct protocol cannot have that property.

The replacement converts an arithmetic share of `x` into `nbits` shared bits
(LSB first) via ABY3 bit decomposition:

1. **Binary-share the three additive components — free.** A replicated sharing
   places a value in one additive component and zero elsewhere. Component `j` is
   held as `left` by party `j` and as `right` by party `j−1`, so exactly those
   two parties supply it and the third contributes nothing and learns nothing.
   No communication.
2. **Carry-save reduce 3 → 2.** In the arithmetic bit encoding,
   `x⊕y⊕z = x+y+z − 2(xy+xz+yz) + 4xyz` and `maj(x,y,z) = xy+xz+yz − 2xyz`, so
   the sum and carry vectors both fall out of the *same* four products (`xy`,
   `xz`, `yz`, `xyz`). Two rounds, not four.
3. **Add the remaining two** with a Kogge–Stone parallel-prefix adder.
   Combining `(g,p)` pairs uses `G = g ⊕ (p ∧ g')`; a group cannot both generate
   and propagate a carry, so those terms are mutually exclusive and the XOR is
   local addition. Only the ANDs communicate.

Depth is `2 + 1 + ⌈log₂ nbits⌉ + 1` ≈ 9 rounds at `nbits = 20`, against the old
(leaky) version's 1. Verified against plaintext binary expansion including
two's-complement negatives, since the MSB is what `lt` returns.

**Measured cost of correctness** (`baru bench-cnn`, bit-exact in both cases —
only privacy and cost differ):

| D | MPC, leaky `a2b` | MPC, correct `a2b` | wire bytes, leaky | correct |
|---|---|---|---|---|
| 8 | 1079 ms | 4538 ms | 70 KB | 9.1 MB |
| 12 | 2619 ms | 11789 ms | — | 24.1 MB |
| 28 | 17593 ms | 76892 ms | 1.2 MB | 159.4 MB |

Each `a2b` at `nbits = 20` needs 257 multiplications (80 carry-save + 20
generate/propagate + 138 prefix + 19 final). At D=28 that is 2375 `a2b` calls
(1352 ReLU + 1014 max-pool comparisons + 9 argmax) ≈ 610k multiplications. In
the first cut each drew a Beaver triple dealt online — ~117 MB of the observed
159 MB, since removed (I.1).

### I.3 Batching — the round count, not the multiplication count

After triples were gone the dominant cost was **round latency, not bandwidth**.
The ~610k multiplications were bundled tightly *within* each `a2b`, but the
`a2b` calls themselves ran one-per-element in a loop: 1352 ReLUs, each a
~9-round decomposition, is ~12 000 sequential rounds. So the fix was to batch
across elements, not to touch the comparison algorithm.

Two backend methods do it, both with a default that just loops (so every
non-batching backend is unaffected) and a `TransportBackend` override that runs
the whole set in one id block:

- **`a2b_batch`** decomposes many shares at once; every internal carry-save and
  prefix-adder round is flattened across all lanes, so a batch of any width
  costs the same ~9 rounds as one value.
- **`lt_batch`** compares many pairs at once (the MSB of each `a2b_batch` lane),
  which lets **`maxpool2d`** reduce all its windows in lockstep — `pool²−1`
  batched steps for the whole layer instead of one `max_reduce` per window.

**Measured** (`baru bench-cnn`, D=28 MPC, all rows bit-exact):

| stage | MPC time | messages | wire bytes |
|---|---|---|---|
| leaky `a2b` (insecure baseline) | 17.6 s | 114 790 | 1.2 MB |
| correct `a2b`, Beaver triples | 76.9 s | 114 790 | 159.4 MB |
| + ABY3 resharing (no triples) | 40.8 s | 29 211 | 10.4 MB |
| + batched ReLU | 20.7 s | 15 701 | 10.2 MB |
| + batched max-pool | 6.1 s | 5 591 | 10.0 MB |
| + batched scheme-switch import | 1.2 s | 147 | 10.0 MB |
| + tournament `argmax` | **1.15 s** | **88** | 10.0 MB |

Batching alone was 40.8 → 6.1 s (6.7×) and cut messages 5.2×. The endpoint is
the payoff: the **private, triple-free** pipeline ended up far **faster than the
leaky/insecure baseline** (17.6 s) it started from.

**The row that mattered is not a protocol change.** After every *comparison* had
been batched, a per-phase breakdown showed the comparisons were never the cost:
**19.1 s of the remaining 20.4 s was the scheme-switch `import`** —
reconstructing an opened layer into shares ran three blocking input-sharing
rounds *per activation*, so D=28's 5408-wide conv output cost **16 224
sequential round trips**, while the ReLU it was importing values for cost 0.6 s.
`share_input_batch` sends the same bytes with the same randomness in **three**
rounds: import 19.1 s → **8 ms**, end to end **21.8 s → 2.6 s (8.3×)**.

Worth remembering: *"the MPC segment dominates"* was true, but *"therefore the
MPC protocols are the bottleneck"* did not follow. The cost was in the plumbing
between the two representations — the part a switch-based construction crosses
most often, and the part neither the FHE nor the MPC literature owns. The
per-phase split that `bench-cnn` now prints exists so this is not repeatable.

**The last *dependent* reduction.** Batching needs independent operations, and
`gadgets::argmax` folded `nclasses−1` comparisons left to right, each waiting on
the previous winner. It did the least work of any phase and, under latency, took
the most time — **1.31 s of a 2.03 s MPC segment at RTT 20 ms**. Reassociating
the fold into a **tournament** (pair adjacent candidates; one `lt_batch` decides
a whole level, one `mul_batch` selects both value and index) makes it
`⌈log₂ n⌉` batched levels instead of `n−1` serial ones: **2.4× at RTT 20 ms**,
and `bench-cnn --img 8 --rtt-ms 20` as a whole 3.31 s → 2.45 s. Tie-breaking
(first maximum wins) is preserved because the bracket pairs adjacent elements in
index order — verified **exhaustively** over small alphabets, since a tie split
across sibling subtrees is precisely what random logits never generate.

One latent bug surfaced while batching and was fixed: the α re-randomiser
counter (`op_id + i`) could repeat across multiplication rounds once a batch
exceeded the id spacing. Harmless for correctness (α always cancels) but a
privacy leak (two products under the same α expose their difference); the
counter is now a collision-free `(op_id ‖ i)` encoding (`alpha_ctr`).

### I.4 The abstraction tax, and where it was not free

`baru bench-abstraction` measures whether Baru's layering costs anything against
hand-written protocol calls. Per-operation it does not. But composition had to
be free *in cost* too, and once it wasn't: every gadget was originally written
the way its definition reads — one `lt`+`mul` per element for `relu`, one `mul`
per weight for `dense`, a left fold for `max_reduce`. All correct, all secure,
and all serializing a layer into `O(n)` dependent rounds. On a 64-wide layer
over the real transport: **5–14× the messages and 9–45× the wall-clock at RTT
20 ms**, with byte counts within 5% and outputs bit-identical.

The gadgets are now defined over the *batched* primitives and match hand-written
batched code message for message; a test pins that, and the benchmark keeps the
per-element form as a control so the gap stays reproducible. **Only round counts
catch this class of defect** — per-op timing and byte counts both look fine.

### I.4b The garbled ReLU: the same defect, a fifth time

Once the FHE side got ~3× cheaper (III.5, III.6), the garbled ReLU became the
pipeline's largest single phase — 409 ms of a 736 ms `bench-cnn --img 28 --gc`
run over one conv layer's 5408 comparisons. `bench-gc-compare` showed the cost
was **linear in the item count and flat in RTT**, so the instinct was that this
one really is per-item compute rather than round structure.

Half of that was right. Attributing it against the real functions
(`crypto::garbled::where_the_garbled_relu_spends_its_time`):

| | |
|---|---|
| AND gates for the layer | 167 648 |
| label hash | **178 ns — a full SHA-256 over 24 bytes** |
| garbling the layer, measured | 169 ms (120 of it hashing) |
| evaluation hashing | 60 ms |
| building the 7.1 MB wire payload | 2 ms |

Garbling really is compute, and the label hash is 44% of it — see below. But
that left **~230 ms unattributed**, and it turned out to be the round structure
after all, hiding in the conversion back to arithmetic.

**`gc_protocol::y2a` re-shared the garbled output one bit at a time.** It called
`share_input` per (item, bit) — at `d=28` with width 12 that is
`2 × 5408 × 12 = 129 792` calls for a single ReLU — while
`aby3_protocol::share_input_batch` had existed the whole time. Replacing the two
loops with two batched calls, for bit-identical output:

| `bench-cnn --img 28 --gc --precompute` | before | after |
|---|---|---|
| ReLU phase | 409 ms | **323 ms** |
| total | 736 ms | **628 ms** |
| **MPC messages** | **319 139** | **47** |
| MPC bytes | 21.8 MB | 18.0 MB |

**A 6 790× reduction in messages, and the bytes went *down* too** — batching
removes per-message framing rather than adding any. On the trained medical model
the same change is 1086 → 1000 ms with traffic falling from 379 725 messages to
61.

This is the **fifth** instance of one defect. The scheme-switch import,
`gadgets::relu`, `bench-hybrid-nn` and `bench-deep-nn` all had it, and in every
case the batched primitive already existed. Pinned by
`garbled_nonlinearity_does_not_scale_with_layer_width`, which asserts both a
ratio (constant in layer width) and an absolute ceiling, since a ratio test alone
passes if both sides regress together.

**And a sixth, found by looking instead of waiting.** Four previous fixes had not
prevented the fifth, because each was repaired where it surfaced rather than
searched for elsewhere. So after `y2a` the obvious move was to grep every
interactive primitive (`share_input`, `share_secret`, `mul`, `lt`, `a2b`)
appearing inside a loop over a layer, and check each against its `_batch` form.
Most hits are deliberate per-element *controls* — `NaiveMatVec`,
`bench-nn`'s naive path, `bench-abstraction`'s control — which exist precisely to
keep the gap reproducible. One was not: **`bench-train`**, both trainers, issuing
one `mul` per weight and one `arith_shr` (hence one `a2b`) per gradient.

Batched — `gadgets::dense` for the matvecs, one `mul_batch` per outer product,
one new `gadgets::arith_shr_batch` per learning rate — with **bit-identical
models and identical loss curves**:

| `baru bench-train` | messages before | after |
|---|---|---|
| linear (12 samples, 6 features, 3 classes, 20 epochs) | 62 683 | **3 643** |
| MLP (`--hidden 8`, backprop through ReLU) | 199 655 | **7 271** |

The MLP forward pass also stopped recomputing its own comparison: the backward
pass needs `step(z1)`, which *is* the ReLU's mask, so it is now computed once and
reused rather than derived twice. Pinned by
`training_round_structure_is_constant_in_model_size`.

**What was left really was compute, and it was the wrong primitive.**
`garbled::hash` was SHA-256; half-gates call it four times per AND to garble and
twice to evaluate, so it ran 1.0 M times for one layer. It is now **TMMO over a
fixed-key AES permutation** — `H(x, i) = π(π(x) ⊕ i) ⊕ π(x)` — at **5 ns against
178**, taking garbling from 169 ms to 10 ms and the ReLU phase from 323 ms to
**88 ms**.

**The first attempt made it three times slower, and that is the transferable
part.** On a default `cargo build` the RustCrypto `aes` crate falls back to
constant-time *bitsliced software*: the hash measured **525 ns**, worse than the
SHA-256 it replaced, because `sha2` is hardware-accelerated on this machine and
`aes` was not. Enabling the ARMv8 crypto extensions
(`.cargo/config.toml`, scoped to `aarch64-apple-darwin`) is what produced the
36× — so *"use the faster primitive"* was not the finding; **which build gets the
hardware** was. The measurement test now warns above 50 ns/hash so a build that
has silently lost the hardware path is visible rather than merely slow.

Half-gates with free-XOR needs a **tweakable circular correlation robust** hash,
which is stronger than collision resistance: the garbler hashes labels differing
by the global offset `R` and reuses `R` across every gate. TMMO is proven to
achieve it (Guo–Katz–Wang–Yu, S&P 2020); plain MMO `π(x) ⊕ x` is not tweakable
and `π(x ⊕ i) ⊕ x` leaks under free-XOR. **The tests here are evidence of
correctness, not of security** — a wrong gate hash computes correct outputs — so
this one is flagged for review against the paper in a way the NTT rewrite (II.2)
never needed to be.

**One hypothesis died in the measurement**, kept so it is not re-formed: the wire
payload is built 16 bytes at a time into a `Vec` with no reserved capacity, which
looks like a textbook allocation problem. It costs 2 ms. The obvious-looking
defect was not the cost — the round structure and the cryptographic primitive
were.

### I.5 Precompiled gadgets: batching a whole matvec

`src/precompile/` is a registry of named Rust gadgets per (operation, backend
family) with an explicit **champion** (the dispatch target, set by a maintainer
— see the module doc for why this is a reviewed decision rather than an
automated market), plus a Baru builtin (`matvec`) that calls the champion
directly instead of a Baru program re-deriving the loop.

The first gadget, `mpc_matvec::MpcMatVec`, ships `NaiveMatVec` (per-element
loop) and `BatchedMatVec`, which flattens the entire matrix's multiplications
into one `Backend::mul_batch` call — one batched ABY3 reshare collapsing the
resharing from O(n) rounds to one, regardless of shape.

```sh
baru bench-nn --workload matvec --backends mpc --competition
```

| shape | naive | batched | speedup | messages (naive → batched) |
|---|---|---|---|---|
| matvec(8×8) | 178.8 ms | 16.4 ms | 10.9× | 944 → 308 |
| matvec(64×32) | 5772.2 ms | 56.8 ms | 101.6× | 28 992 → 8 516 |
| matvec(128×64) | 25.3 s | 127.2 ms | **199×** | 115 328 → 33 412 |

Both agree with the plaintext reference at every shape, permanently — a
precompile that disagrees with plaintext is a bug independent of how fast it is.

### I.6 The network cost model, validated against a real socket

`baru bench-net` checks the cost formula (`rounds × latency + bytes / bandwidth`)
against a real TCP socket rather than trusting it. `src/bench/comm.rs` wraps the
actual transport in counting/latency decorators that observe traffic without
touching protocol code, so a `--rtt-ms`/`--bandwidth-mbps` profile can "move"
parties from one rack to different continents without deploying anything.

Two harness bugs were found and fixed by building this, both of which had been
silently distorting results:

- **Message reordering.** Per-message latency timers could reorder back-to-back
  messages, corrupting the protocol. The model now preserves per-connection FIFO
  like a real TCP stream, with a regression test.
- **Bandwidth undercharging.** The old per-message bandwidth model did not
  serialize bulk transfers cumulatively, so it undercharged exactly the
  byte-heavy phases the crossover experiments are about. Fixed, with a
  regression test pinning the corrected behaviour.

Also worth recording: the in-process `Aby3Simulator` **is not the protocol**
(it was called `Aby3Backend` until the naming was fixed on 2026-07-30). Its share multiplication reconstructs and reshares locally — free
and insecure — and it understated the pure-MPC row by ~2.7× (6.7 ms simulated
vs 18 ms real-protocol). Every MPC number in this document is the real wire
protocol (`TransportBackend`, three party threads exchanging serialized
messages).

---

## Part II — FHE engines

### II.1 Exact ring products over a power-of-two modulus

The polynomial multiply is the innermost cost of every FHE scheme here, and it
runs on a **multi-modular NTT** rather than an O(n²) schoolbook loop. A textbook
NTT needs a modulus with a primitive 2n-th root of unity (`q ≡ 1 mod 2n`,
prime) — but `q = 2^100` is a power of two *deliberately*, because that is what
makes the `t/q` rescale a rounded shift and the ring arithmetic native `u128`. A
direct NTT is therefore impossible.

The fix keeps the scheme and changes only the *computation*: what the schemes
need is the **exact integer** negacyclic product, so it is computed modulo four
~63-bit NTT-friendly primes (O(n log n) each) and CRT-reconstructed exactly
(Garner), then the unchanged rescale is applied. The BFV entry point shares
forward transforms across the tensor (4 forward + 3 inverse NTTs per prime, not
12). Decryption uses **Horner** (`c₀ + s·(c₁ + s·c₂)`), one ring multiply per
key power instead of six.

Correctness is not argued but tested: differentially against the schoolbook
implementation it replaced, at full ±2⁹⁹ coefficient width and n=4096.

| operation | before | after | speedup |
|---|---|---|---|
| BFV multiply (n=4096) | 515 ms | **45 ms** | 11× |
| BFV decrypt | 130 ms | **37 ms** | 3.5× |

### II.2 NTL as an oracle — and the Shoup butterfly that made it one

The multi-modular NTT is ~200 lines of hand-written modular arithmetic on the
hottest path in the repo, validated against a schoolbook reference too slow to
run at the sizes that matter. Both facts warranted a second opinion, so the same
operation is also implemented against **NTL** (arbitrary-precision `ZZX`
multiplication over GMP) through a C++ shim (`cpp/ntl_polymul.cpp`) wrapped in
Rust. NTL picks its own strategy by operand size — plain, Karatsuba, or
homomorphic imaging — so it is an independent answer to the same question.

That buys a much stronger correctness argument: `cargo test --features ntl`
checks the hand-written NTT against NTL at **n = 4096 with ±2⁹⁹ coefficients**,
and at the widest ±2¹¹⁹ / n = 2048 corner the CRT modulus can represent, where
the O(n²) oracle is impractical. They agree exactly.

It first produced a result we did not want, and then — after acting on it — the
opposite one. `baru bench-polymul --schoolbook-max 4096` (per product, 20 reps,
marshalling included, NTL 11.5.1 / GMP 6.3.0):

| n | schoolbook | multi-modular NTT | NTL+GMP | NTT vs schoolbook | **NTT vs NTL** |
|---|---|---|---|---|---|
| 256 | 0.280 ms | **0.101 ms** | 0.188 ms | 2.8× | 1.9× |
| 512 | 1.125 ms | **0.203 ms** | 0.375 ms | 5.5× | 1.8× |
| 1024 | 4.500 ms | **0.422 ms** | 0.754 ms | 10.7× | 1.8× |
| 2048 | 17.979 ms | **0.865 ms** | 1.555 ms | 20.8× | 1.8× |
| 4096 | 71.859 ms | **1.771 ms** | 3.146 ms | 40.6× | 1.8× |

**The hand-written NTT is ~1.8× faster than NTL at every size.** It did not
start that way: it was 8.232 ms at n=4096 and *lost* to NTL by 2.6×. Profiling
NTL showed exactly where the gap was, and closing it gave **4.65× on our own
transform**.

NTL is **not a different algorithm**, which is worth stating precisely because
it changes what the gap means. Probing the linked library directly
(`NTL_SP_NBITS = 60`): a single-FFT-prime `zz_pX` multiply at n=4096 takes
**0.151 ms**, and the 99-bit `ZZX` multiply takes 1.763 ms — a ratio of 11.7
where the coefficient bound needs only `(2·99 + 12 + 1)/60 = 3.5` primes. So NTL
does exactly what `ntt.rs` does — multi-modular FFT over ~60-bit primes plus CRT
— and spends the rest on GMP bignum↔residue conversion. Two consequences:

1. **Our transforms were the slow part, by ~13×.** Four primes × (2 forward + 1
   inverse) is 12 transforms; NTL ran those in ~0.60 ms, we took 8.2 ms. The
   cause was visible in the butterfly: **two** `mulmod`s, each a `u128 %` (a
   `__umodti3` libcall on aarch64), where NTL uses a precomputed root table and
   Shoup-preconditioned multiplication with no division at all.
2. **Our conversion was the fast part.** Fixed-width `i128`/`I256` coefficients
   make `reduce_mod` and Garner CRT cheap, where NTL pays generic bignum costs.
   That is why NTL won by only 2.5× and not by 13×.

So a Shoup butterfly should *beat* NTL rather than merely match it — keeping the
cheap fixed-width conversion **and** getting the fast transform. That work is
done and the prediction held. Two changes, both licensed by the primes being
`< 2^63`:

1. **Precomputed twiddle tables.** The old inner loop spent one `mulmod` on the
   butterfly and a second walking `wn` forward. Twiddles are fixed per level, so
   they are built once per plan.
2. **Shoup-preconditioned multiplication.** For a fixed multiplier `w`, cache
   `w' = ⌊w·2^64/p⌋`; then `w·x mod p` is a multiply-high, a multiply, a subtract
   and one conditional subtract — **no division at all**. The inverse
   transform's `n^{-1}` scale and `ψ^{-i}` untwist were two `mulmod`s and are now
   one fused Shoup multiply.

`8.232 → 1.771 ms` at n=4096 (**4.65×**), and the schemes followed: a BFV ct×ct
multiply went **20.7 → 5.4 ms**. Two guard tests pin the preconditions —
`primes_admit_shoup` (the `p < 2^63` bound the prime search must keep supplying,
or every transform silently corrupts) and `shoup_multiplication_matches_mulmod`.

**So NTL is no longer a performance option — it is an oracle.**
`poly::use_fastest()` returns `Ntt` unconditionally, and the C++ toolchain buys
correctness assurance rather than speed. That is a better outcome than adopting
NTL would have been: the default build stays pure Rust *and* is the fast one.
The NTL differential tests are what made the rewrite safe to do. The path stays
wired in as a **runtime-selectable backend** (`src/fhe/poly.rs`) so one binary
can measure either, and `tests/fhe_poly_backends.rs` checks that every scheme
encrypts→multiplies→decrypts identically under both at deployed parameters.

### II.3 The minimal-FHE inner product, and which scheme to run it on

The construction the whole hybrid rests on is deliberately the *cheapest* FHE
primitive: **one ciphertext×ciphertext multiplication, no relinearization keys,
no bootstrapping, no evaluation keys at all**, decrypting the degree-2 product
directly under the squared key. Two tricks make a whole matvec almost free:

- **Reversed-coefficient packing.** Input in coefficients `0..d` of one message
  polynomial, every weight row *reversed* at stride `d` in another — the whole
  matvec is ONE ct×ct multiplication, row `r`'s inner product landing in
  coefficient `r·d + d−1` (cross-term-free while `(rows+1)·d ≤ n`; asserted).
  Weights *and* input are encrypted: fully private linear algebra.
- **No relinearization.** The only consumer of the product is a *decryption* —
  the masked decryption of an FHE→MPC scheme switch — so the degree-2 key never
  needs switching back down.
- **Poly-level scheme switch.** One masked decryption converts the *entire*
  output vector to ABY3 shares simultaneously.

`baru bench-fhe-innerprod` compares the secure schemes on exactly this
primitive. Both use the same schoolbook-Rust polynomial style and the same
packing, so any difference is the encoding:

```text
BFV:  c(s) = Δ·m + e      message in the HIGH bits  ⇒ tensor needs a t/q rescale
BGV:  c(s) = m + t·e      message in the LOW bits   ⇒ tensor is reduced mod q, full stop
```

BGV's multiply is therefore BFV's multiply *minus* the rescale, and its decrypt
is a mask rather than a rounded divide. Measured
(`baru bench-fhe-innerprod --poly all --reps 15`, packed 8×4 matvec, both
operands encrypted, `n = 4096, q = 2^100, t = 2^20` for both):

| scheme | multiply (NTT) | multiply (NTL) | ct bytes |
|---|---|---|---|
| BFV | 5.4 ms | 11.2 ms | 131072 |
| **BGV** | **5.3 ms** | **11.1 ms** | 131072 |

**BGV is indistinguishable from BFV.** Repeated runs put the multiply gap at
+0.5%, +0.5%, −0.1% — inside the 1–2% run-to-run spread. The rescale is
genuinely removed; it is just `3n = 12288` shifts against 28 size-4096
transforms. *The encoding is not where the cost is*, and `bgv.rs` is the
evidence for that rather than a speedup.

BGV's noise behaves differently from BFV's in one way worth having: the dominant
term is `t·e·e'`, so the budget is **message-independent**. Measured at n=4096,
`|m·m' + t·E| ≈ 2^60.2` against a `q/2 = 2^99` bound — **38.8 bits of slack** —
and sweeping the plaintext magnitude 8× moves it by 0.1 bits. (The worst-case
analytic bound is 2^79, so the ternary convolutions concentrate about 19 bits
better than the bound assumes.) BFV's post-multiply budget is measured the same
way in `fhe::bfv`'s `measures_post_multiply_noise_budget`.

So the practical answer to "which secure scheme for the depth-1 product" is that
BFV and BGV are interchangeable on cost, and the lever that actually moves FHE
is the ring arithmetic (II.2), not the scheme.

### II.4 CKKS comparisons: approximate, not impossible

`RealFheBackend`'s doc comment says comparisons are "impossible" on CKKS's
approximate ciphertexts — true for *exact* comparison, but the CKKS/HEAAN
literature has an established answer for *approximate* comparison, and since
other groups build private-inference systems on exactly this technique, treating
it as out of scope would leave CKKS's real performance profile unmeasured rather
than characterized. So: don't compare — approximate `sign(x)` with a polynomial,
and comparison/ReLU/max fall out arithmetically (`relu(x) ≈ x·(sign(x)+1)/2`).

**The polynomial.** `f(x) = x·(3 − x²)/2` — the classical Newton–Schulz
iteration for the scalar sign function, and, not coincidentally, the exact "f"
polynomial family Cheon, Kim, Kim and Song introduced for *homomorphic*
comparison in [Numerical Method for Comparison on Homomorphically Encrypted
Numbers](https://eprint.iacr.org/2019/417) (Asiacrypt 2019). `f` has fixed
points at `−1, 0, 1`; `0` is repelling and `±1` are superattracting, so
iterating pushes `x ∈ (−√3, √3)` towards `sign(x)` — slowly near 0, then
quadratically fast. The follow-up work ([Efficient Homomorphic Comparison
Methods with Optimal Complexity](https://eprint.iacr.org/2019/417), Asiacrypt
2020; [Minimax Approximation of Sign Function by Composite
Polynomial](https://eprint.iacr.org/2020/834)) composes `f` with a second family
`g` (fast near ±1) via dynamic programming for asymptotically optimal depth.
`src/fhe/ckks_compare.rs` implements only the foundational `f`-iteration — correct
and independently verified, but costlier in depth for a given precision than the
state of the art. The `f`+`g` composite is future work.

**Why it needs its own CKKS context.** `CkksBackend` (the shallow path used for
`matvec`/`dot`) deliberately never calls SEAL's `rescale_to_next` — it keeps
every ciphertext at the top modulus level and tracks scale via a `depth`
counter, exactly right for depth-≤2 linear workloads but meaning each
multiplication *compounds* the required modulus budget instead of resetting it.
It cannot survive the sequential multiplications a sign approximation needs.
`CkksCompareBackend` is an independent context that does what real CKKS
deployments do: a genuine `rescale_to_next` after every multiplication, so depth
cost is linear in the number of primes rather than exponential — at the cost of
a much bigger ring dimension. It picks the smallest dimension (8192 / 16384 /
32768) that fits the requested level count using SEAL's own
HomomorphicEncryption.org security table, looked up at run time rather than
guessed.

**Measured** (`baru bench-ckks-compare --iterations 1,2,3`, one scalar
`relu_approx` per row, bound = 5.0):

| iterations | levels | ring degree | keygen | relu (per op) | ciphertext | error vs exact ReLU | error vs plain approx |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 16384 | 5.3 s | 1.7 s | 1.8 MB | 0.39 | 3.4×10⁻⁸ |
| 2 | 9 | 32768 | 18.9 s | 8.7 s | 5.2 MB | 0.19 | 2.3×10⁻⁸ |
| 3 | 12 | 32768 | 30.0 s | 17.2 s | 6.8 MB | 0.17 | 1.9×10⁻⁸ |

Two error columns on purpose: "vs plain approx" (the same Newton iteration in
plain `f64`) stays at ~2×10⁻⁸ across the board — proof that real SEAL ciphertext
noise is *not* the dominant error source, the polynomial's own approximation
error is. And that error is large at these iteration counts specifically because
the sample points include inputs close to the bound's boundary (the documented
slow-near-zero regime) — the `f`-only simplification's known weakness. For
comparison, MPC's `relu` costs **~10 ms per element** end to end. That is the
"totally different performance profile" made concrete: three to four orders of
magnitude slower per comparison than MPC, before reaching useful precision, and
paid in a separate, deeper, dimension-inflated context the cheap linear path
never has to touch.

### II.5 Widening the field: 256-bit coefficients for LowGear

LowGear (VI.4) originally ran at `n = 4096, q = 2^100`, which the flooding
budget `t²·n²·3·2^(σ−1) < q/2` pins to a **~17-bit field**. Enough to multiply,
not enough to *compare*: masked bit-decomposition needs `bit_width + KAPPA` ≈ 58
bits, so the engine refused comparison-shaped programs and `bench-layer`'s SHE
row had to be **composed** from two measurements rather than run.

Closing that meant widening the arithmetic. `q = 2^190` needs 256-bit
coefficients and a 512-bit product accumulator:

- **`src/fhe/bigint.rs`** — `IntN<const L: usize>`, replacing the bespoke
  `I256`. The narrow path measured **1.770 ms → 1.770 ms** at n=4096: zero
  regression, because the const-generic loops unroll as the hand-written `u128`
  halves did.
- **`src/fhe/ntt.rs`** — a **wide profile**: `MAX_N` 4096 → 8192, 7 primes (CRT
  modulus ~2^441) against the narrow 4. One prime table serves both, since
  `p ≡ 1 mod 16384` implies `p ≡ 1 mod 8192`.
- **`src/fhe/bgv_wide.rs`** — BGV at 256-bit coefficients, carrying only what
  the OLE needs (ct×*plaintext*, so no tensor, no relinearization, no key power
  above one).

LowGear now runs at `n = 8192, q = 2^190, t ≈ 2^61` and **computes ReLU
correctly**. Two bugs surfaced getting there, both ours: the flooding mask was
`log2_bound + 1` bits instead of `log2_bound`, doubling the noise past `q/2` and
silently invalidating every triple; and the parameter search accepted a set with
**0.29 bits** of slack. `validate` now demands 2 bits, and the
deployed-parameter triple test is **no longer `#[ignore]`d** — it was ignored,
which is why a wrong answer three layers up was the first symptom.

---

## Part III — The hybrid construction

### III.1 Scheme switching, both directions

MPC does comparisons/ReLU in ~10 ms but pays communication for every
multiplication; CKKS does a whole linear layer in one SIMD ciphertext op but
pays *seconds* per approximate comparison (II.4). Scheme switching turns that
asymmetry from a limitation into a design space: convert values **between MPC
shares and FHE ciphertexts, mid-computation, without ever exposing them**, and
run each phase on whichever cryptosystem is cheapest there. This is the
architecture of Gazelle (Juvekar et al., USENIX Security 2018) and its
successors; the cross-scheme idea in general is
[CHIMERA](https://eprint.iacr.org/2018/758), and the trust model is threshold
FHE ([Asharov et al., EUROCRYPT 2012](https://eprint.iacr.org/2011/613)).

Both directions are one short protocol each (`src/hybrid/mod.rs`):

- **MPC → FHE** (`shares_to_fhe`): each party encrypts its *own additive share
  component* under the FHE public key; the evaluator homomorphically sums them.
  The sum of components *is* the secret, so the result is an encryption of it —
  and nobody ever reconstructed it. Needs only homomorphic addition (no
  relin/Galois keys), so it works on every backend here.
- **FHE → MPC** (`fhe_to_shares`): mask-and-open. Each party contributes an
  encrypted uniform mask; only the *masked* value (uniformly random, revealing
  nothing) is ever decrypted, and the masks turn it into additive shares. This is
  exactly the data flow of a threshold decryption; the in-process implementation
  stands a single client key in for the distributed key operation — documented,
  not hidden.
- **Re-encryption instead of bootstrapping** (`mpc_reencrypt`): compose the two.
  FHE → shares → fresh FHE ciphertext resets encryption noise and, for levelled
  CKKS, restores the full depth budget, for the price of one interactive round
  instead of a bootstrapping circuit.
  `tests/hybrid.rs::test_real_ckks_share_conversion_resets_depth` proves it on
  real SEAL: multiply a ciphertext (depth 1), re-encrypt, depth is 0 again and
  the value is intact.

**Generic by construction**: conversions are written against two small
capability traits, so new FHE backends compose with new MPC backends without
touching the module — `hybrid::FheAdapter` on the FHE side (`IntFhe` blankets
every integer `FheBackend`; `RealFhe` adapts any `RealFheBackend`), and
`Backend::export_additive`/`import_additive` on the MPC side.

One documented caveat: a converted ciphertext holds the value's *residue class*
mod `2^nbits` (share-sum carries may sit above it), so `+`/`−`/`×` keep working
but FHE-side comparisons on converted values would see the carry — convert back
to MPC for those, which is the point.

### III.2 The headline: deep private inference with ZERO bootstrapping

`baru bench-deep-nn` — the experimental core of the paper claim.

**The claim.** In the hybrid pipeline, the scheme switch that computes each ReLU
*also refreshes the ciphertext*: the switch-back re-encrypts from scratch
(`shares_to_fhe` → `key_pow = 1`, fresh noise), so **the FHE noise budget resets
at every layer boundary**. The MPC round trip *is* an interactive functional
bootstrap. Therefore a network of **arbitrary depth** runs with **zero FHE
bootstraps**, at cost **linear in depth**, using an FHE engine that supports
only a *single* multiplication. A leveled-FHE competitor (CKKS/BGV) must
bootstrap every few layers or exhaust its modulus chain; we never bootstrap.

Two invariants make the chain exact at any depth, both verified empirically:

1. **Noise never accumulates.** Each layer's FHE ciphertext is depth-1 fresh, so
   BFV's depth-1 budget is the *only* budget that ever matters, independent of L.
2. **Values never overflow.** Layers are fixed-point: the nonlinearity is the
   fused `a' = ReLU((W·a + β≪s) ≫ s)`, which rescales by `s` bits every layer and
   so keeps activation magnitude bounded regardless of L. The rescale needs only
   a *logical* shift — correct because it is applied solely on the non-negative
   (kept) branch — computed from the same single `a2b` that yields the sign bit.

**Measured** (deep fully-connected ReLU net, width 16, weights *and* activations
encrypted, real ABY3 protocol, BFV at `n = 4096, q = 2^100`, exact integer
fixed-point):

```sh
baru bench-deep-nn --depths 2,4,8,16,32 --width 16
```

| depth | exact? | total | fhe | switch | mpc | msgs | bytes | our bootstraps | CKKS-competitor bootstraps |
|---|---|---|---|---|---|---|---|---|---|
| 2 | ✓ | 145 ms | 20 ms | 59 ms | 54 ms | 38 | 121 KB | **0** | 0 |
| 4 | ✓ | 280 ms | 40 ms | 119 ms | 112 ms | 76 | 241 KB | **0** | 1 |
| 8 | ✓ | 539 ms | 81 ms | 239 ms | 210 ms | 152 | 483 KB | **0** | 3 |
| 16 | ✓ | 1050 ms | 158 ms | 480 ms | 403 ms | 304 | 966 KB | **0** | 7 |
| 32 | ✓ | 2119 ms | 333 ms | 958 ms | 820 ms | 608 | 1.93 MB | **0** | 15 |

Every row is **bit-exact** against the plaintext integer network — if noise
accumulated, deep outputs would diverge, which is the empirical proof of the
noise-reset invariant. Total time, messages and bytes are all **linear in depth**
(2→32 is 16×; total 145 ms → 2119 ms is 14.6×). Our bootstrap count is a flat
**0**; the competitor's grows ∝ depth.

The competitor column is not fabricated: it uses *our own measured* per-ReLU
level cost from `bench-ckks-compare` (`levels_for_relu` = 9 levels at 2
iterations, + 1 for the linear map = 10 levels/layer), with a stated
conservative 20-levels-per-bootstrap assumption — `⌈(10·L − 20)/20⌉` → 0, 1, 3,
7, 15. The *shape* (∝ depth vs flat 0) is the point, and each CKKS bootstrap
costs seconds in a degree-32768 context (II.4). Everything is `--json`-emittable
and reproducible; nothing is hand-entered.

Also load-bearing: **per-layer independent MPC sessions** whose only carried
state is the scheme-switch ciphertext — the interactive bootstrap made concrete.

### III.3 One private dense layer, five ways

`baru bench-hybrid-nn` — the same layer (`matvec(8×2) + bias + ReLU`, inputs
spanning both signs) end to end under every strategy, verified against the
plaintext reference. Every MPC segment — the pure strategy, the hybrids' ReLU,
and both scheme-switch directions — is the real ABY3 wire protocol, and the
network model charges the FHE upload/download and conversion round trips too.

At `rtt = 0` (all parties co-located; one run, `baru bench-hybrid-nn --in 8
--out 2`):

| strategy | total | linear | MPC segment | wire (MPC) | correct? |
|---|---|---|---|---|---|
| **mpc** (pure ABY3, weights *also* private) | **25 ms** | 5.0 ms | relu 13.0 ms | 120 msgs / 17.9 KB | ✓ exact |
| **hybrid-bfv** (BFV linear → ABY3 ReLU, weights *also* encrypted) | **79 ms** | 5.6 ms (one ct×ct) | relu 27.7 ms (+4.0 ms switch) | 15 msgs / 8.9 KB | ✓ exact |
| **hybrid-ckks** (CKKS SIMD linear → ABY3 ReLU) | 918 ms | 81 ms | relu 23.0 ms (+82 ms switch) | 13 msgs / 14.8 KB | ✓ exact |
| **ckks** (pure deep CKKS, polynomial ReLU) | 15.2 s | 412 ms | — | — | ✗ off by 282 ¹ |
| **tfhe** (pure TFHE, PBS ReLU) | 59.2 s | 57.0 s | — | — | ✓ exact |
| **hybrid-tfhe** (TFHE linear → ABY3 ReLU) | 60.0 s | 54.4 s | relu 22.8 ms (+4.7 s switch) | 13 msgs / 14.8 KB | ✓ exact |

Note how much of each pure-FHE row is *keygen*: 722 ms of hybrid-ckks's 918 ms
and 13.6 s of pure CKKS's 15.2 s. The deep polynomial-ReLU context is expensive
before the layer even starts, which is itself the argument for not needing one.

¹ The pure-CKKS output matches its own plain-`f64` Newton-approximation
reference (the crypto is correct) but deviates 282 from the exact ReLU on the
negative pre-activation (−818 at bound 5820 — the documented slow-near-zero
regime at 2 iterations).

Latency sweep (`--rtt-ms`), layer 8×2:

| network | mpc | hybrid-bfv | hybrid-ckks |
|---|---|---|---|
| rtt = 0 | 25 ms | 79 ms | 918 ms |
| rtt = 20 ms | 206 ms | 309 ms | 1.18 s |
| rtt = 100 ms | 825 ms | 1.14 s | 1.99 s |

The hybrid's *relative* penalty shrinks as the link gets worse (3.2× → 1.5× →
1.4×) because both sides pay the same ReLU rounds and the hybrid's extra
conversion RTTs stop mattering next to them.

What the numbers say:

- **Under latency, MPC cost is rounds × RTT — and ABY3's batched matvec is
  remarkably round-cheap.** At 100 ms RTT the whole fully-private layer is ~8
  round trips (825 ms), of which the batched matvec is ~1 regardless of shape;
  it is the ReLU's ~9-round decomposition that stacks. Precisely the structure an
  in-process simulator cannot show.
- **The hybrids pay the same ReLU rounds plus ~1.5 conversion RTTs**, so scheme
  switching does not beat pure MPC on *latency* at these shapes. Its win is
  elsewhere: deployment shape (the client sends ciphertexts and leaves; only the
  tiny nonlinear segment needs live parties) and wire footprint — **15 messages
  against 9276** at 256→8, because the bytes-heavy linear phase lives in FHE
  where it crosses the wire once as ciphertext.
- **Scheme switching rescues CKKS entirely** (exact ReLU in milliseconds vs an
  approximate one costing 8.5 s/element in a degree-32768 context).
- **hybrid-TFHE is an honest negative result**: TFHE's linear phase (67.5 s)
  dominates and the conversion costs more than its native PBS ReLU. Switching
  pays only when it replaces *many* FHE ops, not one comparison.

### III.4 Does faster ring arithmetic bring FHE closer to MPC? No

`baru bench-hybrid-nn --poly all` runs the whole private-layer pipeline with
only the polynomial arithmetic swapped. Two measurement traps had to be cleared
first, and both are worth recording because each produced a confident wrong
answer.

**Trap 1: process warm-up.** Running both backends in one process gave the
polynomial-free `mpc` row a spurious 1.21× "speedup". Every number below is
medians of 8, each backend in its **own process, alternating order**, with `mpc`
as a control that must read 1.00×.

**Trap 2 (the expensive one): the benchmark's ReLU was not batched.**
`bench-hybrid-nn` applied a per-element `mpc_relu` in a `.map()` loop, costing
`O(out_dim)` communication rounds — while `gadgets::relu` had done one
`lt_batch` + one `mul_batch` for the whole slice all along, and its own docstring
recorded that the per-element form costs "5× the messages and 9× the
wall-clock". The benchmark never adopted its own gadget. Fixing the four call
sites, for **bit-identical output**:

| | ReLU, per-element | ReLU, batched | speedup |
|---|---|---|---|
| pure MPC, out=8 | 95.7 ms | 14.3 ms | 6.7× |
| pure MPC, out=32 | 451.8 ms | 19.0 ms | 23.8× |
| pure MPC, out=64 | 802.4 ms | 23.3 ms | **34.4×** |

The batched ReLU is nearly **flat in layer width** (14.3 → 23.3 ms for 8× the
outputs) because independent comparisons cost a constant number of rounds. The
per-element version was linear.

**The same defect was in `bench-deep-nn`**, whose `mpc_relu_rescale_layer` ran
`a2b`/`b2a`/`mul` inside a per-element `.map`, and in **both benchmarks' FHE→MPC
import**, which called `share_secret` three times per output. Fixing all of them
— `a2b_batch` + one `mul_batch` for the nonlinearity (`b2a` is local, so it
stays per element), and `share_secret_batch` per owner for the import:

| | messages before | after | MPC segment before | after |
|---|---|---|---|---|
| `bench-deep-nn`, depth 4, width 16 | 1412 | **76** | 1273 ms | **102 ms** |
| `bench-deep-nn`, depth 4, width 32 | 2820 | **76** | 2372 ms | **100 ms** |
| `bench-hybrid-nn`, out=8 | 145 | **15** | 158 ms | **21 ms** |
| `bench-hybrid-nn`, out=64 | 1153 | **15** | 1186 ms | **30 ms** |

Bit-exact throughout, 0 bootstraps unchanged. The number to look at is not the
speedup but that **message count is now constant in width** — 76 at both
deep-nn widths, 15 at every hybrid-nn width. `share_secret_batch` had existed on
`TransportBackend` the whole time; `cnn.rs`, `compiled_nn.rs`, `abstraction.rs`
and `mixed_layer.rs` all used it. These two benchmarks simply never adopted it —
the same story as `gadgets::relu`, and four instances of one defect.

With both traps cleared (one layer 256→8, medians of 8, separate processes,
alternating order):

| strategy | poly | total | FHE | MPC seg | msgs |
|---|---|---|---|---|---|
| mpc (control) | ntt | 34.1 ms | 6.1 ms | 13.9 ms | 9276 |
| **hybrid-bfv** | **ntt** | **71.5 ms** | 45.9 ms | 25.4 ms | **15** |
| hybrid-bfv | ntl | 100.1 ms | 75.4 ms | 24.4 ms | 15 |

- **NTL is now the *slower* backend** (since the Shoup rewrite, II.2), which is
  the expected direction; before it, NTL sped the FHE work up 2.41×, reproducing
  `bench-polymul`'s isolated ratio through a different code path. The `mpc`
  control reads 1.00× either way, as it must.
- **The gap to pure MPC did not close.** The secure hybrid is **2.09× pure
  MPC**. That figure moved a lot as measurement errors were removed and real
  work landed: 1.85× (the benchmark's own round-structure bug inflating both
  sides) → 3.80× (bug fixed on one side) → 2.77× (imports batched) → **2.09×**
  (Shoup butterfly). Only the last was a genuine speedup; the others were
  corrections.

So the honest answer to "will faster polynomials bring FHE close to MPC" is
**no**. Faster ring arithmetic is real and worth having, but pure ABY3 does this
layer in 34 ms and the secure hybrid needs 72 ms. FHE's case here was never raw
speed; it is the *communication* profile — 15 messages and 34 KB against MPC's
9276 messages and 350 KB — which is what pays on a WAN.

### III.5 Offline/online split for the scheme switch

With the multiply cheap, the *conversions* became the bottleneck (149 ms +
108 ms = 59% of the secure hybrid), dominated by BFV encryptions of masks. Two
standard optimizations from the literature apply directly:

- **Delphi-style offline/online split**: the to-MPC masks `rᵢ` are
  *query-independent*, so their encryption is precomputed before the query
  arrives. Online, that leg is 3 ciphertext additions + 1 decryption.
- **Mask-and-correct on the return path**: each party precomputes `Enc(ρᵢ)`
  offline and, online, publishes only `δᵢ = cᵢ − ρᵢ` — uniform, so it leaks
  nothing — letting the evaluator fold it in with a *plaintext* addition
  (`Bfv::add_plain_poly`, O(n), no ring multiply). Since `Σ(ρᵢ + δᵢ) = Σcᵢ`, it
  is the same interactive bootstrap with **no online encryption at all**.

| conversion leg | before | after (online) |
|---|---|---|
| FHE → MPC (`to-mpc`) | 149 ms | **37 ms** |
| MPC → FHE (`to-fhe`) | 108 ms | **0.4 ms** |
| **secure hybrid, online latency** | 437 ms | **160 ms** |

Stated honestly: this does **not** reduce total work (the same six encryptions
happen — 282 ms of them now precomputed); it moves query-independent work *off
the critical path*, which is precisely what the split buys. Output remains
bit-exact.

### III.6 Lifting the layer-size ceiling, and why the packing is already optimal

The secure hybrid previously asserted a single weight ciphertext, capping layers
at `(out+1)·in ≤ n = 4096`. It now **chunks**: `⌈out/(n/in − 1)⌉` weight
ciphertexts, each multiplied against the same encrypted input, whose forward NTT
is **computed once and reused** (`Bfv::mul_pre`) — saving 2 of 4 forward
transforms per prime per chunk.

Worth stating as a finding rather than a to-do: **the reversed-coefficient
packing is already at the fundamental limit for dense layers.** A degree-n
product yields at most `n/in` *separable* outputs, because distinct rows need
disjoint weight windows — so a matvec inherently costs `⌈out·in/n⌉`
multiplications, which is exactly what chunking does. Gazelle's SIMD packing
achieves the same MAC throughput but needs **rotations and Galois keys**, which
would forfeit this construction's no-evaluation-keys property; ours is the
rotation-free, Cheetah-style layout. Convolution is the case where coefficient
packing does strictly better (III.7).

Measured (secure hybrid, exact at every size):

| layer | multiplications | online | offline |
|---|---|---|---|
| 8×2 | 1 | 164 ms | 282 ms |
| 256×32 | 3 | 656 ms | 590 ms |
| 784×16 | 4 | 505 ms | 734 ms |

Two real bugs were caught getting here, both fixed and tested:

- **Mask reuse across chunks leaked.** Every chunk opens the same coefficient
  positions, so sharing one mask set would reveal `pre_i − pre_j` to anyone
  seeing two openings. Masks are now fresh per chunk.
- **Fixed `t = 2^20` silently overflowed.** A 784-term dot product of
  ~50-magnitude values reaches ~2²¹, wrapping the plaintext modulus and
  producing wrong (not merely noisy) results. `t` is now sized from the measured
  activation range, with the depth-1 noise budget asserted.

### III.7 Cheetah-style convolution packing: 1352 multiplications become 8

Convolution is exactly the case the dense-layer limit does *not* apply to.
Polynomial multiplication **is** convolution, so if the image is flattened
row-major and the kernel is packed **reversed on every axis** (channel, row,
column) at matching strides, one ciphertext multiplication computes the entire
sliding-window convolution for a whole output channel. Two properties make it
exact rather than approximate:

- **Channels pair off automatically.** The reversed channel offset makes
  `c + (C−1−c′) = C−1` hold only when `c = c′`, so every input channel's
  contribution lands on the *same* coefficient and they sum — precisely what a
  multi-channel convolution wants.
- **No row-wrap contamination.** A neighbouring row could only leak into an
  output coefficient via a column carry (`col + b ≥ W`), which would require
  `ox ≤ −1`. Impossible for a valid output, so the read-out positions are clean.
  (Other coefficients *are* garbage; we never read them.)

Conv output `(oy,ox)` lands at `(C−1)·HW + (oy+kh−1)·W + (ox+kw−1)`. The cost is
`out_c` multiplications — **independent of the image size** — against the
`⌈out_c·oh·ow·CHW/n⌉` an im2col/Toeplitz matvec needs.

Measured A/B on the same classifier (`baru bench-cnn --img D`), both paths exact
against the plaintext reference, 0 bootstraps. (Measured while the pipeline was
still on the cheaper insecure engine — the *ratio* is what transfers; the
absolute FHE column is superseded by the secure-parameter numbers in IV.3.)

| D | Toeplitz mults | packed mults | FHE time (Toeplitz) | FHE time (packed) |
|---|---|---|---|---|
| 8 | 3 | 2 | 289 ms | 222 ms |
| 12 | 16 | 2 | 1037 ms | 209 ms |
| 28 | 1352 | 2 | **77497 ms** | **264 ms** |

At MNIST resolution that is a **294× faster convolution** and 5.5× faster end to
end (94.0 s → 17.1 s) — and it puts a full 28×28 image within reach, which the
module previously documented as out of scope (`D ≤ ~12`).

**Half the remaining FHE time was query-independent (`--precompute`).** With MPC
down to a fraction, the same per-phase treatment on the FHE side gave: model
encryption 218 ms, ct×ct products 210 ms, decrypting the masked openings 217 ms,
re-encrypting shares 50 ms, input 17 ms — and encrypting the **scheme-switch
masks 656 ms**, 48% of FHE and 39 of the inference's 52 encryptions.

Masks are uniform random values depending on neither the image nor the weights,
only on the network's *shape*, so they can be generated before the query exists
— Delphi's preprocessing idea applied to the *switch* rather than the linear
layer. They must still be **fresh per query** (two queries masked identically
would reveal the difference of their pre-activations), so this moves work off
the critical path rather than removing it — which is exactly the deployment
question: a server idle between patients has the time, a patient waiting does
not. Precomputing also lets the three parties' mask ciphertexts be *summed* in
advance, so the online step is one ciphertext addition instead of three
encryptions plus three additions.

D=28: online **1.99 s → 1.31 s** against 680 ms offline. On the trained two-conv
medical model, which switches four times rather than twice: **4.94 s → 2.90 s**
online against 2.03 s offline. A test runs the identical inference both ways and
requires the same class, and the mask pool asserts exact consumption so its size
prediction cannot drift from the pipeline.

**The bottleneck moved off FHE — and has since moved back.** Right after
packing, FHE was 264 ms of a 17.1 s run (1.5%) and the rest was the MPC segment,
which motivated the batching work in Part I. But most of that MPC time was the
unbatched scheme-switch import, not the comparisons; with the import batched and
the nonlinearities garbled, the same D=28 classifier ran in 1.31 s of which
695 ms (53%) was FHE, and the trained two-conv medical model — four encrypted
linear layers to this one's two — was 73% FHE. What is left is irreducible: the
ct×ct products and the model encryption depend on the query or the weights, so
no preprocessing reaches them. The highest-value remaining optimization is
`fhe::bfv` itself (single-threaded, no GPU), not the MPC layer.

---

## Part IV — Applications

### IV.1 Real MNIST: a trained model, real test digits, encrypted

`baru bench-mnist`. Everything in Part III proves the pipeline is *exact*. This
adds the orthogonal question a reviewer will ask: **what accuracy does it get on
real data?**

The model is a small but real **two-convolution** CNN —
`1×28×28 → conv(8,3×3) → ReLU → pool → conv(16,3×3) → ReLU → pool → dense(→128) → ReLU → dense(→10) → argmax`
— **trained** on the full 60 000-image MNIST training set (numpy SGD; regenerate
with `tools/train_medmnist.py`) and then **quantised into the very integer
fixed-point scheme `CnnModel::forward` runs** — so the number below is the
accuracy of the *deployed integer model*, not a float model that would lose
something afterward. It has **four encrypted linear layers** (two conv, two
dense), three MPC nonlinearities and three scheme switches, and still **zero
bootstraps** — the second, multi-channel convolution composes in with nothing
but "switch again." Weights and a real test subset are embedded in
`examples/data/mnist_cnn.rs`.

```
plaintext-integer accuracy : 299/300 = 99.67%   (in-repo test subset)
full MNIST test set (10 000): 98.89%             (identical weights, offline)
ENCRYPTED pipeline (4 imgs) : 4/4 match plaintext exactly, 0 bootstraps
```

The encrypted run matters more than the accuracy: because the hybrid is
bit-exact against `forward`, **every encrypted prediction equals the plaintext
one** — verified on real digits through all four encrypted linear layers, not
asserted. So the encrypted accuracy *is* 98.89%; privacy costs nothing in
accuracy. The Rust integer forward was checked to agree with the numpy reference
digit-for-digit on the embedded subset (299/300 both), so the offline
10 000-image number transfers faithfully.

These weights were **regenerated on 2026-07-28** by `tools/train_medmnist.py`
after the original trainer was lost, which is why the number moved from 98.05%
to 98.89%: same architecture, same quantisation scheme, a fresh 30-epoch run.
The embedded 300-image test subset is byte-identical to the original file, so the
two models are directly comparable. (Lesson, now a rule: generators live in
`tools/`, not in a scratch directory.)

**Where this sits vs the literature.** The honest framing is like-for-like on
*what is hidden*, since that — not accuracy — is the axis this work moves.

| system | model | MNIST acc | what is hidden |
|---|---|---|---|
| **baru** (this repo) | conv(8)+conv(16)+128-hidden+dense | **98.9%** | **image *and* weights** (ct×ct) |
| Gazelle (2018) | larger CNN | ~99% | image only (server holds plaintext weights) |
| Delphi (2019) | larger CNN | ~99% | image only |
| CrypTFlow2 (2020) | larger CNN | ~99% | image only |
| MiniONN / Cheetah | larger CNN | ~99% | image only |

The residual ~0.1% accuracy gap is model size, and closing it is a *training*
question the no-bootstrap construction is indifferent to (depth is free). The
column that distinguishes this work is the last one: the systems above are
two-party client/server protocols where **the server evaluates on plaintext
weights** — the model is not hidden. Here the linear layers are
**ciphertext×ciphertext**, so the weights are encrypted too, and the only value
that ever leaves the cryptographic domain is the class index. That is a
different (stronger) privacy model, not a faster route to the same one.

### IV.2 Private *diagnosis*: real biomedical images

`baru bench-medmnist`. Handwritten digits are not a medical claim. **MedMNIST
v2** (<https://medmnist.com>, CC BY 4.0) distributes twelve 2D biomedical
datasets at **28×28 — exactly MNIST's shape**, so the pipeline runs on them
unchanged:

```sh
baru bench-medmnist --dataset all --hybrid 3 --both
```

| | pneumonia (chest X-ray, 2 classes) | organa (abdominal CT, 11 classes) |
|---|---|---|
| integer accuracy, full test set | **84.78%** (624 imgs) | **81.18%** (17 778 imgs) |
| float accuracy of the same net | 84.78% | 81.03% |
| majority-class baseline | 62.50% | 18.48% |
| published ResNet-18 (28×28) | 0.854 | 0.935 |
| 95% CI on the full-test figure | [81.7, 87.4] | [80.6, 81.8] |
| **AUC, full test set** | **0.9314** | **0.9590** (macro OvR) |
| published ResNet-18 AUC | 0.944 | 0.997 |
| **balanced accuracy, full test set** | **80.64%** | **80.49%** |
| sensitivity / specificity | 97.18% / **64.10%** | — |
| encrypted == plaintext diagnosis | **300/300** | **300/300** |
| FHE bootstraps | 0 | 0 |

The AUC and balanced-accuracy rows were measured against the real MedMNIST test
splits from these exact committed weights, not from a regeneration — and
recomputing accuracy the same way reproduced 84.78% / 81.18% to six decimals,
which is what ties them to these weights.

**Read the sensitivity/specificity row before quoting the accuracy.** Pneumonia's
test set is ~60% positive, and the model earns its 84.78% largely by
over-calling disease: it catches 97.18% of pneumonia but clears only 64.10% of
healthy scans. Balanced accuracy is 80.64%, not 84.78%. Organa's weakest classes
are kidney-left and spleen at 64% recall each. None of this is a cryptographic
result — it is what the 53 k-parameter model does — but reporting accuracy alone
would overstate it.

Latencies **re-measured 2026-08-01, Apple M1 Pro (8 cores), 20 images/config**.
The previous revision of this table carried 5.71 / 4.94 / 2.90 s, which had gone
stale: the session-key hoists of 2026-07-30 (`3cd9fe4`) refreshed `README.md`
and [private-inference-performance.md](private-inference-performance.md) to
488 ms but never reached this table. The check that this is a code change and
not a faster machine: HEAD-before-optimisation measures **488 ms** here, exactly
the figure that commit recorded.

| | pneumonia | organa |
|---|---|---|
| latency/diagnosis, `a2b` | 1.63 s | 1.74 s |
| + `--gc` | 0.88 s | 0.90 s |
| + `--precompute`, **single core** | 0.435 s | 0.460 s |
| + **4 NTT threads** (**online**) | **0.268 s** | **0.291 s** |
| offline, **per query** (mask pool, cannot be amortised) | 0.14 s/image | 0.14 s/image |
| online at WAN (`--rtt-ms 20 --bandwidth-mbps 1000`) | 2.65 s | 2.81 s |
| party-to-party traffic (`--gc`) | 61 msgs, 10.1 MB | 76 msgs, 11.3 MB |

Both a single-core and a threaded figure are quoted deliberately: the threading
is in the multi-modular NTT (independent primes, plus a pure-map CRT), is
bit-identical to the serial path, and is disabled by `BARU_NTT_THREADS=1`. A
wall-clock number that silently depends on core count cannot be compared against
published work, so the comparable one stays measurable.

Quote the online figure *with* the per-query offline one: 0.32 s/image of mask
generation must happen for every query and only moves off the critical path, so
"0.41 s per diagnosis" is the latency, not the work.

Three readings. **Quantisation and encryption are both free**: the integer net
tracks its float parent to within 0.2 points (agreeing on 99.7% / 98.7% of test
predictions), and the encrypted pipeline reproduces the plaintext diagnosis
bit-exactly — privacy costs wall-clock, not accuracy. **The model, not the
crypto, is the ceiling**: this 2-conv classifier is sized to the envelope the
pipeline allows, which is three constraints and not one. Spatially, 3×3 valid
convolution and floor-pooling take 28 → 13 → 5 → 1, so a 28×28 input admits at
most three convs. By *packing*, the Cheetah layout needs `2·in_c·H·W ≲ n`, so
at the second conv's 13×13 input the first conv may emit at most **12**
channels before the product degree passes `n = 4096` (16 would need 5266) —
this is what stops the model getting wider, and it is a packing question, not a
limit of the no-bootstrap construction: it relaxes at `n = 8192` or by
splitting channels across ciphertexts. Only then does the depth-1 noise budget
bind, on the activation magnitude. The result is enough to match published
ResNet-18 on pneumonia and 12 points short of it on the 11-class organ task.

And **the cost centre is the MPC nonlinearities again**: of the 268 ms
chest-X-ray diagnosis, 99 ms is FHE (mul 44, decrypt 45, re-encrypt 10) and
**168 ms (63%) is the MPC segment** (import 22, ReLU 38, pool 69, argmax 8).
That is the fourth time the answer has flipped, which is the point: *which side
dominates* has a half-life of about one optimisation, so no plan should rest on
it. A diagnosis used to take ~33 s: the batched scheme-switch import took it to
5.71 s, garbled nonlinearities (`--gc`) to 4.94 s, preprocessed masks
(`--precompute`) to 2.90 s, the fixed-key AES gate hash and the three
session-key hoists to 488 ms, **commuting the ReLU past the max-pool to 435 ms**,
and **threading the multi-modular NTT to 268 ms on four cores** — 123× in total,
with the construction unchanged and every step exact.

That last step is worth stating as an identity rather than a tuning result:
`relu∘rescale` is monotonic non-decreasing, so it commutes with `max`, and
pooling *first* shrinks the ReLU layer by `POOL²` = 4× at no cost to the pool
itself. It is exact, not approximate — the bit-exactness tests are unchanged.
It buys 1.17× here and 1.38× on the `a2b` path, where the nonlinearity is the
dominant cost rather than a sixth of it.

Anyone optimising further should go to `fhe::bfv`, not the MPC layer. On a
wide-area link a diagnosis is 2.65 s — the pipeline is not round-bound, and
garbling is worth 2.0× on the MPC segment there.

The same swap applies to the linear-discriminator demo:

```sh
cargo run --release --example mnist -- --pneumonia --mpc --ckks
```

giving one 1669-gate circuit whose Q16.16 score is bit-identical across plain,
fhe-mock and ABY3, and equal to ~1e-7 under CKKS.

**Two silent failure modes to know about** when swapping datasets: a CNN whose
quantisation is not recalibrated silently wraps the plaintext modulus (III.6),
and a per-dataset normalisation mismatch produces a plausible-but-wrong accuracy
rather than an error. Both are checked now; neither announces itself.

### IV.3 A real CNN through the hybrid pipeline

`baru bench-cnn`. The gadget library composes into a full convolutional
classifier that runs end to end through the actual hybrid pipeline — FHE ct×ct
linear layers, MPC nonlinearities, **zero FHE bootstraps** — verified
**bit-exact** against the plaintext reference:

```
1×D×D image → conv(OC, 3×3) → ReLU∘rescale → maxpool(2×2) → dense(→10) → argmax → class
```

Both the image *and* the weights are encrypted. The convolution uses the
reversed-axis coefficient packing of III.7 and the same scheme-switch machinery
as `bench-deep-nn`; only the nonlinearities differ (ReLU fused with the
fixed-point rescale; max-pool and argmax from `crate::gadgets`). `--gc` swaps the
nonlinearity protocol, `--both` runs the pair and prints the ratio, and every
run prints the MPC **phase breakdown** — the thing whose absence hid an 8.3× win
for months.

The FHE side runs on **BFV at RLWE-secure parameters** (n=4096, q=2^100, at or
below the ~109-bit ceiling for n=4096 at the 128-bit level), and `CnnRun` records
`ring_n`/`log2_q`/`log2_t` so the security claim is a test
(`uses_rlwe_secure_parameters`) rather than only prose. Two supporting properties
came with that: `t` is **calibrated from the measured activation range** with the
depth-1 noise budget asserted (a fixed `t` is a correctness hazard, not a tuning
knob — III.6), and the image's forward NTT is computed **once and reused** across
output channels and weight chunks.

Measured (D=28, garbled nonlinearities, precomputed masks):

```
predicted class : exact vs plaintext        FHE bootstraps  : 0
conv ct×ct mults: 8 (packed) vs 1352 (im2col/Toeplitz) — 169× fewer
time            : total 840 ms (fhe 179 ms, mpc 658 ms)
offline         : 158 ms (query-independent; not in total)
mpc breakdown   : import 9 ms, relu 447 ms, pool 166 ms, argmax 30 ms
```

Earlier per-configuration breakdowns, useful for the shape rather than the
absolute numbers:

| network | class | bootstraps | total | fhe | mpc | breakdown |
|---|---|---|---|---|---|---|
| D=8, rtt=0 | exact ✓ | **0** | 1.17 s | 957 ms | 184 ms | import 8 / relu 47 / pool 59 / argmax 60 ms |
| D=8, rtt=20 ms | exact ✓ | **0** | 2.45 s | 1.19 s | 1.24 s | import 40 / relu 176 / pool 420 / argmax 546 ms |
| D=28, rtt=0 | exact ✓ | **0** | 2.54 s | 1.36 s | 1.15 s | import 9 / relu 615 / pool 467 / argmax 55 ms |
| D=28, rtt=0, `--gc` | exact ✓ | **0** | 1.99 s | 1.35 s | 609 ms | import 11 / relu 407 / pool 155 / argmax 27 ms |

Correctness is exact; *accuracy* on a real dataset is the orthogonal ML step
(IV.1, IV.2). Coefficient packing bounds this to small D; CIFAR-scale needs
SIMD-slot conv packing (Gazelle-style) — the noted next crypto-engineering step,
independent of the no-bootstrap result.

> Building this surfaced two real transport-only bugs the earlier benchmarks
> couldn't reach (they only ever compared secrets against zero):
> `aby3_protocol::share_public` produced an inconsistent replicated share (party
> 2's wraparound component was zeroed — invisible to `reveal`, wrong under
> multiplication), and `b2a` returned a share carrying the bit-width of its input
> bits (`1`) rather than the recomposed value's, so a downstream comparison
> decomposed a single bit. Both fixed, with regression tests. That the
> *composition* caught bugs the *unit* paths didn't is itself the argument for the
> gadget approach.

### IV.4 Private *training*, not just inference

`baru bench-train`. The stack so far assumes a model already exists. This closes
the loop: a least-squares classifier is **trained by SGD entirely under the real
3-party ABY3 protocol** — the training data stays secret-shared, and every
forward pass, gradient and weight update is an oblivious computation composed
from `crate::gadgets` — then the trained model does private inference on a
held-out example.

The one design choice that makes it tractable: **MSE loss instead of softmax.**
Softmax-cross-entropy's gradient needs `exp`/`log`/division (the transcendental
wall); regressing to one-hot targets with squared error gives a gradient that is
purely `sub`/`mul`/shift — no nonlinearity in the loss at all (`forward
z=W·x+b; δ=z−y; ∂W=δ⊗x; W-=∂W≫lr`, the shift being `gadgets::arith_shr`). Every
step is exact integer fixed-point, so the encrypted run matches a plaintext
integer-SGD reference **bit-for-bit**. Measured (12 samples, 6 binary features,
3 classes, 20 epochs, lr = 2⁻⁴, real protocol):

```
loss (MSE): 65536 → 7172   (89% reduction, monotone)
train accuracy: 100%
encrypted model == plaintext model: EXACT (bit-for-bit)
held-out example: classified correctly (only the class index revealed)
FHE bootstraps: 0     MPC wire: 105,162 msgs / 3.8 MB
```

Honest framing: training is naturally an MPC computation (as in SecureML /
ABY3's own training experiments), so "0 bootstraps" is trivially true on this
side — the no-bootstrap *FHE* contribution is the *inference* deployment, and
this module completes the private-ML lifecycle around it, showing the same
gadget library composes a full SGD trainer. Correctness is exact; *accuracy* is
a property of the (synthetic, separable) data, reported separately.

**Backpropagation *through ReLU* (`--hidden H`).** The linear case converges
provably; adding a hidden layer makes it a real (non-convex) network whose
backward pass differentiates a nonlinearity — and it stays inside the gadget
library, because `ReLU′(z) = (z > 0)` is exactly the mask `gadgets::relu`
already computes (reused from the forward pass for the backward). The whole loop
runs under the real protocol and the trained model matches the plaintext integer
reference **bit-for-bit**:

```sh
baru bench-train --hidden 8 --epochs 16 --lr-shift 6 --lr2-shift 9
# 6 -> 8 (ReLU) -> 3 | loss 64366 -> 39013 (39%), acc 50%
# encrypted model (backprop through ReLU) == plaintext: EXACT (bit-for-bit)
```

Two honest notes: the two layers need **different learning rates**
(`∂W1 = W2ᵀδ2 ⊙ step ⊗ x` is smaller than `∂W2 = δ2 ⊗ a1` by the scale of `W2`,
so one rate freezes one layer or destabilizes the other), and unregularized
*integer* SGD **diverges past a stable regime** — an ML tuning matter, entirely
orthogonal to the crypto claim, which is that **private backprop through a
nonlinearity is exact**. Deeper nets and softmax loss remain future work.

### IV.5 The composable gadget library

`src/gadgets/` is a small library of MPC building blocks, each a pure function
of the oblivious primitives (`add`/`sub`/`mul`/`lt`/`share_public`): `relu`,
`max2`/`max_reduce`/`maxpool2d`, `argmax`, `dense`, `conv2d`. Every one ships
with a plaintext reference and is tested for **bit-exact** equality, not
approximation.

Why this matters: Baru is effectively an **Arithmetic Black Box** — it
manipulates secrets only through opaque handles and a fixed primitive set, never
inspecting the representation. So if each primitive UC-realizes its ideal
operation, each gadget does too (UC composition), and **any network composed
from these gadgets is secure and correct by construction** — no per-architecture
proof. The research question becomes "which architecture is *fastest*," not "is
this one secure."

As evidence the composition is semantically exact rather than merely
type-correct, the library includes an end-to-end LeNet-shaped classifier —
`conv → relu → maxpool → flatten → dense → relu → dense → argmax` — that runs
entirely through the gadgets, matches its plaintext reference bit-for-bit, and
reveals **only the class index** (`argmax` is the escape from the transformer
softmax wall: a CNN *classifier* needs argmax over logits, never
`exp`/division). Adding channels, conv blocks or residual connections needs no
new security argument. Convolution here is the correctness/composition form
(direct MACs / im2col-matvec); the efficient FHE conv *packing* is III.7.

The cost caveat is I.4: composition is free per-operation but was not free in
round structure until the gadgets were rewritten over batched primitives.

### IV.6 Flagship demo

`baru demo-private-mnist`. The audience-facing version. A handwritten digit
(real MNIST sample) and a trained discriminator are **both encrypted**; the
784-pixel dot product is **one** BFV ciphertext×ciphertext multiplication
(decrypted under `s²` — no bootstrapping, no evaluation keys); the class
decision is a real 3-party ABY3 threshold comparison that reveals **only the
final bit**. Nobody — not the evaluator, not the MPC parties, not even the
client — ever sees the score, only the predicted digit.

That last property is the point of thresholding under MPC: a plain-FHE
classifier must decrypt the score to decide, and the score leaks information
about the input. (It is exactly the leak in the reference `hom-ham` iris
pipeline, whose `DecryptMul` reveals the raw Hamming distance.) Encrypting the
*model* is a strictly stronger privacy setting than Gazelle/Delphi/Cheetah,
whose server holds the model in plaintext.

Measured, per sample, `n = 4096, q = 2^100, t = 2^21`: keygen 2 ms, image
encryption 3.7 ms, weight encryption 3.7 ms, the whole 784-element dot product
6–8 ms, switch to MPC ~16 ms, MPC threshold ~20 ms — **~55 ms total**, 20 MPC
messages and 4.9 KB of wire traffic for the decision, 128 KB per ciphertext.
Both repo samples classify correctly.

---

## Part V — The circuit compiler

### V.1 Circuit generation: where Baru beats hand-written code

`baru bench-circuit`. The **compiled** path — `Expr` → `CircuitCompiler` →
`FlatOp` → `mpcc` — differs from the eager APIs because there Baru is a circuit
language and can rewrite before anything runs:

```
  program          n      ops ops/no-cse   interactive   rounds
  relu-layer       1        3          3             3        3
  relu-layer      64      192   255 (1x)           129        3
  dot             64      127        127            65        2
  max-fold         8       28 4372 (156x)            15       15
```

**Independent work batches automatically.** ReLU-layer rounds are *flat in
width* — 3 at n=1 and 3 at n=64, while interactive ops grow 3 → 129. ASAP
scheduling over the dependence DAG puts every independent operation at one
level, with no vectorized IR and no user effort. That is what the hand-written
route needs explicit `lt_batch`/`mul_batch` for, and what the gadget library had
to be patched three times to get right — so **on this axis the compiler is
already ahead of hand-writing.**

**Dependent reductions needed a rewrite, and now have one.** A fold is a chain,
and ASAP is optimal for a fixed lowering but cannot reassociate.
`Rule::ReassociateReduction` recognises an associative reduction whose operator
is a repeated *subgraph shape* — a maximum is `acc + (x>acc)·(x−acc)`, four
instructions, which `Rule::Reassociate` cannot see — and rebalances the chain
into a tournament, `2n−1` rounds → `2⌈log₂ n⌉+1`:

| max-fold | as written | rebalanced |
|---|---|---|
| n=4 | 7 rounds | 5 |
| n=8 | 15 | **7** |
| n=10 | 19 | **9** |

End to end under Shamir: **65 → 29 rounds (2.21×)**, instruction count
unchanged, differentially verified. This is the transformation `gadgets::argmax`
and `max_reduce` needed *by hand*; the compiler now derives it for any
associative operator — and refuses it for non-associative ones, since a
subtraction fold is left alone. Associativity is tested by evaluation, not
assumed, because the optimizer accepts rewrites on cost rather than semantics.

**CSE was missing from the compile path entirely, and now isn't.** `mpcc::opt`
had a `Cse` rule reachable only from `baru mpcc optimize`; `compile_fhe` ran
constant folding and dead-wire elimination and nothing else, so every circuit
handed to FHE or ZKP carried the source's redundancy. A fold written the obvious
way repeats its accumulator subtree per step and flattens *exponentially*. The
pass now runs inside `CircuitCompiler`:

| circuit | before | after |
|---|---|---|
| max-fold n=8 | 4372 ops | 28 ops (156×) |
| **MNIST demo** | **2365 gates** | **1764 gates (−25%)** |

MNIST scores bit-identical, all 18 FHE differential tests pass (real TFHE and
SEAL CKKS included). At TFHE's ~0.2–0.5 s/gate, those 601 gates are minutes per
run.

### V.2 The same program, a different circuit per backend

`baru bench-backend-aware` — the claim nobody else makes. MP-SPDZ optimizes
rounds and is blind to FHE; FHE compilers optimize multiplicative depth and are
blind to MPC; HyCC → Silph → SING assign *protocols* to parts of a program but
do not rewrite the circuit differently per target.

One program, optimized once per backend, then every circuit priced under every
backend (WAN, 3 parties, n=8, all differentially verified):

```
max-fold        as shamir  as semi-bin     as bmr    rules kept
  shamir       1453.010ms*  3761.043ms   281.564ms   reassociate-reduction
  semi-bin     1453.010ms   3761.043ms*  281.564ms   reassociate-reduction
  bmr          3253.010ms   8761.043ms   281.564ms*  -
```

`*` marks each circuit under the backend it was optimized for. **The same
rewrite is a 2.24× win under Shamir and worth nothing under BMR** — depth is the
round count for an interactive protocol and free for a constant-round garbled
one — so the optimizer keeps it for one and declines it for the other. Running
BMR's circuit under Shamir costs 2.24×.

Across `relu-layer`, `max-fold` and `dot-const`: **2, 2 and 3 distinct circuits
from one program**, worst wrong-circuit penalties 1.66× / 2.33× / 1.81×. The
command fails if the diagonal is ever beaten.

### V.3 A whole network: hand-written vs compiled

`baru bench-compiled-nn` — the question the circuit-optimization paper is
actually about. The same network — `x(n) → dense(h) → ReLU → dense(c) → max`,
**secret weights** — evaluated two ways over the *same* 3-party transport, both
checked bit-exact against a plaintext integer reference:

- `gadget` — Rust calling `gadgets::{dense, relu, max_reduce}` (the batched
  expert baseline)
- `compiled` — the same network as a **Baru expression**, through
  `CircuitCompiler` → `FlatOp` → `mpcc::exec`, a level-batched executor

The source says nothing about batching — it is scalar arithmetic, a mux-style
ReLU and a left-fold `max`. n=32, h=16, c=8, RTT 20 ms:

| route | ops | interactive calls | messages | wall-clock |
|---|---|---|---|---|
| `gadget` (expert) | — | 39 | 95 | 677 ms |
| `compiled` | 1317 | 27 | 87 | 1272 ms |
| `compiled+reassoc` | 1317 | **15** | **47** | 663 ms |

**0.38× the interactive calls and 0.49× the messages of expert hand-written
code**, at parity on wall-clock. The command also prints an **ablation ladder**
attributing each pass — ASAP batching is 285× on interactive calls, CSE is 68×
on circuit size and 30× on bytes, reassociation is 1.25× on rounds. The passes
act on different axes and none substitutes for another, which is why the ladder
is reported rather than one number. The advantage *grows with width*: the
compiled route stays at 15 interactive calls while the expert route goes 31 →
39, because ASAP scheduling batches every independent operation however many
there are.

Two honest notes. **Without `ReassociateReduction` the compiled route loses**
(1.88× wall-clock) — the expert's `max_reduce` is a hand-written tournament, and
that ablation is what shows which pass buys what. And **the wall-clock is
parity, not a win**: both routes perform the same number of comparison layers,
which dominate latency; the compiler's gain is in multiplies and sharing.

### V.4 Cost models validated, not asserted

`Semi2k` and `LowGear` are wired into the `mpcc` VM, so `baru mpcc run` and
`baru mpcc calibrate` drive them like any other protocol — and both flip from
`planned` to `impl` in `baru mpcc protocols`. The point is the calibration loop:
a protocol you can only *cost* is a protocol whose cost model nobody has
checked.

```text
baru mpcc calibrate --expr '(+ (* a b) (* c d))' --parties 2 ...

protocol   parties  rounds pred/meas  online bytes pred/meas   result
dealer          2        3/3                     72/72         42 ✓
semi2k          2        3/3                     72/72         42 ✓
lowgear         2        3/3                     27/27         42 ✓

  dealer   rounds +0.0%  bytes +0.0%
  semi2k   rounds +0.0%  bytes +0.0%
  lowgear  rounds +0.0%  bytes +0.0%
```

All three at **0% on rounds and bytes**. The online phase is Beaver's in every
case, which is exactly why this is a useful check: with the online phase held
constant against two already-validated protocols, any disagreement would have to
come from the part that differs — where the triples come from.

Getting there caught a real accounting bug. LowGear is the first protocol whose
field is not byte-aligned (17-bit modulus), and `cost.rs` priced `field_bits / 8`
while the VM charged `ceil(order_bits / 8)` — for every previous protocol those
agree, so nothing had exercised the difference. It showed up as a −33% byte
error. The fix follows Shamir's existing convention: the registry's `field_bits`
is the **byte-aligned wire width** (24 for a 17-bit element, as Shamir declares
64 for a 61-bit prime), and a test now fails if the model's bytes-per-element
and the VM's ever drift apart again.

**LowGear's capability boundary is enforced, not documented-and-hoped.** Its
~17-bit field could not supply the `KAPPA = 26` bits of masking headroom that
bit-decomposition needs, so comparison-shaped programs were **refused** rather
than run and silently wrong. `mpcc::engine_can_run_program` is the predicate;
sweeps that enumerate `RUNNABLE` skip that row and keep going, the same way an
unsupported party count is handled. (The wide-field work in II.5 removed the
limitation itself.)

---

## Part VI — Building blocks, measured

### VI.1 Oblivious transfer, base and extended

`baru bench-ot`. OT is the primitive that pays twice: private set intersection
is built on it, and so are Semi2k's and MASCOT's triples. `src/ot/` has
Chou–Orlandi base OT over Ristretto255 and the IKNP extension on top.

```
  protocol      OTs       us/OT   bytes/OT   rounds
      base      128      126.35       64.2        2
 extension      128      127.99       48.2        3
      base     8192      123.92       64.0        2
 extension     8192        4.67       16.5        3
```

Base OT is flat in the batch size — each transfer is its own pair of scalar
multiplications. The extension starts level with it at 128 OTs (that batch *is*
the `κ = 128` base OTs it must pay for first) and is **27× cheaper per OT and 4×
smaller by 8192**. Bytes per OT converge to `κ/8 = 16` from above, which is the
protocol's analytic online cost of one column bit per output OT — the measured
curve landing on that floor is what says the accounting is the protocol's and
not the harness's.

Both layers are **semi-honest**, and for IKNP that is a concrete gap rather than
a caveat: nothing forces the receiver to use the same choice vector in every
column, and an inconsistent one leaks the sender's secret bit by bit. KOS15's
correlation check is what closes it.

### VI.2 Private set intersection

`baru bench-psi`. Two hospitals want to know which patients they share, and
nothing else. `src/psi/` implements DH-PSI over Ristretto255 (semi-honest,
one-sided output), and the benchmark prices the OT-based alternative from the
*measured* OT cost rather than assuming it wins:

| | hides non-shared items? | bytes/item | µs/item |
|---|---|---|---|
| naive hash exchange | **no** | 32 | **0.43** |
| DH-PSI (measured) | yes | 96 | 165 |
| OT-based (projected) | yes | ~58 | ~1 |

DH-PSI is flat in both — three flights of compressed points, two scalar
multiplications per item. The insecure baseline is **~380× faster**, which is
exactly why hash-exchange keeps getting shipped; what the 380× buys is an
offline dictionary attack that recovers the sender's *entire set*, and patient
identifiers are precisely the low-entropy input that attack needs.

The projection says an OT-based PSI wins by only ~1.7× on bytes but ~100× on
time, so the deciding axis is **compute, not bandwidth**. That is worth knowing
before building the harder protocol — and it is a projection (measured per-OT
cost × a literature OPRF code width), not a measured PSI.

### VI.3 Private information retrieval — and why we are not using it

`baru bench-pir`. Encrypting records at rest does not hide *which* record you
read, and an access pattern over a patient database is itself a disclosure.
`src/pir/` implements the textbook single-server scheme over the same BFV: the
database becomes one plaintext polynomial per shard, the query is an encrypted
monomial, and since `xⁿ = −1` a single plaintext-ciphertext multiply drops the
wanted record into the low coefficients. One round, `O(1)` query, no server key,
**zero multiplicative depth**.

It is also, measured honestly, the wrong tool — 256-byte records, n=4096:

```
 records  shards      query     answer   whole db  vs plain  vs whole  server ms
      32       1    104.0 KB   104.0 KB     8.0 KB      832x     0.04x  16
     256       8    104.0 KB   832.0 KB    64.0 KB     3744x     0.07x  130
    2048      64    104.0 KB     6.5 MB   512.0 KB    27040x     0.08x  1032
```

Retrieval is correct at every size, and the ratio converges to `1/13` rather
than improving with scale. A response ciphertext costs
`2·⌈log₂q/8⌉ / ⌊log₂t/8⌋` = **13×** the entire plaintext content of the shard
that produced it, and this scheme returns one response per shard — so its
traffic is ~13× a whole-database download regardless of `N`. There is no
crossover.

So by this repo's own admission rule for new primitives, PIR has not earned its
stage: the cheapest way to hide the access pattern here is still to download the
shard. What it has earned is a correct implementation, a measured baseline, and
a specific next step (SealPIR-style recursion with server-side query expansion,
which needs Galois automorphisms BFV here lacks). Reporting that is more useful
than a table where every primitive wins.

### VI.4 SHE as an MPC building block: LowGear triples

`baru bench-triples`. MP-SPDZ builds its offline phase on somewhat-homomorphic
encryption (Overdrive/LowGear/HighGear). Baru's cost model already carried that
protocol — as an *estimate*: `Status::Planned`, ~384 B/triple from the
literature, plus an explicit disclaimer that its CPU cost was not modelled.
`src/mpcc/lowgear.rs` runs it instead.

The protocol is structurally identical to `semi2k` — both compute
`c = a₀b₀ + a₁b₁ + a₀b₁ + a₁b₀` and need the two cross terms from an OLE. Only
the OLE differs: Semi2k uses `2k` correlated OTs; LowGear uses one **ciphertext ×
plaintext** multiply over `n` CRT slots (depth 0 — no relin, no bootstrap, no
eval keys, and not even a ct×ct multiply).

**What made it possible:** `src/fhe/slots.rs`. Every other packing in this repo
is coefficient-based, where a ring product is a *convolution*; an OLE needs `n`
**independent** products. CRT slot packing under a prime `t ≡ 1 (mod 2n)` gives
that — which is why `BgvParams::t` had to stop being a power of two, and why BGV
rather than BFV is the natural host (no `Δ = q/t` to divide, so `t` need not
divide `q`). Without slots one ciphertext pair yields *one* product, i.e.
~131 KB/triple: **40× worse than the OT it replaces.** Slots are the difference
between winning by 7× and losing by 40×.

Measured, two parties, semi-honest, 4096 triples:

| route | domain | B/triple | triples/s | rounds | ok |
|---|---|---|---|---|---|
| **lowgear (SHE)** | `F_941356466589171713` | **192** | 73 400 | 2 | ✓ |
| semi2k (OT) | `Z_2^64` | 3074 | 2 620 | 3 | ✓ |

**⚠️ CORRECTED 2026-08-01 — the two rows are over different domains.** This
section previously read "LowGear's ~61-bit field covers Semi2k's 64-bit values
without CRT limbs, so the comparison is direct: 192 against 3074 — a 16.3×
communication advantage". That is wrong: a ~61-bit prime field does **not**
represent every 64-bit value, so the raw table compares a 61-bit field triple
against a 64-bit ring triple.

At **equal 64-bit width** LowGear needs two CRT limbs, giving **384 B/triple
against OT's 3074 — an 8.0× communication advantage**, which is the figure to
quote. `bench-triples` now prints the equal-width line itself so the
apples-to-apples number cannot be read off the raw table by mistake. The 192 B
figure remains correct *for a 60-bit prime field*, and is still 2× better than
the literature's 384 B estimate at that width.

The direction of the finding is unchanged; only its magnitude was overstated.

**The surprise, and a correction to our own model.** `mpcc::cost` asserted that
*"LowGear trades bandwidth for CPU: its preprocessing is compute-bound."* That
note was never measured, and it is **wrong** here: LowGear generates triples
**~28× faster** than the OT route. There is no trade at these parameters — SHE
wins on bytes *and* on CPU, by a wide margin. The note has been replaced with
the measurement.

**Scope, stated plainly.** Semi-honest only: no MACs, no sacrifice, and **no
zero-knowledge proofs of plaintext knowledge**. A malicious party can send a
malformed ciphertext and learn about the other's input from decryption
behaviour. ZKPoPK is the bulk of the Overdrive paper and is deliberately absent
— this is the LowGear analogue of Semi2k, not of SPDZ.

### VI.5 Where the FHE-vs-MPC crossover actually is

`baru bench-layer`, WAN, 2 parties, preprocessing charged per triple
*consumed*:

| width | dealer (lower bound) | semi2k (OT) | **lowgear (SHE)** | hybrid/bfv |
|---|---|---|---|---|
| 64 | 600.6 ms | 694.5 ms | **606.2 ms** | 692.1 ms |
| 1024 | 603.0 ms | 942.8 ms | **623.4 ms** | 692.1 ms |
| 4096 | 610.9 ms | 1.74 s | **678.4 ms** | 733.7 ms |

**The hybrid never beats SHE-preprocessed MPC at any width measured**, and
LowGear tracks the trusted-dealer *lower bound* within 5 ms at width 64 —
because its preprocessing is **384 B/triple against OT's 3074 at equal 64-bit
width — 8.0× cheaper** (see the correction in VI.4; the raw 192 B figure is for
a narrower field), and ~28× faster. An earlier composed estimate said the crossover moved from
width 64 to 4096; measuring it properly says it is gone in this range entirely.

That is a real qualification to the paper outline's E25, whose conclusion — *"the
entire case for FHE here is the cost of preprocessing"* — turns out to be
exactly right, and to cut the other way once MPC gets the better preprocessing.
**The crossover belongs to the baseline's preprocessing, not to either
cryptosystem.**

---

## Part VIII — The cost of active security

*Command:* `baru bench-malicious --n 64 --bits 32`. Both legs run the same
workload through the **identical** transport stack (`LatencyTransport` +
`CountingTransport`), so the only difference is the protocol.
`docs/malicious-hybrid.md` describes the constructions.

### The headline

| Workload | bytes | interactive calls | wall clock (WAN, 25 ms one-way) |
|---|---|---|---|
| Comparisons, n=32, 32-bit | **2.63×** | **1.06×** | **1.23×** |
| Dot product, n=64, 32-bit | 2.44× | 2.00× | 2.87× |

The comparison row is the one that matters for private inference: every ReLU,
max-pool and argmax reduces to a comparison, so it is the entire nonlinear
surface. Active security there costs **2.6× the bytes and 1.06× the rounds**.

That is the predicted shape, and the prediction was worth making explicit
because it is the repository's recurring lesson applied to a new protocol: the
MAC doubles what is *sent* but batches into the *same* round, so on a link where
latency dominates the upgrade is nearly free. Bit decomposition's two extra
rounds — the random-pair pool and the MAC-attaching opening — are per *batch*,
not per value, so they amortise: 1.06× at n=32.

The dot-product row is a small-workload artifact and is reported rather than
hidden. The semi-honest leg needs only 4 interactive calls in total, so the
fixed cost of generating a pair pool is not amortised against anything. It is
2.00× calls because the base is 4, not because multiplication doubles rounds.

### The ring sacrifice is free at these widths

The malicious multiplication check needs statistical slack in the ring, which
the literature charges as a widening to `Z_2^{k+s}`. Baru already shares over
`Z_2^128` while values are 32-bit, so the slack is already paid for:

```
values occupy 70 bits of the 128-bit ring
  → 58 bits of statistical security at no widening cost
```

70 bits is a 32-bit product (64) plus the accumulation depth of a 64-term dot
product (6). Reported by the benchmark from the widths actually in use, not
asserted — `aby3_verify::headroom_bits` computes it, and it returns 0 rather
than a comfortable number when the width leaves no room.

### Method note — an unbatched loop, again

The first run of this benchmark reported **39×–52× wall clock** on the dot
product against only 2× interactive calls. The cause was in the benchmark, not
the protocol: it shared inputs with scalar `share_secret` in a loop, and each
call costs one extra synchronisation in the malicious backend (the MAC-attaching
opening). `Mailbox::recv_matching` polls on a 2 ms sleep, so 128 extra
synchronisations are ~256 ms of pure waiting.

Switching to `share_secret_batch` — which both backends have, and which every
real workload in this repository already uses — took the dot product from 39× to
2.18×. This is the fourth time in this record that a segment-level measurement
said "the protocol is slow" and a per-phase breakdown said "one unbatched loop
inside it is slow." It is recorded here because the earlier number was wrong in
a way that would have been quotable.

---

## Part IX — Proving a quantized layer (ZKP)

*Command:* `baru bench-zkp-layer --widths 1,4,16,64,256,784`. Real Binius64
prover and verifier (`zkp::binius_real`); weights public, inputs private.

Baru's comparison table carried an empty ZKP column for anything resembling
inference. The reason was not that it had not been measured — it was that
before 2026-07-31 the ZKP backend proved nothing at all (Part VII), and the
real one that replaced it could not compile a ReLU until the circuit fragment
gained comparisons, `if`, shifts and division.

### A Q16.16 quantized neuron: `relu((Σ wᵢ·xᵢ) >> 16 + b)`

| width | gates | constraints | prove | verify | proof bytes |
|---|---|---|---|---|---|
| 1 | 11 | 14 | 6.3 ms | 1.0 ms | 18 544 |
| 4 | 23 | 42 | 4.6 ms | 0.8 ms | 21 120 |
| 16 | 71 | 151 | 5.2 ms | 1.0 ms | 31 456 |
| 64 | 263 | 589 | 8.0 ms | 1.3 ms | 47 808 |
| 256 | 1 031 | 2 341 | 14.6 ms | 2.3 ms | 71 968 |
| **784** | **3 143** | **7 159** | **32.3 ms** | **4.8 ms** | **111 712** |

784 is the MNIST input width, so the last row is one full first-layer neuron of
the network the rest of this record benchmarks.

Two things are worth reading off the sweep rather than the last row alone.
**Constraints grow linearly** in width (≈9 per input — a multiply, an add, and
the shared rescale/ReLU tail amortising away). **Proof size does not**: a 286×
larger circuit produces a 6× larger proof, which is the succinctness that makes
a SNARK worth using. Verification stays in single-digit milliseconds throughout.

The small-width rows are dominated by fixed setup, which is why width 4 proves
marginally *faster* than width 1 — not noise worth explaining away, just the
floor.

### What this does not say

This is **integrity, not confidentiality**, and the two are easy to conflate
when the numbers sit in adjacent columns:

* MPC and FHE keep data secret **from the parties computing on it**.
* A SNARK proves the computation was done correctly while hiding the witness
  **from a verifier**. The prover holds everything in the clear.

So "ZKP proves a 784-wide neuron in 32 ms" must not be set against "MPC
evaluates a layer in *N* ms" as if they were alternatives. They answer different
questions, and a deployment that needs both pays for both. The benchmark prints
this caveat with its results so the numbers cannot travel without it.

---

## Part VII — Negative, superseded and withdrawn results

Kept because a repository that only records its wins is not reporting.

### VII.1 SI-NTRU: removed as insecure (2026-07-30)

For most of this project's life the fast FHE engine was a Rust port of the
**scale-invariant NTRU (LTV-family)** scheme from
[Inversed-Tech's `hom-ham`](https://github.com/Inversed-Tech/hom-ham): `f = t·f′+1`,
`h = t·g·f⁻¹`, `c = e₁ + h·e₂ + Δ·m`, multiplication as an exact integer product
rescaled by `t/q`, and no relinearization, no bootstrapping and no evaluation
keys at all — a product ciphertext simply decrypts under `f²`. It is where the
*construction* in II.3 was first worked out, and at `n = 2048` it was genuinely
the cheapest arithmetic available here: a **1.0 ms** packed ct×ct matvec against
BFV/BGV's 5.4 ms.

**It was removed because our own analysis concluded it is insecure at the
parameters this construction needs — and depth-1 does not rescue it.**
LTV/YASHE-family NTRU at these shapes is **overstretched** (Albrecht–Bai–Ducas,
CRYPTO 2016; Kirchner–Fouque 2017; Ducas–van Woerden, "NTRU Fatigue",
EUROCRYPT 2021). One might hope the minimal single-multiply regime stays below
the fatigue point. It does not:

- Fatigue point at `n = 2048` (ternary secret) is `q ≈ n^2.484 ≈ 2^19`;
  generously adjusting for the large LTV secret `f = t·f′+1` (‖f‖ ~ t·√n) by
  ~`t² = 2^40` raises it to ≈ **2^59**.
- One multiply with a *usable* plaintext modulus (`t ≈ 2^20`, needed to hold NN
  inner-product results) already forces **`q ≳ 2^84`** by the noise budget.
- 2^84 is ~25 bits above the adjusted fatigue point, and the port's 2^120 was
  ~60 above. Closing the gap by raising `n` needs `n ≈ 2^35`; lowering `q` below
  fatigue is incompatible with one multiply at `t ≈ 2^20`. The two constraints
  cannot both hold at any practical ring dimension.

(That was a back-of-envelope estimate rather than a full run of the DvW
estimator, but the margin is far too wide for the conclusion to flip.)

**What replaced it.** The construction is scheme-agnostic, so every pipeline was
ported to **BFV** — RLWE, no NTRU lattice, `n = 4096, q = 2^100` inside the
128-bit bound — which supports the identical skip-relinearization,
decrypt-under-`s²` trick. `bench-cnn` moved first; `bench-deep-nn`,
`demo-private-mnist`, `bench-hybrid-nn`'s fully-private strategy and
`bench-fhe-innerprod` followed, and the scheme was deleted from the tree. Every
result transferred **bit-exact**; only the FHE column got slower.

**The one finding worth keeping from it**, since it can no longer be reproduced
in-tree: the ~5× RLWE-vs-NTRU gap was **structural, not an artifact**.

| | ring dim | ring mults per ct×ct | transforms per prime |
|---|---|---|---|
| SI-NTRU | 2048 | 1 (ct is **one** ring element) | 2 fwd + 1 inv |
| BFV / BGV | 4096 | 3 (ct is **two** ring elements) | 4 fwd + 3 inv |

~2× from the ring dimension and ~2.3× from the tensor ⇒ ≈5×, which is what the
measurements showed (1.0 ms vs 5.3–5.4 ms). No choice of RLWE encoding closes
it — and the single-ring-element structure that bought the 5× is the same
structure that makes NTRU overstretched. **That factor is the price of
security**, and the way to pay less for it is faster ring arithmetic (II.2), not
a different scheme.

The security cost was visible end to end and **conv packing more than paid for
it**: at 28×28 the secure BFV pipeline ran in 18.7 s against 94.0 s for the old
*insecure* Toeplitz-on-SI-NTRU path — 5× faster *and* actually private.

### VII.2 The crossover, re-measured on BFV — and gone

The old sweep found a real hybrid-vs-pure-MPC crossover: parameters ≳ 4 K with
few outputs, bandwidth-gated, flipping back to MPC at 50 Mbps. It was measured
on the removed engine, so it was withdrawn when the engine went. **Re-measured
on `hybrid-bfv` (2026-07-30), the crossover does not exist anywhere in the
range** — pure MPC wins at every shape and every bandwidth tried.

`baru bench-hybrid-nn --strategies mpc,hybrid-bfv --rtt-ms 20 --bandwidth-mbps 10`:

| layer (params) | mpc | hybrid-bfv | winner | (old, SI-NTRU) |
|---|---|---|---|---|
| 8×2 (16) | **224 ms** | 868 ms | mpc | mpc |
| 512×4 (2 K) | **516 ms** | 874 ms | mpc | mpc |
| 1024×4 (4 K) | **769 ms** | 1.36 s | mpc | *hybrid* |
| 1024×8 (8 K) | **1.28 s** | 1.94 s | mpc | *hybrid* |
| 1024×16 (16 K) | **2.29 s** | 3.28 s | mpc | *hybrid* |
| 1024×32 (32 K) | **4.29 s** | 5.69 s | mpc | — |
| 1024×64 (64 K) | **8.33 s** | 11.0 s | mpc | — |
| 2048×32 (64 K) | **7.77 s** | 15.5 s | mpc | — |

And it is not rescued by a thinner pipe, which is where the old finding said the
hybrid's advantage lived (layer 1024×32, rtt 20 ms):

| bandwidth | mpc | hybrid-bfv |
|---|---|---|
| 10 Mbps | **4.29 s** | 5.69 s |
| 5 Mbps | **7.95 s** | 10.8 s |
| 2 Mbps | **18.8 s** | 26.2 s |

**Why it flipped, and the methodological point.** The hybrid at 1024×32 /
10 Mbps spends **3810 ms of 5685 in `to-mpc`** and 1283 ms in `link-io` —
i.e. two thirds of it is *ciphertext transfer*, against 49 ms of actual linear
algebra. A BFV ciphertext at `n = 4096` is **131 KB** (two ring elements),
against the removed scheme's 33 KB (one ring element at `n = 2048`), and the
packing bound forces `⌈out·in/n⌉` of them — 11 weight chunks here, each needing
three mask ciphertexts on the switch. So the hybrid moves *more* bytes than pure
MPC at these shapes, not fewer.

That is worth stating carefully, because the old claim was not wrong so much as
**partial**: "19× fewer messages, 46× fewer bytes" counted the *MPC segment*
only. That segment really is tiny and still is — 15 messages and 153 KB against
MPC's 135 372 and 4.5 MB. But the ciphertexts crossing the link are charged by
`sleep_link` and were never in that comparison, and at a secure ring dimension
they dominate. **Quote the MPC-segment footprint as what it is — the footprint
of one phase — not as the hybrid's wire cost.**

Two things moved between the old table and this one, and the measurement cannot
cleanly separate them: pure MPC got ~2.3× faster at 1024×16 (the batching work
in Part I), and BFV's ciphertexts are 4× larger than the removed scheme's. Both
push the same way.

**One modelling caveat, stated rather than fixed.** The masks are precomputed
under `--precompute`, but only their *encryption* is moved offline — the
harness still charges their transfer online, which is that 3810 ms. A
deployment that precomputes masks would also pre-*ship* them, which would move
most of the hybrid's cost off the critical path here. Changing the model is a
real decision about what the offline phase is allowed to include, so it is
flagged rather than assumed. It would not change the conclusion against
SHE-preprocessed MPC (VI.5), which does not depend on it.

### VII.2b The `bench-cnn` security-cost table, withdrawn

The side-by-side table (FHE 222 → 1108 ms at D=8 etc.) compared the insecure and
secure engines directly and is no longer runnable. Its conclusion — the upgrade
costs ~4× on the FHE side — stands as the reason the numbers in III.7 and IV.3
differ.

### VII.3 Other honest negatives

- **PIR loses to downloading the database** by ~13×, with no crossover in `N`
  (VI.3).
- **hybrid-TFHE does not pay**: TFHE's linear phase dominates and the conversion
  costs more than its native PBS ReLU (III.3).
- **Faster polynomials do not bring FHE close to MPC** (III.4). The hybrid's
  case is the communication profile, not speed.
- **The hybrid does not beat pure ABY3** at any shape or bandwidth measured
  (VII.2), and does not beat SHE-preprocessed MPC either (VI.5). Both are direct
  qualifications of our own earlier claims.
- **`mpcc::cost`'s "LowGear is compute-bound" note was wrong** (VI.4), and was
  replaced by a measurement rather than softened.
- **The abstraction was not free** until the gadgets were rewritten over batched
  primitives: 9–45× wall-clock on a 64-wide layer, invisible to per-op timing
  and byte counts (I.4).
