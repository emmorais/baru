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

### I.2c `b2a` returned a share tagged **one bit wide**

Found 2026-08-21 by writing the MedMNIST CNN as a Baru program
(`examples/medmnist.baru`) and getting a *wrong diagnosis* out of the private run
while the plaintext run was correct.

`Backend::b2a` recomposes `nbits` bit-shares into one arithmetic share. It set
the result's `ShareRef.nbits` from **`bits.first().nbits`** — the width of a
*bit*, i.e. `1` — instead of `bits.len()`. Both backends had it
(`Aby3Simulator` and `TransportBackend`). Nothing about the *value* was wrong:
`(reconstruct (b2a (a2b (share 12345 32) 32)))` returned 12345, and the
reconstruct path consults `nbits` only for fixed-point shares.

What was wrong was every **signed comparison downstream**, because `lt`/`le` size
the two's-complement comparison as `a.nbits.max(b.nbits)`. One mistagged operand
was survivable — the other operand's 64 still won the `max`. Two mistagged
operands were not: the comparison ran at one bit and decided `a > b` on the
parity of the low bit.

`shift_right` is `a2b` → drop bits → `b2a`, so `ash-right` on a share returned a
1-bit-tagged share. That makes the failing pattern exactly **fused ReLU∘rescale
followed by a max-pool** — compare two rescaled activations — which is the shape
of every CNN in this repository:

```lisp
(let ((a (* (ash-right (share 5000 64) 8) (>= (share 5000 64) 0)))   ; = 19
      (b (* (ash-right (share 1000 64) 8) (>= (share 1000 64) 0))))  ; = 3
  (reconstruct (> a b)))            ; 0 before the fix, 1 after
```

Three things worth keeping.

**It was invisible to 553 unit tests, and that is not their fault.** Every
existing MPC test either compares a *freshly shared* value against a derived one
(the `max` picks the good width) or reconstructs without comparing. The gadget
suite exercises `gadgets::relu`/`maxpool2d` on the Rust batched path, which sizes
its own widths and never routes through `b2a`. Reaching this needed a program
that composes `ash-right` with `>` twice in a row, which only the Lisp CNN does.

**The failure was silent and plausible.** A wrong pool output is a wrong
diagnosis, not a crash: the private run returned a confident answer that
disagreed with plaintext. It was caught only because the Baru program is checked
against `CnnModel::forward` image by image (`tests/medmnist_baru.rs`) — the
argument for keeping two independent implementations of one model.

**Bisection was by layer checksum, not by debugger.** Reconstructing `vsum` of
each feature map located it in one pass: conv1 agreed (154 588 both ways), the
pool did not (42 544 plaintext vs 38 625 private). Any per-layer invariant would
have done; none existed for the Lisp path before this.

### I.2d Public × secret was paying a communication round

Same day, same program. `LANGUAGE.md` §6.2 has always said scalar multiplication
is "local on shares, no communication" — the interpreter's `*` did not implement
it. `share_mul_typed` injected the public operand as a share (`share_operand`)
and called the interactive `Backend::mul`, so `weight * pixel` with a *public*
weight cost one resharing round.

`crypto::arith::scalar_mul` and `aby3_protocol::scalar_mul_party` already existed
and are local; nothing reached them from the language. `Backend::mul_public` now
does, with a default that falls back to inject-and-multiply so a backend that
cannot do it locally still computes the right answer. Restricted to the
all-integer case on purpose: a Q16.16 operand needs the existing scale-lifting
and post-rescale, and duplicating that is how two paths drift apart (I.4).

Why it survived: with public weights this is *every multiplication in a
convolution*, and no test measured a convolution from Lisp. The Rust gadgets
never hit it because `precompile::mpc_matvec` handles the public-weight case
directly.

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

> [!IMPORTANT]
> **The NTL backend was removed on 2026-08-03.** Everything below is the record
> of what it was for and what it taught us, kept because the Shoup-preconditioned
> butterfly — the single biggest win on this path — came directly from profiling
> it. It is written in the present tense of the time; read it as history.
>
> Why it went: once the NTT led NTL by 1.8× at every size, the backend was a C++
> dependency (NTL + GMP + a `cc` build step) whose only remaining job was
> differential validation. That job is covered by `fhe::ntt`'s schoolbook tests,
> which run at **full ring dimension and ±2⁹⁹ coefficients** in 0.3 s — so the
> claim below that "the O(n²) oracle is impractical" at those sizes no longer
> holds, and had probably stopped holding once the NTT itself got fast. The
> `--features ntl` invocations below will not work; `cpp/`, `src/fhe/ntl.rs` and
> `build.rs` are gone, and the default build now needs no C++ toolchain at all.

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
(`baru bench-fhe-innerprod --reps 15`, packed 8×4 matvec, both
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

> [!IMPORTANT]
> **The CKKS backends were removed on 2026-08-03, and `bench-ckks-compare` with
> them.** This section is the reason — and the reason it was worth doing anyway.
>
> The finding stands: approximate comparison on CKKS is *possible*, contradicting
> our own earlier doc comment, and the measurement below is what established that.
> It is also what condemned it. Seconds per comparison with real error, against an
> MPC comparison in milliseconds and bit-exact, is not a trade this pipeline can
> take — and the backend carried 3.4 MB of vendored SEAL, a cmake/MSVC/bindgen
> build, and a Windows helper script for a header the modern SDK stopped shipping.
>
> One thing was kept, because a live claim depends on it. The **level cost** of the
> polynomial sign approximation grounds `bench-deep-nn`'s competitor-bootstraps
> model — the baseline the zero-bootstrap headline is measured against (§III.2).
> That arithmetic now lives in `fhe::leveled_relu_cost`, pure integer, no
> dependency, with the figures pinned by a test precisely because the apparatus
> that produced them is gone. The numbers below cannot be re-derived in-tree.

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

> [!NOTE]
> `--poly all` no longer exists. The runtime polynomial-backend switch it swept
> was removed on 2026-08-03 with the NTL backend that motivated it (§II.2), so
> this experiment cannot be re-run as written. The finding stands on its own —
> swapping the ring arithmetic moved the FHE column and left the conclusion
> unchanged — and is why the switch was not worth keeping once there was nothing
> to swap.

`baru bench-hybrid-nn --poly all` ran the whole private-layer pipeline with
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
| integer accuracy, full test set | **84.78%** (624 imgs) | **82.93%** (17 778 imgs) |
| float accuracy of the same net | 84.78% | 81.03% |
| majority-class baseline | 62.50% | 18.48% |
| published ResNet-18 (28×28) | 0.854 | 0.935 |
| 95% CI on the full-test figure | [81.7, 87.4] | [82.4, 83.5] |
| **AUC, full test set** | **0.9314** | **0.9605** (macro OvR) |
| published ResNet-18 AUC | 0.944 | 0.997 |
| **balanced accuracy, full test set** | **80.64%** | **82.21%** |
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

### IV.2b The same diagnosis written *in Baru* — and what the language costs

Added 2026-08-21. Everything in IV.2 is hand-written Rust calling the backends.
[`examples/medmnist.baru`](../examples/medmnist.baru) is the same trained network
written as a **Baru program**: the convolution, the fused ReLU∘rescale, the
2×2 max-pool, the dense stack and the argmax are Lisp. Only *loading* is a
builtin (`medmnist-model` / `medmnist-image`, LANGUAGE.md §5.8), which hands over
the identical `i64`s the Rust pipeline multiplies — so the two are two
implementations of one model, and `tests/medmnist_baru.rs` pins them together
image by image on all three datasets.

It needed `baru run` (LANGUAGE.md §10.5) to exist at all: the REPL reads one
expression per line, and it needed the trampoline's step budget raised, since a
single 28×28 convolution exceeds the REPL's 1 000 000-frame runaway guard.

**Pneumonia, one diagnosis, Apple M1 Pro. Same weights, same images, every
integer path bit-identical.**

| implementation | what is hidden | per diagnosis | messages |
|---|---|---|---|
| PyTorch float32, batch of 20 | nothing | **0.06 ms** | — |
| NumPy, integer | nothing | **1.53 ms** | — |
| pure Python, integer | nothing | **33.2 ms** | — |
| **Baru program, plain interpreter** | nothing | **2 783 ms** | — |
| **Rust hybrid pipeline** (`bench-medmnist --gc --precompute`) | image **and** model | **307 ms** | 61 |
| **Baru program, 3-party MPC** (`--mpc --private 1`) | the image | **1 216 404 ms** | 864 361 |

Organa and derma on the plain interpreter: **10 476 ms** and **3 985 ms**.
The float32 row is not bit-exact (17/20 against the integer net's 18/20) and is
a batched call, so it is a throughput figure; every other row is one image at a
time.

Four readings, and the ordering of the middle two is the point.

**Interpretation costs ~1 800× more than vectorised integer arithmetic, and the
crypto is not why.** Baru's plaintext CNN is 2 783 ms against NumPy's 1.53 ms and
pure Python's 33.2 ms. A tree-walking evaluator against a bytecode one is most of
that 84× gap to pure Python; the rest is that every scalar is an `Rc`-boxed
`Types`. Nothing about privacy has happened yet at this row.

**The encrypted Rust pipeline is 9× FASTER than the plaintext Baru program.**
307 ms with both the model and the image encrypted, against 2 783 ms with nothing
hidden at all. Privacy is not what makes a private-inference system slow —
*evaluation strategy* is. This is the cleanest statement of that in the
repository, and it only became measurable once the same network existed on both
sides.

**The Lisp private path is 4 000× the Rust one, for a reason already written
down.** 864 361 messages against 61. §I.3 says the cost is the round count and
not the multiplication count, and §6.2 warns that calling the per-element
primitive in a loop pays `O(n)` rounds for nothing. The Lisp program does exactly
that: one `lt` per comparison, ~12 600 comparisons per diagnosis, each ~9 rounds.
The Rust pipeline calls `lt_batch`/`mul_batch` and spends one round per *layer*.
The gap is not the language being interpreted — it is that the batched primitives
`src/gadgets/` is written over are **not exposed to the language**. That is the
concrete next piece of work, and it is a frontend question, not a protocol one.

**The threat models on those two rows differ, and the weaker one is the fast
one.** `--private 1` hides the image and leaves the model public — the setting
most 2-party private-inference papers measure. The 307 ms pipeline hides *both*.
`--private 2` hides both in Baru too and is verified exact
(`tests/medmnist_baru.rs`), but its wall clock on the real transport is **not
measured**: at `--private 1` a convolution's multiplications are share×public and
therefore local (I.2d), while at `--private 2` every one of them becomes an
interactive resharing, and the run did not finish inside the ~20 minutes
`--private 1` took. So the honest comparison across those two rows is
*model-public Lisp* against *model-hidden Rust*, and the Lisp side would only get
slower if the models were matched.

Writing this program is also what surfaced I.2c (`b2a` mistagging its width, a
silent wrong diagnosis) and I.2d (public × secret paying a round). Both were
invisible to 553 unit tests and to every Rust benchmark.

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

**The conditional in that paragraph is where the work is, and it is not free.**
"If each primitive UC-realizes its ideal operation" is a much stronger hypothesis
than "if each primitive is secure": it requires that every value the primitive
protocols open *internally* be simulatable from the ideal operation's output. That
is a finite checklist — there are nine opening sites in the whole artifact — and
**three of the nine needed a protocol change before the checklist could be
completed**:

| protocol | was | now |
|---|---|---|
| `Π_Trunc` | mask sized to the no-wrap bound, so the opened `a+r` correlated with `a` | mask covers the ring, overflow corrected by a wrap bit |
| `Π_SS` (borrow + wrap) | one selector daBit reused for two lifted bits | one selector each |
| `Π_PC` unit sampling | the mask `m` itself was opened to reject zeros, making `Π cᵢ = d/m` public | opens `m·s` for a fresh `s`; `m` stays uniform on `Z_p*` |

The `Π_PC` case is the sharpest: the `cᵢ` carry no randomness of their own, so `m`
was the only blinding, and at ℓ=6 a single predicate class contains **31 distinct
products** — about 5 bits where the predicate is entitled to 1. Since `Π_WA`
compares the wrap mask `x` against the public `r = a + x`, those are bits of the
activation. No correctness test could see it; it was found by writing the
enumeration the composability argument requires. The gadget-level claim is sound
and it transfers security rather than creating it — see `paper-survey` §5 for the
functionality, the opening lemma and the theorem."

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

> [!IMPORTANT]
> **`src/pir/` and `bench-pir` were removed on 2026-08-03.** The result below is
> why: PIR loses to downloading the whole database by ~13×, with **no crossover in
> N** — so there is no database size at which it becomes the right answer here.
> A component that only ever demonstrates its own inapplicability does not need to
> stay in the tree; the measurement does.

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

#### Re-measured once random bits became real (2026-08-17)

Until now `ShareEngine::random_bits` **drew bits locally and charged nothing for
them** — the "fake offline" mode MP-SPDZ uses to isolate an online phase. The
bits are now *generated*: each party contributes a private bit and the shared bit
is their XOR (`x ⊕ y = x + y − 2xy`), so a bit costs one Beaver triple, and at
two parties a whole batch costs one round (`vm::random_bits_batch`). The same
table, re-run:

| width | dealer (lower bound) | semi2k (OT) | **lowgear (SHE)** | hybrid/bfv |
|---|---|---|---|---|
| 64 | 600.6 ms | 709.6 ms | **607.1 ms** | 692.1 ms |
| 1024 | 603.0 ms | 957.9 ms | **624.3 ms** | 692.1 ms |
| 4096 | 610.9 ms | 1.75 s | **679.3 ms** | 733.7 ms |

**The conclusion above survives and the mechanism sharpens.** The ranking is
unchanged, LowGear still beats the hybrid at every width, and dealer is still
flat. What moved is Semi2k, by exactly the ReLU's bit bill: **+59 triples at
every width**, which is `nb + KAPPA = 33 + 26` random bits at one triple each.
At 3136 B/triple that is 185 kB of OT per comparison — against the layer's
**6.8 kB of online traffic at width 64**, so a single comparison's preprocessing
is **27× the entire online phase**.

Two things follow, and the second is the one that matters for a paper.

**The cost model was under-charging a random bit by ~400×.** `mpcc::cost`
charged `random_bits × elem_bytes` — one ring element, 8 bytes — against a real
cost of one triple, ~3.1 kB. The term is now fitted to the measurement
(`vm::the_cost_of_a_random_bit_is_a_triple`), and the direction of the error is
the uncomfortable one: it under-priced *precisely* the phase that decides a
comparison-heavy workload, and it did so in MPC's favour. Any assignment or
crossover this model produced for a program containing comparisons was biased
toward MPC.

**At narrow widths the OT-preprocessed comparison is now a tie rather than a
win.** `bench-layer --widths 1,8` gives semi2k 693.3 / 695.2 ms against the
hybrid's 692.1 — **0.2% and 0.4% apart**, inside the run-to-run spread §7.3
records. The benchmark's derived summary reports "from width 1 onward" because it
tests a threshold; the honest reading is that the two are indistinguishable below
width 64 and FHE pulls clearly ahead by width 1024 (38%). The published crossover
of *width 64* is therefore still the right number to quote, and it is now the
boundary of a **tie region** rather than of a Semi2k win.

`spdz2k` deliberately still deals its bits, and that is disclosed rather than
fixed: the XOR construction is only sound if each contribution really is a bit,
which a malicious party can violate without any MAC detecting it. Its comparison
costs are consequently a lower bound and are not commensurable with the two
measured engines — which is why `RUNNABLE` excludes it and no sweep quotes it.

---

### VI.6 A whole classifier at a matched threat model (2026-08-17)

`baru bench-2pc-nn`. Every private-inference figure elsewhere in this document
comes from the **3-party honest-majority** path. That is not FHE's threat model:
single-server FHE assumes one server and nothing about collusion, so a 3-party
honest-majority system beating it on latency is a consequence of assuming away
the hard part — and it is why every hybrid inference system in the literature
(Gazelle, Delphi, Cheetah) is 2-party. This runs a whole classifier under
**2-party dishonest-majority** MPC, where the comparison is commensurable.
Input, every weight and every bias secret; only the class index revealed;
bit-exact against `QuantizedNetwork::forward` before any cost is quoted.

LAN (0.1 ms RTT, 10 Gbit/s), Q8 fixed point, 32-bit values:

| shape | rounds | muls | decomps | online | offline (semi2k) | offline (lowgear) | warm | cold (semi2k) | cold (lowgear) |
|---|---|---|---|---|---|---|---|---|---|
| 16x8x4 | 56 | 7 119 | 23 | 112.9 KiB | 25.9 MiB | 1.6 MiB | 3 ms | 25 ms | **5 ms** |
| 16x8x8x4 | 74 | 12 023 | 39 | 190.2 KiB | 43.8 MiB | 2.7 MiB | 4 ms | 41 ms | **7 ms** |
| 16x8x8x8x4 | 92 | 16 927 | 55 | 267.5 KiB | 61.7 MiB | 3.8 MiB | 5 ms | 57 ms | **9 ms** |
| 784x32x10 | 110 | 50 523 | 83 | 995.0 KiB | 169.3 MiB | 10.3 MiB | 6 ms | 149 ms | **16 ms** |

Four results, and the last one is the paper's.

**The layers are 1.7% of the multiplications.** At `16x8x8x8x4` the network's own
arithmetic is 288 multiplications of 16 927; the other 98.3% is *inside* the bit
decompositions that the rescales and ReLUs need. Cost tracks **neuron count**,
not multiplication count — 52 of the 55 decompositions are the network's
nonlinear surface and only 3 are the argmax. This is the fixed-point rescale
being as expensive as the nonlinearity, which `mpcc::ir` flags in a comment
(*"the classic MPC gotcha that Baru's Q16.16 rescale runs into"*) and which is
now measured.

**Round depth dominates in every regime measured**, not just on a WAN: 95.5% of
the online phase is round latency even on a 10 Gbit/s LAN (92 rounds × 0.05 ms
against 267 KiB of traffic), and 100.0% on the WAN profile. So the online cost of
2-party fixed-point inference is set by *depth*, and depth is set by the
nonlinearity count.

**Rounds are a function of depth alone.** `128x64x10`, `256x64x10` and
`784x32x10` all cost **110 rounds** — width buys bytes and multiplications and
nothing else, because every decomposition at one dependence level batches into
one. Pinned by `rounds_are_set_by_depth_not_width`.

**Cost is linear in depth, with no discontinuity** — and this is the 2-party
restatement of the repo's zero-bootstrap thesis. Each added 8-neuron layer costs
exactly **+18 rounds and +16 decompositions**, forever. Pinned as a *second
difference of zero* rather than as those constants, so the claim survives a
change to the decomposition's internals.

That last point is where the interesting comparison lives, and it reframes the
axis. §7.1 and the FHE/MPC literature study the crossover in layer **width**. At
a matched threat model the decisive axis is **depth**, because depth is what
exhausts an FHE level budget and forces a bootstrap. HE-LRM (Garimella, Ebel, De
Micheli & Reagen, arXiv 2506.18150 — the Orion group) reports a single CKKS
bootstrap taking **75% of a 24.22 s inference**, and ~20 s on a single-threaded
CPU. MPC pays +18 rounds; FHE pays either nothing or ~20 s, depending on whether
the layer crossed a level boundary. A step function against a straight line is a
more interesting comparison than two straight lines.

**Preprocessing route decides the cold number, by 16×.** LowGear's SHE triples
put `784x32x10` at 16 ms cold against Semi2k's 149 ms — the same conclusion §VI.5
reached at a single layer, now at whole-network scale. Quoting "2-party MPC is
slow because preprocessing" is a statement about OT, not about MPC.

**Scope, stated.** Wall clock is computed from measured counts through a network
profile, not measured on a socket — the same status as `bench-layer`, and
`bench-net` validates the model to 13.6% worst case. `dealer` is printed as a
lower bound and marked, not as an option.

**A correction, and it was my own diagnosis that was wrong.** This section
originally said "the practical ceiling is `mpcc::lower` being **scalar**, so
compilation takes ~50–60 s at 784 inputs while the protocol run itself is
milliseconds", and named IR vectorization as the fix. **Both halves were false,
and neither had been measured.** Profiling the phases at `784x32x10`:

| phase | time |
|---|---|
| generate source | 72 ms |
| parse | 11 ms |
| flatten (`CircuitCompiler`) | 681 ms |
| lower (`mpcc::lower`) | 28 ms |
| schedule | 1 ms |
| **execute, dealer** | 435 ms |
| **execute, semi2k** | **45.8 s** |
| execute, lowgear | 1.02 s |

So compilation was **0.8 s**, not 50; `mpcc::lower` — the thing blamed — was
**28 ms of it**; and the 50–60 s was **Semi2k generating 55 420 real IKNP-OT
triples**. LowGear does the same work in 1.02 s, **45× faster**, which is the
per-triple SHE-versus-OT gap of §VI.5 showing up as wall clock at scale.

#### Three quadratics, all in the one dimension the repo never tested

Profiling further found that flatten *and* execute were both superlinear. All
three causes were quadratic in the **number of variables** — which is exactly the
dimension a private network grows in, one variable per weight, and therefore the
one dimension a repo full of small test programs never exercised.

| where | what it did | effect at `784x256x10` |
|---|---|---|
| `CircuitCompiler::{compile, compile_fhe}` | deduped variables with `Vec::contains` on a growing `Vec` | flatten 36.7 s → 9.3 s (**3.9×**) |
| `CircuitCompiler::flatten_let` | `env.clone()` per `let` — and the parser desugars `let*` into one **nested** `let` per binding, so ~530 clones of a 203 000-entry map | 9.3 s → **0.68 s** (**14×**; 54× overall, and now linear) |
| `vm::execute` | found each input's owning party with `prog.code.iter().find(..)` — 203 000 inputs against 611 694 instructions, ~10¹¹ operations | execute 73.2 s → **0.32 s** (**232×**) |

End to end at `784x256x10`: **112.6 s → 5.25 s, 21×**, and every phase is now
linear in width (doubling the hidden layer roughly doubles each). Guarded by
`the_pipeline_stays_near_linear_in_width`, which asserts a growth *ratio* under 3×
rather than an absolute budget — a reintroduced quadratic shows as ~4×.

The lesson is the one this document keeps relearning: **the bottleneck was named
without being profiled.** "Vectorize the IR" was a plausible story that survived
because nobody timed the phases, and the actual fixes were a `HashSet`, a scope
restore, and a one-pass index — none of them in the IR at all.

**The ceiling moved accordingly.** `784x128x10` — an MNIST-shaped MLP — now runs
bit-exact at **40 ms cold on LowGear** (524 ms on Semi2k, 9 ms warm). SIMD
vectorization remains a legitimate future item for *communication* batching, but it
is no longer what stands between this benchmark and a realistic network.

**One earlier finding needs qualifying at this scale.** "The layers are 1.7% of
the multiplications" holds for the tiny shapes it was measured on; at
`784x128x10` the layers are **55.0%** (101 632 of 184 827), because MACs grow with
width while decompositions grow with neuron count. The benchmark derives the split
per row, so it reports the right number for whatever it ran — but the 1.7% figure
is a small-network statement, not a general one.

---

### VI.8 Where the crossover against Orion must be, from Orion's own table (2026-08-17)

`bench-2pc-cnn` runs LeNet-5 through two-party MPC. The obvious next question is
whether MPC overtakes the state of the art in FHE deep learning, and Orion's
evaluation table (ASPLOS'25, Table 2) answers the *mechanism* more sharply than
our own measurements do. Reproduced here because every claim below reads off it:

| model | params (M) | FLOPs (M) | act. | depth | **boots** | time (s) |
|---|---|---|---|---|---|---|
| MLP (MNIST) | 0.12 | 0.12 | x² | 5 | **0** | **0.29** |
| LoLA (MNIST) | 0.10 | 0.13 | x² | 5 | **0** | **0.23** |
| LeNet (MNIST) | 1.66 | 4.30 | x² | 7 | **0** | **2.93** |
| AlexNet (CIFAR-10) | 23.3 | 188 | ReLU | 109 | 15 | 337.2 |
| VGG-16 (CIFAR-10) | 14.7 | 314 | ReLU | 227 | 28 | 588.6 |
| **ResNet-20 (CIFAR-10)** | **0.27** | **41.2** | ReLU | **287** | **37** | **618.2** |
| MobileNet (Tiny IN) | 3.25 | 47.4 | SiLU | 218 | 42 | 892.4 |
| ResNet-18 (Tiny IN) | 11.3 | 2 260 | SiLU | 138 | 61 | 1 447 |
| ResNet-34 (ImageNet) | 21.8 | 3 670 | SiLU | 267 | 146 | 14 338 |
| ResNet-50 (ImageNet) | 25.6 | 4 110 | SiLU | 395 | 351 | 32 324 |

**Latency tracks bootstraps, not work.** ResNet-20 has **4.6× fewer FLOPs than
AlexNet and is 1.8× slower** — 41.2 against 188 million, 618 s against 337 — because
it is deeper: 37 bootstraps against 15. The same inversion appears again between
MobileNet and ResNet-18. And the boundary at zero is a cliff: every network Orion
evaluates **without** a bootstrap finishes in **0.23–2.93 s**, and every network
**with** one takes **190–32 324 s**, a gap of two to four orders of magnitude with
nothing in between.

The mechanism is stated in their own text. A minimax composite approximation of
ReLU has **multiplicative depth 14** (13 for the sign, 1 for the product); SiLU has
7. So depth accumulates at 7–14 levels *per activation layer*, the level budget is
exhausted after a handful of layers, and the remainder of the network is paid for
in bootstraps. Note also that Orion, like our `lenet5-fhe`, **replaces max pooling
with average pooling** throughout — for the reason §VI.7 gives, that a
comparison-free scheme cannot evaluate a maximum.

This is the same discontinuity §VI.7 found from the other side, where HEIR's
levelled pipeline simply stops at depth 11 instead of bootstrapping. Orion's
contribution is precisely to place the bootstraps automatically so that it does not
stop; the table is what that costs.

#### Where MPC must overtake it — to be measured, not projected

Secret sharing has **no depth budget**. Its cost is linear in depth (18 rounds per
layer, §VI.6) and linear in work, so it has no cliff, and against a system whose
cost is dominated by a cliff it must overtake it somewhere. **Where** is a question
this document will answer by measurement or not at all.

An earlier revision of this section carried an extrapolation from our LeNet-5
figures to ResNet-20, quoting a projected latency for each side. **It has been
removed.** A projected number is not a result, and placing one beside a measured
one invites exactly the instability of conclusions that the rest of this record
exists to prevent. Nothing in this document may quote an MPC-versus-FHE ratio at a
network we have not run.

Three things are required before the question can be answered, and each is
tracked as work and not as a prediction.

1. **Elapsed wall clock over a real transport.** Every MPC latency in §VI.6–VI.7
   is *composed*: rounds and bytes are measured counts put through a network
   model, and compute is measured but divided by the party count, because
   `mpcc`'s engines hold every party's share in one process. That is not a running
   time. The parties must execute concurrently over a transport that really
   waits, as the three-party path already does.
2. **A generator that is not scalar.** LeNet-5 emits 8.9 MB of source and
   636 066 instructions; a network at 146× the multiply--accumulate count would
   emit on the order of a gigabyte, which will not compile in available memory.
   A convolution must emit one instruction of width `out_c · out_h · out_w`.
3. **Residual connections**, which `bench::conv_nn` does not have, and without
   which a deep stack is not ResNet-20.

### VI.7 The FHE arm: HEIR as a baseline, and two gates (2026-08-17)

`baru bench-heir`. **No FHE latency has been measured yet, and this section says
so rather than implying otherwise.** What exists is the harness and its gates,
tested end to end; the measurement needs an environment the development machine
does not have.

**Why HEIR is a baseline and not a target.** HEIR is compiler *infrastructure*.
Its paper (arXiv:2508.11095) claims generality — "support all mainstream
techniques… integrate with all major software libraries and hardware
accelerators" — and reports no private-ML performance evaluation. Beating its
numbers would not be a result. What it buys is an FHE arm running on **tuned C++**
(OpenFHE, Lattigo) instead of on Baru's hand-written NTT, which removes the
confound §1 of the survey concedes — *"absolute figures here are not competitive
with tuned implementations"* — in the one direction that matters, because the
objection to "MPC beat our FHE" is that our FHE was ours. It also replaces
`bench-layer`'s FHE column, which is not an FHE run at all but a per-operation
constant from `mpcc::fhe_cost` extrapolated through a network profile.

**Both arms run the same network.** `bench-2pc-nn`'s fixture is now the single
source of truth (`fixture_for` / `input_for` / `argmax_of`), and the FHE arm
receives it dequantized by the fixed-point scale. A test asserts the float
evaluation reproduces the *quantized* classification, so the two arms are
comparable by construction rather than by coincidence.

**Gate 1 — security parameters.** google/heir#1232 reports an MNIST MLP at
**~30 s and 3 GB with `HEStd_NotSet`** — OpenFHE's "do not check" mode, i.e. no
security level at all — and **~6 min and 26 GB** at the secure ring dimension. A
**12× factor** sits between an insecure and a secure baseline, and the insecure
one is what a casual run produces. `HeirReport::gate` refuses to let a latency be
quoted unless the run reports `HEStd_{128,192,256}_classic`. This extends
`crypto::attestations`' discipline — every engine states what it implements, and
asking for more is an error — to an *external* baseline. **An insecure baseline is
worse than no baseline**: it manufactures a speedup out of a parameter choice.

**Gate 2, and the harder problem it exposed — an exact ReLU is not expressible on
the FHE arm at all.** CKKS cannot compare. The MPC arm computes ReLU exactly. So
the two arms cannot be made to compute the same function by tuning a parameter,
and the two available choices measure *different things*:

* **Both arms compute `x²`.** Exact on each side, and what Orion does for its
  MNIST networks. **This is the route taken**, and the worry about it turned out
  to be unfounded — see the measurement below.
* **Keep the exact ReLU and approximate it on the FHE arm.** Then the polynomial
  degree needs justifying and the accuracy it costs needs reporting — Orion
  measures 2.1% accuracy for 1.77× speed on that axis. Not implemented; `x²` made
  it unnecessary.

`Refusal::DifferentNetwork` refuses to choose, and is checked **before** every
other condition: "the two arms computed different functions" is a stronger
objection than any disagreement in their outputs, and a class disagreement is
exactly how it would otherwise present — misread as approximation error. Beneath
it, `Refusal::Disagrees` still requires the FHE arm to reproduce the plaintext
classification, because timing an approximation that gets the answer wrong is not
a comparison.

#### `x²` costs the MPC arm nothing, measured

The worry was that switching to `x²` would quietly improve the MPC arm and make
the comparison flattering: over plain integers `x²` is one multiplication where
ReLU needs a masked-opening bit decomposition. **In fixed point it is not.**
Squaring a `Q(s)` value doubles its scale, so the product needs a rescale — and a
rescale is the same bit decomposition the comparison was. Two per neuron either
way.

`bench-2pc-nn --activation relu` against `--activation square`, LAN, both
bit-exact against their own plaintext reference:

| shape | rounds | muls | decomps | online | offline (semi2k) | cold (semi2k) |
|---|---|---|---|---|---|---|
| 16x8x4 | 56 | 7 119 | 23 | 112.9 KiB | 25.9 MiB | 25 ms |
| 16x8x8x8x4 | 92 | 16 927 | 55 | 267.5 KiB | 61.7 MiB | 57 ms |

**Both activations produce these numbers identically** — not approximately, digit
for digit, on every column and every protocol. Pinned by
`the_fhe_friendly_activation_costs_the_mpc_arm_nothing`.

Two consequences. The comparison against a CKKS arm is now available **at zero
cost to the MPC side**, so every finding in §VI.6 carries over unchanged rather
than needing re-measurement. And the result is worth reporting on its own: the
"FHE-friendly" activation, chosen in the FHE literature precisely because it
avoids a comparison, buys a *fixed-point* MPC implementation nothing at all. What
would make it cheaper is folding the doubled scale into the next layer's rescale
(shift by `2s` there instead of `s`), which removes the extra decomposition at the
cost of wider intermediates. Not done; the measured equality is reported instead.

**What it took to get a number, all three found by measurement.**

1. *`heir-py` alone cannot evaluate anything.* `heir.compile()` resolves an OpenFHE
   **C++ install** — headers under `/usr/local/include/openfhe`, or the
   `OPENFHE_INCLUDE_DIR` / `OPENFHE_LIB_DIR` / `OPENFHE_LINK_LIBS` variables — and
   otherwise fails with *"Could not build development config. Did you run
   bazel?"*. HEIR emits C++ and compiles it per program, so headers are a hard
   requirement. The PyPI `openfhe` wheel does not satisfy it (Python bindings, not
   a toolchain), there is no Homebrew formula, and `heir.backends.cleartext` is not
   a shortcut — its `eval` forwards straight to the original Python function and
   says so in a warning. OpenFHE 1.5.1 was therefore built from source (needed
   `brew install libomp`; Apple clang does not ship `omp.h`, and the emitted C++ is
   linked with `-L$OPENFHE_LIB_DIR` only, so libomp has to be reachable from
   there).
2. *The decorator frontend is not the ML path.* It runs the function through
   numba, and a dense layer is not expressible there: on 2026.8.1, `x[0]` raises
   "Unsupported IR Element: getitem", `@` raises "Extension handling for
   non-integer (e.g., floats, tensors) types is not yet supported", and
   `np.sum`/`np.matmul` are unsupported. Elementwise tensor ops *do* work. So the
   driver emits **`linalg` MLIR** and uses `compile(mlir_str=...)` — the level
   HEIR's ML pipeline actually targets, which the Torch route reaches via
   torch-mlir. Going direct skips torch entirely. Conventions that took
   measurement to pin: the attribute is `{secret.secret}`, the generated encryptor
   is named after the *argument* (`encrypt_arg_0`), and `arith.addf`/`arith.mulf`
   are elementwise on shaped types.
3. *HEIR never sets a security level.* Its emitted C++ calls only
   `SetMultiplicativeDepth` and `SetPlaintextModulus`, so the level in force is
   OpenFHE's default, which its headers pin at `HEStd_128_classic`. **This corrects
   the framing above**: the `HEStd_NotSet` in google/heir#1232 was a *user's
   explicit choice*, not HEIR's default, so the gate catches a real mistake but not
   one a casual run makes. The driver reads the emitted C++ and reports the level
   *and its source*.

#### The first head-to-head at a matched threat model

Both arms compute the **same exact function** — `x²` activation, same weights,
same inputs — and both classifications were verified against the plaintext
fixed-point reference before any timing was quoted. HEIR → OpenFHE CKKS at
`HEStd_128_classic`, multiplicative depth **3, chosen by HEIR**; keygen excluded
as a per-session cost.

| shape | FHE (HEIR/OpenFHE) | MPC lowgear, LAN | MPC semi2k, LAN | MPC lowgear, WAN | MPC semi2k, WAN |
|---|---|---|---|---|---|
| 6x4x3 | **58.9 ms** | **3 ms** | 15 ms | 2 680 ms | 3 820 ms |
| 16x8x4 | **80.1 ms** | **5 ms** | 25 ms | 3 243 ms | 5 283 ms |

**The crossover is a network crossover, and it is large in both directions.** On a
LAN, SHE-preprocessed 2-party MPC beats HEIR/OpenFHE by **16–20×**; OT-preprocessed
MPC still beats it by **3–4×**. On the WAN profile (50 ms one-way) FHE wins by
**41–45×**, because MPC pays 47–56 rounds of pure latency while FHE is
non-interactive and network-insensitive. That is the shape Huang, Maeng & Suh
report in *Beyond Latency*, now reproduced in one harness with both arms measured
and the function held identical.

**Caveats, and they matter.** (i) The two wall clocks are not measured the same
way: FHE's is a real elapsed time, MPC's is **computed** from measured counts
through the network profile, so the MPC column inherits the model's 13.6%
worst-case error and has no compute term for a real socket. (ii) FHE's figure
excludes the ciphertext upload/download, which on a WAN is not free — the pybind
wrapper does not expose ciphertext size, and that is reported as unavailable
rather than as zero. (iii) OpenFHE was built with OpenMP, so its evaluation may use
multiple cores, while the MPC counts are core-agnostic. (iv) These networks are
tiny — `16x8x4` is 160 MACs against LeNet-5's 281 640 — so the *ratios* are the
result and the absolute numbers are not comparable to Orion's 0.23 s MNIST or
HE-LRM's 24 s DLRM. Scaling up is blocked on `mpcc::lower` being scalar, not on
either protocol.

#### At MNIST scale, with a baseline that is not a strawman

The comparison above is on toy shapes. Repeated at **MNIST-shaped** networks —
784 inputs, 10 classes — with `x²` on both arms, both classifications verified,
CKKS at `HEStd_128_classic`:

| shape | FHE eval | FHE + keygen | MPC LAN warm | MPC LAN cold | MPC WAN cold |
|---|---|---|---|---|---|
| 784x32x10 | **212.5 ms** | 322.9 ms | **20.4 ms** | 318.8 ms | ~7.3 s |
| 784x128x10 | **490.3 ms** | 631.2 ms | ~30 ms | ~0.9 s | ~9.8 s |

**On a LAN with preprocessing hoisted on both sides, MPC wins by ~10–16×; cold
against cold the two are within ~1.0–1.4× at this depth.** On the WAN profile FHE
wins by ~20–30×, because MPC pays 110 rounds of latency and FHE is
non-interactive. Both margins move with depth (see the sweep below) and with width,
because CKKS packing amortizes width — 32× the MACs for 2.3× the time — while MPC's
traffic does not.

**The LAN figures include local compute.** An earlier version of this section
omitted it entirely and reported 13× and 12×; the correction is documented under
the depth sweep below. The WAN figures are latency-dominated and moved far less.

**The baseline is credible, and that is what makes the claim mean anything.**
Orion (ASPLOS'25) reports MNIST at **0.23 s** single-threaded; this harness gets
**0.21–0.49 s** for MNIST-shaped MLPs through HEIR → OpenFHE. Same ballpark as the
published state of the art, so "MPC beats it by 12–13× on a LAN" is a statement
about MPC against competent FHE, not against a strawman — which was the entire
reason for wiring HEIR in rather than using Baru's own BFV.

It is also **far** better than the only published HEIR ML datapoint:
google/heir#1232 reports 784→512→10 at ~30 s with insecure parameters and ~6 min
secure. The difference is not our doing — it is the route. That write-up went
through the numba decorator frontend with a padded 1024×1024 layer; this goes
through `linalg` MLIR, which is the level HEIR's ML pipeline is actually built
for. Worth reporting back upstream.

**What is still not comparable to the literature**: these are MLPs with `x²`,
not CNNs with polynomial ReLU, and depth 3 rather than the depth that forces
bootstrapping. Orion's 0.23 s is a different network. The honest comparison is of
*ratios within this harness*, with the absolute numbers offered only as evidence
that neither arm is incompetent.

#### The depth sweep, and FHE's ceiling arrives before the crossover does

`baru bench-heir --activation square --shapes 6x4x3,6x4x4x3,...` against
`baru bench-2pc-nn` on the same shapes. Every FHE row verified against the
plaintext reference; every MPC row bit-exact. Depth is chosen by HEIR.

| hidden layers | shape | FHE depth | FHE eval | MPC rounds | MPC lowgear LAN | MPC lowgear WAN |
|---|---|---|---|---|---|---|
| 1 | 6x4x3 | 3 | **47.8 ms** | 47 | **3 ms** | 2 680 ms |
| 2 | 6x4x4x3 | 5 | 164.1 ms | 65 | 5 ms | — |
| 3 | 6x4x4x4x3 | 7 | 263.5 ms | 83 | 6 ms | — |
| 4 | 6x4x4x4x4x3 | 9 | **389.7 ms** | 101 | **8 ms** | 5 827 ms |
| 5 | 6x4x4x4x4x4x3 | 11 | **SIGSEGV in key generation** | 119 | 9 ms | 6 876 ms |

Each added `x²` layer costs **two levels**, and HEIR gets the depth right
automatically every time. Three results.

**FHE has a hard ceiling, not a bootstrap step.** At depth 11 the run dies with
**SIGSEGV, inside `f.setup()` — key generation** — after HEIR's MLIR→C++→clang
pipeline succeeded. Not OOM, not a clean "no secure parameters" abort: a
segmentation fault in OpenFHE 1.5.1 for a context that sets only
`SetMultiplicativeDepth` and `SetPlaintextModulus`. **HEIR's default pipeline is
leveled and places no bootstraps**, so depth is a wall rather than a step. That
reframes what Orion's automatic bootstrap placement *is*: not an optimization, but
the thing that makes depth > 9 possible at all — at ~20 s per bootstrap (HE-LRM).

**On a LAN, MPC's advantage widens with depth.** FHE/MPC is 16×, 33×, 44×, 49×
across the four depths that run. MPC is linear (+18 rounds and +2 ms per layer,
forever); FHE grows ~110 ms per layer *and* stops.

**On a WAN, FHE wins at every depth it can run — and that is the interesting
part.** FHE 48–390 ms against MPC 2 680–5 827 ms, an advantage narrowing from 55×
to 15× as depth grows. Extrapolating the two slopes (~110 ms/layer against
~900 ms/layer) the crossover is far away — but **FHE's ceiling arrives first**. So
the honest statement is not "they cross at depth *d*": it is that **on a WAN,
leveled FHE is faster until it stops working, and the depth where it stops is
below the depth where MPC would have caught it.** A system that bootstraps to get
past that ceiling pays ~20 s, which is 3× MPC's WAN total and 2000× its LAN total
at depth 11 — so **the bootstrap is exactly where MPC wins on both axes at once.**

That is the sharpest form of this repo's zero-bootstrap thesis, arrived at from
the FHE side: the scheme switch that refreshes a ciphertext costs 160 ms (E11), and
the alternative is either a ~20 s bootstrap or a segfault.

**Caveat on the ceiling**: depth 11 fails *in this configuration* — HEIR's default
CKKS context, OpenFHE 1.5.1, no explicit ring dimension or scaling technique, no
`heir_opt_options`. It is a limit of the default pipeline, not a proof about CKKS,
and a hand-tuned context or HEIR's bootstrapping passes would move it. What is
*measured* is that the out-of-the-box path stops there, and the deepest network it
evaluates is four hidden layers.

#### Repeated at MNIST width, with compute charged

`x²` on both arms, Xavier-scaled weights, every row verified. FHE is HEIR →
OpenFHE CKKS at `HEStd_128_classic`, depth chosen by HEIR. MPC is LowGear on a LAN
(0.1 ms RTT, 10 Gbit/s). **Both sides are split warm/cold**: FHE's key generation
and MPC's correlated randomness are both per-session, query-independent costs, so
warm-vs-warm and cold-vs-cold are the two honest comparisons.

| depth | shape | FHE eval | FHE + keygen | MPC warm | MPC cold | warm | cold |
|---|---|---|---|---|---|---|---|
| 3 | 784x32x10 | 212.5 ms | 322.9 ms | **20.4 ms** | 318.8 ms | 10.4× | **1.0×** |
| 5 | 784x32x32x10 | 762.2 ms | 1 159.9 ms | **26.6 ms** | 454.2 ms | 28.7× | 2.6× |
| 7 | 784x32x32x32x10 | 1 230.5 ms | 1 681.4 ms | **31.2 ms** | 588.0 ms | 39.5× | 2.9× |
| 9 | 784x32x32x32x32x10 | 1 775.5 ms | 2 327.2 ms | **37.4 ms** | 723.2 ms | 47.4× | 3.2× |
| 11 | +1 hidden layer | ✗ **SIGSEGV in keygen** | — | **42 ms** | 858 ms | — | — |

**Warm, MPC wins by 10× at depth 3 rising to 47× at depth 9. Cold, the two are a
dead tie at depth 3 (1.0×) and MPC pulls to 3.2× by depth 9.** Then FHE stops
working while MPC continues linearly. The direction — MPC's advantage widening with
depth — holds on both accountings; the magnitude depends entirely on which one you
quote.

**These numbers replace an earlier version that was wrong, and the error is worth
stating.** `bench-2pc-nn` charged MPC `rounds × RTT + bytes / bandwidth` and **no
compute term at all**. On the WAN profile that is nearly harmless — 110 rounds at
50 ms swamps everything — but on a LAN the network term is *microseconds*, and
compute is the entire cost. Measured at `784x32x10`: modelled network **16 ms**
against **598 ms** of real local compute. So the published LAN figures understated
MPC by ~19×, and the ratios (13×–57×) additionally compared a *warm* MPC against a
*cold* FHE. Both halves flattered MPC. Compute is now measured and split:

* `offline_micros` times correlated-randomness generation inside the engines, so it
  can be charged to the cold column only — the same treatment `offline_bytes` and
  `random_bit_bytes` already had.
* Per-party compute is the in-process total divided by the party count. The
  simulator runs every party serially in one process, so this is a **division, not
  a measurement of one party**; the protocols are symmetric, but it is an
  approximation and is labelled as one.
* The first attempt double-counted: `random_bits_batch` timed its whole span
  including the `mul_batch` calls that trigger the engine's own refill, which times
  itself. The residual saturated to zero and reported the online phase as free.
  Fixed by subtracting the nested delta.

Two internal checks say the split is sound. **Online compute is identical across
protocols** — 14.3 ms (Semi2k) against 14.1 ms (LowGear) at depth 3, 26.2 against
26.7 at depth 9 — which is exactly right, because the two share the same Beaver
online phase and differ only in where triples come from. And **offline compute
separates them by 78×**: 22.8 s against 293 ms at depth 3, the OT-versus-SHE gap of
§VI.5 appearing in CPU rather than bytes. Semi2k on a LAN is 22.9 s cold, which is
the honest verdict on OT preprocessing at this scale.

#### The fixture had to be fixed first, and that produced a withdrawn finding

The first run of this sweep used **unscaled** fixture weights, and `x²` failed at
depth 9 on *both* arms — the FHE gate refused because the CKKS error (5.3 × 10⁸)
exceeded the reference's top-two class margin (1.9 × 10⁷), and every MPC protocol
disagreed with the plaintext reference. It was written up as "the depth ceiling is
the activation, not the crypto", with an inversion in which MPC's exact ReLU is what
makes depth possible. Consistent with both measurements, and **wrong**.

The caveat flagged beside it — that an *untrained* `x²` fixture might be the cause —
was the entire cause. A dot product over `n_in` inputs has gain ~`√n_in·|w|`, and
squaring turns any gain above 1 into doubly exponential growth. Worst accumulator
against the `2^32` a 33-bit masked decomposition admits:

| hidden layers | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| unscaled `\|w\| ≤ 0.625` | 10^5.5 | 10^6.7 | 10^9.0 | **10^13.5** | 10^22.6 | 10^38.2 |
| Xavier `\|w\| ~ 1/√fan_in` | 10^3.9 | 10^3.9 | 10^3.9 | 10^3.9 | 10^3.9 | 10^4.0 |

Dividing by `√fan_in` — what He/Xavier initialisation does, and what training
produces — puts the gain at the fixed point where `x²` is stable, flat from one
hidden layer to eight. The fixture is now scaled that way, and
`the_fixture_stays_in_range_as_depth_grows` asserts both failure directions: the
accumulator under `2^32`, and the logits not decayed to zero. **No cost figure moved**
— rounds, bytes and decomposition counts depend on program shape, not weight
magnitude — only correctness at depth.

Worth keeping from the wrong version: exact ReLU really is better behaved than `x²`
(1-Lipschitz and non-expansive; it reached seven hidden layers on the unscaled
fixture where `x²` died at four). It just is not what limits depth once the weights
are sane. And both gates caught their own arm rather than a reader having to.

---

### VI.9 Elapsed wide-area time, measured — and two optimizations that follow from it (2026-08-18)

*Probes:* `examples/probe_wan.rs`, `examples/probe_rounds.rs`, `examples/probe_accuracy.rs`.
*Engine:* `src/mpcc/dist.rs` — two parties on separate threads over
`bench::comm::LatencyTransport`, which really sleeps. Every figure below is
`max` over the two parties of a wall-clock interval measured with
`Instant::elapsed`. **Nothing in this section is composed from a cost model.**

#### Why this section exists

Every wide-area figure recorded before it was assembled: a round count from the
scheduler, a byte count from the engine, and a network profile multiplied together.
That is a prediction, and §VI.5–VI.8 record three occasions on which a prediction of
ours was wrong in a direction we did not anticipate. `dist.rs` removes the
composition: the two parties run concurrently, each message is really waited for, and
elapsed time is read off a clock.

The first thing this bought was a validation. On the `784×32×10` classifier the cost
model predicted 110 rounds and the distributed engine spent exactly 110 exchanges. The
model's *round arithmetic* was right; what it could not tell us was which term of
`compute + bytes/bandwidth + rounds × RTT` dominates, and that turned out to decide
what to optimize.

#### The three terms, separated by measurement

The same program was run four times per configuration: at unbounded and at 100 Mbps
bandwidth, and at 0 ms and 25 ms one-way latency. The differences are themselves
measurements, so no term is modelled.

| workload | exch | bytes | compute | +bandwidth | +latency | both |
|---|---:|---:|---:|---:|---:|---:|
| one ReLU | 11 | 5 KiB | 0 ms | 1 ms | 317 ms | 320 ms |
| `16×8×4` | 47 | 112 KiB | 2 ms | 13 ms | 1347 ms | 1341 ms |
| `784×32×10` | 65 | 892 KiB | 17 ms | 102 ms | 1872 ms | 1936 ms |
| LeNet-5 (FHE-friendly) | 119 | 47.9 MiB | 254 ms | 4408 ms | 3631 ms | 7637 ms |

Read the last two rows against each other, because they disagree. On the classifier,
bandwidth costs 85 ms and latency costs 1855 ms: **latency is 96% of wide-area time**,
and one could afford roughly forty times the traffic before bytes began to matter. On
LeNet-5 that headroom is gone — bandwidth (4.2 s) exceeds latency (3.4 s), and the two
terms are very nearly additive rather than overlapped, so neither hides behind the
other.

So "MPC is latency-bound" is *workload-dependent*, and the two workloads want opposite
optimizations. Both were then done, and both were measured.

#### Optimization 1 — argmax as a tournament, not a fold

`probe_rounds.rs` attributes each scheduled round to the deepest protocol it contains.
On the classifier, ten of twenty-five rounds were charged to comparison — and the
network's own definition contains no comparison at all. They came from the argmax.

The argmax had been emitted as a left fold carrying the running best: `n−1`
comparisons, which is optimal, and what a plaintext implementation would write. Under a
secret-sharing protocol it is the wrong program, because each comparison reads the
previous one's output and a comparison is a bit decomposition followed by a
logarithmic-depth borrow chain, several rounds deep. The fold's *sequential depth* is
`n−1` in units of that subprotocol.

A single-elimination tournament performs exactly the same `n−1` comparisons — the
arithmetic is untouched — but makes each bracket's comparisons mutually independent, so
the scheduler collapses each bracket into one round. Depth falls from `n−1` to
`⌈log₂ n⌉`: four instead of nine for ten classes.

`src/bench/depth.rs`, with `bench::depth::argmax_tree`. Measured, at 25 ms:

| workload | exchanges | elapsed | |
|---|---:|---:|---|
| `784×32×10` fold | 110 | 3228 ms | |
| `784×32×10` tournament | 65 | 1931 ms | **1.67×** |

Same classification on every input, and `every_protocol_classifies_exactly` still
passes bit-exactly against `QuantizedNetwork::forward`. This is a free 1.67× — no
approximation, no extra assumption, purely the shape of the dependence graph.

The same module also balances the dot-product and convolution-tap sums. Addition does
not communicate, so this buys no rounds; it is there because a left fold over 784 terms
is a 784-deep expression that the compiler front end recurses through, which is what
forced LeNet-5 onto a 512 MiB spawned stack.

#### Optimization 2 — rescale with a truncation pair, not a bit decomposition

The byte figure for LeNet-5 could not be explained by the network: it has about 0.28 M
multiply-accumulates and the engine reported **3.12 M multiplications**. Counting the
random bits located the other 91%: `553,833` bits, about 59 per rescale, two rescales
per neuron. In an *FHE-friendly* LeNet-5 — average pooling, `x²` activation, no
comparison anywhere in its definition — nine tenths of the traffic was fixed-point
rescaling.

That is the structural gap against an approximate homomorphic scheme, which rescales as
a ciphertext-level operation for nothing. The fix is a construction from the
secret-sharing literature that we had not implemented: mask the value with `r` drawn
uniformly over `2^{k+κ}`, open, shift the *public* result locally, and subtract the
pre-shifted share `⌊r/2^m⌋`. One opening, no multiplication.

The reason it is worth stating carefully is what makes it cheap. The pair `(r, ⌊r/2^m⌋)`
does not depend on the value being truncated, so it is **query-independent**: a
deployment generates it before the query arrives, in exactly the sense that a
homomorphic deployment generates its evaluation keys before the query arrives. A bit
decomposition of `x` admits no such treatment, because it is a function of `x`. The two
are therefore not merely fast and slow versions of one thing; they sit on opposite sides
of the online/offline boundary.

`vm::trunc_pairs`, opted into per engine via `ShareEngine::trunc_pair_mode` (default
`false`, so no previously recorded figure moved). Measured at 25 ms, 100 Mbps:

| workload | rescale | exch | bytes | muls | elapsed | |
|---|---|---:|---:|---:|---:|---|
| `784×32×10` | decomposition | 65 | 892 KiB | 50,519 | 1936 ms | |
| `784×32×10` | pair | 51 | 694 KiB | 37,835 | 1539 ms | **1.26×** |
| LeNet-5 | decomposition | 119 | 47.9 MiB | 3,121,211 | 7637 ms | |
| LeNet-5 | pair, warm | 56 | 4.66 MiB | 289,055 | **2084 ms** | **3.67×** |
| LeNet-5 | pair, bits inline | 82 | 4.66 MiB | 289,055 | 3922 ms | 1.95× |

Bytes fall 10.3× and multiplications 10.8×, which is the 91% figure recovered from the
other side. Note the last row: with bit generation *not* hoisted the gain is 1.95×
rather than 3.67×, and reporting only the warm number would be the §7.3 error again.

#### What it costs, measured rather than argued

The pair route computes a different function. It returns `⌊x/2^m⌋ + c` where `c` is one
with probability equal to the discarded fraction, so the cheap rescale is *approximate*
and the speedups above are not quotable until that is characterised.

`mpcc::vm::tests::stochastic_rescaling_is_off_by_at_most_one_and_is_unbiased` measures
the primitive over 400 trials at each of six values, and asserts three things: the
deviation is 0 or 1 and never anything else; the carry rate matches the discarded
fraction to within four binomial standard deviations; and when nothing is discarded the
result is exact every time. So the rounding is **unbiased**, where an exact truncation
always rounds toward negative infinity and therefore contributes a systematic drift
once per layer. For fixed-point inference the approximate rescale is arguably the more
faithful of the two, and it is certainly the fairer counterpart to a scheme that does
not compute exactly either.

At the network level, `probe_accuracy.rs` runs both routes over identical programs and
inputs: class agreement was 40/40 on `16×8×4`, 40/40 on `784×32×10`, 10/10 on LeNet-5.
Ninety trials is a real measurement and a small one; it bounds nothing on its own, and
the primitive-level characterisation above is what the claim rests on.

#### Optimization 3 — every pairwise comparison at once

With rescaling cheap, the argmax became the dominant term, and measuring it in
isolation said by how much: **38 of LeNet-5's remaining 56 exchanges, 1.09 s of its
2.07 s**. The tournament had taken the argmax from nine sequential comparisons to
four, and four is the floor for a tournament, because each bracket must know who won
the last.

Constant depth requires abandoning the tournament. Computing *every* pairwise
comparison makes all `n(n−1)/2` of them mutually independent, so they batch into one
round, and the winner is then read off the comparison matrix: index `i` wins exactly
when it beats every other candidate, which is a product of `n−1` shared bits. Every
candidate's product tree is independent of every other's, so all `n` trees together
cost `⌈log₂(n−1)⌉` rounds of *multiplication* — one exchange each, against roughly
nine for a comparison. The index is a public-constant weighted sum and so is free.

The trade is forty-five comparisons against nine. Whether that is worth constant depth
is a property of the link, and the byte-headroom measurement above is what licensed
spending it. Measured at 25 ms, 100 Mbps, on top of optimizations 1 and 2:

| workload | exch | bytes | elapsed | |
|---|---:|---:|---:|---|
| argmax over 10, tournament | 38 | 43 KiB | 1094 ms | |
| argmax over 10, pairwise | 14 | 214 KiB | **420 ms** | **2.6×** |
| `784×32×10` | 27 | 866 KiB | **847 ms** | 1.8× |
| LeNet-5 | 32 | 4.94 MiB | **1412 ms** | 1.47× |

Bytes rose 3.6% on LeNet-5 and 25% on the classifier — the headroom absorbed it, as
the earlier table said it would. `bench::depth::argmax_pairwise`, with the tie-breaking
rule checked against the tournament on eleven cases including all-equal inputs, since
a tie is where the two could silently disagree.

`argmax_tree` is kept rather than replaced: on a link where bandwidth dominates it is
the right choice, and the point of measuring the terms separately is that the answer
depends on the link.

#### Where the three optimizations leave the two workloads

| | start of session | after all three | |
|---|---:|---:|---|
| `784×32×10` | 3228 ms | **847 ms** | 3.8× |
| LeNet-5 | 7763 ms | **1412 ms** | 5.5× |

LeNet-5's balance has flipped back: 549 ms of bandwidth against 984 ms of latency, so
it is latency-bound again and further round reduction is again worth having. Its 32
exchanges account exactly — nine layers at two each (one multiplication round, one
rescale opening) plus fourteen for the argmax.

#### Where this leaves the comparison

LeNet-5 on a 50 ms round-trip link, at a matched threat model, two-party dishonest
majority, with preprocessing hoisted: **1.41 s**, from 7.76 s at the start of the
session. Orion reports 2.93 s for its LeNet-5, single machine, no network.

That comparison is **not yet valid** and is not being made. Orion's LeNet-5 is
4.30 MFLOPs against our 0.28 M multiply-accumulates — the same architecture name over a
15× difference in work — and the withdrawal in §VI.8 was for exactly this error. What
closes the gap is our own implementation of the Orion pipeline on our own hardware, on
the same network, which is the outstanding task and is not something to estimate around.

#### Two accounting distinctions now explicit in the engine

`DistEngine` carries two flags rather than one configuration, because conflating them
is how an online figure gets quoted against a total:

- `hoist_bits` — shared random bits taken from preprocessing (measuring the online
  phase, comparable to a warm homomorphic server) against generated inline by the
  exclusive-or protocol (measuring a server handed a query with nothing prepared).
- `trunc_pairs` — the rescale route above.

Every table in this section names which was in force.

---

### VI.10 Our own homomorphic arm, and the first same-network comparison (2026-08-18)

*Tool:* `tools/ckks_cnn.cpp` — CKKS on OpenFHE 1.5.1, built and run per
`tools/README.md`. *Fixture:* `bench::conv_nn::export`, so both arms evaluate one
definition of the network on one input. Machine: the same one every figure in this
document was measured on.

#### Why the arm was built rather than cited

§VI.8 quoted Orion's 2.93 s for LeNet-5 and then had to withdraw a comparison against
it, because Orion's LeNet-5 is 4.30 MFLOPs against our 0.28 M multiply-accumulates —
the same architecture name over a fifteenfold difference in work. Nothing about that
was fixable by requoting. A comparison needs the same network, on the same machine,
with both answers checked against the same plaintext reference, and that requires
implementing the homomorphic side.

What it is: a depth-conscious CKKS implementation of the same pipeline — one ciphertext
per channel, convolution by rotation, `x²` activation, average pooling with the window
mean folded into a multiply it needed anyway, and a bootstrap placed exactly when the
level budget would otherwise be exceeded. What it is not: Orion. Orion's contribution
is automating that last decision over arbitrary graphs; here the graph is a list and
the placement is computed from it directly.

#### One bootstrap, measured

Before the network, the primitive that governs it. At `HEStd_128_classic` with a
bootstrapping level budget, OpenFHE derives ring dimension 2¹⁷ and 65 536 slots:

| | |
|---|---:|
| key generation | 30.4 s |
| one `EvalMult` | 0.130 s |
| **one `EvalBootstrap`** | **110.7 s** |
| precision remaining afterwards | 11 bits |

A single bootstrap on this machine costs **78× our entire LeNet-5 inference under
secret sharing on a wide-area link**. That is the cliff Orion's Table 2 reports from
the other side — every network they evaluate without a bootstrap finishing in
0.23–2.93 s and every network with one taking 190–32 324 s — reproduced here as a
measurement rather than read off their table.

#### LeNet-5, both arms, same network

The average-pooling `x²` variant, which is the only one on which a same-function
comparison is possible: max pooling is a comparison, and the arm refuses it rather than
approximate it.

| | secret sharing, 2-party | CKKS, single server |
|---|---:|---:|
| threat model | dishonest majority, 2 of 2 | single server, no honest party needed |
| security | statistical, κ = 26 | `HEStd_128_classic` |
| ring / parameters | `Z_2^64` | ring 2¹⁵, 1 024 slots, depth 11 |
| **bootstraps** | not applicable | **0** — LeNet-5 fits the budget |
| multiplications | 289 055 | 4 378 ciphertext |
| rotations | not applicable | 2 043 |
| preprocessing / key generation | hoisted | 4.66 s |
| **inference** | **1.41 s** (50 ms RTT, 100 Mbps) | **43.8 s** (no network) |
| verified | class 6, bit-exact reference | class 6, reference class 6 |

**32×, on the same network and the same machine.** And the comparison is conservative
against the secret-sharing arm in two ways that should be stated rather than quietly
banked. First, it computes the argmax *inside* the protocol — fourteen of its
thirty-two exchanges — while the homomorphic protocol returns all ten logits and lets
the client take the maximum in the clear. So our arm does strictly more work. Second,
for the same reason it reveals strictly less: one class against a full logit vector.

The asymmetry that runs the other way is the one §VI.6 records and that matching threat
models does not remove: CKKS lets the client go offline, and two-party computation
needs it present for the whole circuit.

#### What this does *not* establish

Our CKKS arm is **15× slower than Orion's published LeNet-5 figure**, so the 32× above
is a margin over our own implementation and not over the state of the art. The reason
is measured, not guessed: slot utilisation. The arm holds one channel per ciphertext,
so a ciphertext carries at most 576 live values in 1 024 slots — and at ring 2¹⁵ those
1 024 slots are themselves a sixteenth of the ring — so utilisation against the ring is
**3.5%**. Orion's packing is
precisely what closes that. Until multi-channel packing lands, the honest statement is
that a competent, secure, bootstrap-free CKKS implementation of this network takes
43.8 s on this machine, and that a better-packed one would take less.

Three things were fixed on the way to 45.1 s from a first working 104.9 s, and the
sizes are worth recording because two of the three were not where a reader would guess:

1. **Masked plaintext multiplies replaced by scalar ones** in the convolution taps.
   The mask exists to clear the positions a rotation drags in from beyond the border,
   and it is unnecessary: every consumer of a feature map addresses only live
   positions, so the garbage never reaches a logit. 104.9 → 95.0 s, and the logits came
   out bit-identical — a **9%** saving, so the encoding cost that looked like the
   obvious waste was not the problem.
2. **Convolution rotations hoisted out of the output-channel loop.** The direct
   transcription rotates in the innermost loop, so the second convolution performed
   16 × 6 × 25 = 2400 rotations where only 6 × 25 = 150 distinct shifts exist.
3. **Reduction along each axis at its own stride.** The layout widens the slot stride
   at each pooling instead of compacting, which costs no levels; after two poolings the
   maps are 4 × 4 on a row stride of 112, so a contiguous rotate-and-sum walks 448
   slots in nine rotations where four suffice.

Together (2) and (3) took rotations from 4 239 to 1 583 and the network from 95.0 to
**45.1 s** — and improved accuracy as a side effect, the logits moving from 0.2335 to
0.2477 against a reference of 0.2617, because a rotation is a key switch and every one
of them adds noise.

A fourth change came later, from a failure rather than a profile: forcing the
bootstrap regime (§VI.11) killed the process, because a rotation key holds two
polynomials per modulus limb and at ring 2¹⁷ the 180-key set exhausted memory before a
single ciphertext existed. Two fixes, both of which the sparse packing had made
available all along and neither of which changes the computed function. **Sparse slot
encoding**: the ring dimension is fixed by security, but the slot count is not, and this
network's widest layout spans 784 slots, so encoding 1 024 rather than 16 384 costs
nothing in a levelled multiply while bootstrapping's cost and key material both scale
with it. **A logarithmic gather**: collecting scalars into slots `0..n` by shifting each
from slot 0 asks for a key per destination — 119 for the widest hidden layer — where
composing power-of-two shifts reaches every destination with `⌈log₂ n⌉` keys, at the
cost of a population count of rotations instead of one. Together: **180 keys to 48**,
which is what let the bootstrap measurement run at all, at a 5% cost in the
bootstrap-free case (43.8 s against the 41.4 s that full-ring encoding with a direct
gather reached). Every figure in §VI.10 and §VI.11 is from this final configuration.

#### The next measurement this sets up

LeNet-5 needs eleven levels and gets them, so it never bootstraps, which is why 43.8 s
rather than minutes. The interesting regime is the one where the budget runs out, since
that is where a 110.7 s primitive starts appearing in the total and where secret
sharing — which has no depth budget at all — must win. Reaching it needs depth, and
`--bootstrap` is wired for exactly that. That is a measurement, not a projection, and
it has not been made yet.

---

### VI.11 The depth sweep, both arms, and a precision result that was not expected (2026-08-18)

*Commands:* `cargo run --release --example probe_depth` and, per depth,
`export_cnn`/`ckks_cnn` (`tools/README.md`). *Network:* `Cnn::lenet5_deep`, LeNet-5 with
`extra` additional dense-plus-activation blocks — two levels each, the same unit of
depth on both sides, so the sweep is over one variable.

#### Why depth

The two paradigms pay for depth in incomparable ways, and that is the whole reason a
crossover exists. Secret sharing has no depth budget: a layer adds a fixed number of
communication rounds and a fixed amount of traffic. An approximate homomorphic scheme
has a level budget fixed at key generation, and the budget is not free to widen —
the ring dimension must grow with the modulus chain to hold the security level. Past the
budget the scheme must bootstrap, at 110.7 s against 0.130 s for a multiplication
(§VI.10).

#### Measured

| levels | instrs | exch | MiB | MPC compute | **MPC, 50 ms WAN** | ring | CKKS mults | CKKS rots | **CKKS** | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 626 110 | 32 | 4.82 | 128 ms | **1517 ms** | 2¹⁵ | 4 378 | 2 043 | **43.8 s** | 29× |
| 13 | 647 614 | 36 | 4.96 | 145 ms | **1683 ms** | 2¹⁵ | 4 463 | 2 883 | **61.1 s** | 36× |
| 15 | 669 118 | 40 | 5.10 | 238 ms | **1804 ms** | 2¹⁶ | 4 548 | 3 723 | **160.0 s** | 89× |
| 17 | 690 622 | 44 | 5.23 | 141 ms | **1937 ms** | — | — | — | — | |
| 19 | 712 126 | 48 | 5.37 | 147 ms | **2110 ms** | — | — | — | — | |

The secret-sharing curve is a line, exactly as the structure predicts: **+4 exchanges,
+0.14 MiB and +148 ms per block**, with no discontinuity anywhere.

**And a second cliff, before bootstrapping enters at all.** Going from 13 to 15 levels
forces the ring dimension from 2¹⁵ to 2¹⁶ and the cost from 61.1 s to 160.0 s — a
**2.6× jump for two levels**, against 7% for the same two levels on the secret-sharing
side. Every ciphertext operation in the network became more expensive because the
security level demanded a larger ring; the operation *counts* barely moved (4 463 →
4 548 multiplications). This is not the bootstrap cliff Orion's Table 2 describes, it
sits below it, and it is a discontinuity in the same direction.

#### The precision result

The fixture's class margins are tight at depth, and disclosing them turned out to
matter. Reference top-two margins, in units of the fixed-point quantum (2⁻⁸):

| levels | reference class | margin | CKKS worst logit error | CKKS verdict | MPC class |
|---:|---:|---:|---:|---|---:|
| 11 | 6 | 58 q | 16 q | agrees | **6** |
| 13 | 0 | **2 q** | 15 q | **margin below error** — returned 7 | **0** |
| 15 | 8 | 10 q | 21 q | agrees, but fortuitously | **8** |
| 17 | 2 | **1 q** | — | — | **2** |
| 19 | 3 | 72 q | — | — | **3** |

**The secret-sharing arm returned the reference class at all five depths, including the
two whose margin is one and two quanta. The CKKS arm returned the wrong class at
thirteen levels.** At fifteen its error (21 q) exceeded the margin (10 q) and it agreed
anyway, which is luck and is labelled as such.

The mechanism is not subtle and it is worth separating from the fixture. CKKS
accumulates approximation error through every rescale and key switch, and at thirteen
levels that error was fifteen times the quantization step. Stochastic truncation's error
is a carry in the last fixed-point bit, unbiased, so it stays *below* one quantum per
rescale and the errors do not systematically add. So exactness here is not an aesthetic
preference: at these depths it is the difference between the right answer and the wrong
one.

Two honest qualifications. The tight margins are a property of an *untrained* fixture —
a trained network separates its classes, and §VI.7 records a previous version of a depth
sweep being written up with a fixture artefact mistaken for a finding. And the latency
figures are unaffected by any of this, because cost depends on program shape rather than
weight magnitude, which §VI.7 established by measurement.

So what the table supports is: the **latency** comparison at every depth, and the
**mechanism** that approximate arithmetic degrades with depth while exact arithmetic
does not. What it does not support is a quantitative accuracy claim for trained
networks, which needs a trained fixture.

#### The bootstrap regime, measured

Every CKKS row above is bootstrap-free, so none contains the primitive that dominates
the literature's deeper networks, and the 29–89× margins are therefore drawn from the
regime *most favourable* to the homomorphic arm. Capping the computational level budget
at six (`--levels=6`) forces the other regime on the same network and the same input:

| LeNet-5, 11 levels needed | ring | slots | bootstraps | keygen | **inference** | class |
|---|---:|---:|---:|---:|---:|---:|
| budget 11 — never bootstraps | 2¹⁵ | 1 024 | 0 | 1.21 s | **43.8 s** | 6 ✓ |
| budget 6 — must bootstrap | 2¹⁷ | 1 024 | **20** | 43.9 s | **1345.1 s** | 6 ✓ |
| secret sharing, 50 ms RTT | — | — | — | hoisted | **1.52 s** | 6 ✓ |

**A 31× discontinuity inside the homomorphic arm**, from a change that leaves the
network, the input and the answer alone — the decrypted logits are *identical* to the
bootstrap-free run to every digit printed, which is the strongest available check that
bootstrapping was performed correctly rather than skipped. Against secret sharing the
margin goes from 29× to **885×**.

The implied per-bootstrap cost, `(1345.1 − 43.8)/20 ≈ 65 s`, is consistent with the
110.7 s measured at 65 536 slots in §VI.10, since bootstrapping cost scales with the
slot count and this run uses 1 024.

**One caveat, and it cuts against our own number.** The count of twenty is inflated by
our packing. A bootstrap refreshes one ciphertext, and this arm holds one channel per
ciphertext, so six to sixteen ciphertexts are live at once and each needs its own
refresh whenever the budget runs out. An implementation packing all channels together
would exhaust the budget the same number of *times* while having far fewer ciphertexts
to refresh — plausibly two to four bootstraps rather than twenty. So 1345 s is an upper
bound on the regime and the defensible reading of the row is narrower than the number:
the *primitive* cost is ours and library-comparable, the *count* is an artefact of
3.5% slot utilisation, and even against a well-packed arm needing four bootstraps
(≈ 260 s) secret sharing at 1.52 s remains ahead by more than two orders of magnitude.

This is the shape §VI.8 read off Orion's Table 2 — latency tracking bootstraps rather
than work — reproduced here as a measurement on one network, one input and one machine,
with both arms' answers checked against the same plaintext reference.

#### Still to measure

Multi-channel packing, which is the single change that would make the homomorphic arm
competitive with Orion and would shrink both the 29× and the 885×. Until it lands, every
margin in §VI.10 and §VI.11 is a margin over our own implementation, and that is how
they are stated.

---

### VI.12 Throughput, and why it changes the answer (2026-08-18)

*Commands:* `cargo run --release --example probe_batch`; `ckks_cnn --batch=N`.
*Network:* LeNet-5 average-pool/`x²`, the same fixture as §VI.10–VI.11.

Every figure so far has been a *latency*: one inference, start to finish. That is the
right measurement for an interactive query and the wrong one for a server, and the two
paradigms differ far more under batching than the latency tables suggest — in opposite
directions, which is why this is not a footnote.

An approximate homomorphic scheme carries thousands of slots per ciphertext and this
network's layout occupies a few hundred, so further inferences ride in the unused slots
at **no additional operation at all**: the same rotation shifts every inference's map
identically, and because the weights are shared, a plaintext mask is the single-inference
pattern repeated per block. Secret sharing gets nothing for free in traffic — every
inference's shares must actually be sent — but it gets everything for free in *rounds*,
because independent inferences are independent instructions and the scheduler puts them
in one round. One paradigm amortises bytes, the other amortises latency, and on a
wide-area link those are precisely the two terms that matter.

#### Measured, both arms

| batch | MPC exch | MPC MiB | MPC total | **MPC per inference** | CKKS mults | CKKS total | **CKKS per inference** | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **32** | 4.82 | 1492 ms | **1492 ms** | 4 378 | 45.10 s | **45.10 s** | 30× |
| 2 | **32** | 9.47 | 2051 ms | **1026 ms** | — | — | — | |
| 4 | **32** | 18.78 | 3156 ms | **789 ms** | — | — | — | |
| 8 | **32** | 37.39 | 5648 ms | **706 ms** | 4 378 | 45.45 s | **5.68 s** | **8.0×** |
| 16 | — | — | — | — | 4 378 | 45.82 s | **2.86 s** | |

Both predictions hold exactly, and each is visible as a column that does not move. **The
secret-sharing arm's exchange count is 32 at every batch** — rounds amortise completely.
**The homomorphic arm's operation counts are 4 378 multiplications and 2 043 rotations at
every batch** — work amortises completely, the whole 1.6% growth from batch 1 to 16 being
plaintext encoding over more slots. All sixteen blocks decided the same class, which is
checked rather than assumed: a packing bug that corrupted the later blocks would
otherwise be reported as a speedup.

#### The margin narrows by seven times

**30× at batch 1 becomes 8.0× at batch 8**, and at batch 16 the homomorphic arm reaches
2.86 s per inference against a secret-sharing figure converging on roughly 0.7 s — about
**4×**. That is the same experiment, the same network and the same machine as §VI.10's
32×; the only change is asking for throughput instead of latency, and it costs seven
eighths of the margin.

Both arms then saturate, for different reasons, and the reasons are measured:

- **Secret sharing saturates on bandwidth.** Per-inference cost falls 1492 → 1026 → 789
  → 706 ms and is converging on the traffic term: 4.67 MiB per inference at 100 Mbps is
  392 ms that no amount of batching removes, because those bytes are Beaver openings and
  there is one opening per multiplication. At 289 055 multiplications and 16 bytes each
  that is 4.6 MiB — so the arm is already at the floor its packing allows, and only a
  narrower ring element would move it.
- **The homomorphic arm saturates on slots.** Batch 16 fills the 16 384 slots that ring
  2¹⁵ provides. Batch 32 needs ring 2¹⁶, where §VI.11 measured every operation becoming
  2.6× more expensive — so the next doubling of the batch costs more than it saves, and
  2.86 s per inference is this configuration's optimum rather than a point on a curve.

#### What this does and does not change

It does not change the depth result. §VI.11's bootstrap cliff is a cost per ciphertext
refresh, and batching does not reduce the number of refreshes — a batched ciphertext
still exhausts its budget at the same depth. Under batching the 885× would narrow by the
same factor as the 30×, to something near 30×, and the *shape* — a line against a
staircase — is untouched.

It does change the headline. The defensible claim from this session is **not** that
secret sharing beats an approximate homomorphic scheme by thirty times; it is that the
margin is thirty times on latency, falls to about four times on throughput, and returns
to two orders of magnitude the moment the network exceeds its level budget. Any single
number drawn from that would be a choice of regime disguised as a result.

And it sharpens what the remaining packing work is for. Our arm's 3.5% slot utilisation
costs it twice under batching — once in operations per inference, and once in the batch
size a ring can hold — so the 15× gap to Orion bites hardest in exactly the regime where
our margin is thinnest. Closing it is the next thing to build, and until it is built the
four times is a margin over our own implementation.

---

### VI.13 Opening each operand once, and finding the structure automatically (2026-08-18)

*Code:* `mpcc::vm::{bilinear_batch, multiply_level}`, `ShareEngine::bilinear_triple`,
`DistEngine::with_bilinear`. *Probes:* `examples/probe_bilinear.rs`,
`examples/probe_batch.rs`. *Tests:*
`opening_each_operand_once_is_cheaper_and_gives_the_same_answer`,
`fusing_a_convolution_does_not_change_what_it_computes`.

#### The constraint this attacks

§VI.12 measured the two-party arm's throughput as bandwidth-bound, and located the
traffic exactly: one Beaver opening per multiplication, 289 055 multiplications, 16 bytes
each, 4.6 MiB. Multiplications already equal multiply-accumulates one for one, so scalar
Beaver is at its floor and no scheduling change reaches it.

The multiply-accumulates are not evenly distributed, and that is the opening. Counting
them per layer against the number of distinct operand values each layer touches:

| layer | MACs | scalar Beaver elements | operand elements | reuse |
|---|---:|---:|---:|---:|
| conv, 6×24×24, k5, 1 in | 86 400 | 172 800 | 934 | **185×** |
| conv, 16×8×8, k5, 6 in | 153 600 | 307 200 | 3 264 | **94×** |
| dense 120×256 | 30 720 | 61 440 | 30 976 | 2.0× |
| dense 84×120 | 10 080 | 20 160 | 10 200 | 2.0× |
| dense 10×84 | 840 | 1 680 | 924 | 1.8× |

The same hundred and fifty kernel weights are multiplied against a feature map at tens of
thousands of positions, while a dense layer uses each weight once. So 85% of the network's
arithmetic sits where operand reuse is two orders of magnitude, and Beaver charges for it
as though every product were unrelated.

#### The construction

For any map `F` linear in each argument separately — a convolution and a matrix product
are both such — and a triple `(a, b, F(a,b))` with `a`, `b` uniform, bilinearity gives

```text
F(x, y) = F(a, b) + F(a, e) + F(d, b) + F(d, e),    d = x − a,   e = y − b
```

Opening `d` and `e` reveals nothing, `a` and `b` being uniform and unknown. Each of the
last three terms has at least one *public* argument, so each is a linear combination of
shares with public coefficients and costs no communication. Traffic is `|x| + |y|`
elements rather than twice the product count, in one round however many products the form
contains.

What it costs instead is local work: each product term is evaluated three times rather
than once. §VI.12's measurement is what says that is the right trade — 392 ms of every
706 ms was traffic — and on a link fast enough that bytes are cheap the scalar route is
better, so the engine opts in rather than switching.

#### Measured, two parties over a 50 ms link at 100 Mbps

| layer | products | operands | scalar | bilinear | traffic | elapsed |
|---|---:|---:|---:|---:|---:|---:|
| conv 28×28 k5 1→6 | 86 400 | 934 | 1383.1 KiB | **40.4 KiB** | **34×** | 176 → **63 ms** |
| conv 12×12 k5 6→16 | 153 600 | 3 264 | 2426.8 KiB | **52.2 KiB** | **46×** | 260 → **60 ms** |

Both routes return the same values, checked against a plaintext reference computed
directly from the definition, and the test asserts on the byte *ratio* rather than a
constant — because the whole saving is a function of operand reuse and a form over
unrelated products would show none, which hard-coding a byte count would conceal.

#### Reaching it automatically, without a new instruction

The obvious way to wire this in is an IR operation carrying the index pattern, emitted by
a vectorised lowering. It turned out not to be necessary. `vm::execute` already gathers
every independent multiplication at a schedule level into one batch, and the structure the
scalar lowering discarded is still present in the *operand registers*: a convolution's
level reads a few hundred distinct registers across tens of thousands of multiplications.
Two multiplications sharing an operand are recognisable only by register identity —
comparing share values would be both wrong and impossible — so the batch now carries the
register pair alongside each product, and `vm::multiply_level` decides per level.

If the level's operands repeat by a factor of four or better it is evaluated as one
bilinear form; otherwise it falls through to Beaver, which is optimal there. The form
produces one output per product, leaving the additions that consume them untouched, which
costs nothing: the traffic is the operand count either way. So **no IR operation, no
lowering change, no scheduler change**, and the optimisation applies to any program whose
levels happen to have reuse rather than only to the ones this benchmark emits.

The threshold exists because the route trades bytes for local work — each product is
evaluated three times — so fusing a level with no reuse would triple its arithmetic and
save nothing. The convolutions this is aimed at have measured reuse of 94 and 185.

#### Measured on the compiled network

LeNet-5 through the full pipeline, two parties, 50 ms RTT, 100 Mbps, preprocessing
hoisted, stochastic rescaling:

| | exchanges | traffic | compute | **elapsed** | class |
|---|---:|---:|---:|---:|---:|
| Beaver | 32 | 4.82 MiB | 178 ms | **1463 ms** | 6 ✓ |
| fused | 32 | **1.19 MiB** | 151 ms | **1166 ms** | 6 ✓ |

Traffic falls **4.05×** and the round count is unchanged, which is the claim. Latency
improves only 1.25×, because at 32 exchanges the link's 1.6 s of round trips dominates and
bandwidth was never the binding term for a *single* inference.

#### Where it actually matters: throughput

§VI.12 established that throughput is the regime where the two-party arm is bandwidth-bound
and where its margin over the homomorphic arm is thinnest. That is the regime this changes.

| fusion | batch | traffic | total | **per inference** | inferences/s |
|---|---:|---:|---:|---:|---:|
| Beaver | 1 | 4.82 MiB | 1488 ms | **1488 ms** | 0.67 |
| Beaver | 8 | 37.39 MiB | 5936 ms | **742 ms** | 1.35 |
| fused | 1 | 1.19 MiB | 1153 ms | **1153 ms** | 0.87 |
| fused | 2 | 2.19 MiB | 1385 ms | **692 ms** | 1.44 |
| fused | 4 | 1.99 MiB | 1785 ms | **446 ms** | 2.24 |
| fused | 8 | **3.47 MiB** | 2905 ms | **363 ms** | **2.75** |

At batch eight, traffic falls **10.8×** and per-inference cost **2.04×**. The reason the
gain grows with the batch is that a dense layer's weights are shared across the batch while
only its activations are fresh, so a layer with no reuse at batch one acquires reuse
proportional to the batch and crosses the threshold. That is visible directly in the
non-monotonic traffic column — 2.19 MiB at batch two against 1.99 MiB at batch four — where
a larger batch sends *less in total* because more of its levels became fusable.

**Against the homomorphic arm this roughly doubles the thinnest margin.** CKKS's best
throughput is 2.86 s per inference (batch 16, §VI.12); the two-party arm now reaches 363 ms
at batch eight, so **~4× becomes ~8×**.

#### And the bottleneck has moved back

At batch eight the arm sends 0.43 MiB per inference — 35 ms at 100 Mbps — against 1.6 s of
round trips amortised over eight inferences, or about 200 ms each. So more than half the
per-inference cost is once again *latency*, and the curve was still falling at batch eight
rather than flattening. Rounds amortise with batch, so the next thing to measure is simply
a larger batch, and the round-reduction work of §VI.9 has become relevant again for a
reason it was not chosen for.

Correctness is checked on both paths: the isolated primitive against a plaintext reference
computed from the definition, and the automatic path by running a compiled convolutional
network with fusion on and off and requiring identical outputs, an unchanged round count,
and traffic that never increases — the last because a level without reuse must be left
alone rather than fused at a loss.

---

### VI.14 Porting Orion's linear-transform evaluator (2026-08-18)

*Code:* `tools/ckks_cnn.cpp` (`--lt`), `tools/ckks_bsgs.cpp`. *Reference:* Orion,
ASPLOS'25 (Ebel, Garimella, Reagen), `github.com/baahl-nyu/orion`.

#### Why a port rather than more of our own packing

§VI.12 left our homomorphic arm about fifteen times slower than Orion's published LeNet-5,
and the cause measured as 3.5% slot utilisation. The obvious remedy — pack channels
together — turns out to be a *loss*, and the measurement already in hand says so. CKKS eval
time is nearly independent of batch (45.1 s at batch 1, 45.8 s at 16, identical operation
counts), so throughput is `eval / batch` and anything reducing the achievable batch costs
proportionally. Channel packing needs `C×` more slots per inference and saves far less than
`C×` in operations, so it would take per-inference cost from 2.86 s to roughly 7.9 s.

What Orion does instead is different in kind. Every layer becomes **one linear transform**,
evaluated by generalized diagonals: for an `n × n` matrix and a vector in `n` slots,
`A·v = Σ_d diag_d(A) ⊙ rot(v, d)`, so the cost is one plaintext multiply per **nonzero
diagonal** and not per matrix entry. Baby-step/giant-step then reduces the rotations from
`D` to about `2√D` by sharing the inner rotations of `v` across diagonal groups.

This corrects a claim that stood in our own source until now. A comment there read that
realising a convolution as a Toeplitz matrix-vector product would be 2.7 M
multiply-accumulates against the convolution's 86 400. That is true of a *dense* matvec and
false of the same matrix evaluated by its diagonals, which is the entire basis of the
method.

`tools/ckks_bsgs.cpp` verifies the transform in isolation first — 9 diagonals over 1024
slots, 5 rotations instead of 9, worst error 5.4·10⁻⁹ against a plaintext reference —
because the index arithmetic of a pre-rotated diagonal is easy to get wrong in a way that
still yields plausible numbers.

#### What multiplexed packing is actually for, measured

The diagonal count is not a property of the layer; it is a property of the **layout**. The
first attempt let a convolution's output grid shrink with its data, 28 wide in and 24 out.
The offset from an input slot to the output slot it feeds is then `y·4 + a·28 + b − oc·576`,
which depends on the row, so every row contributes its own diagonal:

| | conv 1→6 k5 | conv 6→16 k5 |
|---|---:|---:|
| output grid shrinks with the data | **1 225** | **1 500** |
| grid held fixed, valid region shrinks inside it | **136** | **592** |

Holding the grid fixed and doubling the multiplexing gap at each stride — so that `h·g`, and
therefore the block size, is invariant across the network — makes a tap's offset `a·W·g + b·g`,
the same at every position. That is what the gap is for, and the table is the measurement of
it rather than an argument about it.

#### Measured, LeNet-5, both evaluators

| | mults | rotations | rotation keys | keygen | **eval** | class |
|---|---:|---:|---:|---:|---:|---:|
| tap by tap, channel per ciphertext | 4 378 | 2 043 | 48 | 1.21 s | **43.8 s** | 6 ✓ |
| linear transform, multiplexed | **1 638** | **266** | 133 | 2.71 s | **14.7 s** | 6 ✓ |

**3.0× on latency**, from 2.7× fewer multiplies and 7.7× fewer rotations. Diagonals per
layer: 136, 23, 592, 32 for the convolutions and poolings, and 555, 203, 93 for the dense
layers. Worst logit error 0.065 against a reference margin of 0.227, and the class agrees.

#### Neither evaluator dominates, and that is the result

| | latency | best batch | **per inference** |
|---|---:|---:|---:|
| tap by tap | 43.8 s | 16 (1024-slot block) | **2.86 s** |
| linear transform | **14.7 s** | 2 (8192-slot block) | 7.56 s |

The multiplexed layout needs 8192 slots because the first convolution's six output channels
sit at gap one and cannot be folded — the gap may only grow where the spatial extent shrinks,
which is at a stride. So the transform evaluator wins single-inference latency by 3.0× and
loses throughput by 2.6×, and the honest FHE figure is the better of the two per regime,
stated as such.

#### Where this leaves the comparison

| regime | MPC | CKKS, best of both | margin | was |
|---|---:|---:|---:|---:|
| latency | **1.17 s** | 14.7 s | **12.5×** | 37× |
| throughput | **363 ms/inf** | 2.86 s/inf | **7.9×** | 7.9× |

The latency margin falls by three times, which is what porting a better baseline is *for*.
Our arm is now about **5× off** Orion's published LeNet-5 rather than 15×, and the remaining
gap is the part of their design not yet ported: they choose each layer's gap by solving for
it rather than inheriting it, which is what would shrink the 8192-slot block and let the
transform evaluator batch as well as it computes.

Not ported, and deliberately: their bootstrap placement, which is a shortest-path problem on
a DAG of `(layer, level)` nodes weighted by latency. LeNet-5 never bootstraps, so it would
change nothing measured here; it becomes relevant at the depths of §VI.11.

---

## Part VIII — The cost of active security

*Command:* `baru bench-malicious --n 64 --bits 32`. Both legs run the same
workload through the **identical** transport stack (`LatencyTransport` +
`CountingTransport`), so the only difference is the protocol.
`docs/malicious-hybrid.md` describes the constructions.

### The headline

| Workload | bytes, equal width | bytes, **packed base** | interactive calls | wall clock (WAN, 25 ms one-way) |
|---|---|---|---|---|
| Comparisons, n=32, 32-bit | **2.63×** | **9.70×** | **1.06×** | **1.23×** |
| Dot product, n=64, 32-bit | 2.44× | 2.85× | 2.00× | 2.87× |

The comparison row is the one that matters for private inference: every ReLU,
max-pool and argmax reduces to a comparison, so it is the entire nonlinear
surface.

**Two byte columns, added 2026-08-03, and they differ by 3.7×.** The verified leg
cannot pack its wire — `aby3_verify::headroom_bits` is `128 − used_bits`, so its
statistical security *is* the slack between the values' width and the full ring, and
narrowing to the value width would set λ to 0. The semi-honest leg has no such
constraint: on the comparison row it packs 234 036 B → 63 416 B while the verified leg
stays at 615 446 B. So *equal width* is the cost of the MAC when both run unpacked,
and *packed base* is the cost to a deployment that would otherwise have packed.

Active security on the nonlinear surface therefore costs **2.63× the bytes at equal
width, 9.70× against a packed baseline, and 1.06× the rounds** either way. Default
`bench-malicious` reports the first; `BARU_WIRE_BITS=auto bench-malicious` the second.
The default was left at equal width so that flipping the repo's global encoding could
not silently redefine a published ratio.

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
