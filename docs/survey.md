# Measuring Privacy-Preserving Computation in One Place: MPC, FHE, ZKP and Their Hybrids

**A systematization and measurement study built on the Baru artifact.**

Draft, 2026-08-01. Target venue: PoPETs (SoK track) — see
[§12 on framing](#12-on-framing-survey-sok-or-measurement-study).

> **LaTeX version:** [`../paper-survey/`](../paper-survey/) — 16 pages, builds
> clean with `pdflatex && bibtex && pdflatex && pdflatex`. This markdown file is
> the working draft; the LaTeX is the submission artifact. Keep them in sync or
> retire one.

---

## Abstract

Privacy-preserving computation is not one field but four — secure multi-party
computation, fully homomorphic encryption, garbled circuits and zero-knowledge
proofs — each with its own literature, its own benchmarks, and its own
accounting conventions. A practitioner choosing between them, or a reviewer
assessing a claim that one beats another, must today assemble numbers from
different papers, measured on different hardware, over different networks,
counting different things. Cross-paradigm comparison is therefore mostly
*citation assembly* rather than measurement.

This paper takes a different route. We survey the paradigms through a single
artifact, **Baru**, in which one source program runs on all of them — a plain
interpreter, 3-party replicated-secret-sharing MPC, garbled circuits, BFV/BGV/
TFHE homomorphic encryption, a Binius64 SNARK, and *combinations* in which
values cross between representations mid-computation without being exposed.
Every backend is driven through the same transport, charged by the same byte and
round accounting, and checked bit-exactly against the same plaintext reference.
A network cost model (`rounds × RTT + bytes / bandwidth`) is validated against a
real socket to within 13.6% worst-case, so a claim about a WAN deployment is
computed from measured counts rather than asserted.

We report what falls out. Several widely-repeated intuitions survive; several do
not. Among the results that only a single-harness study can produce: the
FHE-versus-MPC crossover is a property of *the baseline's preprocessing*, not of
either cryptosystem, and it disappears entirely once MPC is given SHE-based
preprocessing; and active security on the nonlinear surface of private inference
costs 2.63× bytes but only 1.06× rounds — and 9.70× bytes once the semi-honest
baseline it is measured against is allowed to pack, which only the unverified side
can do; and the choice between a garbled and a
`wrap3`-style comparison protocol has a boundary at 30–40 Mbps that moves with RTT.
We also report our own withdrawn claims — a ZKP backend that proved nothing, an MPC
bit-decomposition that leaked its inputs, and that last boundary, whose location we
published wrongly three times before measuring it — because a survey that reports
only what worked is not reporting.

---

## 1. The comparability problem

Ask how fast private inference is and the literature answers in incompatible
units. Gazelle, Delphi and Cheetah report 2-party hybrid systems where the
server holds the model in plaintext. Falcon reports 3-party replicated sharing
on a LAN. CKKS papers report amortized throughput over batched slots. SNARK
papers report prover time for a circuit that a different paper would express
with a different constraint count. Each is internally rigorous. Together they do
not compose.

Four specific obstacles recur:

1. **Different hardware.** A 5× difference between two systems is unfalsifiable
   when one ran on a server-class Xeon and the other on a laptop.
2. **Different accounting.** "Communication" may mean payload bytes, serialized
   bytes, or bytes per party. "Rounds" may mean protocol rounds, messages, or
   interactive calls. These differ by more than the effects being measured.
3. **Different security models.** Semi-honest and malicious numbers appear in
   adjacent columns of secondary literature without the distinction surviving
   the citation.
4. **Different functions.** Two "ReLU" costs may be for different bit widths,
   different fixed-point scales, or — in the worst case we found in our own
   work — subtly different functions that nobody checked against a common
   reference.

A survey assembled from such numbers inherits every one of these. The
alternative we pursue is to re-measure the paradigms in a single harness where
the four obstacles are controlled by construction: same machine, same
accounting code, machine-checked security levels, and a bit-exactness assertion
against a shared plaintext reference before any cost is quoted.

The cost of this approach is equally clear and we state it up front: our
implementations are research-quality Rust, not the optimized C++ of the systems
we discuss. **Absolute figures here are not competitive with tuned
implementations and should not be read as such.** What the single harness buys
is *internally valid ratios* — comparisons in which the confounds above are
held fixed. Where we compare to published numbers, we say so and treat the
comparison as weaker evidence than our internal ones.

### 1.1 Contributions

- A **systematization** of four privacy-preserving paradigms organized by what
  they actually cost, with the cost dimensions (rounds, bytes, compute,
  preprocessing) separated rather than collapsed into a single "performance"
  number.
- A **single-artifact measurement study**: ~70,000 lines of Rust in which one
  program runs on every paradigm, with a validated network model and uniform
  accounting.
- **Cross-paradigm findings** that are not visible from any single paradigm's
  literature (§7), including two crossovers whose *location* turns out to depend on
  parameters the original claims left implicit — one of which we mislocated three
  times ourselves before measuring it (§7.3).
- A **methodology for honest benchmarking** (§6) — bit-exactness gating,
  machine-checked security attestations, and a discipline of recording
  withdrawn results — together with the record of what it caught in our own
  work (§9).

---

## 2. Scope

**In scope.** Semi-honest and malicious 3-party MPC over rings; garbled circuits
in the honest-majority setting; RLWE-based FHE (BFV, BGV) and TFHE;
FHE↔MPC scheme switching; SNARKs over 64-bit words (Binius64); and the
supporting primitives these need — oblivious transfer, private set intersection,
and Beaver-triple preprocessing from both OT and SHE.

**Out of scope.** Differential privacy and federated learning (different threat
model: they protect an *aggregate*, not a computation); trusted execution
environments (hardware trust assumption); ORAM (surveyed but not implemented —
see §10); and transformer-specific nonlinearities such as softmax and LayerNorm,
which need exp/reciprocal/rsqrt subprotocols we do not have.

**Application focus.** We use private neural-network inference as the running
workload, because it is the one application that exercises every paradigm: dense
linear algebra (where FHE packing shines), comparisons (where MPC and garbled
circuits compete), and verifiable computation (where SNARKs apply). Where a
result is specific to this workload we say so.

---

## 3. Background: four paradigms and four cost structures

The paradigms are usually taxonomized by security property. For a practitioner
choosing among them, the more useful axis is **what each one is expensive in**.

| Paradigm | Secret from | Expensive in | Cheap in | Round complexity |
|---|---|---|---|---|
| Secret-sharing MPC | the computing parties | **rounds**, bandwidth | local arithmetic | O(depth), interactive |
| Garbled circuits | the computing parties | **bandwidth** | rounds | O(1), constant-round |
| FHE | the computing server | **compute**, ciphertext size | rounds (none) | non-interactive |
| ZK proofs | the verifier | **prover compute** | verifier, proof size | non-interactive |

Three consequences drive most of what follows.

**Secret-sharing MPC and garbled circuits trade rounds against bytes.** This is
the classic Yao-versus-GMW trade. It means neither dominates: the winner depends
on RTT and bandwidth, and the boundary is an empirical question. §7.3 measures
it.

**FHE and MPC trade compute against communication.** FHE sends a lot once and
computes hard; MPC sends a little many times. On a fast LAN MPC usually wins; on
a thin pipe the balance shifts. But this framing omits *preprocessing*, and §7.1
shows that the omission is what makes most published FHE-vs-MPC crossovers
fragile.

**ZK is orthogonal, and conflating it with the others is a category error.** MPC
and FHE keep data secret *from the parties computing on it*. A SNARK proves a
computation was done correctly while hiding the witness *from a verifier* — the
prover holds everything in the clear. A deployment needing both confidentiality
and integrity pays for both. We keep them in separate tables for this reason,
and our benchmark prints the caveat alongside its numbers so they cannot travel
without it.

### 3.1 The hybrid, and why it exists

Neural-network inference alternates linear layers (matrix multiply, convolution)
with nonlinearities (ReLU, max-pool, argmax). FHE is good at the former and bad
at the latter — comparison is not a low-degree polynomial. MPC is the reverse:
comparison costs a bit decomposition, but a linear layer costs a round
regardless of width.

The hybrid architecture — linear layers under FHE, nonlinearities under MPC,
with a *scheme switch* between them — is therefore the natural design, and it is
what Gazelle, Delphi and Cheetah all instantiate. Baru implements it, and adds
two things to the standard construction (§5.4): the weights are encrypted too,
and the switch is used as a *noise refresh*, which removes bootstrapping from
deep networks entirely.

---

## 4. The Baru artifact

### 4.1 One program, many backends

Baru is a Lisp-like language whose interpreter manipulates secret values only
through opaque handles and a fixed set of primitive operations. This is the
**Arithmetic Black Box** abstraction of the SPDZ line: an ideal functionality
storing values by handle and exposing `add / mul / lt / share_public / reveal`.
A backend implements those primitives; the evaluator never inspects a
representation.

```lisp
; private ReLU on one neuron — every value is secret-shared
(let ((w :sint8 -1) (x :sint8 3) (b :sint8 0))
  (let ((z (+ (* w x) b)))
    (reconstruct (* z (> z 0)))))          ; => 0

; a value computed under MPC crosses into FHE without being revealed
(let ((keys (fhe-keygen)) (x (share 6 32)) (y (share -7 32)))
  (fhe-dec keys (share->fhe keys (* x y))))   ; => -42
```

The consequence for a survey is direct: *the same source text* is the workload
for every paradigm, so a cross-paradigm comparison cannot be confounded by two
implementations computing subtly different things. This is not a hypothetical
concern — §9 records a case where it caught exactly that.

### 4.2 Composability

Because every gadget (`relu`, `max2`, `maxpool2d`, `argmax`, `dense`, `conv2d`)
is a pure composition of ABB primitives, UC composition gives security of any
network built from them for free, given the primitives. The practical
consequence is that architecture search does not require a new security
argument per architecture; the research question becomes "which architecture is
*fastest* under the cost model?" rather than "is this one secure?".

### 4.3 What is implemented

| Layer | Contents |
|---|---|
| **MPC** | ABY3 replicated shares over `Z_{2^128}`; **triple-free** multiplication from pairwise PRFs; bit decomposition (carry-save 3→2 + Kogge–Stone); batched primitives; **malicious security with abort** (≤1 of 3 corrupt); in-process and TCP transports |
| **Garbled circuits** | Half-gates (Zahur–Rosulek–Evans), free-XOR, fixed-key AES; comparator and fused ReLU∘rescale circuits; OT-free label delivery in the 3-party honest-majority setting |
| **Falcon** | `Π_PC` private compare over `Z_p`, `Π_WA` wrap3, `Π_DReLU`, `Π_SS`, `Π_Maxpool`, exact truncation — a reproduction of Falcon (PETS'21) inside the same harness |
| **FHE** | BFV and BGV (depth-1, no evaluation keys), TFHE-rs, an exact real-arithmetic backend; multi-modular NTT with Shoup butterflies; CRT slot packing; Cheetah-style convolution packing |
| **Hybrid** | MPC↔FHE switching both directions; re-encryption as a bootstrapping alternative; replicated threshold-key switch |
| **ZKP** | Expression→circuit compiler; real Binius64 prover/verifier |
| **Compiler** | Backend-aware circuit IR, rewrite rules, cost models, protocol registry (dealer / Shamir / Semi2k / LowGear), per-operation protocol assignment |
| **Primitives** | Chou–Orlandi and IKNP OT, DH-PSI |

---

## 5. The paradigms, surveyed and measured

All figures below are from the maintainer's machine (Apple M1 Pro, 8 cores),
release build, unless noted. Provenance for every row is in
[`experiments.md`](experiments.md); commands are reproducible from the artifact.

### 5.1 Secret-sharing MPC

Baru's MPC layer is ABY3-style: `x = s₀+s₁+s₂ mod 2^128`, party `i` holding the
replicated pair `(sᵢ, sᵢ₊₁)`. Addition and scalar multiplication are local.
Multiplication is **triple-free** — each party sends one ring element to one
neighbour, with correlated randomness regenerated locally from a one-time
pairwise PRF key exchange. There is no dealer and no preprocessing phase.

The engineering lesson this layer taught, four separate times, is worth
generalizing beyond the artifact:

> **The arithmetic was never the bottleneck; the round structure was.** Each
> time a segment-level measurement said "MPC is slow", a per-phase breakdown
> said "one unbatched loop inside it is slow."

Batched forms (`a2b_batch`, `lt_batch`, `mul_batch`) cost **the same number of
rounds for any batch width**. Adopting them moved the CNN's MPC segment from
40.8 s to 6.1 s (6.7×) with messages falling 29,211 → 5,591. A survey reader
should take from this that published MPC costs are highly sensitive to batching
discipline, and that comparing two systems' round counts without checking
whether both batch is comparing implementations, not protocols.

**Cost of active security** (verified 2026-08-01, `bench-malicious --n 32
--bits 32`):

| workload | bytes, equal width | bytes, **packed baseline** | interactive calls | wall clock (WAN, 25 ms) |
|---|---|---|---|---|
| **comparisons, n=32, 32-bit** | **2.63×** | **9.70×** | **1.06×** | **1.26×** |
| dot product, n=64, 32-bit | 2.44× | 2.85× | 2.00× | 2.26× |

**Two byte columns, because the honest answer depends on what you compare against,
and they differ by 3.7×.** The verified leg cannot narrow its wire: its statistical
security is `128 − used_bits` (`headroom_bits`), so packing it to the values' own
width would set λ to zero. The semi-honest leg has no such constraint and packs
freely — 234 036 B → 63 416 B on the comparison row, with the verified leg unchanged
at 615 446 B.

So *equal width* answers "what does the MAC cost if both protocols run unpacked",
and *packed baseline* answers "what does active security cost a deployment that would
otherwise have packed" — which is the question a practitioner buying it is asking.
**On the workload that matters for inference the answer is 9.70×, not 2.63×.**

The comparison row is the one that matters, since every ReLU, max-pool and argmax
reduces to a comparison. **Active security costs bytes, not rounds**: the MAC
multiplies what is sent but batches into the same round. The dot-product row moves
much less between the two columns (2.44 → 2.85×) because its traffic is input sharing
and reveals rather than ring resharing, so there was little padding to recover — the
same reason `bench-nn` was unaffected by the encoding fix. Its high interactive-call
ratio is a small-workload artifact, a fixed pair-pool cost amortized against almost
nothing, and we report it rather than hide it.

Reproduce both with `bench-malicious` (equal width, the default) and
`BARU_WIRE_BITS=auto bench-malicious` (packed baseline). We kept the default at equal
width so that flipping the repo's global encoding could not silently redefine a
published ratio.

A second result generalizes, **with a correction we owe to §7.3's encoding fix.**
The statistical slack that ring-based verification normally charges as a widening to
`Z_{2^{k+s}}` is already present when the implementation shares over a ring much
wider than its values: Q16.16 values in a 128-bit ring leave **59 bits** of
statistical security. That much stands and is what `headroom_bits` computes.

What we previously added — that this comes **at no widening cost** — is false, and
the re-baseline is what exposed it. The 128-bit ring was not free; it was costing
3.1–5.6× in bytes on every semi-honest path, which is precisely what packing those
paths to their values' own width recovered (§7.3). The verified path *cannot* recover
it: its security is `128 − used_bits`, so narrowing the wire to the value width sets
λ to zero. The slack was never free — it was invisible, because nothing in the repo
was packed and so nothing revealed its price.

The consequence is the second byte column above, and it is now measured rather than
bounded: **9.70× against a packed baseline, against 2.63× at equal width.** The
prediction was that the gap would fall inside the packing factors measured on
comparable paths (3.1–5.6×); the measured ratio of ratios is 3.69×, so it did.

### 5.2 Garbled circuits

In the *3-party honest-majority* setting, garbled-circuit input labels are
delivered by the co-garbler with no oblivious transfer. Since OT is the
expensive part of 2-party garbling, this changes the economics substantially,
and it is the reason garbling is competitive here at all.

Measured head-to-head against bit decomposition through the identical transport
(1352 comparisons, 20-bit): `a2b` needs 9 rounds and **1.29 MB**; the garbled circuit
needs 1 round and 1.69 MB. So **bit decomposition is the byte-cheaper of the two**,
by 1.3×, and garbling's advantage is entirely in rounds — wall clock favours it
18.8× on a LAN and 4.0× on an 80 ms / 100 Mbps WAN, the ratio shrinking as bandwidth
rather than latency starts to bind.

**This reverses a figure we published.** The same experiment previously reported
`a2b` at 5.94 MB, making garbling look 3.5× cheaper in bytes *as well as* faster.
That 4.6× difference was `u128` serialisation padding: a 20-bit ring share travelled
in 128 bits, and `a2b` is almost entirely ring traffic while garbled tables are
already compact — so the overhead fell on one side of the comparison only. With the
encoding fixed (§7.3), the byte ordering flips and only the round advantage survives.
Both routes remain bit-exact against the plaintext reference.

**An important correction to our own earlier claim.** The 18.8–4.0× above is the
*garbling primitive in isolation*. The deployable 3-party protocol, including the Y2A
return leg that converts the boolean output back to an arithmetic share, is smaller:
**7.7× on a LAN falling to 1.8× at 80 ms / 100 Mbps** for the comparison, and 3.9× to
2.4× for the fused ReLU∘rescale. We record both because the gap between them is
exactly the kind of thing that gets lost in citation.

**A second correction, to a number this paragraph used to quote.** It gave
**1.4–1.7×** for the deployable path, taken not from this experiment but from the
CNN's *MPC segment* — a different benchmark on a different workload, presented as if
it corrected this one. Measured today that segment is **6.1×** (a2b 1102 ms against
gc 180 ms). The old figure was right when it was written and is stale for a reason
recorded in §9: at that time an unbatched scheme-switch import was **94% of the MPC
segment**, so it loaded both routes equally and compressed the ratio toward 1. Import
is now 23 ms of 1102, and the ratio reflects the protocols instead of the plumbing.

Both corrections illustrate the same rule, and it is the one worth taking away:
**quote the deployable protocol, on the workload you are claiming about, and say
which benchmark produced the number.** A ratio is only meaningful together with what
was in the denominator.

### 5.3 Fully homomorphic encryption

Baru uses the cheapest FHE primitive that exists: **one ciphertext×ciphertext
multiplication, no relinearization keys, no bootstrapping, no evaluation keys**,
decrypting the degree-2 product directly under `s²`. Parameters are `n = 4096,
q = 2^100, t = 2^20` for BFV — inside the ~109-bit `log q` ceiling for 128-bit
security at that dimension.

Two packing results are worth surveying:

- **Reversed-coefficient packing** makes a whole matvec one multiplication. We
  find this is **already optimal for dense layers**: a degree-`n` product admits
  at most `n/in` separable outputs, so `⌈out·in/n⌉` multiplications is the
  limit. Gazelle's SIMD packing matches the throughput but requires
  rotations and Galois keys, forfeiting the no-evaluation-keys property.
- **Cheetah-style convolution packing** makes a whole output channel one
  multiplication, independent of image size. Measured A/B at 28×28: **1352 → 2
  multiplications**, FHE time 77.5 s → 264 ms.

A negative result that narrowed the design space: **BGV is indistinguishable
from BFV** at identical parameters (repeated runs put the gap at +0.5%, +0.5%,
−0.1%). Where the message sits in the ciphertext, high bits or low, is not where
the cost is.

On arithmetic, our multi-modular NTT — NTT-friendly primes as a computational
device, CRT-reconstructed, because a power-of-two `q` admits no direct NTT — is
**1.8× faster than NTL+GMP at every size** after moving its butterfly to
Shoup-preconditioned multiplication. NTL is retained as a *differential oracle*
rather than a dependency, which is what made rewriting the hottest code in the
repository safe.

### 5.4 The hybrid, and the no-bootstrap result

The construction's central observation: the scheme switch performed for a ReLU
*also refreshes the ciphertext*. After switching back, each party freshly
encrypts its share component and the evaluator sums; the sum of fresh
encryptions is a fresh encryption. **The FHE noise budget therefore resets at
every layer boundary**, and a network of arbitrary depth runs with zero FHE
bootstraps at cost linear in depth, on an engine supporting only a *single*
multiplication.

| depth | exact? | total | fhe | switch | mpc | our bootstraps | levelled-competitor bootstraps |
|---|---|---|---|---|---|---|---|
| 2 | ✓ | 145 ms | 20 | 59 | 54 | **0** | 0 |
| 8 | ✓ | 539 ms | 81 | 239 | 210 | **0** | 3 |
| 32 | ✓ | 2119 ms | 333 | 958 | 820 | **0** | 15 |

Linear in depth, bit-exact at every depth. The exactness is the evidence for the
noise claim: had noise accumulated, deep outputs would diverge.

This construction also encrypts the **weights**, a stronger privacy model than
Gazelle/Delphi/Cheetah, whose server holds the model in plaintext.

### 5.5 Zero-knowledge proofs

Proving a Q16.16 quantized neuron, `relu((Σ wᵢ·xᵢ) >> 16 + b)`, with public
weights and private inputs, on a real Binius64 prover:

| width | constraints | prove | verify | proof bytes |
|---|---|---|---|---|
| 1 | 14 | 6.3 ms | 1.0 ms | 18,544 |
| 64 | 589 | 8.0 ms | 1.3 ms | 47,808 |
| **784** | **7,159** | **32.3 ms** | **4.8 ms** | **111,712** |

Constraints grow linearly (≈9 per input); **proof size does not** — a 286×
larger circuit yields a 6× larger proof. Verification stays in single-digit
milliseconds.

Two implementation findings generalize to anyone building arithmetic circuits
over 64-bit words. First, comparison gates are **unsigned**, so signed
comparison must flip sign bits first; without it `relu(-1)` returns `-1` — a
circuit perfectly sound about the wrong function. Second, and sharper:
**division cannot be checked the obvious way.** A circuit cannot divide, so the
prover supplies quotient and remainder as witnesses; but checking `a = b·q + r`
with `|r| < |b|` is *unsound* because circuit multiplication wraps mod 2⁶⁴. For
`a=1, b=3` a cheating prover sets `r=0` and `q=3⁻¹ mod 2⁶⁴` and both checks
pass. Requiring the 128-bit product's high word to be the sign extension of the
low word closes it. We drive that forgery through the real prover as a test.

### 5.6 Preprocessing: OT versus SHE

Beaver-triple preprocessing is where the FHE-vs-MPC comparison is usually
decided, and it is usually decided implicitly. Measured (verified 2026-08-01,
`bench-triples`, two parties, semi-honest):

| route | domain | B/triple | triples/s | rounds |
|---|---|---|---|---|
| LowGear (SHE) | `F_p`, 60-bit prime | **192** | 97,304 | 2 |
| Semi2k (OT) | `Z_2^64` | 3,074 | 2,681 | 3 |

**A comparability caveat that the literature routinely drops, and which we
initially dropped ourselves.** These two rows are over *different domains*. At
**equal 64-bit field width** LowGear needs two CRT limbs, giving 384 B/triple
against OT's 3,074 — an **8.0×** communication advantage, not the 16× a naive
reading of the table suggests. Our own `experiments.md` quoted 16.3× and our
README quoted "192 vs 3074" without qualification; both compare across widths.
This is precisely the kind of error a single-harness study is positioned to
catch, and we correct it here.

The measured OT cost also validates the analytic model: 3,074 measured against
3,136 predicted, a 2% error.

---

## 6. Methodology

This section is the part we would most like other artifacts to adopt.

### 6.1 Identical transport, uniform accounting

Every protocol — arithmetic MPC, garbled, Falcon, hybrid — runs over the same
transport stack, wrapped in counting and latency decorators that observe traffic
without touching protocol code. Both legs of any comparison pay real
serialization, real message passing and real byte counting. There is no
measurement asymmetry in which one side is simulated and the other is not.

This matters more than it sounds. The in-process simulator **is not the
protocol**: its share multiplication reconstructs and reshares locally, which is
free and insecure, and it understated the pure-MPC baseline by ~2.7× (6.7 ms
simulated against 18 ms real). Every MPC number in this study is the real wire
protocol with three party threads exchanging serialized messages.

### 6.2 A validated network model

`NetworkProfile` charges `rounds × RTT + bytes / bandwidth`, letting a benchmark
"move" parties between continents without deploying anything. The model is
**validated against a real loopback TCP socket**: worst error 13.6%, under 1%
through the middle of both sweeps, with the link's constants measured rather
than assumed. So cost really is linear in rounds and linear in bytes with no
missing cross-term — which is what licenses computing wall-clock from measured
counts.

Building this validation found two harness bugs that had been silently
distorting results: per-message latency timers could **reorder** back-to-back
messages, and the bandwidth model did not serialize bulk transfers cumulatively,
**undercharging exactly the byte-heavy phases** the crossover experiments are
about. Both now have regression tests. We mention this because a cost model that
has never been checked against a socket should be assumed to have bugs of this
kind.

The honest limitation: the constants are loopback's (RTT 23 µs), not a WAN's, so
a WAN deployment needs its own two constants.

### 6.3 Bit-exactness as a gate on quoting cost

**A cost comparison between routes computing different things is meaningless.**
Every backend is therefore checked against a common plaintext reference before
any number is reported, and the benchmarks refuse to present a clean table when
a cell disagrees. The encrypted CNN reproduces the plaintext prediction on
600/600 embedded medical images, run twice — before and after two optimizations
— with both passes agreeing on all 600. That is what makes those optimizations
*exact* rather than merely close.

### 6.4 Machine-checked security attestations

Every engine attests the security level it actually implements, in one auditable
table, and asking a session for more than an engine attests is an **error rather
than a silent downgrade**. Exactly one engine attests malicious security. The
editing rule is that an attestation records what the code *does*, never what the
roadmap intends; raising one is the last step of an upgrade, after the
cheating-detection tests pass.

We consider this the most transferable idea in the artifact. Security level is
the dimension most often lost when numbers move between papers, and making it a
machine-checked property of the running code — rather than prose in a
limitations section — is cheap to implement and hard to get wrong by accident.

### 6.5 Recording withdrawn results

Negative, superseded and withdrawn results are kept in the experimental record
alongside the wins, with the reasoning that produced them. §9 summarizes what
this caught.

**And the losing component is deleted, while its result is kept.** This is the
half that is easy to skip. An artifact that accumulates every paradigm it ever
measured becomes unmaintainable and, worse, misleading — a reader assumes
everything present is recommended. So when a measurement condemns a component, we
remove the code and retain the number:

| removed | the measurement that condemned it |
|---|---|
| single-server PIR | loses to downloading the database by ~13×, with **no crossover in N** |
| SEAL/CKKS backends | approximate comparison costs seconds per operation with real error, against milliseconds and bit-exact under MPC; and 3.4 MB of vendored C++ |
| NTL polynomial backend | the hand-written NTT beat it 1.8× at every size, once profiling NTL revealed the Shoup butterfly it was missing |
| an unsound ZKP placeholder | its `verify` never checked constraint satisfaction — see §9 |

Two consequences worth stating. First, **the numbers in this paper for a removed
component are not reproducible from the current artifact**; §II.4 and §VI.3 of the
experimental record say so at the point of measurement. Second, where a removed
measurement still feeds a live claim, it survives as the smallest thing that can
carry it: the levelled-competitor bootstrap counts in §5.4 come from the level
arithmetic of the CKKS sign approximation, which now lives as a 40-line
dependency-free module with its figures pinned by a test — precisely because the
apparatus that produced them is gone.

---

## 7. Cross-paradigm findings

These are the results that motivated the single-harness approach, in the sense
that no single paradigm's literature could produce them.

### 7.1 The FHE-vs-MPC crossover belongs to the baseline's preprocessing

The standard claim is that FHE overtakes MPC past some layer width, because MPC
preprocessing grows linearly while a packed ciphertext is flat. Baru's own cost
model predicts it, and measurement confirms it — *against an OT-preprocessed
baseline*:

| width | dealer (lower bound) | semi2k (OT) | **lowgear (SHE)** | hybrid/BFV |
|---|---|---|---|---|
| 64 | 600.6 ms | 694.5 ms | **606.2 ms** | 692.1 ms |
| 1024 | 603.0 ms | 942.8 ms | **623.4 ms** | 692.1 ms |
| 4096 | 610.9 ms | 1.74 s | **678.4 ms** | 733.7 ms |

Against **SHE-preprocessed** MPC the crossover is *gone in this range entirely*.
LowGear tracks the trusted-dealer lower bound within 5 ms at width 64.

The honest statement is therefore not "the hybrid beats dishonest-majority MPC
past width 64" but **"the hybrid beats *OT-preprocessed* dishonest-majority MPC
past width 64, and loses to SHE-preprocessed MPC at every width measured."** The
mechanism survives — preprocessing volume decides it — but the boundary is a
property of the baseline, not of either cryptosystem.

A second boundary moves with the **threat model**. Under an honest majority
(ABY3, multiplication from correlated randomness, zero preprocessing) the layer
stays on MPC at *every* width. Under a dishonest majority with OT preprocessing,
FHE takes it past ~256 multiplications. This explains a structural fact about
the literature that is otherwise puzzling: **every hybrid FHE+MPC inference
system is 2-party.** Gazelle, Delphi and Cheetah sit precisely in the regime
where the model says the hybrid pays.

### 7.2 Protocol assignment can be derived, not annotated

Given measured costs for each backend, a search over per-operation-kind
assignments returns `mul → fhe`, `cmp-lt → binary` for a 256-wide encrypted
layer with a comparison — that is, it *derives* the Gazelle/Delphi/Cheetah
architecture rather than having it annotated. The prior work on protocol
assignment (HyCC → Silph → SING) is **MPC-only**; extending the cost model to
span MPC *and* FHE is what makes the hybrid architecture expressible as an
output of the search rather than an input.

Two structural non-linearities make this objective non-separable, which is worth
noting for anyone tempted to formulate it as an ILP: a dependence level costs
the **maximum** over its instructions, not their sum; and conversions **batch
per level**, so a crossing's cost depends on whether another crossing already
happens there.

### 7.3 The nonlinearity crossover, and three wrong answers on the way to it

Baru implements three interchangeable comparison protocols — batched bit
decomposition, garbled circuits, and Falcon's `wrap3`-based DReLU — behind one
interface, verified to compute the same function bit-exactly. Latency per encrypted
MedMNIST diagnosis, with **both** protocols charged for their preprocessing and
neither paying for serialisation padding (2026-08-03):

Medians of three repeats, on a route that is Falcon end to end and privately so
(2026-08-04 — the earlier single-run grid measured a Falcon that used `a2b` for its ReLUs
and, once it did not, leaked; see the truncation note above):

| RTT | 40 | 30 | 20 | 15 | 10 | 8 Mbps |
|---|---|---|---|---|---|---|
| **0 ms** gc | **6552** | 8453 | 12337 | 16639 | 24269 | 29982 |
| **0 ms** falcon | 7255 | 8607 | **11859** | **15109** | **20979** | **24717** |
| **20 ms** gc | **8157** | **10098** | **13925** | **17770** | **25744** | 31504 |
| **20 ms** falcon | 13572 | 14969 | 17860 | 20784 | 27476 | 31836 |

**Crossover between 20 and 30 Mbps at RTT 0; no crossover down to 8 Mbps at RTT 20 ms.**
Garbled circuits win above, Falcon below; higher RTT moves the boundary *down*, shrinking
Falcon's region. The mechanism is visible in the same run: Falcon sends 3.4 MB against gc's
10.1 (2.94× cheaper) in 450 messages against 61 (7.4× dearer), so the two cost terms trade
and the boundary is where the trade balances.

**There are two boundaries, and a warm server's sits a decade of bandwidth higher.**
Falcon's preprocessing is query-independent, so a deployment that generates it before the
query arrives pays only the online phase. Measured per MPC segment (pneumonia):

|  | RTT 0 ms | RTT 20 ms |
|---|---|---|
| cold — server pays the total | crossover 20–30 Mbps | none down to 8 Mbps |
| warm — preprocessing hoisted | falcon wins to ≥100 Mbps | crossover 20–40 Mbps |

Garbled circuits have **no** query-independent phase to hoist here, so their column is
identical warm and cold. That asymmetry is what the comparison is about, and it is also why
the warm boundary must be quoted *alongside* the cold one and never instead of it —
otherwise it becomes one protocol's online cost against another's total, which is exactly
how our first crossover went wrong. Online MPC latency falls 2.9–3.2× across the three
datasets, on pools of 7 137 / 14 154 / 7 142 comparisons split across four segments.

**The resolution is ±1 grid step, and the repeats do not improve that.** They re-run inside
one session on warm caches and spread under 2%, but an *unchanged* `gc` configuration
re-run as a separate process moved a cell by 13%. So the 30 Mbps and 8 Mbps cells — 1.8%
and 1.1% apart — are ties in substance whatever the marker says.

**This section previously reported three other answers, and the sequence is the more
useful result.** Each was reproducible, computed the correct diagnosis, and agreed
with our cost model; the first even matched a break-even prediction of 17.1 Mbps
against 15–20 measured.

| reported | why it was wrong |
|---|---|
| 15–20 Mbps | Falcon's preprocessing was dealt from a shared seed, free because it was not secret — so this compared Falcon's *online* cost against garbled circuits' *total* |
| no crossover at any bandwidth | with a real offline phase Falcon measured 32.5 MB against 10.1 and looked dearer in both terms; **82% of those bytes were padding** |
| a tie, no boundary worth quoting | the grid stopped at 10 Mbps, above the boundary |

Two things are worth extracting. First, **the first answer was wrong in two
directions at once**: dealt preprocessing flattered Falcon while `u128` padding
penalised it, and correcting both moves the boundary *up* to 30–40 Mbps — Falcon wins
a **wider** region than the original claim, for entirely different reasons. The
errors had partly cancelled, which is why the number looked plausible.

Second, and this is the methodological point: **neither bit-exactness gating nor
cost-model agreement detects any of the three.** Both mechanisms confirm that a
computation is correct and that its cost is explicable. Neither asks whether two
measurements are *commensurable*. What caught these was decomposing each total into
the terms the model charges — phases, messages, bytes, and the padded share of bytes
— and checking that both protocols were charged for the same things. The sweep now
prints those terms beside every table.

A consequence we now apply generally: **a spot check is not evidence about an
encoding.** Narrowing the ring wire to a fixed 21 bits was bit-exact for the `2^20`
dataset and returned the wrong diagnosis for *every* image of the `2^22` one — the
floor is a property of the model, not a constant. Only the full-scan check over every
embedded image of both datasets caught it.

**What this cost, and what it bought.** Fixing the encoding took `a2b` from 29.9 MB
to 5.3 and Falcon from 10.1 to 3.5, leaving the garbled route untouched at 10.1
because it is garbled-dominated. So the headline configuration's figures are
unchanged, while the *ranking* of the three routes on bytes reversed: `a2b`, which
§7 previously dismissed as never winning, is now the middle route rather than the
worst by 3×.

---

## 8. Case study: private medical image diagnosis

MedMNIST v2 distributes twelve biomedical datasets at 28×28 — exactly MNIST's
shape — so a private-inference pipeline runs on them unchanged, and the artifact
stops being about handwritten digits.

| | PneumoniaMNIST | OrganAMNIST | DermaMNIST |
|---|---|---|---|
| task | chest X-ray, pneumonia vs normal | abdominal CT, 11 organs | dermatoscopy, 7 lesions |
| input | 1×28×28 | 1×28×28 | **3×28×28 (RGB)** |
| integer accuracy, full test set | 84.78% (624) | **82.93%** (17,778) | 72.12% (2,005) |
| 95% Wilson CI | [81.7, 87.4] | [82.4, 83.5] | [70.1, 74.0] |
| balanced accuracy / AUC | 80.64% / **0.9314** | **82.21%** / **0.9605** | 30.14% / **0.8519** |
| sensitivity / specificity | 97.18% / **64.10%** | — | — |
| published ResNet-18 (28×28) | 0.854 / 0.944 | 0.935 / 0.997 | 0.735 / 0.917 |
| majority-class baseline | 62.50% | 18.48% | **66.88%** |
| **encrypted == plaintext** | **300/300** | **300/300** | **300/300** |
| FHE bootstraps | 0 | 0 | 0 |
| online latency / diagnosis | **268 ms** | 291 ms | 271 ms |

DermaMNIST is read on balanced accuracy and AUC, not accuracy: its largest class is
66.88% of the test set and the published ResNet-18 beats that by under seven points,
so accuracy carries almost no signal at that skew.

**Colour is worth having and costs nothing in ciphertext multiplications.** Against a
grayscale-folded control trained identically, RGB gives AUC **0.8519 against 0.7667**
and balanced accuracy **30.14% against 23.30%** — dermatoscopy is where colour should
carry diagnostic signal, and it does. The packing takes it for free: an RGB 28×28
convolution needs degree 3977 against `n = 4096`, so it fits one ciphertext at **one
multiplication per output channel**. Three channels sum on shared coefficients rather
than costing three times the work, which is why the RGB row's latency matches the
grayscale ones.

Three points a privacy venue should care about.

**Privacy costs nothing in accuracy, and this is shown rather than asserted.**
The integer model matches its float parent to within 0.2 points, and the
encrypted pipeline reproduces the plaintext diagnosis bit-exactly on every one
of 600 images.

**Accuracy alone misreads a medical classifier, and a privacy paper reporting
only accuracy is reporting badly.** PneumoniaMNIST's test set is ~60% positive,
and this model earns its 84.78% largely by *over-calling disease*: 97.2%
sensitivity against 64.1% specificity. Balanced accuracy is 80.64%. None of this
is visible while only accuracy is reported, and none of it is a cryptographic
result — but a paper claiming private diagnosis has to report it.

**The accuracy ceiling is the model, and specifically the packing, not the
crypto.** We initially attributed the model's size to the FHE noise budget. That
was wrong. Three limits bind, and noise is the third: spatially, 3×3 valid
convolution plus pooling takes 28 → 13 → 5 → 1, admitting at most three convs;
by *packing*, `2·in_c·H·W ≲ n` caps the first conv at 12 channels before the
second conv's input passes `n = 4096`. Depth, meanwhile, really is free.

**The packing limit is now lifted, and without growing the ring** (2026-08-04).
Convolution is linear in the input channels, so a layer too wide for one ciphertext
evaluates as equal channel groups whose products are **added** — free homomorphically,
needing no evaluation key and no depth. Doubling to `n = 8192` would also have worked
and would have charged every operation in the pipeline; splitting charges only the
layers that overrun.

**And it buys accuracy — on one of the two datasets.** We retrained both at
`conv(16,32)`, whose second convolution takes 16 channels at 13×13 (degree 5266
against `n = 4096`) and therefore runs as two groups of 12:

| | pneumonia (4 708 train) | organa (34 561 train) |
|---|---|---|
| conv(8,16) baseline | **84.78%** [81.7, 87.4] | 81.64% [81.1, 82.2] |
| conv(16,32), split | 83.97% [80.9, 86.6] | **82.93%** [82.4, 83.5] |
| verdict | no gain — CIs overlap, validation rose while test fell | **+1.29 points, CIs disjoint** |

Both are bit-exact under encryption with zero bootstraps; the split costs conv2 twice
the ciphertext multiplications (32 → 64) and no evaluation key.

**So "the packing was the accuracy ceiling" is true conditionally, and the condition
is data.** On organa, 34 561 training images can use the extra capacity and the
packing limit really was what bound. On pneumonia, 4 708 cannot: the wider model
overfits — its validation accuracy *improves* while its test accuracy falls — so the
ceiling there is the dataset, and no amount of packing headroom would have moved it.
Reporting only one of these experiments would have been misleading in whichever
direction it pointed.

---

## 9. What the methodology caught in our own work

A survey's credibility rests partly on what its authors were willing to
withdraw. Each item below was found by the discipline in §6 and is recorded in
the artifact rather than quietly fixed.

**A ZKP backend that proved nothing.** Until 2026-07-31 the shipped `binius64`
backend's `verify` never checked that a witness satisfied any constraint —
anyone holding the public circuit could forge an accepting proof. It is retained,
unused, so a soundness test can keep demonstrating precisely what it failed to
do. **Every ZKP number predating that date is withdrawn.**

**A privacy break in bit decomposition.** The previous `a2b` masked its input
with a value derived from a *public* lockstep counter, so the input was
recoverable from the transcript by any observer. Everything built on it — ReLU,
max-pool, argmax — leaked its activations. The advertised "no extra
communication per bit" was a *symptom*: the carry chain was free because every
operand was already public. Fixing it cost 4.4× MPC time and ~130× bandwidth,
later recovered by triple-free multiplication and batching.

**A truncation whose mask was sized for correctness, not for privacy.** Falcon's
exact truncation opens `c = a + r`. Its stated precondition is that `a + r` must not
wrap — a *ceiling* on the mask — and the first pure-Falcon ReLU met it the cheapest
way, with `r` no wider than the value. That is correct and hides almost nothing: the
posterior on `a` given `c` collapses to a point at `c = 0`, and for uniform `a` the leak
is ≈0.72 bits per truncation. The aggregate over an image was not measured — activations
are correlated, so the per-truncation figures do not add — but the size of the number is
not what matters. The `a2b` route it replaced leaks **nothing**, so making the route
"purer" replaced a strictly private construction with a less private one.

Nothing in the test suite could have caught it — leakage is not a wrong answer — and
the three tests covering truncation preprocessing all passed a mask narrower than the
ring, so none ever exercised a wrapping one. The fix is not slack but a **wrap bit**:
let `r` cover the whole ring, making the opened value uniform and *perfectly*
independent of `a`, then correct the resulting overflow with `[c < r]`, which is
computable from a public `c` and a secret `r` and costs one narrowed `a2b` and one
`Π_SS` lift. The alternative — `κ` bits of statistical slack — needs a ring wider than
the narrowed wire encoding permits and measured 1.9× *dearer* in bytes than `a2b`. The
narrow request is now unrepresentable rather than merely documented.

**An insecure FHE scheme, removed rather than caveated.** The fast engine was a
scale-invariant NTRU port. Analysis showed it insecure at the parameters the
construction needs even at depth 1 — the large LTV secret raises the fatigue
point to ≈2^59 but a single multiply with usable `t` forces `q ≳ 2^84`. It was
deleted, not kept as a "fast reference"; every result transferred bit-exact and
the FHE column got slower.

**A selective-failure channel that passes every functional test.** Attaching
MACs to bit-decomposition leaves by multiplication is unsound: a corrupt party
injects `δ`, the first downstream gate carries MAC error `δ·y` with `y` a secret
bit, so the run aborts when `y=1` and completes when `y=0`. **Deferring the
abort does not close this** — batching hides *when* an abort happens, not
*whether* it happens. We had asserted the opposite before building it.

**A crossover reported wrong three times, for three different reasons.** The
boundary in §7.3 was published at ~15 Mbps (Falcon's *online* cost against garbled
circuits' *total*, because Falcon's preprocessing was dealt from a shared seed), then
as no boundary at all (Falcon inflated 3.2× by `u128` serialisation padding), then as
a tie (the grid stopped above it). It is at **30–40 Mbps** at RTT 0.

**All three passed our bit-exactness gate and agreed with our cost model**, and the
first matched a break-even prediction of 17.1 Mbps against 15–20 measured. That is
the uncomfortable finding: those two mechanisms establish that a computation is
correct and that its cost is explicable, and neither asks whether two measurements
are commensurable.

The first answer was also wrong in *both* directions at once — dealt preprocessing
flattered Falcon, padding penalised it — so the errors partly cancelled and the
result looked mechanistically sound. Correcting both moved the boundary up rather
than down.

One rule came out of it that we now apply generally: a spot check is not evidence
about an encoding. Narrowing the ring wire to a fixed 21 bits was exact for the
`2^20` dataset and wrong for **every** image of the `2^22` one, because the safe
width is a property of the model rather than a constant.

**A byte comparison that was measuring our serialisation.** For a year the
comparison-primitive experiment reported `a2b` at 5.94 MB against garbled circuits'
1.69 MB, i.e. garbling 3.5× cheaper in bytes as well as faster. Ring shares were
serialised as `u128` whatever their width, so a 20-bit value travelled in 128 bits;
`a2b` is ~99% ring traffic and garbled tables are already compact, so the overhead
fell on one side only. Re-measured with the encoding fixed, `a2b` is **1.29 MB** and
is the byte-*cheaper* route. **Garbling's byte advantage is withdrawn; its round
advantage stands and is larger than we reported** (18.8× on a LAN).

A control run at the old width reproduces 5 938 236 B exactly, so the benchmark was
never unstable — it was faithfully measuring a quantity nobody meant to compare. The
same encoding suppressed the §7.3 crossover by 15×, and its cost had been estimated
in advance at 5.9× and dismissed as not worth fixing because "the headline path is
garbled-dominated". That reasoning was sound for the headline path and wrong for every
comparison the padding was actually load-bearing in.

**Comparison errors of framing.** Pricing FHE's *linear* part against MPC's
*whole layer* (omitting the ReLU that FHE cannot do at all); computing a
crossover against a trusted-dealer baseline (asking whether FHE beats free
preprocessing); and the cross-domain triple comparison corrected in §5.6.

**Honest negatives retained.** PIR loses to downloading the database by ~13×
with no crossover in `N`. Hybrid-TFHE does not pay. Faster ring arithmetic does
*not* bring FHE close to MPC. Per-layer bit-width tuning was measured at ~10% and
not built. A fused garbled max-of-4 was measured as a 1.68× *loss* and not built.

---

## 10. Threats to validity

**Implementation quality.** Research Rust, not tuned C++. Absolute figures are
not competitive; ratios measured within the harness are the defensible output.
Where we compare against published numbers — e.g. our sign extraction at
~0.043 ms against Falcon's optimized C++ — the gap is implementation, not
protocol, and we say so.

**Single machine.** All figures from one Apple M1 Pro. Cross-machine variance is
uncharacterized. The network model is validated on loopback, so WAN constants
are modelled.

**The simulated network is a model.** Validated to 13.6% worst case, but it is
still `rounds × RTT + bytes / bandwidth`, not a real WAN with congestion, jitter
or packet loss.

**One disclosed cryptographic gap.** FHE threshold *key generation* is
dealer-based. It is recorded on the code and in the attestation table rather than
only in prose.

The Falcon backend's preprocessing used to be the second such gap — dealt from a
common seed, so all three parties could reconstruct the mask. It is now generated
by bit composition (`random_wrap_masks`), which closed the gap and cost the §7.3
crossover its validity: preprocessing turned out to cost an `a2b` per element, the
protocol Falcon's online phase exists to avoid.

**No offline/online split.** Related, and the limit that now matters most for §7.3:
`lt_falcon_batch` runs preprocessing inline, so every latency we report is a
*total*. Protocols whose design puts work offline — Falcon among them — are
therefore charged for it here in a way their own papers do not, and we cannot
currently produce the online-only figure that would make the comparison fair.

**Workload breadth.** Findings are anchored on feed-forward CNN inference at
28×28. Transformers, larger images, and non-inference workloads are not covered.

**Not an adversarial evaluation.** Cheating-detection tests drive real attacks
over the transport for the malicious ABY3 layer, but this is not a security
audit, and no formal proofs are machine-checked.

---

## 11. Open problems

1. **A CIFAR-scale private CNN** needs SIMD-slot packing; coefficient packing
   bounds image size. The composition and security story is independent of it.
2. **Transformer nonlinearities** (softmax, LayerNorm) need exp/reciprocal/rsqrt
   subprotocols.
3. **Malicious security at the FHE boundary.** Replication cannot catch a
   *malformed input ciphertext* — both honest co-holders compute on it faithfully
   and agree. This sharpens the positioning: **ZK is needed only at the input
   boundary, never for the evaluation**, where single-server verifiable FHE needs
   it over the whole computation. The boundary proof is not implemented.
4. **Secret-index array access**, priced by the cost model, before ORAM. CirC
   implements exactly the two cases either side of that line and does not
   implement ORAM at all.
5. **Per-instruction protocol assignment**, versus the current per-operation-kind
   granularity.
6. **Cross-machine and real-WAN validation** of the cost model's constants.

---

## 12. On framing: survey, SoK, or measurement study

A note on positioning, since it affects venue choice.

This document is **not a pure survey**, and framing it as one would be a
strategic error. A survey's contribution is organization and coverage; reviewers
judge it on completeness of the literature. This document's distinguishing asset
is different: it contains **original cross-paradigm measurements** in a
controlled harness, and several of its findings *contradict or qualify* claims
in the literature — including our own. Coverage of the literature is real but
not exhaustive.

The right framing is a **Systematization of Knowledge with an artifact**, whose
thesis is:

> Cross-paradigm performance claims in privacy-preserving computation are not
> currently falsifiable, because the numbers come from incommensurable sources.
> Here is a harness in which they are, here is what changes when you re-measure
> under uniform accounting, and here are three claims — including two of our own
> — that do not survive it.

That thesis is defensible, is supported by the measurements, and gives reviewers
something to disagree with, which a survey does not. Recommended venues, in
order:

1. **PoPETs / PETS, SoK track** — best fit; values artifacts, and the
   methodology section is the kind of contribution the venue rewards.
2. **ACM Computing Surveys** — only if reframed toward coverage and away from
   original measurement, which we would advise against.
3. **IEEE TDSC** or **Journal of Cryptographic Engineering** — good fits for the
   measurement-study framing.
4. **USENIX Security / CCS artifact-oriented submission** — possible, but these
   prefer a single sharp result over breadth; the no-bootstrap construction
   (§5.4) is the better candidate for that route, and is a *separate* paper.

**Relationship to the other papers in this repository.** The no-bootstrap deep
inference result and the Falcon reproduction are each a distinct paper with a
single sharp claim. This document is deliberately the complement: breadth,
methodology, and the cross-paradigm comparisons no single-claim paper has room
to make. It should cite them rather than absorb them.

---

## Reproducibility

Single Rust crate, no GPU. `cargo test` covers correctness; every `bench-*`
subcommand emits `--json`. All numbers are produced by these commands; none are
hand-entered.

| Result | Command |
|---|---|
| Deep inference, zero bootstraps | `bench-deep-nn --depths 2,4,8,16,32` |
| Private diagnosis | `bench-medmnist --dataset pneumonia` |
| Nonlinearity crossover (§7.3) | `bench-medmnist --sweep` |
| Cost of active security | `bench-malicious --n 32 --bits 32` |
| Triples: SHE vs OT | `bench-triples` |
| FHE↔MPC crossover | `bench-layer` |
| Cost model vs real socket | `bench-net` |
| Proving a quantized neuron | `bench-zkp-layer --widths 1,4,16,64,256,784` |
| Falcon vs a2b vs garbled | `bench-falcon` |

Full provenance, including the corrections behind each row, is in
[`experiments.md`](experiments.md).
