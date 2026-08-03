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
preprocessing; active security on the nonlinear surface of private inference
costs 2.63× bytes but only 1.06× rounds; and the choice between a garbled and a
`wrap3`-style comparison protocol has a *measured* crossover in bandwidth that
moves with RTT. We also report our own withdrawn claims, including a ZKP backend
that proved nothing and an MPC bit-decomposition that leaked its inputs, because
a survey that reports only what worked is not reporting.

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
- A **single-artifact measurement study**: ~72,000 lines of Rust in which one
  program runs on every paradigm, with a validated network model and uniform
  accounting.
- **Cross-paradigm findings** that are not visible from any single paradigm's
  literature (§7), including two crossovers whose *location* turns out to depend
  on a parameter the original claims left implicit.
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

| workload | bytes | interactive calls | wall clock (WAN, 25 ms) |
|---|---|---|---|
| **comparisons, n=32, 32-bit** | **2.63×** | **1.06×** | **1.26×** |
| dot product, n=32, 32-bit | 2.47× | 1.75× | 2.80× |

The comparison row is the one that matters for inference, since every ReLU,
max-pool and argmax reduces to a comparison. **Active security costs bytes, not
rounds**: the MAC doubles what is sent but batches into the same round. The
dot-product row is a small-workload artifact — the semi-honest leg needs only 4
interactive calls, so a fixed pair-pool cost is amortized against nothing — and
we report it rather than hide it.

A second result generalizes: the statistical slack that ring-based verification
normally charges as a widening to `Z_{2^{k+s}}` is **already paid for** when the
implementation shares over a ring much wider than its values. Q16.16 values in a
128-bit ring leave **59 bits** of statistical security at no widening cost.

### 5.2 Garbled circuits

In the *3-party honest-majority* setting, garbled-circuit input labels are
delivered by the co-garbler with no oblivious transfer. Since OT is the
expensive part of 2-party garbling, this changes the economics substantially,
and it is the reason garbling is competitive here at all.

Measured head-to-head against bit decomposition through the identical transport
(1352 comparisons, 20-bit): `a2b` needs 9 rounds and 5.94 MB; the garbled
circuit needs 1 round and 1.69 MB. Wall clock favours garbling 4.2× on a LAN and
4.9× on a 20 ms WAN.

**An important correction to our own earlier claim.** That 4.2–4.9× is the
*garbling primitive in isolation*. The deployable 3-party protocol, including
the Y2A return leg that converts the boolean output back to an arithmetic share,
is **1.4–1.7×**, worth 1.9× on the CNN's MPC segment. We record both because the
gap between them is exactly the kind of thing that gets lost in citation.

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

### 7.3 The nonlinearity crossover is in bandwidth, and moves with RTT

Baru implements three interchangeable comparison protocols — batched bit
decomposition, garbled circuits, and Falcon's `wrap3`-based DReLU — behind one
interface, verified to compute the same function bit-exactly. This makes the
Yao-vs-GMW trade of §3 directly measurable on a real workload rather than
argued. Latency per encrypted MedMNIST diagnosis (2026-08-01):

| RTT | 1000 | 100 | 40 | 30 | 20 | 15 | 10 | 5 Mbps |
|---|---|---|---|---|---|---|---|---|
| **0 ms** gc | **638** | **3227** | **6554** | **8511** | 12357 | 16215 | 23915 | 46869 |
| **0 ms** falcon | 1279 | 3528 | 6750 | 8583 | **12167** | **15740** | **22828** | **43794** |
| **20 ms** gc | **2411** | **4839** | **8395** | **10292** | **14174** | **18046** | 25594 | 48659 |
| **20 ms** falcon | 3873 | 6276 | 9585 | 11393 | 14998 | 18431 | **25543** | **46850** |

Falcon — the lowest-bandwidth route — wins below **~20 Mbps at RTT 0** and below
**~10 Mbps at RTT 20 ms**. Higher RTT shifts the crossover *down*, shrinking the
region where it wins, because it sends more messages. The mechanism checks out
quantitatively: Falcon starts ~641 ms behind on compute and pays 67 extra
messages, against which it saves 1.6 MB; break-even is predicted at 20.0 and 6.5
Mbps against 20–30 and 10–15 measured.

**The boundary is a band, not a line.** Around 20–30 Mbps the winner flips
between repeat runs (0.1–0.3% spread). No ordering inside that band should be
quoted; the finding is that the protocols are *indistinguishable* there, which
is what a crossover is.

The generalizable claim: given a deployment's RTT and bandwidth, one should
**pick** the nonlinearity protocol rather than fixing it. Systems that hardwire
one choice are optimal only in part of the deployment space.

---

## 8. Case study: private medical image diagnosis

MedMNIST v2 distributes twelve biomedical datasets at 28×28 — exactly MNIST's
shape — so a private-inference pipeline runs on them unchanged, and the artifact
stops being about handwritten digits.

| | PneumoniaMNIST | OrganAMNIST |
|---|---|---|
| task | chest X-ray, pneumonia vs normal | abdominal CT, 11 organs |
| integer accuracy, full test set | 84.78% (624) | 81.18% (17,778) |
| 95% Wilson CI | [81.7, 87.4] | [80.6, 81.8] |
| balanced accuracy / AUC | 80.64% / **0.9314** | 80.49% / **0.9590** |
| sensitivity / specificity | 97.18% / **64.10%** | — |
| published ResNet-18 (28×28) | 0.854 / 0.944 | 0.935 / 0.997 |
| majority-class baseline | 62.50% | 18.48% |
| **encrypted == plaintext** | **300/300** | **300/300** |
| FHE bootstraps | 0 | 0 |
| online latency / diagnosis | **268 ms** | 291 ms |

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
second conv's input passes `n = 4096`. That is a packing constraint that relaxes
at `n = 8192`, not a limit of the construction. Depth, meanwhile, really is free.

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

**Two disclosed cryptographic gaps.** The Falcon backend's preprocessing is
currently **dealt from a common seed and is not secure** — all three parties can
reconstruct the mask; its online protocol and measured costs are real, its
offline phase is not. FHE threshold *key generation* is dealer-based. Both are
recorded on the code and in the attestation table rather than only in prose.

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
| Nonlinearity crossover | `bench-medmnist --sweep` |
| Cost of active security | `bench-malicious --n 32 --bits 32` |
| Triples: SHE vs OT | `bench-triples` |
| FHE↔MPC crossover | `bench-layer` |
| Cost model vs real socket | `bench-net` |
| Proving a quantized neuron | `bench-zkp-layer --widths 1,4,16,64,256,784` |
| Falcon vs a2b vs garbled | `bench-falcon` |

Full provenance, including the corrections behind each row, is in
[`experiments.md`](experiments.md).
