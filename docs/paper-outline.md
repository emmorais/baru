# Paper outline: Bootstrapping-free deep private inference via scheme switching

Working title candidates:
- *"The Switch Is the Bootstrap: Unbounded-Depth Private Inference without FHE
  Bootstrapping"*
- *"Interactive Refresh: Deep FHE Inference at Cost Linear in Depth"*

This is a working outline mapping the claims to the artifact in this
repository (`baru`). It is written to be honest about what is implemented,
what is simulated, and what is analysis — the distinctions a program
committee will probe.

---

## 1. One-sentence thesis

In a hybrid FHE+MPC inference pipeline, the scheme switch that already
exists to compute each nonlinearity **also refreshes the FHE ciphertext to
a fresh, depth-1 state**, so a network of *arbitrary depth* is evaluated
with **zero FHE bootstrapping** and total cost **linear in depth** — using
an FHE engine (scale-invariant NTRU / LTV) that supports only a *single*
homomorphic multiplication.

## 2. Two claims, stated separately (reviewers will separate them anyway)

- **C1 — Switch-as-bootstrap.** The MPC round trip performed for a ReLU
  returns the value to FHE via fresh re-encryption. Formally: after
  `shares_to_fhe`, the ciphertext is distributed identically to a fresh
  encryption (`key_pow = 1`, independent fresh noise). Therefore per-layer
  FHE noise is bounded by the depth-1 budget *independent of network
  depth*. Bootstrapping — the operation leveled FHE uses to reset noise —
  is never invoked. **This is the contribution.** It is scheme-agnostic:
  any additively-homomorphic FHE with a cheap depth-1 multiply instantiates
  it.
- **C2 — Both-operands-private linear layers.** Instantiating the FHE side
  with scale-invariant NTRU and reversed-coefficient packing, a full matvec
  with *both* the weight matrix and the activation vector encrypted is a
  single ciphertext×ciphertext multiplication, decrypted under `f²` (no
  relinearization). This strengthens the privacy model of prior hybrid
  systems (Gazelle/Delphi/Cheetah keep the model in plaintext).

Papers can lead with C1 (the general result) and present C2 + SI-NTRU as
"a concrete, unusually cheap instantiation."

## 3. Why C1 is true — proof sketch

Let `Enc` be the FHE scheme, `x` the secret value entering a layer as a
ciphertext, `L(·)` the linear map (plaintext or encrypted weights), `σ` the
ReLU. The layer computes `σ(L(x))`:

1. `c = Enc(x)` — fresh, noise `ν₀`, one available multiplicative "step".
2. `c' = L(c)` — one multiplication; noise grows to `ν₁ = poly(ν₀)`; for
   SI-NTRU the product decrypts under `f²` (`fhe::si_ntru`), still within
   the depth-1 budget (measured slack ≈ 36 bits at the paper parameters).
3. **Switch to MPC** (`fhe_to_shares`): parties mask, one masked
   ciphertext is decrypted, additive shares result. The value now lives in
   MPC; *the ciphertext noise is discarded.*
4. `σ` is computed on the shares (any MPC-friendly nonlinearity).
5. **Switch back** (`shares_to_fhe`): each party freshly encrypts its share
   component; the evaluator sums. The sum of `n` fresh encryptions is a
   fresh encryption (linear in fresh ciphertexts), noise `≈ n·ν₀`, and
   `key_pow = 1`. **This is indistinguishable from a layer-1 input.**

Induction over layers: every layer begins in the same fresh state, so the
FHE noise bound is a per-layer constant. No dependence on `L`. ∎

Artifact: the invariant is `assert`-verified indirectly by *exact*
plaintext agreement at every measured depth (`bench-deep-nn`, `ok` column) —
if noise had accumulated, deep outputs would diverge; they do not.

## 4. Why the values stay correct — the fixed-point argument

Independent of noise, integer activations must not overflow the share
modulus `2^b`. The artifact uses a fused nonlinearity
`a' = ReLU((W·a + (β≪s)) ≫ s)` where the `≫ s` rescale holds activation
magnitude ~constant per layer. Two subtleties worth a paragraph each:

- The rescale is a **logical** shift, yet the value can be negative before
  ReLU. Correctness holds because the shift result is *kept only on the
  non-negative branch* of the ReLU (where logical = arithmetic); on the
  negative branch the ReLU zeroes it. Hence a single `a2b` yields both the
  sign bit and the shifted magnitude — one bit-decomposition per layer, not
  two.
- The construction is exact integer fixed-point: the encrypted result
  equals the plaintext integer network *bit for bit* (not "up to
  approximation"), which the microbenchmark reports as `correct = true`.

## 5. Cost model (what to plot)

Per layer, independent of depth:
- FHE: 1 encrypt (weights) + 1 ct×ct multiply + O(1) mask ciphertexts.
- Switch: 1 masked decryption (round trip), O(width) opened residues.
- MPC: `width` fused ReLU-rescales; each = 1 `a2b` (≈ `b` AND-depth) + a
  handful of local ops. Round complexity per layer = O(1) conversion RTT +
  O(bit-depth) for the `a2b`s (batchable).
- Communication: measured with real serialized messages (`CountingTransport`).

Headline plots:
1. **wall-time vs depth** — linear; annotate the 0-bootstrap line.
2. **our bootstraps (0) vs competitor bootstraps (∝ depth)** — the
   `competitor_bootstraps` model, grounded in *our measured*
   `CkksCompareBackend::levels_for_relu` (from `bench-ckks-compare`), with
   the per-bootstrap level budget as a stated conservative assumption.
3. **per-phase breakdown** (fhe / switch / mpc) vs depth and vs width — to
   show where cost actually lives (the MPC nonlinearity + conversion, as in
   `bench-hybrid-nn`).
4. **wall-time vs network RTT/bandwidth** — the WAN story (`--rtt-ms`,
   `--bandwidth-mbps`), reusing the corrected pipe-serializing latency
   model.

## 5b. The UC gadget library — why composition to any architecture is free

The construction's real power is *composability*: define a small set of
building blocks once, prove each secure once, and every network a user
writes by chaining them in Baru is secure and correct **by the UC
composition theorem** — no per-architecture proof.

**Baru is an Arithmetic Black Box (ABB).** The interpreter manipulates
secret values only through opaque handles (`ShareRef`, `FheCiphertext`) and
a fixed set of primitive operations; it never inspects the representation.
This is exactly the ABB abstraction (Damgård–Nielsen and the SPDZ line): an
ideal functionality `F_ABB` storing values by handle and exposing
`add / mul / lt / share_public / reveal`, plus an *internal, invisible*
representation switch (share ↔ ciphertext). A Baru program is a sequence of
`F_ABB` calls.

**The gadget library** (`src/gadgets/`) is the layer between `F_ABB` and
neural-network architectures. Every gadget is a pure function of the ABB
primitives:

| gadget | definition | primitives used |
|---|---|---|
| `relu` | `x·(0<x)` | `sub`, `lt`, `mul` |
| `max2` | `mux(a<b, b, a)` | `lt`, `sub`, `mul`, `add` |
| `max_reduce` / `maxpool2d` | fold of `max2` over a window | (above) |
| `argmax` | scan tracking `(best, idx)` via `mux` | `lt`, `mul`, `add`, `share_public` |
| `dense` / `conv2d` | `Σ W·x + b` (conv = im2col matvec) | `mul`, `add` |

Because each gadget is a composition of ABB operations, **if each primitive
UC-realizes its `F_ABB` operation, each gadget UC-realizes the
corresponding ideal gadget** (UC composition), and a full network — any
sequence of gadgets — UC-realizes the ideal inference functionality. The
simulator for a network is the composition of the per-gadget simulators.

**Consequence for the paper's scope.** We do not have to prove "private
CNN inference" as a monolith. We prove a handful of gadgets, and then
*exhibit* architectures composed from them. `src/gadgets/mod.rs` includes an
end-to-end LeNet-shaped classifier (`conv → relu → maxpool → dense → relu →
dense → argmax`) that is **bit-exact** against its plaintext reference and
reveals only the class index — evidence that the composition is not just
type-correct but semantically exact. Swapping in more channels, more conv
blocks, or residual adds requires *no* new security argument.

**This is what makes "explore combinations of building blocks" rigorous.**
Baru becomes an instrument for searching the private-inference architecture
space: any architecture expressible from the gadget library inherits
security and correctness for free, so the research question shifts from "is
*this* network secure?" to "which architecture is *fastest* under the cost
model of §5?"

## 6. Related work & positioning

- **Hybrid linear-FHE / nonlinear-MPC inference**: Gazelle (USENIX Sec
  '18), Delphi (USENIX Sec '20), Cheetah (USENIX Sec '22), Iron/BOLT/
  BumbleBee (transformers). *Difference*: these treat the switch as an
  engineering necessity and keep the model in plaintext; we (a) identify
  the switch as a **bootstrapping-elimination** mechanism enabling
  unbounded depth with a depth-1 FHE engine, and (b) encrypt the weights.
- **Scheme switching between FHE schemes**: CHIMERA, PEGASUS. *Difference*:
  they switch FHE↔FHE to pick the cheapest scheme per op; we switch FHE↔MPC
  and exploit the re-encryption as a noise refresh.
- **Threshold / multi-key FHE**: Asharov et al. (EC '12) and the MHE line —
  the honest instantiation of our masked decryption (see §8).
- **Scale-invariant NTRU / LTV / YASHE**: López-Alt–Tromer–Vaikuntanathan
  (STOC '12), Bos et al. (YASHE); reference C++ (`hom-ham`, this port). We
  use only its depth-1 regime.

## 7. Experiments (all reproducible from this repo)

| # | Command | Shows |
|---|---|---|
| E1 | `bench-deep-nn --depths 2,4,8,16,…` | correctness (exact) at every depth; 0 bootstraps; linear time |
| E2 | `bench-deep-nn --json` + competitor model | bootstraps: 0 vs ∝ depth |
| E3 | `bench-ckks-compare` | grounds the per-ReLU level cost used by E2 |
| E4 | `bench-hybrid-nn --rtt-ms … --bandwidth-mbps …` | WAN cost, single-layer crossover vs pure MPC |
| E5 | `demo-private-mnist` | end-to-end, encrypted model + image, only the class bit revealed |
| E6 | `gadgets` CNN composition test | full LeNet-shaped classifier (conv/pool/dense/argmax) composed from the gadget library, bit-exact, class-only reveal |
| E7 | `bench-cnn` | that CNN through the *real hybrid pipeline* (FHE ct×ct conv + MPC relu/pool/argmax), 0 bootstraps, bit-exact, per-phase cost under a WAN model |
| E8 | `bench-train` | private *training* (SGD, MSE loss to sidestep softmax) under the real ABY3 protocol, bit-exact vs plaintext integer-SGD, loss ↓ + 100% train acc, then private inference — the full encrypted-data→train→infer lifecycle from the gadget library |

## 8. Security model — state honestly, do not overclaim

- **Threat model**: semi-honest, honest-majority 3-party MPC for the
  nonlinear segment; the FHE evaluator is an untrusted server. Only the
  final output is revealed; intermediate activations remain shared /
  encrypted; each switch opens only a uniformly-masked value (simulatable).
- **Gaps in the *implementation* (not the design)**, disclosed as
  limitations:
  1. The masked decryption uses a **single client key** as a stand-in for
     threshold decryption. A deployable instantiation secret-shares the FHE
     key (threshold FHE) so no party can decrypt alone. The data flow is
     already correct (only masked values are opened); only the key
     operation is simulated.
  2. **SI-NTRU at the paper parameters is overstretched** (large `q` vs
     `n`), hence not IND-CPA-secure against subfield/dense-sublattice
     attacks (Albrecht–Bai–Ducas '16; Kirchner–Fouque '17; Ducas–van
     Woerden fatigue). Mitigation for a real deployment: instantiate C1
     with a standard-parameter scheme (BGV/BFV/CKKS at depth 1, or a
     non-overstretched NTRU regime), accepting a slower multiply but the
     same 0-bootstrap property. The paper's contribution (C1) does not
     depend on SI-NTRU's security; SI-NTRU is a *speed* choice for the
     benchmark.
  3. Malicious security is out of scope (semi-honest, as standard for this
     line of work).

## 9. Limitations & honest negatives (from the measurements)

- The MPC nonlinearity + conversion dominates cost (from `bench-hybrid-nn`):
  the construction wins on *depth scaling and deployment shape*, and on the
  large-param / thin-pipe regime — **not** on a single small layer on a fast
  LAN, where pure MPC is faster. Report this table as-is.
- Softmax/LayerNorm (transformers) need additional MPC subprotocols
  (exp/reciprocal/rsqrt) — the method extends in principle but those
  nonlinearities are not implemented; deep *MLP/CNN/RNN* ReLU stacks are.
- Width per FHE ciphertext is bounded by the ring dimension via the packing
  constraint `(rows+1)·in ≤ n`; larger layers chunk (implemented in
  `bench-hybrid-nn`).
- CNN gadgets (`conv2d`, `maxpool2d`, `argmax`) are demonstrated *composed
  and exact* (E6). The **efficient FHE convolution packing** (laying out
  channels × spatial so a conv is one/few ciphertext multiplies, à la CKKS
  conv packing) is the remaining crypto-engineering step to run a
  CIFAR-scale CNN under the hybrid at speed; the composition/security story
  (this section, §5b) is independent of it.

## 10. Reproducibility

Single Rust crate, no GPU. `build-windows.cmd` documents the Windows/SEAL
toolchain; `cargo test` covers correctness; each `bench-*` subcommand emits
`--json`. All numbers in the paper are produced by these commands on
commodity hardware; none are hand-entered.
