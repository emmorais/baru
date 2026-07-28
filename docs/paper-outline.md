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
- **Offline/online preprocessing**: Delphi (USENIX Sec '20) moves
  query-independent HE work off the critical path; we apply the same idea
  to the *scheme switch* (mask/ρ encryptions precomputed) and combine it
  with mask-and-correct so the return leg is a plaintext addition (E11).
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
| E8b | `bench-train --hidden H` | private *backpropagation through ReLU* (`ReLU′ = step = lt`, reused from the forward mask): a 1-hidden-layer MLP trained under the real protocol, model bit-exact vs plaintext; honest ML caveats (per-layer LR; integer-SGD divergence past a stable regime) are separate from the exact-backprop crypto claim |
| E9 | `bench-fhe-innerprod` | head-to-head of the minimal-FHE inner product (one ct×ct multiply, **no relinearization, no bootstrapping, no eval keys**, decrypt under `s²`/`f²`) — **BFV (secure, RLWE, n=4096)** vs **SI-NTRU (fast reference, insecure)**. Both exact, both eval-key-free. Result (this machine, matvec 8×4, **both NTT-accelerated**): BFV multiply 45 ms / decrypt 37 ms / ct 131 KB vs SI-NTRU 9.0 ms / 17 ms / 33 KB — the secure scheme pays ~5× (2× ring dimension + degree-2 tensor), quantifying the cost of security. CKKS is structurally the same (RLWE, no eval keys, coefficient-packed) and measured via SEAL elsewhere. **Conclusion: BFV is the secure instantiation to build on; SI-NTRU stays a reference.** |
| E9b | `bench-hybrid-nn --strategies hybrid-bfv` | the **full no-bootstrap pipeline end to end on the SECURE scheme**: packed ct×ct matvec (BFV, no relin, decrypt under s²) → mask-open switch → real-protocol ABY3 ReLU → packed re-encryption. Bit-exact vs plaintext. Three-way (layer 8×2, all exact): mpc 21 ms / hybrid-ntru(insecure) 187 ms / hybrid-bfv(secure) 437 ms — the construction is exact and complete on secure parameters, and the remaining gap is now dominated by the scheme-switch conversions, not the FHE multiply. |
| E10 | `fhe::ntt` differential tests | **multi-modular NTT** for the exact negacyclic product: a power-of-two `q` admits no direct NTT, so NTT-friendly primes are used as a computational device and the exact integer product is CRT-reconstructed (Garner), preserving the rounded-shift rescale. Differentially tested against the schoolbook implementation it replaces at full ±2^99 width and n=4096. Speedups: BFV multiply 515→45 ms (11×), BFV decrypt 130→37 ms (Horner), SI-NTRU multiply 31.6→9.0 ms, end-to-end secure hybrid 1149→437 ms. |
| E11 | `bench-hybrid-nn --strategies hybrid-bfv` (offline/online) | **offline/online split of the scheme switch**, adapting Delphi's preprocessing idea plus the standard mask-and-correct trick: to-MPC masks and return-path randomness are query-independent, so their encryptions are precomputed; online, the return leg is a *plaintext* addition (`Enc(ρ)+δ`, `δ=c−ρ` uniform ⇒ leaks nothing). Measured: to-mpc 149→37 ms, to-fhe 108→**0.4 ms**, secure-hybrid **online latency 437→160 ms** (282 ms precomputed). Honest framing: total work is unchanged — it is moved off the critical path. Output bit-exact. |
| E12 | `bench-hybrid-nn --in/--out` (chunked) | **layer-size ceiling lifted**: `⌈out/(n/in−1)⌉` weight ciphertexts against one encrypted input, whose forward NTT is reused (`mul_pre`). *Finding*: the reversed-coefficient packing is already **optimal for dense layers** — a degree-n product admits at most `n/in` separable outputs (distinct rows need disjoint weight windows), so `⌈out·in/n⌉` multiplications is the limit; Gazelle's SIMD packing matches the MAC throughput but requires rotations/Galois keys, forfeiting the no-eval-keys property (ours is the rotation-free Cheetah-style layout). Convolution is where coefficient packing strictly wins (overlapping windows) — the natural next step. Exact at 8×2 / 256×32 / 784×16 (online 164/656/505 ms). Two bugs found and fixed: mask reuse across chunks leaked `pre_i−pre_j`; a fixed `t=2^20` overflowed at in_dim=784 (t is now range-adaptive with the noise budget asserted). |
| E13 | `bench-cnn --img D` (Cheetah conv packing) | **convolution packing**: kernel packed reversed on channel/row/column at the input's strides, so one ct×ct multiply yields an entire output channel — `out_c` multiplications *independent of image size*, versus `⌈out_c·oh·ow·CHW/n⌉` for im2col/Toeplitz. Exactness argued in two steps: reversed channel offsets make `c+(C−1−c′)=C−1` iff `c=c′` (channels sum on the same coefficient, as multi-channel conv requires), and row-wrap contamination would need `ox ≤ −1` (impossible for a valid output). Measured A/B, both exact, 0 bootstraps: D=8 3→2 mults (289→222 ms FHE), D=12 16→2 (1037→209 ms), D=28 **1352→2 (77497→264 ms, 294×)**. Lifts `bench-cnn` from its documented `D ≤ ~12` ceiling to full 28×28 MNIST resolution. *Finding*: this moves the bottleneck off FHE — at D=28, FHE is 264 ms of 17.1 s (1.5%), the rest being ~1352 MPC ReLUs plus max-pool. **CrypTFlow2-style comparison/truncation protocols are now the highest-value optimization**, not further FHE packing. |
| E14 | `bench-cnn` on BFV (`uses_rlwe_secure_parameters`) | **the flagship CNN now runs on RLWE-secure parameters** (BFV n=4096, q=2^100 ≤ the ~109-bit ceiling), closing a claims-integrity gap: it previously ran on SI-NTRU, which E-sec shows is insecure here even at depth 1 (fatigue ~2^59 adjusted vs required q ≳ 2^84). BFV supports the identical no-relin trick (decrypt the degree-2 product under s²), so the construction transferred unchanged. Supporting fixes: `t` is now calibrated from the measured activation range with the depth-1 budget asserted (a fixed `t` wraps silently → wrong, not noisy, results); the image's forward NTT is computed once and reused across output channels and chunks (`mul_pre`). FHE cost of the upgrade ≈4× (D=28: 264→1032 ms), but conv packing more than paid for it — **the secure pipeline at 28×28 runs in 18.7 s vs 94.0 s for the old insecure Toeplitz path**. `CnnRun` records `ring_n`/`log2_q`/`log2_t` so the security claim is a regression test, not prose. *Standing finding*: FHE is now 5.5% of runtime; MPC nonlinearities are the bottleneck. |
| E15 | `a2b_matches_plaintext_binary_expansion`, `component_sharings_are_free_and_consistent` | **found and fixed a privacy break in the MPC layer.** The previous `a2b` masked its input with a value derived from `op_id` alone — a public lockstep counter — then opened `a − r`; the mask carried no secret entropy, so the input was recoverable from the transcript by any party or observer. Everything built on `a2b` (`lt`/`eq`/`le`, hence ReLU, max-pool, argmax) leaked its activations. The advertised "no extra communication per bit" was a *symptom*: the carry chain was free because every operand was already public. Replaced with ABY3 bit decomposition — free binary sharing of the three additive components (each already known to exactly the two parties a replicated sharing needs), carry-save 3→2 reduction exploiting that `x⊕y⊕z` and `maj(x,y,z)` share the same four products (2 rounds not 4), then a Kogge–Stone prefix adder whose `G = g ⊕ (p∧g')` combine is local because generate and propagate are mutually exclusive. Depth ≈9 rounds vs 1. **Cost of correctness: MPC 4.4× slower, ~130× bandwidth** (D=28: 17.6→76.9 s, 1.2 MB→159.4 MB); all results remain bit-exact, and the FHE-side claims (exactness, zero bootstraps, BFV security) are untouched. *Next*: ~117 MB of the 159 MB is online Beaver-triple dealing (610k multiplications × 6 ring elements/peer); ABY3's native multiplication needs no triples — one element per party per multiplication from PRF-derived zero-sum randomness — so that is the highest-value MPC optimization outstanding. |
| E16 | `triple_free_mul_is_correct_and_reshares_are_masked` | **eliminated Beaver triples from the MPC multiplication.** Replaced the online triple-dealing protocol (party 0 deals 6 ring elements + 2 opened per multiplication) with ABY3 resharing: `zᵢ = xᵢyᵢ + xᵢyᵢ₊₁ + xᵢ₊₁yᵢ + αᵢ` where `αᵢ = F(kᵢ,ctr) − F(kᵢ₋₁,ctr)` is zero-sum correlated randomness from a one-time pairwise-PRF-key handshake — one ring element sent to one neighbour, no dealer, nothing precomputed. Note this answers "why not precompute triples?" by removing the phase entirely: the correlated randomness regenerates locally per operation. **Measured (D=28): MPC 76.9→40.8 s (1.9×), wire 159.4→10.4 MB (15×)**; ~117 MB of the old traffic was triple distribution alone. All results remain bit-exact across the full hybrid suite (35 slow tests). Dead triple-dealing machinery (`deal_triple*`, `receive_triple*`, and the `TripleComponent`/`TripleBatch`/`TaggedD`/`TaggedE`/`TaggedDEBatch` wire messages) removed. Combined with E15: the MPC layer is now both **private** (E15) and **triple-free** (E16), at 40.8 s vs the original insecure 17.6 s baseline — 2.3× the cost of the version that was neither. |
| E17 | `a2b_batch_matches_plaintext`, `conv2d_and_maxpool2d_match_reference` | **batched the nonlinearities: the bottleneck was round latency, not bandwidth.** With triples gone, the ~610k multiplications were tight *within* each `a2b`, but `a2b` ran one-per-element (1352 ReLUs × ~9 rounds ≈ 12k sequential rounds). Added `a2b_batch` (all lanes' carry-save + Kogge–Stone rounds flattened into shared multiplication rounds — a batch of any width costs the same ~9 rounds as one value) and `lt_batch`, letting the ReLU layer become one `a2b_batch` + one batched multiply and `maxpool2d` reduce all windows in lockstep (`pool²−1` batched steps vs one `max_reduce`/window). Both bit-exact vs per-element. **D=28 MPC: 40.8→6.1 s (6.7×), messages 29 211→5 591 (5.2×).** Endpoint: the private, triple-free pipeline (6.1 s) is now **2.9× faster than the original leaky/insecure baseline** (17.6 s). Latent bug fixed en route: the α counter (`op_id+i`) could repeat once a batch exceeded the id spacing — harmless for correctness, a privacy leak (same-α products expose their difference); now a collision-free `(op_id‖i)` encoding (`alpha_ctr`). Sim `TIMEOUT_MS` raised (large batched messages flaked under parallel test load, passed in isolation). |
| E18 | `bench-mnist`, `plaintext_accuracy_is_reasonable`, `encrypted_matches_plaintext_on_real_digits` | **real-dataset accuracy pipeline on real MNIST test digits.** Trained a CNN of the `bench-cnn` shape on the full 60k MNIST train set (numpy SGD), then **post-training quantised into the integer fixed-point scheme `CnnModel::forward` runs** — so the reported number is the deployed integer model, not a float model that loses accuracy afterward. The Rust integer forward agrees with the numpy reference digit-for-digit (embedded 300-image subset), so the offline 10 000-image number transfers; the **encrypted** BFV×ABY3 hybrid reproduces the plaintext prediction on every real digit tested (0 bootstraps), so encrypted accuracy == plaintext accuracy — privacy costs nothing in accuracy, shown on real inputs not asserted. Infrastructure: MNIST loader, `CnnModel::from_trained`, weight/subset embedding, `bench-mnist` CLI. (Superseded on accuracy by E19's deeper model.) |
| E19 | `bench-mnist` (deeper), `hybrid_cnn_classifies_exactly` (OC=8) | **larger trained model → 97.6% on real MNIST, still zero bootstraps.** Added an optional **encrypted hidden dense layer** to `CnnModel`/`run_cnn_hybrid` and widened conv to 8 channels: `conv(8,3×3)→ReLU→maxpool→dense(→128)→ReLU→dense(→10)`. Now **three encrypted linear layers, three MPC nonlinearities, two scheme switches — and still 0 FHE bootstraps**: the extra depth composes in with nothing but "switch again," the central thesis made concrete. Architecture search (numpy) picked conv(8)+hidden(128) as the accuracy/packing sweet spot (FLAT=1352 ≤ the n/2=2048 dense-packing bound; OC=12 gained only +0.1% at FLAT=2028, near the limit). Uniform single-shift quantisation (DSCALE=2^shift) keeps `CnnModel` to one `shift` field. **Full 10 000-image integer test accuracy: 97.63%** (nbits=23, within the depth-1 budget); Rust==numpy digit-for-digit (298/300 subset); encrypted==plaintext on every real digit tested (8/8, 0 bootstraps) through all three encrypted layers. Moves the SOTA table's accuracy column from 91.6% to 97.6% without touching the distinguishing privacy axis (image *and* weights encrypted). |
| E20 | `two_conv_hybrid_matches_plaintext`, `bench-mnist` (two conv) | **a genuine two-convolution CNN, fully encrypted → 98.0% on real MNIST, still zero bootstraps.** Generalised `CnnModel` to an arbitrary stack of conv layers (`extra_conv: Vec<ConvLayer>`) and FC layers (`hidden: Vec<..>`); `run_cnn_hybrid` now composes them by looping the same three-step block (FHE conv/matvec → open → MPC ReLU∘rescale+pool → scheme-switch). The **second convolution is multi-channel** (in_c=8) and packed with the same reversed-axis Cheetah layout — its channels pair off automatically. Architecture: `conv(8)→pool→conv(16)→pool→dense(128)→dense(10)`, i.e. **four encrypted linear layers, three MPC nonlinearities, three scheme switches, 0 bootstraps**. Uniform single-shift conv convention (`cbias<<shift`, rescale `>>shift`) so a `ConvLayer` needs no per-layer special case; only kernel pre-scaling differs at quantise time (conv1 ÷255 for raw pixels). **Full 10 000-image integer accuracy: 98.05%** (nbits=24, within budget); Rust==numpy digit-for-digit (296/300 subset); encrypted==plaintext on every real digit tested (4/4, 0 bootstraps) through all four encrypted layers. Moves the SOTA accuracy column 97.6%→98.0%, ~1% from the ~99% two-party systems whose server sees plaintext weights — while here every layer is ct×ct. *Architecture search finding*: two conv layers (98.05%) beat both wider single-conv (OC=12: 95.7%) and deeper FC (two hidden: 97.9%); spatial depth is what MNIST wanted, and the construction supplies it for free. |
| E21 | `bench-gc-compare`, `garbled_comparator_matches_plaintext` | **do garbled circuits beat bit decomposition for the comparison?** Built a *real* half-gates garbling engine (`crypto::garbled`): free-XOR (offset R, lsb 1), Zahur–Rosulek–Evans AND gates (2 ciphertexts/gate, SHA-256 as H), a `MSB(x+y)` comparator (`b−1` AND gates, deterministic per-gate tweaks so the evaluator needs none on the wire), bit-exact vs plaintext. Head-to-head **through the identical transport** (both `a2b` and the garbled circuit pay real bincode serialization, message passing, latency injection and byte counting — no measurement asymmetry). **Result (1352 comparisons, 20-bit): a2b 9 rounds / 5.94 MB vs GC 1 round / 1.69 MB (3.5×); measured latency GC wins 4.2× (LAN, RTT 0.1ms) → 4.9× (WAN, RTT 20ms) → 4.3× (WAN 80ms/100Mbps).** GC wins at *every* profile, not just WAN. The enabler is the **3-party honest-majority setting: garbled-circuit input labels are delivered by the co-garbler with no oblivious transfer** (OT is the expensive part of 2-party GC, absent here). Bit-width sweep confirms a2b rounds grow as `⌈log₂ b⌉` while GC stays constant. *Honesty*: the garbling engine, byte and round counts, and the transport measurement are all real and implementation-identical for both methods; the GC path is run 2-party (garbler+evaluator) as a faithful cost proxy for 3PC-no-OT (same tables, same labels, same single round). **Implication: a 3PC garbled comparison is the highest-value remaining MPC optimization**; full integration (real 3PC label delivery + B2A of the sign bit into the fused ReLU∘rescale) is the follow-up. |
| E22 | `garbled_relu_rescale_matches_plaintext`, `bench-gc-compare` (ReLU rows) | **garbled the *whole* fused nonlinearity, not just the comparison.** A garbled `fused ReLU∘rescale` circuit — `out = (v>=0) ? (v>>shift) : 0` for `v = x+y` — as `(b−1)` adder ANDs + `(b−shift)` mux ANDs (NOT is free-XOR: flip the zero-label), bit-exact vs the plaintext fused op. Proves a garbled circuit computes an entire ReLU layer in **one round** (vs a2b's 9-round decomposition + 1-round sign-gate multiply = 10). Measured through the identical transport, charging the GC path one modeled B2A round to return its boolean-shared output to arithmetic for the next FHE switch (the a2b path already lands in arithmetic): **GC ReLU wins 2.7× (LAN) → 3.4× (WAN)** — more conservative than the bare-comparison 3.8–4.5× because it includes the fair B2A cost, and more directly relevant since it is the pipeline's actual operation. *Projection*: the CNN MPC segment is dominated by ReLU + max-pool comparisons, so a 3PC garbled nonlinearity projects to a ~3× MPC-segment speedup; measured integration into `run_cnn_hybrid` (real 3PC label delivery + B2A) is the remaining engineering. |
| E23 | `bench-medmnist`, `embedded_accuracy_is_reasonable`, `encrypted_matches_plaintext_on_real_scans` | **private *diagnosis*: the same pipeline on real biomedical images.** MedMNIST v2 (CC BY 4.0) ships twelve 2D biomedical datasets at 28×28 — exactly MNIST's shape — so the E20 pipeline runs unchanged on **PneumoniaMNIST** (pediatric chest X-ray, 2 classes) and **OrganAMNIST** (abdominal CT, 11 classes; `NCLASSES`/`OC` became model fields rather than consts to allow it). Full-test-set integer accuracy **84.78%** (624 imgs) and **81.18%** (17 778), against majority-class baselines of 62.50% / 18.48% and published ResNet-18 (28×28) of 0.854 / 0.935 — i.e. at the published number on pneumonia, 12 points short on the 11-class task, the gap being the 2-conv model sized to the depth-1 budget rather than anything cryptographic. Integer==float to within 0.2 points (99.7% / 98.7% prediction agreement) and **encrypted==plaintext on every scan tested (3/3 each, 0 bootstraps)**, so neither the fixed-point scheme nor the encryption costs accuracy; the price is ~33 s/image. The linear TFHE/CKKS demo swaps over too (`--example mnist -- --pneumonia`, 2380 gates, bit-identical across plain/mock/ABY3, ~1e-7 under CKKS). *Two methodological notes worth reporting*: the quantisation gain is selected on validation, never test; and it is two-sided — too large overflows the plaintext modulus, too small starves the output layer's weights to zero and collapses the classifier onto its majority class. |
| E24 | `bench-mnist` (regenerated), `tools/train_medmnist.py` | **the E20 result made reproducible, and improved in passing.** E20's weights were produced by a numpy script in a scratch directory that was subsequently lost, leaving a trained artifact that could not be retrained — the accuracy column rested on weights nobody could regenerate. The trainer now lives in the repo (`tools/`), reads MNIST idx *and* MedMNIST npz, and reproduces the E20 architecture and quantisation scheme exactly. A fresh 30-epoch run gives **98.89%** on the full 10 000-image test set (was 98.05%), with the embedded 300-image subset **byte-identical** to the original file so the two models are directly comparable, Rust==numpy digit-for-digit (299/300), and encrypted==plaintext (4/4, 0 bootstraps). `FULL_TEST_ACCURACY` is now re-exported *from* the weight file instead of restated in Rust, so the claim cannot drift from the weights again. Moves the SOTA accuracy column 98.0%→98.9%, ~0.1% from the ~99% two-party systems whose server sees plaintext weights. |
| E25 | `mpcc assign`, `the_model_derives_the_hybrid_split_for_a_neural_layer` | **the cost model independently derives the hybrid architecture — and the depth-1 budget is why it is legal.** `src/mpcc/assign.rs` + `src/mpcc/fhe_cost.rs` assign each operation *kind* to an arithmetic sharing, a binary sharing, or FHE, searching Baru's exact cost model. Inputs are measured only: BFV ct×ct 20.7 ms and ct 128 KB (`bench-fhe-innerprod`), TFHE 0.2–0.5 s/gate (`baru bench`), Beaver triples priced from measured OT (`mpcc::ot_cost`), plus two structural facts — BFV is arithmetic-only and depth-1. On one NN layer (256 secret×secret products then a comparison, WAN, 2 parties) the search returns **`mul → fhe`, `cmp-lt → binary`**: the Gazelle/Delphi/Cheetah split, and the one `src/hybrid/` implements by hand, *derived* rather than annotated. 1.26× over the better pure protocol, offline 1.7 MiB → 101 KiB. **The connection to this paper's thesis is causal, not decorative**: at depth 1 a second ciphertext multiplication is *infeasible*, so the only reason FHE can host the layer at all is that the nonlinearity's switch to MPC refreshes the ciphertext. Depth-is-free is the precondition for the assignment, not an optimisation on top of it. There is also a real crossover — 8 and 64-wide layers stay on MPC (batched multiplies cost one round, which FHE cannot amortize a switch against), 256 and 512 move to FHE — so the claim is a *prediction* with a boundary, not a preference. Three modelling errors were found by insisting the model agree with `bench-cnn`'s working hybrid: missing SIMD packing, missing preprocessing cost, and charging a ciphertext per crossing *edge* instead of per crossing value. |

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
  2. **SI-NTRU is overstretched even at depth 1** — do not claim it secure.
     Fatigue point (ternary) at n=2048 is q ≈ 2^19; the large LTV secret
     f=t·f'+1 raises the effective point to ≈ 2^59; but a single multiply
     with a usable t≈2^20 already forces q ≳ 2^84 by the noise budget, and
     no practical n closes the gap. (Back-of-envelope; a full DvW-estimator
     run is a to-do, but the 25–65-bit margin is decisive.) **The secure
     instantiation is RLWE**: BGV/BFV/CKKS have no NTRU lattice / fatigue
     point, so q=2^120 at n=2048 is standard, provably-secure territory —
     *and the same depth-1 trick applies*: a ct×ct product is a degree-2
     ciphertext; skip relinearization and decrypt under `s²` (exactly the
     mask-and-open switch already does). So C1's minimal FHE (one multiply,
     no evaluation keys) is realized securely by RLWE. SI-NTRU stays as a
     *fast reference point* (cheapest arithmetic) for isolating
     packing/switching cost from FHE-scheme cost — never as the security
     foundation. **Contribution opportunity**: a head-to-head of
     BGV/BFV/CKKS/NTRU on this minimal-FHE inner product (no relin, no
     bootstrap) — see §7 (E9, planned).
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
