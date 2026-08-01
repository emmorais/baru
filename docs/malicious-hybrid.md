# Malicious security for the encrypted-model honest-majority hybrid

**Status: design sketch + self-pressure-test.** This is the candidate research
contribution identified in the roadmap discussion (direction *A*): a
maliciously-secure, honest-majority private-inference system that keeps the
model as a *published FHE ciphertext*, where the integrity of the FHE
operations is enforced by **replication**, not by verifiable-FHE / SNARKs.

The point of writing it down is to find where it breaks *before* building it.
Sections 5 and 6 are the honest caveats — including one claim I initially made
and then had to retract.

---

## 1. Deployment scenario (why bother with FHE at all)

If you already assume an honest majority of non-colluding servers, you can run
the whole inference in pure secret-sharing (ABY3 / SWIFT) with the model
*secret-shared* among the servers — no FHE. So FHE must earn its place. It does
so **operationally**, not by a stronger active-security threshold:

- **The model owner is offline.** The model is a *static published artifact* —
  a set of BFV ciphertexts under a threshold key. The owner encrypts once and
  disappears. With RSS-of-the-model the owner must interactively re-share to
  each specific server set and re-share whenever the set changes.
- **Servers are stateless w.r.t. the model.** They hold ciphertexts, not model
  shares. (See the snapshot-compromise benefit in §5.)
- **Communication.** The linear layer is one ct×ct product (§ the main paper),
  not an interactive matmul — the WAN win.

The client encrypts its input under the same threshold key. Three
non-colluding servers run inference; nobody learns the model or the input;
only the class index is revealed.

This is a coherent, deployable niche: *a proprietary model served by a
consortium of non-colluding clouds.* Malicious security matters because any one
cloud could be compromised.

---

## 2. Threat model

- **Parties.** Three servers `P0, P1, P2`, replicated 2-of-3 sharing (RSS):
  value `v = v0 + v1 + v2`, `Pi` holds `(vi, v_{i+1})`. Each pair shares a PRF
  key (`k01, k12, k20`) — the same keys the semi-honest multiplication already
  uses.
- **FHE key.** BFV secret key `s` is **replicated-shared** across the three
  servers and threshold-decryption needs **all three** (3-of-3 for the key;
  2-of-3 for the MPC). No proper subset can decrypt.
- **Adversary.** Static, **malicious**, corrupts **at most one** server
  (honest majority). Goal: security *with abort* — either the honest parties
  output the correct class, or they abort; the adversary learns nothing either
  way (beyond the output, on success).
- **Boundaries.** The model owner and the client are semi-honest-with-proof:
  they must ship *well-formed* ciphertexts (§4.4). Malicious model owner /
  client integrity is out of scope except for that boundary check.

---

## 3. The one idea

Every value in the pipeline is held **redundantly** (RSS ⇒ each share is known
to two parties; the FHE key share likewise). So every operation can be
performed by **two** parties and their results compared. With ≤1 corruption,
at least one of the two computers is honest, so a cheat is always caught.

The only obstacle is **randomness**: two honest parties re-encrypting the same
value produce *different* ciphertexts (fresh noise), so equality-comparison
fails. The fix is that all FHE randomness is **derandomized** from the pairwise
PRF keys the two co-holders already share. Then the two co-holders produce a
*bit-identical* ciphertext, and comparison works again.

So the whole construction is: **classify every FHE operation as deterministic
or randomized; verify deterministic ops by recompute-and-compare; derandomize
randomized ops so they, too, become recompute-and-compare.**

---

## 4. The construction, operation by operation

### 4.1 FHE linear map (ct×ct multiply) — deterministic

BFV multiplication (tensor + rescale) is a deterministic function of the two
input ciphertexts; no randomness. Both `P0` and `P1` hold the (public-to-them)
input ciphertexts, so both compute the product and exchange a **hash** of the
result. Disagreement ⇒ abort. A malicious computer producing a wrong ciphertext
is caught by the honest one. Cost: 2× FHE compute + one hash.

### 4.2 Switch to MPC (mask + threshold-decrypt) — randomized, derandomized

To open `v` with privacy: parties add a random mask `r` (RSS-shared) and
threshold-decrypt `c_v + Enc(r)`, revealing `v + r`; shares of `v` follow.

- **Mask contribution** `r_i`: RSS-shared, so `r_i` is known to two parties.
  They derandomize its sampling from their shared PRF key ⇒ identical ⇒
  comparable. A wrong mask is caught by the co-holder.
- **Partial decryption** `d_i = c1 · s_i + e_i^{smudge}`: the key share `s_i`
  is held by two parties; the **smudging noise** `e_i^{smudge}` is derandomized
  from their shared PRF key. Both co-holders compute the *same* `d_i` and
  compare. A wrong partial decryption is caught.
- The public sum `Σ d_i` yields `v + r`; each honest party checks it against
  the two partials it can recompute.

### 4.3 Switch back to FHE (re-encrypt) — randomized, derandomized

Value `a` is RSS-shared after the nonlinearity. We want `c_a = Σ_i Enc(a_i)`.
Each `a_i` is known to two parties; the encryption randomness `(u, e_1, e_2)`
is derandomized from their shared PRF key ⇒ both produce an identical
`Enc(a_i)` ⇒ comparable. The sum is deterministic; recompute-and-compare.

### 4.4 Boundary: well-formed input ciphertexts — the *only* ZK

Recompute-and-compare secures the *computation*, but it cannot vouch for
ciphertexts that enter from outside (the published model, the client input): a
malicious client could ship a **malformed** ciphertext (e.g. adversarial noise
engineered to force a decryption error, or a "ciphertext" that is not an
encryption of any bounded-noise plaintext). Both co-holders would faithfully
compute on the same bad input and *agree*, so replication does **not** catch
it.

Therefore each externally-supplied ciphertext carries a **zero-knowledge proof
of well-formedness**: knowledge of a bounded-noise `(m, u, e)` opening. This is
a standard, cheap, amortizable lattice ZKPoK — *not* a SNARK over a
computation. It is the honest price of a malicious boundary, and it is charged
**once per input**, never per operation.

> **This is the crisp contrast with single-server verifiable FHE.** There, ZK
> must cover the *entire evaluation*. Here, ZK covers only the *inputs*;
> replication + honest majority covers the evaluation for free.

### 4.5 MPC nonlinearity — standard malicious honest-majority

ReLU / max-pool / argmax on the shares use the existing malicious
honest-majority machinery (SWIFT / Fantastic-Four-style replicate-and-compare,
or sublinear distributed-ZK on secret-shared data — fully-linear PCPs of
Boneh–Boyle–Corrigan-Gibbs–Ishai). No new technique; it composes with the
above because both live in the same RSS world.

> **Correction (2026-07-31, from implementing it).** "No new technique" was too
> quick. Building the nonlinearity path surfaced a subtlety that is *not*
> handled by the deferred-abort discipline of §4.6, and that a careless
> implementation gets wrong while appearing to work. It is recorded in §4.5a.

### 4.5a Why bit-decomposition MACs cannot be multiplied into existence

The malicious ABY3 layer (`src/crypto/aby3_verify.rs`) carries every wire as
`([x], [r·x])` and verifies all multiplications with one batched sacrifice.
Bit decomposition breaks the inductive step: its **leaves are built locally,
with no communication** (`share_component`), so they arrive without MACs.

The leaf *values* are safe — nothing goes on the wire, so the only available
deviation is to use a value the co-holder disagrees with, and replication
catches that through the pairwise transcript.

The leaf *MACs* are the problem. The obvious construction, `[r·x] = [r]·[x]`,
is unsound, and unsound in a way that is worse than a missed cheat. A corrupt
party injects `δ` into that multiplication; the first downstream gate
`[z] = [x]·[y]` then carries the MAC error

```text
([r·x] + δ)·[y] = [r·z] + δ·y
```

where `y` is a **secret bit**. When `y = 0` the error cancels by itself and the
run completes; when `y = 1` the check fails and the parties abort. *Whether the
computation aborts is now a function of the data.* The adversary learns a
predicate on the secret inputs from nothing more than whether the run survived.

This is a selective-failure attack, and §4.6 does **not** stop it. Deferring
the abort to a single point hides *when* the abort happens; it does not hide
*whether* it happens. The two are different channels and only the first is
closed by batching.

**The fix** (Chida et al., CRYPTO'18 §3–4) is to never multiply a MAC into
existence. Prepare random pairs `([ρ], [r·ρ])` in advance and attach MACs
affinely:

```text
d = open(x − ρ)                 uniform: ρ is a fresh one-time pad
[r·x] = [r·ρ] + d·[r]           local — no multiplication, no deviation
```

A corrupt party can still cheat while the *pairs* are produced, and the same
sacrifice still catches it — but `ρ` is random and **independent of the
inputs**, so that abort carries no information about the data. The channel
closes because the only place deviation remains possible has been moved off
the data path.

Implemented as `MacPair` / `generate_mac_pairs` / `mac_existing_batch`, with
the whole pool produced in one batched round and attachment costing one batched
opening. Cost is the same order as the multiplication it replaces; the
difference is entirely in soundness.

**Reading for the paper.** This is worth stating explicitly rather than
burying, because the naive construction *passes every functional test* — it
computes the right answer, and it aborts when cheated. It fails only against an
adversary willing to spend the abort to learn one bit, which no correctness
test will ever exhibit.

### 4.6 Selective-failure defense

Per-operation abort could leak: a malicious party might inject an error that
cancels only for specific secret values, making *abort* itself input-dependent.
Standard mitigation: do **not** compare per operation. Accumulate all
recompute-and-compare residuals and check a **single random linear
combination** of them at the end of the inference (à la SWIFT's batched
verification). One check, secret-independent abort, negligible soundness error.

---

## 5. Security argument (sketch) and the caveat I had to retract

**Integrity (≤1 malicious).** Every value has ≥2 honest-or-adversary computers
with ≥1 honest; every operation is deterministic after derandomization; the
batched random-linear-combination check catches any single injected error with
overwhelming probability. External inputs are well-formed by §4.4. Hence the
honest parties either output the correct class or abort. ∎(sketch)

**Privacy (≤1 malicious).** The adversary is missing one RSS share of every
value (so every masked-open is uniform to it) and one FHE key share (so it
cannot decrypt). Derandomized randomness uses PRF keys the adversary lacks for
at least one co-pair. Standard simulation. ∎(sketch)

**Caveat — the "graceful degradation" claim is FALSE; do not make it.** My
first instinct was that the FHE-encrypted model buys *model privacy under a
dishonest majority* (2 colluding servers). It does not. Under 2-collusion:

- The two colluders jointly hold all three RSS shares of every mask, so every
  **masked-open reconstructs the intermediate activation**. The activations
  depend on the model, so a chosen-input probe leaks model behavior and enables
  model-extraction — the model is *not* safe under 2-collusion.
- So the active-protocol trust threshold for **both privacy and integrity is
  honest majority**, exactly as in pure MPC. FHE does not raise it.

The one *narrow* real benefit that survives: against a **passive
snapshot/storage-compromise** adversary that exfiltrates the state of up to two
servers *without participating in the protocol*, the model stays confidential,
because decrypting it needs the full 3-of-3 key that is never assembled. With
RSS-of-the-model, two snapshots reconstruct the model. This is a genuine but
narrow at-rest guarantee, not an active-security upgrade. State it as such.

---

## 6. Honest costs and open problems

- **Threshold decryption needs smudging** (noise flooding to hide the key).
  We **measured** the actual gate (`bfv::tests::measures_post_multiply_noise_budget`)
  rather than estimate it, and the news is good — see §6a.
- **~2× FHE compute** (two parties recompute each op). Acceptable for a
  malicious upgrade; parallelizable across the honest pair.

### 6a. The smudging gate — measured, and it passes

The correctness budget for threshold decryption is
`q/2t − |e_mult|`; smudging noise `E` must fit inside it while statistically
hiding `|e_mult|`. By the smudging lemma, hiding `|e_mult|` at statistical
security `λ` needs `log2 E ≥ log2|e_mult| + λ`, and correctness needs
`log2 E < log2(q/2t)`. So the affordable statistical security is
`λ ≤ log2(q/2t) − log2|e_mult|` — the **headroom**.

We measured `|e_mult|` on the real depth-1 ct×ct product (the multi-modular
NTT gives *exact* products, so the noise is just fresh-encryption noise blown
up by the multiply — far smaller than a textbook worst case):

| params | `|e_mult|` (measured) | `q/2t` | headroom `λ_max` |
|---|---|---|---|
| n=4096, q=2^100, t=2^20 | `2^38.6` | `2^79` | **≈ 40 bits** |
| n=4096, q=2^100, t=2^24 | `2^42.5` | `2^75` | **≈ 32 bits** |

Reading this into the parameter choice:

- **λ = 40 statistical** (lenient): **fits at the current n=4096, q=2^100** for
  `t ≤ 2^20`, with *no parameter change*. (`t = 2^24` is ~8 bits short.)
- **λ = 80 statistical** (conservative): needs `log2 q ≳ 140`, hence
  **n=8192** (its 128-bit ceiling is `log2 q ≤ 218`). Ciphertexts grow from
  ~100 KB to ~290 KB (≈ 3×). A bounded cost, not a blocker.
- **Rényi-divergence flooding** (sound for a bounded number of decryptions,
  which inference is) tightens the requirement by roughly half, likely fitting
  λ = 80 at n=4096 or comfortably at n=8192.
- **Decrypt-to-shares** (§ below) can avoid public flooding almost entirely.

**Verdict: the direction is viable.** My original pen-and-paper estimate
(`|e_mult| ≈ 2^55`–`2^60`, "needs n=8192") was too pessimistic; the *measured*
noise leaves 32–40 bits of headroom at current parameters. Conservative
statistical security costs at most a doubling of `n`.

**Decrypt-to-shares, the smudging-light alternative.** Public threshold
decryption reveals `v + r` *and its noise*, which is what forces flooding. An
alternative outputs an additive *sharing* of `v` without a public opening
(each party derives a share via its partial decryption), so the noise is never
revealed and the flooding requirement collapses. This is the SPDZ/Overdrive
"distributed decryption to shares" trick; adapting it to our mask-and-open
switch (which currently *does* open `v + r`) is the way to keep λ = 80 at
n=4096 if the 3× ciphertext of n=8192 is unwelcome.
### 6b. The boundary proof's soundness slack is paid out of the smudging budget

**Added 2026-07-31, and it changes the parameter conclusion of §6a.**

§6a measured `|e_mult|` and found ~40 bits of smudging headroom at n=4096,
concluding the parameters were fine. That analysis assumed **honest inputs** —
which is precisely what the boundary proof exists to stop assuming.

Every practical lattice Σ-protocol is *relaxed*: the extractor recovers a
witness whose norm exceeds the honest one by a slack factor, from the challenge
difference and from rejection sampling. A verifier accepting such a proof is
accepting inputs with noise up to `2^slack · B`. That slack does not stay at the
boundary — the depth-1 multiply is linear in each operand's noise, so it passes
straight through to the magnitude threshold decryption must hide:

> **Each bit of soundness slack in the boundary proof costs one bit of
> statistical security in the threshold decryption.**

Computed by `fhe::boundary::achievable_lambda`, with the one-for-one relation
pinned by a test. The consequences:

| boundary proof slack | λ achievable at n=4096, q=2^100, t=2^20 |
|---|---|
| 0 (idealised) | ~38–40 |
| 2^10 (tight Σ-protocol) | ~28 |
| 2^20 (typical) | ~18 |

So **n=4096 supports a conservative λ only with an unusually tight proof**.
§6a's "fits at n=4096, no parameter change needed" is correct for honest inputs
and optimistic for the malicious setting the rest of this document is about.

> **Corrected by measurement (§6d).** This section originally concluded "n=8192
> makes the combination comfortable". That was inferred, and it was wrong: the
> modulus it assumed (q=2^190) is not representable by this implementation, and
> the measured result at the largest one that is leaves **λ = 30, not 40**.

The design constraint for whoever builds the proof is
`max_tolerable_slack_bits`, which answers the only question that matters: *how
loose may this proof be?* At n=4096 targeting λ=30, the answer is about 8 bits.

This is a conclusion neither analysis reaches alone; it only appears when the
boundary proof and the threshold decryption are budgeted together.

### 6c. Implementing the boundary proof: rejection sampling is a grinding oracle

**Added 2026-07-31.** `fhe::boundary_proof` implements §4.4 as a Σ-protocol
over the ring with Fiat–Shamir and rejection sampling (the Lyubashevsky shape).
Two things went wrong while building it, both of which an implementation can
have while passing every obvious test.

**1. Global retry makes parallel repetitions unreachable.** Fiat–Shamir derives
all challenges from all commitments at once, so a rejection *anywhere* forces
the whole proof to restart. With a mask width of `M = 3n·B` — the value that
gives a comfortable per-coefficient acceptance rate — overall acceptance is
`0.37^reps`: at 8 repetitions, about one attempt in three thousand, and the
prover appears to hang. The mask must be sized against `n · reps`, not `n`.

**2. The retry loop grinds the challenge.** This is the serious one. A rejected
transcript is discarded and the masks resampled, which resamples the
commitments and therefore *the challenges*. If the challenge vector ever comes
out all-zero, every response is a pure mask: no witness is ever added, nothing
is rejected, and **the transcript verifies while proving nothing**. An honest
prover reaches that state by accident in about `2^reps` attempts; a malicious
one aims for it.

At `reps = 8` this was reproducible in seconds. The prover "proved" a
ciphertext whose noise was `2^40` times the permitted bound — precisely the
malformed input the boundary exists to reject — and the verifier accepted.

So with binary challenges, `reps` is **not** a soundness-error knob to set to
taste; it is a *grinding-resistance* parameter and must equal the security
level. `MIN_REPS = 128` is enforced by both prover and verifier — the verifier
side matters, since accepting a short proof is accepting a forgery.

**Cost.** 128 repetitions × 4 response polynomials. The slack this construction
achieves is ~25 bits, which §6b prices: it does not fit at n=4096, and n=8192 is
the parameter set that absorbs it. The proof is charged once per input
ciphertext, never per operation, which remains the crisp contrast with
single-server verifiable FHE.

**Status.** This is a working, tested implementation of the relation, not a
production-hardened proof system: the security argument above is a sketch, the
constant-time behaviour of the sampler has not been audited, and the `n=8192`
parameter recommendation follows from §6b's model rather than from a fresh
noise measurement at that dimension. It closes §4.4 as *implemented and
testable*, not as *reviewed*.

### 6d. Measured at n=8192 — and the binding constraint is the modulus, not the ring

**Added 2026-07-31.** §6b concluded that n=8192 would absorb the boundary
proof's slack. That was a model, and measuring it produced two corrections.

**1. The recommended parameters were not representable.**
`boundary::smallest_ring_for` proposed `n = 8192, q = 2^190`. This BFV holds the
modulus in a `u128` (`q_mask` is `(1 << log2_q) - 1`), so `log2 q ≤ 127` and the
proposal overflows on construction. The search now caps at
`MAX_REPRESENTABLE_LOG2_Q` and returns `None` when a target is unreachable —
`None` being a real answer, not an invitation to take the largest entry.

**So the binding constraint is the modulus representation, not the ring
dimension.** Going further needs multi-precision or RNS arithmetic, which is an
implementation change rather than a parameter choice.

**2. The achievable λ is 30, not 40.** Measured at the largest representable
parameter set (`bfv::tests::measures_post_multiply_noise_at_n8192`):

```
n=8192, q=2^120, t=2^20
  |e_mult|              2^41      (measured)
  boundary-proof slack  26 bits   (from the real construction, §6c)
  achievable lambda     30
```

λ = 30 is a usable statistical security parameter, and it is what this
construction actually delivers today. It is **not** the λ = 40 §6a reported for
honest inputs, and the gap is entirely the price of the boundary proof.

To reach λ = 40 one of three things has to change: a tighter proof (smaller
slack), a multi-precision modulus (bigger budget), or accepting a smaller `t`.
Stating which is a parameter decision for the paper, not a caveat to bury.

**Method note.** The first version of this measurement asserted that the
decrypted product equalled the elementwise product of the two plaintexts. It
does not — BFV multiplication is a *polynomial* (negacyclic) product, and the
whole packing scheme in the main paper depends on exactly that. Correctness at
these parameters is now checked as "the noise stays under q/2t", which is what
correctness means here.

- **Noise analysis under adversarial inputs.** §4.4 argues malformed inputs are
  excluded by the boundary ZK, so honest noise bounds hold. This needs to be
  made rigorous: the ZKPoK's noise bound must be tight enough that the
  worst-case *accepted* input still leaves the smudging noise dominant.
- **Derandomization vs. security of encryption.** Using PRF-derived encryption
  randomness must be argued pseudorandom to the third party. The two co-holders
  know it (fine — they encrypt their own share), the third does not. Needs the
  standard PRF-security argument, and care that the *same* randomness is never
  reused across two different plaintexts (distinct counters, as in `alpha_ctr`).
- **The MPC malicious layer** is assumed (SWIFT-style) but not yet implemented
  here; the current code is semi-honest. Integrating it is real work.

---

## 7. Novelty and positioning

- **What is new:** extending the honest-majority *replicate-and-compare*
  malicious philosophy (SWIFT, Fantastic Four) to **threshold-FHE operations**,
  by derandomizing the randomized ones (encryption, partial decryption,
  smudging) with the pairwise PRF keys already present. To our knowledge no
  prior system combines a *published-ciphertext (offline-owner) model* with
  *honest-majority MPC* and gets malicious security this way.
- **The clean statement:** *ZK is needed only at the input boundary
  (well-formedness), never for the computation.* Single-server verifiable FHE
  needs ZK over the whole evaluation; honest majority replaces it with
  replication.
- **What is NOT new / must not be overclaimed:** the trust threshold is honest
  majority for both privacy and integrity (no graceful degradation, §5); the
  semi-honest hybrid, the depth-1 no-eval-key FHE, and the GC-vs-a2b finding
  are all prior or folklore.

---

## 8. Next steps to de-risk before building

1. ~~Concrete threshold-BFV parameters *with smudging*.~~ **Done (§6a).**
   Measured: fits at n=4096 for λ=40; needs n=8192 (~3× ciphertext) for λ=80,
   or decrypt-to-shares to stay at n=4096. **Gate passed — not paper-blocking.**

   **Implemented and re-measured, 2026-07-31.** `src/fhe/threshold.rs` and
   `src/hybrid/threshold_switch.rs` now carry replicated key shares,
   derandomized encryption, checkable partial decryptions, and both directions
   of the scheme switch. The λ=40 claim is no longer an estimate from a noise
   measurement — `decryption_survives_realistic_smudging` and
   `threshold_switch_still_works_at_realistic_smudging` sweep the flooding
   magnitude and confirm correctness holds with three parties each contributing
   a flooding term. §4.1–4.3 are done; §4.4 (the boundary proof) is what remains,
   and it is now the *only* thing between this and a malicious attestation.
2. Pin the boundary ZKPoK: which lattice proof, what noise bound, what cost per
   input. (Still the top open item — but the *bound* is now pinned; see §6b.)
3. Write the ideal functionality `F_infer` and the simulator for the one
   malicious party, formally, to confirm §5 holds without a hidden selective-
   failure channel.
4. Decide public-open + flood (n=8192) vs decrypt-to-shares (n=4096); the
   latter changes the switch sub-protocol but keeps parameters small.
5. Only then implement: derandomized threshold decrypt/encrypt + the batched
   consistency check, on top of a malicious SWIFT-style MPC layer.
