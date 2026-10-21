# ARA Fall 2026 — AWS Cryptography proposal

**Compiling Across the FHE/MPC Boundary** — a continuation-based compiler
architecture for programs whose cheapest backend changes mid-computation.

Call: **AWS Cryptography, Amazon Research Awards Fall 2026**.
Window **1 Oct – 4 Nov 2026, 23:59 PT**. Decisions Feb 2027.
Portal: `home.academic-research-portal.amazon.science`.
Award: up to **$80k unrestricted** + **$40k AWS credits**.

```sh
pdflatex main.tex && pdflatex main.tex
# §1–6 (all technical content) fit the 4-page limit; §7 (students + budget,
# ~10 lines of administrative content) and references run onto p5; appendices
# A+B follow and are excluded by the call. ARA states 4pp as "encouraged, not
# mandatory", so this is deliberate rather than an overrun to fix.
```

## Reproducing the HEIR measurements

The toolchain is installed; **do not rebuild it**.

```sh
export BARU_OPENFHE_PREFIX="$HOME/.local/openfhe"      # OpenFHE 1.5.1, from source
export BARU_PYTHON="$HOME/.local/heir-venv/bin/python" # heir-py 2026.8.1, py3.12, numpy 1.26.4

baru bench-heir   --shapes 64x32x10 --activation square --inputs 3 --repeats 3
baru bench-2pc-nn --shapes 64x32x10 --activation square [--lan]
```

Full results and method: [docs/experiments.md §VI.15](../docs/experiments.md).

---

## The framing, and why it is not "we beat HEIR"

[docs/experiments.md §VI.7](../docs/experiments.md) argues HEIR is compiler
*infrastructure*, that its paper reports no private-ML performance evaluation,
and that **"beating its numbers would not be a result."** That judgement is
correct and the proposal does not contradict it. Instead:

> The contribution is **expressiveness, not speed**. HEIR's architecture draws a
> boundary; we compile across it and measure what the boundary costs.

"Baru must be at least as good as HEIR in general" survives as **Tier A: a
control condition.** If our FHE arm is slow, then "switching to MPC helps" is an
artifact of a weak baseline rather than a property of the architecture, and no
other number is attributable. That is a stronger reason to want parity than
wanting to win, and a reviewer will accept it where they would reject a
scoreboard.

### Three-tier comparison design (§4 of the proposal)

| Tier | What it shows | Status |
|---|---|---|
| **A — Parity** | Baru within a small factor of HEIR-generated OpenFHE at matched security | **achieved for MLPs: within 13%** (§VI.16); CNNs need T1 |
| **B — Expressiveness** | programs HEIR cannot express, each with the blocking pass cited | the contribution |
| **C — Crossover** | mixed assignment beats both pure strategies | **both directions measured**; prediction outstanding |

## Tier A: Baru's own FHE arm is at HEIR parity (§VI.16)

The control condition the whole proposal rests on. Baru already had a CKKS arm on
the same OpenFHE build HEIR targets — it had just never been pointed at a dense
MLP, only at LeNet-5. `export_mlp_fixture` points it there from the *same*
fixture the MPC arm and HEIR consume, so all three arms evaluate one network.

| shape | depth | direct | **`--lt`** | HEIR | |
|---|---|---|---|---|---|
| `64x32x10` | 3 | 1001 ms | **181 ms** | 167 ms | 1.08× HEIR |
| `784x32x10` | 3 | 1383 ms | **751 ms** | 178 ms | **4.2× HEIR** |
| `64x32x32x10` | 5 | 2173 ms | **384 ms** | 659 ms | **1.7× ours** |
| `64x32x32x32x32x10` | 9 | 9450 ms | **1687 ms** | 1493 ms | 1.13× HEIR |
| `64x32(×5)x10` | 11 | 13095 ms | **2281 ms** | **SIGSEGV** | **ours runs** |

**Within 13% at four of five comparable shapes, faster at depth 5, and running
at depth 11 where HEIR segfaults.** So §VI.15's crossover claim is about
paradigms, not about our implementation quality.

**Two findings worth more than the parity itself:**

1. **The evaluator choice was worth 5.4–8.1× and is opt-in behind a flag.** The
   Orion-style BSGS transform spends *more* ciphertext multiplications (43 → 106)
   to spend *far fewer* rotations (337 → 26). Same program, same backend, same
   security — an 8× swing decided by a choice no cost model was making. *Which
   evaluator is cheapest is a property of the layer shape*, exactly as which
   *backend* is cheapest is a property of the program. The thesis recurs one
   level down, which is the best evidence yet that it's structural.
   **Not flipped to default**, because every §VI.10–VI.12 figure was measured
   without it; it belongs in the cost model, not a flag.
2. **`784x32x10` is the one real deficit (4.2×)** — the only wide-input shape,
   needing 68 rotations where same-depth `64x32x10` needs 26. HEIR's
   shift-network passes do something ours doesn't. Narrow and actionable.

Also: two of six rows are *refused* — at depth 3 the fixture margin (0.039–0.063)
sits below the CKKS logit error (0.085–0.203). **The two-party arm returns the
reference class on all six.** §VI.11's precision result, reproduced on a second
network family — and this time it's *our own* FHE arm failing, so it can't be
read as flattering our own code.

## What the October 2026 code push produced

**The headline result changed.** Tier A is no longer a plan — the HEIR baseline
runs, and it found something better than parity:

> Same MLP, same `x²` function, same machine, `HEStd_128_classic`, both arms
> verified against the same plaintext reference, **at matched disclosure**
> (both return the logit vector):
>
> | depth | HEIR | 2PC LAN | 2PC WAN | |
> |---|---|---|---|---|
> | 3 | 167 ms | **2 ms** | 434 ms | 2.6× HEIR |
> | 5 | 659 ms | **2 ms** | 654 ms | even |
> | 9 | 1493 ms | **4 ms** | **1099 ms** | 1.4× ours |
> | 10 | **SIGSEGV** | **5 ms** | **1322 ms** | ours runs |

**On LAN we win at every depth by 101–377×, widening. On WAN the crossover is at
depth 5, and we lead from there until HEIR stops running.** HEIR costs ~442 ms
per added layer against our ~222 ms — half — so the ratio moves our way
monotonically. There is no depth at which an FHE-only pipeline is both faster
*and* able to run.

**Two thirds of the earlier gap was disclosure policy, not protocol.** Our arm
computed the argmax *inside* the protocol (revealing only the class) while HEIR
returns logits — so the comparison charged us for more work *and* credited us
with less disclosure. Matching disclosure takes WAN exchanges from 20 to **8**
at depth 3: **the private argmax was 60% of wide-area cost.** It is also a
guarantee HEIR cannot offer at any price, so both numbers are reported: matched
(crossover at depth 5) and our default (2.5× more, still ahead from depth 9).

This is the first time the record shows FHE beating the MPC arm, and it is the
thesis confirming itself rather than a setback. §VI.10–VI.12 had LeNet-5 at
depth 11 with MPC ahead 32× (885× past the level budget). Both are true, which
is exactly the claim: *"is MPC or FHE faster" is not well-posed.*

**Also found, and both reportable upstream:**

1. **HEIR's `eval()` is uncallable for some networks.** Its layout passes
   promote the `linalg.fill` zero-initializer into a second ciphertext argument
   of the emitted C++ while the Python layer still registers one
   (`arg_names == ['arg_0']`), so pybind raises an arity error. Weight-dependent,
   not size-dependent. `tools/heir_baseline.py` now reads the true arity and
   drives the exported entry points directly; order `(arg0, zeros)` was pinned
   against the plaintext reference.
2. **SIGSEGV at multiplicative depth 10**, at 554 MB peak against the 564 MB of
   the depth-9 run that succeeds — **not** an OOM kill, and no parameter-abort
   message. Depth 9 works, depth 10 does not run at all.

**A correction we had to make to ourselves.** §VI.7 said the insecure
`HEStd_NotSet` configuration "is what a casual run produces." That is wrong and
unfair to HEIR: HEIR never calls `SetSecurityLevel`, so OpenFHE's default
`HEStd_128_classic` applies, and issue #1232's insecure run was the reporter's
choice. Corrected in both the proposal and `docs/experiments.md`; the gate is
unchanged but its justification is now "a harness must verify, not assume."

**Two corrections we had to make to ourselves — both found while building the
comparison, and both material:**

1. **The first version of §VI.15 quoted a cost model as a latency.**
   `bench-2pc-nn`'s wide-area column is `rounds × RTT + bytes/bandwidth +
   compute`, which its own footer says — the error was quoting it without
   reading. Everything is now wall clock from
   `examples/probe_heir_match.rs` (two parties, two threads, a transport that
   really sleeps). `bench-2pc-nn` now prints a `NOT quotable as a latency`
   warning pointing at the probe.
2. **The arm HEIR was compared against was missing two of the repo's biggest
   optimizations.** `bench-2pc-nn` drives `run_protocol`, which enables neither
   truncation pairs (§VI.9) nor bilinear operand fusion (§VI.13). Turning them
   on cuts exchanges 137 → 36 at depth 10 and WAN elapsed 7695 → 1999 ms —
   **2.4–3.8×, no new algorithm.** The defect presented not as a wrong number
   but as a *correct* number for a configuration nobody would deploy.

**Housekeeping that was blocking the proposal:** `LICENSE` (real Apache-2.0,
202 lines) and `Cargo.toml` metadata added; 831 tests green.

---

## What I verified about HEIR (arXiv:2508.11095)

Sourced from the paper, not inferred. These are what Appendix B's capability
matrix rests on — check them against the published version before submitting.

- MLIR-based. Dialects: `secret`, `comb`, `mgmt`, `tensor_ext`, `lwe`, `bgv`,
  `ckks`, `cggi`, library APIs (`openfhe`, `lattigo`, `tfhe_rust`, `jaxite`),
  `mod_arith`/`rns`/`polynomial`, `pisa` for hardware.
- **No MPC, secret sharing, garbled circuits, or any interactive/multi-round
  protocol appears anywhere in the stack.** No FHE↔MPC switching.
- **Secret-dependent control flow is eliminated, not executed.**
  `convert-if-to-select` turns a secret branch into `arith.select` (both arms
  evaluated). `convert-secret-for-to-static-for` needs user-supplied static
  bounds; secret-dependent loop bounds are unsupported.
- **Polynomial approximation requires user annotation** for degree and interval
  — their own stated limitation. This is the only nonlinearity route available
  to an FHE-only compiler, which is what makes our wrong-class measurement an
  architectural result rather than a tuning observation.
- BGV/BFV bootstrapping unsupported (no backend supports it).
- No benchmark timings in the paper, consistent with §VI.7.

---

## What I verified about Baru's architecture — read this carefully

The proposal distinguishes **implemented** from **proposed**, because the
distinction is load-bearing and a reviewer will probe it.

**Implemented** (verified in [src/interpreter/eval.rs](../src/interpreter/eval.rs)):

- A genuine defunctionalized continuation machine: `Continuation` ADT,
  `Task::Eval`/`Task::Apply` trampoline over an explicit stack, step budget.
- Secret-dependent `if` is a first-class continuation frame (`If`, then
  `IfResultNormalize`, which reconciles a numeric result back to a share of the
  right owner and width) — executed under protocol, not compiled to a select.
- Per-operation backend dispatch through the `Backend` trait, which is what
  makes a mid-program scheme switch an ordinary operation.

**NOT implemented — stated as proposed work (T3), do not let this drift into a
claim:**

- `Continuation` derives only `Debug, Clone`. **It is not `Serialize`.** The
  "migrate a computation / suspend at a round boundary" story is future work.
- The transport backend's `mul`/`reveal` are synchronous and **block on IO**.
  Rounds are scheduled statically by `mpcc`, not dynamically by the evaluator.

So the architectural claim as written rests on what exists (secret-dependent
control flow, per-op dispatch, the switch) and proposes the rest. If T3's
framing is softened or dropped, the proposal still stands on §1–§4.

---

## Status: complete, zero placeholders

| | |
|---|---|
| **PI** | Thales B. Paiva, Assistant Professor (Professor Doutor), PCS / Escola Politécnica da USP, `thalespaiva@larc.usp.br`, Lattes `5406900278440032` |
| **Co-I** | Eduardo Morais, Postdoctoral Researcher, LARC / PCS, USP |
| **Students** | 1 doctoral + 2 *iniciação científica* |
| **Budget** | $80,000 unrestricted + $40,000 AWS credits, itemised at FAPESP published rates (DR-I R$5,790/mo, IC R$1,140/mo, from 2025-08-01) converted at R$5.03/USD on 2026-10-21 |

### Three things still to check — none of them mine

1. **"Full-time" faculty status.** ARA's wording is explicit. A USP Professor
   Doutor appointment normally qualifies (RDIDP/RTC), but confirm the formal
   contract type. It is the one eligibility clause that could still disqualify.
2. **One proposal per PI per call period** — apparently program-wide rather than
   per track. Verify on the portal, and confirm Thales is not PI on another
   Fall 2026 ARA submission.
3. **Thales reads the draft.** He is PI; the scientific claims are made in his
   name. In particular §7 cites his TCHES/EuroS&P record and ties the
   bootstrapping paper to our depth-10 cliff — he should agree that framing is
   one he wants to stand behind.

### Why this PI strengthens the proposal

Not a formality. *Faster amortized bootstrapping using the incomplete NTT for
free* (TCHES 2025) is simultaneously (a) a listed topic of this exact call
— "improved practical FHE bootstrapping algorithms for cloud environments" —
and (b) the operation whose absence produces §3's depth-10 cliff. Add
*Incompleteness in Number-Theoretic Transforms* (EuroS&P 2025), which bears on
Baru's own hand-written NTT, two further TCHES 2025 papers, and side-channel
work on Kyber and HQC. The team is not approaching FHE from the outside, and §7
says so where reviewers look for it.

## Number provenance## Number provenance

Every figure traces to [docs/experiments.md](../docs/experiments.md). Nothing is
modelled or extrapolated; the appendix says so explicitly.

| Claim | Source |
|---|---|
| CKKS **wrong class** at 13 levels (margin 2q, error 15q) | §VI.11 precision table |
| 1.41 s vs 43.8 s, 32× latency | §VI.10 — same network, same machine, both verified class 6 |
| HEIR 167 ms vs 2 ms LAN / 434 ms WAN (matched) | §VI.15, `examples/probe_heir_match.rs` (measured elapsed) |
| private argmax = 60% of WAN cost (20 → 8 exch) | §VI.15, same probe, `opt` vs `opt+logits` |
| 2.4–3.8× from enabling existing optimizations | §VI.15, same probe, base vs opt columns |
| SIGSEGV at mult. depth 10, 554 MB peak | §VI.15 |
| 0.363 s vs 2.86 s, ~8× throughput | §VI.12 batch sweep + §VI.13 fusion |
| 1.52 s vs 1345.1 s, 885× | §VI.11 forced-bootstrap (`--levels=6`) |
| 2.6× ring cliff, 61.1 → 160.0 s | §VI.11, 13 → 15 levels |
| 872× from frontend vectorization | §IV.2b |
| 507,920 → 33,106 messages | §IV.2b |
| Lisp overhead: 299 ms plaintext vs 268 ms encrypted Rust | §IV.2b |
| Argmax tournament 1.67× / truncation pair 3.67× / all-pairs 1.47× | §VI.9 |
| Operand fusion 10.8× traffic at batch 8 | §VI.13 |
| HEIR 12× insecure-vs-secure gate | §VI.7 gate 1 (google/heir#1232) |
| 15× off Orion, 3.5% slot utilisation | §VI.10 |
| 129 commits / 3 contributors | `git shortlog -sne --all` |

---

## Highest-value work before 4 November

Tier A and the `LICENSE` are **done**. What remains, in order:

1. **Commit the branch.** Still the cheapest visible win — `malicious-security`
   has the whole October push uncommitted.
1b. **Multi-output lowering.** The matched-disclosure figures use a summed
   output as a stand-in, which is sound for round cost but is not a deployable
   logit-returning classifier. `mpcc::lower` takes a single `output_wire`, and
   the circuit compiler renames let-bindings to temporaries, so returning ten
   logits needs a binding→wire map. Contained, and it would turn a probe into a
   real deployment mode.
2. **Report both HEIR bugs upstream.** Citing filed issues in the proposal is
   much stronger than "we observed"; it also demonstrates the ecosystem
   contribution the call's open-source topic asks about.
3. **Isolate the depth-10 segfault** — HEIR parameter selection vs. OpenFHE
   1.5.1 vs. genuinely needing bootstrapping. This is the one claim in §3 that
   rests on an unexplained crash, and a reviewer will ask.
4. **A trained fixture for the depth/precision sweep**, which is the standing
   caveat on the wrong-class result.
