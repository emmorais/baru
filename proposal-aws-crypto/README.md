# ARA Fall 2026 — AWS Cryptography proposal

**Compiling Across the FHE/MPC Boundary** — a continuation-based compiler
architecture for programs whose cheapest backend changes mid-computation.

Call: **AWS Cryptography, Amazon Research Awards Fall 2026**.
Window **1 Oct – 4 Nov 2026, 23:59 PT**. Decisions Feb 2027.
Portal: `home.academic-research-portal.amazon.science`.
Award: up to **$80k unrestricted** + **$40k AWS credits**.

```sh
pdflatex main.tex && pdflatex main.tex
# 6 pages: body is exactly 4 (the limit), appendices A+B on 5–6 (excluded)
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
| **A — Parity** | Baru within a small factor of HEIR-generated OpenFHE at matched security | **done for MLPs** (§VI.15); CNNs need T1 |
| **B — Expressiveness** | programs HEIR cannot express, each with the blocking pass cited | the contribution |
| **C — Crossover** | mixed assignment beats both pure strategies | **both directions measured**; prediction outstanding |

## What the October 2026 code push produced

**The headline result changed.** Tier A is no longer a plan — the HEIR baseline
runs, and it found something better than parity:

> Same MLP, same `x²` function, same machine, `HEStd_128_classic`, both arms
> verified against the same plaintext reference. On `64x32x10`:
> **LAN — secret sharing wins 11.9× (14 ms vs 167 ms).
> WAN — HEIR wins 12.7× (167 ms vs 2113 ms).**
> A **151× swing from the network profile alone.**

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

## Blockers

1. **PI must be full-time USP faculty.** ARA excludes independent and part-time
   researchers. Eduardo is co-investigator in the draft. Need name, title,
   institutional e-mail, Lattes/ORCID.
2. **One proposal per PI per call period**, apparently program-wide rather than
   per track — verify on the portal.
3. **No `LICENSE` file is committed.** The call asks for it and §6 points at it.
   Apache-2.0 recommended. Five-minute fix.
4. **Commit the `malicious-security` branch.** Last commit 2026-08-21; nine
   files uncommitted. Real answer is 129 commits in three months, but GitHub
   currently looks stalled.
5. **Budget figures** — all amounts are `\TODO`.

Placeholders render red in the PDF. Count: run `grep -c 'TODO{' main.tex`.

---

## Number provenance

Every figure traces to [docs/experiments.md](../docs/experiments.md). Nothing is
modelled or extrapolated; the appendix says so explicitly.

| Claim | Source |
|---|---|
| CKKS **wrong class** at 13 levels (margin 2q, error 15q) | §VI.11 precision table |
| 1.41 s vs 43.8 s, 32× latency | §VI.10 — same network, same machine, both verified class 6 |
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
2. **Report both HEIR bugs upstream.** Citing filed issues in the proposal is
   much stronger than "we observed"; it also demonstrates the ecosystem
   contribution the call's open-source topic asks about.
3. **Isolate the depth-10 segfault** — HEIR parameter selection vs. OpenFHE
   1.5.1 vs. genuinely needing bootstrapping. This is the one claim in §3 that
   rests on an unexplained crash, and a reviewer will ask.
4. **A trained fixture for the depth/precision sweep**, which is the standing
   caveat on the wrong-class result.
