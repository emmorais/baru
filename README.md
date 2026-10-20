# Baru

[![CI](https://github.com/emmorais/baru/actions/workflows/ci.yml/badge.svg)](https://github.com/emmorais/baru/actions/workflows/ci.yml)

A Lisp-like language and research engine for **privacy-preserving computation**.
One program can run on four cryptographic backends — a plain interpreter, 3-party
MPC (ABY3-style replicated secret sharing), FHE (BFV/BGV/TFHE), and
zero-knowledge proofs (Binius64) — and, crucially, on **combinations** of them:
values cross between MPC shares and FHE ciphertexts mid-computation without ever
being exposed.

The research result the stack was built for: **deep private inference with zero
FHE bootstraps.** The scheme switch that computes each ReLU under MPC also
refreshes the ciphertext, so the FHE noise budget resets at every layer
boundary. An arbitrarily deep network then runs at cost linear in depth on an FHE
engine that supports only a *single* multiplication — with both the model and the
input encrypted, and only the class index ever revealed.

| | |
|---|---|
| **Language reference** | [`LANGUAGE.md`](LANGUAGE.md) — syntax, types, every builtin, the REPL, worked examples |
| **All measured results** | [`docs/experiments.md`](docs/experiments.md) — the full experimental record, including the negative results |
| **Where to pick up performance work** | [`docs/private-inference-performance.md`](docs/private-inference-performance.md) — written to be read cold |
| **Cross-paradigm survey (SoK)** | [`docs/survey.md`](docs/survey.md) — MPC, FHE, ZKP and hybrids measured in one harness; the methodology, and what it caught. LaTeX in [`paper-survey/`](paper-survey/) |
| **Publication framing** | [`docs/paper-outline.md`](docs/paper-outline.md), [`docs/paper-circuit-optimization.md`](docs/paper-circuit-optimization.md) |

---

## Quickstart

```sh
cargo build                          # build everything
cargo test                           # unit + integration tests
cargo run --bin baru                 # single-process REPL (no networking)

cargo build --release
./target/release/baru demo-private-mnist    # the flagship demo, ~55 ms/digit
```

```lisp
; private ReLU on a single neuron — every value is secret-shared
(let ((w :sint8 -1) (x :sint8 3) (b :sint8 0))
  (let ((z (+ (* w x) b)))
    (reconstruct (* z (> z 0)))))          ; => 0

; a secret computed under MPC crosses into FHE without being revealed
(let ((keys (fhe-keygen)) (x (share 6 32)) (y (share -7 32)))
  (fhe-dec keys (share->fhe keys (* x y))))   ; => -42
```

### Building on Windows

`cargo build` works from a plain shell: the default build is pure Rust with no
C++ toolchain, no cmake and no bindgen. That was not true before 2026-08-03,
when the vendored SEAL/CKKS backend needed all three plus a header the modern
Windows SDK stopped shipping; removing CKKS removed the build helper with it.

`--features pq-tls` additionally needs **NASM** on `PATH`
(`winget install NASM.NASM`), which `aws-lc-rs` uses.

Optional features: `--features pq-tls` enables post-quantum TLS transports.
There is no longer an NTL+GMP polynomial backend — it was removed on 2026-08-03
once the hand-written NTT beat it by 1.8× at every size, so **the default build
is the only build** and needs no C++ toolchain. What profiling NTL taught us
(Shoup-preconditioned butterflies) is what made the NTT faster, and the
differential validation it provided is covered by `fhe::ntt`'s schoolbook tests,
which run at full ring dimension and ±2⁹⁹ coefficients.

---

## What is in the box

| Layer | Location | What it provides |
|---|---|---|
| **Interpreter** | `src/interpreter/` | Lisp parser, AST, trampolined evaluator, REPL, pluggable `Backend` |
| **MPC** | `src/crypto/` | ABY3 replicated shares, **triple-free** multiplication, bit decomposition, batched primitives, in-process and TCP transports. **Malicious security with abort** (`crypto::aby3_verify`, `interpreter::malicious_backend`) |
| **FHE** | `src/fhe/` | BFV and BGV (secure, depth-1, no eval keys), TFHE-rs, an exact real-arithmetic backend, multi-modular NTT, CRT slot packing |
| **Hybrid** | `src/hybrid/` | MPC ↔ FHE scheme switching in both directions; re-encryption as a bootstrapping alternative; threshold-key switch with derandomized, checkable operations (`hybrid::threshold_switch`) |
| **ZKP** | `src/zkp/` | Expression → circuit compiler, **real Binius64 prover/verifier** (`zkp::binius_real`; arithmetic, signed comparisons, `if`, `let`), proof serialization — see the ZKP note below |
| **Gadgets** | `src/gadgets/` | UC-composable MPC building blocks: `relu`, `maxpool2d`, `argmax`, `dense`, `conv2d` — each bit-exact against a plaintext reference |
| **MPC compiler** | `src/mpcc/` | Backend-aware circuit IR, rewrite rules, cost models, protocol registry (dealer / Shamir / Semi2k / LowGear), executable VM |
| **Primitives** | `src/ot/`, `src/psi/` | Chou–Orlandi + IKNP OT (with KOS check and Schnorr PoK), DH-PSI (with a batched DLEQ proof) — measured, including where they lose |
| **ML plumbing** | `src/ml/`, `src/precompile/` | Model import + Q16.16 quantization with an accuracy report; champion-dispatched batched gadgets |
| **Node** | `src/bin/tcp_party.rs` | Multi-party node: a distributed MPC evaluator over real sockets (plaintext TCP or optional PQ-TLS) |

The interpreter's `Backend` trait bridges the layers: every arithmetic or
comparison operation on a shared value dispatches through a backend that runs the
appropriate protocol round.

### Security posture

Every engine **attests** what it actually implements
(`crypto::attestations` — one table, readable in a sitting), and asking a
session for more than an engine attests is an error rather than a silent
downgrade. Today exactly one engine attests malicious security:
`MaliciousBackend`, the ABY3 layer, secure **with abort** against one corrupt
party in three. Everything else is semi-honest and says why, naming the specific
gap.

That gate is deliberate. The malicious upgrade landed in stages, and a
half-finished stage must not be mistakable for a finished one — by a caller, a
benchmark, or a paper table.

---

## Results at a glance

Numbers from the maintainer's machine, release build. Full method, provenance and
the corrections behind each row are in [`docs/experiments.md`](docs/experiments.md).

| Result | Number | Command |
|---|---|---|
| **Deep inference, zero bootstraps** | depth 32 × width 16 in **2.1 s**, bit-exact, **0** bootstraps, linear in depth | `bench-deep-nn --depths 2,4,8,16,32` |
| One private dense layer, secure hybrid | **79 ms** (BFV linear → ABY3 ReLU) vs 25 ms pure MPC — but **15 wire messages** vs 9276 | `bench-hybrid-nn` |
| Private CNN, 28×28, image *and* weights encrypted | **260 ms**, 0 bootstraps, exact | `bench-cnn --img 28 --gc --precompute` |
| Real MNIST accuracy, deployed integer model | **98.89%**; every encrypted prediction equals the plaintext one | `bench-mnist` |
| Private *diagnosis* (MedMNIST chest X-ray) | **84.78%**, **268 ms** online per diagnosis (435 ms single-core), 0 bootstraps, encrypted==plaintext on **300/300** | `bench-medmnist` |
| **The same CNN written *in Baru*** (`examples/medmnist.baru`) | **299 ms** plaintext, **911 ms** with the image secret, **4.17 s** with image *and* model secret (`--nonlin gc`) — bit-exact against the Rust model on all 3 datasets. The *encrypted* Rust pipeline (268 ms) is faster than the *plaintext* Lisp one | `run examples/medmnist.baru --mpc --gc` |
| Private *training* (SGD under MPC, backprop through ReLU) | trained model **bit-identical** to plaintext integer SGD, in **3 643** wire messages | `bench-train --hidden 8` |
| Convolution packing | 1352 ct×ct multiplications → **8** | `bench-cnn --img 28` |
| Our NTT vs schoolbook | **40.6× faster** at n=4096 (1.77 ms vs 71.9 ms); it also beat NTL+GMP by 1.8× before that backend was removed | `bench-polymul` |
| Triples from SHE vs from OT | **384 B/triple** vs 3074 at equal 64-bit width (**8.0×**), and ~28× faster | `bench-triples` |
| Compiler vs expert hand-written network | **0.38×** the interactive calls, **0.49×** the messages | `bench-compiled-nn` |
| **MPC vs FHE at a matched threat model** (2-party dishonest majority, same exact function, MNIST-shaped MLP) | LAN, preprocessing hoisted on both sides: **20 ms** MPC vs **213 ms** FHE (**10×**). Cold: **319 ms** vs **323 ms** — a tie. WAN: FHE wins ~**30×** | `bench-2pc-nn`, `bench-heir` |
| FHE baseline is not a strawman | **0.21–0.49 s** via HEIR→OpenFHE, against Orion's published **0.23 s** MNIST | `bench-heir --shapes 784x32x10` |
| MPC's advantage **widens with depth**, then FHE stops | LAN warm ratio **10× → 47×** over depth 3–9 (cold **1.0× → 3.2×**); FHE **SIGSEGVs at depth 11** while MPC runs on linearly | `bench-heir --shapes …`, `bench-2pc-nn` |
| **Cost of active security** (malicious ABY3, with abort) | comparisons: **2.63×** bytes but only **1.06×** rounds, **1.23×** wall clock on a WAN | `bench-malicious` |
| **Proving** a Q16.16 quantized neuron (784 inputs) | **32 ms** prove, **4.8 ms** verify, 112 KB proof; constraints linear in width, proof size far sublinear | `bench-zkp-layer` |

Honest negatives are first-class here and kept alongside the wins: PIR loses to
downloading the database by 13×; CKKS's approximate comparison costs seconds per
operation with real error; hybrid-TFHE does not pay; faster ring arithmetic does
**not** bring FHE close to MPC; and on a secure ring dimension the hybrid does
not beat pure ABY3 at any shape or bandwidth measured, nor SHE-preprocessed MPC
at any width.

**Losing components are deleted, not kept.** PIR, the SEAL/CKKS backends, an
unsound ZKP placeholder, the NTL polynomial backend and a runtime backend switch
that existed only to compare against it were all removed once the measurement
came in; each one's *result* stays in `docs/experiments.md`, and where a result
still feeds a live claim it survives as the smallest thing that can carry it
(`fhe::leveled_relu_cost` is the CKKS experiment reduced to the level arithmetic
`bench-deep-nn` needs). The hybrid's case is the *deployment shape* — the client
sends ciphertexts and leaves, only the nonlinear segment needs live parties —
not raw cost. See
[Part VII](docs/experiments.md#part-vii--negative-superseded-and-withdrawn-results).

---

## A note on the ZKP backend

`src/zkp/binius_real.rs` is a **real proof system**: it builds a Binius64
constraint system with `binius_frontend`, proves with `binius_prover`, and
verifies with `binius_verifier`. `tests/zkp_real.rs` checks the properties that
matter — an honest proof verifies; proving without a witness fails; a witness
that does not satisfy the statement is rejected; a proof does not transfer to a
different statement; a tampered transcript is rejected.

Scope is stated rather than implied. It compiles `+`, `−`, `×`, `/`, `mod`,
the signed comparisons `<`, `>`, `<=`, `>=`, `==`, arithmetic shifts by a public
constant, plus `if`, `let` and sequences, over 64-bit words — enough to prove a
**ReLU neuron** and a **Q16.16 multiply-rescale**, which are the shapes private
inference is built from. Lambdas and list operations are **refused with an
error**; `bench-suite` reports anything it cannot compile as unsupported instead
of skipping it.

Three details in that circuit are easy to get silently wrong, so they are worth
knowing about if you extend it.

The comparison gates are **unsigned**, so signed comparison flips both sign bits
first — without that, every negative value compares as larger than every
positive one and `relu(-1)` returns `-1`, a circuit that is perfectly sound
about the wrong function. Those gates also define only the *most significant
bit* of their output, the rest being explicitly undefined, so the result is
extracted with a logical shift rather than a mask.

**Division is the sharp one.** A circuit cannot divide, so the prover supplies
the quotient and remainder as witnesses and the circuit checks them — and the
obvious check `a == b*q + r` with `|r| < |b|` is *unsound*, because circuit
multiplication wraps mod 2⁶⁴. For `a = 1, b = 3` a cheating prover sets `r = 0`
and `q = 3⁻¹ mod 2⁶⁴`; both checks pass and the quotient is nonsense. What
closes it is requiring the 128-bit product's high word to be exactly the sign
extension of the low word, so `b*q` cannot wrap. `tests/zkp_real.rs` drives that
forgery through the real prover and asserts it is rejected.

**The history is worth keeping.** Until 2026-07-31 this backend was
`binius64_backend`, which concatenated hashes and whose `verify` never checked
that a witness satisfied any constraint — anyone holding the public circuit
could forge an accepting proof. **Any ZKP numbers published before that date
should not be quoted.** The placeholder itself was deleted on 2026-08-03: it was
554 lines existing only so a test could re-demonstrate a bug in code nothing
used, and `tests/zkp_real.rs` runs the same attacks against the real prover,
where they are rejected. That is the guard that matters.

---

## How the MPC works, briefly

Every secret `x` is split into three additive shares over a 128-bit ring
(`s₀ + s₁ + s₂ = x mod 2¹²⁸`), with party `i` holding the replicated pair
`(sᵢ, sᵢ₊₁)`. No single party learns anything from its own pair. Signed values
use two's-complement in the declared bit-width, so `(share -3 8)` shares 253.

| Operation | Protocol | Cost |
|---|---|---|
| `+`, `−`, scalar × | local on shares | no communication |
| `*` | ABY3 resharing — **no Beaver triples**, no dealer | 1 ring element to 1 neighbour, 1 round |
| `a2b` | carry-save 3→2 + Kogge–Stone prefix adder | ~9 rounds at 20 bits |
| `<`, `>`, `≤`, `≥`, `==` | `a2b(a−b)`, sign bit or zero test | as `a2b` |
| `reconstruct` | each party sends its component | 1 round |
| batched forms (`a2b_batch`, `lt_batch`, `mul_batch`) | same protocol, all lanes flattened | **the same round count for any batch width** |

That last row is the one that matters in practice. Multiplication needs no
precomputed triples because the correlated randomness is regenerated locally from
pairwise PRF keys, so the only preprocessing is one 32-byte key exchange per
party pair per session. And the recurring lesson of this repository is that **the
arithmetic was never the bottleneck — the round structure was**: four separate
times, a segment-level measurement said "MPC is slow" while a per-phase
breakdown said "one unbatched loop inside it is slow." The details, with numbers,
are in [`docs/experiments.md` Part I](docs/experiments.md#part-i--mpc-protocol-engineering).

Branching on a shared boolean (`(if (> z 0) z 0)`) reveals the branch taken —
one bit. For oblivious code, write a mux: `(* z (> z 0))`.

---

## The hybrid construction

`src/hybrid/` converts values between representations without exposing them:

- **MPC → FHE** — each party encrypts its own additive share component; the
  evaluator homomorphically sums them. The sum *is* the secret, and nobody
  reconstructed it. Needs only homomorphic addition, so every backend supports it.
- **FHE → MPC** — mask-and-open. Each party contributes an encrypted uniform
  mask; only the masked value (uniformly random) is ever decrypted, and the masks
  turn it into additive shares.
- **Re-encryption instead of bootstrapping** — compose the two: noise resets and,
  for levelled CKKS, the full depth budget comes back, for one interactive round
  instead of a bootstrapping circuit.

The FHE side uses the cheapest primitive that exists: **one
ciphertext×ciphertext multiplication, no relinearization keys, no bootstrapping,
no evaluation keys**, decrypting the degree-2 product directly under `s²`.
Reversed-coefficient packing makes a whole matvec one multiplication, and
Cheetah-style reversed-axis packing makes a whole convolution channel one
multiplication. Weights *and* input are encrypted, which is a stronger privacy
setting than Gazelle/Delphi/Cheetah, whose server holds the model in plaintext.

Parameters are `n = 4096, q = 2^100, t = 2^20` (BFV) — inside the ~109-bit
`log q` ceiling for 128-bit security at that dimension. The plaintext modulus is
calibrated from the measured activation range rather than fixed, because a fixed
`t` silently wraps and gives wrong rather than merely noisy results.

Two honesty notes carried everywhere: the scheme switch stands a single client
key in for a threshold decryption (a real deployment uses distributed-key
threshold FHE), and everything here is **semi-honest**.

---

## Command-line surface

```sh
baru                       # REPL
baru bench [--json]        # every backend on a small expression suite
baru quantize --in model.json --out model.quant.json --scale-bits 16 --validate samples.json
baru mpcc <subcommand>     # the MPC compiler: dump / analyse / compare / run / calibrate / optimize / assign
baru db <subcommand>       # the experiment store: put / query / export / get / stats
baru demo-private-mnist    # encrypted image × encrypted model → only the class bit
```

Benchmarks (`baru bench-<name>`; all accept `--json`, most accept
`--rtt-ms` / `--bandwidth-mbps` to model a network):

| Command | Measures |
|---|---|
| `bench-nn` | size-swept private inference: MPC vs FHE cost shapes, real wire counts |
| `bench-hybrid-nn` | one private dense layer under mpc / tfhe / hybrid-tfhe / hybrid-bfv |
| `bench-deep-nn` | **the headline**: deep private inference, zero bootstraps, linear in depth |
| `bench-cnn` | full conv classifier through the hybrid pipeline, 0 bootstraps |
| `run FILE.baru` | evaluate a Baru **source file** on the plain interpreter or on 3-party MPC (`--mpc`), timed — see [`LANGUAGE.md` §10.5](LANGUAGE.md) |
| `bench-mnist`, `bench-medmnist` | trained+quantised CNNs on real MNIST / real biomedical images |
| `bench-train` | private SGD training, including backprop through ReLU |
| `bench-fhe-innerprod` | the minimal-FHE inner product across secure schemes |
| `bench-polymul` | exact negacyclic product: schoolbook vs our NTT, checked for agreement |
| `bench-triples` | Beaver triples from SHE (LowGear) vs from OT (Semi2k) |
| `bench-ot`, `bench-psi` | OT (base vs IKNP), DH-PSI |
| `bench-layer`, `bench-net` | the MPC↔FHE crossover; the cost model against a real socket |
| `bench-2pc-nn` | a whole private classifier at a **matched threat model** — 2-party dishonest majority, the setting where a comparison against single-server FHE is commensurable. Reports warm *and* cold |
| `bench-heir` | drives [HEIR](https://github.com/google/heir) → OpenFHE as the FHE arm of that comparison, so the baseline is tuned C++ rather than our own NTT. Sweeps **depth**, the axis that decides whether FHE must bootstrap. Refuses to quote a latency at insecure parameters, on a disagreeing classification, or when the two arms computed different functions. Needs Python ≥3.10 and an OpenFHE build — see [`tools/README.md`](tools/README.md); `--self-test` exercises the contract without either |
| `bench-circuit`, `bench-backend-aware`, `bench-compiled-nn` | circuit generation, per-backend rewriting, compiled vs hand-written |
| `bench-abstraction`, `bench-gc-compare`, `bench-mixed-layer` | the abstraction tax; garbled vs arithmetic nonlinearities; mixed protocols |
| `bench-cohort`, `bench-cohort-eval` | cohort-conditioned private evaluation over the encrypted store |

---

## Running a multi-party node

```sh
cargo build --bin tcp_party
./target/debug/tcp_party        # no args: in-process demo, 3 threaded parties

# TCP mode — three terminals, same address roster, differing only in the party id
./target/debug/tcp_party 0 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
./target/debug/tcp_party 1 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
./target/debug/tcp_party 2 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
```

Party 0 drives an interactive REPL and broadcasts each expression; every party
evaluates it jointly via `TransportBackend`, so all parties run in lockstep. A
parse or runtime error on party 0 broadcasts `:abort` so the workers re-sync
instead of hanging. `share-from` / `:sintN-from-P` let each party contribute its
own private input.

The two transports run on ports derived from the MPC base port: MPC at `base`,
expression distribution at `base+100`. Setting `BARU_TLS` switches both to
PQ-TLS transports (needs `--features pq-tls`), with each party minting one
long-term Ed25519 identity shared across its channels.

REPL commands on party 0: `:help`, `:examples [category]`, `:example <n>`,
`:verbose on|off`, `:quit`.

> **History.** Until 2026-07-30 this node also carried consensus with
> reputation-weighted leader election, a p2p chat/forum, a REST control plane and
> FHE/MPC/storage job markets (~5,000 lines, and the `axum`/`tokio`/`x25519`/
> `chacha20poly1305` dependencies). None of it was used by the cryptographic
> research this repository is about, so it was removed. It is recoverable from
> git history if that direction resumes.

---

## Repository map

```
src/
  interpreter/  parser, AST, trampolined evaluator, Backend trait,
                Aby3Simulator (values only) / TransportBackend (real protocol)
  crypto/       Fr and Share3, ABY3 protocols (mul, a2b, batched forms),
                garbled circuits, Transport (InprocHandle / TcpTransport), TLS,
                identity; sim.rs wraps the protocols for in-process use
  fhe/          bfv, bgv, bgv_wide (256-bit coeffs), tfhe_backend, mock_backend,
                exact_real_backend, leveled_relu_cost (the CKKS experiment's
                surviving level model),
                ntt (multi-modular, exact over power-of-two q),
                slots (CRT packing),
                conv_pack (Cheetah-style), circuit_emitter
  hybrid/       MPC <-> FHE scheme switching, mpc_reencrypt
  gadgets/      relu, max2/max_reduce/maxpool2d, argmax, dense, conv2d
  mpcc/         circuit IR, rewrite rules (opt), cost + fhe_cost + ot_cost models,
                protocol registry, semi2k, lowgear, mixed, scheduler, VM, exec
  zkp/          CircuitCompiler (FlatIR), binius_real (real Binius64), serialization
  ot/ psi/      Chou-Orlandi + IKNP; DH-PSI
  ml/           NetworkSpec/QuantizedNetwork, `baru quantize`
  precompile/   Gadget/Registry/champion dispatch, batched matvec
  db/           the experiment store (BaruDB client / local JSONL backends)
  bench/        one module per `baru bench-*` command
  demo/         `baru demo-private-mnist`
  bin/          tcp_party — the multi-party node
docs/
  experiments.md                  every measured result, organized by topic
  private-inference-performance.md where the pipeline's time goes; what to do next
  paper-outline.md                 claims, proof sketch, security model, related work
  paper-circuit-optimization.md    the circuit-optimization paper's plan and gaps
  mpc-compiler-roadmap.md          where `mpcc` is going
  research-agenda.md               private AI for healthcare; the admission rule
  gadget-costs.md                  curated cost table with per-row provenance
  related-work.md                  what SING / CirC say, what Baru took and did not
  malicious-hybrid.md              what malicious security would cost this design
  barudb-api.md                    the external interface of the encrypted store
tools/train_medmnist.py            trains + quantises the models the benches use
paper/, paper-circuits/            LaTeX sources
tests/                             interpreter, crypto, FHE differential, TCP
```

---

## Known limitations

- **Semi-honest only.** No authentication, MACs or abort handling; a malicious
  party can deviate undetected. IKNP has a concrete gap here (an inconsistent
  choice vector leaks the sender's bits — KOS15's correlation check is what
  closes it), and LowGear lacks the zero-knowledge proofs of plaintext knowledge
  that make Overdrive malicious-secure. What malicious security would cost is
  sketched in [`docs/malicious-hybrid.md`](docs/malicious-hybrid.md).
- **The scheme switch simulates threshold decryption** with a single client key.
  A real deployment needs distributed-key threshold FHE; the data flow is already
  the right one.
- **Condition leakage.** `if` on a shared boolean reveals the branch taken. Use
  mux-style expressions for oblivious code.
- **Ring size.** The 128-bit ring wraps on overflow; there is no prime-order
  field.
- **Single-evaluator model.** All parties evaluate the same expression text.
- **Coefficient packing bounds image size.** CIFAR-scale convolution needs
  SIMD-slot packing (Gazelle-style) — a known next step, independent of the
  no-bootstrap result.

---

Contributions and issues are welcome.
