# Baru

[![CI](https://github.com/emmorais/baru/actions/workflows/ci.yml/badge.svg)](https://github.com/emmorais/baru/actions/workflows/ci.yml)

A Lisp-like language and research engine for **privacy-preserving computation**.
One program can run on four cryptographic backends — a plain interpreter, 3-party
MPC (ABY3-style replicated secret sharing), FHE (BFV/BGV/TFHE/CKKS), and
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

The vendored SEAL/CKKS backend (`third_party/sealy`) is a C++ library built via
cmake + MSVC, so a plain `cargo build` from a bare shell fails — bindgen can't
find the Windows SDK headers. Use the helper, which enters the Visual Studio
developer environment and adds the include/PATH entries SEAL needs:

```bat
build-windows.cmd build
build-windows.cmd test --lib
build-windows.cmd build --features pq-tls
```

One-time prerequisites: **Visual Studio 2022** (or Build Tools) with the *Desktop
development with C++* workload; **CorError.h**, which SEAL's C wrapper includes
but the modern Windows SDK no longer ships
(`winget install Microsoft.WindowsSDK.10.0.18362`); and **NASM** for
`--features pq-tls` (`winget install NASM.NASM`). Source patches to the vendored
crate are documented in
[`third_party/sealy/BARU_PATCHES.md`](third_party/sealy/BARU_PATCHES.md).

Optional features: `--features ntl` builds the NTL+GMP polynomial backend (needs
a C++ toolchain plus NTL and GMP) — kept as a **differential oracle** for our own
NTT, which is now the faster of the two. `--features pq-tls` enables
post-quantum TLS transports.

---

## What is in the box

| Layer | Location | What it provides |
|---|---|---|
| **Interpreter** | `src/interpreter/` | Lisp parser, AST, trampolined evaluator, REPL, pluggable `Backend` |
| **MPC** | `src/crypto/` | ABY3 replicated shares, **triple-free** multiplication, bit decomposition, batched primitives, in-process and TCP transports |
| **FHE** | `src/fhe/` | BFV and BGV (secure, depth-1, no eval keys), TFHE-rs, SEAL CKKS (+ exact mock twins), multi-modular NTT, CRT slot packing |
| **Hybrid** | `src/hybrid/` | MPC ↔ FHE scheme switching in both directions; re-encryption as a bootstrapping alternative |
| **ZKP** | `src/zkp/` | Expression → circuit compiler, Binius64 SNARK prover/verifier, proof serialization |
| **Gadgets** | `src/gadgets/` | UC-composable MPC building blocks: `relu`, `maxpool2d`, `argmax`, `dense`, `conv2d` — each bit-exact against a plaintext reference |
| **MPC compiler** | `src/mpcc/` | Backend-aware circuit IR, rewrite rules, cost models, protocol registry (dealer / Shamir / Semi2k / LowGear), executable VM |
| **Primitives** | `src/ot/`, `src/psi/`, `src/pir/` | Chou–Orlandi + IKNP OT, DH-PSI, single-server BFV PIR — all measured, including where they lose |
| **ML plumbing** | `src/ml/`, `src/precompile/` | Model import + Q16.16 quantization with an accuracy report; champion-dispatched batched gadgets |
| **Node** | `src/bin/tcp_party.rs` | Multi-party node: a distributed MPC evaluator over real sockets (plaintext TCP or optional PQ-TLS) |

The interpreter's `Backend` trait bridges the layers: every arithmetic or
comparison operation on a shared value dispatches through a backend that runs the
appropriate protocol round.

---

## Results at a glance

Numbers from the maintainer's machine, release build. Full method, provenance and
the corrections behind each row are in [`docs/experiments.md`](docs/experiments.md).

| Result | Number | Command |
|---|---|---|
| **Deep inference, zero bootstraps** | depth 32 × width 16 in **2.1 s**, bit-exact, **0** bootstraps, linear in depth | `bench-deep-nn --depths 2,4,8,16,32` |
| One private dense layer, secure hybrid | **79 ms** (BFV linear → ABY3 ReLU) vs 25 ms pure MPC — but **15 wire messages** vs 9276 | `bench-hybrid-nn` |
| Private CNN, 28×28, image *and* weights encrypted | **840 ms**, 0 bootstraps, exact | `bench-cnn --img 28 --gc --precompute` |
| Real MNIST accuracy, deployed integer model | **98.89%**; every encrypted prediction equals the plaintext one | `bench-mnist` |
| Private *diagnosis* (MedMNIST chest X-ray) | **84.78%**, **2.90 s** online per diagnosis, 0 bootstraps | `bench-medmnist` |
| Private *training* (SGD under MPC, backprop through ReLU) | trained model **bit-identical** to plaintext integer SGD | `bench-train --hidden 8` |
| Convolution packing | 1352 ct×ct multiplications → **8** | `bench-cnn --img 28` |
| Our NTT vs NTL+GMP | **1.8× faster** at every size (after the Shoup butterfly rewrite) | `bench-polymul` |
| Triples from SHE vs from OT | **192 B/triple** vs 3074, and ~28× faster | `bench-triples` |
| Compiler vs expert hand-written network | **0.38×** the interactive calls, **0.49×** the messages | `bench-compiled-nn` |

Honest negatives are first-class here and kept alongside the wins: PIR loses to
downloading the database by 13×; hybrid-TFHE does not pay; faster ring arithmetic
does **not** bring FHE close to MPC; and the hybrid does not beat
SHE-preprocessed MPC at any width measured. See
[Part VII](docs/experiments.md#part-vii--negative-superseded-and-withdrawn-results).

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
| `bench-hybrid-nn` | one private dense layer under mpc / tfhe / ckks / hybrid-tfhe / hybrid-ckks / hybrid-bfv |
| `bench-deep-nn` | **the headline**: deep private inference, zero bootstraps, linear in depth |
| `bench-cnn` | full conv classifier through the hybrid pipeline, 0 bootstraps |
| `bench-mnist`, `bench-medmnist` | trained+quantised CNNs on real MNIST / real biomedical images |
| `bench-train` | private SGD training, including backprop through ReLU |
| `bench-fhe-innerprod` | the minimal-FHE inner product across secure schemes |
| `bench-polymul` | exact negacyclic product: schoolbook vs our NTT vs NTL+GMP |
| `bench-ckks-compare` | polynomial-approximation sign/ReLU on real SEAL CKKS |
| `bench-triples` | Beaver triples from SHE (LowGear) vs from OT (Semi2k) |
| `bench-ot`, `bench-psi`, `bench-pir` | OT (base vs IKNP), DH-PSI, single-server PIR |
| `bench-layer`, `bench-net` | the MPC↔FHE crossover; the cost model against a real socket |
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
  fhe/          bfv, bgv, bgv_wide (256-bit coeffs), ckks_backend (shallow),
                ckks_compare (deep, polynomial sign), tfhe_backend, mock_backend,
                ntt (multi-modular, exact over power-of-two q), ntl (oracle),
                poly (runtime backend switch), slots (CRT packing),
                conv_pack (Cheetah-style), circuit_emitter
  hybrid/       MPC <-> FHE scheme switching, mpc_reencrypt
  gadgets/      relu, max2/max_reduce/maxpool2d, argmax, dense, conv2d
  mpcc/         circuit IR, rewrite rules (opt), cost + fhe_cost + ot_cost models,
                protocol registry, semi2k, lowgear, mixed, scheduler, VM, exec
  zkp/          CircuitCompiler (FlatIR), Binius64 backend, proof serialization
  ot/ psi/ pir/ Chou-Orlandi + IKNP; DH-PSI; single-server BFV PIR
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
- **CKKS is approximate**, and its exact-comparison story is polynomial
  approximation with real error (measured, not hand-waved).
- **Ring size.** The 128-bit ring wraps on overflow; there is no prime-order
  field.
- **Single-evaluator model.** All parties evaluate the same expression text.
- **Coefficient packing bounds image size.** CIFAR-scale convolution needs
  SIMD-slot packing (Gazelle-style) — a known next step, independent of the
  no-bootstrap result.

---

Contributions and issues are welcome.
