# Private inference performance: state, method, and what to do next

Working notes for the **switch-as-bootstrap** paper (`paper/main.tex`,
outline in [`paper-outline.md`](paper-outline.md)) — *not* the cost-model
paper, whose agenda is [`research-agenda.md`](research-agenda.md).

Last measured **2026-07-28** on an Apple-silicon laptop, `--release`,
single-threaded. Ratios transfer between machines; absolute seconds do not.

---

## 1. Where it stands

One encrypted diagnosis, PneumoniaMNIST, trained two-conv CNN, image **and**
weights encrypted, zero bootstraps, bit-exact against the plaintext integer
reference:

```sh
baru bench-medmnist --dataset pneumonia --hybrid 2 --gc --precompute
```

```
latency  : 2896 ms   (fhe 2123 ms, mpc 756 ms)
fhe split: weights 670 ms, mul 634 ms, mask-enc 0 ms, decrypt 668 ms, re-encrypt 151 ms
mpc split: import 21 ms, relu 505 ms, pool 216 ms, argmax 7 ms
offline  : 2028 ms query-independent (keygen + mask pool), NOT in the latency above
```

It was **32.7 s** the same morning. The construction never changed — only
round batching, the protocol chosen for comparisons, and what sits on the
critical path.

| step | pneumonia latency | what changed |
|---|---|---|
| start of day | 32.7 s | — |
| batched scheme-switch import | 5.71 s | 3 rounds per *layer*, not per activation |
| garbled nonlinearities (`--gc`) | 4.94 s | constant-round ReLU + max-pool |
| tournament `argmax` | 4.94 s | (invisible at rtt=0; 2.4× at RTT 20 ms) |
| preprocessed masks (`--precompute`) | **2.90 s** | + 2.03 s offline |

WAN (RTT 20 ms, 1 Gbps): 5.00 s online. The pipeline is **not round-bound** —
that is what the constant-round nonlinearities bought.

### The synthetic control (`bench-cnn --img 28`)

Smaller model (one conv, no hidden dense), so proportions differ — it has two
scheme switches to the medical model's four. Useful because it runs in ~2 s
and prints everything:

```sh
baru bench-cnn --img 28 --gc --precompute
# total 1312ms (fhe 695ms, mpc 600ms) | offline 673ms
# fhe: keygen 10 / input 17 / weights 218 (13x) / mul 209 / mask 0 (0x) / decrypt 216 / re-encrypt 51 (3x)
# mpc: import 11 / relu 412 / pool 158 / argmax 28
```

---

## 2. The method — read this before optimising anything

Three wins today, and **all three were invisible at the granularity we were
measuring**. Each time, the component-level number (`fhe_ms` vs `mpc_ms`) was
correct and the inference drawn from it was wrong.

1. "The MPC segment dominates" was true. "Therefore the MPC protocols are the
   bottleneck" did not follow — 94% of that segment was an unbatched
   scheme-switch import. The paper and README had both already committed to
   the wrong conclusion in print, naming garbled circuits as "the
   highest-value remaining optimization".
2. "Garbled circuits are 2.7–3.4× faster" was true of the garbling
   *primitive*, measured in isolation. The deployable three-party protocol
   including the Y2A return leg is 1.4–1.7×.
3. "FHE is now 84%" was true, and 48% of *that* was encrypting masks — work
   that need not be on the critical path at all.

The rule that falls out: **measure phases, not components, and when a bench
reports both an isolated primitive and a full protocol, quote the protocol.**
`MpcPhases` and `FhePhases` in `src/bench/cnn.rs` exist to make this cheap.
Add a phase before optimising a phase.

Corollary that keeps biting: a **cross-stage ratio** (e.g. `bench-cohort-eval`'s
"selection is N× cheaper than inference") is invalidated whenever either stage
improves. It went 1900× → 388× today. Quote the command's output, never a
remembered number.

---

## 3. Next steps, in priority order

### 3.1 Precompute the encrypted **model** — best remaining win, low risk

**670 ms, 23% of the online diagnosis**, and it is query-independent for a
strictly better reason than the masks were.

`hybrid_conv_open` and `hybrid_matvec_open` call `s.encrypt_poly(ck, &kpoly)`
inside their per-channel / per-chunk loop, so every inference re-encrypts the
whole model. But the model does not change between patients. Under a
per-session key the encrypted weights are uploaded **once** and reused for
every query in that session.

Note this is *better than the mask result*, and the distinction belongs in the
paper:

- **Masks** are per-query — precomputing shifts latency but buys **no
  throughput**. A saturated server does the same total work.
- **Weights** are per-*session* — precomputing is genuine **amortization**.
  Total work drops by 670 ms per query after the first.

**How:** generalise `MaskPool` into a session preamble holding both the mask
pool and the encrypted model. Follow the existing shape exactly: build it
before `start`, report it in `offline_ms`, assert exact consumption so the
size prediction cannot drift from the pipeline
(`MaskPool::for_model` + `rows_per_chunk` is the pattern).

**Trap:** the model ciphertexts must be encrypted under the *same* key as
everything else. This already bit once — the first `--precompute` draft called
`s.keygen()` a second time for the pool, which produces masks that do not
decrypt. Hoist keygen above the preamble.

**Verify:** extend `precomputed_masks_change_nothing_but_the_timing` — same
class, both nonlinearity routes, and the two-conv model (multi-layer chunking
is where a size prediction goes wrong).

**Expected:** 2.90 s → ~2.23 s online.

### 3.2 Store the weight ciphertexts **pre-transformed** (NTT domain)

Follows directly from 3.1 and needs it first.

`mul_pre(a: &BfvCtNtt, b: &BfvCt)` takes the input pre-transformed and
transforms `b` (the weights) inside `exact_bfv_tensor_pre`, once per output
channel. With one input and *N* weight ciphertexts per layer, almost all the
forward-NTT work is on the weight side — and if the weights are already
offline, their transforms can be too.

**How:** add a both-operands-transformed entry point
(`exact_bfv_tensor_pre_pre` / `Bfv::mul_pre_pre`) and store `BfvCtNtt` in the
preamble instead of `BfvCt`. `BfvCtNtt` is currently not `Clone`/serialisable
— it will need to be.

**Expected:** roughly 2 of the ~5 NTTs per channel, so ~40% of the 634 ms
`mul` phase. Estimate only — **measure before quoting**. Cumulative with 3.1:
~2.0 s online.

### 3.3 Parallelism

Everything is single-threaded. The per-output-channel FHE work (encrypt, mul,
decrypt) is embarrassingly parallel — conv2 has 16 independent channels — and
garbling parallelises across gates.

**Do this last of the three, and keep a single-thread control.** The paper
currently says "single-thread on commodity hardware ... read as ratios";
threading changes what every number means, so it needs its own flag
(`--threads N`, default 1) and its own row, not a silent global change.

**Expected:** near-linear in cores on the FHE phases. This is the largest
absolute win available and the least interesting scientifically.

### 3.4 Things measured and deliberately *not* worth doing

- **Faster decryption by computing only the coefficients we read.** We decrypt
  a full polynomial and then read `oh·ow` positions. Selective evaluation of a
  polynomial product is not obviously cheaper than the NTT that computes all of
  it. Price it analytically before writing code.
- **Optimising `import`.** It is 21 ms. It was 19 100 ms this morning; that is
  finished.
- **Optimising `argmax` further.** 7 ms on 2 classes, 30 ms on 11.
- **More MPC protocol work generally.** The MPC segment is 756 ms of 2896 ms
  and its largest phase (ReLU, 505 ms) is already constant-round garbled.

---

## 4. Open paper gaps (independent of performance work)

- **No latency comparison against prior systems.** `tab:sota` compares only
  *what is hidden*. We now claim seconds per inference; Gazelle / Delphi /
  Cheetah / CrypTFlow2 all report latencies. A reviewer will ask. The honest
  framing is a two-axis table — they are faster *and* hide less (plaintext
  weights, 2-party) — but the numbers have to be in it. **Do not** put their
  numbers and ours in one column without the hardware and threat-model caveat.
- **`tab:arch` mixes training budgets.** The winning row is a longer retrain
  (98.89%); the other rows are the original sweep. Flagged in the caption, but
  re-running the sweep under one recipe would remove the asterisk.
- **The throughput caveat for preprocessing** (see 3.1) is stated for masks in
  §7.5 but should be sharpened once weights are preprocessed too, because the
  two differ in kind.
- **`--precompute` offline cost exceeds online cost** for the synthetic model
  (673 ms vs 1312 ms is fine; on the medical model 2028 vs 2896 is also fine —
  but check this stays true as online shrinks). If offline ever exceeds online
  by a lot, a continuously-loaded server cannot keep up and the framing has to
  change from "latency" to "latency, given spare offline capacity".
- **MedMNIST scope.** Already stated: downsampled, pre-processed, no clinical
  claim. Keep it stated.

---

## 5. Reproduction

```sh
cargo build --release

# headline: private diagnosis
./target/release/baru bench-medmnist --dataset all --hybrid 3 --gc --precompute
./target/release/baru bench-medmnist --dataset all --hybrid 2 --gc --precompute \
    --rtt-ms 20 --bandwidth-mbps 1000        # WAN

# the fast control, prints every phase
./target/release/baru bench-cnn --img 28 --gc --precompute
./target/release/baru bench-cnn --img 28 --both     # a2b vs gc, refuses to
                                                    # print a ratio unless both
                                                    # agree bit-exactly
# isolated protocol comparison
./target/release/baru bench-gc-compare

# correctness
cargo test --release --lib -- --ignored bench::cnn gadgets
```

`bench::cohort` needs `BARUDB_BIN=~/github/barudb/target/release/barudb` and
`BLINDSEARCH_PASSWORD` (not `BARUDB_PASSWORD`). The two `bench::net` tests
measure a real socket and are load-sensitive — they fail under a loaded
machine and pass in isolation; that is not a regression.

Key tests to keep green when touching any of this:

| test | pins |
|---|---|
| `two_conv_hybrid_matches_plaintext` | the flagship exactness claim |
| `gc_and_a2b_nonlinearities_agree` | the two nonlinearity routes are the same function |
| `precomputed_masks_change_nothing_but_the_timing` | preprocessing is not cheating |
| `scheme_switch_import_does_not_scale_with_layer_width` | import stays batched (message count, not wall-clock) |
| `argmax_breaks_ties_to_first_over_every_small_input` | tournament tie-breaking, exhaustive |
| `batched_input_sharing_matches_the_scalar_protocol` | batched == scalar sharing |

---

## 6. Where the code is

| thing | location |
|---|---|
| pipeline | `bench::cnn::run_cnn_hybrid_full` |
| phase structs | `bench::cnn::{MpcPhases, FhePhases}` |
| nonlinearity switch | `bench::cnn::Nonlin`, `TransportBackend::set_gc_nonlinearities` |
| batched import | `aby3_protocol::share_input_batch`, `bench::cnn::import_raw_batch` |
| mask preprocessing | `bench::cnn::{MaskPool, Masks}` |
| tournament argmax | `gadgets::argmax` |
| BFV / NTT | `fhe::bfv`, `fhe::ntt` |
| conv packing | `fhe::conv_pack` |

Docstrings in `bench::cnn` and `bench::medmnist` carry the current measured
tables. **If you change a number, change them in the same commit** — a stale
docstring citing a measurement is worse than no docstring
(see the reproducibility note in `tools/README.md`).
