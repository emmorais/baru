# tools/ — regenerating the embedded model weights

Every weight file under `examples/data/` is generated, and this directory holds
the script that generates it. **This file exists because the original MNIST
trainer did not live here.** It was written to a scratch directory, the 98.05%
CNN weights in `examples/data/mnist_cnn.rs` outlived it, and the experiment
became unreproducible — a trained artifact with no way to retrain it. If you
add another generated file, add its command below in the same breath.

## The generated files

| file | consumed by | command |
|---|---|---|
| `mnist_cnn.rs` | `baru bench-mnist` | `train_medmnist.py mnist/ examples/data/mnist_cnn.rs --epochs 30 --test-order first --classes 0,1,2,3,4,5,6,7,8,9 --dataset MNIST --task "handwritten digit recognition"` |
| `pneumonia_cnn.rs` | `baru bench-medmnist --dataset pneumonia` | `train_medmnist.py pneumoniamnist.npz examples/data/pneumonia_cnn.rs --epochs 25 --classes normal,pneumonia --dataset PneumoniaMNIST --task "pediatric chest X-ray: pneumonia vs normal"` |
| `organa_cnn.rs` | `baru bench-medmnist --dataset organa` | `train_medmnist.py organamnist.npz examples/data/organa_cnn.rs --epochs 45 --classes bladder,femur-left,femur-right,heart,kidney-left,kidney-right,liver,lung-left,lung-right,pancreas,spleen --dataset OrganAMNIST --task "abdominal CT: 11-organ classification (axial views)"` |
| `derma_cnn.rs` | `baru bench-medmnist --dataset derma` | `train_medmnist.py dermamnist.npz examples/data/derma_cnn.rs --epochs 40 --rgb --classes actinic-keratoses,basal-cell-carcinoma,benign-keratosis,dermatofibroma,melanoma,melanocytic-nevi,vascular-lesion --dataset DermaMNIST --task "dermatoscopy: 7-class skin lesion"` |
| `pneumonia_linear.rs` | `cargo run --example mnist -- --pneumonia` | `train_medmnist.py pneumoniamnist.npz examples/data/pneumonia_linear.rs --linear --epochs 30 --classes normal,pneumonia --dataset PneumoniaMNIST --task "pediatric chest X-ray: pneumonia vs normal"` |

**`derma_cnn.rs` is the RGB one, and `--rgb` is load-bearing.** Without it the
dataset is folded to luminance and the result is a different experiment: measured at
40 epochs, colour is worth **AUC 0.8519 against 0.7667** and balanced accuracy
**30.14% against 23.30%**. Dermatoscopy is where colour should matter, and it does.
The fold remains the default for every other set because every figure in the repo was
measured that way.

**`organa_cnn.rs` is `--conv 16,32`, not the default `8,16`.** Its second convolution
takes 16 channels at 13×13, which needs degree 5266 against a ring dimension of 4096
and therefore runs as two channel groups — free homomorphically, at two ciphertext
multiplications per output channel instead of one. That widening is worth **82.93%
against 81.64%** on 17 778 test images, with disjoint confidence intervals. Pneumonia
was left at `8,16` because the same widening gained it nothing: 4 708 training images
overfit the extra capacity, and its validation accuracy rose while its test accuracy
fell.

`examples/data/mnist_data.rs` is the one exception: its 784 weights are
hand-written (extracted from `tests/nn.rs`), not trained, so there is nothing
to regenerate. It exists to exercise the circuit, not to classify well.

The three CNN files above were emitted before the trainer took `--conv`/`--fc`
and before it recorded the float parent's accuracy, so each carries
`FLOAT_TEST_ACCURACY = None` and `bench-medmnist` omits the float-parent line
for them. Rerunning the command in the table populates it. The number is
deliberately not filled in by hand: a hand-maintained copy of a measured value
is exactly the drift these generated files exist to prevent.

Requires `torch` and `numpy`. Training takes minutes (pneumonia) to ~20 min
(organa, 34 561 images × 45 epochs); `--device` picks the backend and defaults
to Apple MPS when available. Only the float training runs there — the integer
reference is numpy on the CPU either way, and it is the integer model that gets
deployed, so the device can never move the reported number.

## Choosing the architecture

The Rust side fixes only K=3 and POOL=2: `CnnModel` carries `extra_conv` and
`hidden` as arbitrary-length `Vec`s, so conv depth and dense depth are data,
not code. `--conv 8,16 --fc 128` is the default and reproduces the files above;
`--conv 32,64 --fc 256,128` and friends go deeper.

Three different limits bind, and it is worth knowing which one you hit:

- **Spatially**, 3×3 valid convolution plus floor-pooling takes 28 → 13 → 5 → 1,
  so a 28×28 input admits at most three convs and the third leaves no extent at
  all. A stack that does not fit is refused with the offending layer named.
- **By packing**, the Cheetah-style conv packing encodes a whole layer into one
  polynomial, so the product degree must stay under the ring dimension (4096).
  This binds on *input channels × spatial extent*, so it bites hardest at the
  **second** conv, where the channels have grown but the image has been pooled
  only once: at 13×13 the packing admits at most **12** input channels. So
  `--conv 16,32` is already too wide — it trains and scores fine in plaintext
  and cannot be run encrypted at all. `check_packing` mirrors
  `ConvShape::required_degree` and refuses it before training starts, rather
  than letting it surface twenty minutes later from inside the FHE code.
- **Numerically**, every activation must stay inside the depth-1 FHE plaintext
  modulus, and a wider layer sums more terms into each accumulator — so capacity
  is capped by the noise budget, not by memory. `--gains` extends the sweep
  downwards for wide nets that overflow at the default ladder.

The last two mean a net that trains fine in float can still be refused. **That
refusal is a result — report it rather than lowering `shift` until it passes.**
Going deeper is the way to spend the budget that is left: `--conv 8,12,32` is
three convs, runs encrypted bit-exactly, and needs no bootstrap.

Four of the twelve MedMNIST v2 2D sets ship RGB (Path, Derma, Blood, Retina).
**`--rgb` keeps the colour; without it they are folded to BT.601 luminance.**

This used to say the FHE conv packing "assumes one input channel", and that was
wrong — the fourth place in this repo to say it. `fhe::conv_pack` has always been
multi-channel, because channels summing on a shared coefficient is the entire point
of its reversed offset; what was single-channel was the Rust plumbing, which
hardcoded `in_c = 1`. An RGB 28×28 convolution needs degree 3977 against a ring
dimension of 4096, so it fits one ciphertext at **one multiplication per output
channel** — three channels do not cost three times the work.

The fold is therefore a choice, and it stays the default because every other weight
file and every accuracy figure in the repo was measured with it. When it is applied
the run says so and the generated file records it, and such an accuracy must be read
against a grayscale baseline rather than the published RGB ResNet-18.

## Getting the data

Neither dataset is vendored — both are a download away, and both are cited
rather than redistributed.

```sh
# MedMNIST v2 (CC BY 4.0) — https://medmnist.com
for d in pneumoniamnist organamnist dermamnist; do
  curl -sLo $d.npz "https://zenodo.org/records/10519652/files/$d.npz?download=1"
done

# MNIST — idx.gz, the classic layout
mkdir -p mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sLo mnist/$f.gz "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
done
```

MNIST ships no validation split, so the trainer carves the last 1/12 of the
training set off for one. That matters: the quantisation gain is chosen on
validation and never on test.

## What the script actually has to get right

The long comments in `train_medmnist.py` are the real documentation; the short
version is that the *integer* model is the deployed one, so quantisation is not
a post-processing detail but part of the model:

- Activations stay at the raw 0..255 pixel scale, which pins the mapping from
  float weights trained on `x/255`: weights `round(W·2^shift)`, conv bias
  `round(B·255)`, dense bias `round(B·255·2^shift)`.
- A per-layer gain must be applied to the **output** layer too. ReLU
  homogeneity lets a gain on one layer be undone on the next, but the last
  layer has no next — and argmax is scale-invariant, so its gain needs no
  compensation, which is exactly why it must get one. Omit it and the logit
  weights round to zero and the classifier collapses onto its majority class,
  which reads as a bad model rather than a bug.
- Gains multiply activations layer over layer, so they can overflow the 2^28
  the depth-1 FHE plaintext modulus allows. The sweep rejects on a measured
  activation peak.
- For `--linear`, the binding constraint is instead the Q16.16 accumulator
  against the demo's 32-bit MPC shares (2^30). Chest X-rays are dense where
  digits are mostly zero, so this genuinely binds; the script measures the
  running partial sum rather than assuming.

Each run prints the sweep, the integer-vs-float accuracy and the
integer/float prediction agreement. If the integer accuracy sits at the
majority-class baseline, it is the output-layer gain.

## Metrics: why accuracy is not enough

The run also reports **balanced accuracy, AUC, per-class recall, a confusion
matrix and a 95% Wilson interval** over the full test set, and records balanced
accuracy and AUC into the weight file. This is not decoration — on this repo's
own pneumonia weights, accuracy actively misleads:

| | |
|---|---|
| accuracy | 84.78% (test set is ~60% positive) |
| sensitivity / specificity | **98.3% / 65.3%** |
| balanced accuracy | 81.8% |
| AUC | 0.925 (published ResNet-18: 0.944) |

The model earns its accuracy largely by over-calling disease — it flags 42 of
121 healthy scans. Accuracy cannot see that; balanced accuracy and the
confusion matrix can, and the trainer now shouts when sensitivity and
specificity diverge by more than 15 points.

AUC matters for a second reason: MedMNIST's published tables lead with it, so a
benchmark reporting only accuracy cannot be placed against them on their own
headline metric. AUC needs a *score*, so the trainer scores logits
(`int_logits`) rather than just the argmax — which is all the deployed
encrypted pipeline ever reveals.

`src/bench/metrics.rs` mirrors these definitions on the Rust side so
`bench-medmnist` reports the same quantities on the embedded subset. The two
implementations were cross-checked by embedding a whole test set and confirming
they agree to the digit.

---

## `heir_baseline.py` — the FHE baseline driver (not a generator)

The other script here is different in kind: it generates nothing and is not
consumed by a committed file. It is a **driver** that `baru bench-heir` invokes as
a subprocess, reading one request JSON on stdin and writing one report JSON on
stdout. The contract is documented in its own header; `src/bench/heir.rs` is the
other end of it.

| what | value |
|---|---|
| consumed by | `baru bench-heir` |
| path override | `BARU_HEIR_PY` (defaults to `tools/heir_baseline.py`) |
| exercise the contract, no HEIR needed | `baru bench-heir --shape 6x4x3 --self-test` |
| real measurement | see the setup below, then `baru bench-heir --shape 16x8x4 --activation square` |
| interpreter override | `BARU_PYTHON` (must be >= 3.10) |
| OpenFHE install | `BARU_OPENFHE_PREFIX` — the driver derives the four `OPENFHE_*` variables from it |

**Python ≥ 3.10 is required for the real path** — `heir-py` publishes
`cp310-abi3` wheels only. This script itself runs on 3.9 so that `--self-test`
works anywhere.

### Setting up the real path

`heir-py` alone cannot evaluate anything: `heir.compile()` needs an OpenFHE **C++
install** (it emits C++ and compiles it per program) and otherwise dies with
*"Could not build development config. Did you run bazel?"*. The PyPI `openfhe`
wheel does not satisfy it — that is Python bindings, not a toolchain — and there
is no Homebrew formula.

```sh
brew install python@3.12 libomp        # Apple clang ships no omp.h
python3.12 -m venv venv && ./venv/bin/pip install "heir-py[python,openfhe]" "numpy<2.4"

git clone --depth 1 https://github.com/openfheorg/openfhe-development.git ofhe
cmake -S ofhe -B ofhe/build -DCMAKE_INSTALL_PREFIX=$PWD/ofhe-install \
  -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-I/opt/homebrew/opt/libomp/include \
  -DCMAKE_C_FLAGS=-I/opt/homebrew/opt/libomp/include
cmake --build ofhe/build -j8 && cmake --install ofhe/build

export BARU_PYTHON=$PWD/venv/bin/python
export BARU_OPENFHE_PREFIX=$PWD/ofhe-install
```

The driver derives `OPENFHE_INCLUDE_DIR`, `OPENFHE_LIB_DIR`, `OPENFHE_LINK_LIBS`
and `DYLD_LIBRARY_PATH` from the prefix, and symlinks libomp into the OpenFHE lib
dir — the emitted C++ is linked with `-L$OPENFHE_LIB_DIR` only, so `-lomp` has to
resolve from there.

**`numpy<2.4` is load-bearing**: numba 0.62 refuses newer numpy, and the frontend
imports numba even on the MLIR path.

### It emits MLIR, not a decorated function

HEIR's Python decorator runs the function through numba, and a dense layer is not
expressible there — measured on 2026.8.1, `x[0]` is "Unsupported IR Element:
getitem", `@` is "Extension handling for non-integer types is not yet supported",
and `np.sum`/`np.matmul` are unsupported. Elementwise tensor ops do work. So the
driver emits **`linalg` MLIR** and calls `compile(mlir_str=...)`, which is the
level HEIR's ML pipeline actually targets (the Torch route reaches it via
torch-mlir; going direct skips torch entirely).

### Why it refuses things

Two gates, both in `src/bench/heir.rs`, and both there because the failure mode
is documented rather than hypothetical.

**Security parameters.** [google/heir#1232](https://github.com/google/heir/issues/1232)
reports an MNIST MLP at ~30 s and 3 GB with `HEStd_NotSet` — OpenFHE's "do not
check" mode — and **~6 min and 26 GB** at the secure ring dimension. A 12×
factor sits between the insecure and the secure baseline, and the insecure one is
what a casual run produces. So a latency is refused unless the run reports one of
`HEStd_128_classic` / `HEStd_192_classic` / `HEStd_256_classic`. An insecure
baseline is worse than no baseline: it manufactures a speedup out of a parameter
choice.

**The activation, and it is a modelling decision.** CKKS cannot compare, so an
exact ReLU is not expressible on the FHE arm at all, while Baru's MPC arm computes
it exactly. The two arms therefore cannot be made to compute the same function by
tuning a parameter, and the driver refuses to choose:

* `activation: "square"` gives **both** arms `x²` — exact on each side, and what
  Orion does for its MNIST networks. This is the route taken. It costs the MPC arm
  *nothing*: measured identical to ReLU on rounds, multiplications,
  decompositions and bytes, because squaring doubles the fixed-point scale and the
  rescale that fixes that is the same decomposition the comparison was.
* `activation: "relu_poly"` keeps the MPC arm's exact ReLU and approximates it on
  the FHE arm. Not implemented: it needs a minimax polynomial over a justified
  input range, which is a decision and not a default.

Underneath that, a latency is still refused unless the FHE arm reproduces the
plaintext classification — otherwise a too-coarse approximation looks fast.

`--self-test` evaluates in plaintext and honestly reports `HEStd_NotSet`, so it
round-trips the contract *and* is rejected. That is deliberate: it is the fixture
that proves the security gate fires.
