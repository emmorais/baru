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
