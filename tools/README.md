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
| `pneumonia_linear.rs` | `cargo run --example mnist -- --pneumonia` | `train_medmnist.py pneumoniamnist.npz examples/data/pneumonia_linear.rs --linear --epochs 30 --classes normal,pneumonia --dataset PneumoniaMNIST --task "pediatric chest X-ray: pneumonia vs normal"` |

`examples/data/mnist_data.rs` is the one exception: its 784 weights are
hand-written (extracted from `tests/nn.rs`), not trained, so there is nothing
to regenerate. It exists to exercise the circuit, not to classify well.

Requires `torch` and `numpy`. Training is CPU-only and takes minutes
(pneumonia) to ~20 min (organa, 34 561 images × 45 epochs).

## Getting the data

Neither dataset is vendored — both are a download away, and both are cited
rather than redistributed.

```sh
# MedMNIST v2 (CC BY 4.0) — https://medmnist.com
for d in pneumoniamnist organamnist; do
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
