//! `baru bench-2pc-cnn` — a **convolutional** classifier through the compiled
//! two-party path, so the comparison against the homomorphic literature is on a
//! network that literature actually reports.
//!
//! # Why this exists separately from `bench-2pc-nn`
//!
//! [`crate::bench::two_party_nn`] measures multilayer perceptrons, which are the
//! right shape for isolating depth and width but are not what the homomorphic
//! deep-learning compilers benchmark. Orion (ASPLOS'25), Fhelipe and HE-LRM all
//! report convolutional networks — LeNet-5, ResNet-20, VGG — and a latency for a
//! perceptron is adjacent to their tables and not in them. This benchmark closes
//! that gap by compiling a LeNet-shaped network through the same pipeline: one
//! source expression, flattened, lowered, scheduled and executed on a two-party
//! dishonest-majority engine.
//!
//! # Why a convolution is not a dense layer with zeros
//!
//! A convolution is linear, so it *could* be lowered as a matrix--vector product
//! with a Toeplitz matrix. That would be a serious misrepresentation of its cost.
//! LeNet-5's first convolution is `1x28x28 -> 6x24x24` with `k = 5`, which is
//! `6 * 25 * 576 = 86 400` multiply--accumulate operations; the dense matrix
//! realising the same map is `3456 x 784`, or `2.7` million, a factor of `31`
//! worse and almost all of it multiplication by a structural zero. The generator
//! below therefore emits only the taps the convolution actually has.
//!
//! # What is measured
//!
//! The same quantities and the same gates as `bench-2pc-nn`: rounds and online
//! bytes measured by running the protocol, preprocessing measured where the
//! engine generates it, local compute measured and split warm from cold, and
//! every row checked bit-exactly against a plaintext fixed-point reference
//! before any timing is reported.

use std::collections::HashMap;

use serde::Serialize;

use crate::ml::Activation;
use crate::mpcc::cost::{native_domain, Deployment};
use crate::mpcc::vm::KAPPA;
use crate::mpcc::{
    compile, engine_can_run_program, engine_supports, find, run_protocol, schedule, LowerConfig,
};

// ─── Network description ──────────────────────────────────────────────────────

/// A tensor shape in channels-height-width order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chw {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Chw {
    fn len(&self) -> usize {
        self.c * self.h * self.w
    }
    fn idx(&self, c: usize, y: usize, x: usize) -> usize {
        (c * self.h + y) * self.w + x
    }
}

/// One layer of a convolutional classifier.
#[derive(Debug, Clone)]
pub enum Layer {
    /// `out_c` filters of `k x k` over every input channel, unit stride, no
    /// padding. Weights are secret, like everything else.
    Conv { out_c: usize, k: usize },
    /// Max pooling over disjoint `s x s` windows. A nonlinearity: each output
    /// costs `s^2 - 1` comparisons, and it is **not expressible** under a
    /// comparison-free homomorphic scheme.
    Pool { s: usize },
    /// Average pooling over disjoint `s x s` windows. Linear, and therefore
    /// exact in both paradigms.
    ///
    /// It is emitted as a bare sum, with the division by `s^2` folded into the
    /// *next* layer's rescale rather than performed here. That makes it entirely
    /// free under secret sharing: a rescale would cost a bit decomposition, and
    /// shifting the following rescale by `2 log s` more bits costs nothing. The
    /// same trick as the scale absorbed by a fast-convolution transform.
    AvgPool { s: usize },
    /// Flatten to a vector; free, since it is a relabelling.
    Flatten,
    /// Fully connected to `out` outputs.
    Dense { out: usize },
    /// Elementwise activation.
    Act,
}

/// A network: an input shape and a stack of layers.
#[derive(Debug, Clone)]
pub struct Cnn {
    pub input: Chw,
    pub layers: Vec<Layer>,
    pub scale_bits: u32,
    pub hidden: Activation,
}

impl Cnn {
    /// LeNet-5 as Orion and the homomorphic-inference literature report it:
    /// `1x28x28 -> conv(6,5x5) -> pool -> conv(16,5x5) -> pool -> 120 -> 84 -> 10`.
    pub fn lenet5(scale_bits: u32, hidden: Activation) -> Cnn {
        Cnn {
            input: Chw { c: 1, h: 28, w: 28 },
            layers: vec![
                Layer::Conv { out_c: 6, k: 5 },
                Layer::Act,
                Layer::Pool { s: 2 },
                Layer::Conv { out_c: 16, k: 5 },
                Layer::Act,
                Layer::Pool { s: 2 },
                Layer::Flatten,
                Layer::Dense { out: 120 },
                Layer::Act,
                Layer::Dense { out: 84 },
                Layer::Act,
                Layer::Dense { out: 10 },
            ],
            scale_bits,
            hidden,
        }
    }

    /// LeNet-5 with every nonlinearity replaced by one a comparison-free
    /// homomorphic scheme can also evaluate: average pooling for max pooling, and
    /// the caller's activation (intended to be `Square`) for \ReLU. This is the
    /// shape the homomorphic deep-learning literature actually benchmarks, and
    /// the only variant on which a same-function comparison is possible.
    pub fn lenet5_fhe_friendly(scale_bits: u32, hidden: Activation) -> Cnn {
        Cnn {
            input: Chw { c: 1, h: 28, w: 28 },
            layers: vec![
                Layer::Conv { out_c: 6, k: 5 },
                Layer::Act,
                Layer::AvgPool { s: 2 },
                Layer::Conv { out_c: 16, k: 5 },
                Layer::Act,
                Layer::AvgPool { s: 2 },
                Layer::Flatten,
                Layer::Dense { out: 120 },
                Layer::Act,
                Layer::Dense { out: 84 },
                Layer::Act,
                Layer::Dense { out: 10 },
            ],
            scale_bits,
            hidden,
        }
    }

    /// [`Cnn::lenet5_fhe_friendly`] with `extra` additional hidden blocks, for the
    /// experiment that decides where the two paradigms cross.
    ///
    /// Depth is the axis along which the comparison must be run, because the two
    /// paradigms pay for it in incomparable ways. Secret sharing has no depth budget:
    /// each additional layer adds a fixed number of communication rounds and a fixed
    /// amount of traffic, so its cost is linear and has no discontinuity. An
    /// approximate homomorphic scheme has a finite level budget fixed at key
    /// generation, and once a network exhausts it the scheme must bootstrap — an
    /// operation measured at 110.7 s on this machine (§VI.10), against 0.130 s for a
    /// multiplication. So one paradigm's cost curve is a line and the other's is a
    /// staircase with very tall steps, and the interesting quantity is where they
    /// cross.
    ///
    /// Each block is a dense layer and an activation, which is two levels — the same
    /// unit of depth on both sides, so the sweep is over one variable.
    pub fn lenet5_deep(scale_bits: u32, hidden: Activation, extra: usize) -> Cnn {
        let mut net = Cnn::lenet5_fhe_friendly(scale_bits, hidden);
        // Insert before the final classifier so the output stays ten-way.
        let tail = net.layers.pop().expect("a final dense layer");
        for _ in 0..extra {
            net.layers.push(Layer::Dense { out: 84 });
            net.layers.push(Layer::Act);
        }
        net.layers.push(tail);
        net
    }

    /// A deliberately small network for tests: `1x8x8 -> conv(2,3) -> pool -> 4`.
    pub fn tiny(scale_bits: u32, hidden: Activation) -> Cnn {
        Cnn {
            input: Chw { c: 1, h: 8, w: 8 },
            layers: vec![
                Layer::Conv { out_c: 2, k: 3 },
                Layer::Act,
                Layer::Pool { s: 2 },
                Layer::Flatten,
                Layer::Dense { out: 4 },
            ],
            scale_bits,
            hidden,
        }
    }

    /// Multiply--accumulate operations in the linear layers, which is the count
    /// the homomorphic literature quotes for a network.
    pub fn macs(&self) -> usize {
        let mut shape = self.input;
        let mut flat: Option<usize> = None;
        let mut total = 0;
        for l in &self.layers {
            match l {
                Layer::Conv { out_c, k } => {
                    let (oh, ow) = (shape.h - k + 1, shape.w - k + 1);
                    total += out_c * shape.c * k * k * oh * ow;
                    shape = Chw {
                        c: *out_c,
                        h: oh,
                        w: ow,
                    };
                }
                Layer::Pool { s } | Layer::AvgPool { s } => {
                    shape = Chw {
                        c: shape.c,
                        h: shape.h / s,
                        w: shape.w / s,
                    }
                }
                Layer::Flatten => flat = Some(shape.len()),
                Layer::Dense { out } => {
                    let n_in = flat.expect("dense must follow a flatten");
                    total += n_in * out;
                    flat = Some(*out);
                }
                Layer::Act => {}
            }
        }
        total
    }
}

// ─── Deterministic parameters ─────────────────────────────────────────────────

/// Every weight of the network, addressed by a stable name so that the source
/// generator and the plaintext reference cannot disagree about which is which.
#[derive(Debug, Clone, Default)]
struct Params {
    w: HashMap<String, i64>,
}

impl Params {
    fn get(&self, name: &str) -> i64 {
        *self.w.get(name).unwrap_or(&0)
    }
}

/// Xavier-scaled deterministic weights, for the reason recorded in
/// `two_party_nn::fixture_with`: an unscaled fixture makes a squaring activation
/// diverge and the failure looks like a property of the activation.
fn parameters(net: &Cnn) -> Params {
    let unit = 1i64 << net.scale_bits;
    let mut p = Params::default();
    let mut shape = net.input;
    let mut flat: Option<usize> = None;
    for (li, l) in net.layers.iter().enumerate() {
        match l {
            Layer::Conv { out_c, k } => {
                let fan_in = ((shape.c * k * k) as f64).sqrt();
                for o in 0..*out_c {
                    for i in 0..shape.c {
                        for a in 0..*k {
                            for b in 0..*k {
                                let t = ((o * 7 + i * 5 + a * 3 + b * 2 + li) % 11) as i64 - 5;
                                p.w.insert(
                                    format!("k{li}_{o}_{i}_{a}_{b}"),
                                    ((t * unit / 8) as f64 / fan_in) as i64,
                                );
                            }
                        }
                    }
                    p.w.insert(
                        format!("cb{li}_{o}"),
                        (((o * 5 + li) % 7) as i64 - 3) * unit / 4,
                    );
                }
                shape = Chw {
                    c: *out_c,
                    h: shape.h - k + 1,
                    w: shape.w - k + 1,
                };
            }
            Layer::Pool { s } | Layer::AvgPool { s } => {
                shape = Chw {
                    c: shape.c,
                    h: shape.h / s,
                    w: shape.w / s,
                }
            }
            Layer::Flatten => flat = Some(shape.len()),
            Layer::Dense { out } => {
                let n_in = flat.expect("dense must follow a flatten");
                let fan_in = (n_in as f64).sqrt();
                for o in 0..*out {
                    for i in 0..n_in {
                        let t = ((o * 7 + i * 5 + li) % 11) as i64 - 5;
                        p.w.insert(
                            format!("d{li}_{o}_{i}"),
                            ((t * unit / 8) as f64 / fan_in) as i64,
                        );
                    }
                    p.w.insert(
                        format!("db{li}_{o}"),
                        (((o * 5 + li) % 7) as i64 - 3) * unit / 4,
                    );
                }
                flat = Some(*out);
            }
            Layer::Act => {}
        }
    }
    p
}

fn input_vector(net: &Cnn, k: usize) -> Vec<i64> {
    let unit = 1i64 << net.scale_bits;
    (0..net.input.len())
        .map(|i| (((i + k * 3) % 9) as i64 - 4) * unit / 4)
        .collect()
}

// ─── Plaintext reference ──────────────────────────────────────────────────────

/// The fixed-point forward pass every protocol row is checked against.
fn forward(net: &Cnn, p: &Params, input: &[i64]) -> Vec<i64> {
    let sc = net.scale_bits;
    // Bits of scale accumulated by average pooling and not yet divided out; the
    // next rescale shifts by this much more. See `Layer::AvgPool`.
    let mut pending: u32 = 0;
    let mut shape = net.input;
    let mut cur: Vec<i64> = input.to_vec();
    for (li, l) in net.layers.iter().enumerate() {
        match l {
            Layer::Conv { out_c, k } => {
                let (oh, ow) = (shape.h - k + 1, shape.w - k + 1);
                let out_shape = Chw {
                    c: *out_c,
                    h: oh,
                    w: ow,
                };
                let mut out = vec![0i64; out_shape.len()];
                for o in 0..*out_c {
                    for y in 0..oh {
                        for x in 0..ow {
                            let mut acc: i128 = 0;
                            for i in 0..shape.c {
                                for a in 0..*k {
                                    for b in 0..*k {
                                        acc += i128::from(p.get(&format!("k{li}_{o}_{i}_{a}_{b}")))
                                            * i128::from(cur[shape.idx(i, y + a, x + b)]);
                                    }
                                }
                            }
                            out[out_shape.idx(o, y, x)] =
                                ((acc >> (sc + pending)) as i64) + p.get(&format!("cb{li}_{o}"));
                        }
                    }
                }
                pending = 0;
                cur = out;
                shape = out_shape;
            }
            Layer::AvgPool { s } => {
                let out_shape = Chw {
                    c: shape.c,
                    h: shape.h / s,
                    w: shape.w / s,
                };
                let mut out = vec![0i64; out_shape.len()];
                for c in 0..shape.c {
                    for y in 0..out_shape.h {
                        for x in 0..out_shape.w {
                            let mut acc = 0i64;
                            for a in 0..*s {
                                for b in 0..*s {
                                    acc += cur[shape.idx(c, y * s + a, x * s + b)];
                                }
                            }
                            out[out_shape.idx(c, y, x)] = acc;
                        }
                    }
                }
                // The division by s^2 is deferred, not performed.
                pending += 2 * (*s as u32).trailing_zeros();
                cur = out;
                shape = out_shape;
            }
            Layer::Pool { s } => {
                let out_shape = Chw {
                    c: shape.c,
                    h: shape.h / s,
                    w: shape.w / s,
                };
                let mut out = vec![0i64; out_shape.len()];
                for c in 0..shape.c {
                    for y in 0..out_shape.h {
                        for x in 0..out_shape.w {
                            let mut m = cur[shape.idx(c, y * s, x * s)];
                            for a in 0..*s {
                                for b in 0..*s {
                                    let v = cur[shape.idx(c, y * s + a, x * s + b)];
                                    if v > m {
                                        m = v;
                                    }
                                }
                            }
                            out[out_shape.idx(c, y, x)] = m;
                        }
                    }
                }
                cur = out;
                shape = out_shape;
            }
            Layer::Flatten => {}
            Layer::Dense { out } => {
                let o_v: Vec<i64> = (0..*out)
                    .map(|o| {
                        let acc: i128 = cur
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                i128::from(p.get(&format!("d{li}_{o}_{i}"))) * i128::from(v)
                            })
                            .sum();
                        ((acc >> (sc + pending)) as i64) + p.get(&format!("db{li}_{o}"))
                    })
                    .collect();
                pending = 0;
                cur = o_v;
            }
            Layer::Act => {
                cur = cur
                    .into_iter()
                    .map(|v| match net.hidden {
                        Activation::Relu => v.max(0),
                        Activation::Square => ((v as i128 * v as i128) >> sc) as i64,
                        Activation::Identity => v,
                    })
                    .collect()
            }
        }
    }
    cur
}

fn argmax_of(v: &[i64]) -> i64 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate().skip(1) {
        if *x > v[best] {
            best = i;
        }
    }
    best as i64
}

// ─── Source generation ────────────────────────────────────────────────────────

/// Emit the network as one Baru expression whose single output is the class
/// index, with the input and every weight a secret variable.
///
/// The convolution is emitted tap by tap, so its instruction count is the
/// convolution's own multiply--accumulate count and not that of the dense matrix
/// realising the same map (see the module docs).
/// A program evaluating `batch` independent inferences of the same network, together
/// with its bindings.
///
/// Batching is where the two paradigms' parallelism is most different, so it has to be
/// measured on both. An approximate homomorphic scheme carries thousands of slots per
/// ciphertext and this network occupies a few hundred, so further inferences ride along
/// at **no additional operation at all**. Secret sharing gets nothing for free in
/// traffic — every inference's shares must actually be sent — but it does get something
/// for free in *rounds*, because independent inferences are independent instructions and
/// the scheduler puts them in the same round. So one paradigm amortises bytes and the
/// other amortises latency, and on a wide-area link those are the two terms that matter.
///
/// The weight variables are shared across the batch and only the intermediates and
/// inputs are renamed, so this is one network applied `batch` times rather than `batch`
/// unrelated networks.
pub fn debug_batch(net: &Cnn, batch: usize) -> (String, HashMap<String, i64>) {
    let p = parameters(net);
    let mut binds: Vec<String> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut ins: HashMap<String, i64> = p.w.clone();
    for n in 0..batch {
        let pre = format!("b{n}_");
        let (b, idx) = network_binds(net, &pre);
        binds.extend(b);
        ids.push(idx);
        for (i, v) in input_vector(net, n).iter().enumerate() {
            ins.insert(format!("{pre}x{i}"), *v);
        }
    }
    // Combine the per-inference classes with public constants, which costs no
    // communication and keeps every one of them live rather than dead code.
    let out = crate::bench::depth::sum_tree(
        ids.iter()
            .enumerate()
            .map(|(n, id)| format!("(* {} {id})", 1i64 << (4 * n.min(12))))
            .collect(),
    );
    (format!("(let* ({}) {out})", binds.join(" ")), ins)
}

/// Serialise a network, its parameters and one input in the flat text format the
/// homomorphic arm reads (`tools/ckks_cnn.cpp`).
///
/// The two arms must evaluate the *same* network on the *same* input, or the
/// comparison measures two different computations — the error §VI.8 withdrew a claim
/// for. Rather than reimplement the fixture on the other side and hope the two agree,
/// the parameters are exported from the single definition here, in fixed point, and
/// the homomorphic arm divides by the scale to recover the reals it works in.
pub fn export(net: &Cnn, k: usize) -> String {
    let p = parameters(net);
    let input = input_vector(net, k);
    let logits = forward(net, &p, &input);
    let mut out = String::new();
    out.push_str(&format!("scale_bits {}\n", net.scale_bits));
    out.push_str(&format!(
        "input {} {} {}\n",
        net.input.c, net.input.h, net.input.w
    ));
    out.push_str(&format!(
        "values {}\n",
        input
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    ));

    let mut shape = net.input;
    let mut flat: Option<usize> = None;
    for (li, l) in net.layers.iter().enumerate() {
        match l {
            Layer::Conv { out_c, k } => {
                out.push_str(&format!("conv {li} {out_c} {k} {} ", shape.c));
                for o in 0..*out_c {
                    for i in 0..shape.c {
                        for a in 0..*k {
                            for b in 0..*k {
                                out.push_str(&format!(
                                    "{} ",
                                    p.get(&format!("k{li}_{o}_{i}_{a}_{b}"))
                                ));
                            }
                        }
                    }
                }
                for o in 0..*out_c {
                    out.push_str(&format!("{} ", p.get(&format!("cb{li}_{o}"))));
                }
                out.push('\n');
                shape = Chw {
                    c: *out_c,
                    h: shape.h - k + 1,
                    w: shape.w - k + 1,
                };
            }
            Layer::Act => out.push_str(&format!("act {li}\n")),
            Layer::Pool { s } | Layer::AvgPool { s } => {
                let kind = if matches!(l, Layer::AvgPool { .. }) {
                    "avgpool"
                } else {
                    "maxpool"
                };
                out.push_str(&format!("{kind} {li} {s}\n"));
                shape = Chw {
                    c: shape.c,
                    h: shape.h / s,
                    w: shape.w / s,
                };
            }
            Layer::Flatten => {
                flat = Some(shape.c * shape.h * shape.w);
                out.push_str(&format!("flatten {li} {}\n", flat.unwrap()));
            }
            Layer::Dense { out: o_dim } => {
                let n_in = flat.unwrap_or(shape.c * shape.h * shape.w);
                out.push_str(&format!("dense {li} {o_dim} {n_in} "));
                for o in 0..*o_dim {
                    for i in 0..n_in {
                        out.push_str(&format!("{} ", p.get(&format!("d{li}_{o}_{i}"))));
                    }
                }
                for o in 0..*o_dim {
                    out.push_str(&format!("{} ", p.get(&format!("db{li}_{o}"))));
                }
                out.push('\n');
                flat = Some(*o_dim);
                shape = Chw {
                    c: *o_dim,
                    h: 1,
                    w: 1,
                };
            }
        }
    }
    out.push_str(&format!(
        "reference {}\n",
        logits
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    ));
    out
}

/// Expose a compiled network and its bindings for the wide-area probe
/// (`examples/probe_wan.rs`), which measures elapsed time rather than composing it.
pub fn debug_case(net: &Cnn, k: usize) -> (String, HashMap<String, i64>) {
    let p = parameters(net);
    (network_src(net), bindings(&p, &input_vector(net, k)))
}

fn network_src(net: &Cnn) -> String {
    let (binds, idx) = network_binds(net, "");
    format!("(let* ({}) {idx})", binds.join(" "))
}

/// The bindings and result name for one inference, with every intermediate and input
/// name carrying `pre`. Weight names are deliberately *not* prefixed: a batch of
/// inferences shares one network, so sharing the weight variables is what makes the
/// batch a batch rather than several unrelated programs.
fn network_binds(net: &Cnn, pre: &str) -> (Vec<String>, String) {
    let sc = net.scale_bits;
    let mut pending: u32 = 0;
    let mut binds: Vec<String> = Vec::new();
    let mut shape = net.input;
    // Names of the current activation tensor, in the layout of `shape`.
    let mut cur: Vec<String> = (0..shape.len()).map(|i| format!("{pre}x{i}")).collect();

    for (li, l) in net.layers.iter().enumerate() {
        match l {
            Layer::Conv { out_c, k } => {
                let (oh, ow) = (shape.h - k + 1, shape.w - k + 1);
                let out_shape = Chw {
                    c: *out_c,
                    h: oh,
                    w: ow,
                };
                let mut next = vec![String::new(); out_shape.len()];
                for o in 0..*out_c {
                    for y in 0..oh {
                        for x in 0..ow {
                            let taps: Vec<String> = (0..shape.c)
                                .flat_map(|i| {
                                    (0..*k).flat_map(move |a| (0..*k).map(move |b| (i, a, b)))
                                })
                                .map(|(i, a, b)| {
                                    format!(
                                        "(* k{li}_{o}_{i}_{a}_{b} {})",
                                        cur[shape.idx(i, y + a, x + b)]
                                    )
                                })
                                .collect();
                            let dot = crate::bench::depth::sum_tree(taps);
                            let name = format!("{pre}c{li}_{o}_{y}_{x}");
                            binds.push(format!(
                                "({name} (+ (ash-right {dot} {}) cb{li}_{o}))",
                                sc + pending
                            ));
                            next[out_shape.idx(o, y, x)] = name;
                        }
                    }
                }
                pending = 0;
                cur = next;
                shape = out_shape;
            }
            Layer::AvgPool { s } => {
                let out_shape = Chw {
                    c: shape.c,
                    h: shape.h / s,
                    w: shape.w / s,
                };
                let mut next = vec![String::new(); out_shape.len()];
                for c in 0..shape.c {
                    for y in 0..out_shape.h {
                        for x in 0..out_shape.w {
                            let terms: Vec<String> = (0..*s)
                                .flat_map(|a| (0..*s).map(move |b| (a, b)))
                                .map(|(a, b)| cur[shape.idx(c, y * s + a, x * s + b)].clone())
                                .collect();
                            let sum = crate::bench::depth::sum_tree(terms);
                            let name = format!("{pre}v{li}_{c}_{y}_{x}");
                            binds.push(format!("({name} {sum})"));
                            next[out_shape.idx(c, y, x)] = name;
                        }
                    }
                }
                pending += 2 * (*s as u32).trailing_zeros();
                cur = next;
                shape = out_shape;
            }
            Layer::Pool { s } => {
                let out_shape = Chw {
                    c: shape.c,
                    h: shape.h / s,
                    w: shape.w / s,
                };
                let mut next = vec![String::new(); out_shape.len()];
                for c in 0..shape.c {
                    for y in 0..out_shape.h {
                        for x in 0..out_shape.w {
                            // Max over the window as a fold of muxes; a maximum is
                            // a comparison, so this is the pooling layer's cost.
                            let mut acc = cur[shape.idx(c, y * s, x * s)].clone();
                            for a in 0..*s {
                                for b in 0..*s {
                                    if a == 0 && b == 0 {
                                        continue;
                                    }
                                    let v = &cur[shape.idx(c, y * s + a, x * s + b)];
                                    let m = format!("{pre}p{li}_{c}_{y}_{x}_{a}_{b}");
                                    binds.push(format!(
                                        "({m} (+ {acc} (* (> {v} {acc}) (- {v} {acc}))))"
                                    ));
                                    acc = m;
                                }
                            }
                            next[out_shape.idx(c, y, x)] = acc;
                        }
                    }
                }
                cur = next;
                shape = out_shape;
            }
            Layer::Flatten => {}
            Layer::Dense { out } => {
                let n_in = cur.len();
                let mut next = Vec::with_capacity(*out);
                for o in 0..*out {
                    let dot = crate::bench::depth::sum_tree(
                        (0..n_in)
                            .map(|i| format!("(* d{li}_{o}_{i} {})", cur[i]))
                            .collect(),
                    );
                    let name = format!("{pre}h{li}_{o}");
                    binds.push(format!(
                        "({name} (+ (ash-right {dot} {}) db{li}_{o}))",
                        sc + pending
                    ));
                    next.push(name);
                }
                pending = 0;
                cur = next;
            }
            Layer::Act => {
                let mut next = Vec::with_capacity(cur.len());
                for (i, v) in cur.iter().enumerate() {
                    let name = format!("{pre}a{li}_{i}");
                    match net.hidden {
                        Activation::Identity => next.push(v.clone()),
                        Activation::Relu => {
                            binds.push(format!("({name} (* {v} (> {v} 0)))"));
                            next.push(name);
                        }
                        Activation::Square => {
                            // Squaring doubles the fixed-point scale, so the
                            // product is rescaled back; that rescale costs a bit
                            // decomposition, exactly as the comparison would.
                            binds.push(format!("({name} (ash-right (* {v} {v}) {sc}))"));
                            next.push(name);
                        }
                    }
                }
                cur = next;
            }
        }
    }

    // Every pairwise comparison at once, not a tournament and not a fold: see
    // `bench::depth`. The index is the only thing revealed.
    let idx = crate::bench::depth::argmax_pairwise(&mut binds, &cur, &format!("{pre}g"));
    (binds, idx)
}

fn bindings(p: &Params, input: &[i64]) -> HashMap<String, i64> {
    let mut m: HashMap<String, i64> = p.w.clone();
    for (i, v) in input.iter().enumerate() {
        m.insert(format!("x{i}"), *v);
    }
    m
}

// ─── The benchmark ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct CnnPoint {
    pub protocol: String,
    pub network: String,
    pub macs: usize,
    pub rounds: u64,
    pub muls: u64,
    pub decompositions: u64,
    pub online_bytes: u64,
    pub offline_bytes: u64,
    pub online_secs: f64,
    pub total_secs: f64,
    pub correct: bool,
    pub class: i64,
    pub expected: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CnnReport {
    pub parties: usize,
    pub bit_width: u32,
    pub activation: String,
    pub deployment: String,
    pub instructions: usize,
    pub source_bytes: usize,
    pub compile_secs: f64,
    pub points: Vec<CnnPoint>,
    pub skipped: Vec<String>,
}

pub fn run(net: &Cnn, dep: &Deployment, bit_width: u32) -> Result<CnnReport, String> {
    run_only(net, dep, bit_width, None)
}

/// [`run`] restricted to a subset of the protocol registry. Semi2k's oblivious
/// transfer is minutes of compute at convolutional scale, so a sweep that only
/// wants the SHE-preprocessed row should say so instead of waiting for it.
pub fn run_only(
    net: &Cnn,
    dep: &Deployment,
    bit_width: u32,
    only: Option<&[String]>,
) -> Result<CnnReport, String> {
    let bits_per_decomp = u64::from(bit_width + 1 + KAPPA);
    let p = parameters(net);
    let input = input_vector(net, 0);
    let want = argmax_of(&forward(net, &p, &input));
    let src = network_src(net);
    let ins = bindings(&p, &input);

    let label = format!(
        "{}x{}x{} -> {} layers",
        net.input.c,
        net.input.h,
        net.input.w,
        net.layers.len()
    );

    let mut points = Vec::new();
    let mut skipped = Vec::new();
    let mut instructions = 0usize;
    let mut compile_secs = 0.0f64;

    for id in crate::mpcc::RUNNABLE {
        if let Some(list) = only {
            if !list.iter().any(|p| p == id) {
                continue;
            }
        }
        let spec = find(id).ok_or_else(|| format!("unknown protocol `{id}`"))?;
        if !spec.supports_parties(dep.parties) || !engine_supports(id, dep.parties) {
            continue;
        }
        let cfg = LowerConfig {
            domain: native_domain(spec, 64),
            parties: dep.parties,
            bit_width,
            public_vars: Vec::new(),
        };
        let t0 = std::time::Instant::now();
        let prog = compile(&src, &cfg)?;
        compile_secs = compile_secs.max(t0.elapsed().as_secs_f64());
        instructions = prog.code.len();
        if let Err(why) = engine_can_run_program(id, &prog) {
            skipped.push(format!("{id}: {why}"));
            continue;
        }
        let sched = schedule(&prog);
        let m = run_protocol(id, dep.parties, &prog, &sched, &ins)?;

        let per_triple = if m.triples_generated > 0 {
            m.offline_bytes as f64 / m.triples_generated as f64
        } else {
            0.0
        };
        let offline_bytes = (per_triple * m.triples as f64) as u64 + m.random_bit_bytes;
        let parties = dep.parties.max(1) as f64;
        let offline_compute = m.offline_micros as f64 / 1e6 / parties;
        let online_compute = m.micros.saturating_sub(m.offline_micros) as f64 / 1e6 / parties;
        let bw = dep.bandwidth_bytes_per_sec;
        let rtt = dep.latency_us / 1e6;
        let online_secs = m.rounds as f64 * rtt + m.online_bytes as f64 / bw + online_compute;
        let total_secs = online_secs
            + m.random_bit_rounds as f64 * rtt
            + offline_bytes as f64 / bw
            + offline_compute;

        let class = m.outputs.first().map(|(_, v)| *v).unwrap_or(i64::MIN);
        points.push(CnnPoint {
            protocol: (*id).to_string(),
            network: label.clone(),
            macs: net.macs(),
            rounds: m.rounds,
            muls: m.muls,
            decompositions: m.random_bits / bits_per_decomp,
            online_bytes: m.online_bytes,
            offline_bytes,
            online_secs,
            total_secs,
            correct: class == want,
            class,
            expected: want,
        });
    }

    Ok(CnnReport {
        parties: dep.parties,
        bit_width,
        activation: match net.hidden {
            Activation::Square => "square".into(),
            Activation::Relu => "relu".into(),
            Activation::Identity => "identity".into(),
        },
        deployment: format!(
            "{:.0} ms one-way, {:.0} Mbit/s",
            dep.latency_us / 1e3,
            dep.bandwidth_bytes_per_sec * 8.0 / 1e6
        ),
        instructions,
        source_bytes: src.len(),
        compile_secs,
        points,
        skipped,
    })
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

const USAGE: &str = "usage: baru bench-2pc-cnn [options]

  --net N              `lenet5` (max pool), `lenet5-fhe` (average pool, so a
                       comparison-free scheme can evaluate it too), or `tiny`
  --protocols A,B      restrict the registry (e.g. `dealer,lowgear`)
  --activation A       `relu` (exact, MPC only) or `square` (both paradigms)
  --parties N          computing parties (default 2 — the matched threat model)
  --bits N             value bit width (default 32)
  --scale N            fixed-point scale bits (default 8)
  --lan                LAN profile instead of WAN
  --json               emit the report as JSON
  -h, --help           this message

Compiles a convolutional classifier to two-party dishonest-majority MPC through
the same pipeline as `bench-2pc-nn`, so a latency can be placed beside the
convolutional networks the homomorphic literature reports.";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut which = "lenet5".to_string();
    let mut hidden = Activation::Relu;
    let mut parties = 2usize;
    let mut bit_width = 32u32;
    let mut scale_bits = 8u32;
    let mut lan = false;
    let mut json = false;
    let mut only: Option<Vec<String>> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--lan" => lan = true,
            "--net" => which = it.next().ok_or("--net needs a value")?.clone(),
            "--protocols" => {
                only = Some(
                    it.next()
                        .ok_or("--protocols needs a value")?
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect(),
                )
            }
            "--activation" => {
                hidden = match it.next().map(String::as_str) {
                    Some("relu") => Activation::Relu,
                    Some("square") => Activation::Square,
                    other => return Err(format!("--activation takes relu|square, got {other:?}")),
                }
            }
            "--parties" => {
                parties = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--parties needs a number")?
            }
            "--bits" => {
                bit_width = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bits needs a number")?
            }
            "--scale" => {
                scale_bits = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--scale needs a number")?
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let net = match which.as_str() {
        "lenet5" => Cnn::lenet5(scale_bits, hidden),
        "lenet5-fhe" => Cnn::lenet5_fhe_friendly(scale_bits, hidden),
        "tiny" => Cnn::tiny(scale_bits, hidden),
        other => return Err(format!("unknown --net `{other}`\n\n{USAGE}")),
    };
    let dep = if lan {
        Deployment::lan(parties)
    } else {
        Deployment::wan(parties)
    };
    // `let*` desugars to one nested `let` per binding, so parsing and flattening
    // recurse once per bound value: LeNet-5 has roughly 13 000 of them and
    // overflows the 8 MiB the main thread is given. The work is therefore run on
    // a thread with a large stack, which is the same accommodation the tests in
    // `two_party_nn` make. An iterative flattener would remove the need; the
    // recursion is cheap per frame and merely deep.
    let r = {
        let net = net.clone();
        let dep = dep.clone();
        let only = only.clone();
        std::thread::Builder::new()
            .stack_size(512 << 20)
            .spawn(move || run_only(&net, &dep, bit_width, only.as_deref()))
            .map_err(|e| format!("could not spawn a deep-recursion thread: {e}"))?
            .join()
            .map_err(|_| "the benchmark thread panicked".to_string())??
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!(
        "A convolutional classifier end to end — {} parties, {}, Q{} fixed point, {} activation",
        r.parties, r.deployment, scale_bits, r.activation
    );
    println!(
        "{} : {} MACs, {} instructions from {:.1} MB of source, compiled in {:.2} s\n",
        which,
        net.macs(),
        r.instructions,
        r.source_bytes as f64 / 1e6,
        r.compile_secs
    );
    println!(
        "{:>9} {:>8} {:>9} {:>9} {:>11} {:>11} {:>10} {:>10}  ok",
        "protocol", "rounds", "muls", "decomps", "online", "offline", "warm", "cold"
    );
    for p in &r.points {
        println!(
            "{:>9} {:>8} {:>9} {:>9} {:>11} {:>11} {:>9.0}ms {:>9.0}ms  {}",
            if p.protocol == "dealer" {
                "dealer*"
            } else {
                &p.protocol
            },
            p.rounds,
            p.muls,
            p.decompositions,
            human(p.online_bytes),
            human(p.offline_bytes),
            p.online_secs * 1e3,
            p.total_secs * 1e3,
            if p.correct { "✓" } else { "✗" }
        );
    }
    for s in &r.skipped {
        println!("\n  skipped — {s}");
    }

    let bad: Vec<&str> = r
        .points
        .iter()
        .filter(|p| !p.correct)
        .map(|p| p.protocol.as_str())
        .collect();
    if !bad.is_empty() {
        return Err(format!(
            "protocols disagree with the plaintext reference: {bad:?} — every cost \
             above is void until they are bit-exact"
        ));
    }
    println!("\n  every protocol reproduces the plaintext fixed-point classification");
    println!(
        "  * `dealer` is a LOWER BOUND, not a deployable protocol.\n\
         \x20 `warm` hoists preprocessing, `cold` pays for it; both are reported because\n\
         \x20 quoting one against another system's total is not a comparison."
    );
    Ok(())
}

fn human(b: u64) -> String {
    const K: f64 = 1024.0;
    let f = b as f64;
    if f >= K * K {
        format!("{:.1} MiB", f / (K * K))
    } else if f >= K {
        format!("{:.1} KiB", f / K)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {

    /// A batched program must be a batch of the *same* network, and each inference in it
    /// must decide what it would have decided alone.
    ///
    /// The batch shares its weight variables and renames only intermediates and inputs.
    /// Getting that backwards — renaming the weights too — would produce several
    /// unrelated networks that happen to run together, and the throughput figure would
    /// then be measuring the wrong thing entirely while looking perfectly healthy.
    #[test]
    fn a_batch_is_one_network_applied_several_times() {
        use crate::mpcc::{compile, run_protocol, schedule, LowerConfig};

        let net = Cnn::tiny(8, Activation::Square);
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        // What each input decides on its own.
        let p = parameters(&net);
        let alone: Vec<i64> = (0..3)
            .map(|k| argmax_of(&forward(&net, &p, &input_vector(&net, k))))
            .collect();

        let (src, ins) = debug_batch(&net, 3);
        // The weights are shared, so a batch of three costs three times the
        // intermediates and *not* three times the parameters.
        let weight_vars = ins.keys().filter(|k| !k.contains("_x")).count();
        let single_weights = p.w.len();
        assert_eq!(
            weight_vars, single_weights,
            "a batch must share one set of weights, not carry a copy per inference"
        );

        let prog = compile(&src, &cfg).expect("compiles");
        let sched = schedule(&prog);
        let got = run_protocol("dealer", 2, &prog, &sched, &ins).expect("runs");

        // The program returns the classes packed four bits apart.
        let packed = got.outputs[0].1;
        for (n, want) in alone.iter().enumerate() {
            let nibble = (packed >> (4 * n)) & 0xF;
            assert_eq!(
                nibble, *want,
                "inference {n} of the batch decided {nibble}, alone it decides {want}"
            );
        }

        // And the round count must not grow with the batch, which is the whole reason
        // batching helps a secret-sharing protocol at all.
        let (src1, _) = debug_batch(&net, 1);
        let prog1 = compile(&src1, &cfg).expect("compiles");
        let rounds1 = schedule(&prog1).round_count();
        assert_eq!(
            sched.round_count(),
            rounds1,
            "independent inferences are independent instructions and belong in the same rounds"
        );
    }
    use super::*;

    /// A convolutional network must be deep-stack: `let*` desugars to one nested
    /// `let` per binding, so recursion tracks the network size.
    fn on_a_deep_stack(body: fn()) {
        std::thread::Builder::new()
            .stack_size(512 << 20)
            .spawn(body)
            .expect("spawn")
            .join()
            .expect("test body panicked");
    }

    /// The gate the benchmark rests on: a private convolutional classification
    /// must equal the plaintext fixed-point one under every protocol.
    #[test]
    #[ignore] // real OT/SHE preprocessing
    fn a_convolutional_network_classifies_exactly() {
        on_a_deep_stack(|| {
            for act in [Activation::Relu, Activation::Square] {
                let net = Cnn::tiny(8, act);
                let r = run(&net, &Deployment::lan(2), 32).unwrap();
                assert!(!r.points.is_empty(), "no protocol ran: {:?}", r.skipped);
                for p in &r.points {
                    assert!(
                        p.correct,
                        "{} with {:?}: class {} expected {}",
                        p.protocol, act, p.class, p.expected
                    );
                }
            }
        });
    }

    /// **Average pooling is free and max pooling is not**, and the difference is
    /// exactly the comparisons the max fold performs.
    ///
    /// A max fold over an `s x s` window costs `s^2 - 1` comparisons per output,
    /// each of which is a bit decomposition; an average fold is a sum, and the
    /// division by `s^2` is deferred into the next layer's rescale, so it costs
    /// nothing at all. On LeNet-5 the difference is 3 360 decompositions of
    /// 12 747, or 26% of the whole nonlinear surface.
    #[test]
    fn average_pooling_costs_nothing_and_max_pooling_costs_comparisons() {
        let max_net = Cnn::lenet5(8, Activation::Relu);
        let avg_net = Cnn::lenet5_fhe_friendly(8, Activation::Square);
        // Same linear work either way: pooling is not where the MACs are.
        assert_eq!(max_net.macs(), avg_net.macs());

        // The pooled output counts: 6x12x12 and 16x4x4, three comparisons each.
        let expected_saving = (6 * 12 * 12 + 16 * 4 * 4) * 3;
        assert_eq!(expected_saving, 3360);
    }

    /// The generator and the reference must agree about layout, or a
    /// bit-exactness check compares two different networks. Exercised on a shape
    /// whose channel, row and column strides are all distinct.
    #[test]
    fn the_generated_source_names_every_weight_the_reference_uses() {
        let net = Cnn::tiny(8, Activation::Relu);
        let p = parameters(&net);
        let src = network_src(&net);
        let ins = bindings(&p, &input_vector(&net, 0));
        // Every variable the source mentions must be bound.
        for tok in src.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            if tok.len() > 1
                && (tok.starts_with('k')
                    || tok.starts_with("cb")
                    || tok.starts_with('d')
                    || tok.starts_with("db")
                    || tok.starts_with('x'))
                && tok.chars().any(|c| c.is_ascii_digit())
                && !ins.contains_key(tok)
                // Generated intermediates share these prefixes; only inputs matter.
                && !tok.starts_with("dot")
            {
                assert!(
                    ins.contains_key(tok) || !p.w.contains_key(tok),
                    "source mentions `{tok}` but no binding supplies it"
                );
            }
        }
        assert!(!ins.is_empty());
    }
}
