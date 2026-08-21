//! Dataset builtins: the trained MedMNIST weights and test images, exposed to
//! Baru as **ordinary Lisp data**.
//!
//! # Why these are data builtins and not a `medmnist-forward` builtin
//!
//! The point of `examples/medmnist.baru` is that the network is written *in
//! Baru* — the convolution, the fused ReLU∘rescale, the max-pool, the dense
//! layers and the argmax are Lisp, so the same source text runs on the plain
//! interpreter and on the MPC backend and the comparison means something. A
//! builtin that evaluated the CNN would move all of that back into Rust and
//! make the Lisp file a wrapper around [`crate::bench::medmnist`], which is
//! exactly the thing it exists to stop being.
//!
//! So the boundary is drawn at *loading*: these builtins hand over the
//! quantised weights and the embedded test images and do no arithmetic at all.
//! That is the same boundary [`super::eval`]'s `nn-load-model` already draws —
//! a quantised model arrives "as ordinary Lisp data, inspectable like any other
//! value rather than an opaque handle".
//!
//! # The model layout
//!
//! `(medmnist-model name)` returns a nine-element list. It is positional rather
//! than keyed because this language has no `assoc`, and `examples/medmnist.baru`
//! defines named accessors over it so no call site indexes it by hand:
//!
//! | index | field | shape |
//! |---|---|---|
//! | 0 | `d` | input side length (28) |
//! | 1 | `in-c` | input channels (1 grayscale, 3 RGB) |
//! | 2 | `shift` | the quantisation shift the fused ReLU∘rescale applies |
//! | 3 | `nclasses` | output width |
//! | 4 | conv1 | `(kernels cbias k)`, `kernels[oc]` flat over `(in_c, ky, kx)` |
//! | 5 | extra convs | list of `(kernels cbias in-c in-h in-w k)` |
//! | 6 | hidden dense | list of `(weights biases)` |
//! | 7 | `dense-w` | output layer, `[nclasses][in]` |
//! | 8 | `dense-b` | output layer bias |
//!
//! Every integer is exactly the `i64` the Rust pipeline multiplies, so a
//! disagreement between this path and `bench-medmnist` is a bug in one of the
//! two implementations rather than a difference of inputs. That is the whole
//! reason to share the weight files instead of re-exporting them.

use super::ast::ErrObj;
use super::ast::Types;
use crate::bench::medmnist::{load, TrainedCnn};

/// `(list ...)` of `Num`s from a slice of `i64`.
fn nums(xs: &[i64]) -> Types {
    Types::List(xs.iter().map(|&v| Types::Num(v)).collect())
}

/// A list of rows, each itself a list of `Num`s.
fn rows(xs: &[Vec<i64>]) -> Types {
    Types::List(xs.iter().map(|r| nums(r)).collect())
}

/// The nine-element model list documented at the module level.
fn model_to_lisp(m: &TrainedCnn) -> Types {
    let conv1 = Types::List(vec![
        rows(&m.kernels),
        nums(&m.cbias),
        // The first conv's kernel side, derived from the weight row rather than
        // stored: a row is `in_c · k · k` long.
        Types::Num(kernel_side(m.kernels[0].len(), m.in_c()) as i64),
    ]);
    let extra = Types::List(
        m.extra_conv
            .iter()
            .map(|c| {
                Types::List(vec![
                    rows(&c.0),
                    nums(&c.1),
                    Types::Num(c.2 as i64),
                    Types::Num(c.3 as i64),
                    Types::Num(c.4 as i64),
                    Types::Num(kernel_side(c.0[0].len(), c.2) as i64),
                ])
            })
            .collect(),
    );
    let hidden = Types::List(
        m.hidden
            .iter()
            .map(|(w, b)| Types::List(vec![rows(w), nums(b)]))
            .collect(),
    );
    Types::List(vec![
        Types::Num(m.d as i64),
        Types::Num(m.in_c() as i64),
        Types::Num(m.shift as i64),
        Types::Num(m.dense_b.len() as i64),
        conv1,
        extra,
        hidden,
        rows(&m.dense_w),
        nums(&m.dense_b),
    ])
}

/// `k` from a flat kernel row of length `in_c · k · k`.
///
/// Returns 0 rather than panicking on a non-square row so a malformed weight
/// file surfaces as a Lisp-visible 0 kernel side (which the `.baru` guard
/// rejects) instead of aborting the interpreter.
fn kernel_side(row_len: usize, in_c: usize) -> usize {
    if in_c == 0 || !row_len.is_multiple_of(in_c) {
        return 0;
    }
    let per_channel = row_len / in_c;
    let k = (per_channel as f64).sqrt().round() as usize;
    if k * k == per_channel {
        k
    } else {
        0
    }
}

/// Register `medmnist-model`, `medmnist-image` and `medmnist-count`.
///
/// Returns the bindings rather than inserting them, so the caller keeps the
/// single `self.store` mutation site it already has for every other builtin.
pub fn dataset_builtins() -> Vec<(String, Types)> {
    let model_fn = std::sync::Arc::new(move |args: Vec<Types>| -> Types {
        let name = match args.as_slice() {
            [Types::Str(s)] => s.clone(),
            _ => return ErrObj::raise("medmnist-model expects 1 arg: (medmnist-model \"name\")"),
        };
        match load(&name) {
            Ok(m) => model_to_lisp(&m),
            Err(e) => ErrObj::raise(e),
        }
    });

    let image_fn = std::sync::Arc::new(move |args: Vec<Types>| -> Types {
        let (name, idx) = match args.as_slice() {
            [Types::Str(s), Types::Num(i)] => (s.clone(), *i),
            _ => {
                return ErrObj::raise("medmnist-image expects 2 args: (medmnist-image \"name\" i)")
            }
        };
        let m = match load(&name) {
            Ok(m) => m,
            Err(e) => return ErrObj::raise(e),
        };
        if idx < 0 || idx as usize >= m.images.len() {
            return ErrObj::raise(format!(
                "medmnist-image: index {idx} out of range (dataset {name:?} embeds {} images)",
                m.images.len()
            ));
        }
        let i = idx as usize;
        Types::List(vec![nums(&m.images[i]), Types::Num(m.labels[i] as i64)])
    });

    let count_fn = std::sync::Arc::new(move |args: Vec<Types>| -> Types {
        let name = match args.as_slice() {
            [Types::Str(s)] => s.clone(),
            _ => return ErrObj::raise("medmnist-count expects 1 arg: (medmnist-count \"name\")"),
        };
        match load(&name) {
            Ok(m) => Types::Num(m.images.len() as i64),
            Err(e) => ErrObj::raise(e),
        }
    });

    vec![
        ("medmnist-model".to_string(), Types::NativeFunc(model_fn)),
        ("medmnist-image".to_string(), Types::NativeFunc(image_fn)),
        ("medmnist-count".to_string(), Types::NativeFunc(count_fn)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_list(t: &Types) -> &Vec<Types> {
        match t {
            Types::List(v) => v,
            other => panic!("expected list, got {other:?}"),
        }
    }
    fn as_num(t: &Types) -> i64 {
        match t {
            Types::Num(n) => *n,
            other => panic!("expected num, got {other:?}"),
        }
    }

    /// The Lisp view must carry the *same integers* the Rust pipeline
    /// multiplies. This is the guard that `examples/medmnist.baru` and
    /// `bench-medmnist` are two implementations of one model rather than two
    /// models: if they ever disagree on a diagnosis, this test says the inputs
    /// were not the difference.
    #[test]
    fn lisp_model_carries_the_same_weights_as_the_rust_pipeline() {
        for name in ["pneumonia", "organa", "derma"] {
            let m = load(name).expect("dataset must be embedded");
            let lisp = model_to_lisp(&m);
            let top = as_list(&lisp);
            assert_eq!(top.len(), 9, "{name}: model layout is nine fields");
            assert_eq!(as_num(&top[0]), m.d as i64, "{name}: d");
            assert_eq!(as_num(&top[1]), m.in_c() as i64, "{name}: in_c");
            assert_eq!(as_num(&top[2]), m.shift as i64, "{name}: shift");
            assert_eq!(as_num(&top[3]), m.dense_b.len() as i64, "{name}: nclasses");

            // conv1 kernels, element by element
            let conv1 = as_list(&top[4]);
            let kern = as_list(&conv1[0]);
            assert_eq!(kern.len(), m.kernels.len(), "{name}: conv1 out_c");
            for (oc, row) in m.kernels.iter().enumerate() {
                let lrow = as_list(&kern[oc]);
                assert_eq!(lrow.len(), row.len(), "{name}: conv1 row {oc} length");
                for (i, &w) in row.iter().enumerate() {
                    assert_eq!(as_num(&lrow[i]), w, "{name}: conv1[{oc}][{i}]");
                }
            }
            // the kernel side must be recoverable from the row length, not guessed
            let k = as_num(&conv1[2]);
            assert!(k > 0, "{name}: conv1 k must be square");
            assert_eq!(
                (k * k) as usize * m.in_c(),
                m.kernels[0].len(),
                "{name}: conv1 k must reproduce the row length"
            );

            // the output layer, which is what the argmax reads
            let dw = as_list(&top[7]);
            assert_eq!(dw.len(), m.dense_w.len(), "{name}: dense rows");
            for (r, row) in m.dense_w.iter().enumerate() {
                let lrow = as_list(&dw[r]);
                for (i, &w) in row.iter().enumerate() {
                    assert_eq!(as_num(&lrow[i]), w, "{name}: dense_w[{r}][{i}]");
                }
            }
            let db = as_list(&top[8]);
            for (i, &b) in m.dense_b.iter().enumerate() {
                assert_eq!(as_num(&db[i]), b, "{name}: dense_b[{i}]");
            }
        }
    }

    /// Every embedded image must arrive at full length with its true label, or
    /// the Lisp accuracy figure would be measured against the wrong answers.
    #[test]
    fn lisp_images_match_the_embedded_test_subset() {
        for name in ["pneumonia", "organa", "derma"] {
            let m = load(name).expect("dataset must be embedded");
            let bs = dataset_builtins();
            let image = match &bs[1].1 {
                Types::NativeFunc(f) => f.clone(),
                _ => panic!("medmnist-image must be a native func"),
            };
            let expect_len = m.in_c() * m.d * m.d;
            for i in [0usize, 1, m.images.len() - 1] {
                let got = image(vec![Types::Str(name.to_string()), Types::Num(i as i64)]);
                let pair = as_list(&got);
                let px = as_list(&pair[0]);
                assert_eq!(px.len(), expect_len, "{name}: image {i} length");
                for (j, &v) in m.images[i].iter().enumerate() {
                    assert_eq!(as_num(&px[j]), v, "{name}: image {i} pixel {j}");
                }
                assert_eq!(as_num(&pair[1]), m.labels[i] as i64, "{name}: label {i}");
            }
        }
    }

    /// An out-of-range index must be an error, not a silent wrap onto another
    /// patient's scan.
    #[test]
    fn image_index_out_of_range_is_an_error() {
        let bs = dataset_builtins();
        let image = match &bs[1].1 {
            Types::NativeFunc(f) => f.clone(),
            _ => panic!(),
        };
        let n = load("pneumonia").unwrap().images.len();
        let got = image(vec![
            Types::Str("pneumonia".to_string()),
            Types::Num(n as i64),
        ]);
        assert!(matches!(got, Types::Error(_, _)), "got {got:?}");
    }

    #[test]
    fn unknown_dataset_is_an_error() {
        let bs = dataset_builtins();
        let model = match &bs[0].1 {
            Types::NativeFunc(f) => f.clone(),
            _ => panic!(),
        };
        let got = model(vec![Types::Str("no-such-set".to_string())]);
        assert!(matches!(got, Types::Error(_, _)), "got {got:?}");
    }
}
