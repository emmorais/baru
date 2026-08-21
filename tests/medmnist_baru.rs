//! `examples/medmnist.baru` must agree with [`baru::bench::cnn::CnnModel`] on
//! every diagnosis.
//!
//! There are now two independent implementations of the same quantised CNN: the
//! Rust one `bench-medmnist` drives, and the Baru one in
//! `examples/medmnist.baru`. Two implementations of one model are only useful if
//! they are actually the same model — otherwise "Baru plaintext vs Baru private
//! vs a reference language" compares three different networks and the timings
//! mean nothing. These tests are that guard.
//!
//! The Baru program is the *specification under test*; `CnnModel::forward` is
//! the reference, because it is the one the encrypted pipeline is already
//! bit-exact against (`bench-medmnist` checks encrypted == plaintext on every
//! embedded image). Chaining the two makes the Baru path exact against the
//! encrypted pipeline too, transitively.

use baru::bench::medmnist::load;
use baru::interpreter::{Participant, Types};
use baru::runner::parse_program;

/// Evaluate `examples/medmnist.baru` on the plain interpreter for the first
/// `n` images of `dataset`, returning `(correct, n)` as the program reports it.
fn run_baru(dataset: &str, n: usize, private: usize) -> (i64, i64) {
    let src =
        std::fs::read_to_string("examples/medmnist.baru").expect("the Baru program must exist");
    let preamble = format!(
        "(define DATASET \"{dataset}\")\n(define NIMAGES {n})\n(define PRIVATE {private})\n(define NBITS 64)\n"
    );
    let prog = parse_program(&format!("{preamble}{src}")).expect("the Baru program must parse");
    // Same reason `baru run` does it: the REPL's step budget is sized for a
    // one-line expression, and a convolution is not one.
    baru::interpreter::eval::set_max_frames(64_000_000_000);
    let p = Participant::new_simulated(0);
    let mut last = Types::Num(0);
    for e in &prog {
        last = p.eval_quiet(e);
        if let Types::Error(..) = last {
            panic!("Baru program raised: {last:?}");
        }
    }
    match last {
        Types::List(xs) => match (&xs[0], &xs[1]) {
            (Types::Num(c), Types::Num(t)) => (*c, *t),
            other => panic!("unexpected result shape {other:?}"),
        },
        other => panic!("expected (correct total), got {other:?}"),
    }
}

/// The Baru program must reach exactly the accuracy the Rust model reaches on
/// the same images — not merely a similar one. A single differing prediction
/// means the two implementations disagree, and this is the only place that would
/// be caught before it reached a results table.
#[test]
fn baru_program_agrees_with_the_rust_model_on_pneumonia() {
    let n = 12usize;
    let t = load("pneumonia").expect("dataset embedded");
    let model = t.model();
    let expect: i64 = (0..n)
        .filter(|&i| model.forward(&t.images[i]).0 == t.labels[i] as usize)
        .count() as i64;
    let (correct, total) = run_baru("pneumonia", n, 0);
    assert_eq!(
        total, n as i64,
        "the program must classify every image asked"
    );
    assert_eq!(
        correct, expect,
        "Baru and CnnModel::forward disagree on the first {n} pneumonia images"
    );
}

/// The RGB set, because three input channels are the case where the window
/// flattening order — (ic, ky, kx), matching the kernel row layout — could be
/// wrong without any grayscale test noticing.
#[test]
fn baru_program_agrees_with_the_rust_model_on_rgb_derma() {
    let n = 8usize;
    let t = load("derma").expect("dataset embedded");
    let model = t.model();
    let expect: i64 = (0..n)
        .filter(|&i| model.forward(&t.images[i]).0 == t.labels[i] as usize)
        .count() as i64;
    let (correct, total) = run_baru("derma", n, 0);
    assert_eq!(total, n as i64);
    assert_eq!(
        correct, expect,
        "Baru and CnnModel::forward disagree on the first {n} derma images"
    );
}

/// The wider organa model exercises a second conv with 16 output channels and
/// an 11→5 pool, which is the floor-division case `/` refuses and `ash-right`
/// handles.
#[test]
fn baru_program_agrees_with_the_rust_model_on_organa() {
    let n = 6usize;
    let t = load("organa").expect("dataset embedded");
    let model = t.model();
    let expect: i64 = (0..n)
        .filter(|&i| model.forward(&t.images[i]).0 == t.labels[i] as usize)
        .count() as i64;
    let (correct, total) = run_baru("organa", n, 0);
    assert_eq!(total, n as i64);
    assert_eq!(
        correct, expect,
        "Baru and CnnModel::forward disagree on the first {n} organa images"
    );
}

/// Secret-sharing the input must not change a single diagnosis: the oblivious
/// forms (`mux`, `max2`, `relu-rescale`) are supposed to be exact, not
/// approximate. Slow — this is the MPC simulator on a real CNN — so it is
/// `#[ignore]`d, like the other pipeline exactness tests in this repo.
#[test]
#[ignore = "runs the CNN under the MPC simulator; minutes, and memory-hungry"]
fn secret_sharing_the_image_changes_no_diagnosis() {
    // One image, and run this test in isolation (`--test-threads=1`, on its own
    // invocation). The ABY3 simulator keeps every intermediate share alive for
    // the whole top-level expression — `reset_per_expression` is the only thing
    // that frees them — so memory grows with images-per-expression, and two of
    // these tests in one process is enough to be OOM-killed.
    let n = 1usize;
    let plain = run_baru("pneumonia", n, 0);
    let private = run_baru("pneumonia", n, 1);
    assert_eq!(
        plain, private,
        "PRIVATE=1 disagreed with plaintext — the oblivious rewrite is not exact"
    );
}

/// And sharing the *weights* as well must not change it either, which is the
/// configuration that matches what `bench-medmnist` measures.
#[test]
#[ignore = "runs the CNN under the MPC simulator with shared weights; slowest of the three"]
fn secret_sharing_the_model_changes_no_diagnosis() {
    let n = 1usize;
    let plain = run_baru("pneumonia", n, 0);
    let private = run_baru("pneumonia", n, 2);
    assert_eq!(
        plain, private,
        "PRIVATE=2 disagreed with plaintext — sharing the weights changed a diagnosis"
    );
}
