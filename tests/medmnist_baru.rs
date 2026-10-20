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
    run_file("examples/medmnist.baru", dataset, n, private)
}

fn run_file(path: &str, dataset: &str, n: usize, private: usize) -> (i64, i64) {
    let src = std::fs::read_to_string(path).expect("the Baru program must exist");
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

/// Changing the nonlinearity protocol must not change a single diagnosis.
///
/// This is the property the whole `--nonlin` switch rests on: `a2b`, the
/// 3-party garbled route and Falcon compute the *same function* by different
/// protocols, and the only reason to prefer one is cost. If they disagreed, the
/// fast run would be fast and wrong — and on one image the difference is a
/// coin flip that would pass unnoticed, which is why this compares the routes
/// over several images rather than one.
///
/// Runs on the real 3-party transport (the garbled and Falcon routes do not
/// exist on the in-process simulator), so it is `#[ignore]`d with the other
/// pipeline tests.
#[test]
#[ignore = "three 3-party MPC runs of a real CNN; a few minutes"]
fn every_nonlinearity_protocol_gives_the_same_diagnosis() {
    let n = 3usize;
    let src = std::fs::read_to_string("examples/medmnist.baru").expect("program");
    let preamble = format!(
        "(define DATASET \"pneumonia\")\n(define NIMAGES {n})\n(define PRIVATE 1)\n(define NBITS 64)\n"
    );
    let prog = baru::runner::parse_program(&format!("{preamble}{src}")).expect("parses");
    baru::interpreter::eval::set_max_frames(64_000_000_000);

    // The plaintext answer these must all reproduce.
    let t = load("pneumonia").expect("dataset embedded");
    let model = t.model();
    let expect: i64 = (0..n)
        .filter(|&i| model.forward(&t.images[i]).0 == t.labels[i] as usize)
        .count() as i64;
    let want = format!("({expect} {n})");

    for mode in ["a2b", "gc", "falcon"] {
        let run = baru::runner::run_mpc(&prog, mode, 0, None)
            .unwrap_or_else(|e| panic!("--nonlin {mode} failed: {e}"));
        assert_eq!(
            run.result, want,
            "--nonlin {mode} disagreed with the plaintext model"
        );
    }
}

/// An unknown protocol name must be refused, not silently ignored — running
/// `a2b` while the operator believes they asked for garbled circuits would
/// misattribute every timing taken afterwards.
#[test]
fn an_unknown_nonlinearity_is_refused() {
    let prog = baru::runner::parse_program("(reconstruct (share 7 32))").expect("parses");
    let err = baru::runner::run_mpc(&prog, "no-such-protocol", 0, None)
        .expect_err("must refuse an unknown protocol");
    assert!(err.contains("no-such-protocol"), "unhelpful error: {err}");
}

/// `examples/medmnist_scalar.baru` exists only to reproduce the first row of
/// `docs/experiments.md` IV.2b — the cost of writing the same network one
/// activation at a time. A measurement whose source has drifted from the one it
/// is compared against is worse than no measurement, so the two files must agree
/// on every diagnosis.
#[test]
fn the_scalar_version_computes_the_same_diagnoses_as_the_vector_one() {
    let n = 8usize;
    let vector = run_baru("pneumonia", n, 0);
    let scalar = run_file("examples/medmnist_scalar.baru", "pneumonia", n, 0);
    assert_eq!(
        scalar, vector,
        "the scalar and vector programs disagree — the IV.2b comparison is between \
         two different networks and must not be quoted"
    );
    // And both against the Rust reference, so neither can drift together.
    let t = load("pneumonia").expect("dataset embedded");
    let model = t.model();
    let expect: i64 = (0..n)
        .filter(|&i| model.forward(&t.images[i]).0 == t.labels[i] as usize)
        .count() as i64;
    assert_eq!(vector.0, expect, "vector program vs CnnModel::forward");
}
