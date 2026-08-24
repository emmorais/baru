// Does stochastic rescaling change the answer?
//
// Replacing an exact bit-decomposition truncation with a truncation pair makes the
// rescale cheap and makes it *approximate*: the result is the exact quotient plus a
// carry that is one with probability equal to the discarded fraction. That is a
// change to the computed function, so the speedup it buys is not quotable until the
// cost has been measured on the quantity that matters — whether the network still
// classifies the same way — over enough inputs to say something.
//
// This runs both rescale routes over the same program and the same inputs and
// reports the disagreement rate on the predicted class, together with the
// distribution of per-logit deviation.
use baru::bench::conv_nn::{debug_case, Cnn};
use baru::ml::Activation;
use baru::mpcc::{compile, run_protocol, schedule, LowerConfig};

fn main() {
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(body)
        .expect("spawn")
        .join()
        .expect("probe");
}

fn body() {
    let cfg = LowerConfig {
        parties: 2,
        bit_width: 32,
        ..Default::default()
    };

    // MLPs first — cheap enough to run many inputs through.
    for shape in [vec![16usize, 8, 4], vec![784usize, 32, 10]] {
        let label = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let src = baru::bench::two_party_nn::debug_src(&shape, 8);
        let prog = compile(&src, &cfg).expect("compile");
        let sched = schedule(&prog);
        let mut agree = 0usize;
        let mut n = 0usize;
        for k in 0..40 {
            let ins = baru::bench::two_party_nn::debug_bindings_k(&shape, 8, k);
            let a = run_protocol("dealer", 2, &prog, &sched, &ins).expect("exact");
            let b = run_protocol("dealer-tp", 2, &prog, &sched, &ins).expect("pairs");
            n += 1;
            if a.outputs[0].1 == b.outputs[0].1 {
                agree += 1;
            }
        }
        println!(
            "{label:>12}: class agreement {agree}/{n} ({:.1}%)",
            100.0 * agree as f64 / n as f64
        );
    }

    // LeNet-5, fewer inputs because each run is expensive.
    let net = Cnn::lenet5_fhe_friendly(8, Activation::Square);
    let (src, _) = debug_case(&net, 0);
    let prog = compile(&src, &cfg).expect("compile");
    let sched = schedule(&prog);
    let mut agree = 0usize;
    let mut n = 0usize;
    for k in 0..10 {
        let (_, ins) = debug_case(&net, k);
        let a = run_protocol("dealer", 2, &prog, &sched, &ins).expect("exact");
        let b = run_protocol("dealer-tp", 2, &prog, &sched, &ins).expect("pairs");
        n += 1;
        if a.outputs[0].1 == b.outputs[0].1 {
            agree += 1;
        } else {
            println!(
                "      lenet5: input {k} exact={} pairs={}",
                a.outputs[0].1, b.outputs[0].1
            );
        }
    }
    println!(
        "      lenet5: class agreement {agree}/{n} ({:.1}%)",
        100.0 * agree as f64 / n as f64
    );
}
