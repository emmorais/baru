// What opening each operand once is worth, on the convolution shapes LeNet-5 actually
// contains. See `mpcc::vm::bilinear_batch`.
use baru::bench::comm::{LatencyTransport, NetworkProfile};
use baru::crypto::net::InprocHandle;
use baru::mpcc::dist::DistEngine;
use baru::mpcc::vm::{bilinear_batch, ShareEngine};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The `(input, weight)` pairs of a convolution, and its operand counts.
fn conv(
    h: usize,
    w: usize,
    k: usize,
    in_c: usize,
    out_c: usize,
) -> (Vec<Vec<(u32, u32)>>, usize, usize) {
    let (oh, ow) = (h - k + 1, w - k + 1);
    let n_in = in_c * h * w;
    let n_w = out_c * in_c * k * k;
    let mut terms = Vec::with_capacity(out_c * oh * ow);
    for o in 0..out_c {
        for y in 0..oh {
            for x in 0..ow {
                let mut t = Vec::with_capacity(in_c * k * k);
                for c in 0..in_c {
                    for a in 0..k {
                        for b in 0..k {
                            t.push((
                                ((c * h + y + a) * w + (x + b)) as u32,
                                (((o * in_c + c) * k + a) * k + b) as u32,
                            ));
                        }
                    }
                }
                terms.push(t);
            }
        }
    }
    (terms, n_in, n_w)
}

fn run(terms: &[Vec<(u32, u32)>], n_in: usize, n_w: usize, bilinear: bool, rtt: u64) -> (f64, u64) {
    let profile =
        NetworkProfile::fixed_latency(Duration::from_millis(rtt)).with_bandwidth(100.0 * 1e6 / 8.0);
    let hs = Mutex::new(InprocHandle::create(2));
    let nets: Vec<_> = (0..2)
        .map(|i| {
            Arc::new(LatencyTransport::new(
                hs.lock().unwrap()[i].clone(),
                profile,
                2,
            ))
        })
        .collect();
    let mut joins = Vec::new();
    for (me, net) in nets.into_iter().enumerate() {
        let terms = terms.to_vec();
        joins.push(std::thread::spawn(move || {
            let mut e = DistEngine::new(me, net, 64, 0xB1)
                .unwrap()
                .with_bilinear(bilinear);
            let xs = e.input_batch(
                &(0..n_in)
                    .map(|i| ((i % 13) as u64, 0usize))
                    .collect::<Vec<_>>(),
            );
            let ys = e.input_batch(
                &(0..n_w)
                    .map(|i| ((i % 7) as u64, 1usize))
                    .collect::<Vec<_>>(),
            );
            let t0 = std::time::Instant::now();
            let out = bilinear_batch(&mut e, &xs, &ys, &terms);
            let _ = e.open_batch(&out);
            (t0.elapsed().as_secs_f64(), e.counters().online_bytes)
        }));
    }
    let mut best = (0.0f64, 0u64);
    for j in joins {
        let r = j.join().unwrap();
        if r.1 > best.1 {
            best = r;
        }
    }
    best
}

fn lenet() {
    use baru::bench::conv_nn::{debug_case, Cnn};
    use baru::ml::Activation;
    use baru::mpcc::dist::RunConfig;
    use baru::mpcc::{compile, dist, schedule, LowerConfig};

    let net = Cnn::lenet5_fhe_friendly(8, Activation::Square);
    let (src, ins) = debug_case(&net, 0);
    let cfg = LowerConfig {
        parties: 2,
        bit_width: 32,
        ..Default::default()
    };
    let prog = compile(&src, &cfg).expect("compile");
    let sched = schedule(&prog);
    println!(
        "\n{:>10} {:>6} {:>10} {:>9} {:>10}  class",
        "fusion", "exch", "bytes/MiB", "compute", "wan"
    );
    for bilinear in [false, true] {
        let go = |rtt: u64, mbps: f64| {
            let profile = NetworkProfile::fixed_latency(Duration::from_millis(rtt))
                .with_bandwidth(mbps * 1e6 / 8.0);
            let hs = Mutex::new(InprocHandle::create(2));
            dist::run_two_party(
                &prog,
                &sched,
                &ins,
                RunConfig {
                    bits: 64,
                    seed: 0xBADC0DE,
                    hoist_bits: true,
                    trunc_pairs: true,
                    bilinear,
                },
                |i| LatencyTransport::new(hs.lock().unwrap()[i].clone(), profile, 2),
            )
            .expect("run")
        };
        let local = go(0, 1e12);
        let wan = go(25, 100.0);
        println!(
            "{:>10} {:>6} {:>10.2} {:>8.0}ms {:>9.0}ms  {:?}",
            if bilinear { "bilinear" } else { "beaver" },
            wan.exchanges,
            wan.online_bytes as f64 / (1024.0 * 1024.0),
            local.elapsed_secs * 1e3,
            wan.elapsed_secs * 1e3,
            wan.outputs
        );
    }
}

fn main() {
    std::thread::Builder::new()
        .stack_size(1024 << 20)
        .spawn(body)
        .expect("spawn")
        .join()
        .expect("probe");
}

fn body() {
    println!(
        "{:>28} {:>9} {:>9} {:>12} {:>12} {:>10} {:>10}",
        "layer", "products", "operands", "scalar KiB", "bilin KiB", "scalar", "bilinear"
    );
    for (name, h, w, k, ic, oc) in [
        ("conv0  28x28 k5 1->6", 28, 28, 5, 1, 6),
        ("conv3  12x12 k5 6->16", 12, 12, 5, 6, 16),
    ] {
        let (terms, n_in, n_w) = conv(h, w, k, ic, oc);
        let prods: usize = terms.iter().map(|t| t.len()).sum();
        let (ts, bs) = run(&terms, n_in, n_w, false, 25);
        let (tb, bb) = run(&terms, n_in, n_w, true, 25);
        println!(
            "{name:>28} {prods:>9} {:>9} {:>12.1} {:>12.1} {:>9.0}ms {:>9.0}ms",
            n_in + n_w,
            bs as f64 / 1024.0,
            bb as f64 / 1024.0,
            ts * 1e3,
            tb * 1e3
        );
    }
    lenet();
}
