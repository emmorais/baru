// The depth sweep, secret-sharing arm: elapsed wide-area time against network depth.
//
// The comparison against an approximate homomorphic scheme has to be run along depth,
// because that is the axis on which the two paradigms pay in incomparable ways. Secret
// sharing has no depth budget — each layer adds a fixed number of rounds and a fixed
// amount of traffic — so this curve should be a line. The homomorphic curve is a
// staircase: the ring dimension must grow with the level budget to hold the security
// level, and past the budget the scheme must bootstrap, measured at 110.7 s against
// 0.130 s for a multiplication.
use baru::bench::comm::{LatencyTransport, NetworkProfile};
use baru::bench::conv_nn::{debug_case, Cnn};
use baru::crypto::net::InprocHandle;
use baru::ml::Activation;
use baru::mpcc::dist::RunConfig;
use baru::mpcc::{compile, dist, schedule, LowerConfig};
use std::sync::Mutex;
use std::time::Duration;

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
        "{:>6} {:>6} {:>8} {:>6} {:>10} {:>9} {:>9}  class",
        "extra", "levels", "instrs", "exch", "bytes/MiB", "compute", "wan"
    );
    for extra in 0..=4usize {
        let net = Cnn::lenet5_deep(8, Activation::Square, extra);
        let (src, ins) = debug_case(&net, 0);
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let prog = compile(&src, &cfg).expect("compile");
        let sched = schedule(&prog);
        let rc = RunConfig {
            bits: 64,
            seed: 0xBADC0DE,
            hoist_bits: true,
            trunc_pairs: true,
            bilinear: false,
        };
        let run = |rtt: u64, mbps: f64| {
            let profile = NetworkProfile::fixed_latency(Duration::from_millis(rtt))
                .with_bandwidth(mbps * 1e6 / 8.0);
            let hs = Mutex::new(InprocHandle::create(2));
            dist::run_two_party(&prog, &sched, &ins, rc, |i| {
                LatencyTransport::new(hs.lock().unwrap()[i].clone(), profile, 2)
            })
            .expect("run")
        };
        let local = run(0, 1e12);
        let wan = run(25, 100.0);
        // Levels, counted the way the homomorphic arm counts them, so the sweep is
        // over one variable on both sides.
        let levels = 11 + 2 * extra;
        println!(
            "{extra:>6} {levels:>6} {:>8} {:>6} {:>10.2} {:>8.0}ms {:>8.0}ms  {:?}",
            prog.code.len(),
            wan.exchanges,
            wan.online_bytes as f64 / (1024.0 * 1024.0),
            local.elapsed_secs * 1e3,
            wan.elapsed_secs * 1e3,
            wan.outputs
        );
    }
}
