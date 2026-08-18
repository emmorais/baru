// Throughput, both arms. See `bench::conv_nn::debug_batch` for why batching is the
// measurement on which the two paradigms differ most.
use baru::bench::comm::{LatencyTransport, NetworkProfile};
use baru::bench::conv_nn::{debug_batch, Cnn};
use baru::crypto::net::InprocHandle;
use baru::ml::Activation;
use baru::mpcc::dist::RunConfig;
use baru::mpcc::{compile, dist, schedule, LowerConfig};
use std::sync::Mutex;
use std::time::Duration;

fn main() {
    std::thread::Builder::new()
        .stack_size(2048 << 20)
        .spawn(body)
        .expect("spawn")
        .join()
        .expect("probe");
}

fn body() {
    println!(
        "{:>8} {:>6} {:>9} {:>6} {:>9} {:>10} {:>12} {:>10}",
        "fusion", "batch", "instrs", "exch", "MiB", "wan", "per-inference", "inf/s"
    );
    let net = Cnn::lenet5_fhe_friendly(8, Activation::Square);
    for (bilinear, batch) in [false, true]
        .into_iter()
        .flat_map(|b| [1usize, 2, 4, 8].into_iter().map(move |n| (b, n)))
    {
        let (src, ins) = debug_batch(&net, batch);
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let prog = match compile(&src, &cfg) {
            Ok(p) => p,
            Err(e) => {
                println!("{batch:>6}  compile refused: {e}");
                let _ = bilinear;
                continue;
            }
        };
        let sched = schedule(&prog);
        let profile = NetworkProfile::fixed_latency(Duration::from_millis(25))
            .with_bandwidth(100.0 * 1e6 / 8.0);
        let hs = Mutex::new(InprocHandle::create(2));
        let m = dist::run_two_party(
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
        .expect("run");
        println!(
            "{:>8} {batch:>6} {:>9} {:>6} {:>9.2} {:>9.0}ms {:>11.0}ms {:>10.2}",
            if bilinear { "bilinear" } else { "beaver" },
            prog.code.len(),
            m.exchanges,
            m.online_bytes as f64 / (1024.0 * 1024.0),
            m.elapsed_secs * 1e3,
            m.elapsed_secs * 1e3 / batch as f64,
            batch as f64 / m.elapsed_secs
        );
    }
}
