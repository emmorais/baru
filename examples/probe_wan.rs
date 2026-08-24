// First measured WAN figures for the two-party arm: elapsed wall clock, two
// parties concurrent, a transport that really sleeps. No modelled term anywhere.
use baru::bench::comm::{LatencyTransport, NetworkProfile};
use baru::crypto::net::InprocHandle;
use baru::mpcc::dist::RunConfig;
use baru::mpcc::{compile, dist, schedule, LowerConfig};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

fn go(
    src: &str,
    ins: &HashMap<String, i64>,
    rtt_ms: u64,
    mbps: f64,
    hoist: bool,
    tp: bool,
) -> dist::DistMeasured {
    let cfg = LowerConfig {
        parties: 2,
        bit_width: 32,
        ..Default::default()
    };
    let prog = compile(src, &cfg).expect("compile");
    let sched = schedule(&prog);
    let profile = NetworkProfile::fixed_latency(Duration::from_millis(rtt_ms))
        .with_bandwidth(mbps * 1e6 / 8.0);
    let hs = Mutex::new(InprocHandle::create(2));
    dist::run_two_party(
        &prog,
        &sched,
        ins,
        RunConfig {
            bits: 64,
            seed: 0xBADC0DE,
            hoist_bits: hoist,
            trunc_pairs: tp,
            bilinear: false,
        },
        |i| LatencyTransport::new(hs.lock().unwrap()[i].clone(), profile, 2),
    )
    .expect("run")
}

fn lenet() -> (String, HashMap<String, i64>) {
    let net = baru::bench::conv_nn::Cnn::lenet5_fhe_friendly(8, baru::ml::Activation::Square);
    baru::bench::conv_nn::debug_case(&net, 0)
}

fn mlp(shape: &[usize]) -> (String, HashMap<String, i64>) {
    (
        baru::bench::two_party_nn::debug_src(shape, 8),
        baru::bench::two_party_nn::debug_bindings(shape, 8),
    )
}

fn main() {
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(body)
        .expect("spawn")
        .join()
        .expect("probe");
}

fn body() {
    // Isolate the three terms by measurement. Elapsed time on a link is some
    // combination of local computation, bytes divided by bandwidth, and rounds
    // times round-trip time; which one dominates is a property of the workload and
    // is not safe to assume. Measuring the same program at unbounded and at finite
    // bandwidth, and at zero and at wide-area latency, separates them without any
    // modelling: the differences are themselves measurements.
    const FREE: f64 = 1e12;
    println!(
        "{:>12} {:>10} {:>5} {:>9} {:>8} {:>8} {:>8} {:>8}  detail",
        "workload", "rescale", "exch", "bytes/KiB", "compute", "+bw", "+rtt", "both"
    );
    let relu = (
        "relu".to_string(),
        "(* x (> x 0))".to_string(),
        HashMap::from([("x".to_string(), -9i64)]),
    );
    let mut jobs = vec![relu];
    for shape in [vec![16usize, 8, 4], vec![784usize, 32, 10]] {
        let (src, ins) = mlp(&shape);
        jobs.push((
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            src,
            ins,
        ));
    }
    {
        let names: Vec<String> = (0..10).map(|i| format!("x{i}")).collect();
        let mut binds: Vec<String> = Vec::new();
        let idx = baru::bench::depth::argmax_pairwise(&mut binds, &names, "g");
        let src = format!("(let* ({}) {idx})", binds.join(" "));
        let ins: HashMap<String, i64> = names
            .iter()
            .cloned()
            .zip([3i64, 9, 1, 4, 15, 92, 6, 500, 5, 35])
            .collect();
        jobs.push(("argmax10".to_string(), src, ins));
    }
    let (lsrc, lins) = lenet();
    jobs.push(("lenet5".to_string(), lsrc, lins));

    for (label, src, ins) in &jobs {
        for (prep, hoist, tp) in [
            ("exact", true, false),
            ("pairs", true, true),
            ("pairs/cold", false, true),
        ] {
            let c = go(src, ins, 0, FREE, hoist, tp);
            let b = go(src, ins, 0, 100.0, hoist, tp);
            let r = go(src, ins, 25, FREE, hoist, tp);
            let a = go(src, ins, 25, 100.0, hoist, tp);
            println!(
                "{label:>12} {prep:>10} {:>5} {:>9.0} {:>7.0}ms {:>7.0}ms {:>7.0}ms {:>7.0}ms  {} muls, {} bits -> {:?}",
                a.exchanges,
                a.online_bytes as f64 / 1024.0,
                c.elapsed_secs * 1e3,
                b.elapsed_secs * 1e3,
                r.elapsed_secs * 1e3,
                a.elapsed_secs * 1e3,
                a.muls,
                a.random_bits,
                a.outputs
            );
        }
    }
}
