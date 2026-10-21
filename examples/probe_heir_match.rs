//! The two-party arm of the HEIR comparison, **measured** rather than modelled.
//!
//! `bench-2pc-nn` reports `rounds x RTT + bytes / bandwidth + compute`. That is a
//! prediction, and §VI.9 exists because three of ours were wrong in directions we
//! did not anticipate. §VI.15's first draft quoted those figures against HEIR as
//! if they were elapsed times; this probe replaces them with wall clock from two
//! parties running concurrently over a transport that really sleeps.
//!
//! It also answers a second question the modelled path could not. `bench-2pc-nn`
//! drives `run_protocol`, which does **not** enable the two optimizations §VI.9
//! and §VI.13 measured — truncation pairs (3.67x on LeNet-5) and bilinear operand
//! fusion (10.8x traffic at batch 8). So the arm HEIR was compared against was
//! missing them. Each shape is run twice, with every optimization off and on, so
//! the gap closed by *configuration alone* is visible before any new work.
//!
//! Both arms compute `x^2`, which is the only activation on which a same-function
//! comparison with CKKS is possible (§VI.7 gate 2).
//!
//! The `opt+logits` column additionally matches the **disclosure** model: HEIR
//! returns the logit vector and lets the client take the maximum in the clear,
//! while our arm computes the argmax inside the protocol. That is strictly more
//! work for strictly less disclosure, so the un-matched columns are conservative
//! against us. See `two_party_nn::debug_src_logits` for why a summed output has
//! exactly the round cost of returning the vector.
//!
//! `cargo run --release --example probe_heir_match`

use baru::bench::comm::{LatencyTransport, NetworkProfile};
use baru::crypto::net::InprocHandle;
use baru::mpcc::dist::RunConfig;
use baru::mpcc::ir::Op;
use baru::mpcc::{compile, dist, schedule, LowerConfig};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

/// `(label, one-way latency ms, Mbit/s)` — the two deployments `bench-2pc-nn`
/// names, so the measured figures land beside the modelled ones they replace.
const LINKS: [(&str, u64, f64); 2] = [("LAN", 0, 10_000.0), ("WAN", 50, 100.0)];

/// The shapes HEIR was measured on in §VI.15, in the same order.
const SHAPES: [&[usize]; 6] = [
    &[64, 32, 10],
    &[784, 32, 10],
    &[64, 32, 32, 10],
    &[64, 32, 32, 32, 10],
    &[64, 32, 32, 32, 32, 10],
    // Multiplicative depth 10: HEIR segfaults in key generation here, so the
    // row exists to show the MPC arm continuing past the cliff.
    &[64, 32, 32, 32, 32, 32, 10],
];

/// HEIR `eval` for the same shape, milliseconds (§VI.15). `None` where HEIR
/// could not run it at all — multiplicative depth 10 SIGSEGVs in key generation.
const HEIR_MS: [Option<f64>; 6] = [
    Some(167.0),
    Some(178.0),
    Some(659.3),
    Some(1051.3),
    Some(1493.4),
    None,
];

fn label(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn go(shape: &[usize], rtt_ms: u64, mbps: f64, opt: bool, logits: bool) -> dist::DistMeasured {
    let src = if logits {
        baru::bench::two_party_nn::debug_src_logits(shape, 8)
    } else {
        baru::bench::two_party_nn::debug_src_sq(shape, 8)
    };
    let ins = baru::bench::two_party_nn::debug_bindings_sq(shape, 8);
    let cfg = LowerConfig {
        parties: 2,
        bit_width: 32,
        ..Default::default()
    };
    let prog = compile(&src, &cfg).expect("compile");
    let sched = schedule(&prog);
    let profile = NetworkProfile::fixed_latency(Duration::from_millis(rtt_ms))
        .with_bandwidth(mbps * 1e6 / 8.0);
    let hs = Mutex::new(InprocHandle::create(2));
    dist::run_two_party(
        &prog,
        &sched,
        &ins,
        RunConfig {
            bits: 64,
            seed: 0xBADC0DE,
            // All three are the *online-phase* accounting: preprocessing hoisted,
            // which is the figure comparable against a warm homomorphic server
            // whose keygen is likewise excluded from `eval`.
            hoist_bits: opt,
            trunc_pairs: opt,
            bilinear: opt,
        },
        |i| LatencyTransport::new(hs.lock().unwrap()[i].clone(), profile, 2),
    )
    .expect("run")
}

/// Attribute each scheduled round to the deepest protocol it carries, so the
/// remaining wide-area cost can be read as *which operation forced a round
/// trip*. On this link elapsed time is round count times latency, so this is
/// the only table that says where to optimize next.
fn attribute(shape: &[usize]) {
    let src = baru::bench::two_party_nn::debug_src_sq(shape, 8);
    let cfg = LowerConfig {
        parties: 2,
        bit_width: 32,
        ..Default::default()
    };
    let prog = compile(&src, &cfg).expect("compile");
    let sched = schedule(&prog);
    let kind = |i: usize| -> &'static str {
        match &prog.code[i].op {
            Op::Mul => "mul",
            Op::LtS | Op::LtU => "lt",
            Op::EqZ => "eqz",
            Op::TruncPr { .. } => "truncpr",
            Op::Open => "open",
            Op::Mux => "mux",
            Op::Input { .. } => "input",
            Op::Output => "output",
            _ => "other",
        }
    };
    let mut charged: BTreeMap<&str, usize> = BTreeMap::new();
    let mut widths: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &sched.rounds {
        let mut mix: BTreeMap<&str, usize> = BTreeMap::new();
        for &i in &r.instrs {
            *mix.entry(kind(i)).or_default() += 1;
        }
        let owner = [
            "lt", "eqz", "truncpr", "mux", "mul", "open", "input", "output",
        ]
        .iter()
        .find(|k| mix.contains_key(**k))
        .copied()
        .unwrap_or("?");
        *charged.entry(owner).or_default() += 1;
        *widths.entry(owner).or_default() += r.instrs.len();
    }
    println!(
        "  {:<22} {} instrs, {} scheduled rounds",
        label(shape),
        prog.code.len(),
        sched.rounds.len()
    );
    for (k, v) in &charged {
        println!(
            "      {v:>3} charged to {k:<8} (mean width {:>6.0} ops)",
            widths[k] as f64 / *v as f64
        );
    }
}

fn main() {
    // A 784-wide dense layer is a deep expression for the front end to recurse
    // through; the same reason `probe_wan` spawns with a large stack.
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(body)
        .expect("spawn")
        .join()
        .expect("probe");
}

fn body() {
    println!(
        "Two-party arm vs HEIR, x^2 activation, elapsed wall clock of the slower party.\n\
         `base` is every optimization off, as `bench-2pc-nn` runs it; `opt` enables\n\
         hoisted bits, truncation pairs (VI.9) and bilinear operand fusion (VI.13).\n\
         HEIR column is `eval` from VI.15, keygen excluded on both sides.\n"
    );
    for (link, rtt, mbps) in LINKS {
        println!("--- {link}: {rtt} ms one-way, {mbps:.0} Mbit/s ---");
        println!(
            "{:>22} {:>6} {:>6} {:>6} {:>9} {:>9} {:>11} {:>10} {:>11}",
            "shape", "exch", "opt", "logit", "base", "opt", "opt+logits", "HEIR eval", "matched"
        );
        for (i, shape) in SHAPES.iter().enumerate() {
            let base = go(shape, rtt, mbps, false, false);
            let opt = go(shape, rtt, mbps, true, false);
            let lg = go(shape, rtt, mbps, true, true);
            let (b, o, l) = (
                base.elapsed_secs * 1e3,
                opt.elapsed_secs * 1e3,
                lg.elapsed_secs * 1e3,
            );
            let heir = HEIR_MS[i];
            // The matched-disclosure column is the like-for-like one: both arms
            // return the logit vector and neither computes the argmax privately.
            let vs = match heir {
                Some(h) if l <= h => format!("{:.1}x ours", h / l),
                Some(h) => format!("{:.1}x HEIR", l / h),
                None => "HEIR dead".to_string(),
            };
            println!(
                "{:>22} {:>6} {:>6} {:>6} {:>9} {:>9} {:>11} {:>10} {:>11}",
                label(shape),
                base.exchanges,
                opt.exchanges,
                lg.exchanges,
                format!("{b:.0}ms"),
                format!("{o:.0}ms"),
                format!("{l:.0}ms"),
                heir.map(|h| format!("{h:.0}ms")).unwrap_or("—".into()),
                vs,
            );
        }
        println!();
    }
    println!(
        "`exch` is interactive exchanges — the number of times a party blocked on its\n\
         peer — before and after. Where it does not move, the gain is bytes, not rounds.\n"
    );
    println!("--- where the remaining rounds go (static schedule) ---");
    for shape in [SHAPES[0], SHAPES[4]] {
        attribute(shape);
    }
}
