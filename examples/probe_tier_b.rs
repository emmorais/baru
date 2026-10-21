//! Tier B of the HEIR comparison: **what the architectures can express**, measured.
//!
//! §VI.15/VI.16 compared cost where both systems can run the same program. This
//! probe covers the programs where they cannot, and it exists because the first
//! draft of the proposal asserted three such cases from HEIR's paper without
//! running any of them. One of the three did not survive being run — see the
//! `mux` row below — which is the reason for measuring rather than citing.
//!
//! Each row reports Baru's measured cost and the specific HEIR mechanism that
//! blocks it, sourced to arXiv:2508.11095.
//!
//! `cargo run --release --example probe_tier_b`

use baru::bench::comm::{LatencyTransport, NetworkProfile};
use baru::crypto::net::InprocHandle;
use baru::mpcc::dist::RunConfig;
use baru::mpcc::{compile, dist, schedule, LowerConfig};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

/// 50 ms one-way, 100 Mbit/s — the §VI.15 wide-area profile.
const RTT_MS: u64 = 50;
const MBPS: f64 = 100.0;

fn bind(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

/// Compile, schedule, and run with two real parties over a sleeping transport.
/// Returns `(instructions, scheduled rounds, exchanges, elapsed ms, output)`.
fn run(src: &str, ins: &HashMap<String, i64>) -> (usize, usize, u64, f64, i64) {
    let cfg = LowerConfig {
        parties: 2,
        bit_width: 32,
        ..Default::default()
    };
    let prog = compile(src, &cfg).expect("compile");
    let sched = schedule(&prog);
    let profile = NetworkProfile::fixed_latency(Duration::from_millis(RTT_MS))
        .with_bandwidth(MBPS * 1e6 / 8.0);
    let hs = Mutex::new(InprocHandle::create(2));
    let m = dist::run_two_party(&prog, &sched, ins, RunConfig::default(), |i| {
        LatencyTransport::new(hs.lock().unwrap()[i].clone(), profile, 2)
    })
    .expect("run");
    (
        prog.code.len(),
        sched.rounds.len(),
        m.exchanges,
        m.elapsed_secs * 1e3,
        m.outputs.first().map(|(_, v)| *v).unwrap_or(i64::MIN),
    )
}

fn main() {
    println!(
        "Tier B — expressiveness. Two parties over a transport that really sleeps,\n\
         {RTT_MS} ms one-way / {MBPS:.0} Mbit/s. `want` is the plaintext answer.\n"
    );
    println!(
        "{:<24} {:>6} {:>7} {:>6} {:>9} {:>6} {:>6}",
        "program", "instrs", "rounds", "exch", "elapsed", "got", "want"
    );

    // ── 1. Exact comparison. No target scheme of HEIR's can compare at all, so
    //    its only route is polynomial approximation, and its paper lists the
    //    degree and interval as *user annotations* rather than inferred.
    let cases: Vec<(&str, &str, HashMap<String, i64>, i64)> = vec![
        ("exact ReLU", "(* x (> x 0))", bind(&[("x", -7)]), 0),
        (
            "exact max of two",
            "(if (> a b) a b)",
            bind(&[("a", 3), ("b", 9)]),
            9,
        ),
        (
            "exact argmax of four",
            "(if (> a b) (if (> a c) 0 2) (if (> b c) 1 2))",
            bind(&[("a", 2), ("b", 9), ("c", 4)]),
            1,
        ),
        // ── 2. The oblivious branch, for the row below the table.
        (
            "branch, oblivious",
            "(if (> x 0) (* x 2) (- 0 x))",
            bind(&[("x", 7)]),
            14,
        ),
    ];

    for (name, src, ins, want) in &cases {
        let (i, r, e, ms, got) = run(src, ins);
        println!(
            "{:<24} {:>6} {:>7} {:>6} {:>8.0}ms {:>6} {:>6} {}",
            name,
            i,
            r,
            e,
            ms,
            got,
            want,
            if got == *want { "✓" } else { "✗" }
        );
    }

    // ── The declared-leakage branch. `reconstruct` opens a value mid-protocol,
    //    which the circuit path does not implement -- and until 2026-10-21 it
    //    compiled to the constant zero instead of saying so. Show the refusal.
    {
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let src = "(if (> (reconstruct x) 0) (* (reconstruct x) 2) 0)";
        match compile(src, &cfg) {
            Ok(_) => println!("\nbranch, declared leak    compiles on the circuit path"),
            Err(e) => println!(
                "\nbranch, declared leak    circuit path REFUSES it:\n    {}",
                e.lines().next().unwrap_or("")
            ),
        }
        // The interpreter does implement it, by revealing the condition.
        let got = baru::interpreter::eval::eval_shared(
            "(let ((x (share 7 32))) (reconstruct (if (> x 0) (* x 2) 0)))",
        );
        println!("                         interpreter: {got} (want 14), by revealing 1 bit");
    }

    println!(
        "\n\
        What this says, and one thing it corrects.\n\
        \n\
        1. EXACT COMPARISON is a genuine expressiveness gap. ReLU, max and argmax\n\
           are computed exactly above. None of HEIR's target schemes (BGV/BFV,\n\
           CKKS, CGGI for arithmetic) can compare, so its only route is polynomial\n\
           approximation with user-annotated degree and interval -- listed among\n\
           its own limitations. VI.16 measured the consequence: at depth 3 our CKKS\n\
           arm's logit error exceeded the class margin and the run was REFUSED,\n\
           while the exact arm returned the reference class at every depth.\n\
        \n\
        2. OBLIVIOUS BRANCHING IS NOT A DIFFERENCE, and the proposal claimed it\n\
           was. Baru lowers a secret-dependent `if` to a Mux -- it evaluates both\n\
           arms, exactly as HEIR's convert-if-to-select does. Obliviousness forces\n\
           the same shape on both, and this claim has been withdrawn.\n\
        \n\
        3. DECLARED LEAKAGE is a real axis, but only the interpreter has it. The\n\
           interpreter's shared `if` reveals its condition and says so in the\n\
           source ('leaks 1 bit'), so a programmer can trade one revealed bit for\n\
           the rounds the oblivious form costs. An FHE-only compiler cannot\n\
           express this at ANY cost -- nothing can be opened without the key.\n\
           But Baru's own circuit path cannot either: it has no `reconstruct`,\n\
           and the two backends therefore disagree on the same program. Closing\n\
           that is work, not a result.\n\
        \n\
        4. THREE CORRECTNESS BUGS came out of writing this probe, all in\n\
           secret-conditioned code that no benchmark exercises. See VI.17.\n"
    );
}
