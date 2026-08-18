// Where do the rounds go? On a wide-area link elapsed time is round count times
// round-trip time plus bytes over bandwidth, and for this workload the first term
// dominates by an order of magnitude. So the question that decides the comparison
// against a zero-round FHE server is not how many multiplications the circuit has,
// it is how deep it is. This attributes every scheduled level to the operation that
// forced it.
use baru::mpcc::ir::Op;
use baru::mpcc::{compile, schedule, LowerConfig};
use std::collections::BTreeMap;

fn main() {
    for (label, src) in [
        ("relu", "(* x (> x 0))".to_string()),
        (
            "16x8x4",
            baru::bench::two_party_nn::debug_src(&[16, 8, 4], 8),
        ),
        (
            "784x32x10",
            baru::bench::two_party_nn::debug_src(&[784, 32, 10], 8),
        ),
    ] {
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let prog = compile(&src, &cfg).expect("compile");
        let sched = schedule(&prog);
        let kind = |idx: usize| -> &'static str {
            match &prog.code[idx].op {
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
        println!(
            "\n=== {label}: {} instructions, {} scheduled rounds",
            prog.code.len(),
            sched.rounds.len()
        );
        let mut charged: BTreeMap<&str, usize> = BTreeMap::new();
        let mut widths: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &sched.rounds {
            let mut mix: BTreeMap<&str, usize> = BTreeMap::new();
            for &i in &r.instrs {
                *mix.entry(kind(i)).or_default() += 1;
            }
            // Charge the round to the deepest-protocol kind it contains: a round that
            // carries a comparison exists because of the comparison, whatever else
            // rode along inside it.
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
        for (k, v) in &charged {
            let w = widths[k] as f64 / *v as f64;
            println!("  {v:>4} rounds charged to {k:<8} (mean width {w:>7.0} ops)");
        }
    }
}
