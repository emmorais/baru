//! `baru bench-2pc-nn` — a whole private network at a **matched threat model**.
//!
//! # The question this exists to answer
//!
//! Every private-inference figure in this repo so far comes from the 3-party
//! honest-majority path: ABY3 shares, `crypto::gc_protocol`'s two-garbler
//! comparison, the `gadgets` library. That is a fine threat model and it is *not*
//! FHE's. Single-server FHE assumes one server and no non-collusion whatsoever;
//! a 3-party honest majority assumes two of three parties stay honest and do not
//! collude. A 3-party system beating single-server FHE on latency is therefore
//! not a result — it is a consequence of assuming away the hard part, and it is
//! why every hybrid inference system in the literature (Gazelle, Delphi,
//! Cheetah) is **2-party**.
//!
//! This benchmark runs the same network under **2-party dishonest-majority**
//! MPC, where one of two parties may be corrupt and nothing is assumed about
//! collusion — the setting in which a comparison against FHE is commensurable.
//!
//! One asymmetry survives the matching and is *not* measured away: FHE lets the
//! client go offline after uploading a ciphertext, while 2PC needs it online for
//! the whole evaluation and needs multi-gigabyte preprocessing besides. Same
//! corruption model, different interaction model. Huang, Maeng & Suh
//! ("Beyond Latency", arXiv 2604.00169) report that this is where MPC's real
//! dollar and energy cost lives, and no wall-clock number here contradicts them.
//!
//! # What is measured and what is computed
//!
//! * **Measured, by running the protocol** — rounds, online bytes, triples, and
//!   for Semi2k and LowGear the real offline bytes of OT- and SHE-generated
//!   triples. Also the random-bit preprocessing, which stopped being free in
//!   E47 and is the dominant term here (see below).
//! * **Computed** — wall clock, by putting those counts through a network
//!   profile. There is no real socket in the loop; `bench-net` validates the
//!   model to 13.6% worst case.
//!
//! # Online and total are both reported, always
//!
//! A comparison-heavy network is preprocessing-dominated, and quoting MPC's
//! online phase against an FHE total is the error survey §7.3 records making
//! three times. So every row carries `online` *and* `total`, and the CLI prints
//! both columns rather than choosing. The offline column is the honest one for a
//! cold server; the online column is honest only for a server that generated its
//! preprocessing before the query arrived, which is a deployment claim and is
//! labelled as one.
//!
//! # Why the nonlinearities dominate, which is the finding
//!
//! Each `ash-right` rescale and each ReLU costs one bit decomposition, and a
//! decomposition draws `bit_width + 1 + KAPPA` shared random bits — 59 at the
//! default 32-bit width — each of which now costs a real Beaver triple. So a
//! network's triple bill is driven by its *neuron count*, not by its
//! multiplication count: a layer of `n` inputs and `m` outputs does `n·m`
//! multiplications but `m` rescales, and at `n = 16` the rescales already cost
//! more than the arithmetic. `decompositions` is reported so the ratio is visible
//! instead of inferred.

use std::collections::HashMap;

use serde::Serialize;

use crate::ml::{Activation, QuantizedLayer, QuantizedNetwork};
use crate::mpcc::cost::{native_domain, Deployment};
use crate::mpcc::vm::KAPPA;
use crate::mpcc::{
    compile, engine_can_run_program, engine_supports, find, run_protocol, schedule, LowerConfig,
};

/// One protocol on one network shape.
#[derive(Debug, Clone, Serialize)]
pub struct NnPoint {
    pub protocol: String,
    /// `16x8x4` — input, hidden…, output.
    pub shape: String,
    /// Online communication rounds.
    pub rounds: u64,
    /// Rounds spent generating random bits, which a warm server has already paid.
    pub offline_rounds: u64,
    pub online_bytes: u64,
    /// Preprocessing bytes: amortized triple generation plus the random-bit
    /// protocol's own traffic.
    pub offline_bytes: u64,
    pub triples: u64,
    /// Bit decompositions performed — one per rescale, one per ReLU, one per
    /// argmax step. The quantity that actually drives the cost.
    pub decompositions: u64,
    /// How those decompositions split: the network's own nonlinear surface
    /// (rescales + ReLUs) against the `classes - 1` argmax steps. Reported so a
    /// reader can see that the classifier's decision is the cheap part.
    pub decomps_network: u64,
    pub decomps_argmax: u64,
    /// Secret×secret multiplications in the program itself (excluding the ones
    /// random-bit generation spends).
    pub muls: u64,
    /// Warm server: preprocessing already generated when the query arrives.
    pub online_secs: f64,
    /// Cold server: it pays for everything.
    pub total_secs: f64,
    /// Bit-exact against the plaintext fixed-point reference. **Gates every
    /// number above** — `run_cli` refuses to print a summary if any row is false.
    pub correct: bool,
    pub class: i64,
    pub expected: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct NnReport {
    pub parties: usize,
    pub bit_width: u32,
    pub scale_bits: u32,
    pub network: String,
    pub points: Vec<NnPoint>,
    /// Shapes that no protocol could run, and why — reported rather than
    /// silently absent, since the ceiling is a result.
    pub skipped: Vec<String>,
}

// ─── The workload ─────────────────────────────────────────────────────────────

/// A deterministic integer network of the given shape.
///
/// Weights are small and mixed-sign so that ReLU actually fires and the rescale
/// actually sees negative values — a network whose pre-activations happen to be
/// positive would not have caught the sign-extension bug in `vm`'s `TruncPr`.
fn fixture(shape: &[usize], scale_bits: u32) -> QuantizedNetwork {
    let unit = 1i64 << scale_bits;
    let mut layers = Vec::new();
    for (l, pair) in shape.windows(2).enumerate() {
        let (n_in, n_out) = (pair[0], pair[1]);
        let weights: Vec<Vec<i64>> = (0..n_out)
            .map(|o| {
                (0..n_in)
                    .map(|i| {
                        // Spread over [-unit/2, unit/2], never all one sign.
                        let k = ((o * 7 + i * 5 + l * 3) % 11) as i64 - 5;
                        k * unit / 8
                    })
                    .collect()
            })
            .collect();
        let bias = (0..n_out)
            .map(|o| (((o * 5 + l) % 7) as i64 - 3) * unit / 4)
            .collect();
        // Last layer is the logit layer: no ReLU, or argmax would see a
        // half-clamped vector and the comparison would be uninteresting.
        let activation = if l + 2 == shape.len() {
            Activation::Identity
        } else {
            Activation::Relu
        };
        layers.push(QuantizedLayer {
            weights,
            bias,
            activation,
        });
    }
    QuantizedNetwork { scale_bits, layers }
}

fn fixture_input(n: usize, scale_bits: u32) -> Vec<i64> {
    let unit = 1i64 << scale_bits;
    (0..n).map(|i| ((i % 9) as i64 - 4) * unit / 4).collect()
}

/// First-max-wins argmax, the tie-break the generated circuit implements
/// (`>` is strict, so an equal later logit does not displace the incumbent).
fn argmax_ref(logits: &[i64]) -> i64 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate().skip(1) {
        if *v > logits[best] {
            best = i;
        }
    }
    best as i64
}

/// Emit the network as one Baru expression whose single output is the class
/// index — everything secret: the input, every weight, every bias.
///
/// `compile_fhe` yields exactly one output wire, so a classifier has to reduce to
/// the class rather than expose the logit vector. That is the right shape anyway:
/// it is what a deployment reveals.
fn network_src(net: &QuantizedNetwork) -> String {
    let mut binds: Vec<String> = Vec::new();
    // Names of the current activation vector.
    let mut cur: Vec<String> = (0..net.layers[0].in_dim())
        .map(|i| format!("x{i}"))
        .collect();

    for (l, layer) in net.layers.iter().enumerate() {
        let mut next = Vec::new();
        for o in 0..layer.out_dim() {
            let dot = cur
                .iter()
                .enumerate()
                .map(|(i, x)| format!("(* w{l}_{o}_{i} {x})"))
                .reduce(|a, b| format!("(+ {a} {b})"))
                .expect("a layer has at least one input");
            // `QuantizedNetwork::forward` rescales *then* adds the bias, so the
            // bias lives at output scale. Match it exactly or the reference is
            // measuring a different network.
            let pre = format!("(+ (ash-right {dot} {}) b{l}_{o})", net.scale_bits);
            let name = format!("h{l}_{o}");
            match layer.activation {
                Activation::Identity => binds.push(format!("({name} {pre})")),
                // Mux ReLU: an MPC backend cannot branch on a secret.
                Activation::Relu => {
                    let p = format!("p{l}_{o}");
                    binds.push(format!("({p} {pre})"));
                    binds.push(format!("({name} (* {p} (> {p} 0)))"));
                }
            }
            next.push(name);
        }
        cur = next;
    }

    // Argmax fold: carry (best, idx) and select with the comparison bit.
    let mut best = cur[0].clone();
    let mut idx = "0".to_string();
    for (i, l) in cur.iter().enumerate().skip(1) {
        let c = format!("c{i}");
        binds.push(format!("({c} (> {l} {best}))"));
        let nb = format!("m{i}");
        binds.push(format!("({nb} (+ {best} (* {c} (- {l} {best}))))"));
        let ni = format!("k{i}");
        binds.push(format!("({ni} (+ {idx} (* {c} (- {i} {idx}))))"));
        best = nb;
        idx = ni;
    }

    format!("(let* ({}) {idx})", binds.join(" "))
}

/// Every secret input the generated source refers to.
fn network_inputs(net: &QuantizedNetwork, input: &[i64]) -> HashMap<String, i64> {
    let mut m = HashMap::new();
    for (i, v) in input.iter().enumerate() {
        m.insert(format!("x{i}"), *v);
    }
    for (l, layer) in net.layers.iter().enumerate() {
        for (o, row) in layer.weights.iter().enumerate() {
            for (i, w) in row.iter().enumerate() {
                m.insert(format!("w{l}_{o}_{i}"), *w);
            }
            m.insert(format!("b{l}_{o}"), layer.bias[o]);
        }
    }
    m
}

// ─── The sweep ────────────────────────────────────────────────────────────────

pub fn run(
    shapes: &[Vec<usize>],
    dep: &Deployment,
    bit_width: u32,
    scale_bits: u32,
) -> Result<NnReport, String> {
    // One decomposition draws this many random bits; used to report the count
    // rather than hardcoding 59 and drifting if either constant moves.
    let bits_per_decomp = u64::from(bit_width + 1 + KAPPA);

    let mut points = Vec::new();
    let mut skipped = Vec::new();

    for shape in shapes {
        if shape.len() < 2 {
            return Err(format!(
                "shape {shape:?} needs at least an input and an output"
            ));
        }
        let label = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let net = fixture(shape, scale_bits);
        let input = fixture_input(shape[0], scale_bits);
        let want = argmax_ref(&net.forward(&input));
        let src = network_src(&net);
        let inputs = network_inputs(&net, &input);
        // Derived from the shape, not from the counters, so the two can be
        // cross-checked: one decomposition per rescale (every neuron) and one
        // more per ReLU (every neuron in a hidden layer).
        let decomps_argmax =
            (*shape.last().expect("a shape has an output") as u64).saturating_sub(1);

        let mut ran_any = false;
        for id in crate::mpcc::RUNNABLE {
            let spec = find(id).ok_or_else(|| format!("unknown protocol `{id}`"))?;
            if !spec.supports_parties(dep.parties) || !engine_supports(id, dep.parties) {
                continue;
            }
            let cfg = LowerConfig {
                domain: native_domain(spec, 64),
                parties: dep.parties,
                bit_width,
                public_vars: Vec::new(),
            };
            let prog = compile(&src, &cfg)?;
            if let Err(why) = engine_can_run_program(id, &prog) {
                skipped.push(format!("{label} / {id}: {why}"));
                continue;
            }
            let m = run_protocol(id, dep.parties, &prog, &sched_of(&prog), &inputs)?;
            ran_any = true;

            // Preprocessing per triple *consumed*, the `bench-layer` convention:
            // generation is batched, so dividing by the generated count and
            // multiplying by the consumed one avoids comparing batch sizes.
            let per_triple = if m.triples_generated > 0 {
                m.offline_bytes as f64 / m.triples_generated as f64
            } else {
                0.0
            };
            let offline_bytes = (per_triple * m.triples as f64) as u64 + m.random_bit_bytes;

            let bw = dep.bandwidth_bytes_per_sec;
            let rtt = dep.latency_us / 1e6;
            let online_secs = m.rounds as f64 * rtt + m.online_bytes as f64 / bw;
            let total_secs =
                online_secs + m.random_bit_rounds as f64 * rtt + offline_bytes as f64 / bw;

            let class = m.outputs.first().map(|(_, v)| *v).unwrap_or(i64::MIN);
            points.push(NnPoint {
                protocol: (*id).to_string(),
                shape: label.clone(),
                rounds: m.rounds,
                offline_rounds: m.random_bit_rounds,
                online_bytes: m.online_bytes,
                offline_bytes,
                triples: m.triples,
                decompositions: m.random_bits / bits_per_decomp,
                decomps_network: (m.random_bits / bits_per_decomp).saturating_sub(decomps_argmax),
                decomps_argmax,
                muls: m.muls,
                online_secs,
                total_secs,
                correct: class == want,
                class,
                expected: want,
            });
        }
        if !ran_any {
            skipped.push(format!("{label}: no protocol could run it"));
        }
    }

    Ok(NnReport {
        parties: dep.parties,
        bit_width,
        scale_bits,
        network: format!(
            "{:.0} ms one-way, {:.0} Mbit/s",
            dep.latency_us / 1e3,
            dep.bandwidth_bytes_per_sec * 8.0 / 1e6
        ),
        points,
        skipped,
    })
}

fn sched_of(prog: &crate::mpcc::ir::Program) -> crate::mpcc::schedule::Schedule {
    schedule(prog)
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

const USAGE: &str = "usage: baru bench-2pc-nn [options]

  --shapes A,B         network shapes, `x`-separated dims (default 16x8x4,32x16x4)
  --parties N          computing parties (default 2 — the matched threat model)
  --bits N             value bit width (default 32)
  --scale N            fixed-point scale bits (default 8)
  --lan                LAN profile instead of WAN
  --json               emit the report as JSON
  -h, --help           this message

Runs one private classifier end to end under 2-party dishonest-majority MPC —
the setting in which a comparison against single-server FHE is commensurable.
Input, weights and biases are all secret; only the class index is revealed.";

fn parse_shape(s: &str) -> Result<Vec<usize>, String> {
    s.split('x')
        .map(|d| {
            d.trim()
                .parse::<usize>()
                .map_err(|_| format!("bad dimension `{d}` in shape `{s}`"))
        })
        .collect()
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut shapes = vec![vec![16usize, 8, 4], vec![32, 16, 4]];
    let mut parties = 2usize;
    let mut bit_width = 32u32;
    let mut scale_bits = 8u32;
    let mut lan = false;
    let mut json = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--lan" => lan = true,
            "--shapes" => {
                let v = it
                    .next()
                    .ok_or_else(|| format!("--shapes needs a value\n\n{USAGE}"))?;
                shapes = v.split(',').map(parse_shape).collect::<Result<_, _>>()?;
            }
            "--parties" => {
                parties = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("--parties needs a number\n\n{USAGE}"))?;
            }
            "--bits" => {
                bit_width = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("--bits needs a number\n\n{USAGE}"))?;
            }
            "--scale" => {
                scale_bits = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("--scale needs a number\n\n{USAGE}"))?;
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let dep = if lan {
        Deployment::lan(parties)
    } else {
        Deployment::wan(parties)
    };
    let r = run(&shapes, &dep, bit_width, scale_bits)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!(
        "One private classifier end to end — {} parties, {}, Q{} fixed point",
        r.parties, r.network, r.scale_bits
    );
    println!("Input, weights and biases all secret; only the class index is revealed.\n");
    println!(
        "{:>10} {:>9} {:>7} {:>6} {:>8} {:>10} {:>11} {:>9} {:>9}  ok",
        "shape", "protocol", "rounds", "muls", "decomps", "online", "offline", "online", "total"
    );
    for p in &r.points {
        // `dealer` is not a threat model — it assumes a trusted party hands out
        // correlated randomness for free. Marked so no reader mistakes its row
        // for a deployable option; it is here as the online lower bound.
        let name = if p.protocol == "dealer" {
            format!("{}*", p.protocol)
        } else {
            p.protocol.clone()
        };
        println!(
            "{:>10} {:>9} {:>7} {:>6} {:>8} {:>10} {:>11} {:>8.0}ms {:>8.0}ms  {}",
            p.shape,
            name,
            p.rounds,
            p.muls,
            p.decompositions,
            human(p.online_bytes),
            human(p.offline_bytes),
            p.online_secs * 1e3,
            p.total_secs * 1e3,
            if p.correct { "✓" } else { "✗" }
        );
    }
    println!(
        "\n  * `dealer` is a LOWER BOUND, not a deployable protocol: it assumes a trusted\n\
         \x20   party hands out correlated randomness for free. Compare against `semi2k`\n\
         \x20   (OT preprocessing) and `lowgear` (SHE preprocessing)."
    );

    for s in &r.skipped {
        println!("\n  skipped — {s}");
    }

    let bad: Vec<&str> = r
        .points
        .iter()
        .filter(|p| !p.correct)
        .map(|p| p.protocol.as_str())
        .collect();
    if !bad.is_empty() {
        return Err(format!(
            "protocols disagree with the plaintext reference: {bad:?} — every cost \
             above is void until they are bit-exact"
        ));
    }
    println!("\n  every protocol reproduces the plaintext fixed-point classification");

    // Derive the conclusions from the rows rather than asserting them alongside.
    // The *last* semi2k row is the widest, since shapes are swept in order.
    if let Some(p) = r.points.iter().rev().find(|p| p.protocol == "semi2k") {
        println!(
            "\n  Where the cost is, derived from the widest row measured ({} on {}):",
            p.shape, p.protocol
        );
        println!(
            "  · a cold server pays {:.1}x a warm one — the gap is entirely preprocessing.",
            if p.online_secs > 0.0 {
                p.total_secs / p.online_secs
            } else {
                0.0
            }
        );
        println!(
            "  · {} of {} bit decompositions are the network's rescales and ReLUs; only {} \
             are the\n    argmax. Cost tracks NEURON count, not multiplication count.",
            p.decomps_network, p.decompositions, p.decomps_argmax
        );
        // The multiplication split is the sharpest form of it: the network's own
        // arithmetic is a rounding error against the decompositions' internals.
        let linear: u64 = shapes
            .iter()
            .find(|s| {
                s.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
                    == p.shape
            })
            .map(|s| s.windows(2).map(|w| (w[0] * w[1]) as u64).sum())
            .unwrap_or(0);
        println!(
            "  · the layers themselves are only {} of the {} multiplications ({:.1}%); the rest \
             is\n    inside the bit decompositions.",
            linear,
            p.muls,
            100.0 * linear as f64 / p.muls.max(1) as f64
        );
        let latency_share = p.rounds as f64 * (dep.latency_us / 1e6) / p.online_secs.max(1e-12);
        println!(
            "  · the online phase is {:.1}% round latency ({} rounds) against {:.1}% bandwidth, \
             so here\n    it is DEPTH that decides it, not traffic.",
            latency_share * 100.0,
            p.rounds,
            (1.0 - latency_share) * 100.0
        );
    }
    println!(
        "\nMEASURED by running the protocol: rounds, online bytes, triples, and the real\n\
         offline bytes of OT/SHE-generated triples and of random-bit generation.\n\
         Wall clock is COMPUTED from those counts and the profile — no socket is in the\n\
         loop (`bench-net` validates the model to 13.6% worst case).\n\n\
         `online` is a WARM server, one that generated preprocessing before the query\n\
         arrived; `total` is a COLD one. Both are printed because quoting MPC's online\n\
         phase against an FHE total is the error survey §7.3 records making three times."
    );
    Ok(())
}

fn human(b: u64) -> String {
    const K: f64 = 1024.0;
    let f = b as f64;
    if f >= K * K {
        format!("{:.1} MiB", f / (K * K))
    } else if f >= K {
        format!("{:.1} KiB", f / K)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A width-`n` layer is an `n`-term nested expression and the parser and
    /// circuit compiler each recurse once per term, so libtest's 2 MiB thread
    /// stack is the binding constraint rather than anything about the protocols.
    /// Narrowing the shapes instead would delete the measurement.
    fn on_a_deep_stack(body: fn()) {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(body)
            .expect("spawn a deep-recursion thread")
            .join()
            .expect("test body panicked");
    }

    /// The cheap half of the correctness gate, and the one that runs in CI.
    ///
    /// The risky new code here is [`network_src`] — a generated circuit that has
    /// to agree with [`QuantizedNetwork::forward`] on the rescale convention
    /// (shift *then* bias), on the mux ReLU, and on argmax's tie-break. That is a
    /// semantic question and it does not need a protocol to answer: evaluating
    /// the same expression in the plain interpreter settles it in milliseconds,
    /// whereas the protocol runs below take ~42 s each in a debug build because
    /// Semi2k drives real IKNP OT.
    ///
    /// So this checks the *circuit*, and the `#[ignore]`d tests check that the
    /// protocols compute the circuit. Both matter; only one is cheap.
    #[test]
    fn the_generated_circuit_agrees_with_the_plaintext_reference() {
        for shape in [
            vec![8usize, 4, 3],
            vec![6, 5, 4],
            // A single layer, no hidden ReLU at all — the degenerate case.
            vec![5, 3],
        ] {
            let net = fixture(&shape, 8);
            let input = fixture_input(shape[0], 8);
            let want = argmax_ref(&net.forward(&input));
            let src = network_src(&net);
            let inputs = network_inputs(&net, &input);

            let mut names: Vec<&String> = inputs.keys().collect();
            names.sort();
            let binds: Vec<String> = names
                .iter()
                .map(|k| format!("({} {})", k, inputs[*k]))
                .collect();
            let wrapped = format!("(let ({}) {})", binds.join(" "), src);
            let expr = crate::interpreter::parse(&wrapped)
                .unwrap_or_else(|e| panic!("generated source does not parse: {e}"));
            let got = match crate::interpreter::Participant::new_simulated(0).eval_quiet(&expr) {
                crate::interpreter::Types::Num(n) => n,
                other => panic!("expected a class index, got {other:?}"),
            };
            assert_eq!(
                got, want,
                "generated circuit disagrees with QuantizedNetwork::forward at {shape:?}"
            );
        }
    }

    /// The gate the whole benchmark rests on: a private classification must equal
    /// the plaintext fixed-point one, class for class. Any cost figure from a run
    /// that computed a different network is measuring the wrong thing.
    ///
    /// `#[ignore]`d because Semi2k generates its triples over real IKNP OT, which
    /// is ~42 s in a debug build — the repo's stated policy for tests that touch
    /// real cryptography (see `.github/workflows/nightly.yml`, which runs
    /// `--include-ignored`). The cheap semantic half runs in CI above.
    #[test]
    #[ignore]
    fn every_protocol_classifies_exactly() {
        on_a_deep_stack(|| {
            let r = run(&[vec![8, 4, 3]], &Deployment::wan(2), 32, 8).unwrap();
            assert!(!r.points.is_empty(), "no protocol ran: {:?}", r.skipped);
            for p in &r.points {
                assert!(
                    p.correct,
                    "{} on {}: class {}, expected {}",
                    p.protocol, p.shape, p.class, p.expected
                );
            }
        });
    }

    /// The fixture must actually exercise the paths that matter — a network whose
    /// pre-activations were all positive would not test the rescale's sign
    /// extension, and one whose ReLUs never clamped would not test the mux.
    #[test]
    fn the_fixture_has_negative_pre_activations_and_clamping_relus() {
        let net = fixture(&[8, 4, 3], 8);
        let input = fixture_input(8, 8);
        let mut cur = input;
        let mut saw_negative = false;
        let mut saw_clamp = false;
        for layer in &net.layers {
            let pre: Vec<i64> = layer
                .weights
                .iter()
                .zip(&layer.bias)
                .map(|(row, &b)| {
                    let acc: i128 = row
                        .iter()
                        .zip(&cur)
                        .map(|(&w, &x)| w as i128 * x as i128)
                        .sum();
                    ((acc >> net.scale_bits) as i64).wrapping_add(b)
                })
                .collect();
            saw_negative |= pre.iter().any(|&v| v < 0);
            cur = match layer.activation {
                Activation::Identity => pre,
                Activation::Relu => {
                    saw_clamp |= pre.iter().any(|&v| v < 0);
                    pre.into_iter().map(|v| v.max(0)).collect()
                }
            };
        }
        assert!(saw_negative, "the rescale never sees a negative value");
        assert!(saw_clamp, "no ReLU ever clamps, so the mux is untested");
    }

    /// The cost is driven by neuron count, not multiplication count — one bit
    /// decomposition per rescale, per ReLU and per argmax step. Doubling the
    /// input width doubles the multiplications and leaves the decompositions
    /// alone, which is the shape a reader needs to see.
    #[test]
    #[ignore]
    fn decompositions_track_neurons_and_muls_track_width() {
        on_a_deep_stack(|| {
            let dep = Deployment::wan(2);
            let narrow = run(&[vec![8, 4, 3]], &dep, 32, 8).unwrap();
            let wide = run(&[vec![16, 4, 3]], &dep, 32, 8).unwrap();
            let pick = |r: &NnReport| -> NnPoint {
                r.points
                    .iter()
                    .find(|p| p.protocol == "semi2k")
                    .cloned()
                    .expect("semi2k runs at two parties")
            };
            let (n, w) = (pick(&narrow), pick(&wide));

            // 4 ReLU + 4 rescale + 3 logit rescale + 2 argmax steps, both shapes.
            assert_eq!(
                n.decompositions, w.decompositions,
                "the same neurons should cost the same decompositions: {} vs {}",
                n.decompositions, w.decompositions
            );
            assert!(
                w.muls > n.muls,
                "twice the input width should cost more multiplications: {} vs {}",
                n.muls,
                w.muls
            );
        });
    }

    /// Cold and warm must differ, and by preprocessing. If they ever coincide the
    /// benchmark has stopped charging for random bits and the comparison against
    /// FHE has silently become online-vs-total again.
    #[test]
    #[ignore]
    fn a_cold_server_pays_strictly_more_than_a_warm_one() {
        on_a_deep_stack(|| {
            let r = run(&[vec![8, 4, 3]], &Deployment::wan(2), 32, 8).unwrap();
            for p in &r.points {
                if p.protocol == "dealer" {
                    // A dealer hands preprocessing over for free by definition.
                    continue;
                }
                assert!(
                    p.total_secs > p.online_secs,
                    "{} should pay for its preprocessing: {:.1} vs {:.1} ms",
                    p.protocol,
                    p.total_secs * 1e3,
                    p.online_secs * 1e3
                );
            }
        });
    }

    /// **Cost is linear in depth, with no discontinuity.** This is the 2-party
    /// dishonest-majority restatement of the repo's zero-bootstrap thesis, and it
    /// is the axis on which a comparison against FHE is actually interesting.
    ///
    /// Prior work — and this repo's own §7.1 — studies the FHE/MPC crossover in
    /// layer *width*. At a matched threat model the decisive axis is **depth**,
    /// because depth is what exhausts an FHE level budget and forces a bootstrap:
    /// HE-LRM (arXiv 2506.18150, the Orion group) reports a single CKKS bootstrap
    /// taking **75% of a 24 s inference**, and ~20 s on a single-threaded CPU. MPC
    /// has no such discontinuity — each layer costs the same increment forever.
    ///
    /// Asserted as a *second difference of zero* rather than as the measured
    /// constants (+18 rounds, +16 decompositions per 8-neuron layer), so the claim
    /// survives a change to the decomposition's internals — which is the claim,
    /// while the constants are an artifact of `KAPPA` and the bit width.
    #[test]
    #[ignore]
    fn cost_is_linear_in_depth_with_no_discontinuity() {
        on_a_deep_stack(|| {
            let dep = Deployment::lan(2);
            let shapes = vec![
                vec![16usize, 8, 4],
                vec![16, 8, 8, 4],
                vec![16, 8, 8, 8, 4],
                vec![16, 8, 8, 8, 8, 4],
            ];
            let r = run(&shapes, &dep, 32, 8).unwrap();
            let rows: Vec<&NnPoint> = r.points.iter().filter(|p| p.protocol == "semi2k").collect();
            assert_eq!(rows.len(), 4, "one row per depth");
            for p in &rows {
                assert!(p.correct, "{} misclassified", p.shape);
            }

            // Each added layer is identical, so the increments must be identical.
            for w in rows.windows(3) {
                let d2_rounds = (w[2].rounds as i64 - w[1].rounds as i64)
                    - (w[1].rounds as i64 - w[0].rounds as i64);
                let d2_decomp = (w[2].decompositions as i64 - w[1].decompositions as i64)
                    - (w[1].decompositions as i64 - w[0].decompositions as i64);
                assert_eq!(
                    d2_rounds, 0,
                    "rounds must grow linearly in depth: {} -> {} -> {}",
                    w[0].rounds, w[1].rounds, w[2].rounds
                );
                assert_eq!(
                    d2_decomp, 0,
                    "decompositions must grow linearly in depth: {} -> {} -> {}",
                    w[0].decompositions, w[1].decompositions, w[2].decompositions
                );
            }
            // And strictly growing, or "linear" would be satisfied by a constant.
            assert!(rows[3].rounds > rows[0].rounds);
        });
    }

    /// Rounds are set by **depth alone**: three shapes with the same layer count
    /// and wildly different widths cost the same rounds, because every
    /// decomposition at one dependence level batches into one.
    ///
    /// This is what makes the previous test's claim about depth non-trivial — if
    /// width also bought rounds, "linear in depth" would not isolate anything.
    #[test]
    #[ignore]
    fn rounds_are_set_by_depth_not_width() {
        on_a_deep_stack(|| {
            let dep = Deployment::lan(2);
            let r = run(
                &[vec![16usize, 8, 4], vec![64, 8, 4], vec![128, 8, 4]],
                &dep,
                32,
                8,
            )
            .unwrap();
            let rows: Vec<&NnPoint> = r.points.iter().filter(|p| p.protocol == "semi2k").collect();
            assert_eq!(rows.len(), 3);
            for p in &rows {
                assert!(p.correct, "{} misclassified", p.shape);
                assert_eq!(
                    p.rounds, rows[0].rounds,
                    "{} should cost the same rounds as {}: {} vs {}",
                    p.shape, rows[0].shape, p.rounds, rows[0].rounds
                );
                assert_eq!(
                    p.decompositions, rows[0].decompositions,
                    "and the same decompositions, since the neuron count is equal"
                );
            }
            // Width buys traffic and multiplications, and nothing else.
            assert!(
                rows[2].online_bytes > rows[0].online_bytes,
                "8x the input width should move more bytes"
            );
            assert!(rows[2].muls > rows[0].muls);
        });
    }

    #[test]
    fn cli_rejects_unknown_flags_and_bad_shapes() {
        assert!(run_cli(&["--nope".to_string()]).is_err());
        assert!(run_cli(&["--shapes".to_string(), "16xzz".to_string()]).is_err());
    }
}
