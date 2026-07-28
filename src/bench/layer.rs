//! `baru bench-layer` — the MPC↔FHE crossover, from measured components.
//!
//! This is the experiment `docs/research-agenda.md` §4 was missing. The claim
//! there is that a wide *encrypted* linear layer belongs on FHE under a
//! **dishonest** majority and on MPC under an **honest** one, with a crossover
//! somewhere around a couple of hundred multiplications. Both halves rested on a
//! modelled triple cost until `src/mpcc/semi2k.rs` made it measurable.
//!
//! # What is measured here, and what is still computed
//!
//! Being exact about this is the point of the benchmark:
//!
//! * **Measured, by running the protocol** — rounds, online bytes, triples, and
//!   for Semi2k the **offline bytes of real OT-generated triples**. These come
//!   from `mpcc`'s VM with real counters, not from the cost model.
//! * **Measured, from a different benchmark** — the FHE side's per-operation cost
//!   (`bench-fhe-innerprod`: one BFV ct×ct multiply, 128 KB ciphertext), carried
//!   in `mpcc::fhe_cost`.
//! * **Computed** — wall-clock, by putting those measured counts through a network
//!   profile (latency × rounds + bytes ÷ bandwidth). There is still no real
//!   network in the loop, which is `docs/mpc-compiler-roadmap.md`'s P1 remainder.
//!
//! So this is a crossover derived from *measured components*, not a measured
//! end-to-end crossover. The distinction matters and the output states it.
//!
//! # Why the layer is `Σ aᵢ·bᵢ` and not `Σ wᵢ·xᵢ`
//!
//! Both operands are secret, i.e. the weights are encrypted too. That is Baru's
//! threat model (`docs/paper-outline.md` C2) and it is what makes the comparison
//! interesting: a *public*-weight layer is free under arithmetic sharing (local
//! scalar multiply, no triple), so there would be nothing for FHE to win. Every
//! multiplication here consumes a Beaver triple, which is exactly the cost that
//! grows with width.

use std::collections::HashMap;

use serde::Serialize;

use crate::mpcc::cost::{native_domain, Deployment};
use crate::mpcc::{compile, engine_supports, fhe_cost, find, run_protocol, schedule, LowerConfig};

/// One protocol at one layer width.
#[derive(Debug, Clone, Serialize)]
pub struct LayerPoint {
    pub protocol: String,
    pub width: usize,
    /// Measured by running the protocol.
    pub rounds: u64,
    pub online_bytes: u64,
    pub triples: u64,
    /// Real OT bytes, non-zero only where triples are actually generated.
    pub offline_bytes: u64,
    pub triples_generated: u64,
    /// Wall-clock computed from the measured counts and the network profile.
    pub secs: f64,
    /// The layer's output matched the plain interpreter.
    pub correct: bool,
}

/// The **hybrid** alternative: FHE for the linear part, MPC for the ReLU.
///
/// It has to be the hybrid and not "FHE alone", because BFV cannot evaluate a
/// comparison at any price — so the nonlinearity stays on MPC either way. An
/// earlier version of this benchmark compared FHE's *linear* cost against MPC's
/// *whole layer* and made FHE look like it won at every width; the ReLU rounds
/// were simply missing from one side.
#[derive(Debug, Clone, Serialize)]
pub struct FhePoint {
    pub scheme: String,
    pub width: usize,
    /// Packed ct×ct multiplications — one, while the width fits the slots.
    pub ops: u64,
    /// Ciphertexts moved for the switch in and out.
    pub switch_bytes: u64,
    /// Seconds for the FHE linear part plus the switch.
    pub linear_secs: f64,
    /// Measured MPC cost of the ReLU alone — the part FHE cannot do.
    pub relu_secs: f64,
    pub relu_rounds: u64,
    pub relu_triples: u64,
    /// `linear_secs + relu_secs`, the honest total.
    pub secs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerReport {
    pub widths: Vec<usize>,
    pub mpc: Vec<LayerPoint>,
    pub fhe: Vec<FhePoint>,
    pub latency_us: f64,
    pub bandwidth_bytes_per_sec: f64,
    /// Width at which FHE first undercuts the cheapest MPC protocol, if any.
    pub crossover: Option<usize>,
}

/// `Σ aᵢ·bᵢ` over `width` pairs of secret inputs, then a ReLU.
fn layer_src(width: usize) -> String {
    let mut dot = "(* a0 b0)".to_string();
    for i in 1..width {
        dot = format!("(+ {dot} (* a{i} b{i}))");
    }
    // Mux-style ReLU: the MPC interpreter cannot branch on a secret.
    format!("(* {dot} (> {dot} 0))")
}

/// Just the ReLU, on one secret input — the MPC half of the hybrid.
fn relu_src() -> String {
    "(* x (> x 0))".to_string()
}

fn layer_inputs(width: usize) -> HashMap<String, i64> {
    let mut m = HashMap::new();
    for i in 0..width {
        m.insert(format!("a{i}"), (i as i64 % 7) - 3);
        m.insert(format!("b{i}"), (i as i64 % 5) + 1);
    }
    m
}

fn plain(width: usize, inputs: &HashMap<String, i64>) -> i64 {
    let dot: i64 = (0..width)
        .map(|i| inputs[&format!("a{i}")] * inputs[&format!("b{i}")])
        .sum();
    if dot > 0 {
        dot
    } else {
        0
    }
}

/// Cost of the layer on FHE, from measured per-operation figures.
///
/// One packed ct×ct multiply while the width fits the slots, plus the ciphertexts
/// the switch moves in and out. No rounds — homomorphic evaluation is
/// non-interactive — but the switch itself costs one round each way, which is
/// charged here because the nonlinearity has to leave FHE regardless.
fn fhe_layer(
    width: usize,
    spec: &fhe_cost::FheSpec,
    dep: &Deployment,
    relu: &LayerPoint,
) -> FhePoint {
    let ops = (width as f64 / spec.slots).ceil().max(1.0);
    // In and out: the packed input, and the packed result.
    let cts = 2.0 * (width as f64 / spec.slots).ceil().max(1.0);
    let switch_bytes = cts * spec.ct_bytes;
    let compute = ops * spec.mul_ns / 1e9;
    // Two switch rounds (into FHE, and back out for the nonlinearity).
    let latency = 2.0 * dep.latency_us / 1e6;
    let linear_secs = compute + latency + switch_bytes / dep.bandwidth_bytes_per_sec;
    FhePoint {
        scheme: spec.id.to_string(),
        width,
        ops: ops as u64,
        switch_bytes: switch_bytes as u64,
        linear_secs,
        relu_secs: relu.secs,
        relu_rounds: relu.rounds,
        relu_triples: relu.triples,
        secs: linear_secs + relu.secs,
    }
}

pub fn run(widths: &[usize], dep: &Deployment, bit_width: u32) -> Result<LayerReport, String> {
    let spec_fhe = fhe_cost::find("bfv").ok_or("bfv missing from the FHE registry")?;
    let mut mpc = Vec::new();
    let mut fhe = Vec::new();

    for &width in widths {
        let src = layer_src(width);
        let inputs = layer_inputs(width);
        let want = plain(width, &inputs);

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
            let sched = schedule(&prog);
            let m = run_protocol(id, dep.parties, &prog, &sched, &inputs)?;

            // Wall-clock from the measured counts and the profile.
            let secs = m.rounds as f64 * dep.latency_us / 1e6
                + (m.online_bytes + m.offline_bytes) as f64 / dep.bandwidth_bytes_per_sec;
            mpc.push(LayerPoint {
                protocol: (*id).to_string(),
                width,
                rounds: m.rounds,
                online_bytes: m.online_bytes,
                triples: m.triples,
                offline_bytes: m.offline_bytes,
                triples_generated: m.triples_generated,
                secs,
                correct: m.outputs.first().map(|(_, v)| *v) == Some(want),
            });
        }
        // The hybrid's MPC half: the ReLU alone, measured under the cheapest
        // protocol available, since that is what a deployment would use for it.
        let relu = {
            let mut best: Option<LayerPoint> = None;
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
                let prog = compile(&relu_src(), &cfg)?;
                let sched = schedule(&prog);
                let mut ri: HashMap<String, i64> = HashMap::new();
                ri.insert("x".to_string(), 5);
                let m = run_protocol(id, dep.parties, &prog, &sched, &ri)?;
                let secs = m.rounds as f64 * dep.latency_us / 1e6
                    + (m.online_bytes + m.offline_bytes) as f64 / dep.bandwidth_bytes_per_sec;
                let p = LayerPoint {
                    protocol: (*id).to_string(),
                    width: 1,
                    rounds: m.rounds,
                    online_bytes: m.online_bytes,
                    triples: m.triples,
                    offline_bytes: m.offline_bytes,
                    triples_generated: m.triples_generated,
                    secs,
                    correct: m.outputs.first().map(|(_, v)| *v) == Some(5),
                };
                if best.as_ref().map(|b| p.secs < b.secs).unwrap_or(true) {
                    best = Some(p);
                }
            }
            best.ok_or("no protocol could run the ReLU")?
        };
        fhe.push(fhe_layer(width, spec_fhe, dep, &relu));
    }

    // First width where the hybrid undercuts every *deployable* MPC protocol.
    //
    // Dealer is excluded on purpose: it assumes a trusted third party hands out
    // triples for free, which is not a threat model but a lower bound. Including
    // it would mean asking "does FHE beat free preprocessing?", to which the
    // answer is trivially no and tells you nothing.
    let mut crossover = None;
    for &width in widths {
        let best_mpc = mpc
            .iter()
            .filter(|p| p.width == width && p.protocol != "dealer")
            .map(|p| p.secs)
            .fold(f64::INFINITY, f64::min);
        let f = fhe.iter().find(|p| p.width == width).map(|p| p.secs);
        if let Some(f) = f {
            if f < best_mpc && crossover.is_none() {
                crossover = Some(width);
            }
        }
    }

    Ok(LayerReport {
        widths: widths.to_vec(),
        mpc,
        fhe,
        latency_us: dep.latency_us,
        bandwidth_bytes_per_sec: dep.bandwidth_bytes_per_sec,
        crossover,
    })
}

fn human(bytes: u64) -> String {
    if bytes >= 1 << 20 {
        format!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn secs(s: f64) -> String {
    if s >= 1.0 {
        format!("{s:.2} s")
    } else {
        format!("{:.1} ms", s * 1000.0)
    }
}

const USAGE: &str = "usage: baru bench-layer [options]

The MPC<->FHE crossover for one private linear layer with BOTH operands secret
(encrypted weights), followed by a ReLU.  MPC costs are MEASURED by running the
protocols; the FHE side uses measured per-operation costs from
bench-fhe-innerprod; wall-clock is computed from those counts and a network
profile.

options:
  --widths N,N,...  layer widths to sweep       (default 8,64,256)
  --parties N       party count                 (default 2)
  --lan | --wan     network profile             (default wan)
  --json            machine-readable output
  -h, --help        this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut widths = vec![8usize, 64, 256];
    let mut parties = 2usize;
    let mut wan = true;
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--widths" => {
                let v = it.next().ok_or("--widths needs a value")?;
                widths = v
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad width {s}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--parties" => {
                parties = it
                    .next()
                    .ok_or("--parties needs a value")?
                    .parse()
                    .map_err(|_| "bad --parties")?;
            }
            "--lan" => wan = false,
            "--wan" => wan = true,
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let mut dep = if wan {
        Deployment::wan(parties)
    } else {
        Deployment::lan(parties)
    };
    dep.allow_dealer = true;
    let r = run(&widths, &dep, 32)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("One private linear layer (both operands secret) + ReLU — measured MPC vs FHE");
    println!(
        "{} parties, {} ({:.0} ms one-way, {:.0} Mbit/s)\n",
        parties,
        if wan { "WAN" } else { "LAN" },
        r.latency_us / 1000.0,
        r.bandwidth_bytes_per_sec * 8.0 / 1e6
    );
    println!(
        "{:>8} {:>10} {:>7} {:>8} {:>11} {:>11} {:>10} {:>4}",
        "width", "protocol", "rounds", "triples", "online", "offline", "time", "ok"
    );
    for &w in &r.widths {
        for p in r.mpc.iter().filter(|p| p.width == w) {
            println!(
                "{:>8} {:>10} {:>7} {:>8} {:>11} {:>11} {:>10} {:>4}",
                p.width,
                p.protocol,
                p.rounds,
                p.triples,
                human(p.online_bytes),
                human(p.offline_bytes),
                secs(p.secs),
                if p.correct { "✓" } else { "✗" }
            );
        }
        if let Some(f) = r.fhe.iter().find(|p| p.width == w) {
            println!(
                "{:>8} {:>10} {:>7} {:>8} {:>11} {:>11} {:>10} {:>4}",
                w,
                format!("hybrid/{}", f.scheme),
                2 + f.relu_rounds,
                f.relu_triples,
                human(f.switch_bytes),
                "0 B",
                secs(f.secs),
                "-"
            );
        }
    }
    println!();
    match r.crossover {
        Some(w) => println!(
            "The hybrid undercuts every DEPLOYABLE MPC protocol from width {w} onward, and the\n\
             mechanism is the offline column: real OT preprocessing grows with width while a\n\
             packed ciphertext does not."
        ),
        None => println!(
            "No deployable MPC protocol was beaten by the hybrid at these widths — so at this\n\
             scale the preprocessing bill has not yet overtaken a ciphertext."
        ),
    }
    println!(
        "\n`dealer` is excluded from that comparison and shown only as a LOWER BOUND: it\n\
         assumes a trusted party hands out triples for free, which is not a threat model.\n\
         That it stays flat and cheapest at every width is itself the result — the entire\n\
         case for FHE here is the cost of *preprocessing*, not of computation."
    );
    println!();
    println!("MEASURED by running the protocol: rounds, triples, online bytes, and Semi2k's");
    println!("offline bytes (real OT-generated triples).  The FHE row uses measured per-op");
    println!("costs from bench-fhe-innerprod.  Wall-clock is COMPUTED from those counts and");
    println!("the profile above — there is no real network in the loop (roadmap P1).");

    if r.mpc.iter().any(|p| !p.correct) {
        return Err("a protocol computed the wrong layer output".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every protocol must compute the layer correctly — the benchmark is a
    /// correctness test as well as a cost measurement.
    #[test]
    fn every_protocol_computes_the_layer() {
        let mut dep = Deployment::wan(2);
        dep.allow_dealer = true;
        let r = run(&[8], &dep, 32).unwrap();
        assert!(!r.mpc.is_empty(), "no protocol ran");
        assert!(r.mpc.iter().all(|p| p.correct), "{:?}", r.mpc);
    }

    /// **The measured crossover.** The hybrid must overtake the deployable
    /// dishonest-majority protocol as the layer widens, and `dealer` must be
    /// excluded from that judgement — it assumes free preprocessing, so including
    /// it asks "does FHE beat a trusted third party?", which is not the question.
    #[test]
    fn the_hybrid_overtakes_semi2k_as_the_layer_widens() {
        let mut dep = Deployment::wan(2);
        dep.allow_dealer = true;
        let r = run(&[8, 1024], &dep, 32).unwrap();
        let pick = |proto: &str, w: usize| -> f64 {
            r.mpc
                .iter()
                .find(|p| p.protocol == proto && p.width == w)
                .map(|p| p.secs)
                .unwrap_or_else(|| panic!("no {proto} at {w}"))
        };
        let hybrid =
            |w: usize| -> f64 { r.fhe.iter().find(|p| p.width == w).map(|p| p.secs).unwrap() };

        // Narrow: Semi2k still wins.
        assert!(
            pick("semi2k", 8) < hybrid(8),
            "at width 8 Semi2k should still win: {:.1} vs {:.1} ms",
            pick("semi2k", 8) * 1e3,
            hybrid(8) * 1e3
        );
        // Wide: the triple bill has overtaken the ciphertext.
        assert!(
            hybrid(1024) < pick("semi2k", 1024),
            "at width 1024 the hybrid should win: {:.1} vs {:.1} ms",
            hybrid(1024) * 1e3,
            pick("semi2k", 1024) * 1e3
        );
        assert!(r.crossover.is_some(), "a crossover should have been found");

        // Dealer is a lower bound and must stay flat — that is what shows the
        // whole effect comes from preprocessing rather than computation.
        let (d8, d1024) = (pick("dealer", 8), pick("dealer", 1024));
        assert!(
            (d1024 - d8).abs() / d8 < 0.05,
            "dealer should be ~flat in width: {:.1} vs {:.1} ms",
            d8 * 1e3,
            d1024 * 1e3
        );
    }

    /// The mechanism behind the crossover, measured rather than modelled: Semi2k's
    /// **offline** bytes must grow with layer width, because each multiplication
    /// consumes a real OT-generated triple. Dealer's must not — it is handed
    /// triples for free, which is what makes it a lower bound.
    #[test]
    fn semi2k_offline_grows_with_width_and_dealers_does_not() {
        let mut dep = Deployment::wan(2);
        dep.allow_dealer = true;
        let r = run(&[8, 64], &dep, 32).unwrap();
        let get = |proto: &str, w: usize| -> LayerPoint {
            r.mpc
                .iter()
                .find(|p| p.protocol == proto && p.width == w)
                .cloned()
                .unwrap_or_else(|| panic!("no {proto} at width {w}"))
        };
        let (s8, s64) = (get("semi2k", 8), get("semi2k", 64));
        assert!(
            s64.offline_bytes > s8.offline_bytes,
            "semi2k offline should grow with width: {} -> {}",
            s8.offline_bytes,
            s64.offline_bytes
        );
        assert!(s64.triples > s8.triples, "more width, more triples");

        let d64 = get("dealer", 64);
        assert_eq!(
            d64.offline_bytes, 0,
            "a trusted dealer moves no preprocessing — that is the assumption"
        );
        assert!(d64.triples > 0, "but it still consumes triples");
    }
}
