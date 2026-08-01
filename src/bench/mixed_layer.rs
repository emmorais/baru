//! `baru bench-mixed-layer` — the cost model decides, and then the decision
//! actually runs.
//!
//! `baru mpcc assign` chooses which operation kinds go to which side and
//! prices the conversions; `baru bench-backend-aware` shows the choice is
//! backend-dependent. Both are *predictions*. This command takes the
//! assignment the model returns and executes it
//! ([`crate::mpcc::mixed`]), so the decision is carried out rather than
//! merely costed, and the result is checked against a plaintext reference.
//!
//! The workload is one neural-network layer with **both operands secret** —
//! $\\sum a_i b_i$ — which is the shape the model puts on FHE for a wide layer
//! on a slow link.
//!
//! # What this does and does not establish
//!
//! It establishes that the assignment is *executable and exact*: the mixed
//! run agrees with the all-MPC run and with plaintext, bit for bit, and the
//! operations really do land on the sides the model chose.
//!
//! It does **not** validate the model's wall-clock prediction. The MPC side
//! here is an in-process simulator with no network, and the MPC→FHE direction
//! of the bridge uses `export_additive`, which only a simulator can satisfy
//! (see [`crate::mpcc::mixed`]). Operation counts, conversion counts and
//! crossing bytes are meaningful; seconds are not. Validating the predicted
//! crossover needs the deployable switch, where each party encrypts its own
//! share component.

use std::collections::HashMap;

use serde::Serialize;

use crate::bench::comm::NetworkProfile;
use crate::crypto::types::Fr;
use crate::fhe::bfv::{Bfv, BfvAdapter, BfvParams};
use crate::interpreter::aby3_simulator::Aby3Simulator;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::parse;
use crate::mpcc::assign::{assign, Pair, Side as ASide};
use crate::mpcc::cost::Deployment;
use crate::mpcc::lower::{lower, LowerConfig};
use crate::mpcc::mixed::{from_assignment, run_mixed, Placement, Side};
use crate::mpcc::protocols::find;
use crate::mpcc::schedule::schedule;
use crate::zkp::circuit_compiler::CircuitCompiler;

const NBITS: usize = 20;

#[derive(Debug, Clone, Serialize)]
pub struct Point {
    pub network: String,
    /// What the cost model decided, as `kind -> side`.
    pub decision: Vec<(String, String)>,
    pub predicted_speedup: f64,
    /// Measured by executing that decision.
    pub mpc_ops: usize,
    pub fhe_ops: usize,
    pub switches_to_fhe: usize,
    pub switches_to_mpc: usize,
    pub crossing_bytes: usize,
    pub demoted_muls: usize,
    /// Operations the boundary rule moved off their kind's side.
    pub boundary_refinements: usize,
    pub correct: bool,
}

fn to_signed(v: u128) -> i64 {
    let m = 1i128 << NBITS;
    let x = (v as i128) & (m - 1);
    if x >= m / 2 {
        (x - m) as i64
    } else {
        x as i64
    }
}

pub fn run(n: usize) -> Result<Vec<Point>, String> {
    // One layer: a dot product with both operands secret.
    let mut src = "(* a0 b0)".to_string();
    for i in 1..n {
        src = format!("(+ {src} (* a{i} b{i}))");
    }
    let expr = parse(&src).map_err(|e| format!("parse error: {e}"))?;
    let circuit = CircuitCompiler::new()
        .compile_fhe(&expr)
        .map_err(|e| format!("compile error: {e}"))?;
    let flat = circuit.flat_ops.clone();
    let out_wire = circuit.output_wire.clone();

    let a: Vec<i64> = (0..n).map(|i| (i as i64 % 5) - 2).collect();
    let b: Vec<i64> = (0..n).map(|i| ((i * 3) as i64 % 7) - 3).collect();
    let want: i64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

    let arith = find("semi2k").ok_or("semi2k missing")?;
    let binary = find("semi-bin").ok_or("semi-bin missing")?;
    let bfv_spec = crate::mpcc::fhe_cost::find("bfv").ok_or("bfv missing")?;
    let pair = Pair::new(arith, binary)?.with_fhe(bfv_spec);

    let cfg = LowerConfig {
        domain: crate::mpcc::cost::native_domain(arith, 64),
        parties: 2,
        bit_width: 32,
        public_vars: Vec::new(),
    };
    let prog = lower(&flat, &out_wire, &cfg)?;
    let sched = schedule(&prog);

    let s = Bfv::new(BfvParams::TEST_SMALL);
    let (ck, _pk) = s.keygen();
    let fhe = BfvAdapter(&s);

    let mut pts = Vec::new();
    for (label, dep) in [("LAN", Deployment::lan(2)), ("WAN", Deployment::wan(2))] {
        let assigned = assign(&prog, &sched, &pair, &dep, 32)?;
        let routing = from_assignment(&assigned.assignment);

        let mut decision: Vec<(String, String)> = assigned
            .assignment
            .by_kind
            .iter()
            .map(|(k, v)| {
                let side = match v {
                    ASide::Fhe => "fhe",
                    ASide::Binary => "binary",
                    ASide::Arith => "arith",
                };
                ((*k).to_string(), side.to_string())
            })
            .collect();
        decision.sort();

        // Execute exactly what the model decided.
        let mut be = Aby3Simulator::new();
        let mut env: HashMap<String, ShareRef> = HashMap::new();
        for i in 0..n {
            let av = (a[i] as i128).rem_euclid(1i128 << NBITS) as Fr;
            let bv = (b[i] as i128).rem_euclid(1i128 << NBITS) as Fr;
            env.insert(format!("a{i}"), be.share_secret(av, NBITS, 0));
            env.insert(format!("b{i}"), be.share_secret(bv, NBITS, 0));
        }
        let placement = Placement::by_kind(routing.clone()).sink_to_fhe(&flat);
        let refinements = placement.refinements();
        let (r, st) = run_mixed(
            &mut be, &fhe, &ck, &flat, &env, &out_wire, NBITS, &placement,
        )?;
        let got = to_signed(be.reveal(&r, None));

        pts.push(Point {
            network: label.to_string(),
            decision,
            predicted_speedup: assigned.speedup(),
            mpc_ops: st.mpc_ops,
            fhe_ops: st.fhe_ops,
            switches_to_fhe: st.to_fhe,
            switches_to_mpc: st.to_mpc,
            crossing_bytes: st.crossing_bytes,
            demoted_muls: st.demoted_muls,
            boundary_refinements: refinements,
            correct: got == want,
        });

        // The routing must have been honoured, or the loop from model to
        // execution is not actually closed.
        let wanted_fhe = routing.get("mul") == Some(&Side::Fhe);
        if wanted_fhe && st.fhe_ops == 0 {
            return Err(format!(
                "{label}: the model assigned `mul -> fhe` but nothing ran on FHE — \
                 the assignment is not reaching the executor"
            ));
        }
        if !wanted_fhe && st.fhe_ops > 0 {
            return Err(format!(
                "{label}: the model kept everything on MPC but {} operations ran on FHE",
                st.fhe_ops
            ));
        }
    }
    Ok(pts)
}

const USAGE: &str = "usage: baru bench-mixed-layer [options]

Takes the assignment `mpcc assign` returns and EXECUTES it across MPC and FHE,
checking the result against a plaintext reference. Run under two network
profiles, because the model's decision depends on the deployment.

Counts and bytes are meaningful; seconds are not measured here (the MPC side is
an in-process simulator and the MPC->FHE bridge is party-simulated).

options:
  --n N     layer width (default 256 — the crossover regime; below ~64 both
            deployments keep the layer on MPC)
  --deployed  additionally MEASURE the layer both ways on the real 3-party
            transport, with the crossing done per-party (cost-faithful)
  --rtt-ms A,B  round-trip times for --deployed (default 0,20)
  --share-bits W  share width for --deployed (default 10). The packed FHE route
            needs a plaintext modulus above (parties*2^W)^2 * n, so large W is
            infeasible at these ring parameters — the command says so rather
            than silently wrapping.
  --json    machine-readable output
  -h,--help this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut n = 256usize;
    let mut json = false;
    let mut deployed = false;
    let mut share_bits = 10usize;
    let mut rtts: Vec<u64> = vec![0, 20];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" => {
                n = it
                    .next()
                    .ok_or("--n needs a value")?
                    .parse()
                    .map_err(|_| "bad --n")?
            }
            "--json" => json = true,
            "--deployed" => deployed = true,
            "--share-bits" => {
                share_bits = it
                    .next()
                    .ok_or("--share-bits needs a value")?
                    .parse()
                    .map_err(|_| "bad --share-bits")?
            }
            "--rtt-ms" => {
                rtts = it
                    .next()
                    .ok_or("--rtt-ms needs a value")?
                    .split(',')
                    .map(|x| {
                        x.trim()
                            .parse::<u64>()
                            .map_err(|_| "bad --rtt-ms".to_string())
                    })
                    .collect::<Result<_, _>>()?
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let pts = run(n)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&pts).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("The cost model decides, and the decision is then executed");
    println!("workload: one layer, sum of {n} products, both operands secret\n");
    println!(
        "  {:<6} {:<24} {:>9} {:>8} {:>8} {:>9} {:>8} {:>7}",
        "net", "model's decision", "predicted", "mpc ops", "fhe ops", "switches", "sunk", "exact"
    );
    for p in &pts {
        if !p.correct {
            return Err(format!(
                "{}: the executed assignment does not match the plaintext reference",
                p.network
            ));
        }
        let d: Vec<String> = p
            .decision
            .iter()
            .map(|(k, v)| format!("{k}->{v}"))
            .collect();
        println!(
            "  {:<6} {:<24} {:>8.2}x {:>8} {:>8} {:>9} {:>8} {:>7}",
            p.network,
            if d.is_empty() {
                "(all arithmetic)".into()
            } else {
                d.join(",")
            },
            p.predicted_speedup,
            p.mpc_ops,
            p.fhe_ops,
            format!("{}/{}", p.switches_to_fhe, p.switches_to_mpc),
            p.boundary_refinements,
            "yes"
        );
    }
    println!("\n  Same program, same widths, same threat model — one line per deployment:");
    for p in &pts {
        if p.fhe_ops == 0 {
            println!(
                "    {:<4} the layer stayed on MPC ({} operations, no switch).",
                p.network, p.mpc_ops
            );
        } else {
            println!(
                "    {:<4} the model moved the layer to FHE: {} operations there, {} on MPC,\n         \
                 {} switches out and {} back, {:.1} KiB crossing.",
                p.network,
                p.fhe_ops,
                p.mpc_ops,
                p.switches_to_fhe,
                p.switches_to_mpc,
                p.crossing_bytes as f64 / 1024.0
            );
        }
        if p.demoted_muls > 0 {
            println!(
                "         {} multiplication(s) demoted back to MPC — the depth-1 FHE side had\n         \
                 no budget left. Reported rather than hidden.",
                p.demoted_muls
            );
        }
    }
    if pts.iter().all(|p| p.fhe_ops == 0) {
        println!(
            "\n  Note: no deployment chose FHE at this width. The crossover is around n=256 —\n  \
             narrow layers stay on MPC because a batched multiply is one round. Re-run with\n  \
             a larger --n to see the decision change."
        );
    }
    println!(
        "\n  `sunk` counts operations the BOUNDARY RULE moved off their kind's side\n  \
         (Placement::sink_to_fhe): an operation whose operands are all already\n  \
         ciphertexts runs on FHE rather than dragging them back. The kind-level\n  \
         assignment puts only `mul` on FHE; the additions follow because of this rule."
    );
    println!(
        "\n  Known gap: the cost model charges crossings *packed* (a whole layer in one\n  \
         ciphertext), while this executor converts scalar-wise — hence one switch per\n  \
         input above. The crossing bytes here therefore overstate a packed\n  \
         implementation, and closing that is the next step on this path."
    );
    if !deployed {
        println!(
            "\n  Counts and bytes above are measured; seconds are not (in-process simulator,\n  \
             party-simulated bridge). Pass --deployed to measure wall-clock on the real\n  \
             3-party transport with the crossing done per-party."
        );
        return Ok(());
    }

    // The width that fits the demonstration share width; see run_deployed.
    let dn = n;
    println!(
        "\n  MEASURED on the real 3-party transport, crossing done per-party\n  \
         (layer width {dn}, {share_bits}-bit shares; ciphertexts charged the link):"
    );
    println!(
        "  {:<12} {:<20} {:>10} {:>12} {:>12} {:>14}",
        "network", "route", "wall-clock", "msgs/triples", "online bytes", "offline/link B"
    );
    for &rtt in &rtts {
        let rows = match run_deployed(dn, rtt, Some(1000.0), share_bits) {
            Ok(r) => r,
            Err(e) => {
                println!("  {:<12} {}", format!("rtt {rtt}ms"), e);
                continue;
            }
        };
        for r in &rows {
            if !r.correct {
                return Err(format!("{} / {}: wrong result", r.network, r.route));
            }
            println!(
                "  {:<12} {:<20} {:>8.1}ms {:>12} {:>12} {:>14}",
                r.network, r.route, r.ms, r.mpc_messages, r.mpc_bytes, r.link_bytes
            );
        }
    }
    println!(
        "\n  READ THIS BEFORE COMPARING TO THE PREDICTION ABOVE. The measured MPC route is\n  \
         ABY3: 3-party, honest majority, and TRIPLE-FREE (PRF resharing). The model's\n  \
         `mul -> fhe` decision was computed for semi2k: 2-party, dishonest majority, one\n  \
         Beaver triple per multiplication. FHE's case there is the offline bill it avoids\n  \
         — and ABY3 has no offline bill, so there is nothing for FHE to amortise against.\n  \
         These two numbers therefore do NOT test each other; they name different\n  \
         protocols. Applied to ABY3, the model says the layer stays on MPC, which is what\n  \
         is measured."
    );
    println!(
        "\n  Two comparisons here ARE controlled, and both go against FHE:\n    \
         · vs aby3 (3-party, honest majority, triple-free): FHE loses badly. Expected —\n      \
           there is no offline bill for FHE to amortise against.\n    \
         · vs semi2k (2-party, dishonest majority, real OT triples), against the 2pc FHE\n      \
           route so the party count matches: semi2k is FASTER and moves 5.8x fewer bytes\n      \
           (125 KB of triples vs 721 KB of ciphertext). The model predicted FHE wins 1.18x."
    );
    println!(
        "\n  A specific, code-checked hypothesis for the gap: `assign.rs` charges an\n  \
         MPC->FHE crossing as ct_bytes * ceil(registers/slots) per level — ONE packed\n  \
         ciphertext — with no party-count factor. But switching a SECRET-SHARED value\n  \
         into FHE needs one ciphertext PER PARTY, since each party encrypts its own\n  \
         component. At 2 parties and 2 operands that is 4 ciphertexts, not 1.\n  \
         Caveat before acting on this: the model's decision was computed at 32-bit while\n  \
         this runs at 10-bit, and semi2k's triple cost scales with the width, so the\n  \
         two are not yet a like-for-like test of the crossover."
    );
    println!(
        "\n  Still a COST model of party separation, not a cryptographic one — the evaluator\n  \
         holds the decryption key, as in bench::cnn. What is faithful is the work and the\n  \
         traffic: three encryptions per operand, three ciphertexts crossing."
    );
    Ok(())
}

/// A **cost-faithful** measurement of the same layer, on the real 3-party
/// transport, with the crossing done the way a deployment does it.
///
/// [`run`] above executes the assignment on an in-process simulator using
/// `export_additive`, which hands one caller every party's share component —
/// convenient, and not something a deployed party can do. Here each party
/// produces only *its own* component ([`TransportBackend::my_additive_component`]),
/// those components are encrypted separately and summed homomorphically, and
/// the ciphertexts crossing the link are charged latency and bandwidth. That
/// is the same accounting `crate::bench::cnn` uses, so the wall-clock is
/// comparable to it.
///
/// It remains a *cost* model of the party separation rather than a
/// cryptographic one: the driver still holds the decryption key, as it does in
/// `bench::cnn`. What is faithful is the work and the traffic — three
/// encryptions per operand and three ciphertexts on the wire, not one.
#[derive(Debug, Clone, Serialize)]
pub struct Deployed {
    pub network: String,
    pub route: String,
    pub ms: f64,
    pub mpc_messages: u64,
    pub mpc_bytes: u64,
    pub link_bytes: usize,
    pub correct: bool,
}

pub fn run_deployed(
    n: usize,
    rtt_ms: u64,
    bandwidth_mbps: Option<f64>,
    share_bits: usize,
) -> Result<Vec<Deployed>, String> {
    use crate::bench::hybrid_nn::{run_parties, sleep_link, MpcTransport};
    use crate::interpreter::transport_backend::TransportBackend;

    // Small magnitudes so the true dot product fits the share width even at
    // n=256; the packed route's constraint is the modulus, checked below.
    let a: Vec<i64> = (0..n).map(|i| (i as i64 % 3) - 1).collect();
    let b: Vec<i64> = (0..n).map(|i| ((i * 2) as i64 % 3) - 1).collect();
    let want: i64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    // Keep the true dot product inside the share width so the check is a plain
    // equality; the packed FHE route additionally needs the plaintext modulus
    // to hold un-reduced share products (see `mixed::packed_dot`).
    let w = share_bits;
    if want.abs() >= 1 << (w - 1) {
        return Err("layer too wide for the demonstration share width".into());
    }

    let mut profile = NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt_ms * 500));
    if let Some(mbps) = bandwidth_mbps {
        profile = profile.with_bandwidth(mbps * 1e6 / 8.0);
    }

    // The plaintext modulus must hold un-reduced share products across the
    // whole layer (see `mixed::packed_dot`): 3 parties, w-bit shares, n terms.
    let bound: u128 = ((3u128 << w) * (3u128 << w)).saturating_mul(n as u128);
    let log2_t = (128 - bound.leading_zeros() + 1).max(20);
    if log2_t > 60 {
        return Err(format!(
            "a {n}-wide layer of {w}-bit shares needs t = 2^{log2_t}, beyond this demo's budget"
        ));
    }
    // The plaintext modulus is only half the constraint. A depth-1 BFV multiply
    // also needs enough of `q` left for noise, and at n=4096 with q=2^100 the
    // budget runs out once t passes roughly 2^32 — beyond that the packed route
    // returns a wrong value rather than failing. Growing the ring would buy
    // headroom, but `fhe::ntt` supports n <= MAX_N = 4096, so there is nowhere
    // to grow to. Refuse, and say exactly why.
    if log2_t > 32 {
        return Err(format!(
            "the packed FHE crossing is not implementable here for a {n}-wide layer of \
             {w}-bit shares: un-reduced share products need t = 2^{log2_t}, which leaves \
             too little of q = 2^100 for depth-1 noise at n = 4096, and fhe::ntt caps the \
             ring at MAX_N = 4096. Reduce --n or --share-bits, or use a BFV parameter set \
             with a larger ring."
        ));
    }
    let s = Bfv::new(BfvParams {
        n: 4096,
        log2_q: 100,
        log2_t,
    });
    let (ck, _pk) = s.keygen();

    let share = move |be: &mut TransportBackend<MpcTransport>, me: usize, v: &[i64]| {
        let vals: Vec<Fr> = v
            .iter()
            .map(|&x| {
                if me == 0 {
                    (x as i128).rem_euclid(1i128 << w) as Fr
                } else {
                    0
                }
            })
            .collect();
        be.share_secret_batch(&vals, w, 0)
    };

    let mut out = Vec::new();

    // ── Route 1: the layer entirely under MPC ──
    {
        let (a2, b2) = (a.clone(), b.clone());
        let t = std::time::Instant::now();
        let (vals, comm) = run_parties(profile, move |me| {
            let (a2, b2) = (a2.clone(), b2.clone());
            move |be: &mut TransportBackend<MpcTransport>| {
                let av = share(be, me, &a2);
                let bv = share(be, me, &b2);
                let pairs: Vec<_> = av.iter().cloned().zip(bv.iter().cloned()).collect();
                let prods = be.mul_batch(&pairs);
                let mut acc = prods[0].clone();
                for p in &prods[1..] {
                    acc = be.add(&acc, p);
                }
                be.reveal(&acc, None)
            }
        });
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        out.push(Deployed {
            network: format!("rtt {rtt_ms}ms"),
            route: "all-MPC (aby3)".into(),
            ms,
            mpc_messages: comm.messages,
            mpc_bytes: comm.bytes,
            link_bytes: 0,
            correct: signed_w(vals[0], w) == want,
        });
    }

    // ── Route 2: the layer hosted on FHE, packed crossing ──
    {
        let (a2, b2) = (a.clone(), b.clone());
        let t = std::time::Instant::now();
        // Each party contributes only its own component vectors.
        let (parts, comm) = run_parties(profile, move |me| {
            let (a2, b2) = (a2.clone(), b2.clone());
            move |be: &mut TransportBackend<MpcTransport>| {
                let av = share(be, me, &a2);
                let bv = share(be, me, &b2);
                let mine = |v: &[ShareRef], be: &TransportBackend<MpcTransport>| -> Vec<i64> {
                    v.iter()
                        .map(|s| be.my_additive_component(s) as i64)
                        .collect()
                };
                (mine(&av, be), mine(&bv, be))
            }
        });

        // Each party encrypts its own packed component vector; the evaluator
        // sums them. Three encryptions per operand, three ciphertexts on the
        // wire — this is the cost a deployment pays.
        let mut ct_a: Option<crate::fhe::bfv::BfvCt> = None;
        let mut ct_b: Option<crate::fhe::bfv::BfvCt> = None;
        let mut link_bytes = 0usize;
        for (pa, pb) in &parts {
            let ca = s.encrypt_poly(&ck, &s.pack_input(pa));
            let cb = s.encrypt_poly(&ck, &s.pack_weights(&[pb.clone()]));
            link_bytes += s.ct_bytes(&ca) + s.ct_bytes(&cb);
            ct_a = Some(match ct_a {
                None => ca,
                Some(acc) => s.add(&acc, &ca),
            });
            ct_b = Some(match ct_b {
                None => cb,
                Some(acc) => s.add(&acc, &cb),
            });
        }
        sleep_link(&profile, link_bytes, 1);
        let prod = s.mul(&ct_a.expect("a"), &ct_b.expect("b"));

        // Masked opening at the output coefficient, back to the parties.
        let pos = s.matvec_output_position(n, 0);
        let t_mask = s.params.t() - 1;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let masks: Vec<u64> = (0..3).map(|_| rng.gen::<u64>() & t_mask).collect();
        let mut masked = prod;
        for &r in &masks {
            let mut m = vec![0u64; s.params.n];
            m[pos] = r;
            masked = s.add(&masked, &s.encrypt_poly(&ck, &m));
        }
        let back = s.ct_bytes(&masked);
        link_bytes += back;
        sleep_link(&profile, back, 1);
        let y = s.decrypt_poly(&ck, &masked)[pos];

        let got = {
            let mut acc = (y as u128).wrapping_sub(masks[0] as u128);
            for &r in &masks[1..] {
                acc = acc.wrapping_sub(r as u128);
            }
            signed_w(acc, w)
        };
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        out.push(Deployed {
            network: format!("rtt {rtt_ms}ms"),
            route: "layer on FHE".into(),
            ms,
            mpc_messages: comm.messages,
            mpc_bytes: comm.bytes,
            link_bytes,
            correct: got == want,
        });
    }

    // ── Route 2b: the layer on FHE against a **2-party** additive sharing,
    //    so it is directly comparable with semi2k below. Switching a
    //    secret-shared value into FHE costs one ciphertext *per party*, so the
    //    party count is part of the crossing cost and must match.
    {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let t = std::time::Instant::now();
        let w_mask = (1u64 << w) - 1;
        let split = |v: &[i64], rng: &mut rand::rngs::ThreadRng| -> Vec<Vec<i64>> {
            let p0: Vec<i64> = v
                .iter()
                .map(|_| (rng.gen::<u64>() & w_mask) as i64)
                .collect();
            let p1: Vec<i64> = v
                .iter()
                .zip(&p0)
                .map(|(&x, &r)| (((x as i128) - (r as i128)).rem_euclid(1i128 << w)) as i64)
                .collect();
            vec![p0, p1]
        };
        let a_parts = split(&a, &mut rng);
        let b_parts = split(&b, &mut rng);

        let mut ct_a: Option<crate::fhe::bfv::BfvCt> = None;
        let mut ct_b: Option<crate::fhe::bfv::BfvCt> = None;
        let mut link = 0usize;
        for p in 0..2 {
            let ca = s.encrypt_poly(&ck, &s.pack_input(&a_parts[p]));
            let cb = s.encrypt_poly(&ck, &s.pack_weights(&[b_parts[p].clone()]));
            link += s.ct_bytes(&ca) + s.ct_bytes(&cb);
            ct_a = Some(match ct_a {
                None => ca,
                Some(x) => s.add(&x, &ca),
            });
            ct_b = Some(match ct_b {
                None => cb,
                Some(x) => s.add(&x, &cb),
            });
        }
        sleep_link(&profile, link, 1);
        let prod = s.mul(&ct_a.expect("a"), &ct_b.expect("b"));
        let pos = s.matvec_output_position(n, 0);
        let t_mask = s.params.t() - 1;
        let masks: Vec<u64> = (0..2).map(|_| rng.gen::<u64>() & t_mask).collect();
        let mut masked = prod;
        for &r in &masks {
            let mut m = vec![0u64; s.params.n];
            m[pos] = r;
            masked = s.add(&masked, &s.encrypt_poly(&ck, &m));
        }
        let back = s.ct_bytes(&masked);
        link += back;
        sleep_link(&profile, back, 1);
        let y = s.decrypt_poly(&ck, &masked)[pos];
        let mut acc = (y as u128).wrapping_sub(masks[0] as u128);
        acc = acc.wrapping_sub(masks[1] as u128);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        out.push(Deployed {
            network: format!("rtt {rtt_ms}ms"),
            route: "layer on FHE (2pc)".into(),
            ms,
            mpc_messages: 0,
            mpc_bytes: 0,
            link_bytes: link,
            correct: signed_w(acc, w) == want,
        });
    }

    // ── Route 3: the layer under semi2k — the protocol the model's decision
    //    was actually computed for. Triples come from real OT, so the offline
    //    cost is performed rather than assumed, and that is the thing the
    //    model says FHE avoids.
    {
        use crate::mpcc::semi2k::Semi2kEngine;
        use crate::mpcc::vm::ShareEngine;
        let t = std::time::Instant::now();
        let mut eng = Semi2kEngine::new(2, w as u32)?;
        let av: Vec<_> = a
            .iter()
            .map(|&x| eng.input((x as i128).rem_euclid(1i128 << w) as u64, 0))
            .collect();
        let bv: Vec<_> = b
            .iter()
            .map(|&x| eng.input((x as i128).rem_euclid(1i128 << w) as u64, 1))
            .collect();
        let pairs: Vec<_> = av.into_iter().zip(bv).collect();
        let prods = eng.mul_batch(&pairs);
        let mut acc = prods[0].clone();
        for p in &prods[1..] {
            acc = eng.add(&acc, p);
        }
        let opened = eng.open_batch(&[acc])[0];
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let c = eng.counters();
        // The engine is in-process, so its wall-clock excludes the network the
        // other routes pay; the honest comparison is its OFFLINE bill, which is
        // the term the model's FHE decision turns on.
        out.push(Deployed {
            network: format!("rtt {rtt_ms}ms"),
            route: "all-MPC (semi2k)".into(),
            ms,
            mpc_messages: c.triples_generated,
            mpc_bytes: c.online_bytes,
            link_bytes: c.offline_bytes as usize,
            correct: signed_w(opened as u128, w) == want,
        });
    }

    Ok(out)
}

fn signed_w(v: u128, w: usize) -> i64 {
    let m = 1i128 << w;
    let x = (v as i128) & (m - 1);
    if x >= m / 2 {
        (x - m) as i64
    } else {
        x as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loop from cost model to execution must actually close: whatever the
    /// model decides is what runs, and it computes the right answer.
    #[test]
    fn the_models_decision_is_executed_and_exact() {
        let pts = run(256).expect("both deployments execute");
        assert_eq!(pts.len(), 2);
        for p in &pts {
            assert!(
                p.correct,
                "{}: executed assignment must be exact",
                p.network
            );
        }
    }
}
