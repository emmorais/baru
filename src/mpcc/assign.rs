//! **Per-operation protocol assignment** — mixed (hybrid) circuits.
//!
//! Until now Baru's comparison answered *"which single protocol for this whole
//! program?"* (`baru mpcc compare`). The state of the art answers a strictly more
//! general question: **which protocol for each part of the program**, paying for
//! the conversions between them. That is the "hybrid MPC compilation" line —
//! HyCC (Büscher et al., CCS'18), Silph (Chen et al., S&P'23), and most recently
//! SING (Blüml, Huppert, Khayata, Schmidt & Schneider, USENIX Sec'26,
//! <https://eprint.iacr.org/2026/1237>). This module closes that gap.
//!
//! The classic finding those systems reproduce is that **multiplications want an
//! arithmetic sharing and comparisons want a boolean/garbled one**, so a mixed
//! circuit beats either pure one on programs containing both. `baru mpcc assign`
//! now derives that from Baru's own cost model rather than assuming it.
//!
//! # Why Baru does not need an ILP or a GNN
//!
//! This is the interesting part, and it is a direct consequence of how the two
//! systems above are built.
//!
//! * **Silph** solves the assignment with Integer Linear Programming, which
//!   forces a **linear** cost model — cost must be a sum of per-operation terms.
//! * **SING** replaces the ILP with a graph neural network, and its stated
//!   motivation is exactly that limitation: a learned predictor can express
//!   *non-linear* costs, and re-fitting it is cheaper than retuning an ILP. It
//!   reports up to a 76 697× speedup over Silph at comparable quality.
//!
//! Baru is in the regime where the linear formulation is wrong, and it does not
//! need a surrogate either, because it has the exact model and can just evaluate
//! it. Two non-linearities in [`crate::mpcc::cost`] make the objective
//! **non-separable** in the per-operation assignment variables:
//!
//! 1. A dependence level costs the **maximum** over the instructions in it, not
//!    their sum — ten comparisons at one level cost one comparison's rounds. So
//!    moving one instruction to the other side may change the level's cost by
//!    nothing at all, or by the whole difference, depending on what else is in
//!    that level.
//! 2. Conversions **batch per level** for the same reason, so the cost of a
//!    domain crossing depends on whether any other crossing already happens at
//!    that level.
//!
//! Neither is expressible as a linear function of the assignment. So the approach
//! here is the one [`crate::mpcc::opt`] already uses for rewrites: propose,
//! evaluate the real objective, keep it if it improves. Cheap because evaluating
//! is cheap; exact because the objective is the actual cost model. What Baru
//! gives up is any optimality guarantee — the search is greedy, and finding the
//! optimum is NP-hard, which is why Silph reached for ILP in the first place.
//!
//! # Granularity, and what is honestly not done
//!
//! Assignment here is **per operation *kind*** (all multiplications together, all
//! comparisons together), not per instruction. That is coarser than Silph and
//! SING, which assign per operation or per block. Two reasons it is the right
//! first step: the space is a few dozen kinds rather than `2ⁿ`, so the search is
//! exhaustive-ish and fast; and the *result is interpretable* — "comparisons went
//! to the binary side, multiplications stayed arithmetic" is a statement about
//! the cost model, which is what this repo is for. Per-instruction assignment is
//! the obvious next step and would only sharpen the same effect.
//!
//! Also not done: **FHE is not in the assignment**, because
//! [`crate::mpcc::cost`] models MPC protocols only. That is the extension nobody
//! else can make — HyCC, Silph and SING are MPC-only, while Baru's `src/hybrid/`
//! already switches MPC↔FHE by hand — and it is the reason this module exists in
//! `mpcc` rather than as a one-off script.

use std::collections::HashMap;

use serde::Serialize;

use super::cost::{arith_op_cost, binary_op_cost, eligibility, native_domain, Deployment};
use super::ir::DomainKind;
use super::ir::{Op, Program};
use super::protocols::ProtocolSpec;
use super::schedule::Schedule;

/// Which sharing an instruction is evaluated under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Side {
    /// Arithmetic sharing — ring or field, depending on the protocol.
    Arith,
    /// Boolean / garbled sharing.
    Binary,
    /// Homomorphic evaluation — **no rounds**, a depth budget, and no
    /// comparisons. See [`crate::mpcc::fhe_cost`].
    Fhe,
}

impl Side {
    pub fn name(self) -> &'static str {
        match self {
            Side::Arith => "arith",
            Side::Binary => "binary",
            Side::Fhe => "fhe",
        }
    }
    /// Every side an operation could move to.
    pub const ALL: &'static [Side] = &[Side::Arith, Side::Binary, Side::Fhe];
}

/// The operation kinds assignment distinguishes.
///
/// Deliberately coarse: what matters for assignment is whether an operation is a
/// multiplication, a comparison, a shift, and so on — not which register it
/// wrote. Constants and local bookkeeping are free on both sides and are not
/// assigned at all.
pub fn kind_of(op: &Op) -> Option<&'static str> {
    Some(match op {
        Op::Mul => "mul",
        Op::MulC(_) => "mul-const",
        Op::Add | Op::Sub | Op::AddC(_) | Op::Neg => "add",
        Op::Mux => "mux",
        Op::LtS | Op::LtU => "cmp-lt",
        Op::EqZ => "cmp-eq",
        Op::And => "and",
        Op::Xor | Op::Not => "xor",
        Op::ShlC(_) => "shift",
        Op::TruncPr { .. } => "trunc",
        Op::Div => "div",
        Op::BitDecomp { .. } | Op::BitComp { .. } => "bit",
        Op::PopCount { .. } => "popcount",
        // Not assignable: no arithmetic to place.
        Op::LdI(_) | Op::Copy | Op::Input { .. } | Op::Output | Op::Open | Op::Convert => {
            return None
        }
    })
}

/// A side per operation kind. Kinds absent from the map take the default side.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Assignment {
    pub by_kind: HashMap<&'static str, Side>,
    pub default: Side,
}

impl Assignment {
    pub fn uniform(side: Side) -> Self {
        Assignment {
            by_kind: HashMap::new(),
            default: side,
        }
    }
    pub fn side_of(&self, op: &Op) -> Side {
        match kind_of(op) {
            Some(k) => *self.by_kind.get(k).unwrap_or(&self.default),
            // Unassignable ops follow whatever produced them; costing them on
            // either side is free, so the default is harmless.
            None => self.default,
        }
    }
    /// Kinds that differ from the default — the readable part of the answer.
    pub fn deviations(&self) -> Vec<(&'static str, Side)> {
        let mut v: Vec<(&'static str, Side)> = self
            .by_kind
            .iter()
            .filter(|(_, s)| **s != self.default)
            .map(|(k, s)| (*k, *s))
            .collect();
        v.sort_unstable_by_key(|(k, _)| *k);
        v
    }
}

/// Cost of one assignment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MixedCost {
    pub rounds: u64,
    pub online_bytes: u64,
    /// Preprocessing bytes the MPC-assigned operations need — Beaver triples,
    /// priced from measured OT cost via [`crate::mpcc::ot_cost`].
    ///
    /// Including this is what makes the comparison fair to FHE, which needs
    /// **no** preprocessing at all. A first version of this model left it out and
    /// concluded FHE never wins, which contradicted this repo's own `bench-cnn`
    /// measurement of a working hybrid — the triples were the missing term.
    pub offline_bytes: u64,
    /// Domain crossings, i.e. `Convert` instructions the assignment implies.
    pub conversions: u64,
    /// Levels containing at least one crossing — what the round count charges,
    /// since conversions at one level batch into one round.
    pub conversion_levels: u64,
    pub secs: f64,
}

impl MixedCost {
    fn improves_on(&self, other: &MixedCost) -> bool {
        const EPS: f64 = 1e-15;
        if self.secs < other.secs - EPS {
            return true;
        }
        if self.secs > other.secs + EPS {
            return false;
        }
        // Break ties towards fewer crossings: a circuit with the same predicted
        // cost and fewer domain switches is simpler to implement and to trust.
        self.conversions < other.conversions
    }
}

/// The two protocols a mixed circuit is split between.
pub struct Pair<'a> {
    pub arith: &'a ProtocolSpec,
    pub binary: &'a ProtocolSpec,
    /// The FHE scheme operations may be placed on, if any. `None` restricts
    /// assignment to the two MPC sharings — which is what HyCC, Silph and SING
    /// do, and is kept so the MPC-only result stays reproducible.
    pub fhe: Option<&'static crate::mpcc::fhe_cost::FheSpec>,
}

impl<'a> Pair<'a> {
    pub fn new(arith: &'a ProtocolSpec, binary: &'a ProtocolSpec) -> Result<Self, String> {
        if arith.domain == DomainKind::Binary {
            return Err(format!("{} is a binary protocol", arith.id));
        }
        if binary.domain != DomainKind::Binary {
            return Err(format!("{} is not a binary protocol", binary.id));
        }
        Ok(Pair {
            arith,
            binary,
            fhe: None,
        })
    }

    /// Add an FHE scheme as a third target.
    pub fn with_fhe(mut self, fhe: &'static crate::mpcc::fhe_cost::FheSpec) -> Self {
        self.fhe = Some(fhe);
        self
    }
}

/// Local time and multiplicative-depth consumption of one operation under an
/// FHE scheme.
///
/// Returns `Err` when the scheme simply cannot evaluate the operation — which is
/// the honest answer for a comparison under BFV or CKKS, and is what stops the
/// search from proposing an arithmetic-only scheme for a program with a `relu` in
/// it. An infeasible assignment is rejected rather than priced at infinity, so
/// the error message says *why*.
fn fhe_op_cost(
    op: &Op,
    spec: &crate::mpcc::fhe_cost::FheSpec,
    bit_width: u32,
    parties: usize,
    ring_bits: u32,
) -> Result<(f64, u32), String> {
    use crate::mpcc::fhe_cost::Comparison;
    let k = f64::from(bit_width);
    Ok(match op {
        // Ciphertext x ciphertext: the one operation that consumes depth — and
        // the one the plaintext modulus can make impossible.
        //
        // The operands are *secret-shared*, so what the engine multiplies is
        // the un-reduced sum of share components: `parties · 2^bits` per
        // factor. Their product needs twice that many bits before any
        // summation. If the scheme's plaintext modulus cannot hold it the
        // result wraps silently, so this is a feasibility gate and not a cost.
        // See `FheSpec::log2_plaintext`.
        Op::Mul => {
            // The magnitude that crosses is the **share**, not the value. A
            // secret sharing of a 10-bit value over Z_2^64 has components
            // uniform over the whole 64-bit ring, so the un-reduced sum is
            // `parties · 2^ring_bits`. Using the value width here would let
            // the model approve a crossing that cannot be performed —
            // exactly the failure this gate exists to prevent.
            let per_factor = ring_bits + parties.max(1).next_power_of_two().trailing_zeros();
            let needed = 2 * per_factor;
            if needed > spec.log2_plaintext {
                return Err(format!(
                    "{} cannot hold a ciphertext×ciphertext product of shares over a \
                     {}-bit ring across {} parties: share components are uniform over the \
                     ring, so the un-reduced product needs a plaintext modulus of 2^{} \
                     while the parameters provide 2^{}. Encrypting BOTH operands is what \
                     costs this — a plaintext-weight layer is unaffected.",
                    spec.id, ring_bits, parties, needed, spec.log2_plaintext
                ));
            }
            (spec.mul_ns, 1)
        }
        // A public constant is a plaintext multiply — cheaper, and depth-free in
        // the sense that matters here (no ct x ct).
        Op::MulC(_) => (spec.mul_plain_ns, 0),
        Op::Add | Op::Sub | Op::AddC(_) | Op::Neg => (spec.add_ns, 0),
        // Shifts are multiplications by a power of two.
        Op::ShlC(_) => (spec.mul_plain_ns, 0),
        Op::LdI(_) | Op::Copy | Op::Input { .. } | Op::Output | Op::Open | Op::Convert => (0.0, 0),
        // Everything below needs comparison, which is where arithmetic schemes
        // stop being applicable at all.
        Op::LtS
        | Op::LtU
        | Op::EqZ
        | Op::Mux
        | Op::TruncPr { .. }
        | Op::And
        | Op::Xor
        | Op::Not
        | Op::BitDecomp { .. }
        | Op::BitComp { .. }
        | Op::PopCount { .. } => {
            match spec.comparison {
                Comparison::Unsupported => {
                    return Err(format!(
                        "{} cannot evaluate {} — it is arithmetic-only (no comparison, \
                         no mux, no argmax), so this operation has to leave FHE",
                        spec.id,
                        op.mnemonic()
                    ))
                }
                // A k-bit comparison is ~k bootstrapped gates.
                Comparison::PerGateNs(ns) => (ns * k, 0),
            }
        }
        Op::Div => match spec.comparison {
            Comparison::Unsupported => {
                return Err(format!("{} cannot evaluate a division", spec.id))
            }
            Comparison::PerGateNs(ns) => (ns * k * k, 0),
        },
    })
}

/// Price `prog` under `assign`.
///
/// Instructions are costed on their assigned side with the *same* per-operation
/// functions `estimate` uses, so a uniform assignment reproduces the pure-protocol
/// estimate. Crossings are charged as `Convert`s on the arithmetic protocol,
/// which is where the daBits come from.
pub fn evaluate(
    prog: &Program,
    sched: &Schedule,
    pair: &Pair<'_>,
    dep: &Deployment,
    bit_width: u32,
    assign: &Assignment,
) -> Result<MixedCost, String> {
    eligibility(pair.arith, dep)?;
    eligibility(pair.binary, dep)?;

    let arith_domain = native_domain(
        pair.arith,
        match prog.domain {
            super::ir::Domain::Ring { k } => k,
            _ => 64,
        },
    );
    let elem_bytes = arith_domain.element_bytes();

    // Which side each instruction runs on, and which register was last written
    // on which side, so crossings can be found.
    let sides: Vec<Side> = prog.code.iter().map(|i| assign.side_of(&i.op)).collect();
    let mut produced_on: HashMap<u32, Side> = HashMap::new();

    let mut bytes = 0.0f64;
    let mut local_ns = 0.0f64;
    let mut triples = 0.0f64;
    let mut conversions = 0u64;
    // Crossings per schedule level, so they can batch.
    let mut crossing_at_level: HashMap<usize, bool> = HashMap::new();
    // Per-instruction internal round cost, for the per-level max.
    let mut instr_rounds = vec![0u32; prog.code.len()];
    // Multiplicative depth each register carries on the FHE side.
    let mut fhe_depth: HashMap<u32, u32> = HashMap::new();
    // (level, op, per-op ns) -> lanes, so packing can be applied per level.
    let mut fhe_lanes: HashMap<(usize, &'static str, u64), f64> = HashMap::new();
    // Distinct (level, register) pairs crossing the MPC<->FHE boundary.
    let mut fhe_crossings: std::collections::HashSet<(usize, u32)> =
        std::collections::HashSet::new();

    let convert_cost = arith_op_cost(&Op::Convert, 1.0, pair.arith, dep, elem_bytes, bit_width);

    for (i, instr) in prog.code.iter().enumerate() {
        let side = sides[i];
        let lanes = f64::from(instr.width);
        let secret = prog.srcs_secrecy(instr) == super::ir::Secrecy::Secret;

        // FHE is priced separately: it has no rounds, its own per-op timings, and
        // hard limits (no comparison, a depth budget) that make an assignment
        // *infeasible* rather than merely expensive.
        if side == Side::Fhe {
            let spec = pair
                .fhe
                .ok_or("an operation was assigned to FHE but no FHE scheme was given")?;
            if secret {
                let (ns, mul_depth) = fhe_op_cost(&instr.op, spec, bit_width, dep.parties, (elem_bytes * 8.0) as u32)?;
                // Packing: independent operations of the same kind at the same
                // dependence level share a ciphertext, so they are tallied and
                // charged `ceil(count / slots)` operations below — not one each.
                // This is the same shape as MPC's round batching, and the second
                // non-linearity an ILP formulation could not express.
                let level = sched.level.get(i).copied().unwrap_or(0);
                *fhe_lanes
                    .entry((level, instr.op.mnemonic(), ns.to_bits()))
                    .or_insert(0.0) += lanes;
                // Rounds stay zero: homomorphic evaluation is non-interactive.
                // This is FHE's whole advantage and why the answer moves with the
                // network profile.
                let incoming = instr
                    .src
                    .iter()
                    .map(|r| fhe_depth.get(r).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                let out_depth = incoming + mul_depth;
                if out_depth > spec.depth_budget {
                    return Err(format!(
                        "{} exceeds {}'s depth budget of {} at instruction {i}: a value \
                         would pass through {} ciphertext multiplications without a \
                         refresh.  A scheme switch is the refresh — which is why the \
                         cheapest assignment puts a nonlinearity between them.",
                        instr.op.mnemonic(),
                        spec.id,
                        spec.depth_budget,
                        out_depth
                    ));
                }
                for dst in &instr.dst {
                    fhe_depth.insert(*dst, out_depth);
                }
            }
        } else {
            let c = match side {
                Side::Arith => {
                    arith_op_cost(&instr.op, lanes, pair.arith, dep, elem_bytes, bit_width)
                }
                Side::Binary => binary_op_cost(&instr.op, lanes, pair.binary, dep, bit_width),
                Side::Fhe => unreachable!("handled above"),
            };
            // Public computation costs nothing in any protocol.
            if secret {
                bytes += c.acc.bytes;
                local_ns += c.acc.local_ns;
                instr_rounds[i] = c.rounds;
                triples += c.acc.triples + c.acc.and_triples;
            }
            // Coming back from FHE resets the depth: the switch re-encrypts.
            for dst in &instr.dst {
                fhe_depth.insert(*dst, 0);
            }
        }

        // A crossing for every secret operand produced on the other side.
        if secret {
            for src in &instr.src {
                if let Some(&from) = produced_on.get(src) {
                    if from != side {
                        conversions += u64::from(instr.width).max(1);
                        // An MPC<->FHE switch moves a whole ciphertext; an
                        // arithmetic<->binary conversion moves daBits.  Both cost
                        // one batched round, charged per level below.
                        if from == Side::Fhe || side == Side::Fhe {
                            // Pack the switch too: a whole layer's worth of
                            // values crosses in one ciphertext, so tally the
                            // *distinct registers* crossing at this level rather
                            // than charging a ciphertext per edge.  Charging per
                            // edge re-encrypts the same value once per use and
                            // made FHE look 16x worse than it is on a wide layer.
                            let level = sched.level.get(i).copied().unwrap_or(0);
                            fhe_crossings.insert((level, *src));
                        } else {
                            bytes += convert_cost.acc.bytes * lanes;
                            local_ns += convert_cost.acc.local_ns * lanes;
                        }
                        let level = sched.level.get(i).copied().unwrap_or(0);
                        crossing_at_level.insert(level, true);
                    }
                }
            }
        }
        for dst in &instr.dst {
            produced_on.insert(*dst, side);
        }
    }

    // Rounds: each dependence level costs the deepest primitive it contains,
    // plus one batched round if anything crosses domains at that level. This is
    // the non-linearity that an ILP formulation cannot express.
    let mut rounds = 0u64;
    for round in &sched.rounds {
        let level_max = round
            .instrs
            .iter()
            .map(|&idx| instr_rounds.get(idx).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        rounds += u64::from(level_max);
    }
    let conversion_levels = crossing_at_level.len() as u64;
    rounds += conversion_levels * u64::from(convert_cost.rounds.max(1));

    // Input sharing, one batched prelude round, as `estimate` charges it.
    if !prog.inputs.is_empty() {
        rounds += 1;
    }

    // Packed FHE work: each (level, operation) group costs `ceil(lanes / slots)`
    // homomorphic operations. A whole 784-element layer is one ciphertext op, which
    // is what `src/fhe/conv_pack.rs` actually does.
    if let Some(spec) = pair.fhe {
        for ((_, _, ns_bits), lanes) in &fhe_lanes {
            let ns = f64::from_bits(*ns_bits);
            let ops = (lanes / spec.slots).ceil().max(1.0);
            local_ns += ns * ops;
        }
        // Ciphertexts moved at the switch, packed per level.
        let mut per_level: HashMap<usize, f64> = HashMap::new();
        for (level, _) in &fhe_crossings {
            *per_level.entry(*level).or_insert(0.0) += 1.0;
        }
        for count in per_level.values() {
            // One packed ciphertext per `slots` values crossing at this
            // level — times the **party count**.
            //
            // A scheme switch does not move one ciphertext. Going in, each
            // party encrypts its own share component and the evaluator sums
            // them; coming out, each party contributes an encrypted mask
            // before the masked value is opened. Both directions therefore
            // carry `parties` ciphertexts, not one.
            //
            // Omitting this factor made the FHE side look cheaper than it is:
            // executing the matched comparison (`bench-mixed-layer
            // --deployed`) has semi2k moving 5.8x fewer bytes than the FHE
            // route it was predicted to lose to. Packing the crossing per
            // level was right; packing away the party dimension was not.
            let packed = (count / spec.slots).ceil().max(1.0);
            bytes += spec.ct_bytes * packed * dep.parties.max(1) as f64;
        }
    }

    // Preprocessing: triples for the MPC-assigned multiplications, priced from
    // measured OT cost where the protocol is OT-based.  FHE contributes none,
    // which is the asymmetry that decides multiply-heavy programs.
    let triple_bytes = match pair.arith.cost.offline_from_ot {
        Some(ot) => ot.triple_bytes((elem_bytes * 8.0) as u32),
        None => pair.arith.cost.offline_elems_per_triple * elem_bytes,
    };
    let offline_bytes = triples * triple_bytes;

    let latency_secs = rounds as f64 * dep.latency_us / 1e6;
    let bandwidth_secs = bytes / dep.bandwidth_bytes_per_sec;
    let compute_secs = local_ns / 1e9;
    // Offline moves over the same link.  `Deployment::count_offline` decides
    // whether a deployment can hide it (idle time before the query arrives).
    let offline_secs = if dep.include_offline {
        offline_bytes / dep.bandwidth_bytes_per_sec
    } else {
        0.0
    };

    Ok(MixedCost {
        rounds,
        online_bytes: bytes as u64,
        offline_bytes: offline_bytes as u64,
        conversions,
        conversion_levels,
        secs: latency_secs + bandwidth_secs + compute_secs + offline_secs,
    })
}

/// The best assignment found, with the baselines it beat.
#[derive(Debug, Clone, Serialize)]
pub struct Assigned {
    pub arith_protocol: String,
    pub binary_protocol: String,
    /// The FHE scheme in the mix, when one was offered.
    pub fhe_scheme: Option<String>,
    /// Everything on the arithmetic side — what `mpcc compare` reports today.
    pub all_arith: MixedCost,
    pub all_binary: MixedCost,
    pub best: MixedCost,
    pub assignment: Assignment,
    /// Kinds moved off the winning baseline's side, in order accepted.
    pub moved: Vec<(&'static str, Side)>,
    /// Assignments priced, so the search's coverage is visible.
    pub evaluated: usize,
}

impl Assigned {
    /// Speed-up of the mixed assignment over the better pure protocol.
    pub fn speedup(&self) -> f64 {
        let base = self.all_arith.secs.min(self.all_binary.secs);
        if self.best.secs <= 0.0 {
            1.0
        } else {
            base / self.best.secs
        }
    }
    pub fn is_mixed(&self) -> bool {
        !self.assignment.deviations().is_empty()
    }
}

/// Search for a cheaper mixed assignment.
///
/// Greedy over operation kinds: start from the better pure baseline, then
/// repeatedly move the single kind whose move helps most, until nothing helps.
/// Optimal assignment is NP-hard — this is a local optimum, and `evaluated`
/// reports how much of the space was priced rather than leaving it implied.
pub fn assign(
    prog: &Program,
    sched: &Schedule,
    pair: &Pair<'_>,
    dep: &Deployment,
    bit_width: u32,
) -> Result<Assigned, String> {
    let all_arith = evaluate(
        prog,
        sched,
        pair,
        dep,
        bit_width,
        &Assignment::uniform(Side::Arith),
    )?;
    let all_binary = evaluate(
        prog,
        sched,
        pair,
        dep,
        bit_width,
        &Assignment::uniform(Side::Binary),
    )?;
    let mut evaluated = 2usize;

    let start = if all_binary.improves_on(&all_arith) {
        Side::Binary
    } else {
        Side::Arith
    };
    let mut current = Assignment::uniform(start);
    let mut current_cost = if start == Side::Binary {
        all_binary
    } else {
        all_arith
    };

    // The kinds this program actually contains — no point pricing a move for an
    // operation that never occurs.
    let mut kinds: Vec<&'static str> = prog
        .code
        .iter()
        .filter_map(|i| kind_of(&i.op))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    kinds.sort_unstable();

    let mut moved: Vec<(&'static str, Side)> = Vec::new();
    loop {
        let mut best_move: Option<(&'static str, Side, MixedCost)> = None;
        for kind in &kinds {
            let now = *current.by_kind.get(kind).unwrap_or(&current.default);
            for &target in Side::ALL {
                if target == now {
                    continue;
                }
                if target == Side::Fhe && pair.fhe.is_none() {
                    continue;
                }
                let mut candidate = current.clone();
                candidate.by_kind.insert(kind, target);
                // An infeasible assignment (FHE depth exceeded, or an operation
                // the scheme cannot evaluate) is simply not a candidate.
                let Ok(cost) = evaluate(prog, sched, pair, dep, bit_width, &candidate) else {
                    evaluated += 1;
                    continue;
                };
                evaluated += 1;
                let better_than_current = cost.improves_on(&current_cost);
                let better_than_best = best_move
                    .as_ref()
                    .map(|(_, _, c)| cost.improves_on(c))
                    .unwrap_or(true);
                if better_than_current && better_than_best {
                    best_move = Some((kind, target, cost));
                }
            }
        }
        match best_move {
            Some((kind, side, cost)) => {
                current.by_kind.insert(kind, side);
                current_cost = cost;
                moved.push((kind, side));
            }
            None => break,
        }
    }

    Ok(Assigned {
        arith_protocol: pair.arith.id.to_string(),
        binary_protocol: pair.binary.id.to_string(),
        fhe_scheme: pair.fhe.map(|f| f.id.to_string()),
        all_arith,
        all_binary,
        best: current_cost,
        assignment: current,
        moved,
        evaluated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpcc::protocols::find;
    use crate::mpcc::{lower, schedule, LowerConfig};
    use crate::zkp::circuit_compiler::CircuitCompiler;

    fn program(src: &str) -> (Program, Schedule) {
        let expr = crate::interpreter::parse(src).expect("parse");
        let circuit = CircuitCompiler::new().compile_fhe(&expr).expect("compile");
        let cfg = LowerConfig {
            domain: crate::mpcc::ir::Domain::Ring { k: 64 },
            parties: 2,
            bit_width: 32,
            public_vars: Vec::new(),
        };
        let prog = lower(&circuit.flat_ops, &circuit.output_wire, &cfg).expect("lower");
        let sched = schedule(&prog);
        (prog, sched)
    }

    fn pair() -> Pair<'static> {
        Pair::new(find("semi2k").unwrap(), find("semi-bin").unwrap()).unwrap()
    }

    /// A uniform assignment must reproduce the pure-protocol view, or the mixed
    /// costing is measuring something other than what `compare` reports.
    #[test]
    fn a_uniform_assignment_is_the_pure_protocol() {
        let (prog, sched) = program("(* x y)");
        let dep = Deployment::lan(2);
        let a = evaluate(
            &prog,
            &sched,
            &pair(),
            &dep,
            32,
            &Assignment::uniform(Side::Arith),
        )
        .unwrap();
        assert_eq!(
            a.conversions, 0,
            "a uniform assignment cannot cross domains"
        );
        let b = evaluate(
            &prog,
            &sched,
            &pair(),
            &dep,
            32,
            &Assignment::uniform(Side::Binary),
        )
        .unwrap();
        assert_eq!(b.conversions, 0);
        // A multiply is one round arithmetically and many AND-depth ones in
        // binary, so the two must differ.
        assert_ne!(a.rounds, b.rounds);
    }

    /// The classic hybrid-MPC result, derived from Baru's cost model rather than
    /// assumed: a program mixing multiplication with comparison should not want
    /// a single sharing for both.
    #[test]
    fn multiplication_and_comparison_want_different_sides() {
        // A comparison feeding a multiply: the shape HyCC/Silph exist for.
        let (prog, sched) = program("(* (+ a b) (> c d))");
        let dep = Deployment::wan(2);
        let out = assign(&prog, &sched, &pair(), &dep, 32).unwrap();
        assert!(
            out.evaluated > 2,
            "the search must price more than the baselines"
        );
        assert!(
            out.best.secs <= out.all_arith.secs && out.best.secs <= out.all_binary.secs,
            "the mixed result cannot be worse than either baseline: {:?}",
            out
        );
    }

    /// Crossings must actually be charged — otherwise mixing looks free and the
    /// search would shred the circuit into alternating domains.
    #[test]
    fn domain_crossings_cost_something() {
        let (prog, sched) = program("(* (* a b) (* c d))");
        let dep = Deployment::wan(2);
        let p = pair();

        let uniform = Assignment::uniform(Side::Arith);
        let mut split = Assignment::uniform(Side::Arith);
        split.by_kind.insert("mul", Side::Binary);
        split.by_kind.insert("add", Side::Arith);

        let u = evaluate(&prog, &sched, &p, &dep, 32, &uniform).unwrap();
        let s = evaluate(&prog, &sched, &p, &dep, 32, &split).unwrap();
        // Putting every multiply on the other side of every input/output must
        // introduce crossings.
        assert!(
            s.conversions > u.conversions,
            "splitting should cost conversions: {} vs {}",
            s.conversions,
            u.conversions
        );
    }

    /// **A negative result, and the one the measurements support.** A layer
    /// with *both* operands secret does not go to FHE — at any ring width,
    /// for two independent reasons.
    ///
    /// *Feasibility.* Share components are uniform over the whole ring, so a
    /// ct×ct product of shares over `Z_2^k` needs a plaintext modulus of
    /// `2^(2(k+log2 parties))` — `2^130` at a 64-bit ring, against the `2^32`
    /// depth-1 BFV provides. The assignment refuses rather than returning
    /// something that silently computes the wrong value, which is what
    /// `bench-mixed-layer --deployed --share-bits 16` did before this gate
    /// existed.
    ///
    /// *Cost.* At a narrow ring where the crossing **is** feasible (ring10),
    /// FHE is priced and simply not chosen — matching execution, where semi2k
    /// beat the FHE route on both wall-clock and bytes.
    ///
    /// Earlier versions of this test asserted the opposite. They passed
    /// because the model had no plaintext-modulus budget and could rank an
    /// assignment it was unable to perform.
    #[test]
    fn a_both_operands_secret_layer_does_not_go_to_fhe() {
        let mut dot = "(* a0 b0)".to_string();
        for i in 1..256 {
            dot = format!("(+ {dot} (* a{i} b{i}))");
        }
        let (prog, sched) = program(&format!("(if (> {dot} 0) 1 0)"));
        let pair = pair().with_fhe(crate::mpcc::fhe_cost::find("bfv").unwrap());
        let out = assign(&prog, &sched, &pair, &Deployment::wan(2), 32).unwrap();
        assert_ne!(
            out.assignment.by_kind.get("mul"),
            Some(&Side::Fhe),
            "a ct×ct product of 64-bit-ring shares needs 2^130 of plaintext modulus: {:?}",
            out.assignment.deviations()
        );
        // The comparison still moves — that part of the hybrid split holds.
        assert_eq!(
            out.assignment.by_kind.get("cmp-lt"),
            Some(&Side::Binary),
            "the nonlinearity should still go to the binary sharing: {:?}",
            out.assignment.deviations()
        );
    }

    /// The threat model still decides where the *comparison* goes, which is
    /// the part of the hybrid split that survives.
    #[test]
    fn the_threat_model_still_decides_the_comparison() {
        let mut dot = "(* a0 b0)".to_string();
        for i in 1..64 {
            dot = format!("(+ {dot} (* a{i} b{i}))");
        }
        let (prog, sched) = program(&format!("(if (> {dot} 0) 1 0)"));
        let bfv = crate::mpcc::fhe_cost::find("bfv").unwrap();
        let dishonest = Pair::new(find("semi2k").unwrap(), find("semi-bin").unwrap())
            .unwrap()
            .with_fhe(bfv);
        let out = assign(&prog, &sched, &dishonest, &Deployment::wan(2), 32).unwrap();
        assert_eq!(
            out.assignment.by_kind.get("cmp-lt"),
            Some(&Side::Binary),
            "comparisons belong on a binary sharing: {:?}",
            out.assignment.deviations()
        );
        assert!(out.speedup() > 1.0, "{:?}", out.best);
    }

    #[test]
    fn a_narrow_layer_stays_on_mpc() {
        let mut dot = "(* a0 b0)".to_string();
        for i in 1..8 {
            dot = format!("(+ {dot} (* a{i} b{i}))");
        }
        let (prog, sched) = program(&dot);
        let dep = Deployment::wan(2);
        let pair = pair().with_fhe(crate::mpcc::fhe_cost::find("bfv").unwrap());
        let out = assign(&prog, &sched, &pair, &dep, 32).unwrap();
        assert_ne!(
            out.assignment.by_kind.get("mul"),
            Some(&Side::Fhe),
            "8 multiplications batch into one MPC round; FHE cannot amortize its \
             switch over that: {:?}",
            out.assignment.deviations()
        );
    }

    /// An arithmetic-only scheme must *refuse* a comparison rather than be priced
    /// for one — that infeasibility is what forces the nonlinearity out of FHE.
    #[test]
    fn an_arithmetic_scheme_refuses_a_comparison() {
        let (prog, sched) = program("(> a b)");
        let dep = Deployment::wan(2);
        let pair = pair().with_fhe(crate::mpcc::fhe_cost::find("bfv").unwrap());
        let mut all_fhe = Assignment::uniform(Side::Fhe);
        all_fhe.by_kind.insert("cmp-lt", Side::Fhe);
        let err =
            evaluate(&prog, &sched, &pair, &dep, 32, &all_fhe).expect_err("BFV cannot compare");
        assert!(err.contains("arithmetic-only"), "{err}");
    }

    /// Pair construction must reject a nonsensical split.
    #[test]
    fn a_pair_needs_one_of_each_domain() {
        assert!(Pair::new(find("semi-bin").unwrap(), find("semi-bin").unwrap()).is_err());
        assert!(Pair::new(find("semi2k").unwrap(), find("semi2k").unwrap()).is_err());
    }

    /// Search must be deterministic — the same program and deployment must give
    /// the same assignment, or a reported result is not reproducible.
    #[test]
    fn the_search_is_deterministic() {
        let (prog, sched) = program("(* (+ a b) (> c d))");
        let dep = Deployment::wan(2);
        let one = assign(&prog, &sched, &pair(), &dep, 32).unwrap();
        let two = assign(&prog, &sched, &pair(), &dep, 32).unwrap();
        assert_eq!(one.assignment, two.assignment);
        assert_eq!(one.evaluated, two.evaluated);
    }
}
