//! The analytic cost model.
//!
//! Given a lowered [`Program`], its [`Schedule`], a [`ProtocolSpec`] and a
//! [`Deployment`], predict what running the program under that protocol would
//! cost — **without running it**.  That is what makes "which backend for *this*
//! program?" answerable for arbitrary programs rather than only for the ones in
//! a benchmark suite.
//!
//! ## How composite primitives are priced
//!
//! [`crate::mpcc::protocols`] only specifies the true primitives —
//! multiplication, opening, `GF(2)` AND, and preprocessing material.  Everything
//! else is *composed* here, in one place, from those primitives, so that adding
//! a protocol means supplying four numbers rather than thirty:
//!
//! | primitive | arithmetic protocol | binary protocol |
//! |---|---|---|
//! | `Mul` | 1 multiplication | `k²` AND gates (schoolbook) |
//! | `Add`/`Sub` | free | `k` AND gates (ripple-carry) |
//! | `LtS`/`EqZ` | edaBit + `log k` AND depth | `k` AND gates |
//! | `BitDecomp` | edaBit + a `k`-bit adder | free — already binary |
//! | `TruncPr` | mask, open, correct | **free** — it is wire selection |
//! | `Div` | `k` restoring-division steps | `k²` AND gates |
//!
//! The `TruncPr` row is the interesting one: right-shifting a secret is nearly
//! free in a garbled circuit and genuinely expensive in an arithmetic protocol.
//! Baru's Q16.16 fixed-point lowering emits a rescale after *every* fixed×fixed
//! multiplication, so this single asymmetry decides the MPC-vs-GC question for
//! most of Baru's neural-network workloads — which is exactly the kind of answer
//! the model exists to produce.
//!
//! ## Accuracy
//!
//! Online communication and round counts follow from the protocol definitions
//! and should be exact.  Offline volumes and the local-work constants are
//! literature estimates (see [`crate::mpcc::protocols`]); calibrating them
//! against measured runs is roadmap phase P1.  [`CostEstimate::notes`] carries
//! every caveat that applied to a particular estimate.

use serde::Serialize;

use super::ir::{Domain, DomainKind, Op, Program, Reg, Secrecy};
use super::protocols::{Majority, ProtocolSpec, Security};
use super::schedule::Schedule;
use super::vm::KAPPA;

/// The deployment a program is being costed for.
#[derive(Debug, Clone)]
pub struct Deployment {
    pub parties: usize,
    /// The weakest adversary the deployment must tolerate.
    pub min_security: Security,
    /// When true, honest-majority protocols are rejected.
    pub require_dishonest_majority: bool,
    /// When false, the dealer model is excluded (it is not a real threat model,
    /// only a lower bound).
    pub allow_dealer: bool,
    /// One-way network latency, microseconds.
    pub latency_us: f64,
    /// Per-party bandwidth, bytes per second.
    pub bandwidth_bytes_per_sec: f64,
    /// Include preprocessing in the headline time.
    pub include_offline: bool,
}

impl Deployment {
    /// 10 Gbit/s datacentre link, 0.1 ms round trip.
    pub fn lan(parties: usize) -> Self {
        Deployment {
            parties,
            min_security: Security::SemiHonest,
            require_dishonest_majority: false,
            allow_dealer: false,
            latency_us: 50.0,
            bandwidth_bytes_per_sec: 1.25e9,
            include_offline: true,
        }
    }

    /// 100 Mbit/s intercontinental link, 100 ms round trip — where round count
    /// dominates everything else.
    pub fn wan(parties: usize) -> Self {
        Deployment {
            latency_us: 50_000.0,
            bandwidth_bytes_per_sec: 12.5e6,
            ..Deployment::lan(parties)
        }
    }
}

/// What running `program` under one protocol is predicted to cost.
#[derive(Debug, Clone, Serialize)]
pub struct CostEstimate {
    pub protocol: String,
    pub implemented: bool,
    /// Communication rounds in the online phase, including input sharing.
    pub rounds: u64,
    /// Bytes one party sends online.
    pub online_bytes: u64,
    /// Bytes one party sends during preprocessing.
    pub offline_bytes: u64,
    pub triples: u64,
    pub and_triples: u64,
    /// Secret x secret multiplications.
    pub muls: u64,
    /// Shared random bits consumed by comparisons and bit decompositions.
    pub random_bits: u64,
    pub edabits: u64,
    pub dabits: u64,
    /// Boolean AND gates, for protocols that evaluate a binary circuit.
    pub and_gates: u64,
    /// Predicted online wall-clock, seconds.
    pub online_secs: f64,
    /// Predicted preprocessing wall-clock, seconds.
    pub offline_secs: f64,
    /// `online_secs` plus `offline_secs` when the deployment counts it.
    pub total_secs: f64,
    /// Which term dominates: `"latency"`, `"bandwidth"` or `"compute"`.
    pub bottleneck: &'static str,
    pub notes: Vec<String>,
}

/// Running totals while walking the program.
#[derive(Default, Clone, Copy)]
pub(crate) struct Acc {
    pub(crate) bytes: f64,
    pub(crate) triples: f64,
    pub(crate) and_triples: f64,
    pub(crate) muls: f64,
    pub(crate) random_bits: f64,
    pub(crate) edabits: f64,
    pub(crate) dabits: f64,
    pub(crate) and_gates: f64,
    pub(crate) local_ns: f64,
}

/// Per-instruction cost in one protocol.
#[derive(Default, Clone, Copy)]
pub(crate) struct OpCost {
    /// Internal rounds — a bit-decomposition is log-depth *inside* one
    /// dependence level, so a schedule level costs the max over its members.
    pub(crate) rounds: u32,
    pub(crate) acc: Acc,
}

/// Decide whether `spec` may be used for `dep`, and why not if it may not.
pub fn eligibility(spec: &ProtocolSpec, dep: &Deployment) -> Result<(), String> {
    if !spec.supports_parties(dep.parties) {
        let bound = match spec.max_parties {
            Some(m) if m == spec.min_parties => format!("exactly {}", m),
            Some(m) => format!("{}..={}", spec.min_parties, m),
            None => format!("{}+", spec.min_parties),
        };
        return Err(format!(
            "needs {} parties, deployment has {}",
            bound, dep.parties
        ));
    }
    if spec.security < dep.min_security {
        return Err(format!(
            "only {} secure, deployment requires {}",
            spec.security, dep.min_security
        ));
    }
    if spec.majority == Majority::Dealer && !dep.allow_dealer {
        return Err("requires a trusted dealer".to_string());
    }
    if dep.require_dishonest_majority && spec.majority == Majority::Honest {
        return Err("assumes an honest majority".to_string());
    }
    Ok(())
}

/// The domain a protocol runs in, independent of any particular program —
/// what [`mod@crate::mpcc::lower`] should target so a measured run and a predicted
/// one are comparable.
pub fn native_domain(spec: &ProtocolSpec, ring_bits: u32) -> Domain {
    match spec.domain {
        DomainKind::Ring => Domain::Ring { k: ring_bits },
        DomainKind::Field => Domain::Field {
            bits: spec.cost.field_bits,
        },
        DomainKind::Binary => Domain::Binary,
    }
}

/// The domain a protocol would actually run the program in.
fn protocol_domain(spec: &ProtocolSpec, prog: &Program) -> Domain {
    match spec.domain {
        DomainKind::Ring => match prog.domain {
            Domain::Ring { k } => Domain::Ring { k },
            _ => Domain::Ring { k: 64 },
        },
        // Each protocol declares its own prime size — honest-majority
        // protocols use a ~61-bit Mersenne prime, SPDZ-family ones a large
        // prime for MAC security.  Running a mod-2^k program mod p changes
        // wraparound semantics, which `notes` flags.
        DomainKind::Field => Domain::Field {
            bits: spec.cost.field_bits,
        },
        DomainKind::Binary => Domain::Binary,
    }
}

/// Multiplications one `nb`-bit decomposition costs.
///
/// Mirrors [`crate::mpcc::vm::bit_decompose_batch`] instruction for
/// instruction: each Kogge-Stone level combines `nb - d` positions at two
/// multiplications apiece, and a final layer XORs the borrows into the
/// difference bits.  Keeping the two in step is the whole point of `calibrate`.
fn bit_decompose_muls(nb: u32) -> f64 {
    let mut total = 0u64;
    let mut d = 1u32;
    while d < nb {
        total += 2 * u64::from(nb - d);
        d *= 2;
    }
    (total + u64::from(nb.saturating_sub(1))) as f64
}

fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Cost of one instruction under an *arithmetic* (ring or field) protocol.
pub(crate) fn arith_op_cost(
    op: &Op,
    lanes: f64,
    spec: &ProtocolSpec,
    dep: &Deployment,
    elem_bytes: f64,
    bit_width: u32,
) -> OpCost {
    let c = &spec.cost;
    let n = dep.parties;
    let mul_bytes = lanes * c.mul_elems.total(n) * elem_bytes * c.auth_overhead;
    let open_bytes = lanes * c.open_elems.total(n) * elem_bytes * c.auth_overhead;
    let and_bytes_per_lane = c.and_bits.total(n) / 8.0 * c.auth_overhead;
    let local = |lanes: f64| lanes * c.local_ns_per_lane;

    match op {
        Op::Mul | Op::Mux => OpCost {
            rounds: c.mul_rounds,
            acc: Acc {
                bytes: mul_bytes,
                triples: lanes * c.triples_per_mul,
                muls: lanes,
                local_ns: local(lanes),
                ..Acc::default()
            },
        },
        Op::Open => OpCost {
            rounds: c.open_rounds,
            acc: Acc {
                bytes: open_bytes,
                local_ns: local(lanes),
                ..Acc::default()
            },
        },
        Op::And => OpCost {
            rounds: c.and_rounds,
            acc: Acc {
                bytes: lanes * and_bytes_per_lane,
                and_triples: lanes * c.triples_per_mul,
                and_gates: lanes,
                local_ns: local(lanes),
                ..Acc::default()
            },
        },
        // Everything comparison-shaped is one masked opening plus a
        // Kogge-Stone borrow chain over *arithmetic* 0/1 shares -- there are no
        // GF(2) gates in this construction, which is why `and_gates` stays zero
        // for an arithmetic protocol.  `nb` is one bit wider than the value so
        // a difference still has a meaningful sign bit, matching the VM.
        Op::BitDecomp { .. } | Op::LtS | Op::LtU | Op::TruncPr { .. } => {
            let nb = bit_width + 1;
            let muls = lanes * bit_decompose_muls(nb);
            OpCost {
                rounds: c.open_rounds + (ceil_log2(nb) + 1) * c.mul_rounds,
                acc: Acc {
                    bytes: open_bytes + muls * c.mul_elems.total(n) * elem_bytes * c.auth_overhead,
                    triples: muls * c.triples_per_mul,
                    muls,
                    random_bits: lanes * f64::from(nb + KAPPA),
                    local_ns: local(muls),
                    ..Acc::default()
                },
            }
        }
        // A decomposition, then an AND tree over the complemented bits.
        Op::EqZ => {
            let nb = bit_width + 1;
            let muls = lanes * (bit_decompose_muls(nb) + f64::from(nb - 1));
            OpCost {
                rounds: c.open_rounds + (2 * ceil_log2(nb) + 1) * c.mul_rounds,
                acc: Acc {
                    bytes: open_bytes + muls * c.mul_elems.total(n) * elem_bytes * c.auth_overhead,
                    triples: muls * c.triples_per_mul,
                    muls,
                    random_bits: lanes * f64::from(nb + KAPPA),
                    local_ns: local(muls),
                    ..Acc::default()
                },
            }
        }
        Op::BitComp { .. } | Op::Convert => OpCost {
            rounds: c.open_rounds,
            acc: Acc {
                bytes: lanes * and_bytes_per_lane,
                dabits: lanes * c.dabits_per_bit,
                local_ns: local(lanes),
                ..Acc::default()
            },
        },
        // Restoring long division: one comparison and one multiplication per
        // quotient bit, serially -- exactly what `Backend::divmod` does.
        Op::Div => {
            let nb = bit_width + 1;
            let steps = f64::from(bit_width);
            let cmp_rounds = c.open_rounds + (ceil_log2(nb) + 1) * c.mul_rounds;
            let muls = steps * lanes * (bit_decompose_muls(nb) + 1.0);
            OpCost {
                rounds: bit_width * (cmp_rounds + c.mul_rounds),
                acc: Acc {
                    bytes: steps * open_bytes
                        + muls * c.mul_elems.total(n) * elem_bytes * c.auth_overhead,
                    triples: muls * c.triples_per_mul,
                    muls,
                    random_bits: steps * lanes * f64::from(nb + KAPPA),
                    local_ns: local(muls),
                    ..Acc::default()
                },
            }
        }
        Op::PopCount { nbits } => {
            let nb = bit_width + 1;
            let muls = lanes * (bit_decompose_muls(nb) + f64::from(*nbits));
            OpCost {
                rounds: c.open_rounds + (ceil_log2(nb) + 1) * c.mul_rounds,
                acc: Acc {
                    bytes: open_bytes + muls * c.mul_elems.total(n) * elem_bytes * c.auth_overhead,
                    triples: muls * c.triples_per_mul,
                    muls,
                    random_bits: lanes * f64::from(nb + KAPPA),
                    local_ns: local(muls),
                    ..Acc::default()
                },
            }
        }
        // Local: linear operations, public constants, register plumbing.
        // Input sharing is charged once, as a batched prelude round, in
        // `estimate` — it never appears in a schedule level.
        _ => OpCost::default(),
    }
}

/// Cost of one instruction under a *binary* protocol, which must evaluate the
/// arithmetic program as a boolean circuit.  Gate counts are the textbook
/// constructions: ripple-carry addition, schoolbook multiplication.
pub(crate) fn binary_op_cost(
    op: &Op,
    lanes: f64,
    spec: &ProtocolSpec,
    dep: &Deployment,
    bit_width: u32,
) -> OpCost {
    let c = &spec.cost;
    let k = f64::from(bit_width);
    let and_bytes = c.and_bits.total(dep.parties) / 8.0 * c.auth_overhead;
    // AND-gate count and AND-depth of the sub-circuit for one lane.
    let (gates, depth): (f64, u32) = match op {
        Op::Add | Op::Sub | Op::AddC(_) => (k, ceil_log2(bit_width)),
        Op::Mul => (k * k, 2 * ceil_log2(bit_width)),
        Op::MulC(_) => (k * k / 2.0, 2 * ceil_log2(bit_width)),
        Op::Mux => (k, 1),
        Op::LtS | Op::LtU | Op::EqZ => (k, ceil_log2(bit_width)),
        Op::And => (1.0, 1),
        Op::Div => (k * k, bit_width * ceil_log2(bit_width)),
        Op::PopCount { nbits } => (k * f64::from(ceil_log2(*nbits)), ceil_log2(*nbits)),
        // Free in a binary protocol: shifts and truncations are wire
        // relabelling, XOR is free-XOR, bit (de)composition is the identity.
        Op::Xor
        | Op::Not
        | Op::Neg
        | Op::ShlC(_)
        | Op::TruncPr { .. }
        | Op::BitDecomp { .. }
        | Op::BitComp { .. }
        | Op::Convert
        | Op::Copy
        | Op::LdI(_)
        | Op::Output
        | Op::Open
        | Op::Input { .. } => (0.0, 0),
    };
    let total_gates = lanes * gates;
    OpCost {
        rounds: if c.constant_round {
            0
        } else {
            depth * c.and_rounds
        },
        acc: Acc {
            bytes: total_gates * and_bytes,
            and_triples: total_gates * c.triples_per_mul,
            and_gates: total_gates,
            local_ns: total_gates * c.local_ns_per_lane,
            ..Acc::default()
        },
    }
}

/// Predict the cost of running `prog` under `spec` in `dep`.
///
/// Returns `Err` with a human-readable reason when the protocol cannot serve
/// the deployment at all.
pub fn estimate(
    prog: &Program,
    sched: &Schedule,
    spec: &ProtocolSpec,
    dep: &Deployment,
) -> Result<CostEstimate, String> {
    eligibility(spec, dep)?;

    let mut notes: Vec<String> = Vec::new();
    let pdomain = protocol_domain(spec, prog);
    let elem_bytes = pdomain.element_bytes();
    // The width the lowering used for comparisons and decompositions — not the
    // share width, which may be wider.
    let bit_width = prog.bit_width;

    if spec.domain != prog.domain.kind() {
        match spec.domain {
            DomainKind::Field => notes.push(format!(
                "program is {} but {} computes {}: wraparound semantics differ, \
                 and shares are {}-bit instead of {}-bit",
                prog.domain,
                spec.id,
                pdomain,
                pdomain.element_bits(),
                prog.domain.element_bits()
            )),
            DomainKind::Binary => notes.push(format!(
                "program is {} but {} is a binary protocol: costed as a boolean \
                 circuit with textbook gate counts ({}-bit ripple-carry adders, \
                 schoolbook multipliers)",
                prog.domain, spec.id, bit_width
            )),
            DomainKind::Ring => notes.push(format!(
                "program is {} but {} computes mod 2^k",
                prog.domain, spec.id
            )),
        }
    }

    let is_binary_protocol = spec.domain == DomainKind::Binary;
    let mut acc = Acc::default();
    let mut rounds: u64 = 0;

    // Input sharing, batched into a single prelude round.
    let n_inputs = prog.inputs.len() as f64;
    if n_inputs > 0.0 {
        if is_binary_protocol {
            // Oblivious transfer for the evaluator's input bits.
            acc.bytes += n_inputs * f64::from(bit_width) * 32.0;
        } else {
            acc.bytes += n_inputs
                * elem_bytes
                * (dep.parties.saturating_sub(1)) as f64
                * spec.cost.auth_overhead;
            rounds += 1;
        }
    }

    // Walk the schedule: every dependence level is one *logical* round, whose
    // real cost is the deepest primitive it contains.
    for round in &sched.rounds {
        let mut level_rounds = 0u32;
        for &idx in &round.instrs {
            let instr = &prog.code[idx];
            let lanes = f64::from(instr.width);
            let cost = if is_binary_protocol {
                binary_op_cost(&instr.op, lanes, spec, dep, bit_width)
            } else {
                arith_op_cost(&instr.op, lanes, spec, dep, elem_bytes, bit_width)
            };
            level_rounds = level_rounds.max(cost.rounds);
            acc.bytes += cost.acc.bytes;
            acc.triples += cost.acc.triples;
            acc.and_triples += cost.acc.and_triples;
            acc.muls += cost.acc.muls;
            acc.random_bits += cost.acc.random_bits;
            acc.edabits += cost.acc.edabits;
            acc.dabits += cost.acc.dabits;
            acc.and_gates += cost.acc.and_gates;
            acc.local_ns += cost.acc.local_ns;
        }
        rounds += u64::from(level_rounds);
    }

    // A binary protocol still has to evaluate the *local* arithmetic
    // instructions as gates — additions are free in a secret-sharing protocol
    // but cost a ripple-carry adder here.  Walk them too.
    if is_binary_protocol {
        // Their AND-depth has to be accumulated along the dependence DAG, not
        // maxed over the program: `is_interactive` is domain-agnostic, so a
        // chain of adders is "local" and every one of them inherits its
        // operands' schedule level. Taking a max would charge a chain of `n`
        // adders the depth of *one*, which silently under-prices exactly the
        // deep local arithmetic (fixed-point rescales, adder trees) that
        // decides MPC-vs-GC for the NN workloads. Longest path it is.
        let mut depth_of: Vec<u32> = vec![0; prog.code.len()];
        let mut producer: std::collections::HashMap<Reg, usize> = std::collections::HashMap::new();
        let mut extra_depth = 0u32;
        for (i, instr) in prog.code.iter().enumerate() {
            let mut incoming = 0u32;
            for src in &instr.src {
                if let Some(&j) = producer.get(src) {
                    incoming = incoming.max(depth_of[j]);
                }
            }
            let counted = !prog.is_interactive(instr) && prog.srcs_secrecy(instr) != Secrecy::Clear;
            let own = if counted {
                let cost = binary_op_cost(&instr.op, f64::from(instr.width), spec, dep, bit_width);
                acc.bytes += cost.acc.bytes;
                acc.and_triples += cost.acc.and_triples;
                acc.and_gates += cost.acc.and_gates;
                acc.local_ns += cost.acc.local_ns;
                cost.rounds
            } else {
                0 // already counted as an interactive round, or public
            };
            depth_of[i] = incoming + own;
            extra_depth = extra_depth.max(depth_of[i]);
            for dst in &instr.dst {
                producer.insert(*dst, i);
            }
        }
        if spec.cost.constant_round {
            rounds = u64::from(spec.cost.constant_rounds);
            notes.push(format!(
                "{} is constant-round: {} rounds regardless of circuit depth",
                spec.id, spec.cost.constant_rounds
            ));
        } else {
            rounds += u64::from(extra_depth);
        }
    }

    // Preprocessing volume.  For the OT-based protocols this is measured OT cost
    // times a protocol-level OT count (see `super::ot_cost`); for the rest it is
    // still a per-triple literature figure, and the notes below say which.
    // Gilboa's OLE needs one OT per bit of the *share*, not per bit of the
    // comparison width — a ring2^64 triple costs 64 OTs whatever `--bits` says.
    let share_bits = (elem_bytes * 8.0) as u32;
    let triple_bytes = match spec.cost.offline_from_ot {
        Some(ot) => ot.triple_bytes(share_bits),
        None => spec.cost.offline_elems_per_triple * elem_bytes,
    };
    // A random bit is **generated**, not dealt.  `vm::random_bits_batch` has each
    // party contribute a private bit and XOR-reduces them as a balanced tree, and
    // `x ⊕ y = x + y − 2xy`, so one bit costs `⌈log₂(parties)⌉` multiplications —
    // and therefore that many Beaver triples — plus the input sharing.
    //
    // This term used to be `acc.random_bits * elem_bytes`: one ring element per
    // bit.  That under-charged it by ~400× at two parties, where a bit really
    // costs one triple (≈3.2 kB of OT at 64-bit shares) rather than 8 bytes —
    // and it under-charged *exactly* the phase that decides a comparison-heavy
    // workload, since a 32-bit comparison draws `nb + KAPPA = 59` of them. The
    // measurement it is now fitted to is `the_cost_of_a_random_bit_is_a_triple`
    // in `mpcc::vm`.
    let bit_levels = f64::from(ceil_log2(dep.parties as u32).max(1));
    let random_bit_triples = acc.random_bits * bit_levels * spec.cost.triples_per_mul;
    let random_bit_bytes = acc.random_bits
        * (dep.parties as f64 * spec.cost.open_elems.total(dep.parties) * elem_bytes
            + bit_levels
                * spec.cost.mul_elems.total(dep.parties)
                * elem_bytes
                * spec.cost.auth_overhead);

    let offline_bytes = (acc.triples + acc.and_triples + random_bit_triples) * triple_bytes
        + (acc.edabits + acc.dabits) * elem_bytes
        + random_bit_bytes;

    let latency_secs = rounds as f64 * dep.latency_us / 1e6;
    let bandwidth_secs = acc.bytes / dep.bandwidth_bytes_per_sec;
    let compute_secs = acc.local_ns / 1e9;
    let online_secs = latency_secs + bandwidth_secs + compute_secs;
    let offline_secs = offline_bytes / dep.bandwidth_bytes_per_sec;

    let bottleneck = if latency_secs >= bandwidth_secs && latency_secs >= compute_secs {
        "latency"
    } else if bandwidth_secs >= compute_secs {
        "bandwidth"
    } else {
        "compute"
    };

    match spec.cost.offline_from_ot {
        Some(ot) => {
            if acc.triples + acc.and_triples > 0.0 {
                notes.push(format!(
                    "offline volume is derived from MEASURED OT cost ({:.1} B/OT, `baru bench-ot`) \
                     x {:.0} correlated OTs per triple; the offline *time* still ignores \
                     preprocessing CPU",
                    super::ot_cost::BYTES_PER_OT,
                    ot.triple_ots(share_bits),
                ));
            }
        }
        None => {
            if spec.cost.offline_elems_per_triple > 0.0 {
                notes.push(
                    "offline volume is a literature estimate, not a measurement (roadmap P1 \
                     calibrates it)"
                        .to_string(),
                );
            }
        }
    }
    if spec.id == "lowgear" {
        // This note used to say "LowGear trades bandwidth for CPU: its
        // preprocessing is compute-bound". `baru bench-triples` measured it and
        // the CPU half runs the *other* way — see `mpcc::lowgear`.
        notes.push(
            "LowGear's offline cost is measured (`baru bench-triples`): 192 B/triple at its \
             ~61-bit field vs Semi2k's 3074, and ~28x FASTER per triple, not slower — the \
             expected bandwidth-for-CPU trade did not appear at these parameters"
                .to_string(),
        );
        notes.push(
            "the implemented engine is semi-honest (no MACs, no ZKPoPK); Overdrive's malicious \
             guarantees are not provided"
                .to_string(),
        );
    }

    Ok(CostEstimate {
        protocol: spec.id.to_string(),
        implemented: spec.status.is_implemented(),
        rounds,
        online_bytes: acc.bytes as u64,
        offline_bytes: offline_bytes as u64,
        triples: acc.triples as u64,
        and_triples: acc.and_triples as u64,
        muls: acc.muls as u64,
        random_bits: acc.random_bits as u64,
        edabits: acc.edabits as u64,
        dabits: acc.dabits as u64,
        and_gates: acc.and_gates as u64,
        online_secs,
        offline_secs,
        total_secs: online_secs
            + if dep.include_offline {
                offline_secs
            } else {
                0.0
            },
        bottleneck,
        notes,
    })
}

#[cfg(test)]
mod tests {
    /// Replacing the hand-entered offline constants with a derivation from
    /// measured OT cost changed a *ranking*: BFV-based preprocessing (LowGear)
    /// moves far fewer bytes per triple than OT-based preprocessing. That is
    /// LowGear's whole selling point, and the old constants understated it — so
    /// it is worth a test rather than a comment.
    #[test]
    fn fhe_preprocessing_moves_far_less_than_ot_preprocessing() {
        let src = "(* (* x y) (* z w))";
        let mut dep = Deployment::lan(3);
        dep.require_dishonest_majority = true;
        let mut offline = std::collections::HashMap::new();
        for id in ["semi2k", "spdz2k", "mascot", "lowgear"] {
            let spec = crate::mpcc::protocols::find(id).unwrap();
            let cfg = crate::mpcc::LowerConfig {
                domain: native_domain(spec, 64),
                parties: 3,
                bit_width: 32,
                public_vars: Vec::new(),
            };
            let expr = crate::interpreter::parse(src).unwrap();
            let circuit = crate::zkp::circuit_compiler::CircuitCompiler::new()
                .compile_fhe(&expr)
                .unwrap();
            let prog = crate::mpcc::lower(&circuit.flat_ops, &circuit.output_wire, &cfg).unwrap();
            let sched = crate::mpcc::schedule(&prog);
            let est = estimate(&prog, &sched, spec, &dep).unwrap();
            offline.insert(id, est.offline_bytes);
        }
        let lowgear = offline["lowgear"];
        for id in ["semi2k", "spdz2k", "mascot"] {
            assert!(
                offline[id] > lowgear * 4,
                "{id} offline {} B should dwarf LowGear's {} B",
                offline[id],
                lowgear
            );
        }
    }

    use super::*;
    use crate::mpcc::lower::{lower, LowerConfig};
    use crate::mpcc::protocols::find;
    use crate::mpcc::schedule::schedule;

    fn compile(src: &str) -> (Program, Schedule) {
        let expr = crate::interpreter::parse(src).expect("parse");
        let mut cc = crate::zkp::circuit_compiler::CircuitCompiler::new();
        let circuit = cc.compile_fhe(&expr).expect("compile");
        let prog = lower(
            &circuit.flat_ops,
            &circuit.output_wire,
            &LowerConfig::default(),
        )
        .expect("lower");
        let sched = schedule(&prog);
        (prog, sched)
    }

    #[test]
    fn eligibility_rejects_for_the_right_reasons() {
        let mut dep = Deployment::lan(2);
        let err = eligibility(find("aby3").unwrap(), &dep).unwrap_err();
        assert!(err.contains("exactly 3"), "{}", err);

        dep = Deployment::lan(3);
        dep.min_security = Security::Malicious;
        let err = eligibility(find("aby3").unwrap(), &dep).unwrap_err();
        assert!(err.contains("semi-honest"), "{}", err);

        dep = Deployment::lan(3);
        dep.require_dishonest_majority = true;
        let err = eligibility(find("shamir").unwrap(), &dep).unwrap_err();
        assert!(err.contains("honest majority"), "{}", err);

        assert!(eligibility(find("dealer").unwrap(), &Deployment::lan(3)).is_err());
    }

    #[test]
    fn a_multiplication_chain_is_latency_bound_on_a_wan() {
        let (prog, sched) = compile("(* (* (* x y) x) y)");
        let est = estimate(&prog, &sched, find("aby3").unwrap(), &Deployment::wan(3)).unwrap();
        assert_eq!(est.bottleneck, "latency");
        // 1 input round + 3 serial multiplications + 1 open.
        assert_eq!(est.rounds, 5);
    }

    #[test]
    fn replicated_sharing_needs_no_triples_but_additive_sharing_does() {
        let (prog, sched) = compile("(* x y)");
        let dep = Deployment::lan(3);
        let aby3 = estimate(&prog, &sched, find("aby3").unwrap(), &dep).unwrap();
        let semi2k = estimate(&prog, &sched, find("semi2k").unwrap(), &dep).unwrap();
        assert_eq!(aby3.triples, 0);
        assert_eq!(semi2k.triples, 1);
        assert!(semi2k.offline_bytes > aby3.offline_bytes);
    }

    #[test]
    fn malicious_security_costs_communication() {
        let (prog, sched) = compile("(* x y)");
        let mut dep = Deployment::lan(3);
        dep.min_security = Security::SemiHonest;
        let semi = estimate(&prog, &sched, find("semi2k").unwrap(), &dep).unwrap();
        let mal = estimate(&prog, &sched, find("spdz2k").unwrap(), &dep).unwrap();
        assert!(
            mal.online_bytes > semi.online_bytes,
            "authentication should cost bytes: {} vs {}",
            mal.online_bytes,
            semi.online_bytes
        );
    }

    #[test]
    fn truncation_is_free_in_a_garbled_circuit_and_costly_in_a_ring() {
        let (plain, s_plain) = compile("(* x y)");
        let (shifted, s_shifted) = compile("(ash-right (* x y) 16)");
        let dep = Deployment::lan(3);

        let aby3_plain = estimate(&plain, &s_plain, find("aby3").unwrap(), &dep).unwrap();
        let aby3_shift = estimate(&shifted, &s_shifted, find("aby3").unwrap(), &dep).unwrap();
        assert!(
            aby3_shift.rounds > aby3_plain.rounds,
            "a ring protocol must pay for truncation"
        );

        let dep2 = Deployment::lan(2);
        let yao_plain = estimate(&plain, &s_plain, find("yao").unwrap(), &dep2).unwrap();
        let yao_shift = estimate(&shifted, &s_shifted, find("yao").unwrap(), &dep2).unwrap();
        assert_eq!(
            yao_plain.and_gates, yao_shift.and_gates,
            "shifting is wire relabelling in a garbled circuit"
        );
        assert_eq!(yao_shift.rounds, 3, "Yao is constant-round");
    }

    /// **Fitting `local_ns_per_lane`, the one term `baru mpcc calibrate` never checked.**
    ///
    /// `calibrate` validates predicted rounds and predicted online bytes against a measured run,
    /// and reports 0% error on both. It does not touch the compute term, whose constants have
    /// carried the comment "an estimate; calibrated in P1" since they were written. That gap
    /// stopped being academic when a network-only model picked a comparison backend that measured
    /// 1.24x slower than the alternative, because the two differed by 6x in local work
    /// (`precompile::mpc_compare`).
    ///
    /// The method matters more than the number. Wall clock at
    /// `NetworkProfile::INSTANT` is per-lane work *plus* a fixed cost — thread spawn, share
    /// setup, the timing call itself — and dividing a single measurement by its width charges the
    /// fixed part to every lane. So this measures several widths and fits the **slope**, which is
    /// the per-lane cost, and reports the intercept separately rather than smearing it.
    #[test]
    #[ignore] // three-party transport, large batches
    fn the_local_work_per_multiplication_is_what_the_spec_says() {
        use crate::bench::comm::NetworkProfile;
        use crate::bench::hybrid_nn::{run_parties, MpcTransport};
        use crate::crypto::types::Fr;
        use crate::interpreter::backend::Backend;
        use crate::interpreter::transport_backend::TransportBackend;
        use std::time::Instant;

        const NBITS: usize = 32;
        let widths = [1024usize, 4096, 16384, 65536];

        let mut points: Vec<(f64, f64)> = Vec::new(); // (lanes, secs)
        for w in widths {
            let (times, _comm) = run_parties(NetworkProfile::INSTANT, move |me| {
                move |be: &mut TransportBackend<MpcTransport>| {
                    let vals: Vec<Fr> = (0..w)
                        .map(|i| if me == 0 { (i as Fr) & 0xffff } else { 0 })
                        .collect();
                    let xs = be.share_secret_batch(&vals, NBITS, 0);
                    let ys = be.share_secret_batch(&vals, NBITS, 0);
                    let pairs: Vec<_> = xs.iter().cloned().zip(ys.iter().cloned()).collect();
                    // Time only the multiplication, so share setup is not in the slope.
                    let t = Instant::now();
                    let out = be.mul_batch(&pairs);
                    let ns = t.elapsed().as_nanos() as u64;
                    (be.reveal(&out[0], None), ns)
                }
            });
            // Party 0's clock; all three do the same work.
            let ns = times[0].1 as f64;
            points.push((w as f64, ns / 1e9));
            println!("    {w:6} lanes: {:9.3} ms", ns / 1e6);
        }

        // Least squares on secs = a + b * lanes.
        let n = points.len() as f64;
        let sx: f64 = points.iter().map(|p| p.0).sum();
        let sy: f64 = points.iter().map(|p| p.1).sum();
        let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
        let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n;
        let measured_ns_per_lane = slope * 1e9;
        let spec = find("aby3")
            .expect("aby3 is registered")
            .cost
            .local_ns_per_lane;
        println!(
            "  fitted {measured_ns_per_lane:.1} ns/lane (fixed cost {:.3} ms), spec says {spec:.1}",
            intercept * 1000.0
        );
        println!(
            "  ratio measured/spec = {:.2}x",
            measured_ns_per_lane / spec
        );

        assert!(
            measured_ns_per_lane > 0.0,
            "a per-lane cost must be positive; the fit found {measured_ns_per_lane:.1} ns, which \
             means the widths swept are all inside the fixed-cost regime"
        );

        // ── What the per-lane figure is actually made of ──
        //
        // The fit above times `mul_batch` over a real transport, so it contains serialization, a
        // channel round and a blocking receive as well as the field arithmetic the spec constant
        // is nominally about. To say whether 5 ns/lane is a wrong constant or a different
        // quantity, we need the arithmetic on its own, so we fit a purely local operation on the
        // same backend: `add` communicates nothing, and its per-lane cost is share bookkeeping
        // plus a ring addition — a floor on what any interactive lane must also pay.
        //
        // (`Aby3Simulator` is *not* the instrument for this, which is worth recording because it
        // looks like it should be: it has no batched multiply, and `sim::mul_ab3_concurrent`
        // spawns three OS threads and three inproc handles per multiplication, so it measures
        // \SI{58}{\micro\second} per lane — 54x the real backend. It is a correctness aid.)
        let add_ns: f64 = {
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for w in widths {
                let (times, _c) = run_parties(NetworkProfile::INSTANT, move |me| {
                    move |be: &mut TransportBackend<MpcTransport>| {
                        let vals: Vec<Fr> = (0..w)
                            .map(|i| if me == 0 { (i as Fr) & 0xffff } else { 0 })
                            .collect();
                        let xs = be.share_secret_batch(&vals, NBITS, 0);
                        let ys = be.share_secret_batch(&vals, NBITS, 0);
                        let t = Instant::now();
                        let mut last = xs[0].clone();
                        for (a, b) in xs.iter().zip(&ys) {
                            last = be.add(a, b);
                        }
                        let ns = t.elapsed().as_nanos() as u64;
                        (be.reveal(&last, None), ns)
                    }
                });
                pts.push((w as f64, times[0].1 as f64 / 1e9));
            }
            let syl: f64 = pts.iter().map(|p| p.1).sum();
            let sxyl: f64 = pts.iter().map(|p| p.0 * p.1).sum();
            let sl = (n * sxyl - sx * syl) / (n * sxx - sx * sx);
            sl * 1e9
        };
        // 5.0 ns was the estimate this field carried before it was fitted. Quoted here because
        // the interesting fact is not the new number but that the arithmetic floor alone already
        // exceeded the old one by 29x, before any communication was counted at all.
        const PRE_FIT_ESTIMATE_NS: f64 = 5.0;
        println!(
            "  local-only `add`: {add_ns:.1} ns/lane — share bookkeeping plus a ring addition, \
             no communication at all, already {:.0}x the {PRE_FIT_ESTIMATE_NS:.0} ns this field \
             assumed before it was fitted",
            add_ns / PRE_FIT_ESTIMATE_NS
        );
        println!(
            "  and a multiply's {measured_ns_per_lane:.0} ns is {:.1}x an add, the rest being \
             serialization and the channel round",
            measured_ns_per_lane / add_ns
        );

        // A wide band, deliberately. This runs on whatever hardware the artifact is built on, so
        // a tight bound would be a portability bug rather than a calibration. What it does catch
        // is the thing that was actually wrong: an order-of-magnitude error in a constant nobody
        // had ever checked. The spec said 5 ns; the fit said 1072; the spec now says 1072.
        let ratio = measured_ns_per_lane / spec;
        assert!(
            (0.25..=4.0).contains(&ratio),
            "measured {measured_ns_per_lane:.1} ns/lane against a spec of {spec:.1} ({ratio:.2}x) \
             — the compute term of every prediction naming `aby3` rides on this, and on a batched \
             program it decides both the total and the reported bottleneck"
        );
        // The arithmetic floor must stay below the full lane cost, or the decomposition above is
        // measuring something other than it claims.
        assert!(
            add_ns < measured_ns_per_lane,
            "a local-only add ({add_ns:.0} ns) cannot cost more than a multiply \
             ({measured_ns_per_lane:.0} ns); the two fits are not measuring what they say"
        );
    }

    #[test]
    fn division_is_the_expensive_primitive() {
        let (m, sm) = compile("(* x y)");
        let (d, sd) = compile("(/ x y)");
        let dep = Deployment::lan(3);
        let mul = estimate(&m, &sm, find("aby3").unwrap(), &dep).unwrap();
        let div = estimate(&d, &sd, find("aby3").unwrap(), &dep).unwrap();
        assert!(
            div.rounds > 20 * mul.rounds,
            "restoring division is per-bit serial: {} vs {}",
            div.rounds,
            mul.rounds
        );
    }
}
