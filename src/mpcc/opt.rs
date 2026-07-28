//! **Backend-aware** circuit optimization.
//!
//! Every optimizer for secure computation that I know of is blind in one eye.
//! MP-SPDZ minimizes communication rounds and cannot see FHE.  FHE compilers
//! minimize multiplicative depth and cannot see MPC.  Neither can ask the
//! question a user actually has — *should this be an FHE program at all?* —
//! because neither has a cost model spanning the alternatives.
//!
//! Baru does ([`crate::mpcc::cost`]), so this pass takes the target protocol as
//! a **parameter** and keeps a rewrite only when that protocol's predicted cost
//! improves.  The point is not that this makes circuits faster; it is that the
//! resulting circuits are *different per backend*, which means a backend-blind
//! optimizer has to pick one behaviour and be wrong the rest of the time.
//!
//! Two rules already demonstrate the effect:
//!
//! * **Strength reduction** — `x * 5` → `(x << 2) + x`.  Under Yao a `k × k`
//!   schoolbook multiplier is `k²` AND gates and an adder is `k`, so at
//!   `k = 64` this is a ~64× saving.  Under ABY3, multiplying by a *public*
//!   constant is already a free local scalar multiply, so the rewrite adds an
//!   instruction and buys nothing.
//! * **Shift fusion and CSE** are cheap everywhere, so they survive under every
//!   backend — which is exactly why they make a good control.
//!
//! ## Correctness is checked, not argued
//!
//! A rewritten circuit is accepted only after [`equivalent`] has evaluated both
//! versions on random inputs through the *existing* `FlatOp` evaluator
//! ([`evaluate_fhe`] on [`MockFheBackend`], whose `Ct = u64` makes it an exact
//! plaintext interpreter).  Reusing that evaluator rather than writing a second
//! one is deliberate: a private copy of the semantics would drift, and then the
//! oracle would be validating the optimizer against the optimizer's own idea of
//! what the ops mean.
//!
//! ## Known limitation: rule granularity
//!
//! The search accepts or rejects each rule for the *whole circuit*, not per
//! site.  So a program mixing `x * 4` (a pure win for everyone — one shift
//! replaces a constant load and a multiply) with `x * 100` (a win only for
//! binary protocols) is scored as one bundle, and an arithmetic backend may
//! accept the bundle for the sake of the first even though the second does
//! nothing for it.  Per-site search is the obvious next step and would sharpen
//! the per-backend divergence rather than change its direction.
//!
//! Random testing is not proof.  It is, however, the same standard the FHE
//! backends are held to in `tests/fhe_differential.rs`, and it has caught real
//! bugs there.  A rewrite that is wrong on a measure-zero set would survive;
//! none of the current rules have that shape (they are exact algebraic
//! identities over wrapping two's-complement arithmetic).

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::fhe::backend::FheCircuit;
use crate::fhe::circuit_emitter::evaluate_fhe;
use crate::fhe::MockFheBackend;
use crate::zkp::circuit_compiler::FlatOp;

use super::cost::{estimate, Deployment};
use super::lower::{flat_dsts, input_wires, lower, LowerConfig};
use super::protocols::ProtocolSpec;
use super::schedule::schedule;

// ─── Objective ────────────────────────────────────────────────────────────────

/// What "better" means for one backend.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Objective {
    pub rounds: u64,
    pub online_bytes: u64,
    pub and_gates: u64,
    pub triples: u64,
    /// Predicted total time — the scalar the search minimizes.
    pub secs: f64,
    /// `FlatOp` count, used only to break ties so the search prefers the
    /// smaller of two circuits the cost model cannot distinguish.
    pub ops: usize,
}

impl Objective {
    /// Strictly better: cheaper, or the same cost with fewer instructions.
    fn improves_on(&self, other: &Objective) -> bool {
        const EPS: f64 = 1e-15;
        if self.secs < other.secs - EPS {
            return true;
        }
        if self.secs > other.secs + EPS {
            return false;
        }
        self.ops < other.ops
    }
}

/// Cost `flat` for one protocol.
pub fn objective(
    flat: &[FlatOp],
    output_wire: &str,
    spec: &ProtocolSpec,
    dep: &Deployment,
    cfg: &LowerConfig,
) -> Result<Objective, String> {
    let prog = lower(flat, output_wire, cfg)?;
    let sched = schedule(&prog);
    let est = estimate(&prog, &sched, spec, dep)?;
    Ok(Objective {
        rounds: est.rounds,
        online_bytes: est.online_bytes,
        and_gates: est.and_gates,
        triples: est.triples,
        secs: est.total_secs,
        ops: flat.len(),
    })
}

// ─── Rules ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Rule {
    /// `x*1 → x`, `x*0 → 0`, `x±0 → x`, shift by 0, mux on a known condition.
    Algebraic,
    /// Reuse an identical earlier computation instead of repeating it.
    Cse,
    /// `(x << a) << b → x << (a+b)`, and the same for right shifts.
    FuseShifts,
    /// `x * c` for a public `c` with few set bits → shifts and adds.
    StrengthReduce,
}

impl Rule {
    pub const ALL: &'static [Rule] = &[
        Rule::Algebraic,
        Rule::Cse,
        Rule::FuseShifts,
        Rule::StrengthReduce,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Rule::Algebraic => "algebraic",
            Rule::Cse => "cse",
            Rule::FuseShifts => "fuse-shifts",
            Rule::StrengthReduce => "strength-reduce",
        }
    }

    fn apply(&self, flat: &[FlatOp], output_wire: &str) -> Vec<FlatOp> {
        match self {
            Rule::Algebraic => algebraic(flat, output_wire),
            Rule::Cse => cse(flat, output_wire),
            Rule::FuseShifts => fuse_shifts(flat, output_wire),
            Rule::StrengthReduce => strength_reduce(flat),
        }
    }
}

/// Rewrite every operand of `op` through `subst`.
fn rename(op: &FlatOp, subst: &HashMap<String, String>) -> FlatOp {
    let r = |w: &String| subst.get(w).cloned().unwrap_or_else(|| w.clone());
    match op {
        FlatOp::Const { dst, value } => FlatOp::Const {
            dst: dst.clone(),
            value: *value,
        },
        FlatOp::ConstF { dst, value } => FlatOp::ConstF {
            dst: dst.clone(),
            value: *value,
        },
        FlatOp::IAdd32 { dst, lhs, rhs } => FlatOp::IAdd32 {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::ISub32 { dst, lhs, rhs } => FlatOp::ISub32 {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::IMul { dst, lhs, rhs } => FlatOp::IMul {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::IDiv {
            dst_q,
            dst_r,
            lhs,
            rhs,
        } => FlatOp::IDiv {
            dst_q: dst_q.clone(),
            dst_r: dst_r.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::ICmpEq { dst, lhs, rhs } => FlatOp::ICmpEq {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::ICmpUlt { dst, lhs, rhs } => FlatOp::ICmpUlt {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::ICmpSlt { dst, lhs, rhs } => FlatOp::ICmpSlt {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::BAnd { dst, lhs, rhs } => FlatOp::BAnd {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::BOr { dst, lhs, rhs } => FlatOp::BOr {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::BXor { dst, lhs, rhs } => FlatOp::BXor {
            dst: dst.clone(),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        FlatOp::BNot { dst, src } => FlatOp::BNot {
            dst: dst.clone(),
            src: r(src),
        },
        FlatOp::Shl { dst, src, amount } => FlatOp::Shl {
            dst: dst.clone(),
            src: r(src),
            amount: *amount,
        },
        FlatOp::Shr { dst, src, amount } => FlatOp::Shr {
            dst: dst.clone(),
            src: r(src),
            amount: *amount,
        },
        FlatOp::Mux {
            dst,
            cond,
            t_val,
            e_val,
        } => FlatOp::Mux {
            dst: dst.clone(),
            cond: r(cond),
            t_val: r(t_val),
            e_val: r(e_val),
        },
        FlatOp::PopCount { dst, src } => FlatOp::PopCount {
            dst: dst.clone(),
            src: r(src),
        },
    }
}

/// Algebraic identities.  Every one holds exactly over wrapping two's-complement
/// arithmetic, which is what `FlatOp` means by `IAdd32`/`IMul`.
fn algebraic(flat: &[FlatOp], output_wire: &str) -> Vec<FlatOp> {
    let mut consts: HashMap<String, i64> = HashMap::new();
    let mut subst: HashMap<String, String> = HashMap::new();
    let mut out: Vec<FlatOp> = Vec::with_capacity(flat.len());

    for op in flat {
        let op = rename(op, &subst);
        // The circuit's signature is `(input_wires, output_wire)`, and callers
        // hold the output wire's *name*.  Substituting it away would silently
        // change the signature — the wire would then look like an undefined
        // input.  So the instruction that defines it is never elided, even when
        // it is a no-op like `x * 1`.
        if op.dst() == output_wire && !matches!(op, FlatOp::Const { .. }) {
            out.push(op);
            continue;
        }
        // `elide` maps this op's destination onto an existing wire and drops it.
        let elide = |dst: &String, onto: &String, subst: &mut HashMap<String, String>| {
            let target = subst.get(onto).cloned().unwrap_or_else(|| onto.clone());
            subst.insert(dst.clone(), target);
        };
        match &op {
            FlatOp::Const { dst, value } => {
                consts.insert(dst.clone(), *value);
                out.push(op.clone());
            }
            FlatOp::IMul { dst, lhs, rhs } => match (consts.get(lhs), consts.get(rhs)) {
                (Some(1), _) => elide(dst, rhs, &mut subst),
                (_, Some(1)) => elide(dst, lhs, &mut subst),
                (Some(0), _) | (_, Some(0)) => {
                    consts.insert(dst.clone(), 0);
                    out.push(FlatOp::Const {
                        dst: dst.clone(),
                        value: 0,
                    });
                }
                _ => out.push(op.clone()),
            },
            FlatOp::IAdd32 { dst, lhs, rhs } => match (consts.get(lhs), consts.get(rhs)) {
                (Some(0), _) => elide(dst, rhs, &mut subst),
                (_, Some(0)) => elide(dst, lhs, &mut subst),
                _ => out.push(op.clone()),
            },
            // Only `x - 0` folds; `0 - x` is a real negation.
            FlatOp::ISub32 { dst, lhs, rhs } if consts.get(rhs) == Some(&0) => {
                elide(dst, lhs, &mut subst)
            }
            FlatOp::Shl { dst, src, amount } | FlatOp::Shr { dst, src, amount } if *amount == 0 => {
                elide(dst, src, &mut subst)
            }
            FlatOp::Mux {
                dst,
                cond,
                t_val,
                e_val,
            } => match consts.get(cond) {
                // Mux semantics are "nonzero is true" in every backend.
                Some(0) => elide(dst, e_val, &mut subst),
                Some(_) => elide(dst, t_val, &mut subst),
                None => out.push(op.clone()),
            },
            _ => out.push(op.clone()),
        }
    }
    out
}

/// Common subexpression elimination.  Commutative operations are keyed on
/// sorted operands so `a+b` and `b+a` collide.
fn cse(flat: &[FlatOp], output_wire: &str) -> Vec<FlatOp> {
    let mut seen: HashMap<String, String> = HashMap::new();
    let mut subst: HashMap<String, String> = HashMap::new();
    let mut out: Vec<FlatOp> = Vec::with_capacity(flat.len());

    for op in flat {
        let op = rename(op, &subst);
        // As in `algebraic`: the output wire's definition must survive, or the
        // circuit's signature changes underneath the caller.
        if op.dst() == output_wire {
            out.push(op);
            continue;
        }
        // `IDiv` writes two wires; keeping the key/substitution bookkeeping
        // honest for it is not worth the complexity, and divisions are rare.
        let key = match &op {
            FlatOp::IDiv { .. } | FlatOp::ConstF { .. } => None,
            other => Some(cse_key(other)),
        };
        match key {
            Some(k) => match seen.get(&k) {
                Some(prev) => {
                    subst.insert(op.dst().to_string(), prev.clone());
                }
                None => {
                    seen.insert(k, op.dst().to_string());
                    out.push(op);
                }
            },
            None => out.push(op),
        }
    }
    out
}

fn cse_key(op: &FlatOp) -> String {
    let commutative = matches!(
        op,
        FlatOp::IAdd32 { .. }
            | FlatOp::IMul { .. }
            | FlatOp::ICmpEq { .. }
            | FlatOp::BAnd { .. }
            | FlatOp::BOr { .. }
            | FlatOp::BXor { .. }
    );
    let mut ins: Vec<&str> = op.inputs();
    if commutative {
        ins.sort_unstable();
    }
    let tag = match op {
        FlatOp::Const { value, .. } => format!("const:{}", value),
        FlatOp::ConstF { value, .. } => format!("constf:{}", value),
        FlatOp::IAdd32 { .. } => "add".into(),
        FlatOp::ISub32 { .. } => "sub".into(),
        FlatOp::IMul { .. } => "mul".into(),
        FlatOp::IDiv { .. } => "div".into(),
        FlatOp::ICmpEq { .. } => "eq".into(),
        FlatOp::ICmpUlt { .. } => "ult".into(),
        FlatOp::ICmpSlt { .. } => "slt".into(),
        FlatOp::BAnd { .. } => "and".into(),
        FlatOp::BOr { .. } => "or".into(),
        FlatOp::BXor { .. } => "xor".into(),
        FlatOp::BNot { .. } => "not".into(),
        FlatOp::Shl { amount, .. } => format!("shl:{}", amount),
        FlatOp::Shr { amount, .. } => format!("shr:{}", amount),
        FlatOp::Mux { .. } => "mux".into(),
        FlatOp::PopCount { .. } => "popcount".into(),
    };
    format!("{}({})", tag, ins.join(","))
}

/// `(x << a) << b → x << (a+b)`, when the intermediate has no other use.
/// Fusing a *right* shift chain is equally valid (both are logical shifts on
/// every backend — see the `Shr` note in `mpcc::lower`), but mixing the two
/// directions is not, because the bits dropped in between do not come back.
fn fuse_shifts(flat: &[FlatOp], output_wire: &str) -> Vec<FlatOp> {
    let uses = use_counts(flat, output_wire);
    let mut producer: HashMap<&str, (&str, u32, bool)> = HashMap::new(); // dst -> (src, amt, is_left)
    let mut out: Vec<FlatOp> = Vec::with_capacity(flat.len());

    for op in flat {
        match op {
            FlatOp::Shl { dst, src, amount } | FlatOp::Shr { dst, src, amount } => {
                let is_left = matches!(op, FlatOp::Shl { .. });
                let fused = producer.get(src.as_str()).and_then(|(psrc, pamt, pleft)| {
                    let total = pamt + amount;
                    // Same direction, single-use intermediate, and a shift
                    // amount the backends still define (`FlatOp` amounts are
                    // compile-time literals under 64).
                    if *pleft == is_left && uses.get(src.as_str()) == Some(&1) && total < 64 {
                        Some((psrc.to_string(), total))
                    } else {
                        None
                    }
                });
                let (real_src, real_amt) = match fused {
                    Some((s, a)) => (s, a),
                    None => (src.clone(), *amount),
                };
                producer.insert(
                    dst.as_str(),
                    (
                        Box::leak(real_src.clone().into_boxed_str()),
                        real_amt,
                        is_left,
                    ),
                );
                out.push(if is_left {
                    FlatOp::Shl {
                        dst: dst.clone(),
                        src: real_src,
                        amount: real_amt,
                    }
                } else {
                    FlatOp::Shr {
                        dst: dst.clone(),
                        src: real_src,
                        amount: real_amt,
                    }
                });
            }
            other => out.push(other.clone()),
        }
    }
    out
}

/// `x * c` for a public `c` → a chain of shifts and adds.
///
/// Exact over wrapping arithmetic: `c = Σ 2^bᵢ` gives `x·c = Σ (x << bᵢ)`, and
/// both the shifts and the additions wrap the same way the multiplication would.
/// Negative constants go through `0 - (x · |c|)`.
///
/// Only applied when `|c|` has at most `MAX_TERMS` set bits, so the replacement
/// is a handful of instructions rather than a full multiplier.
fn strength_reduce(flat: &[FlatOp]) -> Vec<FlatOp> {
    const MAX_TERMS: u32 = 3;
    let mut consts: HashMap<String, i64> = HashMap::new();
    let mut out: Vec<FlatOp> = Vec::with_capacity(flat.len());
    let mut fresh = 0usize;
    let taken: HashSet<String> = flat
        .iter()
        .flat_map(|o| flat_dsts(o).into_iter().map(String::from))
        .collect();
    let name = |fresh: &mut usize| loop {
        *fresh += 1;
        let n = format!("opt${}", fresh);
        if !taken.contains(&n) {
            return n;
        }
    };

    for op in flat {
        if let FlatOp::Const { dst, value } = op {
            consts.insert(dst.clone(), *value);
        }
        let (dst, x, c) = match op {
            FlatOp::IMul { dst, lhs, rhs } => match (consts.get(lhs), consts.get(rhs)) {
                // A constant × constant product is the constant folder's job.
                (Some(_), Some(_)) => {
                    out.push(op.clone());
                    continue;
                }
                (Some(c), None) => (dst, rhs, *c),
                (None, Some(c)) => (dst, lhs, *c),
                _ => {
                    out.push(op.clone());
                    continue;
                }
            },
            other => {
                out.push(other.clone());
                continue;
            }
        };

        let mag = c.unsigned_abs();
        // `|c| == 1` is the algebraic rule's job (`x*1 → x`, `x*-1` is already
        // a single instruction); `c == 0` likewise folds to a constant.
        if mag <= 1 || mag.count_ones() > MAX_TERMS {
            out.push(op.clone());
            continue;
        }

        // Σ (x << bᵢ) over the set bits of |c|, low to high.  `mag > 1`
        // guarantees at least one nonzero shift, so no term needs a copy of
        // `x` that `FlatOp` has no instruction for.
        let bits: Vec<u32> = (0..64).filter(|b| mag >> b & 1 == 1).collect();
        let last = bits.len() - 1;
        let mut terms: Vec<String> = Vec::with_capacity(bits.len());
        for (i, b) in bits.iter().enumerate() {
            if *b == 0 {
                terms.push(x.clone());
                continue;
            }
            // The final shift can write straight into `dst` when it is also
            // the final instruction (single term, positive constant).
            let into = if i == last && last == 0 && c > 0 {
                dst.clone()
            } else {
                name(&mut fresh)
            };
            out.push(FlatOp::Shl {
                dst: into.clone(),
                src: x.clone(),
                amount: *b,
            });
            terms.push(into);
        }

        if terms.len() == 1 && c > 0 {
            continue; // the shift above already wrote `dst`
        }

        // Fold the terms, writing the last addition into `dst` when the sign
        // does not need a final negation.
        let mut acc = terms[0].clone();
        for (i, term) in terms.iter().enumerate().skip(1) {
            let into = if i == terms.len() - 1 && c > 0 {
                dst.clone()
            } else {
                name(&mut fresh)
            };
            out.push(FlatOp::IAdd32 {
                dst: into.clone(),
                lhs: acc,
                rhs: term.clone(),
            });
            acc = into;
        }
        if c < 0 {
            let z = name(&mut fresh);
            out.push(FlatOp::Const {
                dst: z.clone(),
                value: 0,
            });
            out.push(FlatOp::ISub32 {
                dst: dst.clone(),
                lhs: z,
                rhs: acc,
            });
        }
    }
    out
}

/// How many times each wire is read, counting the output wire as one use.
fn use_counts(flat: &[FlatOp], output_wire: &str) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for op in flat {
        for w in op.inputs() {
            *counts.entry(w.to_string()).or_insert(0) += 1;
        }
    }
    *counts.entry(output_wire.to_string()).or_insert(0) += 1;
    counts
}

/// Drop instructions whose results nothing reads.
pub fn dead_code_eliminate(flat: &[FlatOp], output_wire: &str) -> Vec<FlatOp> {
    let mut live: HashSet<String> = HashSet::new();
    live.insert(output_wire.to_string());
    let mut keep = vec![false; flat.len()];
    for (i, op) in flat.iter().enumerate().rev() {
        if flat_dsts(op).iter().any(|d| live.contains(*d)) {
            keep[i] = true;
            for w in op.inputs() {
                live.insert(w.to_string());
            }
        }
    }
    flat.iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(op, _)| op.clone())
        .collect()
}

// ─── Correctness oracle ───────────────────────────────────────────────────────

/// Evaluate a `FlatOp` block on plaintext inputs.
///
/// Reuses the FHE emitter with [`MockFheBackend`] (whose `Ct = u64`), so the
/// semantics here are *by construction* the semantics every other backend is
/// tested against, rather than a second opinion about what the ops mean.
pub fn eval_flat(flat: &[FlatOp], output_wire: &str, inputs: &HashMap<String, i64>) -> u64 {
    let circuit = FheCircuit {
        flat_ops: flat.to_vec(),
        encrypted_inputs: vec![],
        public_inputs: input_wires(flat, output_wire),
        output_wire: output_wire.to_string(),
        num_gates: flat.len(),
        output_fixed: false,
        real_arith: false,
    };
    evaluate_fhe(&circuit, inputs, &HashMap::new(), &MockFheBackend)
}

/// Do two circuits agree on `trials` pseudorandom input assignments?
///
/// Inputs are drawn from a mix of small values, boundary values and full-width
/// randoms: a rewrite that is wrong only at a sign boundary or only on wrap is
/// the failure mode worth designing the sampler around.
pub fn equivalent(
    original: &[FlatOp],
    rewritten: &[FlatOp],
    output_wire: &str,
    trials: usize,
) -> Result<(), String> {
    let names = input_wires(original, output_wire);
    for t in 0..trials {
        let mut inputs = HashMap::new();
        for (i, n) in names.iter().enumerate() {
            let v: i64 = match (t + i) % 5 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => (rand::random::<i32>() % 1000) as i64,
                _ => rand::random::<i32>() as i64,
            };
            inputs.insert(n.clone(), v);
        }
        // A division by zero panics in the evaluator rather than diverging;
        // skip those assignments instead of failing the comparison.
        let a = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eval_flat(original, output_wire, &inputs)
        }));
        let b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eval_flat(rewritten, output_wire, &inputs)
        }));
        match (a, b) {
            (Ok(a), Ok(b)) if a == b => {}
            (Ok(a), Ok(b)) => {
                return Err(format!(
                    "rewrite changed the result on {:?}: {} became {}",
                    inputs, a as i64, b as i64
                ))
            }
            (Err(_), Err(_)) => {}
            _ => {
                return Err(format!(
                    "rewrite changed which inputs are defined: {:?}",
                    inputs
                ))
            }
        }
    }
    Ok(())
}

// ─── The pass ─────────────────────────────────────────────────────────────────

/// One rule that survived the cost check.
#[derive(Debug, Clone, Serialize)]
pub struct Applied {
    pub rule: &'static str,
    pub ops_before: usize,
    pub ops_after: usize,
    pub secs_before: f64,
    pub secs_after: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Optimized {
    pub protocol: String,
    #[serde(skip)]
    pub flat: Vec<FlatOp>,
    pub before: Objective,
    pub after: Objective,
    /// Rules kept, in the order the search accepted them.
    pub applied: Vec<Applied>,
    /// Rules tried and rejected because they did not help *this* backend.
    pub rejected: Vec<&'static str>,
    /// Whether the result was differentially checked against the original.
    pub verified: bool,
}

impl Optimized {
    /// Predicted speed-up for the target backend.  `1.0` means no change.
    pub fn speedup(&self) -> f64 {
        if self.after.secs <= 0.0 {
            1.0
        } else {
            self.before.secs / self.after.secs
        }
    }
}

/// Optimize `flat` for one protocol.
///
/// Greedy and cost-guided: each rule is applied speculatively, priced with
/// [`objective`], and kept only if it improves.  Repeats to a fixed point (rules
/// enable each other — strength reduction creates the `x + 0` that the algebraic
/// rule then folds away).
pub fn optimize(
    flat: &[FlatOp],
    output_wire: &str,
    spec: &ProtocolSpec,
    dep: &Deployment,
    cfg: &LowerConfig,
    verify: bool,
) -> Result<Optimized, String> {
    const MAX_PASSES: usize = 4;

    let base = dead_code_eliminate(flat, output_wire);
    let before = objective(&base, output_wire, spec, dep, cfg)?;
    let mut current = base;
    let mut current_obj = before;
    let mut applied: Vec<Applied> = Vec::new();
    let mut rejected: HashSet<&'static str> = HashSet::new();

    for _ in 0..MAX_PASSES {
        let mut changed = false;
        for rule in Rule::ALL {
            let candidate = dead_code_eliminate(&rule.apply(&current, output_wire), output_wire);
            if candidate.len() == current.len() && candidate == current {
                continue;
            }
            let obj = match objective(&candidate, output_wire, spec, dep, cfg) {
                Ok(o) => o,
                // A rewrite that produces something the lowering rejects is
                // simply not a candidate.
                Err(_) => continue,
            };
            if obj.improves_on(&current_obj) {
                applied.push(Applied {
                    rule: rule.name(),
                    ops_before: current.len(),
                    ops_after: candidate.len(),
                    secs_before: current_obj.secs,
                    secs_after: obj.secs,
                });
                rejected.remove(rule.name());
                current = candidate;
                current_obj = obj;
                changed = true;
            } else {
                rejected.insert(rule.name());
            }
        }
        if !changed {
            break;
        }
    }

    if verify {
        equivalent(flat, &current, output_wire, 32)?;
    }

    let mut rejected: Vec<&'static str> = rejected.into_iter().collect();
    rejected.sort_unstable();
    Ok(Optimized {
        protocol: spec.id.to_string(),
        flat: current,
        before,
        after: current_obj,
        applied,
        rejected,
        verified: verify,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpcc::cost::native_domain;
    use crate::mpcc::protocols::find;
    use crate::zkp::circuit_compiler::CircuitCompiler;

    fn compile(src: &str) -> (Vec<FlatOp>, String) {
        let expr = crate::interpreter::parse(src).expect("parse");
        let mut cc = CircuitCompiler::new();
        let c = cc.compile_fhe(&expr).expect("compile");
        (c.flat_ops, c.output_wire)
    }

    fn cfg_for(id: &str) -> LowerConfig {
        LowerConfig {
            domain: native_domain(find(id).unwrap(), 64),
            parties: if id == "yao" { 2 } else { 3 },
            bit_width: 32,
            public_vars: Vec::new(),
        }
    }

    fn dep_for(id: &str) -> Deployment {
        Deployment::lan(if id == "yao" { 2 } else { 3 })
    }

    fn opt(src: &str, id: &str) -> Optimized {
        let (flat, out) = compile(src);
        optimize(
            &flat,
            &out,
            find(id).unwrap(),
            &dep_for(id),
            &cfg_for(id),
            true,
        )
        .expect("optimize")
    }

    // ── The headline claim ────────────────────────────────────────────────

    #[test]
    fn strength_reduction_helps_a_garbled_circuit_and_not_a_ring_protocol() {
        // `x * 100`: 100 = 0b1100100, three set bits, so it becomes two shifts
        // and two adds instead of a 64x64 schoolbook multiplier.
        let yao = opt("(* x 100)", "yao");
        let aby3 = opt("(* x 100)", "aby3");

        assert!(
            yao.applied.iter().any(|a| a.rule == "strength-reduce"),
            "Yao should take the rewrite: {:?}",
            yao
        );
        assert!(
            aby3.applied.iter().all(|a| a.rule != "strength-reduce"),
            "ABY3 multiplies by a public constant for free, so the rewrite is \
             pointless there: {:?}",
            aby3
        );
        assert!(
            yao.after.and_gates < yao.before.and_gates,
            "gate count should drop: {} -> {}",
            yao.before.and_gates,
            yao.after.and_gates
        );
        // This is the whole point: the two backends get different circuits.
        let aby3_flat = aby3.flat.len();
        assert_ne!(
            yao.flat.len(),
            aby3_flat,
            "backend-aware optimization must produce backend-specific circuits"
        );
    }

    #[test]
    fn the_wall_clock_win_appears_once_the_circuit_is_big_enough_to_dominate() {
        // A single `x * 100` saves ~32x the AND gates but barely any time: Yao
        // is constant-round, and for a circuit that small the three rounds of
        // latency swamp both the bandwidth and the garbling.  The gate saving
        // only becomes a *time* saving once the circuit is large enough to be
        // the dominant term — which is itself a fact about the cost model worth
        // asserting, because it is the sort of thing a gate-count-only
        // optimizer would get wrong.
        let small = opt("(* x 100)", "yao");
        assert!(small.before.and_gates >= 8 * small.after.and_gates);
        assert!(small.speedup() >= 1.0);

        let big = opt("(dot (list a b c d) '(3 5 9 17))", "yao");
        assert!(
            big.speedup() > 2.0,
            "four constant multiplies should show a real speed-up, got {:.2}x \
             ({} -> {} AND gates)",
            big.speedup(),
            big.before.and_gates,
            big.after.and_gates
        );
    }

    // ── Individual rules ──────────────────────────────────────────────────

    #[test]
    fn algebraic_identities_hold_on_random_inputs() {
        for src in [
            "(* x 1)",
            "(* x 0)",
            "(+ x 0)",
            "(- x 0)",
            "(+ (* x 1) (* y 0))",
        ] {
            let (flat, out) = compile(src);
            let rewritten = dead_code_eliminate(&algebraic(&flat, &out), &out);
            equivalent(&flat, &rewritten, &out, 24).unwrap_or_else(|e| panic!("{}: {}", src, e));
        }
    }

    #[test]
    fn strength_reduction_preserves_semantics_including_negative_constants() {
        for src in [
            "(* x 2)",
            "(* x 5)",
            "(* x 100)",
            "(* x -3)",
            "(* x -1024)",
            "(* 6 x)",
            "(+ (* x 12) (* y -20))",
        ] {
            let (flat, out) = compile(src);
            let rewritten = dead_code_eliminate(&strength_reduce(&flat), &out);
            equivalent(&flat, &rewritten, &out, 32).unwrap_or_else(|e| panic!("{}: {}", src, e));
        }
    }

    #[test]
    fn cse_collapses_a_repeated_subexpression() {
        let (flat, out) = compile("(+ (* x y) (* x y))");
        let rewritten = dead_code_eliminate(&cse(&flat, &out), &out);
        assert!(
            rewritten.len() < flat.len(),
            "expected the duplicate product to go: {} -> {}",
            flat.len(),
            rewritten.len()
        );
        equivalent(&flat, &rewritten, &out, 24).unwrap();
    }

    #[test]
    fn cse_sees_through_commutativity() {
        let (flat, out) = compile("(+ (* x y) (* y x))");
        let rewritten = dead_code_eliminate(&cse(&flat, &out), &out);
        assert!(rewritten.len() < flat.len());
        equivalent(&flat, &rewritten, &out, 24).unwrap();
    }

    #[test]
    fn shifts_fuse_only_in_a_consistent_direction() {
        let (flat, out) = compile("(ash-left (ash-left x 3) 4)");
        let fused = dead_code_eliminate(&fuse_shifts(&flat, &out), &out);
        assert!(fused.len() < flat.len(), "two shifts should become one");
        equivalent(&flat, &fused, &out, 24).unwrap();

        // Left-then-right drops bits; fusing would resurrect them.
        let (flat, out) = compile("(ash-right (ash-left x 8) 4)");
        let fused = dead_code_eliminate(&fuse_shifts(&flat, &out), &out);
        equivalent(&flat, &fused, &out, 32).unwrap();
    }

    // ── Whole-pass invariants ─────────────────────────────────────────────

    #[test]
    fn optimization_never_makes_the_prediction_worse() {
        for id in ["aby3", "yao", "semi2k", "shamir"] {
            for src in [
                "(* x 100)",
                "(+ (* x y) (* x y))",
                "(* x (> x 0))",
                "(+ (* x 3) (* y 7))",
                "(dot (list a b c d) '(2 3 4 5))",
            ] {
                let o = opt(src, id);
                assert!(
                    o.after.secs <= o.before.secs + 1e-15,
                    "{} under {}: {} -> {}",
                    src,
                    id,
                    o.before.secs,
                    o.after.secs
                );
                assert!(o.verified);
            }
        }
    }

    #[test]
    fn every_bench_suite_case_optimizes_and_stays_correct() {
        for case in crate::bench::SUITE {
            let src = case.mpc_expr.unwrap_or(case.expr);
            let (flat, out) = compile(src);
            for id in ["aby3", "yao"] {
                let o = optimize(
                    &flat,
                    &out,
                    find(id).unwrap(),
                    &dep_for(id),
                    &cfg_for(id),
                    true,
                )
                .unwrap_or_else(|e| panic!("{} under {}: {}", case.name, id, e));
                assert!(o.verified, "{} under {}", case.name, id);
            }
        }
    }

    #[test]
    fn a_semantics_breaking_rewrite_is_caught() {
        // Sanity-check the oracle itself: if it cannot catch a deliberate bug,
        // every "verified" above is worthless.
        let (flat, out) = compile("(* x 100)");
        let mut broken = flat.clone();
        broken.push(FlatOp::IAdd32 {
            dst: out.clone(),
            lhs: out.clone(),
            rhs: out.clone(),
        });
        assert!(equivalent(&flat, &broken, &out, 16).is_err());
    }
}
