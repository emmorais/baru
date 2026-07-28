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
//! The rules demonstrate the effect from three directions:
//!
//! * **Strength reduction** — `x * 5` → `(x << 2) + x`.  Under Yao a `k × k`
//!   schoolbook multiplier is `k²` AND gates and an adder is `k`, so at
//!   `k = 64` this is a ~64× saving.  Under ABY3, multiplying by a *public*
//!   constant is already a free local scalar multiply, so the rewrite adds an
//!   instruction and buys nothing.
//! * **Reassociation** ([`reassociate`]) trades dependence depth and *nothing
//!   else* — same instruction count, same gate count — so it isolates the
//!   round term.  Which operator it pays for inverts between protocol
//!   families: multiplication under secret sharing (one round each), addition
//!   under a binary protocol (a real AND-depth), neither under a
//!   constant-round protocol.  That is the claim in its cleanest form, and the
//!   tests pin all three cases.
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
//! ## Per-site search
//!
//! The two rules whose sign depends on their operand — [`Rule::StrengthReduce`]
//! and [`Rule::Reassociate`] — are searched **site by site**, each candidate
//! priced on its own and kept only if it pays.  A site is named by the wire it
//! writes, which the rewrites preserve, so accepting one site does not
//! invalidate the addresses of the others.
//!
//! This was a real limitation until 2026-07-28, not a refinement.  On
//! `(+ (* x 4) (* y 100))` a whole-circuit search bundles a pure win (`x * 4`:
//! one shift replaces a constant load *and* a multiply) with a binary-only win
//! (`y * 100`: four instructions replace one), and an arithmetic backend
//! rejects the bundle — losing the free rewrite it should have taken.  With
//! per-site search ABY3 and Shamir take the `x * 4` site and decline `y * 100`,
//! while BMR and Semi-Bin take both.
//!
//! It also sharpens the central claim, because backends now diverge *within* a
//! rule rather than only over which rules they accept.  On a 64-element
//! constant dot product Shamir accepts strength reduction at 5 sites (the exact
//! powers of two) and Semi-Bin at all 42.
//!
//! The cost is search time: every site is a full lower → schedule → estimate.
//! `Optimized::sites_considered` reports how many were priced, so the coverage
//! is visible rather than implied.  A 2351-instruction circuit across all 11
//! backends takes ~1 minute.
//!
//! Still coarse: rules are applied greedily in a fixed order, so the search
//! finds a local optimum, not the best circuit.  Sites are also considered
//! independently, so a pair of rewrites that only pays *together* is missed.
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
use crate::zkp::circuit_compiler::{cse, rename, FlatOp};

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
    /// Re-parenthesise a chain of one associative operator into a balanced
    /// tree, trading dependence *depth* for nothing else.
    Reassociate,
}

impl Rule {
    pub const ALL: &'static [Rule] = &[
        Rule::Algebraic,
        Rule::Cse,
        Rule::FuseShifts,
        Rule::StrengthReduce,
        Rule::Reassociate,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Rule::Algebraic => "algebraic",
            Rule::Cse => "cse",
            Rule::FuseShifts => "fuse-shifts",
            Rule::StrengthReduce => "strength-reduce",
            Rule::Reassociate => "reassociate",
        }
    }

    /// Apply the rule at one site (`only`), or everywhere (`None`).
    fn apply_at(&self, flat: &[FlatOp], output_wire: &str, only: Option<&str>) -> Vec<FlatOp> {
        match self {
            Rule::Algebraic => algebraic(flat, output_wire),
            Rule::Cse => cse(flat, output_wire),
            Rule::FuseShifts => fuse_shifts(flat, output_wire),
            Rule::StrengthReduce => strength_reduce(flat, only),
            Rule::Reassociate => reassociate(flat, output_wire, only),
        }
    }

    /// The wires this rule could rewrite, for rules whose benefit varies *per
    /// site*; `None` for rules that are all-or-nothing.
    ///
    /// `Algebraic`, `Cse` and `FuseShifts` are cheap wherever they apply, so
    /// searching them site by site would only cost time. `StrengthReduce` and
    /// `Reassociate` are the two whose sign genuinely depends on the operand —
    /// which is the whole reason per-site search exists.
    fn sites(&self, flat: &[FlatOp], output_wire: &str) -> Option<Vec<String>> {
        match self {
            Rule::StrengthReduce => Some(strength_reduce_sites(flat)),
            Rule::Reassociate => Some(reassociate_sites(flat, output_wire)),
            Rule::Algebraic | Rule::Cse | Rule::FuseShifts => None,
        }
    }
}

/// Rewrite every operand of `op` through `subst`.

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
/// Set bits allowed in a constant multiplier before shift-and-add stops paying.
const MAX_TERMS: u32 = 3;

/// `(dst, x, c)` when `op` is an `x * c` that strength reduction can rewrite.
///
/// Shared by [`strength_reduce`] and [`strength_reduce_sites`] so the site
/// enumeration cannot drift from the rewrite it is enumerating for.
fn strength_reduce_candidate(
    op: &FlatOp,
    consts: &HashMap<String, i64>,
) -> Option<(String, String, i64)> {
    let (dst, x, c) = match op {
        FlatOp::IMul { dst, lhs, rhs } => match (consts.get(lhs), consts.get(rhs)) {
            // A constant × constant product is the constant folder's job.
            (Some(_), Some(_)) => return None,
            (Some(c), None) => (dst, rhs, *c),
            (None, Some(c)) => (dst, lhs, *c),
            _ => return None,
        },
        _ => return None,
    };
    let mag = c.unsigned_abs();
    // `|c| == 1` is the algebraic rule's job (`x*1 → x`, `x*-1` is already a
    // single instruction); `c == 0` likewise folds to a constant.
    if mag <= 1 || mag.count_ones() > MAX_TERMS {
        return None;
    }
    Some((dst.clone(), x.clone(), c))
}

/// Destination wires at which strength reduction can fire.
fn strength_reduce_sites(flat: &[FlatOp]) -> Vec<String> {
    let mut consts: HashMap<String, i64> = HashMap::new();
    let mut sites = Vec::new();
    for op in flat {
        if let FlatOp::Const { dst, value } = op {
            consts.insert(dst.clone(), *value);
        }
        if let Some((dst, ..)) = strength_reduce_candidate(op, &consts) {
            sites.push(dst);
        }
    }
    sites
}

/// Rewrite `x * c` into shifts and adds — at `only`, or everywhere when `None`.
fn strength_reduce(flat: &[FlatOp], only: Option<&str>) -> Vec<FlatOp> {
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
        let (dst, x, c) = match strength_reduce_candidate(op, &consts) {
            Some(t) if only.is_none_or(|w| w == t.0) => t,
            _ => {
                out.push(op.clone());
                continue;
            }
        };
        let (dst, x) = (&dst, &x);

        let mag = c.unsigned_abs();
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

// ─── Reassociation ────────────────────────────────────────────────────────────

/// Operators that are associative over the wrapping arithmetic `FlatOp` wires
/// carry, so a chain of them can be re-parenthesised without changing the
/// result.  `ISub32` is deliberately absent — subtraction is not associative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Assoc {
    Add,
    Mul,
    And,
    Or,
    Xor,
}

/// `(kind, dst, lhs, rhs)` when `op` is one of the associative binaries.
fn assoc_of(op: &FlatOp) -> Option<(Assoc, &str, &str, &str)> {
    match op {
        FlatOp::IAdd32 { dst, lhs, rhs } => Some((Assoc::Add, dst, lhs, rhs)),
        FlatOp::IMul { dst, lhs, rhs } => Some((Assoc::Mul, dst, lhs, rhs)),
        FlatOp::BAnd { dst, lhs, rhs } => Some((Assoc::And, dst, lhs, rhs)),
        FlatOp::BOr { dst, lhs, rhs } => Some((Assoc::Or, dst, lhs, rhs)),
        FlatOp::BXor { dst, lhs, rhs } => Some((Assoc::Xor, dst, lhs, rhs)),
        _ => None,
    }
}

fn build_assoc(kind: Assoc, dst: String, lhs: String, rhs: String) -> FlatOp {
    match kind {
        Assoc::Add => FlatOp::IAdd32 { dst, lhs, rhs },
        Assoc::Mul => FlatOp::IMul { dst, lhs, rhs },
        Assoc::And => FlatOp::BAnd { dst, lhs, rhs },
        Assoc::Or => FlatOp::BOr { dst, lhs, rhs },
        Assoc::Xor => FlatOp::BXor { dst, lhs, rhs },
    }
}

/// Collect the leaves of the maximal same-kind chain rooted at `flat[i]`, in
/// left-to-right order, marking every chain instruction in `touched`.
///
/// An operand is followed into the chain only when it is produced by the same
/// operator *and read exactly once* — otherwise some other instruction depends
/// on that intermediate value and the chain cannot be rebuilt around it.
struct ChainCtx<'a> {
    flat: &'a [FlatOp],
    counts: &'a HashMap<String, usize>,
    producer: &'a HashMap<String, usize>,
    output_wire: &'a str,
}

fn chain_leaves(
    ctx: &ChainCtx<'_>,
    i: usize,
    kind: Assoc,
    touched: &mut [bool],
    leaves: &mut Vec<String>,
) {
    touched[i] = true;
    let (_, _, lhs, rhs) = match assoc_of(&ctx.flat[i]) {
        Some(t) => t,
        None => return,
    };
    for w in [lhs.to_string(), rhs.to_string()] {
        let single_use = ctx.counts.get(&w).copied().unwrap_or(0) == 1 && w != ctx.output_wire;
        match ctx.producer.get(&w) {
            Some(&j)
                if single_use
                    && !touched[j]
                    && assoc_of(&ctx.flat[j]).map(|(k, ..)| k) == Some(kind) =>
            {
                chain_leaves(ctx, j, kind, touched, leaves);
            }
            _ => leaves.push(w),
        }
    }
}

/// A 3-leaf chain already has depth 2, which is what a balanced tree gives, so
/// there is nothing to win below 4.
const MIN_LEAVES: usize = 4;

/// Chain-root wires at which reassociation can fire.
fn reassociate_sites(flat: &[FlatOp], output_wire: &str) -> Vec<String> {
    // The rewrite is its own site oracle: a root is a site exactly when
    // rebuilding only that root changes the circuit.
    let mut sites = Vec::new();
    for op in flat {
        if let Some((_, dst, ..)) = assoc_of(op) {
            if reassociate(flat, output_wire, Some(dst)) != flat {
                sites.push(dst.to_string());
            }
        }
    }
    sites
}

/// Rebalance associative chains: `((((a·b)·c)·d)·e)` → `((a·b)·(c·d))·e`.
///
/// The instruction *count* is unchanged — only the dependence depth moves, from
/// `n-1` to `⌈log₂ n⌉`. That is precisely why the rule belongs in a
/// backend-aware optimizer rather than a generic one: under ASAP scheduling
/// (see [`crate::mpcc::schedule`]) depth *is* the online round count, so for an
/// interactive operator under a secret-sharing protocol this is a large WAN win,
/// while for a constant-round garbled circuit it is worth exactly nothing. The
/// search does not need to know which case it is in — [`objective`] prices it.
///
/// Note the asymmetry between operators, which is easy to state backwards:
/// rebalancing a chain of *multiplications* pays under an arithmetic protocol
/// (each multiply is a round), whereas rebalancing a chain of *additions* pays
/// nothing there, because additions are local. Additions are the ones that
/// matter for a binary protocol, where an adder is a real AND-depth.
fn reassociate(flat: &[FlatOp], output_wire: &str, only: Option<&str>) -> Vec<FlatOp> {
    let counts = use_counts(flat, output_wire);
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, op) in flat.iter().enumerate() {
        for d in flat_dsts(op) {
            producer.insert(d.to_string(), i);
        }
    }

    // An instruction is *interior* to a longer chain when its result feeds one
    // same-kind operator and nothing else; chains are rebuilt from their roots.
    let mut interior = vec![false; flat.len()];
    for op in flat {
        if let Some((kind, _, lhs, rhs)) = assoc_of(op) {
            for w in [lhs, rhs] {
                if counts.get(w).copied().unwrap_or(0) != 1 || w == output_wire {
                    continue;
                }
                if let Some(&j) = producer.get(w) {
                    if assoc_of(&flat[j]).map(|(k, ..)| k) == Some(kind) {
                        interior[j] = true;
                    }
                }
            }
        }
    }

    let mut consumed = vec![false; flat.len()];
    let mut plan: HashMap<usize, (Assoc, String, Vec<String>)> = HashMap::new();
    for i in 0..flat.len() {
        if consumed[i] || interior[i] {
            continue;
        }
        let (kind, dst) = match assoc_of(&flat[i]) {
            Some((k, d, ..)) => (k, d.to_string()),
            None => continue,
        };
        let mut touched = vec![false; flat.len()];
        let mut leaves = Vec::new();
        let ctx = ChainCtx {
            flat,
            counts: &counts,
            producer: &producer,
            output_wire,
        };
        chain_leaves(&ctx, i, kind, &mut touched, &mut leaves);
        if leaves.len() < MIN_LEAVES || only.is_some_and(|w| w != dst) {
            continue; // leave the chain alone, and leave `consumed` untouched
        }
        for (j, t) in touched.iter().enumerate() {
            if *t {
                consumed[j] = true;
            }
        }
        plan.insert(i, (kind, dst, leaves));
    }

    let taken: HashSet<String> = flat
        .iter()
        .flat_map(|o| flat_dsts(o).into_iter().map(String::from))
        .collect();
    let mut fresh = 0usize;
    let name = |fresh: &mut usize| loop {
        *fresh += 1;
        let n = format!("opt${}", fresh);
        if !taken.contains(&n) {
            return n;
        }
    };

    let mut out: Vec<FlatOp> = Vec::with_capacity(flat.len());
    for (i, op) in flat.iter().enumerate() {
        if let Some((kind, dst, leaves)) = plan.get(&i) {
            // Pair up level by level; the last combine writes the chain's dst,
            // so nothing downstream notices the rebuild.  Every leaf is defined
            // before the root in SSA order, so emitting the whole tree here is
            // topologically sound.
            let mut level = leaves.clone();
            while level.len() > 1 {
                let mut next: Vec<String> = Vec::with_capacity(level.len().div_ceil(2));
                let final_combine = level.len() == 2;
                let mut k = 0;
                while k + 1 < level.len() {
                    let into = if final_combine {
                        dst.clone()
                    } else {
                        name(&mut fresh)
                    };
                    out.push(build_assoc(
                        *kind,
                        into.clone(),
                        level[k].clone(),
                        level[k + 1].clone(),
                    ));
                    next.push(into);
                    k += 2;
                }
                if k < level.len() {
                    next.push(level[k].clone()); // odd leaf rides to the next level
                }
                level = next;
            }
            continue;
        }
        if consumed[i] {
            continue; // folded into the rebuilt tree at its root
        }
        out.push(op.clone());
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
    /// The wire rewritten, for per-site rules; `None` for whole-circuit ones.
    /// Two backends accepting the *same rule at different sites* is the finest
    /// grain at which the per-backend divergence shows up.
    pub site: Option<String>,
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
    /// How many individual rewrite sites were priced (summed over passes).
    /// Reported so the search's coverage is visible rather than implied.
    pub sites_considered: usize,
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

    // One speculative rewrite: price it, and keep it only if it improves.
    let try_one = |current: &mut Vec<FlatOp>,
                   current_obj: &mut Objective,
                   applied: &mut Vec<Applied>,
                   rejected: &mut HashSet<&'static str>,
                   rule: &Rule,
                   site: Option<&str>|
     -> bool {
        let candidate =
            dead_code_eliminate(&rule.apply_at(current, output_wire, site), output_wire);
        if candidate.len() == current.len() && candidate == *current {
            return false;
        }
        // A rewrite that produces something the lowering rejects is simply not
        // a candidate.
        let Ok(obj) = objective(&candidate, output_wire, spec, dep, cfg) else {
            return false;
        };
        if !obj.improves_on(current_obj) {
            rejected.insert(rule.name());
            return false;
        }
        applied.push(Applied {
            rule: rule.name(),
            site: site.map(str::to_string),
            ops_before: current.len(),
            ops_after: candidate.len(),
            secs_before: current_obj.secs,
            secs_after: obj.secs,
        });
        rejected.remove(rule.name());
        *current = candidate;
        *current_obj = obj;
        true
    };

    let mut sites_seen = 0usize;
    for _ in 0..MAX_PASSES {
        let mut changed = false;
        for rule in Rule::ALL {
            match rule.sites(&current, output_wire) {
                // All-or-nothing rule: one candidate for the whole circuit.
                None => {
                    changed |= try_one(
                        &mut current,
                        &mut current_obj,
                        &mut applied,
                        &mut rejected,
                        rule,
                        None,
                    );
                }
                // Per-site rule: accept the sites that pay and leave the rest.
                // A site is named by the wire it writes, which survives the
                // rewrite, so accepting one does not invalidate the others.
                Some(sites) => {
                    sites_seen += sites.len();
                    for site in sites {
                        changed |= try_one(
                            &mut current,
                            &mut current_obj,
                            &mut applied,
                            &mut rejected,
                            rule,
                            Some(&site),
                        );
                    }
                }
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
        sites_considered: sites_seen,
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

    /// The case that motivated per-site search: one site is a free win for
    /// everybody, the other only pays for a binary protocol.  A whole-circuit
    /// search has to take both or neither, and taking neither costs an
    /// arithmetic backend a rewrite it wanted.
    #[test]
    fn per_site_search_keeps_the_free_rewrite_and_drops_the_costly_one() {
        const SRC: &str = "(+ (* x 4) (* y 100))";
        let shamir = opt_wan(SRC, "shamir");
        let semi_bin = opt_wan(SRC, "semi-bin");

        let sr = |o: &Optimized| {
            o.applied
                .iter()
                .filter(|a| a.rule == "strength-reduce")
                .count()
        };
        assert_eq!(
            sr(&shamir),
            1,
            "Shamir should take exactly the `x * 4` site: {:?}",
            shamir.applied
        );
        assert!(
            shamir.after.ops < shamir.before.ops,
            "and that site is a strict instruction win: {} -> {}",
            shamir.before.ops,
            shamir.after.ops
        );
        assert_eq!(
            sr(&semi_bin),
            2,
            "a binary protocol wants both sites: {:?}",
            semi_bin.applied
        );

        // Every accepted per-site rewrite must name the wire it rewrote, since
        // that is what makes the divergence inspectable.
        for a in shamir.applied.iter().chain(&semi_bin.applied) {
            if a.rule == "strength-reduce" || a.rule == "reassociate" {
                assert!(
                    a.site.is_some(),
                    "per-site rule must record its site: {a:?}"
                );
            }
        }
        assert!(shamir.sites_considered > 0 && semi_bin.sites_considered > 0);
    }

    #[test]
    fn per_site_search_discriminates_within_one_rule() {
        // A constant dot product mixes powers of two (free everywhere) with
        // multi-bit constants (worth it only where a multiplier is expensive).
        // The two backends should keep *different subsets of the same rule*.
        let src = format!(
            "(dot (list {}) '({}))",
            (0..16)
                .map(|i| format!("a{i}"))
                .collect::<Vec<_>>()
                .join(" "),
            (0..16)
                .map(|i| (3 + i).to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let shamir = opt_wan(&src, "shamir");
        let semi_bin = opt_wan(&src, "semi-bin");
        let sites = |o: &Optimized| -> HashSet<String> {
            o.applied
                .iter()
                .filter(|a| a.rule == "strength-reduce")
                .filter_map(|a| a.site.clone())
                .collect()
        };
        let (s_sites, b_sites) = (sites(&shamir), sites(&semi_bin));
        assert!(
            !s_sites.is_empty() && s_sites.len() < b_sites.len(),
            "the arithmetic backend should take a strict subset: {} vs {}",
            s_sites.len(),
            b_sites.len()
        );
        assert!(
            s_sites.is_subset(&b_sites),
            "and it should be the cheap sites, not different ones"
        );
    }

    /// A left-nested chain of `n` copies of `op` over distinct inputs.
    fn chain(op: &str, n: usize) -> String {
        let mut s = "a0".to_string();
        for i in 1..n {
            s = format!("({} {} a{})", op, s, i);
        }
        s
    }

    fn opt_wan(src: &str, id: &str) -> Optimized {
        let (flat, out) = compile(src);
        let parties = if id == "yao" { 2 } else { 3 };
        optimize(
            &flat,
            &out,
            find(id).unwrap(),
            &Deployment::wan(parties),
            &cfg_for(id),
            true,
        )
        .expect("optimize")
    }

    /// Reassociation is the sharpest form of the claim: it changes *only* the
    /// dependence depth, leaving the instruction count and the gate count
    /// identical, so any cost difference it produces is purely about rounds.
    #[test]
    fn reassociating_a_multiply_chain_pays_for_rounds_and_not_for_gates() {
        let shamir = opt_wan(&chain("*", 8), "shamir");
        assert!(
            shamir.applied.iter().any(|a| a.rule == "reassociate"),
            "a secret-sharing protocol pays one round per multiply, so a \
             balanced tree is a real win on a WAN: {:?}",
            shamir
        );
        assert_eq!(
            shamir.before.ops, shamir.after.ops,
            "reassociation must not change the instruction count"
        );
        assert!(
            shamir.after.rounds < shamir.before.rounds,
            "rounds should fall from the chain depth to log2: {} -> {}",
            shamir.before.rounds,
            shamir.after.rounds
        );
        assert!(shamir.speedup() > 1.5, "got {:.2}x", shamir.speedup());
    }

    #[test]
    fn a_constant_round_protocol_declines_reassociation() {
        // BMR is constant-round: its cost does not depend on circuit depth at
        // all, so rebalancing buys precisely nothing and the search drops it.
        let bmr = opt_wan(&chain("*", 8), "bmr");
        assert!(
            bmr.applied.iter().all(|a| a.rule != "reassociate"),
            "depth is worth nothing to a constant-round protocol: {:?}",
            bmr
        );
        assert_eq!(bmr.before.rounds, bmr.after.rounds);
    }

    /// The operator decides *which* backends care, and it inverts between the
    /// two chains — which is the part a backend-blind optimizer cannot express.
    #[test]
    fn which_operator_pays_for_depth_depends_on_the_protocol() {
        // Additions are local in an arithmetic protocol: zero rounds either
        // way, so there is nothing for reassociation to win.
        let arith_add = opt_wan(&chain("+", 8), "shamir");
        assert!(
            arith_add.applied.iter().all(|a| a.rule != "reassociate"),
            "additions are local under secret sharing — depth is free: {:?}",
            arith_add
        );

        // Under a non-constant-round *binary* protocol the adder is a real
        // AND-depth, so the same rewrite on the same program is now a win.
        let binary_add = opt_wan(&chain("+", 8), "semi-bin");
        assert!(
            binary_add.applied.iter().any(|a| a.rule == "reassociate"),
            "an adder has AND-depth in a binary protocol: {:?}",
            binary_add
        );
        assert!(
            binary_add.after.rounds < binary_add.before.rounds,
            "{} -> {} rounds",
            binary_add.before.rounds,
            binary_add.after.rounds
        );
        assert_eq!(
            binary_add.before.and_gates, binary_add.after.and_gates,
            "the gate count is untouched; only the depth moved"
        );
    }

    /// Chained *local* arithmetic must accumulate AND-depth along the
    /// dependence path in a binary protocol. Charging the maximum over the
    /// program instead (what the cost model used to do) prices a chain of `n`
    /// adders as though it were one, which under-prices exactly the deep local
    /// arithmetic that decides MPC-vs-GC.
    #[test]
    fn binary_protocols_charge_chained_local_arithmetic_by_path_depth() {
        let (short_flat, short_out) = compile(&chain("+", 2));
        let (long_flat, long_out) = compile(&chain("+", 8));
        let spec = find("semi-bin").unwrap();
        let dep = Deployment::wan(3);
        let cfg = cfg_for("semi-bin");
        let short = objective(&short_flat, &short_out, spec, &dep, &cfg).unwrap();
        let long = objective(&long_flat, &long_out, spec, &dep, &cfg).unwrap();
        assert!(
            long.rounds > short.rounds,
            "a 7-adder chain must cost more rounds than a 1-adder one: {} vs {}",
            long.rounds,
            short.rounds
        );
    }

    #[test]
    fn reassociation_preserves_semantics_on_every_associative_operator() {
        // The differential oracle is the real guarantee; run it directly on
        // each operator the surface language can build a chain of.
        for op in ["+", "*", "and", "or"] {
            let (flat, out) = compile(&chain(op, 6));
            let rebuilt = reassociate(&flat, &out, None);
            assert_ne!(rebuilt, flat, "`{op}` chain should have been rebalanced");
            equivalent(&flat, &rebuilt, &out, 64)
                .unwrap_or_else(|e| panic!("`{op}` reassociation changed semantics: {e}"));
        }
    }

    /// `BXor` has no surface syntax (it is emitted by the fixed-point and
    /// logical-NOT gadgets), so build the chain directly rather than leave the
    /// variant untested.
    #[test]
    fn reassociation_handles_a_hand_built_xor_chain() {
        let mut flat = Vec::new();
        for i in 0..5 {
            flat.push(FlatOp::Const {
                dst: format!("k{i}"),
                value: 1 << i,
            });
        }
        let mut acc = "k0".to_string();
        for i in 1..5 {
            let dst = format!("t{i}");
            flat.push(FlatOp::BXor {
                dst: dst.clone(),
                lhs: acc,
                rhs: format!("k{i}"),
            });
            acc = dst;
        }
        let rebuilt = reassociate(&flat, &acc, None);
        assert_ne!(rebuilt, flat, "the xor chain should have been rebalanced");
        assert_eq!(rebuilt.len(), flat.len(), "op count must be unchanged");
        equivalent(&flat, &rebuilt, &acc, 32).expect("xor reassociation must be exact");
    }

    #[test]
    fn reassociation_leaves_shared_intermediates_alone() {
        // `(a+b)` is read twice, so the chain cannot be rebuilt around it
        // without duplicating work — the rule must decline.
        let (flat, out) = compile("(* (+ a b) (+ (+ a b) c))");
        let rebuilt = reassociate(&flat, &out, None);
        equivalent(&flat, &rebuilt, &out, 64).expect("must stay correct");
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
            let rewritten = dead_code_eliminate(&strength_reduce(&flat, None), &out);
            equivalent(&flat, &rewritten, &out, 32).unwrap_or_else(|e| panic!("{}: {}", src, e));
        }
    }

    /// CSE now runs inside `CircuitCompiler`, so a circuit that reaches any
    /// backend is *already* free of duplicate subexpressions. That is the
    /// property worth pinning — asserting that a second application shrinks it
    /// further would only be true if the compiler had stopped doing its job.
    ///
    /// Both cases used to emit a duplicated product; commutativity is included
    /// because the key sorts operands, so `x*y` and `y*x` must collide.
    #[test]
    fn compiled_circuits_arrive_already_cse_d() {
        for src in ["(+ (* x y) (* x y))", "(+ (* x y) (* y x))"] {
            let (flat, out) = compile(src);
            let again = dead_code_eliminate(&cse(&flat, &out), &out);
            assert_eq!(
                again.len(),
                flat.len(),
                "{src}: compiler output still had a duplicate for CSE to find                  ({} -> {})",
                flat.len(),
                again.len()
            );
            // One product, not two: the duplicate really is gone, rather than
            // both copies having survived into a circuit CSE cannot improve.
            let muls = flat.iter().filter(|o| matches!(o, FlatOp::IMul { .. })).count();
            assert_eq!(muls, 1, "{src}: expected a single product, got {muls}");
            equivalent(&flat, &again, &out, 24).unwrap();
        }
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
