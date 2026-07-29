//! Executing a compiled circuit on a real MPC backend, level by level.
//!
//! This is the bridge that makes the *compiled* path comparable to
//! hand-written code. [`crate::mpcc::schedule`] already proves, analytically,
//! that ASAP levelling collapses a layer's independent operations into one
//! round; this runs the resulting circuit against the same
//! [`Backend`] the hand-written gadgets use, so the batching stops being a
//! property of a cost model and becomes messages on a wire that
//! [`crate::bench::comm`] can count.
//!
//! # How it batches
//!
//! Every `FlatOp` is placed at its ASAP level: a **local** op (add, sub,
//! constant) inherits the level of its latest operand and costs nothing; an
//! **interactive** op (multiply, comparison) sits one level after its latest
//! operand. All interactive ops of the same kind at the same level are then
//! issued through the backend's *batched* primitive — one `mul_batch`, one
//! `lt_batch` — so a 512-wide layer costs the same number of rounds as a
//! 1-wide one.
//!
//! No vectorized IR is involved and the source program says nothing about
//! batching. That is the point: the user writes a scalar expression and the
//! schedule finds the parallelism, whereas the hand-written route only batches
//! where its author remembered to call `mul_batch`.
//!
//! # Scope
//!
//! The arithmetic subset a neural network needs: constants, add, sub,
//! multiply, signed comparison and mux. Anything else returns an error rather
//! than silently falling back to a per-element loop — a silent fallback would
//! quietly reintroduce exactly the round-serialization this exists to avoid.

use std::collections::HashMap;

use crate::crypto::types::Fr;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::zkp::circuit_compiler::FlatOp;

/// Does this op cost a communication round on a secret-sharing backend?
fn is_interactive(op: &FlatOp) -> bool {
    matches!(
        op,
        FlatOp::IMul { .. } | FlatOp::ICmpSlt { .. } | FlatOp::ICmpUlt { .. } | FlatOp::Mux { .. }
    )
}

/// What the executor did, for a benchmark to report alongside the backend's
/// own communication counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExecStats {
    /// ASAP levels that contained at least one interactive op — the round
    /// count the schedule predicted, now actually executed.
    pub rounds: usize,
    /// Batched primitive calls issued (one per kind per level).
    pub batch_calls: usize,
    /// Interactive operations executed.
    pub interactive_ops: usize,
}

/// How the executor issues interactive operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Batching {
    /// Group by ASAP level and issue one batched primitive per kind per level.
    #[default]
    Level,
    /// One operation at a time, in program order.
    ///
    /// This is the control the level-batched mode is measured against, and it
    /// is what an executor written the obvious way does — walk the circuit and
    /// call the backend for each op. Nothing about the *circuit* differs; only
    /// how many times the parties stop and talk.
    Serial,
}

/// Evaluate `flat` on `be`, given shares for its free wires.
///
/// `inputs` must supply every wire the circuit reads and does not define.
/// Returns the share of `output_wire` plus what it cost.
pub fn run_flat(
    be: &mut dyn Backend,
    flat: &[FlatOp],
    inputs: &HashMap<String, ShareRef>,
    output_wire: &str,
    nbits: usize,
) -> Result<(ShareRef, ExecStats), String> {
    run_flat_with(be, flat, inputs, output_wire, nbits, Batching::Level)
}

/// [`run_flat`] with an explicit batching mode.
pub fn run_flat_with(
    be: &mut dyn Backend,
    flat: &[FlatOp],
    inputs: &HashMap<String, ShareRef>,
    output_wire: &str,
    nbits: usize,
    batching: Batching,
) -> Result<(ShareRef, ExecStats), String> {
    if batching == Batching::Serial {
        return run_serial(be, flat, inputs, output_wire, nbits);
    }
    // ── ASAP levels ──
    let mut level: HashMap<&str, usize> = HashMap::new();
    for w in inputs.keys() {
        level.insert(w.as_str(), 0);
    }
    let mut op_level = vec![0usize; flat.len()];
    for (i, op) in flat.iter().enumerate() {
        let ready = op
            .inputs()
            .iter()
            .map(|w| *level.get(*w).unwrap_or(&0))
            .max()
            .unwrap_or(0);
        let lv = if is_interactive(op) { ready + 1 } else { ready };
        op_level[i] = lv;
        level.insert(op.dst(), lv);
    }
    let max_level = op_level.iter().copied().max().unwrap_or(0);

    // ── Execute level by level ──
    let mut env: HashMap<String, ShareRef> = inputs.clone();
    let mut stats = ExecStats::default();

    for lv in 0..=max_level {
        // Interactive ops first, then locals.
        //
        // A local op inherits the level of its *latest* operand, so a local at
        // level `lv` necessarily has an operand at level `lv`, and that operand
        // can only be interactive (a local would have inherited a lower level).
        // Locals must therefore run *after* this level's interactive results
        // exist. Conversely an interactive op at `lv` has all operands at
        // `lv-1` by construction, so it never waits on a local here.
        let here: Vec<usize> = (0..flat.len())
            .filter(|&i| op_level[i] == lv && is_interactive(&flat[i]))
            .collect();
        if !here.is_empty() {
            stats.rounds += 1;
            stats.interactive_ops += here.len();

            let muls: Vec<usize> = here
                .iter()
                .copied()
                .filter(|&i| matches!(flat[i], FlatOp::IMul { .. }))
                .collect();
            let cmps: Vec<usize> = here
                .iter()
                .copied()
                .filter(|&i| matches!(flat[i], FlatOp::ICmpSlt { .. } | FlatOp::ICmpUlt { .. }))
                .collect();
            let muxes: Vec<usize> = here
                .iter()
                .copied()
                .filter(|&i| matches!(flat[i], FlatOp::Mux { .. }))
                .collect();

            if !cmps.is_empty() {
                let pairs: Vec<(ShareRef, ShareRef)> = cmps
                    .iter()
                    .map(|&i| {
                        let ins = flat[i].inputs();
                        Ok((get(&env, ins[0])?, get(&env, ins[1])?))
                    })
                    .collect::<Result<_, String>>()?;
                let out = be.lt_batch(&pairs);
                stats.batch_calls += 1;
                for (&i, mut r) in cmps.iter().zip(out) {
                    r.nbits = nbits;
                    env.insert(flat[i].dst().to_string(), r);
                }
            }
            if !muls.is_empty() {
                let pairs: Vec<(ShareRef, ShareRef)> = muls
                    .iter()
                    .map(|&i| {
                        let ins = flat[i].inputs();
                        Ok((get(&env, ins[0])?, get(&env, ins[1])?))
                    })
                    .collect::<Result<_, String>>()?;
                let out = be.mul_batch(&pairs);
                stats.batch_calls += 1;
                for (&i, mut r) in muls.iter().zip(out) {
                    r.nbits = nbits;
                    env.insert(flat[i].dst().to_string(), r);
                }
            }
            if !muxes.is_empty() {
                // mux(c, t, e) = e + c·(t − e): one batched multiply for the level.
                let mut diffs = Vec::with_capacity(muxes.len());
                for &i in &muxes {
                    let ins = flat[i].inputs();
                    let (c, t, e) = (get(&env, ins[0])?, get(&env, ins[1])?, get(&env, ins[2])?);
                    diffs.push((c, be.sub(&t, &e)));
                }
                let sel = be.mul_batch(&diffs);
                stats.batch_calls += 1;
                for (&i, s) in muxes.iter().zip(sel) {
                    let e = get(&env, flat[i].inputs()[2])?;
                    let mut r = be.add(&e, &s);
                    r.nbits = nbits;
                    env.insert(flat[i].dst().to_string(), r);
                }
            }
        }

        // Locals for this level, in program order (which is topological).
        for (i, op) in flat.iter().enumerate() {
            if op_level[i] != lv || is_interactive(op) {
                continue;
            }
            let v = eval_local(be, op, &env, nbits)?;
            env.insert(op.dst().to_string(), v);
        }
    }

    let out = get(&env, output_wire)?;
    Ok((out, stats))
}

fn get(env: &HashMap<String, ShareRef>, w: &str) -> Result<ShareRef, String> {
    env.get(w)
        .cloned()
        .ok_or_else(|| format!("circuit reads undefined wire `{w}`"))
}

/// Communication-free ops, evaluated in place.
fn eval_local(
    be: &mut dyn Backend,
    op: &FlatOp,
    env: &HashMap<String, ShareRef>,
    nbits: usize,
) -> Result<ShareRef, String> {
    Ok(match op {
        FlatOp::Const { value, .. } => {
            let v = (*value as i128).rem_euclid(1i128 << nbits) as Fr;
            let mut s = be.share_public(v, nbits);
            s.nbits = nbits;
            s
        }
        FlatOp::IAdd32 { lhs, rhs, .. } => {
            let (a, b) = (get(env, lhs)?, get(env, rhs)?);
            be.add(&a, &b)
        }
        FlatOp::ISub32 { lhs, rhs, .. } => {
            let (a, b) = (get(env, lhs)?, get(env, rhs)?);
            be.sub(&a, &b)
        }
        other => {
            return Err(format!(
                "mpcc::exec does not implement {other:?} — it would have to fall back to a \
                 per-element loop, which is the round serialization this executor exists to avoid"
            ))
        }
    })
}

/// One interactive operation at a time, in program order — the control.
fn run_serial(
    be: &mut dyn Backend,
    flat: &[FlatOp],
    inputs: &HashMap<String, ShareRef>,
    output_wire: &str,
    nbits: usize,
) -> Result<(ShareRef, ExecStats), String> {
    let mut env: HashMap<String, ShareRef> = inputs.clone();
    let mut stats = ExecStats::default();
    for op in flat {
        let v = if is_interactive(op) {
            stats.rounds += 1;
            stats.batch_calls += 1;
            stats.interactive_ops += 1;
            let ins = op.inputs();
            match op {
                FlatOp::IMul { .. } => {
                    let (a, b) = (get(&env, ins[0])?, get(&env, ins[1])?);
                    let mut r = be.mul(&a, &b);
                    r.nbits = nbits;
                    r
                }
                FlatOp::ICmpSlt { .. } | FlatOp::ICmpUlt { .. } => {
                    let (a, b) = (get(&env, ins[0])?, get(&env, ins[1])?);
                    let mut r = be.lt(&a, &b);
                    r.nbits = nbits;
                    r
                }
                FlatOp::Mux { .. } => {
                    let (c, t, e) = (get(&env, ins[0])?, get(&env, ins[1])?, get(&env, ins[2])?);
                    let d = be.sub(&t, &e);
                    let sel = be.mul(&c, &d);
                    let mut r = be.add(&e, &sel);
                    r.nbits = nbits;
                    r
                }
                other => return Err(format!("mpcc::exec: unsupported interactive op {other:?}")),
            }
        } else {
            eval_local(be, op, &env, nbits)?
        };
        env.insert(op.dst().to_string(), v);
    }
    let out = get(&env, output_wire)?;
    Ok((out, stats))
}
