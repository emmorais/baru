//! Executing one circuit across **two backends at once**.
//!
//! [`crate::mpcc::assign`] decides which operation *kinds* belong on which
//! side and prices the conversions at the crossings, but until now that was a
//! costing result: nothing executed a mixed assignment. This module runs one
//! circuit with some operations under MPC and others under FHE, inserting the
//! scheme switch at each crossing, so the assignment's prediction can be
//! checked against a measurement rather than believed.
//!
//! # The depth-1 constraint is the whole design
//!
//! The FHE side is BFV with **no relinearization**: a ciphertext$\times$
//! ciphertext product is a degree-2 ciphertext that decrypts under $s^2$ and
//! *cannot be multiplied again*. So the FHE side can evaluate
//! `mul` on two fresh ciphertexts and then any number of additions — which is
//! exactly the shape of a linear layer — and nothing deeper. When a
//! multiplication's operand is already a product, the executor moves that
//! operation back to MPC rather than failing, and counts the crossing.
//!
//! This is not a limitation being worked around; it is why the assignment is
//! interesting. FHE buys a linear layer with no preprocessing, and the switch
//! back to MPC is what makes the *next* layer possible.
//!
//! # Honest scope
//!
//! The MPC→FHE direction uses [`Backend::export_additive`], which only an
//! in-process simulator can satisfy — a real party can produce its own share
//! component and no one else's. The cryptography here is real (BFV at the
//! configured parameters, a genuine mask-and-open in the other direction) but
//! the **party separation of the bridge is simulated**. A deployed switch has
//! each party encrypt its own component; `crate::bench::cnn` does it that way.
//! Conversion *counts* and *bytes* reported here are therefore meaningful;
//! wall-clock for the switch is not a deployment number.

use std::collections::HashMap;

use crate::crypto::types::Fr;
use crate::hybrid::{fhe_to_mpc_share, mpc_share_to_fhe, FheAdapter};
use crate::interpreter::backend::{Backend, ShareRef};
use crate::zkp::circuit_compiler::FlatOp;

/// The arithmetic an FHE side must provide to host part of a circuit.
///
/// Deliberately separate from [`FheAdapter`], which is additive-only because
/// that is all a scheme switch needs. Hosting a *layer* needs one
/// multiplication, and needs to be asked how deep a ciphertext already is.
pub trait FheArith: FheAdapter {
    /// Ciphertext product. Callers must check [`Self::depth`] first.
    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;
    /// Multiplicative depth already spent. `mul` requires both operands at 0.
    fn depth(&self, ct: &Self::Ct) -> usize;
    /// Deepest product this side can evaluate.
    fn max_depth(&self) -> usize {
        1
    }
}

impl FheArith for crate::fhe::bfv::BfvAdapter<'_> {
    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        self.0.mul(a, b)
    }
    fn depth(&self, ct: &Self::Ct) -> usize {
        ct.key_pow().saturating_sub(1)
    }
}

/// Which side an operation runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Mpc,
    Fhe,
}

/// Translate an [`crate::mpcc::assign::Assignment`] into the kind-level
/// routing this executor takes, closing the loop from *what the cost model
/// decided* to *what actually runs*.
///
/// This is the kind-level decision only. Operations adjacent to the boundary
/// are placed by [`Placement::sink_to_fhe`], which is a separate and
/// principled step rather than a heuristic bolted on here.
///
/// The assignment has three sides; this executor has two. `Arith` and
/// `Binary` both map to [`Side::Mpc`], because both are evaluated by the same
/// secret-sharing backend here — the arithmetic/binary distinction is real in
/// the cost model but is not a *representation* boundary the executor has to
/// cross. Only `Fhe` is a different representation, and only it needs a
/// scheme switch.
pub fn from_assignment(a: &crate::mpcc::assign::Assignment) -> HashMap<String, Side> {
    use crate::mpcc::assign::Side as ASide;
    let map = |s: ASide| match s {
        ASide::Fhe => Side::Fhe,
        ASide::Arith | ASide::Binary => Side::Mpc,
    };
    a.by_kind
        .iter()
        .map(|(k, v)| ((*k).to_string(), map(*v)))
        .collect()
}

/// Where each operation runs.
///
/// [`crate::mpcc::assign`] decides per operation *kind* — every multiply goes
/// to the same side. That is coarse in a way that costs crossings: an
/// operation sitting between two FHE results, which the FHE side could host,
/// forces both of them back across the boundary just because its *kind* was
/// assigned to MPC. A per-operation refinement fixes that without changing
/// the kind-level decision.
#[derive(Debug, Default, Clone)]
pub struct Placement {
    by_kind: HashMap<String, Side>,
    by_wire: HashMap<String, Side>,
}

impl Placement {
    pub fn by_kind(map: HashMap<String, Side>) -> Self {
        Placement {
            by_kind: map,
            by_wire: HashMap::new(),
        }
    }

    fn side_of(&self, op: &FlatOp) -> Side {
        self.by_wire
            .get(op.dst())
            .copied()
            .or_else(|| self.by_kind.get(op_kind(op)).copied())
            .unwrap_or(Side::Mpc)
    }

    /// How many operations were moved off their kind's side.
    pub fn refinements(&self) -> usize {
        self.by_wire.len()
    }

    /// **Sink operations across the boundary when it saves a crossing.**
    ///
    /// The rule: an operation runs on FHE if the FHE side can host it, its
    /// depth budget allows it, and *every* operand is already produced on the
    /// FHE side. The last condition is what makes this safe to apply greedily
    /// — it never forces a fresh value across, it only avoids dragging values
    /// back. An operation with mixed operands stays where its kind put it,
    /// because the crossing is unavoidable there anyway.
    ///
    /// This is a boundary-*placement* decision rather than a protocol
    /// decision, and it is only expressible once execution is mixed: while the
    /// hybrid lives in hand-written orchestration the boundary is wherever the
    /// author drew it. Silph and SING assign per instruction but over MPC
    /// sharings only, so this particular move is not in their space.
    pub fn sink_to_fhe(mut self, flat: &[FlatOp]) -> Self {
        // Depth of each wire on the FHE side; `usize::MAX` marks "not on FHE".
        let mut depth: HashMap<&str, usize> = HashMap::new();
        for op in flat {
            let base = self.side_of(op);
            let ins = op.inputs();
            let operand_depths: Vec<Option<usize>> =
                ins.iter().map(|w| depth.get(*w).copied()).collect();
            let all_on_fhe = !ins.is_empty() && operand_depths.iter().all(|d| d.is_some());
            let max_in = operand_depths.iter().flatten().copied().max().unwrap_or(0);

            let hostable = matches!(op, FlatOp::IMul { .. } | FlatOp::IAdd32 { .. });
            let out_depth = match op {
                FlatOp::IMul { .. } => max_in + 1,
                _ => max_in,
            };
            // Depth-1: a product of products cannot stay on FHE.
            let fits = out_depth <= 1 && !(matches!(op, FlatOp::IMul { .. }) && max_in > 0);

            let side = if base == Side::Fhe {
                Side::Fhe
            } else if hostable && all_on_fhe && fits {
                // Every operand is already a ciphertext: hosting this here
                // avoids dragging them back.
                self.by_wire.insert(op.dst().to_string(), Side::Fhe);
                Side::Fhe
            } else {
                Side::Mpc
            };

            if side == Side::Fhe {
                depth.insert(op.dst(), out_depth);
            } else {
                depth.remove(op.dst());
            }
        }
        self
    }
}

/// A wire's current representation.
enum Val<C> {
    Mpc(ShareRef),
    Fhe(C),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MixedStats {
    pub mpc_ops: usize,
    pub fhe_ops: usize,
    /// Share → ciphertext conversions.
    pub to_fhe: usize,
    /// Ciphertext → share conversions (mask-and-open).
    pub to_mpc: usize,
    /// Bytes of ciphertext crossing the boundary.
    pub crossing_bytes: usize,
    /// Multiplications pushed back to MPC because the FHE side had no depth
    /// left. Reported because a silent fallback would make the FHE assignment
    /// look better than it is.
    pub demoted_muls: usize,
}

/// Evaluate `flat` with operations routed by `assignment` (keyed by operation
/// kind, e.g. `"mul"`), converting representations at every crossing.
pub fn run_mixed<A: FheArith>(
    be: &mut dyn Backend,
    fhe: &A,
    client_key: &[u8],
    flat: &[FlatOp],
    inputs: &HashMap<String, ShareRef>,
    output_wire: &str,
    nbits: usize,
    placement: &Placement,
) -> Result<(ShareRef, MixedStats), String> {
    let mut env: HashMap<String, Val<A::Ct>> = inputs
        .iter()
        .map(|(k, v)| (k.clone(), Val::Mpc(v.clone())))
        .collect();
    let mut st = MixedStats::default();

    for op in flat {
        let want = placement.side_of(op);

        // FHE hosts only `mul` and `add`, and `mul` only on fresh operands.
        let feasible = match (want, op) {
            (Side::Fhe, FlatOp::IMul { lhs, rhs, .. }) => {
                let d = |w: &String| match env.get(w) {
                    Some(Val::Fhe(c)) => fhe.depth(c),
                    _ => 0,
                };
                d(lhs) == 0 && d(rhs) == 0
            }
            (Side::Fhe, FlatOp::IAdd32 { .. }) => true,
            (Side::Fhe, _) => false,
            (Side::Mpc, _) => true,
        };
        let side = if feasible { want } else { Side::Mpc };
        if want == Side::Fhe && side == Side::Mpc && matches!(op, FlatOp::IMul { .. }) {
            st.demoted_muls += 1;
        }

        let out = match side {
            Side::Fhe => {
                st.fhe_ops += 1;
                let mut get_ct = |w: &String,
                                  env: &mut HashMap<String, Val<A::Ct>>,
                                  st: &mut MixedStats,
                                  be: &mut dyn Backend|
                 -> Result<A::Ct, String> {
                    match env.get(w) {
                        Some(Val::Fhe(c)) => Ok(c.clone()),
                        Some(Val::Mpc(s)) => {
                            let c = mpc_share_to_fhe(be, s, fhe, client_key);
                            st.to_fhe += 1;
                            st.crossing_bytes += fhe.ct_bytes(&c);
                            let cc = c.clone();
                            env.insert(w.clone(), Val::Fhe(c));
                            Ok(cc)
                        }
                        None => Err(format!("circuit reads undefined wire `{w}`")),
                    }
                };
                let ins: Vec<String> = op.inputs().iter().map(|s| s.to_string()).collect();
                let a = get_ct(&ins[0], &mut env, &mut st, be)?;
                let b = get_ct(&ins[1], &mut env, &mut st, be)?;
                Val::Fhe(match op {
                    FlatOp::IMul { .. } => fhe.mul(&a, &b),
                    FlatOp::IAdd32 { .. } => fhe.add(&a, &b),
                    other => return Err(format!("FHE side cannot host {other:?}")),
                })
            }
            Side::Mpc => {
                st.mpc_ops += 1;
                let mut get_sh = |w: &String,
                                  env: &mut HashMap<String, Val<A::Ct>>,
                                  st: &mut MixedStats,
                                  be: &mut dyn Backend|
                 -> Result<ShareRef, String> {
                    match env.get(w) {
                        Some(Val::Mpc(s)) => Ok(s.clone()),
                        Some(Val::Fhe(c)) => {
                            st.crossing_bytes += fhe.ct_bytes(c);
                            let s = fhe_to_mpc_share(fhe, client_key, c, be, nbits);
                            st.to_mpc += 1;
                            env.insert(w.clone(), Val::Mpc(s.clone()));
                            Ok(s)
                        }
                        None => Err(format!("circuit reads undefined wire `{w}`")),
                    }
                };
                Val::Mpc(eval_mpc(be, op, &mut env, &mut st, nbits, &mut get_sh)?)
            }
        };
        env.insert(op.dst().to_string(), out);
    }

    // The result leaves as a share, whichever side produced it.
    match env.remove(output_wire) {
        Some(Val::Mpc(s)) => Ok((s, st)),
        Some(Val::Fhe(c)) => {
            st.crossing_bytes += fhe.ct_bytes(&c);
            let s = fhe_to_mpc_share(fhe, client_key, &c, be, nbits);
            st.to_mpc += 1;
            Ok((s, st))
        }
        None => Err(format!("circuit never defines its output `{output_wire}`")),
    }
}

type GetShare<'a, C> = dyn FnMut(
        &String,
        &mut HashMap<String, Val<C>>,
        &mut MixedStats,
        &mut dyn Backend,
    ) -> Result<ShareRef, String>
    + 'a;

fn eval_mpc<C>(
    be: &mut dyn Backend,
    op: &FlatOp,
    env: &mut HashMap<String, Val<C>>,
    st: &mut MixedStats,
    nbits: usize,
    get: &mut GetShare<'_, C>,
) -> Result<ShareRef, String> {
    let ins: Vec<String> = op.inputs().iter().map(|s| s.to_string()).collect();
    Ok(match op {
        FlatOp::Const { value, .. } => {
            let v = (*value as i128).rem_euclid(1i128 << nbits) as Fr;
            be.share_public(v, nbits)
        }
        FlatOp::IAdd32 { .. } => {
            let (a, b) = (get(&ins[0], env, st, be)?, get(&ins[1], env, st, be)?);
            be.add(&a, &b)
        }
        FlatOp::ISub32 { .. } => {
            let (a, b) = (get(&ins[0], env, st, be)?, get(&ins[1], env, st, be)?);
            be.sub(&a, &b)
        }
        FlatOp::IMul { .. } => {
            let (a, b) = (get(&ins[0], env, st, be)?, get(&ins[1], env, st, be)?);
            let mut r = be.mul(&a, &b);
            r.nbits = nbits;
            r
        }
        FlatOp::ICmpSlt { .. } => {
            let (a, b) = (get(&ins[0], env, st, be)?, get(&ins[1], env, st, be)?);
            let mut r = be.lt(&a, &b);
            r.nbits = nbits;
            r
        }
        other => return Err(format!("mpcc::mixed does not implement {other:?}")),
    })
}

/// Compute a dot product of two secret vectors on the FHE side with the
/// **whole layer packed into one ciphertext per operand**.
///
/// The scalar executor above crosses one ciphertext per value, which is what
/// makes its crossing bytes overstate a real implementation: `mpcc::assign`
/// prices a switch as a whole layer in one ciphertext, and so does
/// `crate::bench::cnn`. This is that packing, so the executed cost and the
/// charged cost describe the same thing.
///
/// One vector is packed forward and the other reversed, so a single
/// polynomial multiplication puts $\sum_i a_i b_i$ in coefficient `d-1`
/// (`Bfv::matvec_output_position`). Cost: **two crossings out, one
/// multiplication, one masked opening back** — independent of the layer width.
///
/// # The plaintext modulus has to hold *un-reduced share products*
///
/// This is a real constraint of switching a secret-shared value into FHE, and
/// it is easy to miss. Each party's share is uniform mod `2^nbits`, so the
/// value the FHE side multiplies is not `a` but `a + k·2^nbits` for a small
/// `k` (the un-reduced sum of components). The product is still congruent to
/// `a·b` modulo `2^nbits` — which is why this is correct at all — but it is
/// far larger as an integer, and the plaintext modulus must hold the entire
/// sum over the layer or it wraps and the answer is silently wrong. We assert
/// the bound rather than trust it.
pub fn packed_dot(
    be: &mut dyn Backend,
    s: &crate::fhe::bfv::Bfv,
    client_key: &[u8],
    a: &[ShareRef],
    b: &[ShareRef],
    nbits: usize,
) -> Result<(ShareRef, MixedStats), String> {
    use rand::Rng;
    if a.len() != b.len() || a.is_empty() {
        return Err("packed_dot: vectors must be non-empty and equal length".into());
    }
    let d = a.len();
    let mut st = MixedStats::default();

    // Bound check: parties * 2^nbits per factor, squared, summed over d.
    let parties = be.export_additive(&a[0]).len().max(1) as u128;
    let factor = parties << nbits;
    let bound = (factor * factor).saturating_mul(d as u128);
    let t = 1u128 << s.params.log2_t;
    if bound >= t {
        return Err(format!(
            "packed_dot: a {d}-wide layer of {nbits}-bit shares across {parties} parties needs              a plaintext modulus above 2^{}, but t = 2^{}. The FHE side multiplies un-reduced              share sums, not the {nbits}-bit values.",
            128 - bound.leading_zeros(),
            s.params.log2_t
        ));
    }

    // Each party's component vector becomes one packed ciphertext; summing
    // them is the MPC->FHE switch, done once per operand instead of per value.
    let comps = |be: &mut dyn Backend, v: &[ShareRef]| -> Vec<Vec<i64>> {
        let per: Vec<Vec<Fr>> = v.iter().map(|s| be.export_additive(s)).collect();
        let np = per[0].len();
        (0..np)
            .map(|p| per.iter().map(|c| c[p] as i64).collect())
            .collect()
    };
    let a_comps = comps(be, a);
    let b_comps = comps(be, b);

    let mut ct_a: Option<crate::fhe::bfv::BfvCt> = None;
    for cv in &a_comps {
        let ct = s.encrypt_poly(client_key, &s.pack_input(cv));
        ct_a = Some(match ct_a {
            None => ct,
            Some(acc) => s.add(&acc, &ct),
        });
    }
    let mut ct_b: Option<crate::fhe::bfv::BfvCt> = None;
    for cv in &b_comps {
        let ct = s.encrypt_poly(client_key, &s.pack_weights(&[cv.clone()]));
        ct_b = Some(match ct_b {
            None => ct,
            Some(acc) => s.add(&acc, &ct),
        });
    }
    let ct_a = ct_a.expect("one operand");
    let ct_b = ct_b.expect("one operand");
    st.to_fhe = 2;
    st.crossing_bytes += s.ct_bytes(&ct_a) + s.ct_bytes(&ct_b);

    // One multiplication for the entire layer.
    let prod = s.mul(&ct_a, &ct_b);
    st.fhe_ops = 1;

    // Masked opening at the output coefficient, back into shares.
    let pos = s.matvec_output_position(d, 0);
    let t_mask = s.params.t() - 1;
    let mut rng = rand::thread_rng();
    let n_parties = a_comps.len();
    let masks: Vec<u64> = (0..n_parties).map(|_| rng.gen::<u64>() & t_mask).collect();
    let mut masked = prod;
    for &r in &masks {
        let mut m = vec![0u64; s.params.n];
        m[pos] = r;
        masked = s.add(&masked, &s.encrypt_poly(client_key, &m));
    }
    st.crossing_bytes += s.ct_bytes(&masked);
    let y = s.decrypt_poly(client_key, &masked)[pos];
    st.to_mpc = 1;

    let mut components: Vec<Fr> = Vec::with_capacity(n_parties);
    components.push((y as u128).wrapping_sub(masks[0] as u128));
    for &r in &masks[1..] {
        components.push(0u128.wrapping_sub(r as u128));
    }
    Ok((be.import_additive(&components, nbits), st))
}

fn op_kind(op: &FlatOp) -> &'static str {
    match op {
        FlatOp::IMul { .. } => "mul",
        FlatOp::IAdd32 { .. } => "add",
        FlatOp::ISub32 { .. } => "sub",
        FlatOp::ICmpSlt { .. } | FlatOp::ICmpUlt { .. } => "cmp-lt",
        FlatOp::Const { .. } => "const",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::bfv::{Bfv, BfvAdapter, BfvParams};
    use crate::interpreter::aby3_simulator::Aby3Simulator;
    use crate::interpreter::parse;
    use crate::zkp::circuit_compiler::CircuitCompiler;

    const NBITS: usize = 20;

    fn compile(src: &str) -> (Vec<FlatOp>, String) {
        let e = parse(src).expect("parse");
        let c = CircuitCompiler::new().compile_fhe(&e).expect("compile");
        (c.flat_ops, c.output_wire)
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

    /// A dot product with **both operands secret**, then a comparison — one
    /// neural-network layer and its ReLU condition. This is the shape
    /// `mpcc::assign` puts on FHE, and the point of the test is that the
    /// assignment is now *executable*: the mixed run must agree with the
    /// all-MPC run and with plaintext, bit for bit.
    #[test]
    fn a_mixed_assignment_computes_the_same_value_as_pure_mpc() {
        let n = 8;
        let mut dot = "(* a0 b0)".to_string();
        for i in 1..n {
            dot = format!("(+ {dot} (* a{i} b{i}))");
        }
        let (flat, out) = compile(&dot);

        let a: Vec<i64> = (0..n).map(|i| (i as i64 % 5) - 2).collect();
        let b: Vec<i64> = (0..n).map(|i| ((i * 3) as i64 % 7) - 3).collect();
        let want: i64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        let bind = |be: &mut Aby3Simulator| -> HashMap<String, ShareRef> {
            let mut m = HashMap::new();
            for i in 0..n {
                let av = (a[i] as i128).rem_euclid(1i128 << NBITS) as Fr;
                let bv = (b[i] as i128).rem_euclid(1i128 << NBITS) as Fr;
                m.insert(format!("a{i}"), be.share_secret(av, NBITS, 0));
                m.insert(format!("b{i}"), be.share_secret(bv, NBITS, 0));
            }
            m
        };

        let s = Bfv::new(BfvParams::TEST_SMALL);
        let (ck, _pk) = s.keygen();
        let fhe = BfvAdapter(&s);

        // All-MPC.
        let mut be = Aby3Simulator::new();
        let env = bind(&mut be);
        let (r_mpc, st_mpc) = run_mixed(
            &mut be,
            &fhe,
            &ck,
            &flat,
            &env,
            &out,
            NBITS,
            &Placement::default(),
        )
        .unwrap();
        let got_mpc = to_signed(be.reveal(&r_mpc, None));

        // Multiplications on FHE — the assignment the cost model returns for a
        // wide layer on a WAN.
        let mut asg = HashMap::new();
        asg.insert("mul".to_string(), Side::Fhe);
        asg.insert("add".to_string(), Side::Fhe);
        let mut be2 = Aby3Simulator::new();
        let env2 = bind(&mut be2);
        let (r_mix, st_mix) = run_mixed(
            &mut be2,
            &fhe,
            &ck,
            &flat,
            &env2,
            &out,
            NBITS,
            &Placement::by_kind(asg),
        )
        .unwrap();
        let got_mix = to_signed(be2.reveal(&r_mix, None));

        assert_eq!(got_mpc, want, "all-MPC route must match plaintext");
        assert_eq!(got_mix, want, "mixed route must match plaintext");
        assert_eq!(got_mix, got_mpc, "the two routes must agree");

        // The assignment actually took effect, and depth-1 held: every
        // multiply ran on FHE, none was demoted.
        assert_eq!(st_mpc.fhe_ops, 0, "the control must not touch FHE");
        assert!(st_mix.fhe_ops >= n, "expected the layer on FHE: {st_mix:?}");
        assert_eq!(st_mix.demoted_muls, 0, "depth-1 should suffice: {st_mix:?}");
        assert_eq!(
            st_mix.to_mpc, 1,
            "one switch back for the result: {st_mix:?}"
        );
    }

    /// Packing the crossing must give the same answer as crossing value by
    /// value, at a constant number of switches instead of one per input.
    ///
    /// The share width is small here on purpose: the FHE side multiplies
    /// *un-reduced* share sums, so the plaintext modulus has to hold the whole
    /// layer's products. That is a genuine cost of switching secret-shared
    /// values into FHE, and `packed_dot` refuses rather than wrapping.
    #[test]
    fn packing_the_crossing_matches_the_scalar_route() {
        // Width and layer size chosen so (a) the plaintext modulus holds the
        // un-reduced share products and (b) the true dot product fits in the
        // share width, so the assertion below is a plain equality rather than
        // a comparison against a wrapped value.
        const W: usize = 10;
        let n = 16;
        let a: Vec<i64> = (0..n).map(|i| (i as i64 % 5) - 2).collect();
        let b: Vec<i64> = (0..n).map(|i| ((i * 3) as i64 % 7) - 3).collect();
        let want: i64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        let s = Bfv::new(BfvParams {
            n: 4096,
            log2_q: 100,
            log2_t: 30,
        });
        let (ck, _pk) = s.keygen();

        let mut be = Aby3Simulator::new();
        let sh = |be: &mut Aby3Simulator, v: &[i64]| -> Vec<ShareRef> {
            v.iter()
                .map(|&x| {
                    let r = (x as i128).rem_euclid(1i128 << W) as Fr;
                    be.share_secret(r, W, 0)
                })
                .collect()
        };
        let av = sh(&mut be, &a);
        let bv = sh(&mut be, &b);

        let (r, st) = packed_dot(&mut be, &s, &ck, &av, &bv, W).expect("packed dot");
        let m = 1i128 << W;
        let raw = (be.reveal(&r, None) as i128) & (m - 1);
        let got = if raw >= m / 2 {
            (raw - m) as i64
        } else {
            raw as i64
        };

        assert!(
            want.abs() < (1 << (W - 1)),
            "test data must fit the share width"
        );
        assert_eq!(got, want, "packed dot must equal the plaintext dot product");
        // The whole point: crossings do not scale with the layer.
        assert_eq!(st.to_fhe, 2, "one ciphertext per operand: {st:?}");
        assert_eq!(st.to_mpc, 1, "one masked opening back: {st:?}");
        assert_eq!(st.fhe_ops, 1, "one multiplication for the layer: {st:?}");
    }

    /// The modulus bound is checked, not hoped for: a layer too wide for the
    /// plaintext modulus must fail loudly rather than wrap silently.
    #[test]
    fn packed_dot_refuses_a_layer_its_modulus_cannot_hold() {
        let s = Bfv::new(BfvParams {
            n: 4096,
            log2_q: 100,
            log2_t: 30,
        });
        let (ck, _pk) = s.keygen();
        let mut be = Aby3Simulator::new();
        let v: Vec<ShareRef> = (0..64).map(|_| be.share_secret(1, 20, 0)).collect();
        let err = packed_dot(&mut be, &s, &ck, &v, &v, 20).unwrap_err();
        assert!(err.contains("plaintext modulus"), "{err}");
    }

    /// The boundary rule earns its place on the shape that exposes it: two
    /// layers feeding one addition.
    ///
    /// Assigned per *kind*, `add` sits on MPC, so both dot products have to
    /// come back across the boundary — two masked openings. Sinking the
    /// addition onto the FHE side, where both of its operands already live,
    /// leaves one. The values are unchanged; only the boundary moved.
    #[test]
    fn sinking_an_operation_across_the_boundary_saves_a_crossing() {
        let n = 4;
        let mut d1 = "(* a0 b0)".to_string();
        let mut d2 = "(* c0 e0)".to_string();
        for i in 1..n {
            d1 = format!("(+ {d1} (* a{i} b{i}))");
            d2 = format!("(+ {d2} (* c{i} e{i}))");
        }
        let (flat, out) = compile(&format!("(+ {d1} {d2})"));

        let s = Bfv::new(BfvParams::TEST_SMALL);
        let (ck, _pk) = s.keygen();
        let fhe = BfvAdapter(&s);

        // Only multiplication is assigned to FHE; additions keep their kind's
        // default, which is MPC.
        let mut kinds = HashMap::new();
        kinds.insert("mul".to_string(), Side::Fhe);

        let setup = |be: &mut Aby3Simulator| -> HashMap<String, ShareRef> {
            let mut env = HashMap::new();
            for i in 0..n {
                for nm in ["a", "b", "c", "e"] {
                    let v = ((i + nm.len()) % 3) as Fr;
                    env.insert(format!("{nm}{i}"), be.share_secret(v, NBITS, 0));
                }
            }
            env
        };

        let mut be1 = Aby3Simulator::new();
        let env1 = setup(&mut be1);
        let flat_c = flat.clone();
        let p_kind = Placement::by_kind(kinds.clone());
        let (r1, st1) =
            run_mixed(&mut be1, &fhe, &ck, &flat_c, &env1, &out, NBITS, &p_kind).unwrap();

        let mut be2 = Aby3Simulator::new();
        let env2 = setup(&mut be2);
        let p_sunk = Placement::by_kind(kinds).sink_to_fhe(&flat);
        let (r2, st2) = run_mixed(&mut be2, &fhe, &ck, &flat, &env2, &out, NBITS, &p_sunk).unwrap();

        assert_eq!(
            to_signed(be1.reveal(&r1, None)),
            to_signed(be2.reveal(&r2, None)),
            "moving the boundary must not change the value"
        );
        assert!(
            p_sunk.refinements() > 0,
            "the rule should have moved something"
        );
        assert!(
            st2.to_mpc < st1.to_mpc,
            "sinking the addition should save a crossing back: {} -> {}",
            st1.to_mpc,
            st2.to_mpc
        );
    }

    /// Depth-1 is a real constraint, and the executor must respect it rather
    /// than emit a ciphertext it cannot decrypt. A product of products cannot
    /// stay on FHE, so it is demoted to MPC and counted.
    #[test]
    fn a_second_multiplication_is_demoted_rather_than_attempted() {
        let (flat, out) = compile("(* (* a0 b0) (* a1 b1))");
        let s = Bfv::new(BfvParams::TEST_SMALL);
        let (ck, _pk) = s.keygen();
        let fhe = BfvAdapter(&s);
        let mut asg = HashMap::new();
        asg.insert("mul".to_string(), Side::Fhe);

        let mut be = Aby3Simulator::new();
        let mut env = HashMap::new();
        let vals = [3i64, 2, 4, 5];
        for (i, n) in ["a0", "b0", "a1", "b1"].iter().enumerate() {
            env.insert(n.to_string(), be.share_secret(vals[i] as Fr, NBITS, 0));
        }
        let (r, st) = run_mixed(
            &mut be,
            &fhe,
            &ck,
            &flat,
            &env,
            &out,
            NBITS,
            &Placement::by_kind(asg),
        )
        .unwrap();
        assert_eq!(to_signed(be.reveal(&r, None)), (3 * 2) * (4 * 5));
        assert_eq!(
            st.demoted_muls, 1,
            "the outer product must fall back: {st:?}"
        );
    }
}
