//! `FlatOp` SSA → MPC bytecode.
//!
//! The front-end is unchanged: a Baru expression is compiled by
//! [`CircuitCompiler`](crate::zkp::circuit_compiler::CircuitCompiler) to the
//! same `FlatOp` block the ZKP and FHE backends consume.  This pass then
//! re-types that block for secure computation: it assigns a domain, tracks
//! which registers are clear and which are secret, and — crucially — replaces
//! the boolean-circuit idioms `FlatOp` inherited from its ZKP origins with the
//! arithmetic-domain lowerings a real MPC compiler would use.
//!
//! ## Boolean wires
//!
//! `FlatOp` has no type system, but it does have a strong convention: values
//! produced by comparisons are 0/1, and the `Mux` gadget is built from
//! `BAnd`/`BOr` over such values.  Lowering `BAnd` on two 0/1 arithmetic shares
//! as a full 64-bit bit-decomposition would over-charge it by ~64×, so this
//! pass propagates a *boolean* flag (comparisons, boolean-operand bitwise ops,
//! and 0/1 constants are boolean) and lowers bitwise ops on boolean operands
//! arithmetically, which is what an MPC compiler actually does:
//!
//! | op | boolean operands | general operands |
//! |---|---|---|
//! | `and(a,b)` | `a·b` — one multiplication | bit-decompose, `GF(2)` AND, recompose |
//! | `or(a,b)` | `a+b−a·b` | idem |
//! | `xor(a,b)` | `a+b−2a·b` | idem |
//! | `not(a)` | `1−a` | bit-decompose, `GF(2)` NOT, recompose |
//!
//! ## What is *not* done here
//!
//! No vectorization: `FlatOp` is scalar, so every instruction comes out with
//! `width = 1`.  Batching is nonetheless accounted for, because
//! [`crate::mpcc::schedule`] charges **one round per dependence level**, not one
//! round per instruction — a 784-lane dot product lowered as 784 scalar `Mul`s
//! at the same level costs one round, exactly as a batched protocol would run
//! it.  Turning those into genuine width-784 instructions is a later
//! optimization that changes the listing, not the cost.

use std::collections::{HashMap, HashSet};

use crate::zkp::circuit_compiler::FlatOp;

use super::ir::{Domain, Instr, Op, Program, Reg, Secrecy};

/// How to lower a circuit for secure computation.
#[derive(Debug, Clone)]
pub struct LowerConfig {
    /// Arithmetic domain for the program's secret values.
    pub domain: Domain,
    /// Number of computing parties — inputs are dealt round-robin.
    pub parties: usize,
    /// Bit width used for bit-decompositions, comparisons and truncations.
    /// Baru's REPL convention is 32-bit shares; the circuit compiler's
    /// `IAdd32`/`ISub32` names carry the same assumption.
    pub bit_width: u32,
    /// Variables that are public to every party (weights, class labels, …).
    /// Everything else becomes a secret input.
    pub public_vars: Vec<String>,
}

impl Default for LowerConfig {
    fn default() -> Self {
        LowerConfig {
            domain: Domain::Ring { k: 64 },
            parties: 3,
            bit_width: 32,
            public_vars: Vec::new(),
        }
    }
}

struct Lowerer<'a> {
    cfg: &'a LowerConfig,
    prog: Program,
    /// wire name → register
    map: HashMap<String, Reg>,
    /// wires known to hold a 0/1 value
    boolean: HashSet<String>,
    /// cached clear constants, so `1 - a` does not allocate a fresh 1 each time
    consts: HashMap<i64, Reg>,
}

impl<'a> Lowerer<'a> {
    fn new(cfg: &'a LowerConfig) -> Self {
        Lowerer {
            cfg,
            prog: Program::with_bit_width(cfg.domain, cfg.bit_width),
            map: HashMap::new(),
            boolean: HashSet::new(),
            consts: HashMap::new(),
        }
    }

    fn secrecy_of(&self, r: Reg) -> Secrecy {
        self.prog.reg(r).secrecy
    }

    /// Secret unless every operand is clear.
    fn join(&self, srcs: &[Reg]) -> Secrecy {
        if srcs.iter().any(|r| self.secrecy_of(*r) == Secrecy::Secret) {
            Secrecy::Secret
        } else {
            Secrecy::Clear
        }
    }

    fn arith(&mut self, secrecy: Secrecy, name: Option<String>) -> Reg {
        self.prog.alloc(self.cfg.domain, secrecy, 1, name)
    }

    fn get(&self, wire: &str) -> Result<Reg, String> {
        self.map
            .get(wire)
            .copied()
            .ok_or_else(|| format!("mpcc: use of undefined wire `{}`", wire))
    }

    fn constant(&mut self, value: i64) -> Reg {
        if let Some(r) = self.consts.get(&value) {
            return *r;
        }
        let r = self.arith(Secrecy::Clear, None);
        self.prog.push(Instr {
            op: Op::LdI(value),
            dst: vec![r],
            src: vec![],
            width: 1,
            label: None,
        });
        self.consts.insert(value, r);
        r
    }

    /// Emit `dst <- op(srcs)` in the arithmetic domain, binding `wire`.
    fn emit(&mut self, wire: &str, op: Op, srcs: Vec<Reg>) -> Reg {
        let secrecy = self.join(&srcs);
        let dst = self.arith(secrecy, Some(wire.to_string()));
        self.prog.push(Instr {
            op,
            dst: vec![dst],
            src: srcs,
            width: 1,
            label: None,
        });
        self.map.insert(wire.to_string(), dst);
        dst
    }

    /// Emit into a fresh temporary rather than binding a source wire.
    fn emit_tmp(&mut self, op: Op, srcs: Vec<Reg>) -> Reg {
        let secrecy = self.join(&srcs);
        let dst = self.arith(secrecy, None);
        self.prog.push(Instr {
            op,
            dst: vec![dst],
            src: srcs,
            width: 1,
            label: None,
        });
        dst
    }

    /// `a * b` in the arithmetic domain.
    fn mul(&mut self, a: Reg, b: Reg) -> Reg {
        self.emit_tmp(Op::Mul, vec![a, b])
    }

    fn add(&mut self, a: Reg, b: Reg) -> Reg {
        self.emit_tmp(Op::Add, vec![a, b])
    }

    fn sub(&mut self, a: Reg, b: Reg) -> Reg {
        self.emit_tmp(Op::Sub, vec![a, b])
    }

    /// Arithmetic register → binary register of `bit_width` lanes.
    fn decompose(&mut self, a: Reg) -> Reg {
        let nbits = self.cfg.bit_width;
        let secrecy = self.secrecy_of(a);
        let width = self.prog.reg(a).width * nbits;
        let dst = self.prog.alloc(Domain::Binary, secrecy, width, None);
        self.prog.push(Instr {
            op: Op::BitDecomp { nbits },
            dst: vec![dst],
            src: vec![a],
            width,
            label: None,
        });
        dst
    }

    /// Binary register → arithmetic register.
    fn recompose(&mut self, bits: Reg, wire: &str) -> Reg {
        let nbits = self.cfg.bit_width;
        let secrecy = self.secrecy_of(bits);
        let width = self.prog.reg(bits).width / nbits.max(1);
        let dst = self.prog.alloc(
            self.cfg.domain,
            secrecy,
            width.max(1),
            Some(wire.to_string()),
        );
        self.prog.push(Instr {
            op: Op::BitComp { nbits },
            dst: vec![dst],
            src: vec![bits],
            width: self.prog.reg(bits).width,
            label: None,
        });
        self.map.insert(wire.to_string(), dst);
        dst
    }

    /// Bitwise op over general (non-boolean) operands: decompose, operate in
    /// `GF(2)`, recompose.
    fn bitwise(&mut self, wire: &str, op: Op, srcs: &[Reg]) -> Reg {
        let bit_regs: Vec<Reg> = srcs.iter().map(|r| self.decompose(*r)).collect();
        let width = self.prog.reg(bit_regs[0]).width;
        let secrecy = self.join(&bit_regs);
        let out_bits = self.prog.alloc(Domain::Binary, secrecy, width, None);
        self.prog.push(Instr {
            op,
            dst: vec![out_bits],
            src: bit_regs,
            width,
            label: None,
        });
        self.recompose(out_bits, wire)
    }

    fn all_boolean(&self, wires: &[&str]) -> bool {
        wires.iter().all(|w| self.boolean.contains(*w))
    }
}

/// Lower a `FlatOp` block to MPC bytecode.
///
/// Wires that no instruction defines are the circuit's inputs; they are dealt
/// round-robin across `cfg.parties` unless named in `cfg.public_vars`.
/// `output_wire` is opened and declared as the program result.
pub fn lower(flat: &[FlatOp], output_wire: &str, cfg: &LowerConfig) -> Result<Program, String> {
    if cfg.parties == 0 {
        return Err("mpcc: need at least one party".to_string());
    }
    let mut lo = Lowerer::new(cfg);

    // ── Inputs: every wire read before it is written ──────────────────────
    let defined: HashSet<&str> = flat.iter().flat_map(flat_dsts).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut input_names: Vec<String> = Vec::new();
    for op in flat {
        for w in op.inputs() {
            if !defined.contains(w) && seen.insert(w.to_string()) {
                input_names.push(w.to_string());
            }
        }
    }
    if !defined.contains(output_wire) && seen.insert(output_wire.to_string()) {
        input_names.push(output_wire.to_string());
    }

    let public: HashSet<&str> = cfg.public_vars.iter().map(String::as_str).collect();
    let mut next_party = 0usize;
    for name in &input_names {
        if public.contains(name.as_str()) {
            let r = lo.arith(Secrecy::Clear, Some(name.clone()));
            lo.prog.push(Instr {
                op: Op::LdI(0),
                dst: vec![r],
                src: vec![],
                width: 1,
                label: Some(format!("public {}", name)),
            });
            lo.map.insert(name.clone(), r);
        } else {
            let party = next_party % cfg.parties;
            next_party += 1;
            let r = lo.arith(Secrecy::Secret, Some(name.clone()));
            lo.prog.push(Instr {
                op: Op::Input { party },
                dst: vec![r],
                src: vec![],
                width: 1,
                label: Some(name.clone()),
            });
            lo.map.insert(name.clone(), r);
            lo.prog.inputs.push((name.clone(), r));
        }
    }

    // ── Body ──────────────────────────────────────────────────────────────
    for op in flat {
        lower_op(&mut lo, op)?;
    }

    // ── Open the result ───────────────────────────────────────────────────
    let res = lo.get(output_wire)?;
    let opened = if lo.secrecy_of(res) == Secrecy::Secret {
        let o = lo
            .prog
            .alloc(cfg.domain, Secrecy::Clear, 1, Some("out".into()));
        lo.prog.push(Instr {
            op: Op::Open,
            dst: vec![o],
            src: vec![res],
            width: 1,
            label: None,
        });
        o
    } else {
        res
    };
    lo.prog.push(Instr {
        op: Op::Output,
        dst: vec![],
        src: vec![opened],
        width: 1,
        label: Some(output_wire.to_string()),
    });
    lo.prog.outputs.push((output_wire.to_string(), opened));

    Ok(lo.prog)
}

fn lower_op(lo: &mut Lowerer, op: &FlatOp) -> Result<(), String> {
    match op {
        FlatOp::Const { dst, value } => {
            let r = lo.constant(*value);
            lo.map.insert(dst.clone(), r);
            if *value == 0 || *value == 1 {
                lo.boolean.insert(dst.clone());
            }
        }
        FlatOp::ConstF { .. } => {
            return Err(
                "mpcc: real-arithmetic circuits (ConstF) are FHE-only; compile with \
                 CircuitCompiler::new() rather than new_real() for MPC"
                    .to_string(),
            );
        }
        FlatOp::IAdd32 { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            lo.emit(dst, Op::Add, vec![a, b]);
        }
        FlatOp::ISub32 { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            lo.emit(dst, Op::Sub, vec![a, b]);
        }
        FlatOp::IMul { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            lo.emit(dst, Op::Mul, vec![a, b]);
        }
        FlatOp::IDiv {
            dst_q,
            dst_r,
            lhs,
            rhs,
        } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            let secrecy = lo.join(&[a, b]);
            let q = lo.arith(secrecy, Some(dst_q.clone()));
            let r = lo.arith(secrecy, Some(dst_r.clone()));
            lo.prog.push(Instr {
                op: Op::Div,
                dst: vec![q, r],
                src: vec![a, b],
                width: 1,
                label: None,
            });
            lo.map.insert(dst_q.clone(), q);
            lo.map.insert(dst_r.clone(), r);
        }
        FlatOp::ICmpEq { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            let diff = lo.sub(a, b);
            lo.emit(dst, Op::EqZ, vec![diff]);
            lo.boolean.insert(dst.clone());
        }
        FlatOp::ICmpUlt { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            lo.emit(dst, Op::LtU, vec![a, b]);
            lo.boolean.insert(dst.clone());
        }
        FlatOp::ICmpSlt { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            lo.emit(dst, Op::LtS, vec![a, b]);
            lo.boolean.insert(dst.clone());
        }
        FlatOp::BAnd { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            if lo.all_boolean(&[lhs.as_str(), rhs.as_str()]) {
                lo.emit(dst, Op::Mul, vec![a, b]);
                lo.boolean.insert(dst.clone());
            } else {
                lo.bitwise(dst, Op::And, &[a, b]);
            }
        }
        FlatOp::BOr { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            if lo.all_boolean(&[lhs.as_str(), rhs.as_str()]) {
                // a + b - a*b
                let ab = lo.mul(a, b);
                let sum = lo.add(a, b);
                lo.emit(dst, Op::Sub, vec![sum, ab]);
                lo.boolean.insert(dst.clone());
            } else {
                // GF(2) has no OR primitive: or(a,b) = not(and(not a, not b)),
                // but the cost is one AND either way, so charge that.
                lo.bitwise(dst, Op::And, &[a, b]);
            }
        }
        FlatOp::BXor { dst, lhs, rhs } => {
            let (a, b) = (lo.get(lhs)?, lo.get(rhs)?);
            if lo.all_boolean(&[lhs, rhs]) {
                // a + b - 2ab
                let ab = lo.mul(a, b);
                let two_ab = lo.add(ab, ab);
                let sum = lo.add(a, b);
                lo.emit(dst, Op::Sub, vec![sum, two_ab]);
                lo.boolean.insert(dst.clone());
            } else {
                lo.bitwise(dst, Op::Xor, &[a, b]);
            }
        }
        FlatOp::BNot { dst, src } => {
            let a = lo.get(src)?;
            if lo.boolean.contains(src) {
                let one = lo.constant(1);
                lo.emit(dst, Op::Sub, vec![one, a]);
                lo.boolean.insert(dst.clone());
            } else {
                lo.bitwise(dst, Op::Not, &[a]);
            }
        }
        FlatOp::Shl { dst, src, amount } => {
            let a = lo.get(src)?;
            lo.emit(dst, Op::ShlC(*amount), vec![a]);
        }
        FlatOp::Shr { dst, src, amount } => {
            // Logical right shift of a secret value is *not* local in an
            // arithmetic domain — it is a truncation protocol.
            let a = lo.get(src)?;
            lo.emit(dst, Op::TruncPr { shift: *amount }, vec![a]);
        }
        FlatOp::Mux {
            dst,
            cond,
            t_val,
            e_val,
        } => {
            let (c, t, e) = (lo.get(cond)?, lo.get(t_val)?, lo.get(e_val)?);
            lo.emit(dst, Op::Mux, vec![c, t, e]);
            if lo.all_boolean(&[t_val, e_val]) {
                lo.boolean.insert(dst.clone());
            }
        }
        FlatOp::PopCount { dst, src } => {
            let a = lo.get(src)?;
            let nbits = lo.cfg.bit_width;
            lo.emit(dst, Op::PopCount { nbits }, vec![a]);
        }
    }
    Ok(())
}

/// The wires an op writes.  `FlatOp::dst` only reports one; `IDiv` writes two.
fn flat_dsts(op: &FlatOp) -> Vec<&str> {
    match op {
        FlatOp::IDiv { dst_q, dst_r, .. } => vec![dst_q.as_str(), dst_r.as_str()],
        other => vec![other.dst()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower_src(src: &str) -> Program {
        let expr = crate::interpreter::parse(src).expect("parse");
        let mut cc = crate::zkp::circuit_compiler::CircuitCompiler::new();
        let circuit = cc.compile_fhe(&expr).expect("compile");
        lower(
            &circuit.flat_ops,
            &circuit.output_wire,
            &LowerConfig::default(),
        )
        .expect("lower")
    }

    #[test]
    fn inputs_are_dealt_round_robin_and_result_is_opened() {
        let p = lower_src("(+ (* x 3) (* y 7))");
        assert_eq!(p.inputs.len(), 2, "x and y are secret inputs");
        let parties: Vec<usize> = p
            .code
            .iter()
            .filter_map(|i| match i.op {
                Op::Input { party } => Some(party),
                _ => None,
            })
            .collect();
        assert_eq!(parties, vec![0, 1]);
        assert_eq!(p.outputs.len(), 1);
        assert!(p.code.iter().any(|i| i.op == Op::Open));
    }

    #[test]
    fn multiplying_by_a_constant_costs_no_round() {
        let p = lower_src("(* x 3)");
        let interactive: Vec<&str> = p
            .code
            .iter()
            .filter(|i| p.is_interactive(i))
            .map(|i| i.op.mnemonic())
            .collect();
        // Only the final Open communicates: `x * 3` is a local scalar multiply.
        assert_eq!(interactive, vec!["open"]);
    }

    #[test]
    fn secret_multiplication_costs_a_round() {
        let p = lower_src("(* x y)");
        let muls = p
            .code
            .iter()
            .filter(|i| i.op == Op::Mul && p.is_interactive(i))
            .count();
        assert_eq!(muls, 1);
    }

    #[test]
    fn boolean_bitwise_ops_lower_arithmetically() {
        // The mux gadget the circuit compiler emits for `if` is band/bor over
        // comparison results; none of it should become a bit-decomposition.
        let p = lower_src("(if (> x 0) x 0)");
        assert!(
            !p.code.iter().any(|i| matches!(i.op, Op::BitDecomp { .. })),
            "boolean operands must not trigger bit-decomposition:\n{}",
            p.listing()
        );
        assert!(
            p.code
                .iter()
                .any(|i| matches!(i.op, Op::LtS | Op::LtU | Op::Mux)),
            "expected a comparison in the lowered program:\n{}",
            p.listing()
        );
    }

    #[test]
    fn right_shift_is_interactive() {
        let p = lower_src("(ash-right x 3)");
        let trunc = p
            .code
            .iter()
            .find(|i| matches!(i.op, Op::TruncPr { .. }))
            .expect("shift lowers to a truncation");
        assert!(p.is_interactive(trunc));
    }

    #[test]
    fn real_arithmetic_circuits_are_rejected() {
        let expr = crate::interpreter::parse("(+ x 1.5)").expect("parse");
        let mut cc = crate::zkp::circuit_compiler::CircuitCompiler::new_real();
        let circuit = cc.compile_fhe(&expr).expect("compile");
        let err = lower(
            &circuit.flat_ops,
            &circuit.output_wire,
            &LowerConfig::default(),
        )
        .unwrap_err();
        assert!(err.contains("FHE-only"), "{}", err);
    }
}
