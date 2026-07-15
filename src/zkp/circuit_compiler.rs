// Circuit Compiler for Baru
// ============================================================
// Two-phase compilation pipeline:
//
//   Lisp AST
//     │
//     ▼  Phase 1 – flatten()
//   FlatIR  (SSA block of FlatOps, one per Binius64 gate/opcode)
//     │
//     ▼  Phase 1b – fold_constants() + eliminate_dead_wires()
//   Optimised FlatIR
//     │
//     ▼  Phase 2 – emit_constraints()
//   Vec<Constraint>  (ready for Binius64 CircuitBuilder)
//
// Optimisations applied between phases:
//   Constant folding   — Const op Const → single Const, shrinks circuit
//   Dead-wire elim.    — removes Const nodes whose dst is never read
//
// Non-determinism:
//   Div / Mod  — witness quotient + remainder, check q*d+r==n
//   If         — both branches evaluated; Mux selects result (no branching)
//
// Binius64 gate mapping:
//   Add      → iadd_32        Sub  → isub_32
//   Mul      → imul           Div  → NDivCheck (non-det witness)
//   Eq       → icmp_eq        Lt   → icmp_slt
//   Gt/Le/Ge → derived        And  → band      Or → bor
//   Not      → bnot           Neg  → isub_32(0, x)
//   Shl/Shr  → shl/shr (static amount extracted from AST)
//   Mux      → band+bor gadget

use crate::interpreter::ast::Expr;
use crate::interpreter::parser::{BinOp, UnOp};
use std::collections::{HashMap, HashSet};

// ─── Flat IR ─────────────────────────────────────────────────

/// A wire name (SSA variable).
pub type Wire = String;

/// One instruction in the flat, straight-line SSA block.
/// Every variant maps to exactly one or a small fixed number of
/// Binius64 gates, making constraint emission trivial.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum FlatOp {
    /// dst = constant
    Const { dst: Wire, value: i64 },
    /// dst = iadd_32(lhs, rhs)
    IAdd32 { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = isub_32(lhs, rhs)
    ISub32 { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = imul(lhs, rhs)
    IMul { dst: Wire, lhs: Wire, rhs: Wire },
    /// Non-deterministic division: lhs = dst_q * rhs + dst_r, 0 ≤ dst_r < rhs
    IDiv {
        dst_q: Wire,
        dst_r: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    /// dst = icmp_eq(lhs, rhs)  — 1 if equal, else 0
    ICmpEq { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = icmp_ult(lhs, rhs) — unsigned less-than
    ICmpUlt { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = icmp_slt(lhs, rhs) — signed less-than
    ICmpSlt { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = band(lhs, rhs)
    BAnd { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = bor(lhs, rhs)
    BOr { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = bxor(lhs, rhs)
    BXor { dst: Wire, lhs: Wire, rhs: Wire },
    /// dst = bnot(src)
    BNot { dst: Wire, src: Wire },
    /// dst = shl(src, amount)  — amount is a compile-time literal
    Shl { dst: Wire, src: Wire, amount: u32 },
    /// dst = shr(src, amount)
    Shr { dst: Wire, src: Wire, amount: u32 },
    /// dst = cond ? t_val : e_val   (Mux gadget: band+bor)
    Mux {
        dst: Wire,
        cond: Wire,
        t_val: Wire,
        e_val: Wire,
    },
    /// dst = popcount(src)
    PopCount { dst: Wire, src: Wire },
}

impl FlatOp {
    /// The single wire this op writes to.
    pub fn dst(&self) -> &str {
        match self {
            FlatOp::Const { dst, .. }
            | FlatOp::IAdd32 { dst, .. }
            | FlatOp::ISub32 { dst, .. }
            | FlatOp::IMul { dst, .. }
            | FlatOp::ICmpEq { dst, .. }
            | FlatOp::ICmpUlt { dst, .. }
            | FlatOp::ICmpSlt { dst, .. }
            | FlatOp::BAnd { dst, .. }
            | FlatOp::BOr { dst, .. }
            | FlatOp::BXor { dst, .. }
            | FlatOp::BNot { dst, .. }
            | FlatOp::Shl { dst, .. }
            | FlatOp::Shr { dst, .. }
            | FlatOp::Mux { dst, .. }
            | FlatOp::PopCount { dst, .. } => dst.as_str(),
            // IDiv has two destinations — return the quotient wire.
            FlatOp::IDiv { dst_q, .. } => dst_q.as_str(),
        }
    }

    /// All wires this op reads (inputs).
    pub fn inputs(&self) -> Vec<&str> {
        match self {
            FlatOp::Const { .. } => vec![],
            FlatOp::IAdd32 { lhs, rhs, .. }
            | FlatOp::ISub32 { lhs, rhs, .. }
            | FlatOp::IMul { lhs, rhs, .. }
            | FlatOp::ICmpEq { lhs, rhs, .. }
            | FlatOp::ICmpUlt { lhs, rhs, .. }
            | FlatOp::ICmpSlt { lhs, rhs, .. }
            | FlatOp::BAnd { lhs, rhs, .. }
            | FlatOp::BOr { lhs, rhs, .. }
            | FlatOp::BXor { lhs, rhs, .. } => vec![lhs, rhs],
            FlatOp::BNot { src, .. }
            | FlatOp::Shl { src, .. }
            | FlatOp::Shr { src, .. }
            | FlatOp::PopCount { src, .. } => vec![src],
            FlatOp::IDiv { lhs, rhs, .. } => vec![lhs, rhs],
            FlatOp::Mux {
                cond, t_val, e_val, ..
            } => vec![cond, t_val, e_val],
        }
    }
}

// ─── Constraint (Binius64 gate encoding) ─────────────────────

#[derive(Debug, Clone)]
pub enum Constraint {
    Constant {
        dst: Wire,
        value: i64,
    },
    IAdd32 {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    ISub32 {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    IMul {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    /// Non-det: witness q,r  s.t.  lhs == q*rhs + r
    NDivCheck {
        dst_q: Wire,
        dst_r: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    ICmpEq {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    ICmpUlt {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    ICmpSlt {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    BAnd {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    BOr {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    BXor {
        dst: Wire,
        lhs: Wire,
        rhs: Wire,
    },
    BNot {
        dst: Wire,
        src: Wire,
    },
    Shl {
        dst: Wire,
        src: Wire,
        amount: u32,
    },
    Shr {
        dst: Wire,
        src: Wire,
        amount: u32,
    },
    Mux {
        dst: Wire,
        cond: Wire,
        t_val: Wire,
        e_val: Wire,
    },
    PopCount {
        dst: Wire,
        src: Wire,
    },
}

// ─── Compiled circuit ────────────────────────────────────────

#[derive(Debug)]
pub struct CompiledCircuit {
    /// Optimised flat SSA IR (after constant folding + dead-wire elimination)
    pub flat_ops: Vec<FlatOp>,
    /// Gate-level constraints (one per FlatOp)
    pub constraints: Vec<Constraint>,
    pub witnesses: Vec<Wire>,
    pub public_inputs: Vec<Wire>,
    pub num_gates: usize,
}

// ─── Flattening environment ───────────────────────────────────

/// Maps variable names to SSA wires.
type Env = HashMap<String, Wire>;

/// Maps lambda names to their parameter lists and body, for inlining
/// named functions bound in let/letrec.
type LambdaStore = HashMap<String, (Vec<String>, Expr)>;

// ─── CircuitCompiler ─────────────────────────────────────────

pub struct CircuitCompiler {
    var_counter: usize,
    witnesses: Vec<Wire>,
    public_inputs: Vec<Wire>,
}

impl Default for CircuitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitCompiler {
    pub fn new() -> Self {
        CircuitCompiler {
            var_counter: 0,
            witnesses: Vec::new(),
            public_inputs: Vec::new(),
        }
    }

    // ── SSA helpers ──────────────────────────────────────────

    fn fresh(&mut self, prefix: &str) -> Wire {
        let w = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        w
    }

    fn fresh_witness(&mut self, prefix: &str) -> Wire {
        let w = self.fresh(prefix);
        self.witnesses.push(w.clone());
        w
    }

    // ── Public entry point ───────────────────────────────────

    pub fn compile(
        &mut self,
        expr: &Expr,
        private_vars: &[String],
    ) -> Result<CompiledCircuit, String> {
        // Classify user-named variables as witnesses or public inputs.
        let all_vars = self.extract_variables(expr);
        for var in &all_vars {
            if private_vars.contains(var) {
                if !self.witnesses.contains(var) {
                    self.witnesses.push(var.clone());
                }
            } else if !self.public_inputs.contains(var) {
                self.public_inputs.push(var.clone());
            }
        }

        // ── Phase 1: flatten to SSA ───────────────────────────
        let mut flat: Vec<FlatOp> = Vec::new();
        let mut env: Env = Env::new();
        let mut lambdas: LambdaStore = LambdaStore::new();

        // User variables map to themselves (they are input wires).
        for var in self.witnesses.iter().chain(self.public_inputs.iter()) {
            env.insert(var.clone(), var.clone());
        }

        let result_wire = self.flatten(expr, &mut env, &mut lambdas, &mut flat)?;

        // ── Phase 1b: optimise flat IR ────────────────────────
        let flat = Self::fold_constants(flat);
        let flat =
            Self::eliminate_dead_wires(flat, &result_wire, &self.witnesses, &self.public_inputs);

        // ── Phase 2: emit constraints ─────────────────────────
        let constraints = Self::emit_constraints(&flat);
        let num_gates = constraints.len();

        Ok(CompiledCircuit {
            flat_ops: flat,
            constraints,
            witnesses: self.witnesses.clone(),
            public_inputs: self.public_inputs.clone(),
            num_gates,
        })
    }

    /// Compile an expression to an `FheCircuit` for homomorphic evaluation.
    ///
    /// Unlike ZKP compilation, every input variable is treated as a potential
    /// encrypted input (caller decides at evaluation time which ones are
    /// plaintext public inputs vs. encrypted inputs).  There are no witnesses
    /// (non-determinism) — `IDiv` ops are kept as-is and evaluated via PBS.
    pub fn compile_fhe(&mut self, expr: &Expr) -> Result<crate::fhe::backend::FheCircuit, String> {
        let all_vars = self.extract_variables(expr);
        // All variables are potential inputs; classification is deferred to
        // the caller (fhe-eval supplies them as plain or cipher maps).
        for v in &all_vars {
            if !self.public_inputs.contains(v) {
                self.public_inputs.push(v.clone());
            }
        }

        let mut flat: Vec<FlatOp> = Vec::new();
        let mut env: Env = Env::new();
        let mut lambdas: LambdaStore = LambdaStore::new();

        for var in &self.public_inputs {
            env.insert(var.clone(), var.clone());
        }

        let result_wire = self.flatten(expr, &mut env, &mut lambdas, &mut flat)?;

        let flat = Self::fold_constants(flat);
        let flat = Self::eliminate_dead_wires(
            flat,
            &result_wire,
            &[], // no ZKP witnesses
            &self.public_inputs,
        );

        let num_gates = flat.len();
        Ok(crate::fhe::backend::FheCircuit {
            flat_ops: flat,
            encrypted_inputs: vec![], // filled in by caller / fhe-compile
            public_inputs: self.public_inputs.clone(),
            output_wire: result_wire,
            num_gates,
        })
    }

    // ── Phase 1 – flatten ────────────────────────────────────

    fn flatten(
        &mut self,
        expr: &Expr,
        env: &mut Env,
        lambdas: &mut LambdaStore,
        flat: &mut Vec<FlatOp>,
    ) -> Result<Wire, String> {
        match expr {
            // ── Literals ─────────────────────────────────────
            Expr::Num(n) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: *n,
                });
                Ok(dst)
            }
            // Fixed-point literal: the circuit works on the scaled integer directly.
            // Truncation after multiplication is handled by the FixedMul / Shr gates.
            Expr::Fixed(v) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: *v,
                });
                Ok(dst)
            }
            Expr::Bool(b) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: if *b { 1 } else { 0 },
                });
                Ok(dst)
            }
            Expr::Char(c) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: *c as i64,
                });
                Ok(dst)
            }

            // ── Variable reference ────────────────────────────
            Expr::Var(name) => env
                .get(name)
                .cloned()
                .ok_or_else(|| format!("unbound variable: {}", name)),

            // ── Private annotation (witness metadata) ─────────
            Expr::Private(inner) => self.flatten(inner, env, lambdas, flat),

            // ── Quote → constant 0 (non-evaluatable in circuit)
            Expr::Quote(_) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: 0,
                });
                Ok(dst)
            }

            // ── Let: inline bindings, then compile body ───────
            Expr::Let(bindings, body) => {
                let mut child_env = env.clone();
                let mut child_lam = lambdas.clone();
                for (name, rhs) in bindings {
                    // If the RHS is a lambda, store it for later inlining
                    // instead of collapsing it to a sentinel wire.
                    if let Expr::Lambda(params, lam_body) = rhs {
                        child_lam.insert(name.clone(), (params.clone(), *lam_body.clone()));
                    } else {
                        let wire = self.flatten(rhs, &mut child_env, &mut child_lam, flat)?;
                        child_env.insert(name.clone(), wire);
                    }
                }
                let mut result = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: result.clone(),
                    value: 0,
                });
                for e in body {
                    result = self.flatten(e, &mut child_env, &mut child_lam, flat)?;
                }
                Ok(result)
            }

            // ── Letrec: same as Let for acyclic circuits ──────
            Expr::LetRec(bindings, body) => {
                let mut child_env = env.clone();
                let mut child_lam = lambdas.clone();
                for (name, rhs) in bindings {
                    if let Expr::Lambda(params, lam_body) = rhs {
                        child_lam.insert(name.clone(), (params.clone(), *lam_body.clone()));
                    } else {
                        let wire = self.flatten(rhs, &mut child_env, &mut child_lam, flat)?;
                        child_env.insert(name.clone(), wire);
                    }
                }
                let mut result = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: result.clone(),
                    value: 0,
                });
                for e in body {
                    result = self.flatten(e, &mut child_env, &mut child_lam, flat)?;
                }
                Ok(result)
            }

            // ── Seq ───────────────────────────────────────────
            Expr::Seq(exprs) => {
                let mut result = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: result.clone(),
                    value: 0,
                });
                for e in exprs {
                    result = self.flatten(e, env, lambdas, flat)?;
                }
                Ok(result)
            }

            // ── Binary operations ─────────────────────────────
            Expr::BinOp(op, left, right) => {
                // Special case: extract static shift amounts directly from AST
                // so the Shl/Shr opcodes carry the correct literal amount.
                if matches!(op, BinOp::AshLeft | BinOp::AshRight) {
                    let src = self.flatten(left, env, lambdas, flat)?;
                    let amount = if let Expr::Num(n) = right.as_ref() {
                        *n as u32
                    } else {
                        0
                    };
                    let dst = self.fresh("t");
                    if matches!(op, BinOp::AshLeft) {
                        flat.push(FlatOp::Shl {
                            dst: dst.clone(),
                            src,
                            amount,
                        });
                    } else {
                        flat.push(FlatOp::Shr {
                            dst: dst.clone(),
                            src,
                            amount,
                        });
                    }
                    return Ok(dst);
                }
                let l = self.flatten(left, env, lambdas, flat)?;
                let r = self.flatten(right, env, lambdas, flat)?;
                self.flatten_binop(*op, l, r, flat)
            }

            // ── Unary operations ──────────────────────────────
            Expr::UnOp(op, inner) => {
                let src = self.flatten(inner, env, lambdas, flat)?;
                self.flatten_unop(*op, src, flat)
            }

            // ── If → Mux (both branches evaluated; non-det select)
            Expr::If(cond, then_e, else_e) => {
                let cond_wire = self.flatten(cond, env, lambdas, flat)?;
                let then_wire = self.flatten(then_e, env, lambdas, flat)?;
                let else_wire = self.flatten(else_e, env, lambdas, flat)?;
                let dst = self.fresh("mux");
                flat.push(FlatOp::Mux {
                    dst: dst.clone(),
                    cond: cond_wire,
                    t_val: then_wire,
                    e_val: else_wire,
                });
                Ok(dst)
            }

            // ── Lambda / Call ─────────────────────────────────
            Expr::Lambda(params, body) => {
                // Bare lambda — emit sentinel; meaningful only inside Call.
                let dst = self.fresh("lam");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: 0,
                });
                let _ = (params, body);
                Ok(dst)
            }
            Expr::Call(func, args) => self.flatten_call(func, args, env, lambdas, flat),

            // ── List operations ───────────────────────────────
            Expr::Cons(left, right) => {
                let _l = self.flatten(left, env, lambdas, flat)?;
                let r = self.flatten(right, env, lambdas, flat)?;
                Ok(r)
            }
            Expr::Car(inner) => self.flatten(inner, env, lambdas, flat),
            Expr::Cdr(inner) => self.flatten(inner, env, lambdas, flat),
            Expr::Eval(inner) => self.flatten(inner, env, lambdas, flat),
            Expr::Str(_) => {
                let dst = self.fresh("str");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: 0,
                });
                Ok(dst)
            }
        }
    }

    // ── BinOp lowering ───────────────────────────────────────

    fn flatten_binop(
        &mut self,
        op: BinOp,
        l: Wire,
        r: Wire,
        flat: &mut Vec<FlatOp>,
    ) -> Result<Wire, String> {
        let dst = self.fresh("t");
        match op {
            BinOp::Add => flat.push(FlatOp::IAdd32 {
                dst: dst.clone(),
                lhs: l,
                rhs: r,
            }),
            BinOp::Sub => flat.push(FlatOp::ISub32 {
                dst: dst.clone(),
                lhs: l,
                rhs: r,
            }),
            BinOp::Mul => flat.push(FlatOp::IMul {
                dst: dst.clone(),
                lhs: l,
                rhs: r,
            }),

            // Non-deterministic: witness q and rem, check q*rhs + rem == lhs.
            BinOp::DivExact => {
                let dst_r = self.fresh_witness("rem");
                flat.push(FlatOp::IDiv {
                    dst_q: dst.clone(),
                    dst_r,
                    lhs: l,
                    rhs: r,
                });
            }
            BinOp::Mod => {
                let dst_q = self.fresh_witness("quot");
                flat.push(FlatOp::IDiv {
                    dst_q,
                    dst_r: dst.clone(),
                    lhs: l,
                    rhs: r,
                });
            }

            BinOp::Eq => flat.push(FlatOp::ICmpEq {
                dst: dst.clone(),
                lhs: l,
                rhs: r,
            }),
            BinOp::Lt => flat.push(FlatOp::ICmpSlt {
                dst: dst.clone(),
                lhs: l,
                rhs: r,
            }),
            BinOp::Gt => flat.push(FlatOp::ICmpSlt {
                dst: dst.clone(),
                lhs: r,
                rhs: l,
            }),
            BinOp::Le => {
                // a ≤ b  ⟺  NOT (b < a)
                let lt = self.fresh("lt");
                flat.push(FlatOp::ICmpSlt {
                    dst: lt.clone(),
                    lhs: r,
                    rhs: l,
                });
                flat.push(FlatOp::BNot {
                    dst: dst.clone(),
                    src: lt,
                });
            }
            BinOp::Ge => {
                // a ≥ b  ⟺  NOT (a < b)
                let lt = self.fresh("lt");
                flat.push(FlatOp::ICmpSlt {
                    dst: lt.clone(),
                    lhs: l,
                    rhs: r,
                });
                flat.push(FlatOp::BNot {
                    dst: dst.clone(),
                    src: lt,
                });
            }
            BinOp::And => flat.push(FlatOp::BAnd {
                dst: dst.clone(),
                lhs: l,
                rhs: r,
            }),
            BinOp::Or => flat.push(FlatOp::BOr {
                dst: dst.clone(),
                lhs: l,
                rhs: r,
            }),

            // Static shifts are handled before this call-site (in BinOp arm above).
            // If reached here the amount is unknown; default to 0.
            BinOp::AshLeft => flat.push(FlatOp::Shl {
                dst: dst.clone(),
                src: l,
                amount: 0,
            }),
            BinOp::AshRight => flat.push(FlatOp::Shr {
                dst: dst.clone(),
                src: l,
                amount: 0,
            }),
        }
        Ok(dst)
    }

    // ── UnOp lowering ────────────────────────────────────────

    fn flatten_unop(
        &mut self,
        op: UnOp,
        src: Wire,
        flat: &mut Vec<FlatOp>,
    ) -> Result<Wire, String> {
        let dst = self.fresh("t");
        match op {
            UnOp::Not => flat.push(FlatOp::BNot {
                dst: dst.clone(),
                src,
            }),
            UnOp::Neg => {
                let zero = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: zero.clone(),
                    value: 0,
                });
                flat.push(FlatOp::ISub32 {
                    dst: dst.clone(),
                    lhs: zero,
                    rhs: src,
                });
            }
            UnOp::Atom => {
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: 1,
                });
            }
            UnOp::BitLen => {
                flat.push(FlatOp::PopCount {
                    dst: dst.clone(),
                    src,
                });
            }
        }
        Ok(dst)
    }

    // ── Call / lambda inlining ───────────────────────────────

    fn flatten_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        env: &mut Env,
        lambdas: &mut LambdaStore,
        flat: &mut Vec<FlatOp>,
    ) -> Result<Wire, String> {
        match func {
            // Direct lambda application — beta-reduce immediately.
            Expr::Lambda(params, body) => {
                if params.len() != args.len() {
                    return Err(format!(
                        "arity mismatch: lambda expects {}, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let mut child_env = env.clone();
                let mut child_lam = lambdas.clone();
                for (param, arg) in params.iter().zip(args.iter()) {
                    let wire = self.flatten(arg, env, lambdas, flat)?;
                    child_env.insert(param.clone(), wire);
                }
                self.flatten(body, &mut child_env, &mut child_lam, flat)
            }

            // Named call — look up lambda store and inline if available.
            Expr::Var(name) => {
                if let Some((params, body)) = lambdas.get(name).cloned() {
                    if params.len() != args.len() {
                        return Err(format!(
                            "arity mismatch for `{}`: expects {}, got {}",
                            name,
                            params.len(),
                            args.len()
                        ));
                    }
                    let mut child_env = env.clone();
                    let mut child_lam = lambdas.clone();
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let wire = self.flatten(arg, env, lambdas, flat)?;
                        child_env.insert(param.clone(), wire);
                    }
                    self.flatten(&body, &mut child_env, &mut child_lam, flat)
                } else if env.contains_key(name.as_str()) {
                    // Variable exists but isn't a lambda — emit sentinel.
                    let dst = self.fresh("call");
                    flat.push(FlatOp::Const {
                        dst: dst.clone(),
                        value: 0,
                    });
                    Ok(dst)
                } else {
                    Err(format!("unbound function: {}", name))
                }
            }

            _ => Err("complex call target not supported in circuit compilation".to_string()),
        }
    }

    // ── Phase 1b – constant folding ──────────────────────────

    /// Evaluate any binary / unary op whose operands are both known
    /// constants at compile time, replacing the op with a single Const.
    /// Runs in a single forward pass (SSA ensures defs precede uses).
    fn fold_constants(flat: Vec<FlatOp>) -> Vec<FlatOp> {
        // known[wire] = Some(value) if the wire holds a compile-time constant.
        let mut known: HashMap<Wire, i64> = HashMap::new();
        let mut out: Vec<FlatOp> = Vec::with_capacity(flat.len());

        for op in flat {
            let folded: Option<(Wire, i64)> = match &op {
                FlatOp::Const { dst, value } => {
                    known.insert(dst.clone(), *value);
                    None // keep as-is; will push below
                }
                FlatOp::IAdd32 { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| a.wrapping_add(b))
                        .map(|v| (dst.clone(), v))
                }
                FlatOp::ISub32 { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| a.wrapping_sub(b))
                        .map(|v| (dst.clone(), v))
                }
                FlatOp::IMul { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| a.wrapping_mul(b))
                        .map(|v| (dst.clone(), v))
                }
                FlatOp::ICmpEq { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| if a == b { 1 } else { 0 })
                        .map(|v| (dst.clone(), v))
                }
                FlatOp::ICmpSlt { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| if a < b { 1 } else { 0 })
                        .map(|v| (dst.clone(), v))
                }
                FlatOp::ICmpUlt { dst, lhs, rhs } => Self::fold2(&known, lhs, rhs, |a, b| {
                    if (a as u64) < (b as u64) {
                        1
                    } else {
                        0
                    }
                })
                .map(|v| (dst.clone(), v)),
                FlatOp::BAnd { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| a & b).map(|v| (dst.clone(), v))
                }
                FlatOp::BOr { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| a | b).map(|v| (dst.clone(), v))
                }
                FlatOp::BXor { dst, lhs, rhs } => {
                    Self::fold2(&known, lhs, rhs, |a, b| a ^ b).map(|v| (dst.clone(), v))
                }
                FlatOp::BNot { dst, src } => known.get(src.as_str()).map(|&v| (dst.clone(), !v)),
                FlatOp::Shl { dst, src, amount } => {
                    known.get(src.as_str()).map(|&v| (dst.clone(), v << amount))
                }
                FlatOp::Shr { dst, src, amount } => {
                    known.get(src.as_str()).map(|&v| (dst.clone(), v >> amount))
                }
                _ => None,
            };

            if let Some((dst, value)) = folded {
                known.insert(dst.clone(), value);
                out.push(FlatOp::Const { dst, value });
            } else {
                out.push(op);
            }
        }
        out
    }

    fn fold2(
        known: &HashMap<Wire, i64>,
        lhs: &str,
        rhs: &str,
        f: impl Fn(i64, i64) -> i64,
    ) -> Option<i64> {
        known
            .get(lhs)
            .and_then(|&a| known.get(rhs).map(|&b| f(a, b)))
    }

    // ── Phase 1b – dead-wire elimination ─────────────────────

    /// Remove Const nodes whose destination wire is never used by any
    /// downstream op, by the result wire, or by witness/public-input wires.
    fn eliminate_dead_wires(
        flat: Vec<FlatOp>,
        result_wire: &str,
        witnesses: &[Wire],
        public_inputs: &[Wire],
    ) -> Vec<FlatOp> {
        // Collect every wire that is actually consumed.
        let mut live: HashSet<String> = HashSet::new();
        live.insert(result_wire.to_string());
        for w in witnesses {
            live.insert(w.clone());
        }
        for w in public_inputs {
            live.insert(w.clone());
        }

        // Pre-compute (dst, inputs) pairs so we can do a borrow-free reverse pass.
        let op_data: Vec<(String, Vec<String>)> = flat
            .iter()
            .map(|op| {
                let dst = op.dst().to_string();
                let inputs: Vec<String> = op.inputs().iter().map(|s| s.to_string()).collect();
                (dst, inputs)
            })
            .collect();

        // Single backward pass over SSA: if a dst is live, all its inputs become live.
        for (dst, inputs) in op_data.iter().rev() {
            if live.contains(dst.as_str()) {
                for inp in inputs {
                    live.insert(inp.clone());
                }
            }
        }

        flat.into_iter()
            .filter(|op| {
                // Only prune Const nodes; all other gate ops are kept unconditionally.
                !matches!(op, FlatOp::Const { .. }) || live.contains(op.dst())
            })
            .collect()
    }

    // ── Phase 2 – emit_constraints ───────────────────────────

    fn emit_constraints(flat: &[FlatOp]) -> Vec<Constraint> {
        let mut out = Vec::with_capacity(flat.len());
        for op in flat {
            match op {
                FlatOp::Const { dst, value } => out.push(Constraint::Constant {
                    dst: dst.clone(),
                    value: *value,
                }),

                FlatOp::IAdd32 { dst, lhs, rhs } => out.push(Constraint::IAdd32 {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::ISub32 { dst, lhs, rhs } => out.push(Constraint::ISub32 {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::IMul { dst, lhs, rhs } => out.push(Constraint::IMul {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::IDiv {
                    dst_q,
                    dst_r,
                    lhs,
                    rhs,
                } => out.push(Constraint::NDivCheck {
                    dst_q: dst_q.clone(),
                    dst_r: dst_r.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::ICmpEq { dst, lhs, rhs } => out.push(Constraint::ICmpEq {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::ICmpUlt { dst, lhs, rhs } => out.push(Constraint::ICmpUlt {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::ICmpSlt { dst, lhs, rhs } => out.push(Constraint::ICmpSlt {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::BAnd { dst, lhs, rhs } => out.push(Constraint::BAnd {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::BOr { dst, lhs, rhs } => out.push(Constraint::BOr {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::BXor { dst, lhs, rhs } => out.push(Constraint::BXor {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }),

                FlatOp::BNot { dst, src } => out.push(Constraint::BNot {
                    dst: dst.clone(),
                    src: src.clone(),
                }),

                FlatOp::Shl { dst, src, amount } => out.push(Constraint::Shl {
                    dst: dst.clone(),
                    src: src.clone(),
                    amount: *amount,
                }),

                FlatOp::Shr { dst, src, amount } => out.push(Constraint::Shr {
                    dst: dst.clone(),
                    src: src.clone(),
                    amount: *amount,
                }),

                FlatOp::Mux {
                    dst,
                    cond,
                    t_val,
                    e_val,
                } => out.push(Constraint::Mux {
                    dst: dst.clone(),
                    cond: cond.clone(),
                    t_val: t_val.clone(),
                    e_val: e_val.clone(),
                }),

                FlatOp::PopCount { dst, src } => out.push(Constraint::PopCount {
                    dst: dst.clone(),
                    src: src.clone(),
                }),
            }
        }
        out
    }

    // ── Variable extraction ───────────────────────────────────

    fn extract_variables(&self, expr: &Expr) -> Vec<String> {
        let mut vars = Vec::new();
        self.extract_vars_helper(expr, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn extract_vars_helper(&self, expr: &Expr, vars: &mut Vec<String>) {
        match expr {
            Expr::Var(name) => vars.push(name.clone()),
            Expr::Let(bindings, body) | Expr::LetRec(bindings, body) => {
                for (name, e) in bindings {
                    vars.push(name.clone());
                    self.extract_vars_helper(e, vars);
                }
                for e in body {
                    self.extract_vars_helper(e, vars);
                }
            }
            Expr::BinOp(_, l, r) => {
                self.extract_vars_helper(l, vars);
                self.extract_vars_helper(r, vars);
            }
            Expr::UnOp(_, e) => self.extract_vars_helper(e, vars),
            Expr::Call(f, args) => {
                self.extract_vars_helper(f, vars);
                for a in args {
                    self.extract_vars_helper(a, vars);
                }
            }
            Expr::If(c, t, e) => {
                self.extract_vars_helper(c, vars);
                self.extract_vars_helper(t, vars);
                self.extract_vars_helper(e, vars);
            }
            Expr::Cons(l, r) => {
                self.extract_vars_helper(l, vars);
                self.extract_vars_helper(r, vars);
            }
            Expr::Car(e) | Expr::Cdr(e) => self.extract_vars_helper(e, vars),
            Expr::Quote(e) | Expr::Eval(e) | Expr::Private(e) => self.extract_vars_helper(e, vars),
            Expr::Seq(exprs) => {
                for e in exprs {
                    self.extract_vars_helper(e, vars);
                }
            }
            Expr::Num(_)
            | Expr::Fixed(_)
            | Expr::Str(_)
            | Expr::Char(_)
            | Expr::Bool(_)
            | Expr::Lambda(_, _) => {}
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn add_expr(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Add, Box::new(l), Box::new(r))
    }
    fn sub_expr(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Sub, Box::new(l), Box::new(r))
    }
    fn mul_expr(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(l), Box::new(r))
    }

    #[test]
    fn test_extract_variables() {
        let c = CircuitCompiler::new();
        let expr = add_expr(Expr::Var("x".into()), Expr::Var("y".into()));
        assert_eq!(c.extract_variables(&expr), vec!["x", "y"]);
    }

    #[test]
    fn test_compile_constant() {
        let mut c = CircuitCompiler::new();
        let circuit = c.compile(&Expr::Num(42), &[]).unwrap();
        assert_eq!(circuit.flat_ops.len(), 1);
        assert!(matches!(
            circuit.flat_ops[0],
            FlatOp::Const { value: 42, .. }
        ));
    }

    #[test]
    fn test_compile_add() {
        let mut c = CircuitCompiler::new();
        let expr = add_expr(Expr::Var("x".into()), Expr::Var("y".into()));
        let circuit = c.compile(&expr, &["x".to_string()]).unwrap();
        assert_eq!(circuit.witnesses, vec!["x"]);
        assert_eq!(circuit.public_inputs, vec!["y"]);
        assert!(circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::IAdd32 { .. })));
        assert!(circuit
            .constraints
            .iter()
            .any(|co| matches!(co, Constraint::IAdd32 { .. })));
    }

    #[test]
    fn test_compile_mul() {
        let mut c = CircuitCompiler::new();
        let expr = mul_expr(Expr::Var("x".into()), Expr::Var("y".into()));
        let circuit = c.compile(&expr, &[]).unwrap();
        assert!(circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::IMul { .. })));
    }

    #[test]
    fn test_compile_if_becomes_mux() {
        let mut c = CircuitCompiler::new();
        let cond = Expr::BinOp(
            BinOp::Lt,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let expr = Expr::If(
            Box::new(cond),
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let circuit = c.compile(&expr, &[]).unwrap();
        assert!(circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::Mux { .. })));
        assert!(circuit
            .constraints
            .iter()
            .any(|co| matches!(co, Constraint::Mux { .. })));
    }

    #[test]
    fn test_compile_let_inlining() {
        let mut c = CircuitCompiler::new();
        // (let ((z (+ x 1))) (* z z))
        let inner = add_expr(Expr::Var("x".into()), Expr::Num(1));
        let body = mul_expr(Expr::Var("z".into()), Expr::Var("z".into()));
        let expr = Expr::Let(vec![("z".to_string(), inner)], vec![body]);
        let circuit = c.compile(&expr, &["x".to_string()]).unwrap();
        assert!(circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::IMul { .. })));
    }

    #[test]
    fn test_compile_div_introduces_witness() {
        let mut c = CircuitCompiler::new();
        let expr = Expr::BinOp(
            BinOp::DivExact,
            Box::new(Expr::Var("n".into())),
            Box::new(Expr::Var("d".into())),
        );
        let circuit = c.compile(&expr, &[]).unwrap();
        assert!(circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::IDiv { .. })));
        assert!(circuit
            .constraints
            .iter()
            .any(|co| matches!(co, Constraint::NDivCheck { .. })));
        assert!(!circuit.witnesses.is_empty());
    }

    #[test]
    fn test_lambda_beta_reduction() {
        let mut c = CircuitCompiler::new();
        // ((lambda (x) (* x x)) 5)  — constant folds to Const(25)
        let lam = Expr::Lambda(
            vec!["x".to_string()],
            Box::new(mul_expr(Expr::Var("x".into()), Expr::Var("x".into()))),
        );
        let expr = Expr::Call(Box::new(lam), vec![Expr::Num(5)]);
        let circuit = c.compile(&expr, &[]).unwrap();
        // After beta-reduction and constant folding, result is Const(25).
        let has_25 = circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::Const { value: 25, .. }));
        let has_mul = circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::IMul { .. }));
        assert!(
            has_25 || has_mul,
            "expected Const(25) from folded mul or IMul gate, got {:?}",
            circuit.flat_ops
        );
    }

    /// Named lambda stored in let, then called by name — must be inlined.
    #[test]
    fn test_named_lambda_inlined() {
        let mut c = CircuitCompiler::new();
        // (let ((double (lambda (x) (+ x x)))) (double 7))
        // After inlining and constant folding: Const(14)
        let lam = Expr::Lambda(
            vec!["x".to_string()],
            Box::new(add_expr(Expr::Var("x".into()), Expr::Var("x".into()))),
        );
        let call = Expr::Call(Box::new(Expr::Var("double".into())), vec![Expr::Num(7)]);
        let expr = Expr::Let(vec![("double".to_string(), lam)], vec![call]);
        let circuit = c.compile(&expr, &[]).unwrap();
        let has_14 = circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::Const { value: 14, .. }));
        let has_add = circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::IAdd32 { .. }));
        assert!(
            has_14 || has_add,
            "expected Const(14) from folded add or IAdd32 gate, got {:?}",
            circuit.flat_ops
        );
    }

    /// Constant folding: (+ 3 4) should collapse to a single Const 7.
    #[test]
    fn test_constant_folding() {
        let mut c = CircuitCompiler::new();
        let expr = add_expr(Expr::Num(3), Expr::Num(4));
        let circuit = c.compile(&expr, &[]).unwrap();
        // After folding, only one Const node with value 7.
        let consts: Vec<i64> = circuit
            .flat_ops
            .iter()
            .filter_map(|op| {
                if let FlatOp::Const { value, .. } = op {
                    Some(*value)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            consts.contains(&7),
            "expected folded Const(7), got {:?}",
            circuit.flat_ops
        );
        assert!(
            !circuit
                .flat_ops
                .iter()
                .any(|op| matches!(op, FlatOp::IAdd32 { .. })),
            "IAdd32 should have been folded away"
        );
    }

    /// Dead-wire elimination: `(let ((unused 99)) x)` should drop the Const(99).
    #[test]
    fn test_dead_wire_elimination() {
        let mut c = CircuitCompiler::new();
        let expr = Expr::Let(
            vec![("unused".to_string(), Expr::Num(99))],
            vec![Expr::Var("x".into())],
        );
        let circuit = c.compile(&expr, &["x".to_string()]).unwrap();
        let const_99 = circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::Const { value: 99, .. }));
        assert!(!const_99, "Const(99) should have been eliminated as dead");
    }

    /// Shift with static amount: `(ash-left x 3)` → Shl { amount: 3 }.
    #[test]
    fn test_static_shift_amount() {
        let mut c = CircuitCompiler::new();
        let expr = Expr::BinOp(
            BinOp::AshLeft,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Num(3)),
        );
        let circuit = c.compile(&expr, &[]).unwrap();
        let has_shl3 = circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::Shl { amount: 3, .. }));
        assert!(
            has_shl3,
            "expected Shl {{ amount: 3 }}, got {:?}",
            circuit.flat_ops
        );
    }

    /// Comparison chain: (- (+ x 1) y) compiled; check both Add and Sub emitted.
    #[test]
    fn test_complex_expr_emits_correct_ops() {
        let mut c = CircuitCompiler::new();
        // (- (+ x 1) y)
        let inner = add_expr(Expr::Var("x".into()), Expr::Num(1));
        let expr = sub_expr(inner, Expr::Var("y".into()));
        let circuit = c.compile(&expr, &["x".to_string()]).unwrap();
        assert!(circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::IAdd32 { .. })));
        assert!(circuit
            .flat_ops
            .iter()
            .any(|op| matches!(op, FlatOp::ISub32 { .. })));
    }
}
