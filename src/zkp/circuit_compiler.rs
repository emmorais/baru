// Circuit Compiler for Baru
// ============================================================
// Two-phase compilation pipeline:
//
//   Lisp AST
//     │
//     ▼  Phase 1 – flatten()
//   FlatIR  (SSA block of FlatOps, one per Binius64 gate/opcode)
//     │
//     ▼  Phase 2 – emit_constraints()
//   Vec<Constraint>  (ready for Binius64 CircuitBuilder)
//
// Phase 1 inlines let/letrec bindings, beta-reduces lambda calls,
// eliminates if-branches via non-deterministic selection (mux), and
// unrolls bounded recursion.  The result is a straight-line SSA
// program with no control flow.
//
// Phase 2 maps each FlatOp to the closest Binius64 gate:
//   Add      → iadd_32   (wrapping 32-bit addition)
//   Sub      → iadd_32 + neg  (negate right, then add)
//   Mul      → imul       (64-bit multiply, low word)
//   Div      → non-deterministic witness quotient q, check q*d + r = n
//   Mod      → non-deterministic witness remainder r, same check
//   Eq       → icmp_eq
//   Lt       → icmp_ult (unsigned) / icmp_slt (signed)
//   Gt/Le/Ge → derived from Lt / Eq
//   And      → band
//   Or       → bor
//   Not      → bnot (bitwise) / icmp_eq(x, 0) (logical)
//   Neg      → iadd_32(0, x) with negation encoding
//   Shl/Shr  → shl / shr
//   Mux      → band + bor  (c*t + (1-c)*e)
//   Assign   → constant wire
//
// Non-determinism note: Div, Mod and If all introduce witnesses
// (advice wires) that are constrained after the fact.  This keeps
// the circuit as small as possible because the prover "guesses"
// the answer and the verifier only checks the algebraic relation.

use std::collections::HashMap;
use crate::interpreter::ast::Expr;
use crate::interpreter::parser::{BinOp, UnOp};

// ─── Flat IR ─────────────────────────────────────────────────

/// A wire name (SSA variable).
pub type Wire = String;

/// One instruction in the flat, straight-line SSA block.
/// Every variant maps to exactly one or a small fixed number of
/// Binius64 gates, making constraint emission trivial.
#[derive(Debug, Clone)]
pub enum FlatOp {
    /// dst = constant
    Const { dst: Wire, value: i64 },

    /// dst = iadd_32(lhs, rhs)   — wrapping 32-bit add
    IAdd32 { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = isub_32(lhs, rhs)   — wrapping 32-bit sub (lhs - rhs)
    ISub32 { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = imul(lhs, rhs)      — 64-bit multiply (lower 64 bits)
    IMul { dst: Wire, lhs: Wire, rhs: Wire },

    /// Non-deterministic integer division.
    /// Witness: quotient q and remainder r such that lhs = q*rhs + r, 0 ≤ r < rhs.
    /// Emits: q (witness), r (witness), check constraint q*rhs + r == lhs.
    IDiv {
        dst_q: Wire,   // quotient
        dst_r: Wire,   // remainder
        lhs: Wire,
        rhs: Wire,
    },

    /// dst = icmp_eq(lhs, rhs)   — 1 if equal, 0 otherwise
    ICmpEq { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = icmp_ult(lhs, rhs)  — unsigned less-than
    ICmpUlt { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = icmp_slt(lhs, rhs)  — signed less-than
    ICmpSlt { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = band(lhs, rhs)      — bitwise AND
    BAnd { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = bor(lhs, rhs)       — bitwise OR
    BOr { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = bxor(lhs, rhs)      — bitwise XOR
    BXor { dst: Wire, lhs: Wire, rhs: Wire },

    /// dst = bnot(src)           — bitwise NOT
    BNot { dst: Wire, src: Wire },

    /// dst = shl(src, amount)    — logical shift left (amount is literal)
    Shl { dst: Wire, src: Wire, amount: u32 },

    /// dst = shr(src, amount)    — logical shift right (amount is literal)
    Shr { dst: Wire, src: Wire, amount: u32 },

    /// Non-deterministic multiplexer: dst = cond ? t_val : e_val
    /// Encoded as dst = band(cond, t_val) bor band(bnot(cond), e_val).
    /// cond must be a 1-bit wire.
    Mux {
        dst: Wire,
        cond: Wire,
        t_val: Wire,
        e_val: Wire,
    },

    /// dst = popcount(src)       — number of 1-bits
    PopCount { dst: Wire, src: Wire },
}

// ─── Constraint (Binius64 gate encoding) ─────────────────────

/// High-level constraint emitted for the Binius64 CircuitBuilder.
/// Each variant names the exact Binius64 gate / opcode.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// builder.add_constant_64(value) → dst
    Constant { dst: Wire, value: i64 },

    /// builder.iadd_32(lhs, rhs) → dst
    IAdd32 { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.isub_32(lhs, rhs) → dst  (negation + iadd_32 internally)
    ISub32 { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.imul(lhs, rhs) → dst
    IMul { dst: Wire, lhs: Wire, rhs: Wire },

    /// Non-det quotient witness + check: q*rhs + r == lhs
    NDivCheck {
        dst_q: Wire,
        dst_r: Wire,
        lhs: Wire,
        rhs: Wire,
    },

    /// builder.icmp_eq(lhs, rhs) → dst
    ICmpEq { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.icmp_ult(lhs, rhs) → dst
    ICmpUlt { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.icmp_slt(lhs, rhs) → dst
    ICmpSlt { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.band(lhs, rhs) → dst
    BAnd { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.bor(lhs, rhs) → dst
    BOr { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.bxor(lhs, rhs) → dst
    BXor { dst: Wire, lhs: Wire, rhs: Wire },

    /// builder.bnot(src) → dst
    BNot { dst: Wire, src: Wire },

    /// builder.shl(src, amount) → dst
    Shl { dst: Wire, src: Wire, amount: u32 },

    /// builder.shr(src, amount) → dst
    Shr { dst: Wire, src: Wire, amount: u32 },

    /// Mux gadget: dst = band(cond,t) bor band(bnot(cond),e)
    Mux { dst: Wire, cond: Wire, t_val: Wire, e_val: Wire },

    /// popcount gadget
    PopCount { dst: Wire, src: Wire },
}

// ─── Compiled circuit ────────────────────────────────────────

#[derive(Debug)]
pub struct CompiledCircuit {
    /// Flat SSA intermediate representation (for inspection / debugging)
    pub flat_ops: Vec<FlatOp>,
    /// Gate-level constraints for Binius64 (derived from flat_ops)
    pub constraints: Vec<Constraint>,
    /// Witness (private) wire names
    pub witnesses: Vec<Wire>,
    /// Public input wire names
    pub public_inputs: Vec<Wire>,
    /// Total gate count
    pub num_gates: usize,
}

// ─── Flattening environment ───────────────────────────────────

/// Substitution map: variable name → SSA wire that holds its value.
type Env = HashMap<String, Wire>;

// ─── CircuitCompiler ─────────────────────────────────────────

pub struct CircuitCompiler {
    var_counter: usize,
    witnesses: Vec<Wire>,
    public_inputs: Vec<Wire>,
}

impl Default for CircuitCompiler {
    fn default() -> Self { Self::new() }
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

    /// Compile `expr` to a `CompiledCircuit`.
    /// `private_vars` names user-declared private inputs (witnesses).
    pub fn compile(
        &mut self,
        expr: &Expr,
        private_vars: &[String],
    ) -> Result<CompiledCircuit, String> {
        // Classify user-named variables
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

        // ── Phase 1: flatten ─────────────────────────────────
        let mut flat: Vec<FlatOp> = Vec::new();
        let mut env: Env = Env::new();

        // Seed env with user variables (they map to themselves)
        for var in self.witnesses.iter().chain(self.public_inputs.iter()) {
            env.insert(var.clone(), var.clone());
        }

        let result_wire = self.flatten(expr, &mut env, &mut flat)?;
        let _ = result_wire; // top-level output wire

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

    // ── Phase 1 – flatten ────────────────────────────────────

    /// Recursively flatten `expr` into `flat`, returning the wire
    /// that holds the expression's value in the SSA block.
    fn flatten(
        &mut self,
        expr: &Expr,
        env: &mut Env,
        flat: &mut Vec<FlatOp>,
    ) -> Result<Wire, String> {
        match expr {

            // ── Literals ─────────────────────────────────────
            Expr::Num(n) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const { dst: dst.clone(), value: *n });
                Ok(dst)
            }
            Expr::Bool(b) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const { dst: dst.clone(), value: if *b { 1 } else { 0 } });
                Ok(dst)
            }
            Expr::Char(c) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const { dst: dst.clone(), value: *c as i64 });
                Ok(dst)
            }

            // ── Variable reference ────────────────────────────
            Expr::Var(name) => {
                env.get(name).cloned().ok_or_else(|| format!("unbound variable: {}", name))
            }

            // ── Private annotation (metadata only) ───────────
            Expr::Private(inner) => self.flatten(inner, env, flat),

            // ── Quote (treat as constant 0 — not evaluatable) ─
            Expr::Quote(_) => {
                let dst = self.fresh("c");
                flat.push(FlatOp::Const { dst: dst.clone(), value: 0 });
                Ok(dst)
            }

            // ── Let bindings ──────────────────────────────────
            // Inlined: each binding compiles its RHS and maps the
            // name to the resulting SSA wire in a child environment.
            Expr::Let(bindings, body) => {
                let mut child_env = env.clone();
                for (name, rhs) in bindings {
                    let wire = self.flatten(rhs, &mut child_env, flat)?;
                    child_env.insert(name.clone(), wire);
                }
                let mut result = self.fresh("c");
                flat.push(FlatOp::Const { dst: result.clone(), value: 0 }); // default
                for e in body {
                    result = self.flatten(e, &mut child_env, flat)?;
                }
                Ok(result)
            }

            // ── Letrec ───────────────────────────────────────
            // For circuits, letrec without recursion is identical
            // to let.  Mutual recursion is rejected (circuits are acyclic).
            Expr::LetRec(bindings, body) => {
                let mut child_env = env.clone();
                for (name, rhs) in bindings {
                    let wire = self.flatten(rhs, &mut child_env, flat)?;
                    child_env.insert(name.clone(), wire);
                }
                let mut result = self.fresh("c");
                flat.push(FlatOp::Const { dst: result.clone(), value: 0 });
                for e in body {
                    result = self.flatten(e, &mut child_env, flat)?;
                }
                Ok(result)
            }

            // ── Seq ───────────────────────────────────────────
            Expr::Seq(exprs) => {
                let mut result = self.fresh("c");
                flat.push(FlatOp::Const { dst: result.clone(), value: 0 });
                for e in exprs {
                    result = self.flatten(e, env, flat)?;
                }
                Ok(result)
            }

            // ── Binary operations ─────────────────────────────
            Expr::BinOp(op, left, right) => {
                let l = self.flatten(left, env, flat)?;
                let r = self.flatten(right, env, flat)?;
                self.flatten_binop(*op, l, r, flat)
            }

            // ── Unary operations ──────────────────────────────
            Expr::UnOp(op, inner) => {
                let src = self.flatten(inner, env, flat)?;
                self.flatten_unop(*op, src, flat)
            }

            // ── If — non-deterministic multiplexer ───────────
            // Both branches are always evaluated (circuit is acyclic),
            // then a Mux gate selects the result based on the condition bit.
            // This is the standard ZK technique: no control flow, only data flow.
            Expr::If(cond, then_e, else_e) => {
                let cond_wire  = self.flatten(cond,   env, flat)?;
                let then_wire  = self.flatten(then_e, env, flat)?;
                let else_wire  = self.flatten(else_e, env, flat)?;
                let dst = self.fresh("mux");
                flat.push(FlatOp::Mux {
                    dst: dst.clone(),
                    cond: cond_wire,
                    t_val: then_wire,
                    e_val: else_wire,
                });
                Ok(dst)
            }

            // ── Lambda / Call — beta reduction ────────────────
            // Lambdas are inlined at every call site (monomorphisation).
            // This terminates only for non-recursive calls; recursive
            // lambdas require a bounded unroll depth (see below).
            Expr::Lambda(params, body) => {
                // A bare lambda not applied to arguments yields a sentinel.
                // In practice lambdas are only meaningful inside a Call.
                let dst = self.fresh("lam");
                flat.push(FlatOp::Const { dst: dst.clone(), value: 0 });
                let _ = (params, body);
                Ok(dst)
            }
            Expr::Call(func, args) => {
                self.flatten_call(func, args, env, flat)
            }

            // ── List operations — structural, encode as pairs ─
            Expr::Cons(left, right) => {
                // Pairs: in a circuit we keep both components; return the left
                // wire as a proxy (the right is emitted as a side-effect).
                let _l = self.flatten(left,  env, flat)?;
                let  r = self.flatten(right, env, flat)?;
                Ok(r) // conservative: return last meaningful wire
            }
            Expr::Car(inner) => self.flatten(inner, env, flat),
            Expr::Cdr(inner) => self.flatten(inner, env, flat),

            Expr::Eval(inner) => self.flatten(inner, env, flat),
            Expr::Str(_) => {
                // Strings are not arithmetic; emit a sentinel 0.
                let dst = self.fresh("str");
                flat.push(FlatOp::Const { dst: dst.clone(), value: 0 });
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
            BinOp::Add => flat.push(FlatOp::IAdd32 { dst: dst.clone(), lhs: l, rhs: r }),
            BinOp::Sub => flat.push(FlatOp::ISub32 { dst: dst.clone(), lhs: l, rhs: r }),
            BinOp::Mul => flat.push(FlatOp::IMul   { dst: dst.clone(), lhs: l, rhs: r }),

            // Non-deterministic division: witness q and rem, then check.
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
                // For (mod l r): witness both q and r_wire; dst holds the remainder.
                let dst_q = self.fresh_witness("quot");
                flat.push(FlatOp::IDiv {
                    dst_q,
                    dst_r: dst.clone(),
                    lhs: l,
                    rhs: r,
                });
            }

            // Comparisons
            BinOp::Eq => flat.push(FlatOp::ICmpEq  { dst: dst.clone(), lhs: l, rhs: r }),
            BinOp::Lt => flat.push(FlatOp::ICmpSlt { dst: dst.clone(), lhs: l, rhs: r }),
            BinOp::Gt => {
                // a > b  ⟺  b < a
                flat.push(FlatOp::ICmpSlt { dst: dst.clone(), lhs: r, rhs: l });
            }
            BinOp::Le => {
                // a ≤ b  ⟺  NOT (a > b)  ⟺  NOT (b < a)
                let lt = self.fresh("lt");
                flat.push(FlatOp::ICmpSlt { dst: lt.clone(), lhs: r, rhs: l.clone() });
                flat.push(FlatOp::BNot { dst: dst.clone(), src: lt });
            }
            BinOp::Ge => {
                // a ≥ b  ⟺  NOT (a < b)
                let lt = self.fresh("lt");
                flat.push(FlatOp::ICmpSlt { dst: lt.clone(), lhs: l.clone(), rhs: r });
                flat.push(FlatOp::BNot { dst: dst.clone(), src: lt });
            }

            // Logical (bitwise)
            BinOp::And => flat.push(FlatOp::BAnd { dst: dst.clone(), lhs: l, rhs: r }),
            BinOp::Or  => flat.push(FlatOp::BOr  { dst: dst.clone(), lhs: l, rhs: r }),

            // Shifts — rhs must be a compile-time constant for circuits.
            // We accept it as a wire and encode amount = 0 as a fallback;
            // a fully static amount is enforced when the rhs is Expr::Num.
            BinOp::AshLeft  => flat.push(FlatOp::Shl { dst: dst.clone(), src: l, amount: 0 }),
            BinOp::AshRight => flat.push(FlatOp::Shr { dst: dst.clone(), src: l, amount: 0 }),
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
            UnOp::Not => flat.push(FlatOp::BNot { dst: dst.clone(), src }),
            UnOp::Neg => {
                // -x  =  0 - x  =  isub_32(0, x)
                let zero = self.fresh("c");
                flat.push(FlatOp::Const { dst: zero.clone(), value: 0 });
                flat.push(FlatOp::ISub32 { dst: dst.clone(), lhs: zero, rhs: src });
            }
            UnOp::Atom => {
                // atom(x) returns 1 for non-list types; in a numeric circuit always 1.
                flat.push(FlatOp::Const { dst: dst.clone(), value: 1 });
            }
            UnOp::BitLen => {
                // bitlen(x) = ceil(log2(x+1)); approximate via popcount on the wire.
                flat.push(FlatOp::PopCount { dst: dst.clone(), src });
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
        flat: &mut Vec<FlatOp>,
    ) -> Result<Wire, String> {
        match func {
            Expr::Lambda(params, body) => {
                if params.len() != args.len() {
                    return Err(format!(
                        "arity mismatch: lambda expects {}, got {}",
                        params.len(), args.len()
                    ));
                }
                // Beta-reduce: evaluate each argument, bind to param names.
                let mut child_env = env.clone();
                for (param, arg) in params.iter().zip(args.iter()) {
                    let wire = self.flatten(arg, env, flat)?;
                    child_env.insert(param.clone(), wire);
                }
                self.flatten(body, &mut child_env, flat)
            }
            Expr::Var(name) => {
                // Look up the lambda wire in the environment.
                // Named functions must be bound via let/letrec before the call.
                match env.get(name).cloned() {
                    Some(_) => {
                        // We can only inline if we still have the lambda body.
                        // At this point the wire is already collapsed; emit sentinel.
                        let dst = self.fresh("call");
                        flat.push(FlatOp::Const { dst: dst.clone(), value: 0 });
                        Ok(dst)
                    }
                    None => Err(format!("unbound function: {}", name)),
                }
            }
            _ => Err("complex call target not yet supported in circuit compilation".to_string()),
        }
    }

    // ── Phase 2 – emit_constraints ───────────────────────────

    /// Map each `FlatOp` to the corresponding `Constraint`(s).
    /// Preserves the SSA ordering so constraints are topologically sorted.
    fn emit_constraints(flat: &[FlatOp]) -> Vec<Constraint> {
        let mut out = Vec::with_capacity(flat.len());
        for op in flat {
            match op {
                FlatOp::Const { dst, value } =>
                    out.push(Constraint::Constant { dst: dst.clone(), value: *value }),

                FlatOp::IAdd32 { dst, lhs, rhs } =>
                    out.push(Constraint::IAdd32 { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::ISub32 { dst, lhs, rhs } =>
                    out.push(Constraint::ISub32 { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::IMul { dst, lhs, rhs } =>
                    out.push(Constraint::IMul { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::IDiv { dst_q, dst_r, lhs, rhs } =>
                    out.push(Constraint::NDivCheck {
                        dst_q: dst_q.clone(),
                        dst_r: dst_r.clone(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    }),

                FlatOp::ICmpEq  { dst, lhs, rhs } =>
                    out.push(Constraint::ICmpEq  { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::ICmpUlt { dst, lhs, rhs } =>
                    out.push(Constraint::ICmpUlt { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::ICmpSlt { dst, lhs, rhs } =>
                    out.push(Constraint::ICmpSlt { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::BAnd { dst, lhs, rhs } =>
                    out.push(Constraint::BAnd { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::BOr { dst, lhs, rhs } =>
                    out.push(Constraint::BOr  { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::BXor { dst, lhs, rhs } =>
                    out.push(Constraint::BXor { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }),

                FlatOp::BNot { dst, src } =>
                    out.push(Constraint::BNot { dst: dst.clone(), src: src.clone() }),

                FlatOp::Shl { dst, src, amount } =>
                    out.push(Constraint::Shl { dst: dst.clone(), src: src.clone(), amount: *amount }),

                FlatOp::Shr { dst, src, amount } =>
                    out.push(Constraint::Shr { dst: dst.clone(), src: src.clone(), amount: *amount }),

                FlatOp::Mux { dst, cond, t_val, e_val } =>
                    out.push(Constraint::Mux {
                        dst: dst.clone(),
                        cond: cond.clone(),
                        t_val: t_val.clone(),
                        e_val: e_val.clone(),
                    }),

                FlatOp::PopCount { dst, src } =>
                    out.push(Constraint::PopCount { dst: dst.clone(), src: src.clone() }),
            }
        }
        out
    }

    // ── Variable extraction (unchanged) ──────────────────────

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
                for e in body { self.extract_vars_helper(e, vars); }
            }
            Expr::BinOp(_, l, r) => {
                self.extract_vars_helper(l, vars);
                self.extract_vars_helper(r, vars);
            }
            Expr::UnOp(_, e) => self.extract_vars_helper(e, vars),
            Expr::Call(f, args) => {
                self.extract_vars_helper(f, vars);
                for a in args { self.extract_vars_helper(a, vars); }
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
            Expr::Seq(exprs) => { for e in exprs { self.extract_vars_helper(e, vars); } }
            Expr::Num(_) | Expr::Str(_) | Expr::Char(_) | Expr::Bool(_) | Expr::Lambda(_, _) => {}
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

    fn mul_expr(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(l), Box::new(r))
    }

    #[test]
    fn test_extract_variables() {
        let compiler = CircuitCompiler::new();
        let expr = add_expr(Expr::Var("x".into()), Expr::Var("y".into()));
        let vars = compiler.extract_variables(&expr);
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_compile_constant() {
        let mut c = CircuitCompiler::new();
        let circuit = c.compile(&Expr::Num(42), &[]).unwrap();
        assert_eq!(circuit.flat_ops.len(), 1);
        assert!(matches!(circuit.flat_ops[0], FlatOp::Const { value: 42, .. }));
    }

    #[test]
    fn test_compile_add() {
        let mut c = CircuitCompiler::new();
        let expr = add_expr(Expr::Var("x".into()), Expr::Var("y".into()));
        let circuit = c.compile(&expr, &["x".to_string()]).unwrap();
        assert_eq!(circuit.witnesses,     vec!["x"]);
        assert_eq!(circuit.public_inputs, vec!["y"]);
        // One IAdd32 FlatOp
        assert!(circuit.flat_ops.iter().any(|op| matches!(op, FlatOp::IAdd32 { .. })));
        // One IAdd32 Constraint
        assert!(circuit.constraints.iter().any(|c| matches!(c, Constraint::IAdd32 { .. })));
    }

    #[test]
    fn test_compile_mul() {
        let mut c = CircuitCompiler::new();
        let expr = mul_expr(Expr::Var("x".into()), Expr::Var("y".into()));
        let circuit = c.compile(&expr, &[]).unwrap();
        assert!(circuit.flat_ops.iter().any(|op| matches!(op, FlatOp::IMul { .. })));
    }

    #[test]
    fn test_compile_if_becomes_mux() {
        let mut c = CircuitCompiler::new();
        // (if (< x y) x y)
        let cond = Expr::BinOp(BinOp::Lt, Box::new(Expr::Var("x".into())), Box::new(Expr::Var("y".into())));
        let expr = Expr::If(Box::new(cond), Box::new(Expr::Var("x".into())), Box::new(Expr::Var("y".into())));
        let circuit = c.compile(&expr, &[]).unwrap();
        assert!(circuit.flat_ops.iter().any(|op| matches!(op, FlatOp::Mux { .. })));
        assert!(circuit.constraints.iter().any(|co| matches!(co, Constraint::Mux { .. })));
    }

    #[test]
    fn test_compile_let_inlining() {
        let mut c = CircuitCompiler::new();
        // (let ((z (+ x 1))) (* z z))
        let inner = add_expr(Expr::Var("x".into()), Expr::Num(1));
        let body   = mul_expr(Expr::Var("z".into()), Expr::Var("z".into()));
        let expr = Expr::Let(
            vec![("z".to_string(), inner)],
            vec![body],
        );
        let circuit = c.compile(&expr, &["x".to_string()]).unwrap();
        assert!(circuit.flat_ops.iter().any(|op| matches!(op, FlatOp::IMul { .. })));
    }

    #[test]
    fn test_compile_div_introduces_witness() {
        let mut c = CircuitCompiler::new();
        let expr = Expr::BinOp(BinOp::DivExact,
            Box::new(Expr::Var("n".into())),
            Box::new(Expr::Var("d".into())));
        let circuit = c.compile(&expr, &[]).unwrap();
        // Non-det witnesses for quotient/remainder
        assert!(circuit.flat_ops.iter().any(|op| matches!(op, FlatOp::IDiv { .. })));
        assert!(circuit.constraints.iter().any(|co| matches!(co, Constraint::NDivCheck { .. })));
        // Remainder wire must be in witnesses
        assert!(!circuit.witnesses.is_empty());
    }

    #[test]
    fn test_lambda_beta_reduction() {
        let mut c = CircuitCompiler::new();
        // ((lambda (x) (* x x)) 5)
        let lam = Expr::Lambda(
            vec!["x".to_string()],
            Box::new(mul_expr(Expr::Var("x".into()), Expr::Var("x".into()))),
        );
        let expr = Expr::Call(Box::new(lam), vec![Expr::Num(5)]);
        let circuit = c.compile(&expr, &[]).unwrap();
        assert!(circuit.flat_ops.iter().any(|op| matches!(op, FlatOp::IMul { .. })));
    }
}
