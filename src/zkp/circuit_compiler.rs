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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FlatOp {
    /// dst = constant
    Const { dst: Wire, value: i64 },
    /// dst = real-valued constant — only emitted in real-arithmetic (CKKS)
    /// compilation mode, where fixed-point literals keep their true value
    /// instead of the Q16.16 scaled integer.  Never valid in ZKP circuits.
    ConstF { dst: Wire, value: f64 },
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
            | FlatOp::ConstF { dst, .. }
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

    /// Rewrite **every** wire this op mentions — destination included —
    /// through `f`.
    ///
    /// [`rename`] deliberately leaves the destination alone, because it exists
    /// to substitute *operands* in place. Instantiating a template needs the
    /// opposite: a fresh copy of a subgraph with all of its wires renamed.
    pub fn remap(&self, f: &dyn Fn(&str) -> String) -> FlatOp {
        let w = |s: &Wire| f(s);
        match self {
            FlatOp::Const { dst, value } => FlatOp::Const {
                dst: w(dst),
                value: *value,
            },
            FlatOp::ConstF { dst, value } => FlatOp::ConstF {
                dst: w(dst),
                value: *value,
            },
            FlatOp::IAdd32 { dst, lhs, rhs } => FlatOp::IAdd32 {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::ISub32 { dst, lhs, rhs } => FlatOp::ISub32 {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::IMul { dst, lhs, rhs } => FlatOp::IMul {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::ICmpEq { dst, lhs, rhs } => FlatOp::ICmpEq {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::ICmpUlt { dst, lhs, rhs } => FlatOp::ICmpUlt {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::ICmpSlt { dst, lhs, rhs } => FlatOp::ICmpSlt {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::BAnd { dst, lhs, rhs } => FlatOp::BAnd {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::BOr { dst, lhs, rhs } => FlatOp::BOr {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::BXor { dst, lhs, rhs } => FlatOp::BXor {
                dst: w(dst),
                lhs: w(lhs),
                rhs: w(rhs),
            },
            FlatOp::BNot { dst, src } => FlatOp::BNot {
                dst: w(dst),
                src: w(src),
            },
            FlatOp::Shl { dst, src, amount } => FlatOp::Shl {
                dst: w(dst),
                src: w(src),
                amount: *amount,
            },
            FlatOp::Shr { dst, src, amount } => FlatOp::Shr {
                dst: w(dst),
                src: w(src),
                amount: *amount,
            },
            FlatOp::Mux {
                dst,
                cond,
                t_val,
                e_val,
            } => FlatOp::Mux {
                dst: w(dst),
                cond: w(cond),
                t_val: w(t_val),
                e_val: w(e_val),
            },
            FlatOp::PopCount { dst, src } => FlatOp::PopCount {
                dst: w(dst),
                src: w(src),
            },
            FlatOp::IDiv {
                dst_q,
                dst_r,
                lhs,
                rhs,
            } => FlatOp::IDiv {
                dst_q: w(dst_q),
                dst_r: w(dst_r),
                lhs: w(lhs),
                rhs: w(rhs),
            },
        }
    }

    /// All wires this op reads (inputs).
    pub fn inputs(&self) -> Vec<&str> {
        match self {
            FlatOp::Const { .. } | FlatOp::ConstF { .. } => vec![],
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

/// A compile-time value produced by flattening: either a single SSA wire or a
/// compile-time list of values.  Lists exist only at compile time — recursion
/// over them is fully unrolled, so no list ever reaches the gate level.
#[derive(Debug, Clone)]
pub enum CVal {
    Wire(Wire),
    List(Vec<CVal>),
}

/// Maps variable names to compile-time values.
type Env = HashMap<String, CVal>;

/// Maps lambda names to their parameter lists and body, for inlining
/// named functions bound in let/letrec.
type LambdaStore = HashMap<String, (Vec<String>, Expr)>;

/// Upper bound on nested lambda inlining.  Recursion over a compile-time list
/// terminates via static `if` resolution; this guard turns runaway recursion
/// (e.g. a missing base case) into a compile error instead of a hang.
const MAX_INLINE_DEPTH: usize = 4096;

// ─── CircuitCompiler ─────────────────────────────────────────

pub struct CircuitCompiler {
    var_counter: usize,
    witnesses: Vec<Wire>,
    public_inputs: Vec<Wire>,
    /// Wires whose constant value is known during flattening (from literals).
    /// Used to resolve `if` conditions statically, which is what terminates
    /// unrolled recursion over compile-time lists.
    flat_known: HashMap<Wire, i64>,
    /// Current lambda-inlining depth (guarded by MAX_INLINE_DEPTH).
    inline_depth: usize,
    /// Wires carrying Q16.16 fixed-point values.  Tracked statically so
    /// arithmetic can insert the rescale (>>16 after multiply) and coerce
    /// mixed int/fixed operands, mirroring the interpreter's semantics.
    fixed_wires: HashSet<Wire>,
    /// Real-arithmetic mode (CKKS): wires carry true real values, so
    /// fixed-point literals become `ConstF`, int→fixed coercion is the
    /// identity, and no Q16.16 shift/rescale gadgets are emitted.
    real_arith: bool,
    /// Run [`cse`] as part of compilation. On by default — it is a strict win
    /// for every backend. The switch exists so `bench-circuit` can measure
    /// what the pass contributes; turning it off in production only makes
    /// circuits bigger.
    cse_enabled: bool,
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
            flat_known: HashMap::new(),
            inline_depth: 0,
            fixed_wires: HashSet::new(),
            real_arith: false,
            cse_enabled: true,
        }
    }

    /// Disable common-subexpression elimination. For A/B measurement only —
    /// see `Self::cse_enabled`.
    pub fn without_cse(mut self) -> Self {
        self.cse_enabled = false;
        self
    }

    /// A compiler in real-arithmetic mode, for CKKS-style backends that
    /// compute on approximate reals instead of 64-bit integers.
    pub fn new_real() -> Self {
        CircuitCompiler {
            real_arith: true,
            ..Self::new()
        }
    }

    // ── SSA helpers ──────────────────────────────────────────

    fn fresh(&mut self, prefix: &str) -> Wire {
        let w = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        w
    }

    /// Emit a Const op and record its value for static `if` resolution.
    fn emit_const(&mut self, value: i64, flat: &mut Vec<FlatOp>) -> Wire {
        let dst = self.fresh("c");
        self.flat_known.insert(dst.clone(), value);
        flat.push(FlatOp::Const {
            dst: dst.clone(),
            value,
        });
        dst
    }

    /// Emit a Q16.16 fixed-point literal and mark its wire: the scaled
    /// integer for integer backends, a `ConstF` true real in real mode.
    fn emit_fixed_const(&mut self, v: i64, flat: &mut Vec<FlatOp>) -> Wire {
        let w = if self.real_arith {
            let dst = self.fresh("cf");
            flat.push(FlatOp::ConstF {
                dst: dst.clone(),
                value: v as f64 / 65536.0,
            });
            dst
        } else {
            self.emit_const(v, flat)
        };
        self.fixed_wires.insert(w.clone());
        w
    }

    /// Unwrap a CVal that must be a scalar wire.
    fn expect_wire(val: CVal, ctx: &str) -> Result<Wire, String> {
        match val {
            CVal::Wire(w) => Ok(w),
            CVal::List(_) => Err(format!(
                "{}: expected a scalar value, got a compile-time list",
                ctx
            )),
        }
    }

    fn fresh_witness(&mut self, prefix: &str) -> Wire {
        let w = self.fresh(prefix);
        self.witnesses.push(w.clone());
        w
    }

    // ── Q16.16 fixed-point helpers ───────────────────────────

    fn is_fixed(&self, w: &Wire) -> bool {
        self.fixed_wires.contains(w)
    }

    /// Coerce an integer wire into the Q16.16 domain (<<16), like the
    /// interpreter's Num→Fixed coercion.  In real-arithmetic mode int wires
    /// already carry their true value, so coercion is just a domain marking.
    fn make_fixed_wire(&mut self, w: Wire, flat: &mut Vec<FlatOp>) -> Wire {
        if self.real_arith {
            self.fixed_wires.insert(w.clone());
            return w;
        }
        let dst = self.fresh("fx");
        flat.push(FlatOp::Shl {
            dst: dst.clone(),
            src: w,
            amount: 16,
        });
        self.fixed_wires.insert(dst.clone());
        dst
    }

    /// Arithmetic shift right by 16 on a two's-complement u64 wire, built
    /// from logical gates: (x >> 16) | (sign ? 0xFFFF000000000000 : 0).
    /// This is the rescale after a fixed×fixed multiply; `Shr` is logical
    /// in every backend, so the sign correction is explicit.
    fn emit_ashr16(&mut self, src: Wire, flat: &mut Vec<FlatOp>) -> Wire {
        let lo = self.fresh("t");
        flat.push(FlatOp::Shr {
            dst: lo.clone(),
            src: src.clone(),
            amount: 16,
        });
        let sign = self.fresh("t");
        flat.push(FlatOp::Shr {
            dst: sign.clone(),
            src,
            amount: 63,
        });
        let mask = self.emit_const((u64::MAX << 48) as i64, flat);
        let zero = self.emit_const(0, flat);
        let corr = self.fresh("t");
        flat.push(FlatOp::Mux {
            dst: corr.clone(),
            cond: sign,
            t_val: mask,
            e_val: zero,
        });
        let dst = self.fresh("t");
        flat.push(FlatOp::BOr {
            dst: dst.clone(),
            lhs: lo,
            rhs: corr,
        });
        dst
    }

    /// Multiply with fixed-point awareness:
    ///   fixed×fixed → IMul then >>16 rescale (arithmetic)
    ///   fixed×int   → IMul only ((a·2¹⁶)·b is already scaled)
    ///   int×int     → IMul
    fn emit_mul_typed(&mut self, l: Wire, r: Wire, flat: &mut Vec<FlatOp>) -> Wire {
        let lf = self.is_fixed(&l);
        let rf = self.is_fixed(&r);
        let prod = self.fresh("t");
        flat.push(FlatOp::IMul {
            dst: prod.clone(),
            lhs: l,
            rhs: r,
        });
        let dst = if lf && rf && !self.real_arith {
            // Integer backends emulate reals in Q16.16: rescale the product.
            // Real mode multiplies true values, so no rescale exists at all.
            self.emit_ashr16(prod, flat)
        } else {
            prod
        };
        if lf || rf {
            self.fixed_wires.insert(dst.clone());
        }
        dst
    }

    /// Add with fixed-point awareness: a mixed int operand is coerced <<16.
    fn emit_add_typed(&mut self, l: Wire, r: Wire, flat: &mut Vec<FlatOp>) -> Wire {
        let lf = self.is_fixed(&l);
        let rf = self.is_fixed(&r);
        let (l, r) = if lf || rf {
            (
                if lf { l } else { self.make_fixed_wire(l, flat) },
                if rf { r } else { self.make_fixed_wire(r, flat) },
            )
        } else {
            (l, r)
        };
        let dst = self.fresh("t");
        flat.push(FlatOp::IAdd32 {
            dst: dst.clone(),
            lhs: l,
            rhs: r,
        });
        if lf || rf {
            self.fixed_wires.insert(dst.clone());
        }
        dst
    }

    /// Truncate a Q16.16 wire toward zero (matches the interpreter's
    /// `fixed->int`, i.e. v / 65536 with Rust division semantics):
    ///   x >= 0 →  x >> 16
    ///   x <  0 →  -((-x) >> 16)
    fn emit_fixed_to_int(&mut self, x: Wire, flat: &mut Vec<FlatOp>) -> Wire {
        let zero = self.emit_const(0, flat);
        let neg = self.fresh("t");
        flat.push(FlatOp::ICmpSlt {
            dst: neg.clone(),
            lhs: x.clone(),
            rhs: zero.clone(),
        });
        let negx = self.fresh("t");
        flat.push(FlatOp::ISub32 {
            dst: negx.clone(),
            lhs: zero.clone(),
            rhs: x.clone(),
        });
        let qn = self.fresh("t");
        flat.push(FlatOp::Shr {
            dst: qn.clone(),
            src: negx,
            amount: 16,
        });
        let nqn = self.fresh("t");
        flat.push(FlatOp::ISub32 {
            dst: nqn.clone(),
            lhs: zero,
            rhs: qn,
        });
        let qp = self.fresh("t");
        flat.push(FlatOp::Shr {
            dst: qp.clone(),
            src: x,
            amount: 16,
        });
        let dst = self.fresh("t");
        flat.push(FlatOp::Mux {
            dst: dst.clone(),
            cond: neg,
            t_val: nqn,
            e_val: qp,
        });
        dst
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
        let input_vars: Vec<Wire> = self
            .witnesses
            .iter()
            .chain(self.public_inputs.iter())
            .cloned()
            .collect();
        for var in input_vars {
            env.insert(var.clone(), CVal::Wire(var));
        }

        let result_wire = Self::expect_wire(
            self.flatten(expr, &mut env, &mut lambdas, &mut flat)?,
            "circuit output",
        )?;

        // ── Phase 1b: optimise flat IR ────────────────────────
        let flat = Self::fold_constants(flat);
        // CSE before DCE: collapsing duplicates is what *creates* the dead
        // wires DCE then removes.
        let flat = if self.cse_enabled {
            cse(&flat, &result_wire)
        } else {
            flat
        };
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

        for var in self.public_inputs.clone() {
            env.insert(var.clone(), CVal::Wire(var));
        }

        let result_wire = Self::expect_wire(
            self.flatten(expr, &mut env, &mut lambdas, &mut flat)?,
            "circuit output",
        )?;

        let flat = Self::fold_constants(flat);
        let flat = if self.cse_enabled {
            cse(&flat, &result_wire)
        } else {
            flat
        };
        let flat = Self::eliminate_dead_wires(
            flat,
            &result_wire,
            &[], // no ZKP witnesses
            &self.public_inputs,
        );

        // extract_variables over-collects names that never become input
        // wires (builtin call targets like `dot` or `fixed->int`, letrec
        // binders).  Keep only variables some gate actually reads — or the
        // output itself for the identity circuit.
        let used: HashSet<String> = flat
            .iter()
            .flat_map(|op| op.inputs().into_iter().map(String::from))
            .collect();
        self.public_inputs
            .retain(|v| used.contains(v) || *v == result_wire);

        let num_gates = flat.len();
        let output_fixed = self.is_fixed(&result_wire);
        Ok(crate::fhe::backend::FheCircuit {
            flat_ops: flat,
            encrypted_inputs: vec![], // filled in by caller / fhe-compile
            public_inputs: self.public_inputs.clone(),
            output_wire: result_wire,
            num_gates,
            output_fixed,
            real_arith: self.real_arith,
        })
    }

    // ── Phase 1 – flatten ────────────────────────────────────

    fn flatten(
        &mut self,
        expr: &Expr,
        env: &mut Env,
        lambdas: &mut LambdaStore,
        flat: &mut Vec<FlatOp>,
    ) -> Result<CVal, String> {
        match expr {
            // ── Forms a circuit cannot host ───────────────────
            // Refused explicitly rather than approximated. A circuit is a fixed
            // set of gates with no environment and no runtime control flow, so
            // `define` has nothing to extend, `set!` has nothing to mutate, and
            // `apply` needs an argument count that only exists at run time.
            // Silently mis-lowering any of them would produce a circuit that is
            // perfectly sound about the wrong function.
            Expr::Define(name, _) => Err(format!(
                "cannot compile `(define {name} ...)` to a circuit: a circuit has \
                 no environment to extend. Use `let` instead."
            )),
            Expr::Set(name, _) => Err(format!(
                "cannot compile `(set! {name} ...)` to a circuit: circuits are \
                 single-assignment. Thread the value through `let` instead."
            )),
            // ── Literals ─────────────────────────────────────
            Expr::Num(n) => Ok(CVal::Wire(self.emit_const(*n, flat))),
            // Fixed-point literal: the circuit works on the scaled integer
            // directly; the wire is marked so arithmetic inserts rescales.
            // Real mode carries the true value in a ConstF instead.
            Expr::Fixed(v) => Ok(CVal::Wire(self.emit_fixed_const(*v, flat))),
            Expr::Bool(b) => Ok(CVal::Wire(self.emit_const(if *b { 1 } else { 0 }, flat))),
            Expr::Char(c) => Ok(CVal::Wire(self.emit_const(*c as i64, flat))),

            // ── Variable reference ────────────────────────────
            Expr::Var(name) => env
                .get(name)
                .cloned()
                .ok_or_else(|| format!("unbound variable: {}", name)),

            // ── Private annotation (witness metadata) ─────────
            Expr::Private(inner) => self.flatten(inner, env, lambdas, flat),

            // ── Quote: literal data lists become compile-time lists ──
            Expr::Quote(inner) => {
                if let Some(list) = self.quote_to_cval(inner, flat) {
                    Ok(list)
                } else {
                    // Quoted code is not evaluatable in a circuit — sentinel 0.
                    Ok(CVal::Wire(self.emit_const(0, flat)))
                }
            }

            // ── Let / Letrec: inline bindings, then compile body ──
            // (Letrec lambdas may call themselves — the recursion is unrolled
            // at inlining time and terminated by static `if` resolution.)
            Expr::Let(bindings, body) | Expr::LetRec(bindings, body) => {
                let mut child_env = env.clone();
                let mut child_lam = lambdas.clone();
                for (name, rhs) in bindings {
                    // If the RHS is a lambda, store it for later inlining
                    // instead of collapsing it to a sentinel wire.
                    if let Expr::Lambda(params, lam_body) = rhs {
                        child_lam.insert(name.clone(), (params.clone(), *lam_body.clone()));
                    } else {
                        let val = self.flatten(rhs, &mut child_env, &mut child_lam, flat)?;
                        child_env.insert(name.clone(), val);
                    }
                }
                let mut result = CVal::Wire(self.emit_const(0, flat));
                for e in body {
                    result = self.flatten(e, &mut child_env, &mut child_lam, flat)?;
                }
                Ok(result)
            }

            // ── Seq ───────────────────────────────────────────
            Expr::Seq(exprs) => {
                let mut result = CVal::Wire(self.emit_const(0, flat));
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
                    let src = Self::expect_wire(
                        self.flatten(left, env, lambdas, flat)?,
                        "shift operand",
                    )?;
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
                    return Ok(CVal::Wire(dst));
                }
                let l = Self::expect_wire(self.flatten(left, env, lambdas, flat)?, "left operand")?;
                let r =
                    Self::expect_wire(self.flatten(right, env, lambdas, flat)?, "right operand")?;

                // Q16.16 interception: when either operand is fixed-point,
                // mirror the interpreter — coerce the int side (<<16), rescale
                // after multiply, pre-shift the dividend, and mark results.
                let lf = self.is_fixed(&l);
                let rf = self.is_fixed(&r);
                if lf || rf {
                    match op {
                        BinOp::Mul => return Ok(CVal::Wire(self.emit_mul_typed(l, r, flat))),
                        BinOp::Add | BinOp::Sub | BinOp::Mod => {
                            let l = if lf { l } else { self.make_fixed_wire(l, flat) };
                            let r = if rf { r } else { self.make_fixed_wire(r, flat) };
                            let dst = self.flatten_binop(*op, l, r, flat)?;
                            self.fixed_wires.insert(dst.clone());
                            return Ok(CVal::Wire(dst));
                        }
                        BinOp::DivExact => {
                            // (a << 16) / b, both operands scaled.  Real mode
                            // divides true values — no pre-shift.
                            let l = if lf { l } else { self.make_fixed_wire(l, flat) };
                            let r = if rf { r } else { self.make_fixed_wire(r, flat) };
                            let num = if self.real_arith {
                                l
                            } else {
                                let num = self.fresh("t");
                                flat.push(FlatOp::Shl {
                                    dst: num.clone(),
                                    src: l,
                                    amount: 16,
                                });
                                num
                            };
                            let dst = self.flatten_binop(BinOp::DivExact, num, r, flat)?;
                            self.fixed_wires.insert(dst.clone());
                            return Ok(CVal::Wire(dst));
                        }
                        BinOp::Eq | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                            // Compare in the scaled domain; result is a plain bit.
                            let l = if lf { l } else { self.make_fixed_wire(l, flat) };
                            let r = if rf { r } else { self.make_fixed_wire(r, flat) };
                            return Ok(CVal::Wire(self.flatten_binop(*op, l, r, flat)?));
                        }
                        // Bitwise / shift ops stay in the raw integer domain.
                        _ => {}
                    }
                }
                Ok(CVal::Wire(self.flatten_binop(*op, l, r, flat)?))
            }

            // ── Unary operations ──────────────────────────────
            Expr::UnOp(op, inner) => {
                // null? is a compile-time predicate on lists: it folds to a
                // constant, which is what lets static `if` resolution stop
                // recursion over compile-time lists.
                if matches!(op, UnOp::IsNull) {
                    return match self.flatten(inner, env, lambdas, flat)? {
                        CVal::List(items) => Ok(CVal::Wire(
                            self.emit_const(if items.is_empty() { 1 } else { 0 }, flat),
                        )),
                        CVal::Wire(_) => Err(
                            "null? expects a compile-time list (lists exist only at compile time in circuits)"
                                .to_string(),
                        ),
                    };
                }
                let src = Self::expect_wire(
                    self.flatten(inner, env, lambdas, flat)?,
                    "unary operand",
                )?;
                Ok(CVal::Wire(self.flatten_unop(*op, src, flat)?))
            }

            // ── If ────────────────────────────────────────────
            // Compile-time-known condition → flatten only the taken branch
            // (essential: this is what terminates unrolled list recursion).
            // Unknown condition → Mux over both branches.
            Expr::If(cond, then_e, else_e) => {
                let cond_wire =
                    Self::expect_wire(self.flatten(cond, env, lambdas, flat)?, "if condition")?;
                if let Some(&cv) = self.flat_known.get(&cond_wire) {
                    let taken = if cv != 0 { then_e } else { else_e };
                    return self.flatten(taken, env, lambdas, flat);
                }
                let then_wire =
                    Self::expect_wire(self.flatten(then_e, env, lambdas, flat)?, "then branch")?;
                let else_wire =
                    Self::expect_wire(self.flatten(else_e, env, lambdas, flat)?, "else branch")?;
                // If exactly one branch is fixed-point, coerce the other so
                // the mux output lives in a single domain.
                let tf = self.is_fixed(&then_wire);
                let ef = self.is_fixed(&else_wire);
                let then_wire = if ef && !tf {
                    self.make_fixed_wire(then_wire, flat)
                } else {
                    then_wire
                };
                let else_wire = if tf && !ef {
                    self.make_fixed_wire(else_wire, flat)
                } else {
                    else_wire
                };
                let dst = self.fresh("mux");
                flat.push(FlatOp::Mux {
                    dst: dst.clone(),
                    cond: cond_wire,
                    t_val: then_wire,
                    e_val: else_wire,
                });
                if tf || ef {
                    self.fixed_wires.insert(dst.clone());
                }
                Ok(CVal::Wire(dst))
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
                Ok(CVal::Wire(dst))
            }
            Expr::Call(func, args) => self.flatten_call(func, args, env, lambdas, flat),

            // ── List operations (compile-time only) ───────────
            Expr::Cons(left, right) => {
                let head = self.flatten(left, env, lambdas, flat)?;
                match self.flatten(right, env, lambdas, flat)? {
                    CVal::List(mut items) => {
                        items.insert(0, head);
                        Ok(CVal::List(items))
                    }
                    CVal::Wire(_) => {
                        Err("cons: second argument must be a compile-time list".to_string())
                    }
                }
            }
            Expr::Car(inner) => match self.flatten(inner, env, lambdas, flat)? {
                CVal::List(items) => items
                    .first()
                    .cloned()
                    .ok_or_else(|| "car of empty list".to_string()),
                CVal::Wire(_) => Err(
                    "car expects a compile-time list (lists exist only at compile time in circuits)"
                        .to_string(),
                ),
            },
            Expr::Cdr(inner) => match self.flatten(inner, env, lambdas, flat)? {
                CVal::List(items) => {
                    if items.is_empty() {
                        Err("cdr of empty list".to_string())
                    } else {
                        Ok(CVal::List(items[1..].to_vec()))
                    }
                }
                CVal::Wire(_) => Err(
                    "cdr expects a compile-time list (lists exist only at compile time in circuits)"
                        .to_string(),
                ),
            },
            Expr::Eval(inner) => self.flatten(inner, env, lambdas, flat),
            Expr::Str(_) => {
                let dst = self.fresh("str");
                flat.push(FlatOp::Const {
                    dst: dst.clone(),
                    value: 0,
                });
                Ok(CVal::Wire(dst))
            }
        }
    }

    /// Interpret a quoted expression as a literal data list, emitting Const
    /// wires for its elements.  Mirrors `quoted_literal_list` in the
    /// interpreter: '(1 2 3) parses as Quote(Call(Num(1), [Num(2), Num(3)]))
    /// and '() as Quote(Seq([])).  Returns None for quoted code.
    fn quote_to_cval(&mut self, e: &Expr, flat: &mut Vec<FlatOp>) -> Option<CVal> {
        fn elem(cc: &mut CircuitCompiler, e: &Expr, flat: &mut Vec<FlatOp>) -> Option<CVal> {
            match e {
                Expr::Num(n) => Some(CVal::Wire(cc.emit_const(*n, flat))),
                Expr::Fixed(v) => Some(CVal::Wire(cc.emit_fixed_const(*v, flat))),
                Expr::Bool(b) => Some(CVal::Wire(cc.emit_const(if *b { 1 } else { 0 }, flat))),
                Expr::Char(c) => Some(CVal::Wire(cc.emit_const(*c as i64, flat))),
                Expr::UnOp(UnOp::Neg, inner) => match inner.as_ref() {
                    Expr::Num(n) => Some(CVal::Wire(cc.emit_const(-n, flat))),
                    Expr::Fixed(v) => Some(CVal::Wire(cc.emit_fixed_const(-v, flat))),
                    _ => None,
                },
                Expr::Call(_, _) | Expr::Seq(_) => cc.quote_to_cval(e, flat),
                _ => None,
            }
        }
        match e {
            Expr::Seq(items) if items.is_empty() => Some(CVal::List(vec![])),
            Expr::Call(head, rest) => {
                let mut out = Vec::with_capacity(1 + rest.len());
                out.push(elem(self, head, flat)?);
                for item in rest {
                    out.push(elem(self, item, flat)?);
                }
                Some(CVal::List(out))
            }
            _ => None,
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
                // a ≤ b  ⟺  (b < a) XOR 1
                // XOR with 1 is a *logical* NOT on the 0/1 comparison wire;
                // bitwise BNot would yield 0xFFFF…FE, which is not a boolean.
                let lt = self.fresh("lt");
                flat.push(FlatOp::ICmpSlt {
                    dst: lt.clone(),
                    lhs: r,
                    rhs: l,
                });
                let one = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: one.clone(),
                    value: 1,
                });
                flat.push(FlatOp::BXor {
                    dst: dst.clone(),
                    lhs: lt,
                    rhs: one,
                });
            }
            BinOp::Ge => {
                // a ≥ b  ⟺  (a < b) XOR 1   (logical NOT, see BinOp::Le)
                let lt = self.fresh("lt");
                flat.push(FlatOp::ICmpSlt {
                    dst: lt.clone(),
                    lhs: l,
                    rhs: r,
                });
                let one = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: one.clone(),
                    value: 1,
                });
                flat.push(FlatOp::BXor {
                    dst: dst.clone(),
                    lhs: lt,
                    rhs: one,
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
            UnOp::Not => {
                // Logical NOT on a 0/1 boolean wire: XOR with 1.
                // Bitwise BNot would turn 1 into 0xFFFF…FE (still truthy).
                let one = self.fresh("c");
                flat.push(FlatOp::Const {
                    dst: one.clone(),
                    value: 1,
                });
                flat.push(FlatOp::BXor {
                    dst: dst.clone(),
                    lhs: src,
                    rhs: one,
                });
            }
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
            // Handled in `flatten` (compile-time list predicate); a runtime
            // wire can never be a list.
            UnOp::IsNull => {
                return Err(
                    "null? expects a compile-time list (lists exist only at compile time in circuits)"
                        .to_string(),
                );
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
    ) -> Result<CVal, String> {
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
                    let val = self.flatten(arg, env, lambdas, flat)?;
                    child_env.insert(param.clone(), val);
                }
                self.inline_guard(|cc| cc.flatten(body, &mut child_env, &mut child_lam, flat))
            }

            // Named call — look up lambda store and inline if available.
            Expr::Var(name) => {
                // (list a b c) — builtin list constructor.
                if name == "list" {
                    let mut items = Vec::with_capacity(args.len());
                    for arg in args {
                        items.push(self.flatten(arg, env, lambdas, flat)?);
                    }
                    return Ok(CVal::List(items));
                }
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
                        let val = self.flatten(arg, env, lambdas, flat)?;
                        child_env.insert(param.clone(), val);
                    }
                    self.inline_guard(|cc| cc.flatten(&body, &mut child_env, &mut child_lam, flat))
                } else if matches!(name.as_str(), "dot" | "vsum" | "matvec") {
                    // Vector builtins — a user lambda of the same name (looked
                    // up above) shadows these.
                    self.flatten_vector_builtin(name, args, env, lambdas, flat)
                } else if matches!(name.as_str(), "int->fixed" | "fixed->int") {
                    if args.len() != 1 {
                        return Err(format!("{} expects 1 arg, got {}", name, args.len()));
                    }
                    let w = Self::expect_wire(
                        self.flatten(&args[0], env, lambdas, flat)?,
                        name.as_str(),
                    )?;
                    let already_fixed = self.is_fixed(&w);
                    if name == "int->fixed" {
                        Ok(CVal::Wire(if already_fixed {
                            w
                        } else {
                            self.make_fixed_wire(w, flat)
                        }))
                    } else {
                        // fixed->int on an int wire is the identity, like the
                        // interpreter's Num passthrough.
                        if already_fixed && self.real_arith {
                            // Truncation needs comparisons and muxes, which
                            // approximate-real backends cannot evaluate.
                            return Err(
                                "fixed->int is not supported in real-arithmetic (CKKS) circuits"
                                    .to_string(),
                            );
                        }
                        Ok(CVal::Wire(if already_fixed {
                            self.emit_fixed_to_int(w, flat)
                        } else {
                            w
                        }))
                    }
                } else if env.contains_key(name.as_str()) {
                    // Variable exists but isn't a lambda — emit sentinel.
                    let dst = self.fresh("call");
                    flat.push(FlatOp::Const {
                        dst: dst.clone(),
                        value: 0,
                    });
                    Ok(CVal::Wire(dst))
                } else {
                    Err(format!("unbound function: {}", name))
                }
            }

            _ => Err("complex call target not supported in circuit compilation".to_string()),
        }
    }

    /// Vector builtins over compile-time lists, lowered to flat multiply-add
    /// chains (the native replacement for the recursive car/cdr dot-product):
    ///   (dot a b)     — sum of element products; scalar wire
    ///   (vsum v)      — sum of elements; scalar wire
    ///   (matvec m v)  — list of (dot row v) for each row of m
    /// Elements are integer wires; Q16.16 operands are not rescaled here.
    fn flatten_vector_builtin(
        &mut self,
        name: &str,
        args: &[Expr],
        env: &mut Env,
        lambdas: &mut LambdaStore,
        flat: &mut Vec<FlatOp>,
    ) -> Result<CVal, String> {
        let expect_list = |val: CVal, ctx: &str| -> Result<Vec<CVal>, String> {
            match val {
                CVal::List(items) => Ok(items),
                CVal::Wire(_) => Err(format!(
                    "{}: expects a compile-time list (lists exist only at compile time in circuits)",
                    ctx
                )),
            }
        };
        match name {
            "dot" => {
                if args.len() != 2 {
                    return Err(format!("dot expects 2 args, got {}", args.len()));
                }
                let a = expect_list(self.flatten(&args[0], env, lambdas, flat)?, "dot")?;
                let b = expect_list(self.flatten(&args[1], env, lambdas, flat)?, "dot")?;
                Ok(CVal::Wire(self.emit_dot(&a, &b, flat)?))
            }
            "vsum" => {
                if args.len() != 1 {
                    return Err(format!("vsum expects 1 arg, got {}", args.len()));
                }
                let v = expect_list(self.flatten(&args[0], env, lambdas, flat)?, "vsum")?;
                let mut acc: Option<Wire> = None;
                for item in v {
                    let w = Self::expect_wire(item, "vsum element")?;
                    acc = Some(match acc {
                        None => w,
                        Some(s) => self.emit_add_typed(s, w, flat),
                    });
                }
                Ok(CVal::Wire(acc.unwrap_or_else(|| self.emit_const(0, flat))))
            }
            "matvec" => {
                if args.len() != 2 {
                    return Err(format!("matvec expects 2 args, got {}", args.len()));
                }
                let m = expect_list(self.flatten(&args[0], env, lambdas, flat)?, "matvec")?;
                let v = expect_list(self.flatten(&args[1], env, lambdas, flat)?, "matvec")?;
                let mut out = Vec::with_capacity(m.len());
                for row in m {
                    let row = expect_list(row, "matvec row")?;
                    out.push(CVal::Wire(self.emit_dot(&row, &v, flat)?));
                }
                Ok(CVal::List(out))
            }
            _ => unreachable!("flatten_vector_builtin called with '{}'", name),
        }
    }

    /// Emit the multiply-add chain for a dot product of two element vectors.
    fn emit_dot(&mut self, a: &[CVal], b: &[CVal], flat: &mut Vec<FlatOp>) -> Result<Wire, String> {
        if a.len() != b.len() {
            return Err(format!("dot: length mismatch ({} vs {})", a.len(), b.len()));
        }
        let mut acc: Option<Wire> = None;
        for (x, y) in a.iter().zip(b.iter()) {
            let xw = Self::expect_wire(x.clone(), "dot element")?;
            let yw = Self::expect_wire(y.clone(), "dot element")?;
            let prod = self.emit_mul_typed(xw, yw, flat);
            acc = Some(match acc {
                None => prod,
                Some(s) => self.emit_add_typed(s, prod, flat),
            });
        }
        Ok(acc.unwrap_or_else(|| self.emit_const(0, flat)))
    }

    /// Run a lambda-body flatten with the inlining depth guard.
    fn inline_guard<F>(&mut self, f: F) -> Result<CVal, String>
    where
        F: FnOnce(&mut Self) -> Result<CVal, String>,
    {
        self.inline_depth += 1;
        if self.inline_depth > MAX_INLINE_DEPTH {
            self.inline_depth -= 1;
            return Err(format!(
                "lambda inlining exceeded {} levels — recursion must terminate on a \
                 compile-time condition (e.g. (null? lst) over a literal list)",
                MAX_INLINE_DEPTH
            ));
        }
        let result = f(self);
        self.inline_depth -= 1;
        result
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
                    // Logical shift, matching every backend (values live on
                    // u64 wires); `i64 >>` would sign-extend and diverge.
                    known
                        .get(src.as_str())
                        .map(|&v| (dst.clone(), ((v as u64) >> amount) as i64))
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

        // Pre-compute (dsts, inputs) pairs so we can do a borrow-free reverse pass.
        // IDiv is the one op with two destinations (quotient and remainder);
        // it is live if EITHER is consumed — using only `dst()` (the quotient)
        // would prune the inputs of `(mod x 3)`, whose live output is dst_r.
        let op_data: Vec<(Vec<String>, Vec<String>)> = flat
            .iter()
            .map(|op| {
                let dsts: Vec<String> = match op {
                    FlatOp::IDiv { dst_q, dst_r, .. } => vec![dst_q.clone(), dst_r.clone()],
                    _ => vec![op.dst().to_string()],
                };
                let inputs: Vec<String> = op.inputs().iter().map(|s| s.to_string()).collect();
                (dsts, inputs)
            })
            .collect();

        // Single backward pass over SSA: if any dst is live, all inputs become live.
        for (dsts, inputs) in op_data.iter().rev() {
            if dsts.iter().any(|d| live.contains(d.as_str())) {
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

                FlatOp::ConstF { .. } => {
                    unreachable!("ConstF is only emitted in real-arithmetic (CKKS) mode, never ZKP")
                }

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
            // These cannot reach a circuit (`flatten` refuses them), but variable
            // extraction runs before flattening, so they are walked rather than
            // ignored — otherwise a diagnostic could omit the very name at fault.
            Expr::Define(name, value) | Expr::Set(name, value) => {
                vars.push(name.clone());
                self.extract_vars_helper(value, vars);
            }
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

// ─── Shared FlatOp passes (used by the compile pipeline and `mpcc::opt`) ──────

/// Common subexpression elimination over the `FlatOp` SSA.
///
/// Commutative operations are keyed on sorted operands so `a+b` and `b+a`
/// collide. The output wire's definition is never eliminated — the caller's
/// circuit signature depends on it.
///
/// This lives here, beside `CircuitCompiler::fold_constants`, rather than in
/// `mpcc::opt` where it started, because **every** backend needs it: `mpcc` was
/// the only consumer for a while, so a circuit handed to FHE or ZKP kept every
/// redundant gate the source expression happened to contain. A fold written the
/// obvious way (`acc = f(acc, x)` with `acc` textually repeated) duplicates its
/// accumulator subtree at each step and grows *exponentially*: an 8-step
/// max-fold flattens to 4372 ops and collapses to 28 under CSE.
pub fn rename(op: &FlatOp, subst: &HashMap<String, String>) -> FlatOp {
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

pub fn cse(flat: &[FlatOp], output_wire: &str) -> Vec<FlatOp> {
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
