//! A **real** Binius64 backend — an actual proof system, not a placeholder.
//!
//! [`super::binius64_backend`] ships a stand-in whose `verify` never checks
//! that a witness satisfies the constraints; `tests/zkp_soundness.rs`
//! demonstrates that anyone holding the (public) circuit can forge an accepting
//! proof. This module replaces it with the genuine Binius64 stack:
//! `binius_frontend` to build the constraint system, `binius_prover` to prove,
//! `binius_verifier` to verify.
//!
//! # How a statement is bound
//!
//! The verifier must be convinced of something specific, so the circuit is
//! built with:
//!
//! * one **witness** wire per private variable — never revealed;
//! * one **inout** (public) wire per public variable;
//! * one **inout** wire for the claimed output, constrained by `assert_eq` to
//!   equal the value the circuit actually computes.
//!
//! Soundness therefore says: *the prover knows private values which, combined
//! with the public inputs, make the circuit evaluate to the claimed output.*
//! A prover without such values cannot produce an accepting transcript.
//!
//! # Why an explicit IR
//!
//! Both prover and verifier must build a **bit-identical** constraint system,
//! or their transcripts diverge and every proof fails. Baru's [`Expr`] is not
//! serializable, and round-tripping it through `Display` and the parser would
//! make circuit identity depend on pretty-printing.
//!
//! So [`compile_circuit`](Binius64Real::compile_circuit) lowers the expression
//! once into [`Ir`] — a flat, serializable list of operations over wire indices
//! — and stores *that* in the circuit. [`build`] turns an `Ir` into a Binius
//! circuit deterministically, and is the only path either side uses.
//!
//! # Scope, stated
//!
//! Arithmetic circuits over `+`, `−`, `×` on 64-bit words, with integer
//! semantics matching Baru's `i64` (wrapping, two's-complement). Comparisons,
//! division and control flow are **rejected with an error** rather than
//! silently mis-compiled: each needs bit-decomposition circuitry that is real
//! work, and a backend that quietly accepted them would be repeating the
//! mistake this module exists to correct.

use std::collections::HashMap;

use binius_frontend::{CircuitBuilder, Wire};
use binius_verifier::config::StdChallenger;
use binius_prover::OptimalPackedB128;
use binius_verifier::hash::sha256::Sha256HashSuite;
use serde::{Deserialize, Serialize};

use super::{Circuit, CircuitMetadata, PublicValues, ZKPBackend, ZKProof};
use crate::interpreter::ast::Expr;
use crate::interpreter::parser::BinOp;

/// Reed–Solomon inverse rate. Higher means smaller proofs and slower proving;
/// 1 is the value the upstream examples use by default.
const LOG_INV_RATE: usize = 1;

/// One node of the flattened circuit. Indices refer to earlier entries in
/// [`Ir::ops`], so the list is in topological order by construction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
enum Op {
    Const(i64),
    /// A public input, by name.
    Public(String),
    /// A private input, by name.
    Witness(String),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
}

/// The serializable circuit description both sides rebuild from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Ir {
    ops: Vec<Op>,
    /// Index of the value the circuit computes.
    output: u32,
    /// Public input names, in the order their wires are created.
    public: Vec<String>,
    /// Private input names, in creation order.
    private: Vec<String>,
}

/// The name under which the claimed output is exposed as a public value.
pub const OUTPUT_NAME: &str = "__out";

impl Ir {
    /// Lower a Baru expression, recording which variables are private.
    fn lower(expr: &Expr, private_vars: &[String]) -> Result<Ir, String> {
        let mut ir = Ir {
            ops: Vec::new(),
            output: 0,
            public: Vec::new(),
            private: Vec::new(),
        };
        let mut seen: HashMap<String, u32> = HashMap::new();
        let out = ir.walk(expr, private_vars, &mut seen)?;
        ir.output = out;
        Ok(ir)
    }

    fn push(&mut self, op: Op) -> u32 {
        self.ops.push(op);
        (self.ops.len() - 1) as u32
    }

    fn walk(
        &mut self,
        expr: &Expr,
        private_vars: &[String],
        seen: &mut HashMap<String, u32>,
    ) -> Result<u32, String> {
        match expr {
            Expr::Num(n) => Ok(self.push(Op::Const(*n))),
            Expr::Fixed(v) => Ok(self.push(Op::Const(*v))),
            Expr::Bool(b) => Ok(self.push(Op::Const(*b as i64))),
            Expr::Var(name) => {
                if let Some(&idx) = seen.get(name) {
                    return Ok(idx);
                }
                let idx = if private_vars.iter().any(|p| p == name) {
                    self.private.push(name.clone());
                    self.push(Op::Witness(name.clone()))
                } else {
                    self.public.push(name.clone());
                    self.push(Op::Public(name.clone()))
                };
                seen.insert(name.clone(), idx);
                Ok(idx)
            }
            Expr::Private(inner) => self.walk(inner, private_vars, seen),
            Expr::BinOp(op, a, b) => {
                let x = self.walk(a, private_vars, seen)?;
                let y = self.walk(b, private_vars, seen)?;
                let node = match op {
                    BinOp::Add => Op::Add(x, y),
                    BinOp::Sub => Op::Sub(x, y),
                    BinOp::Mul => Op::Mul(x, y),
                    other => {
                        return Err(format!(
                            "binius64: {other:?} is not supported yet. Only +, - and * are \
                             compiled; comparisons and division need bit-decomposition \
                             circuitry, and accepting them here would silently prove the \
                             wrong statement."
                        ))
                    }
                };
                Ok(self.push(node))
            }
            other => Err(format!(
                "binius64: unsupported expression {other:?}. This backend compiles arithmetic \
                 circuits over +, - and *; anything else is refused rather than mis-compiled."
            )),
        }
    }
}

/// Build the Binius circuit for an [`Ir`]. Deterministic: prover and verifier
/// both call this and must obtain identical constraint systems.
fn build(ir: &Ir) -> (binius_frontend::Circuit, HashMap<String, Wire>, Wire) {
    let b = CircuitBuilder::new();
    let mut wires: Vec<Wire> = Vec::with_capacity(ir.ops.len());
    let mut named: HashMap<String, Wire> = HashMap::new();

    for op in &ir.ops {
        let w = match op {
            Op::Const(v) => b.add_constant_64(*v as u64),
            Op::Public(name) => {
                let w = b.add_inout();
                named.insert(name.clone(), w);
                w
            }
            Op::Witness(name) => {
                let w = b.add_witness();
                named.insert(name.clone(), w);
                w
            }
            Op::Add(x, y) => {
                // `iadd` returns (sum, carry); Baru's i64 arithmetic wraps, so
                // the carry is discarded deliberately rather than overlooked.
                let (sum, _carry) = b.iadd(wires[*x as usize], wires[*y as usize]);
                sum
            }
            Op::Sub(x, y) => {
                // a − b = a + (¬b + 1), the standard two's-complement identity.
                let ones = b.add_constant_64(u64::MAX);
                let neg = b.bxor(wires[*y as usize], ones);
                let one = b.add_constant_64(1);
                let (nb, _c1) = b.iadd(neg, one);
                let (diff, _c2) = b.iadd(wires[*x as usize], nb);
                diff
            }
            Op::Mul(x, y) => {
                // `smul` is the *signed* 64x64 -> 128 multiply, matching Baru's
                // i64 semantics, and it returns `(hi, lo)` — high word first.
                // Destructuring that pair the other way round silently yields
                // the high word, which is 0 for every small product and so looks
                // like "the witness was never populated".
                let (_hi, lo) = b.smul(wires[*x as usize], wires[*y as usize]);
                lo
            }
        };
        wires.push(w);
    }

    // Bind the claimed output to a public wire. Without this the statement
    // would be "some circuit was evaluated", which proves nothing useful.
    let out = b.add_inout();
    named.insert(OUTPUT_NAME.to_string(), out);
    b.assert_eq("output", wires[ir.output as usize], out);

    (b.build(), named, out)
}

/// The real backend.
pub struct Binius64Real;

impl Default for Binius64Real {
    fn default() -> Self {
        Self::new()
    }
}

impl Binius64Real {
    pub fn new() -> Self {
        Binius64Real
    }

    fn ir_of(circuit: &Circuit) -> Result<Ir, String> {
        bincode::deserialize(&circuit.circuit_data)
            .map_err(|e| format!("binius64: malformed circuit data: {e}"))
    }

    /// Evaluate the circuit in the clear, to learn the output a proof would
    /// claim.
    ///
    /// A prover always knows this — it holds the witness — so exposing it is
    /// not a leak, it is a convenience that keeps callers from having to
    /// duplicate the circuit's semantics to fill in [`OUTPUT_NAME`]. Uses the
    /// same wrapping `i64` arithmetic the circuit does, so the value it returns
    /// is exactly the one the constraints accept.
    pub fn evaluate(
        &self,
        circuit: &Circuit,
        private_values: &HashMap<String, i64>,
        public_values: &PublicValues,
    ) -> Result<i64, String> {
        let ir = Self::ir_of(circuit)?;
        let mut vals: Vec<i64> = Vec::with_capacity(ir.ops.len());
        for op in &ir.ops {
            let v = match op {
                Op::Const(c) => *c,
                Op::Public(name) => *public_values
                    .get(name)
                    .ok_or_else(|| format!("binius64: missing public value for `{name}`"))?,
                Op::Witness(name) => *private_values
                    .get(name)
                    .ok_or_else(|| format!("binius64: missing private value for `{name}`"))?,
                Op::Add(x, y) => vals[*x as usize].wrapping_add(vals[*y as usize]),
                Op::Sub(x, y) => vals[*x as usize].wrapping_sub(vals[*y as usize]),
                Op::Mul(x, y) => vals[*x as usize].wrapping_mul(vals[*y as usize]),
            };
            vals.push(v);
        }
        Ok(vals[ir.output as usize])
    }
}

impl ZKPBackend for Binius64Real {
    fn compile_circuit(&self, expr: &Expr, private_vars: &[String]) -> Result<Circuit, String> {
        let ir = Ir::lower(expr, private_vars)?;
        let (circuit, _wires, _out) = build(&ir);
        let stat = binius_frontend::CircuitStat::collect(&circuit);
        let circuit_data =
            bincode::serialize(&ir).map_err(|e| format!("binius64: cannot serialize IR: {e}"))?;
        Ok(Circuit {
            circuit_data,
            witness_vars: ir.private.clone(),
            public_vars: ir.public.clone(),
            metadata: CircuitMetadata {
                num_gates: ir.ops.len(),
                num_constraints: stat.n_and_constraints + stat.n_mul_constraints,
                backend_name: "binius64-real".to_string(),
                timestamp: 0,
            },
        })
    }

    fn prove(
        &self,
        circuit: &Circuit,
        private_values: &HashMap<String, i64>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String> {
        use binius_prover::Prover;
        use binius_transcript::ProverTranscript;
        use binius_verifier::Verifier;

        let ir = Self::ir_of(circuit)?;
        let (built, wires, _out) = build(&ir);

        // Populate the witness. Every declared input must be supplied — a
        // missing one is an error, not a zero, because silently substituting
        // zero would prove a different statement than the caller asked for.
        let mut filler = built.new_witness_filler();
        for name in &ir.private {
            let v = private_values
                .get(name)
                .ok_or_else(|| format!("binius64: missing private value for `{name}`"))?;
            filler[wires[name]] = binius_core::word::Word(*v as u64);
        }
        for name in &ir.public {
            let v = public_values
                .get(name)
                .ok_or_else(|| format!("binius64: missing public value for `{name}`"))?;
            filler[wires[name]] = binius_core::word::Word(*v as u64);
        }
        let claimed = public_values
            .get(OUTPUT_NAME)
            .ok_or_else(|| format!("binius64: the statement needs a claimed `{OUTPUT_NAME}`"))?;
        filler[wires[OUTPUT_NAME]] = binius_core::word::Word(*claimed as u64);

        built
            .populate_wire_witness(&mut filler)
            .map_err(|e| format!("binius64: witness does not satisfy the circuit: {e}"))?;

        let cs = built.constraint_system().clone();
        let verifier = Verifier::<Sha256HashSuite>::setup(cs, LOG_INV_RATE)
            .map_err(|e| format!("binius64: verifier setup failed: {e}"))?;
        let prover = Prover::<OptimalPackedB128, Sha256HashSuite>::setup(verifier)
            .map_err(|e| format!("binius64: prover setup failed: {e}"))?;

        let mut transcript = ProverTranscript::new(StdChallenger::default());
        prover
            .prove(filler.into_value_vec(), &mut transcript)
            .map_err(|e| format!("binius64: proving failed: {e}"))?;

        Ok(ZKProof {
            proof_data: transcript.finalize(),
            public_values: public_values.clone(),
            backend_name: "binius64-real".to_string(),
            timestamp: 0,
        })
    }

    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String> {
        use binius_transcript::VerifierTranscript;
        use binius_verifier::Verifier;

        let ir = Self::ir_of(circuit)?;
        let (built, wires, _out) = build(&ir);

        // The public values the *verifier* was given are the statement. They
        // must also match what the proof carries, or a prover could prove one
        // statement and present it as another.
        for (k, v) in public_values {
            match proof.public_values.get(k) {
                Some(pv) if pv == v => {}
                Some(pv) => {
                    return Err(format!(
                        "binius64: public value `{k}` is {pv} in the proof but {v} in the statement"
                    ))
                }
                None => return Err(format!("binius64: proof does not carry public value `{k}`")),
            }
        }

        // Reconstruct the public segment of the value vector.
        //
        // The public segment is *constants followed by inout values*, and on the
        // proving side `populate_wire_witness` fills the constants. The verifier
        // never calls that (it has no witness to evaluate), so it must fill them
        // itself — otherwise its public input differs from the prover's by
        // exactly the constants and every honest proof is rejected with an
        // opaque `prodcheck` failure.
        let mut filler = built.new_witness_filler();
        {
            let constants = built.constraint_system().constants.clone();
            let vv = filler.value_vec_mut();
            for (i, c) in constants.iter().enumerate() {
                vv.set(i, *c);
            }
        }
        for name in &ir.public {
            let v = public_values
                .get(name)
                .ok_or_else(|| format!("binius64: missing public value for `{name}`"))?;
            filler[wires[name]] = binius_core::word::Word(*v as u64);
        }
        let claimed = public_values
            .get(OUTPUT_NAME)
            .ok_or_else(|| format!("binius64: the statement needs a claimed `{OUTPUT_NAME}`"))?;
        filler[wires[OUTPUT_NAME]] = binius_core::word::Word(*claimed as u64);
        let value_vec = filler.into_value_vec();

        let cs = built.constraint_system().clone();
        let verifier = Verifier::<Sha256HashSuite>::setup(cs, LOG_INV_RATE)
            .map_err(|e| format!("binius64: verifier setup failed: {e}"))?;

        let mut transcript = VerifierTranscript::new(
            StdChallenger::default(),
            proof.proof_data.clone(),
        );
        match verifier.verify(value_vec.public(), &mut transcript) {
            Ok(()) => transcript
                .finalize()
                .map(|_| true)
                .map_err(|e| format!("binius64: trailing transcript data: {e}")),
            Err(e) => Err(format!("binius64: proof rejected: {e}")),
        }
    }

    fn name(&self) -> &str {
        "binius64-real"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }
}

#[cfg(test)]
mod tests {
    use binius_core::word::Word;
    use binius_frontend::CircuitBuilder;

    /// Regression test for the two mistakes that made the first working
    /// version silently wrong, both of which produce *plausible* failures
    /// rather than obvious ones.
    ///
    /// 1. `smul`/`imul` return `(hi, lo)` — **high word first**. Destructuring
    ///    as `(lo, hi)` yields the high word, which is 0 for every small
    ///    product, so the circuit looks exactly as if the witness were never
    ///    populated.
    /// 2. Baru's values are signed `i64`, so `smul` is the correct multiply.
    #[test]
    fn the_multiply_returns_high_word_first() {
        let b = CircuitBuilder::new();
        let x = b.add_witness();
        let y = b.add_inout();
        let (_hi, lo) = b.smul(x, y);
        let out = b.add_inout();
        b.assert_eq("out", lo, out);
        let c = b.build();

        let mut w = c.new_witness_filler();
        w[x] = Word(6);
        w[y] = Word(7);
        w[out] = Word(42);
        c.populate_wire_witness(&mut w).expect("6*7 == 42");
    }
}
