//! FHE circuit emitter: walks an `FheCircuit`'s `FlatOp` SSA block and
//! dispatches each gate to an `FheBackend`.
//!
//! This is the FHE analogue of `binius64_backend::serialize_constraints`.
//! Key differences from ZKP emission:
//!   - There are no witness wires (FHE is fully deterministic from the
//!     server's perspective).
//!   - Public inputs arrive as plaintext; encrypted inputs arrive as
//!     ciphertexts.  The emitter keeps both in a unified `WireEnv`.
//!   - Division maps to `backend.divmod()` (PBS-based, expensive).

use std::collections::HashMap;

use crate::fhe::backend::{FheBackend, FheCircuit};
use crate::zkp::circuit_compiler::FlatOp;

// ─── Wire environment ─────────────────────────────────────────────────────────

/// Every SSA wire holds either a plaintext constant or a ciphertext.
enum WireVal<Ct> {
    Plain(i64),
    Cipher(Ct),
}

/// Evaluate `FheCircuit` homomorphically.
///
/// * `circuit`           — compiled circuit from `compile_fhe`
/// * `public_inputs`     — plaintext values for `circuit.public_inputs` wires
/// * `encrypted_inputs`  — ciphertexts for `circuit.encrypted_inputs` wires
/// * `backend`           — FHE backend to dispatch gates to
///
/// Returns the ciphertext on the output wire.
/// Panics if a required input wire is missing or an unsupported op is encountered.
pub fn evaluate_fhe<B: FheBackend>(
    circuit: &FheCircuit,
    public_inputs: &HashMap<String, i64>,
    encrypted_inputs: &HashMap<String, B::Ct>,
    backend: &B,
) -> B::Ct {
    let mut env: HashMap<String, WireVal<B::Ct>> = HashMap::new();

    // Seed env with public (plaintext) inputs
    for name in &circuit.public_inputs {
        let val = *public_inputs
            .get(name)
            .unwrap_or_else(|| panic!("fhe-eval: missing public input '{}'", name));
        env.insert(name.clone(), WireVal::Plain(val));
    }

    // Seed env with encrypted inputs (ciphertexts)
    for name in &circuit.encrypted_inputs {
        let ct = encrypted_inputs
            .get(name)
            .unwrap_or_else(|| panic!("fhe-eval: missing encrypted input '{}'", name))
            .clone();
        env.insert(name.clone(), WireVal::Cipher(ct));
    }

    // Helper: get or promote a wire to a ciphertext.
    // Plain constants are encrypted on-the-fly using `add_plain(0, v)`.
    macro_rules! ct {
        ($wire:expr) => {
            match env
                .get($wire)
                .unwrap_or_else(|| panic!("fhe-eval: unbound wire '{}'", $wire))
            {
                WireVal::Cipher(c) => c.clone(),
                WireVal::Plain(v) => backend.add_plain(&backend.encrypt_zero(), *v),
            }
        };
    }

    for op in &circuit.flat_ops {
        match op {
            FlatOp::Const { dst, value } => {
                env.insert(dst.clone(), WireVal::Plain(*value));
            }

            FlatOp::IAdd32 { dst, lhs, rhs } => {
                let result = match (env.get(lhs), env.get(rhs)) {
                    (Some(WireVal::Plain(a)), Some(WireVal::Plain(b))) => {
                        WireVal::Plain(a.wrapping_add(*b))
                    }
                    (Some(WireVal::Plain(a)), Some(WireVal::Cipher(_))) => {
                        WireVal::Cipher(backend.add_plain(&ct!(rhs), *a))
                    }
                    (Some(WireVal::Cipher(_)), Some(WireVal::Plain(b))) => {
                        WireVal::Cipher(backend.add_plain(&ct!(lhs), *b))
                    }
                    _ => WireVal::Cipher(backend.add(&ct!(lhs), &ct!(rhs))),
                };
                env.insert(dst.clone(), result);
            }

            FlatOp::ISub32 { dst, lhs, rhs } => {
                let result = match (env.get(lhs), env.get(rhs)) {
                    (Some(WireVal::Plain(a)), Some(WireVal::Plain(b))) => {
                        WireVal::Plain(a.wrapping_sub(*b))
                    }
                    (Some(WireVal::Cipher(_)), Some(WireVal::Plain(b))) => {
                        WireVal::Cipher(backend.add_plain(&ct!(lhs), -b))
                    }
                    _ => WireVal::Cipher(backend.sub(&ct!(lhs), &ct!(rhs))),
                };
                env.insert(dst.clone(), result);
            }

            FlatOp::IMul { dst, lhs, rhs } => {
                let result = match (env.get(lhs), env.get(rhs)) {
                    (Some(WireVal::Plain(a)), Some(WireVal::Plain(b))) => {
                        WireVal::Plain(a.wrapping_mul(*b))
                    }
                    (Some(WireVal::Plain(a)), Some(WireVal::Cipher(_))) => {
                        WireVal::Cipher(backend.mul_plain(&ct!(rhs), *a))
                    }
                    (Some(WireVal::Cipher(_)), Some(WireVal::Plain(b))) => {
                        WireVal::Cipher(backend.mul_plain(&ct!(lhs), *b))
                    }
                    _ => WireVal::Cipher(backend.mul(&ct!(lhs), &ct!(rhs))),
                };
                env.insert(dst.clone(), result);
            }

            FlatOp::IDiv {
                dst_q,
                dst_r,
                lhs,
                rhs,
            } => {
                let (q, r) = backend.divmod(&ct!(lhs), &ct!(rhs));
                env.insert(dst_q.clone(), WireVal::Cipher(q));
                env.insert(dst_r.clone(), WireVal::Cipher(r));
            }

            FlatOp::ICmpEq { dst, lhs, rhs } => {
                let result = backend.cmpeq(&ct!(lhs), &ct!(rhs));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::ICmpSlt { dst, lhs, rhs } => {
                let result = backend.cmplt(&ct!(lhs), &ct!(rhs));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::ICmpUlt { dst, lhs, rhs } => {
                // Unsigned: reinterpret as signed for mock; real TFHE uses separate API
                let result = backend.cmplt(&ct!(lhs), &ct!(rhs));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::BAnd { dst, lhs, rhs } => {
                let result = backend.band(&ct!(lhs), &ct!(rhs));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::BOr { dst, lhs, rhs } => {
                let result = backend.bor(&ct!(lhs), &ct!(rhs));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::BXor { dst, lhs, rhs } => {
                let result = backend.bxor(&ct!(lhs), &ct!(rhs));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::BNot { dst, src } => {
                let result = backend.bnot(&ct!(src));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::Shl { dst, src, amount } => {
                let result = backend.shl(&ct!(src), *amount);
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::Shr { dst, src, amount } => {
                let result = backend.shr(&ct!(src), *amount);
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::Mux {
                dst,
                cond,
                t_val,
                e_val,
            } => {
                let result = backend.mux(&ct!(cond), &ct!(t_val), &ct!(e_val));
                env.insert(dst.clone(), WireVal::Cipher(result));
            }

            FlatOp::PopCount { dst, src } => {
                // Approximate: encrypt the popcount of the plaintext/ciphertext.
                // For a real backend this would be a sequence of bit-extract + add gates.
                let result = backend.shr(&ct!(src), 0); // identity placeholder
                env.insert(dst.clone(), WireVal::Cipher(result));
            }
        }
    }

    // Return the output wire value as a ciphertext
    match env
        .remove(&circuit.output_wire)
        .unwrap_or_else(|| panic!("fhe-eval: output wire '{}' not found", circuit.output_wire))
    {
        WireVal::Cipher(ct) => ct,
        WireVal::Plain(v) => backend.add_plain(&backend.encrypt_zero(), v),
    }
}

/// Helper trait extension — backends need a "zero ciphertext" for plain→cipher promotion.
/// Added as a separate trait to avoid breaking the core `FheBackend` interface.
pub trait FheBackendExt: FheBackend {
    /// Return an encryption of zero (used to lift plaintext constants into the
    /// ciphertext domain when mixing plain and cipher wires in the same gate).
    fn encrypt_zero(&self) -> Self::Ct;
}

// Blanket implementation: keygen + encrypt(0).
impl<B: FheBackend> FheBackendExt for B {
    fn encrypt_zero(&self) -> Self::Ct {
        let (ck, _) = self.keygen();
        self.encrypt(&ck, 0)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::mock_backend::MockFheBackend;
    use crate::zkp::circuit_compiler::CircuitCompiler;

    fn compile_fhe_circuit(src: &str) -> FheCircuit {
        use crate::interpreter::parse;
        let expr = parse(src).unwrap();
        let mut cc = CircuitCompiler::new();
        cc.compile_fhe(&expr).unwrap()
    }

    #[test]
    fn test_fhe_eval_add_public() {
        // (+ x y) with both public
        let circuit = compile_fhe_circuit("(+ x y)");
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let mut pub_in = HashMap::new();
        pub_in.insert("x".to_string(), 10i64);
        pub_in.insert("y".to_string(), 5i64);
        let enc_in: HashMap<String, u64> = HashMap::new();
        let result_ct = evaluate_fhe(&circuit, &pub_in, &enc_in, &be);
        assert_eq!(be.decrypt(&ck, &result_ct), 15);
    }

    #[test]
    fn test_fhe_eval_add_encrypted() {
        // (+ x y) with x encrypted, y public
        let mut circuit = compile_fhe_circuit("(+ x y)");
        // Manually mark x as encrypted (compile_fhe default: all inputs public)
        circuit.encrypted_inputs = vec!["x".to_string()];
        circuit.public_inputs = vec!["y".to_string()];

        let be = MockFheBackend;
        let (ck, _) = be.keygen();

        let mut pub_in = HashMap::new();
        pub_in.insert("y".to_string(), 5i64);
        let mut enc_in = HashMap::new();
        enc_in.insert("x".to_string(), be.encrypt(&ck, 10));

        let result_ct = evaluate_fhe(&circuit, &pub_in, &enc_in, &be);
        assert_eq!(be.decrypt(&ck, &result_ct), 15);
    }

    #[test]
    fn test_fhe_eval_if_mux() {
        // (if (< x 5) 1 0) with x encrypted
        let src = "(if (< x 5) 1 0)";
        let mut circuit = compile_fhe_circuit(src);
        circuit.encrypted_inputs = vec!["x".to_string()];
        circuit.public_inputs = vec![];

        let be = MockFheBackend;
        let (ck, _) = be.keygen();

        // x = 3 → 1
        let mut enc_in = HashMap::new();
        enc_in.insert("x".to_string(), be.encrypt(&ck, 3));
        let r = evaluate_fhe(&circuit, &HashMap::new(), &enc_in, &be);
        assert_eq!(be.decrypt(&ck, &r), 1);

        // x = 7 → 0
        enc_in.insert("x".to_string(), be.encrypt(&ck, 7));
        let r = evaluate_fhe(&circuit, &HashMap::new(), &enc_in, &be);
        assert_eq!(be.decrypt(&ck, &r), 0);
    }
}
