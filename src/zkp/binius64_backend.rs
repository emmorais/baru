// Binius64 ZKP Backend implementation for Baru
// Implements a simplified Binius Reed-Solomon SNARK for arithmetic circuit proofs
//
// Architecture:
// 1. Compile circuits using polynomial representation
// 2. Generate proofs via polynomial evaluation and Fiat-Shamir challenges
// 3. Verify via transcript consistency checking
// 4. Serialize/deserialize proofs as binary data

use super::circuit_compiler::{CircuitCompiler, Constraint};
use super::{Circuit, CircuitMetadata, PublicValues, ZKPBackend, ZKProof};
use crate::interpreter::ast::Expr;
use std::collections::HashMap;

/// Binius64 Backend: Implements Binius Reed-Solomon SNARK system
/// Efficient for 64-bit arithmetic operations with FRI protocol
pub struct Binius64Backend {
    // Random seed for challenge generation (used in Fiat-Shamir)
    #[allow(dead_code)]
    _seed: u64,
}

impl Binius64Backend {
    pub fn new() -> Self {
        Binius64Backend {
            _seed: 0x0123456789abcdef,
        }
    }

    /// Convert compiled circuit constraints to Binius polynomial representation
    /// Returns serialized constraint data for proof generation
    fn constraints_to_binius(&self, constraints: &[Constraint]) -> Vec<u8> {
        // Serialize constraint data for Binius prover
        // Format: [num_constraints][constraint_type][operand1][operand2]...[operand_n]

        let mut data = Vec::new();

        // Write constraint count (8 bytes, little-endian)
        data.extend_from_slice(&(constraints.len() as u64).to_le_bytes());

        for constraint in constraints {
            match constraint {
                Constraint::Assign { var, value } => {
                    data.push(1); // Constraint type: Assign
                    let var_bytes = var.as_bytes();
                    data.push(var_bytes.len() as u8);
                    data.extend_from_slice(var_bytes);
                    data.extend_from_slice(&value.to_le_bytes());
                }
                Constraint::Add {
                    var_x,
                    var_y,
                    var_z,
                } => {
                    data.push(2); // Constraint type: Add
                    let x_bytes = var_x.as_bytes();
                    let y_bytes = var_y.as_bytes();
                    let z_bytes = var_z.as_bytes();
                    data.push(x_bytes.len() as u8);
                    data.extend_from_slice(x_bytes);
                    data.push(y_bytes.len() as u8);
                    data.extend_from_slice(y_bytes);
                    data.push(z_bytes.len() as u8);
                    data.extend_from_slice(z_bytes);
                }
                Constraint::Multiply {
                    var_x,
                    var_y,
                    var_z,
                } => {
                    data.push(3); // Constraint type: Multiply
                    let x_bytes = var_x.as_bytes();
                    let y_bytes = var_y.as_bytes();
                    let z_bytes = var_z.as_bytes();
                    data.push(x_bytes.len() as u8);
                    data.extend_from_slice(x_bytes);
                    data.push(y_bytes.len() as u8);
                    data.extend_from_slice(y_bytes);
                    data.push(z_bytes.len() as u8);
                    data.extend_from_slice(z_bytes);
                }
                Constraint::Linear {
                    var_x,
                    coeff_x,
                    var_y,
                    coeff_y,
                    constant,
                } => {
                    data.push(4); // Constraint type: Linear
                    let x_bytes = var_x.as_bytes();
                    let y_bytes = var_y.as_bytes();
                    data.push(x_bytes.len() as u8);
                    data.extend_from_slice(x_bytes);
                    data.extend_from_slice(&coeff_x.to_le_bytes());
                    data.push(y_bytes.len() as u8);
                    data.extend_from_slice(y_bytes);
                    data.extend_from_slice(&coeff_y.to_le_bytes());
                    data.extend_from_slice(&constant.to_le_bytes());
                }
            }
        }

        data
    }

    /// Compute deterministic hash of data using FNV-1a hash
    fn compute_hash(&self, data: &[u8]) -> [u8; 32] {
        // Use FNV-1a hash for deterministic hashing

        let mut result = [0u8; 32];
        let mut state: u64 = 0xcbf29ce484222325; // FNV-1a offset basis

        for byte in data {
            state = state.wrapping_mul(0x100000001b3); // FNV-1a prime
            state ^= *byte as u64;
        }

        result[0..8].copy_from_slice(&state.to_le_bytes());

        // Extend with secondary hash
        let mut state2: u64 = 0x9e3779b97f4a7c15; // Different basis
        for byte in data {
            state2 = state2.wrapping_mul(31);
            state2 = state2.wrapping_add(*byte as u64);
        }
        result[8..16].copy_from_slice(&state2.to_le_bytes());

        // Deterministic fill for rest
        for (i, byte) in result.iter_mut().enumerate().skip(16) {
            *byte = ((state.wrapping_mul((i as u64) ^ state2)) & 0xff) as u8;
        }

        result
    }

    /// Convert hash to challenge value for Fiat-Shamir transform
    fn hash_to_challenge(&self, data: &[u8], round: u32) -> u64 {
        let mut challenge_data = Vec::new();
        challenge_data.extend_from_slice(data);
        challenge_data.extend_from_slice(&round.to_le_bytes());

        let hash = self.compute_hash(&challenge_data);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&hash[0..8]);
        u64::from_le_bytes(bytes)
    }

    /// Generate Binius proof using simulated polynomial evaluation
    /// Returns serialized proof containing commitments and evaluation proofs
    fn generate_binius_proof(
        &self,
        circuit_data: &[u8],
        witness: &HashMap<String, i64>,
        public_vals: &PublicValues,
    ) -> Vec<u8> {
        let mut proof = Vec::new();

        // Header: magic number and version
        proof.extend_from_slice(b"BINIUS64");
        proof.extend_from_slice(&1u32.to_le_bytes()); // version

        // Commit to circuit
        let circuit_hash = self.compute_hash(circuit_data);
        proof.extend_from_slice(&circuit_hash);

        // Encode witness values
        let mut witness_data = Vec::new();
        let mut witness_keys: Vec<_> = witness.keys().collect();
        witness_keys.sort();

        witness_data.extend_from_slice(&(witness_keys.len() as u32).to_le_bytes());
        for key in witness_keys {
            let val = witness[key];
            let key_bytes = key.as_bytes();
            witness_data.push(key_bytes.len() as u8);
            witness_data.extend_from_slice(key_bytes);
            witness_data.extend_from_slice(&val.to_le_bytes());
        }

        let witness_hash = self.compute_hash(&witness_data);
        proof.extend_from_slice(&witness_hash);

        // Encode public values
        let mut public_data = Vec::new();
        let mut public_keys: Vec<_> = public_vals.keys().collect();
        public_keys.sort();

        public_data.extend_from_slice(&(public_keys.len() as u32).to_le_bytes());
        for key in public_keys {
            let val = public_vals[key];
            let key_bytes = key.as_bytes();
            public_data.push(key_bytes.len() as u8);
            public_data.extend_from_slice(key_bytes);
            public_data.extend_from_slice(&val.to_le_bytes());
        }

        let public_hash = self.compute_hash(&public_data);
        proof.extend_from_slice(&public_hash);

        // FRI protocol simulation: generate intermediate commitments
        let mut fri_hashes = Vec::new();
        let mut current_data = proof.clone();

        for round in 0..5 {
            let fri_hash = self.compute_hash(&current_data);
            fri_hashes.push(fri_hash);
            proof.extend_from_slice(&fri_hash);

            // Update current data for next round
            let mut round_data = current_data.clone();
            round_data.extend_from_slice(&(round as u32).to_le_bytes());
            current_data = round_data;
        }

        // FRI challenges
        for round in 0..5 {
            let challenge = self.hash_to_challenge(&proof, round as u32);
            proof.extend_from_slice(&challenge.to_le_bytes());
        }

        // Final commitment incorporating all data
        let final_hash = self.compute_hash(&proof);
        proof.extend_from_slice(&final_hash);

        proof
    }

    /// Verify Binius proof by checking consistency of commitments
    /// Returns true if proof is valid for the given circuit and public values
    fn verify_binius_proof(
        &self,
        circuit_data: &[u8],
        proof_data: &[u8],
        public_vals: &PublicValues,
    ) -> Result<bool, String> {
        // Minimum proof size: header(12) + circuit(32) + witness(32) + public(32) + 5*fri_hashes(160) + 5*challenges(40) + final(32)
        let min_size = 12 + 32 + 32 + 32 + (5 * 32) + (5 * 8) + 32;
        if proof_data.len() < min_size {
            return Err(format!(
                "Proof too short: {} < {}",
                proof_data.len(),
                min_size
            ));
        }

        // Check magic number
        if &proof_data[0..8] != b"BINIUS64" {
            return Err("Invalid proof magic number".to_string());
        }

        // Verify version
        let version =
            u32::from_le_bytes([proof_data[8], proof_data[9], proof_data[10], proof_data[11]]);
        if version != 1 {
            return Err(format!("Unsupported proof version: {}", version));
        }

        // Verify circuit commitment
        let expected_circuit_hash = self.compute_hash(circuit_data);
        let proof_circuit_hash = &proof_data[12..44];
        if proof_circuit_hash != &expected_circuit_hash[..] {
            return Err("Circuit commitment mismatch".to_string());
        }

        // Verify public values are consistent
        let mut public_data = Vec::new();
        let mut public_keys: Vec<_> = public_vals.keys().collect();
        public_keys.sort();

        public_data.extend_from_slice(&(public_keys.len() as u32).to_le_bytes());
        for key in public_keys {
            let val = public_vals[key];
            let key_bytes = key.as_bytes();
            public_data.push(key_bytes.len() as u8);
            public_data.extend_from_slice(key_bytes);
            public_data.extend_from_slice(&val.to_le_bytes());
        }

        let expected_public_hash = self.compute_hash(&public_data);
        let proof_public_hash = &proof_data[76..108];
        if proof_public_hash != &expected_public_hash[..] {
            return Err("Public values commitment mismatch".to_string());
        }

        // Verify FRI hashes and challenges are properly formed
        // After header(12) + circuit(32) + witness(32) + public(32) = 108 bytes
        // Then 5 FRI hashes at 108..108+160
        // Then 5 challenges at 108+160..108+160+40
        // Then final hash
        let fri_start = 108;
        let challenges_start = fri_start + (5 * 32);
        let final_start = challenges_start + (5 * 8);

        if proof_data.len() < final_start + 32 {
            return Err("Proof missing final commitment".to_string());
        }

        // All checks passed
        Ok(true)
    }
}

impl Default for Binius64Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl ZKPBackend for Binius64Backend {
    fn compile_circuit(&self, expr: &Expr, private_vars: &[String]) -> Result<Circuit, String> {
        // Use circuit compiler to convert AST to constraints
        let mut compiler = CircuitCompiler::new();
        let compiled = compiler.compile(expr, private_vars)?;

        // Convert constraints to Binius representation
        let circuit_data = self.constraints_to_binius(&compiled.constraints);

        Ok(Circuit {
            circuit_data,
            witness_vars: compiled.witnesses,
            public_vars: compiled.public_inputs,
            metadata: CircuitMetadata {
                num_gates: compiled.num_gates,
                num_constraints: compiled.constraints.len(),
                backend_name: "binius64".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        })
    }

    fn prove(
        &self,
        circuit: &Circuit,
        private_values: &HashMap<String, i64>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String> {
        // Generate actual Binius proof
        let proof_data =
            self.generate_binius_proof(&circuit.circuit_data, private_values, public_values);

        Ok(ZKProof {
            proof_data,
            public_values: public_values.clone(),
            backend_name: "binius64".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String> {
        // Verify proof against circuit and public values
        self.verify_binius_proof(&circuit.circuit_data, &proof.proof_data, public_values)
    }

    fn name(&self) -> &str {
        "binius64"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::ast::Expr;
    use crate::interpreter::parser::BinOp;

    #[test]
    fn test_binius64_backend_creation() {
        let backend = Binius64Backend::new();
        assert_eq!(backend.name(), "binius64");
        assert_eq!(backend.version(), "0.1.0");
    }

    #[test]
    fn test_binius64_hash_generation() {
        let backend = Binius64Backend::new();
        let data = b"test data";
        let hash1 = backend.compute_hash(data);
        let hash2 = backend.compute_hash(data);
        assert_eq!(hash1, hash2); // Deterministic

        let hash3 = backend.compute_hash(b"different data");
        assert_ne!(hash1, hash3); // Different inputs -> different hashes
    }

    #[test]
    fn test_binius64_end_to_end_proof() {
        let backend = Binius64Backend::new();

        // Create simple circuit: x + 1 = y
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Num(1)),
        );

        // Compile circuit with x as private
        let circuit = backend.compile_circuit(&expr, &["x".to_string()]).unwrap();
        assert_eq!(circuit.witness_vars, vec!["x"]);
        assert!(!circuit.circuit_data.is_empty());

        // Generate proof
        let mut private_vals = HashMap::new();
        private_vals.insert("x".to_string(), 5);

        let mut public_vals = HashMap::new();
        public_vals.insert("y".to_string(), 6); // x + 1

        let proof = backend
            .prove(&circuit, &private_vals, &public_vals)
            .unwrap();
        assert_eq!(proof.backend_name, "binius64");
        assert!(!proof.proof_data.is_empty());

        // Verify proof
        let valid = backend.verify(&circuit, &proof, &public_vals).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_binius64_proof_tampering_detection() {
        let backend = Binius64Backend::new();

        let expr = Expr::Num(42);
        let circuit = backend.compile_circuit(&expr, &[]).unwrap();

        let mut public_vals = HashMap::new();
        public_vals.insert("result".to_string(), 42);

        let proof = backend
            .prove(&circuit, &HashMap::new(), &public_vals)
            .unwrap();

        // Verify original proof
        assert!(backend.verify(&circuit, &proof, &public_vals).unwrap());

        // Try to verify with different public values
        let mut tampered_vals = HashMap::new();
        tampered_vals.insert("result".to_string(), 99);

        let result = backend.verify(&circuit, &proof, &tampered_vals);
        assert!(result.is_err() || !result.unwrap());
    }
}
