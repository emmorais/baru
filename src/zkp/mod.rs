// ZKP Backend abstraction for Baru
// This module provides the trait-based interface for ZKP backends (e.g., Binius64, Plonky3)

use crate::interpreter::ast::Expr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod binius64_backend;
pub mod circuit_compiler;
pub mod serialization;

/// Public input/output values for a circuit
/// Maps variable names to field element values
pub type PublicValues = HashMap<String, i64>;

/// Represents a compiled circuit ready for proving/verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    /// Backend-specific serialized circuit data
    pub circuit_data: Vec<u8>,
    /// Which variables are witnesses (private)
    pub witness_vars: Vec<String>,
    /// Which variables are public
    pub public_vars: Vec<String>,
    /// Metadata about circuit size, gates, etc.
    pub metadata: CircuitMetadata,
}

/// Metadata about a circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetadata {
    pub num_gates: usize,
    pub num_constraints: usize,
    pub backend_name: String,
    pub timestamp: u64,
}

/// Represents a zero-knowledge proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKProof {
    /// Backend-specific proof data
    pub proof_data: Vec<u8>,
    /// Public input values used in proof
    pub public_values: PublicValues,
    /// Backend that generated this proof
    pub backend_name: String,
    /// Timestamp of proof generation
    pub timestamp: u64,
}

/// ZKP Backend trait - abstraction over different ZKP systems
pub trait ZKPBackend: Send + Sync {
    /// Compile an expression into a circuit
    ///
    /// # Arguments
    /// * `expr` - The expression to compile
    /// * `private_vars` - Variable names that should be treated as witnesses
    ///
    /// # Returns
    /// A compiled circuit ready for proving, or an error message
    fn compile_circuit(&self, expr: &Expr, private_vars: &[String]) -> Result<Circuit, String>;

    /// Generate a proof for a circuit given witness and public values
    ///
    /// # Arguments
    /// * `circuit` - The compiled circuit
    /// * `private_values` - Mapping of witness variable names to their values
    /// * `public_values` - Mapping of public variable names to their values
    ///
    /// # Returns
    /// A ZKProof if successful, or an error message
    fn prove(
        &self,
        circuit: &Circuit,
        private_values: &HashMap<String, i64>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String>;

    /// Verify a proof against a circuit and public values
    ///
    /// # Arguments
    /// * `circuit` - The compiled circuit
    /// * `proof` - The proof to verify
    /// * `public_values` - The public input values
    ///
    /// # Returns
    /// true if proof is valid, false otherwise (or error on verification failure)
    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String>;

    /// Get a human-readable name for this backend
    fn name(&self) -> &str;

    /// Get version information
    fn version(&self) -> &str;
}

/// Mock backend for testing (always succeeds)
pub struct MockZKPBackend;

impl ZKPBackend for MockZKPBackend {
    fn compile_circuit(&self, _expr: &Expr, private_vars: &[String]) -> Result<Circuit, String> {
        Ok(Circuit {
            circuit_data: vec![],
            witness_vars: private_vars.to_vec(),
            public_vars: vec![],
            metadata: CircuitMetadata {
                num_gates: 0,
                num_constraints: 0,
                backend_name: "mock".to_string(),
                timestamp: 0,
            },
        })
    }

    fn prove(
        &self,
        _circuit: &Circuit,
        _private_values: &HashMap<String, i64>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String> {
        Ok(ZKProof {
            proof_data: vec![],
            public_values: public_values.clone(),
            backend_name: "mock".to_string(),
            timestamp: 0,
        })
    }

    fn verify(
        &self,
        _circuit: &Circuit,
        _proof: &ZKProof,
        _public_values: &PublicValues,
    ) -> Result<bool, String> {
        // Mock backend always succeeds
        Ok(true)
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend_compile() {
        let backend = MockZKPBackend;
        let expr = Expr::Num(42);
        let private_vars = vec!["x".to_string()];

        let result = backend.compile_circuit(&expr, &private_vars);
        assert!(result.is_ok());

        let circuit = result.unwrap();
        assert_eq!(circuit.witness_vars, private_vars);
    }

    #[test]
    fn test_mock_backend_prove() {
        let backend = MockZKPBackend;
        let expr = Expr::Num(42);
        let circuit = backend.compile_circuit(&expr, &["x".to_string()]).unwrap();

        let private_values = HashMap::new();
        let public_values = HashMap::new();

        let result = backend.prove(&circuit, &private_values, &public_values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_backend_verify() {
        let backend = MockZKPBackend;
        let expr = Expr::Num(42);
        let circuit = backend.compile_circuit(&expr, &["x".to_string()]).unwrap();

        let proof = backend
            .prove(&circuit, &HashMap::new(), &HashMap::new())
            .unwrap();

        let result = backend.verify(&circuit, &proof, &HashMap::new());
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
}
