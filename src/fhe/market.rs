//! FHE job market grounded on real homomorphic evaluation.
//!
//! This is the cryptographic core of the `OptimisticReExec` work-receipt model:
//!
//! * A **requester** posts an [`FheJob`] — a compiled circuit plus its public
//!   inputs and *encrypted* inputs (ciphertext bytes).  Only the requester holds
//!   the client key, so the data stays private.
//! * A **worker** runs [`evaluate_job`], producing the result ciphertext and a
//!   succinct [`WorkReceipt`] (a commitment to the output).  A newcomer can take
//!   a job off the queue and earn stake immediately — the job *is* the on-ramp.
//! * **Anyone** can check the worker via [`verify_receipt`], which re-executes
//!   the circuit and compares the output commitment.  Because homomorphic
//!   evaluation is a *deterministic* function of `(circuit, inputs, keys)` — the
//!   verifier works entirely on **ciphertexts and never decrypts** — a wrong
//!   result yields a different commitment and is provably fraudulent.
//!
//! Determinism is the load-bearing property: the mock backend is bit-exact, and
//! integer FHE evaluation (e.g. tfhe-rs) injects no fresh randomness during
//! evaluation, so re-execution reproduces the output ciphertext exactly.
//!
//! The escrow / bond / challenge-window state machine (see the design notes)
//! layers *on top* of this core — this module provides the objective
//! validity check those economics adjudicate.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::fhe::backend::{FheBackend, FheCiphertextHandle, FheCircuit};
use crate::fhe::circuit_emitter::evaluate_fhe;

/// A 32-byte content commitment.
pub type Hash = [u8; 32];

fn hash_bytes(bytes: &[u8]) -> Hash {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// A homomorphic-evaluation job posted by a requester.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FheJob {
    /// The circuit to evaluate (its `encrypted_inputs` / `public_inputs` name
    /// the wires supplied below).
    pub circuit: FheCircuit,
    /// Plaintext inputs, by wire name.
    pub public_inputs: HashMap<String, i64>,
    /// Encrypted inputs, by wire name — serialized ciphertext bytes.
    pub encrypted_inputs: HashMap<String, Vec<u8>>,
    /// Backend the job must be evaluated with (rejects mismatched workers).
    pub backend_id: String,
}

impl FheJob {
    /// Canonical content commitment to the request — what a worker must
    /// evaluate.  Deterministic regardless of map iteration order.
    pub fn request_hash(&self) -> Hash {
        let mut h = Sha256::new();
        // The circuit serializes deterministically (Vec-based).
        h.update(bincode::serialize(&self.circuit).expect("serialize circuit"));
        h.update(self.backend_id.as_bytes());

        let mut pubs: Vec<(&String, &i64)> = self.public_inputs.iter().collect();
        pubs.sort_by(|a, b| a.0.cmp(b.0));
        h.update((pubs.len() as u64).to_le_bytes());
        for (k, v) in pubs {
            h.update((k.len() as u64).to_le_bytes());
            h.update(k.as_bytes());
            h.update(v.to_le_bytes());
        }

        let mut encs: Vec<(&String, &Vec<u8>)> = self.encrypted_inputs.iter().collect();
        encs.sort_by(|a, b| a.0.cmp(b.0));
        h.update((encs.len() as u64).to_le_bytes());
        for (k, bytes) in encs {
            h.update((k.len() as u64).to_le_bytes());
            h.update(k.as_bytes());
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(bytes);
        }
        h.finalize().into()
    }
}

/// A worker's succinct commitment to a completed job.  Everything needed to
/// adjudicate correctness is here; the (large) output ciphertext is delivered
/// to the requester separately (see the returned [`FheCiphertextHandle`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkReceipt {
    /// The request this receipt answers.
    pub request: Hash,
    /// Commitment to the output ciphertext: `SHA-256(output bytes)`.
    pub output_root: Hash,
}

/// Result of adjudicating a receipt by re-execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The receipt's output commitment matches an honest re-execution.
    Valid,
    /// The receipt does not answer this job.
    WrongRequest,
    /// Re-execution produced a different output than claimed — provable fraud.
    Fraudulent {
        expected_root: Hash,
        claimed_root: Hash,
    },
}

/// Errors a worker can hit before producing any result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobError {
    /// The worker's backend does not match the job's required backend.
    BackendMismatch { job: String, worker: String },
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobError::BackendMismatch { job, worker } => {
                write!(f, "backend mismatch: job requires '{job}', worker is '{worker}'")
            }
        }
    }
}

impl std::error::Error for JobError {}

/// Worker side: evaluate `job` homomorphically, returning the result ciphertext
/// handle (delivered to the requester) and a succinct [`WorkReceipt`].
pub fn evaluate_job<B: FheBackend>(
    job: &FheJob,
    backend: &B,
) -> Result<(FheCiphertextHandle, WorkReceipt), JobError> {
    if job.backend_id != backend.backend_id() {
        return Err(JobError::BackendMismatch {
            job: job.backend_id.clone(),
            worker: backend.backend_id().to_string(),
        });
    }

    let encrypted: HashMap<String, B::Ct> = job
        .encrypted_inputs
        .iter()
        .map(|(name, bytes)| (name.clone(), backend.deserialize_ct(bytes)))
        .collect();

    let out = evaluate_fhe(&job.circuit, &job.public_inputs, &encrypted, backend);
    let bytes = backend.serialize_ct(&out);
    let output_root = hash_bytes(&bytes);

    let handle = FheCiphertextHandle {
        backend_id: backend.backend_id().to_string(),
        bytes,
        fixed: job.circuit.output_fixed,
        nbits: None,
    };
    let receipt = WorkReceipt {
        request: job.request_hash(),
        output_root,
    };
    Ok((handle, receipt))
}

/// Verifier / challenger side: re-execute the job deterministically and compare
/// the output commitment.  Never decrypts — works purely on ciphertexts.
pub fn verify_receipt<B: FheBackend>(job: &FheJob, receipt: &WorkReceipt, backend: &B) -> Verdict {
    if receipt.request != job.request_hash() {
        return Verdict::WrongRequest;
    }
    let recomputed = match evaluate_job(job, backend) {
        Ok((_, r)) => r,
        Err(_) => return Verdict::WrongRequest,
    };
    if recomputed.output_root == receipt.output_root {
        Verdict::Valid
    } else {
        Verdict::Fraudulent {
            expected_root: recomputed.output_root,
            claimed_root: receipt.output_root,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::mock_backend::MockFheBackend;
    use crate::interpreter::parse;
    use crate::zkp::circuit_compiler::CircuitCompiler;

    /// Compile `src` and mark the given wires as encrypted inputs (the rest
    /// stay public).
    fn compile(src: &str, encrypted: &[&str]) -> FheCircuit {
        let expr = parse(src).unwrap();
        let mut cc = CircuitCompiler::new();
        let mut circuit = cc.compile_fhe(&expr).unwrap();
        let enc: Vec<String> = encrypted.iter().map(|s| s.to_string()).collect();
        circuit.public_inputs.retain(|v| !enc.contains(v));
        circuit.encrypted_inputs = enc;
        circuit
    }

    /// Build a job over the mock backend with the given encrypted-input values.
    fn make_job(
        circuit: FheCircuit,
        be: &MockFheBackend,
        ck: &[u8],
        public: &[(&str, i64)],
        encrypted: &[(&str, u64)],
    ) -> FheJob {
        let public_inputs = public.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let encrypted_inputs = encrypted
            .iter()
            .map(|(k, v)| (k.to_string(), be.serialize_ct(&be.encrypt(ck, *v))))
            .collect();
        FheJob {
            circuit,
            public_inputs,
            encrypted_inputs,
            backend_id: be.backend_id().to_string(),
        }
    }

    #[test]
    fn job_evaluates_verifies_and_decrypts() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job = make_job(
            compile("(+ x y)", &["x", "y"]),
            &be,
            &ck,
            &[],
            &[("x", 10), ("y", 32)],
        );

        // Worker evaluates.
        let (handle, receipt) = evaluate_job(&job, &be).unwrap();

        // Anyone verifies by re-execution — no decryption.
        assert_eq!(verify_receipt(&job, &receipt, &be), Verdict::Valid);

        // Requester decrypts the delivered ciphertext.
        let out_ct = be.deserialize_ct(&handle.bytes);
        assert_eq!(be.decrypt(&ck, &out_ct), 42);
    }

    #[test]
    fn re_execution_is_deterministic() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job = make_job(
            compile("(* a b)", &["a", "b"]),
            &be,
            &ck,
            &[],
            &[("a", 6), ("b", 7)],
        );
        let (_, r1) = evaluate_job(&job, &be).unwrap();
        let (_, r2) = evaluate_job(&job, &be).unwrap();
        // The fraud proof relies on this: identical inputs → identical output root.
        assert_eq!(r1.output_root, r2.output_root);
    }

    #[test]
    fn tampered_output_is_provable_fraud() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job = make_job(
            compile("(+ x y)", &["x", "y"]),
            &be,
            &ck,
            &[],
            &[("x", 10), ("y", 32)],
        );
        let (_, honest) = evaluate_job(&job, &be).unwrap();

        // A lying worker submits a different output commitment.
        let mut lie = honest.clone();
        lie.output_root[0] ^= 0xff;

        match verify_receipt(&job, &lie, &be) {
            Verdict::Fraudulent {
                expected_root,
                claimed_root,
            } => {
                assert_eq!(expected_root, honest.output_root);
                assert_eq!(claimed_root, lie.output_root);
            }
            other => panic!("expected fraud, got {other:?}"),
        }
    }

    #[test]
    fn receipt_for_a_different_job_is_wrong_request() {
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let job_a = make_job(
            compile("(+ x y)", &["x", "y"]),
            &be,
            &ck,
            &[],
            &[("x", 1), ("y", 2)],
        );
        let job_b = make_job(
            compile("(+ x y)", &["x", "y"]),
            &be,
            &ck,
            &[],
            &[("x", 100), ("y", 200)],
        );
        let (_, receipt_b) = evaluate_job(&job_b, &be).unwrap();
        // A receipt answering job B does not satisfy job A.
        assert_eq!(verify_receipt(&job_a, &receipt_b, &be), Verdict::WrongRequest);
    }

    #[test]
    fn request_hash_is_order_independent() {
        // Same logical job built with inputs inserted in different orders must
        // hash identically (canonical, sorted-key commitment).
        let be = MockFheBackend;
        let (ck, _) = be.keygen();
        let circuit = compile("(+ x y)", &["x", "y"]);
        let j1 = make_job(circuit.clone(), &be, &ck, &[], &[("x", 3), ("y", 4)]);
        let j2 = make_job(circuit, &be, &ck, &[], &[("y", 4), ("x", 3)]);
        assert_eq!(j1.request_hash(), j2.request_hash());
    }
}
