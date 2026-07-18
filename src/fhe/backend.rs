//! FHE backend trait and circuit data structures.
//!
//! `FheBackend` is generic over an associated `Ciphertext` type so that:
//!   - `MockFheBackend`  uses plain `u64`  (tests, REPL development)
//!   - `TfheBackend`     will use `FheUint64` from tfhe-rs
//!   - Future CKKS       would use polynomial ciphertexts
//!
//! There is intentionally no `reveal` method during evaluation — revealing
//! requires the client key and is a separate step (`fhe-dec`).

use crate::zkp::circuit_compiler::FlatOp;
use serde::{Deserialize, Serialize};

// ─── Circuit types ────────────────────────────────────────────────────────────

/// A compiled FHE circuit, produced by `compile_fhe`.
/// Structurally identical to the ZKP `CompiledCircuit` except:
///   - No `witnesses` / non-deterministic ops (FHE has no prover/verifier split)
///   - `encrypted_inputs` names the variables that arrive as ciphertexts
///   - `public_inputs`    names the variables that arrive as plaintext constants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FheCircuit {
    /// Optimised flat SSA block (same FlatOp IR as ZKP).
    pub flat_ops: Vec<FlatOp>,
    /// Variables that must be supplied as ciphertexts at evaluation time.
    pub encrypted_inputs: Vec<String>,
    /// Variables that must be supplied as plaintext integers at evaluation time.
    pub public_inputs: Vec<String>,
    /// The wire whose value is the circuit output.
    pub output_wire: String,
    /// Total gate count.
    pub num_gates: usize,
}

// ─── Key handles (serialisable for REPL round-trips) ─────────────────────────

/// Opaque representation of an FHE key pair.
/// The actual cryptographic material lives in the backend; this struct only
/// carries what is needed to identify and (de)serialise keys in JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FheKeyPair {
    /// Backend identifier — used to reject mismatched keys.
    pub backend_id: String,
    /// Serialised client key bytes (backend-specific encoding).
    pub client_key_bytes: Vec<u8>,
    /// Serialised server/evaluation key bytes.
    pub server_key_bytes: Vec<u8>,
}

// ─── Ciphertext handle ────────────────────────────────────────────────────────

/// A serialisable ciphertext handle passed around in the Baru runtime.
/// The `bytes` field contains the backend-specific encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FheCiphertextHandle {
    /// Backend that created this ciphertext.
    pub backend_id: String,
    /// Serialised ciphertext bytes.
    pub bytes: Vec<u8>,
}

// ─── Backend trait ────────────────────────────────────────────────────────────

/// Trait that every FHE backend must implement.
///
/// The associated `Ct` type is the backend's native ciphertext type.
/// All arithmetic / logic methods take ciphertexts by reference so
/// the backend can decide whether to clone internally.
pub trait FheBackend {
    type Ct: Clone + Send + Sync;

    // ── Key management ───────────────────────────────────────────────────────

    /// Generate a fresh key pair.  Returns `(client_key_bytes, server_key_bytes)`.
    fn keygen(&self) -> (Vec<u8>, Vec<u8>);

    /// Encrypt a 64-bit integer value using the client key.
    fn encrypt(&self, client_key_bytes: &[u8], value: u64) -> Self::Ct;

    /// Trivially "encrypt" a public constant, without any key.
    ///
    /// The result is a noiseless ciphertext that is compatible with
    /// ciphertexts under *any* client key of this backend.  The circuit
    /// emitter uses this to promote plaintext constants into the ciphertext
    /// domain.  The value is public — this provides no privacy.
    fn encrypt_trivial(&self, value: i64) -> Self::Ct;

    /// Decrypt a ciphertext using the client key.
    fn decrypt(&self, client_key_bytes: &[u8], ct: &Self::Ct) -> u64;

    /// Serialise a ciphertext to bytes (for JSON transport / storage).
    fn serialize_ct(&self, ct: &Self::Ct) -> Vec<u8>;

    /// Deserialise a ciphertext from bytes.
    fn deserialize_ct(&self, bytes: &[u8]) -> Self::Ct;

    /// Unique identifier for this backend (e.g. "mock", "tfhe-rs-v0.6").
    fn backend_id(&self) -> &'static str;

    // ── Arithmetic ───────────────────────────────────────────────────────────

    fn add(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;
    fn sub(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;
    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;

    /// Add a plaintext constant to a ciphertext (cheaper than ciphertext-ciphertext add).
    fn add_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct;

    /// Multiply a ciphertext by a plaintext constant.
    fn mul_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct;

    // ── Comparisons (return encrypted booleans: 0 or 1) ──────────────────────

    fn cmpeq(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;

    /// Unsigned less-than: interprets both operands as u64.
    fn cmplt(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;

    /// Signed less-than: interprets both operands as two's-complement i64.
    /// This is what Baru's `<` / `>` compile to (`ICmpSlt`).
    fn cmpslt(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;

    // ── Bitwise ──────────────────────────────────────────────────────────────

    fn band(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;
    fn bor(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;
    fn bxor(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;
    fn bnot(&self, a: &Self::Ct) -> Self::Ct;

    // ── Shifts (static amount — same constraint as ZKP) ───────────────────────

    fn shl(&self, a: &Self::Ct, amount: u32) -> Self::Ct;
    fn shr(&self, a: &Self::Ct, amount: u32) -> Self::Ct;

    // ── Control-flow via Mux (no branching in FHE) ───────────────────────────

    /// Multiplexer: `cond` must be an encrypted bit (0 or 1).
    /// Returns `t` when `cond == 1`, else `e`.
    fn mux(&self, cond: &Self::Ct, t: &Self::Ct, e: &Self::Ct) -> Self::Ct;

    // ── Division via PBS (expensive — optional) ───────────────────────────────

    /// Non-deterministic division via programmable bootstrapping.
    /// Returns `(quotient, remainder)`.
    /// Default: panics with an informative message so backends can opt in.
    fn divmod(&self, _a: &Self::Ct, _b: &Self::Ct) -> (Self::Ct, Self::Ct) {
        panic!(
            "divmod is not supported by the '{}' FHE backend. \
             Consider pre-computing the quotient on the client side.",
            self.backend_id()
        )
    }

    // ── Fixed-point helpers (default implementations via shr/mul) ─────────────

    /// Fixed-point multiply: `(a * b) >> 16`.
    /// Uses i128-width intermediate to avoid overflow.
    fn fixed_mul(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        let prod = self.mul(a, b);
        self.shr(&prod, 16)
    }

    /// Fixed-point divide: `(a << 16) / b`.
    fn fixed_div(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        let shifted = self.shl(a, 16);
        let (q, _) = self.divmod(&shifted, b);
        q
    }
}
