//! Fully Homomorphic Encryption (FHE) module for Baru.
//!
//! Architecture mirrors the ZKP module:
//!
//!   Lisp AST
//!     │
//!     ▼  circuit_compiler::compile_fhe()   (shared FlatIR pipeline)
//!   FheCircuit  (optimised SSA block, no non-determinism)
//!     │
//!     ▼  circuit_emitter::evaluate_fhe()
//!   FheBackend calls  (per-gate ciphertext operations)
//!
//! User-facing builtins (registered in eval.rs):
//!
//!   (fhe-keygen)
//!       → returns an opaque key-pair handle (JSON string)
//!
//!   (fhe-enc keys value)
//!       → encrypts `value` (Num or Fixed) using the client key in `keys`
//!       → returns a Types::FheCiphertext handle
//!
//!   (fhe-dec keys ciphertext)
//!       → decrypts `ciphertext` using the client key in `keys`
//!       → returns Types::Num (or Types::Fixed if the circuit was fixed-point)
//!
//!   (fhe-eval circuit-str keys '("pub-x") '(3) '("enc-y") '(ct))
//!       → evaluates the compiled FheCircuit homomorphically
//!       → public inputs are plaintext; encrypted inputs are ciphertexts
//!       → returns a Types::FheCiphertext (the encrypted result)
//!
//!   (fhe-compile 'expr '("enc-y"))
//!       → compiles a Lisp expression to an FheCircuit JSON string
//!       → second arg names the variables that will be encrypted inputs
//!       → analogous to (zkp-compile 'expr '("x"))
//!
//! Comparison with ZKP and MPC:
//!
//!   ZKP: prover holds private witness; verifier checks a proof non-interactively.
//!        Non-determinism (div/mod) allowed.
//!
//!   MPC: each participant holds a secret share; parties communicate per-gate.
//!        Result reconstructed by all.
//!
//!   FHE: one or more participants encrypt their private data under a public key;
//!        any party can evaluate arbitrary functions over ciphertexts without
//!        learning the plaintext.  After evaluation, each holder of the client
//!        key decrypts their result.  Less communication than MPC (send once),
//!        but heavier per-gate computation (bootstrapping).

pub mod backend;
pub mod bfv;
pub mod bgv;
/// BGV at wide parameters (n=8192, q=2^190, 256-bit coefficients) — the regime
/// [`crate::mpcc::lowgear`] needs for a field big enough to bit-decompose.
pub mod bgv_wide;
/// Fixed-width signed integers generic over limb count — the CRT accumulator
/// for [`ntt`], at both the narrow and wide parameter profiles.
pub mod bigint;
pub mod boundary;
pub mod boundary_proof;
pub mod circuit_emitter;
pub mod ckks_backend;
pub mod ckks_compare;
pub mod conv_pack;
pub mod mock_backend;
/// NTL+GMP polynomial multiplication — a second implementation of what
/// [`ntt`] computes, and the differential oracle for it. Opt-in: needs a C++
/// toolchain plus NTL and GMP (`cargo test --features ntl`).
#[cfg(feature = "ntl")]
pub mod ntl;
pub mod ntt;
/// Runtime switch choosing which implementation computes the ring products
/// every FHE scheme here depends on ([`ntt`], or `ntl` when built in).
pub mod poly;
/// CRT slot packing — `n` independent values per ciphertext, for the
/// elementwise products [`crate::mpcc::lowgear`] needs.
pub mod slots;
pub mod tfhe_backend;
pub mod threshold;

pub use backend::{FheBackend, FheCircuit, RealFheBackend};
pub use ckks_backend::{CkksBackend, MockCkksBackend};
pub use mock_backend::MockFheBackend;
pub use tfhe_backend::TfheBackend;
