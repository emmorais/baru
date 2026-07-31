//! Soundness audit of the ZKP backend.
//!
//! Stage 6 of the malicious-security work called for an *audit* of `src/zkp/`
//! rather than an implementation, on the reasoning that a SNARK is already
//! sound against a malicious prover by construction and the job was to confirm
//! the verifier is genuinely invoked on every path.
//!
//! The audit did not confirm that. These tests record what the backend actually
//! does, because the gap between the claim and the code is large enough that it
//! must be demonstrated rather than described.
//!
//! **Superseded.** `zkp::binius_real` is now a real Binius64 prover/verifier and
//! is what the interpreter uses; `tests/zkp_real.rs` runs these same attacks
//! against it, where they are rejected. This file is kept pointing at the old
//! `binius64_backend` so the contrast stays visible and so nothing quietly
//! reintroduces it.

use std::collections::HashMap;

use baru::interpreter::parser::parse;
use baru::zkp::{binius64_backend::Binius64Backend, ZKPBackend};

/// Compile `(* x y)` with `x` private and `y` public.
fn circuit() -> baru::zkp::Circuit {
    let expr = parse("(* x y)").expect("parse");
    let be = Binius64Backend::new();
    be.compile_circuit(&expr, &["x".to_string()])
        .expect("compile")
}

/// An honest proof verifies. This is the only property the backend has.
#[test]
fn an_honest_proof_verifies() {
    let be = Binius64Backend::new();
    let ckt = circuit();
    let mut private = HashMap::new();
    private.insert("x".to_string(), 6i64);
    let mut public = HashMap::new();
    public.insert("y".to_string(), 7i64);

    let proof = be.prove(&ckt, &private, &public).expect("prove");
    assert!(be.verify(&ckt, &proof, &public).expect("verify"));
}

/// **The finding.** A prover that supplies no witness at all still produces a
/// proof the verifier accepts.
///
/// `verify` checks the magic header, that the circuit hash matches a hash it
/// recomputes from the *public* circuit, that the public values echo the ones
/// supplied, and that the proof is long enough. It never checks that any
/// witness satisfies any constraint — it cannot, since nothing in the proof
/// binds a satisfying assignment to the commitment.
///
/// So anyone holding the circuit (which is public) can produce an accepting
/// proof for any public values, without knowing a witness. That is not a
/// soundness *weakness*; it is the absence of soundness.
#[test]
fn a_proof_with_no_witness_is_accepted() {
    let be = Binius64Backend::new();
    let ckt = circuit();
    let mut public = HashMap::new();
    public.insert("y".to_string(), 7i64);

    let forged = be
        .prove(&ckt, &HashMap::new(), &public)
        .expect("proving with an empty witness succeeds");
    assert!(
        be.verify(&ckt, &forged, &public).expect("verify"),
        "documented: the verifier accepts a proof produced without any witness"
    );
}

/// A witness that does not satisfy the circuit is equally acceptable, which is
/// the same gap seen from the other side.
#[test]
fn a_witness_that_satisfies_nothing_is_accepted() {
    let be = Binius64Backend::new();
    let ckt = circuit();
    let mut public = HashMap::new();
    public.insert("y".to_string(), 7i64);

    let mut nonsense = HashMap::new();
    nonsense.insert("completely_unrelated".to_string(), 999i64);

    let proof = be.prove(&ckt, &nonsense, &public).expect("prove");
    assert!(
        be.verify(&ckt, &proof, &public).expect("verify"),
        "documented: witness contents are never checked against the constraints"
    );
}

/// The backend does not use the `binius-field` / `binius-core` crates at all.
///
/// They are declared in `Cargo.toml` and pulled from git, but nothing under
/// `src/` imports them. The "FRI protocol commitment layer" in `prove` is
/// hashes of the proof-so-far — its own comment says "simulated" — and
/// `compute_field_hash` is hand-rolled rather than a standard hash.
///
/// This test is a tripwire: if a real Binius64 proof system is wired in, the
/// soundness tests above should start failing, and this note should be removed
/// along with them.
#[test]
fn the_backend_is_a_placeholder_not_a_proof_system() {
    let be = Binius64Backend::new();
    let ckt = circuit();
    let mut public = HashMap::new();
    public.insert("y".to_string(), 7i64);

    // Two different witnesses produce proofs the verifier treats identically.
    let mut w1 = HashMap::new();
    w1.insert("x".to_string(), 6i64);
    let mut w2 = HashMap::new();
    w2.insert("x".to_string(), -12345i64);

    let p1 = be.prove(&ckt, &w1, &public).expect("prove");
    let p2 = be.prove(&ckt, &w2, &public).expect("prove");
    assert!(be.verify(&ckt, &p1, &public).unwrap());
    assert!(
        be.verify(&ckt, &p2, &public).unwrap(),
        "any witness is accepted for the same public statement"
    );
}
