//! Every engine's attested security level, in one auditable place.
//!
//! The impls could live next to each engine, and that would keep each one
//! close to the code it describes. They are gathered here instead for a
//! specific reason: **the whole security posture of the repo should be
//! readable in one sitting.** A reviewer asking "what is actually maliciously
//! secure in Baru?" gets a single file, not a grep across fourteen modules
//! whose doc comments each tell a local truth.
//!
//! ## The rule for editing this file
//!
//! An attestation records what the code **does**, never what the roadmap
//! intends. Raising one is the *last* step of a malicious upgrade — after the
//! checks are implemented, after the cheating-detection tests pass — not the
//! first. If you are here to raise a level, the corresponding
//! `tests/malicious.rs` case should already be green.
//!
//! ## Current posture
//!
//! | Engine | Level | Gap |
//! |---|---|---|
//! | **`MaliciousBackend`** (ABY3, checked) | **malicious** | with abort, ≤1 of 3 corrupt |
//! | `TransportBackend` (ABY3 over transport) | semi-honest | superseded by `MaliciousBackend`; kept as the measurement baseline |
//! | `Aby3Simulator` | semi-honest | in-process simulator; models values, not adversaries |
//! | `gc_protocol` (garbled leg) | semi-honest | tables cross-checked; input labels and evaluator output are not — see `GC_MALICIOUS_GAPS` |
//! | `Semi2kEngine` | semi-honest | no MACs; `mpcc::spdz2k` is the authenticated upgrade |
//! | `ChouOrlandi` (base OT) | semi-honest | trait path omits the Schnorr proof; `new_proved`/`verify_setup` close it |
//! | `Iknp` (OT extension) | semi-honest | trait path omits the KOS check; `simulate_checked` runs it |
//! | `NaiveHash` (PSI) | semi-honest | leaks by construction; see `psi::Hiding` |
//! | `DhPsi` | semi-honest | trait path ships no proof; `respond_proved`/`verify_exponent` close it |
//! | `Pir` | semi-honest | Stage 6 (gated): no integrity against a malicious server |
//! | `zkp::binius_real` | sound | real Binius64 prover/verifier; arithmetic, signed comparisons, shifts, division, `if`/`let` (enough for ReLU and Q16.16) (`tests/zkp_real.rs`) |
//! | `zkp::binius64_backend` | **none** | retained placeholder, unused: `verify` never checks constraint satisfaction (`tests/zkp_soundness.rs`) |
//! | `Bfv`, `Bgv` | semi-honest | `fhe::threshold` built and tested; boundary proof implemented but not reviewed; measured lambda is 30, not 40 (§6d) |
//! | `CkksBackend`, `TfheBackend` | semi-honest | Stage 6 (gated): third-party randomness, cannot derandomise |
//!
//! [`crate::interpreter::malicious_backend::MaliciousBackend`] is the only
//! engine that attests [`Security::Malicious`], and it attests it because the
//! cheating-detection tests in `tests/malicious.rs` demonstrate every deviation
//! point being caught — corrupted resharing, inconsistent dealing, false
//! reveals, and cheating inside bit decomposition. Its attestation lives on the
//! type itself rather than here, next to the code that earns it.
//!
//! Everything else in this table is semi-honest, and the gate in
//! [`super::security::require_security`] enforces that.

use super::security::{AttestedSecurity, Security};

/// Generate an [`AttestedSecurity`] impl with its justification.
///
/// A macro rather than hand-written impls so the note can never be silently
/// separated from the level it explains — both are arguments to the same
/// invocation.
macro_rules! attest {
    ($ty:ty, $level:expr, $note:literal) => {
        impl AttestedSecurity for $ty {
            fn attested_security(&self) -> Security {
                $level
            }
            fn attestation_note(&self) -> &'static str {
                $note
            }
        }
    };
}

// ─── MPC engines ─────────────────────────────────────────────────────────────

attest!(
    crate::interpreter::aby3_simulator::Aby3Simulator,
    Security::SemiHonest,
    "in-process simulator: it models protocol values, not a network or an adversary"
);

impl<T: crate::crypto::net::Transport> AttestedSecurity
    for crate::interpreter::transport_backend::TransportBackend<T>
{
    fn attested_security(&self) -> Security {
        Security::SemiHonest
    }
    fn attestation_note(&self) -> &'static str {
        "ABY3 resharing has no multiplication check: a corrupt party can add an \
         arbitrary offset to its z_i and no honest party can recompute it"
    }
}

attest!(
    crate::mpcc::semi2k::Semi2kEngine,
    Security::SemiHonest,
    "additive shares carry no MACs; deviation is undetectable (SPDZ2k is the upgrade)"
);

// ─── Oblivious transfer ──────────────────────────────────────────────────────

attest!(
    crate::ot::base::ChouOrlandi,
    Security::SemiHonest,
    "the OtProtocol path uses Sender::new, which omits the Schnorr proof of knowledge; \
     Sender::new_proved + verify_setup close that, and Ristretto already rules out \
     malformed and small-order points"
);

attest!(
    crate::ot::iknp::Iknp,
    Security::SemiHonest,
    "the OtProtocol path uses simulate(), which omits the KOS consistency check; \
     iknp::simulate_checked runs it and rejects the inconsistent-column attack that \
     otherwise leaks the sender's s bit by bit"
);

// ─── Private set intersection ────────────────────────────────────────────────

attest!(
    crate::psi::NaiveHash,
    Security::SemiHonest,
    "hashed items leak to an offline dictionary attack even against a passive receiver"
);

attest!(
    crate::psi::dh::DhPsi,
    Security::SemiHonest,
    "the PsiProtocol path uses Sender::respond, which ships no proof; \
     respond_proved + verify_exponent prove a common exponent across the response, \
     which is the deviation that corrupts the receiver's output"
);

// ─── Private information retrieval ───────────────────────────────────────────

attest!(
    crate::pir::Pir,
    Security::SemiHonest,
    "no integrity: a malicious server can return a wrong or malformed answer and the \
     client cannot tell"
);

// ─── FHE backends ────────────────────────────────────────────────────────────

attest!(
    crate::fhe::bfv::Bfv,
    Security::SemiHonest,
    "the threshold scheme switch is built and checkable (hybrid::threshold_switch), but \
     the boundary proof of well-formedness now exists (fhe::boundary_proof) but is a tested \
     implementation rather than a reviewed one, and measurement shows its 26-bit slack leaves \
     lambda = 30 at the largest modulus this u128-based BFV can represent, not the 40 the \
     honest-input analysis reported; the attestation stays semi-honest until both are settled"
);

attest!(
    crate::fhe::bgv::Bgv,
    Security::SemiHonest,
    "evaluation is unverified and decryption uses a single key, not a threshold key"
);

attest!(
    crate::fhe::tfhe_backend::TfheBackend,
    Security::SemiHonest,
    "third-party library: Baru cannot derandomise its internal sampling, so the \
     recompute-and-compare argument does not apply"
);

attest!(
    crate::fhe::ckks_backend::CkksBackend,
    Security::SemiHonest,
    "third-party library (SEAL): internal randomness is not derandomisable from our \
     PRF keys, and CKKS is approximate, so bit-identical comparison is not available"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::security::require_security;

    /// The load-bearing test for this file: an engine listed here as
    /// semi-honest must still be refused when malicious security is required.
    ///
    /// When a stage lands, this test is what forces the claim to be deliberate
    /// — it fails, and whoever raises the attestation must come here and say so
    /// explicitly. That is exactly what happened for the ABY3 layer: the entry
    /// moved to `MaliciousBackend`, and `TransportBackend` stayed semi-honest
    /// because it *is* still semi-honest.
    #[test]
    fn engines_in_this_table_are_refused_malicious_security() {
        let sim = crate::interpreter::aby3_simulator::Aby3Simulator::new();
        assert_eq!(sim.attested_security(), Security::SemiHonest);
        assert!(require_security("aby3-sim", &sim, Security::Malicious).is_err());

        assert!(require_security("iknp", &crate::ot::iknp::Iknp, Security::Malicious).is_err());
        assert!(require_security("dh-psi", &crate::psi::dh::DhPsi, Security::Malicious).is_err());
    }

    /// The Stage 6 gate: for each scheme whose malicious upgrade is itself a
    /// research project, requesting malicious security must fail rather than
    /// silently downgrade.
    ///
    /// | Scheme | Why it is gated |
    /// |---|---|
    /// | `Pir` | Integrity for single-server PIR is an open problem, and PIR already loses to downloading the database by 13× here |
    /// | `TfheBackend`, `CkksBackend` | Third-party libraries whose internal sampling Baru cannot derandomise, so recompute-and-compare does not apply; CKKS is approximate, so bit-identical comparison is unavailable in principle |
    /// | `Bfv` | `fhe::threshold` is built and tested, but externally supplied ciphertexts still need the boundary proof that replication provably cannot substitute for |
    /// | `Iknp`, `ChouOrlandi`, `DhPsi` | The checked entry points exist; the `OtProtocol`/`PsiProtocol` trait paths still call the unchecked ones |
    #[test]
    fn research_grade_gaps_refuse_malicious_security() {
        let pir = crate::pir::Pir::new(crate::fhe::bfv::BfvParams::TEST_SMALL, 8, 4)
            .expect("a small PIR instance");
        let cases: Vec<(&'static str, &dyn AttestedSecurity)> = vec![
            ("pir", &pir),
            ("tfhe", &crate::fhe::tfhe_backend::TfheBackend),
            ("iknp", &crate::ot::iknp::Iknp),
            ("chou-orlandi", &crate::ot::base::ChouOrlandi),
            ("dh-psi", &crate::psi::dh::DhPsi),
            ("naive-hash", &crate::psi::NaiveHash),
        ];
        for (name, engine) in cases {
            let result = require_security(name, engine, Security::Malicious);
            assert!(result.is_err(), "{name} must refuse malicious security");
            let err = result.unwrap_err();
            assert!(
                !err.note.is_empty(),
                "{name} refuses malicious security without naming the gap"
            );
        }
    }

    /// Every attestation below malicious must say *why*, or the gate's error
    /// message is useless to whoever hits it.
    #[test]
    fn every_gap_is_explained() {
        let engines: Vec<(&str, &dyn AttestedSecurity)> = vec![
            ("iknp", &crate::ot::iknp::Iknp),
            ("chou-orlandi", &crate::ot::base::ChouOrlandi),
            ("dh-psi", &crate::psi::dh::DhPsi),
            ("naive-hash", &crate::psi::NaiveHash),
            ("tfhe", &crate::fhe::tfhe_backend::TfheBackend),
        ];
        for (name, e) in engines {
            if e.attested_security() < Security::Malicious {
                assert!(
                    !e.attestation_note().is_empty(),
                    "{name} attests below malicious without explaining the gap"
                );
            }
        }
    }
}
