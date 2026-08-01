//! What each protocol in this crate is *actually* secure against — and the
//! gate that stops anything claiming more than it implements.
//!
//! # Why this module exists
//!
//! Baru describes a dozen protocols, and until now the honest statement of
//! their threat model lived only in module doc comments: `ot::iknp` says it is
//! broken against a malicious receiver, `mpcc::semi2k` says it "cannot be cited
//! as maliciously secure", `pir` says it has no integrity at all. Those notes
//! are correct and were written in good faith, but a doc comment cannot stop a
//! caller — or a benchmark, or a paper table — from treating the protocol as
//! actively secure anyway.
//!
//! [`Security`] used to be *metadata*: a field on [`crate::mpcc::protocols::ProtocolSpec`]
//! that cost reports printed. This module promotes it to a **load-bearing
//! runtime value**. Every engine attests the level it implements via
//! [`AttestedSecurity`], and every session that asks for a level runs it past
//! [`require_security`] first. Asking for more than the engine attests is an
//! error at session construction, not a silent downgrade.
//!
//! The point is not defence in depth against an attacker — an attacker does not
//! call our constructors. The point is that Baru's own results stay honest
//! while the malicious upgrade lands **incrementally**: a half-finished stage
//! cannot be mistaken for a finished one, by a caller or by us.
//!
//! # What "malicious" means here
//!
//! Throughout Baru, attesting [`Security::Malicious`] means **security with
//! abort** under the protocol's own corruption threshold (for the ABY3 family:
//! a static adversary corrupting at most one of three parties). Honest parties
//! either produce the correct output or return [`Abort`]. It does **not** mean
//! guaranteed output delivery: a corrupt party can always make the computation
//! fail, it just cannot make it lie.

use std::fmt;

use serde::Serialize;

/// Adversary behaviour a protocol is secure against. Ordered: a protocol secure
/// against a stronger adversary also satisfies a weaker requirement, which is
/// exactly what [`require_security`] relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Security {
    /// Corrupt parties follow the protocol and only try to learn from what they
    /// legitimately see.
    SemiHonest,
    /// Deviation is caught with some fixed probability, and the adversary is
    /// deterred rather than stopped. No Baru protocol attests this today; it
    /// exists because the protocol registry costs covert variants.
    Covert,
    /// Corrupt parties may deviate arbitrarily. In Baru this always means
    /// **security with abort** — see the module header.
    Malicious,
}

impl fmt::Display for Security {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Security::SemiHonest => write!(f, "semi-honest"),
            Security::Covert => write!(f, "covert"),
            Security::Malicious => write!(f, "malicious"),
        }
    }
}

impl Security {
    pub fn parse(s: &str) -> Option<Security> {
        match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "semi-honest" | "semihonest" | "passive" => Some(Security::SemiHonest),
            "covert" => Some(Security::Covert),
            "malicious" | "active" => Some(Security::Malicious),
            _ => None,
        }
    }

    /// Whether a protocol attesting `self` may be used where `required` is
    /// asked for. Stronger satisfies weaker, by the enum's ordering.
    pub fn satisfies(self, required: Security) -> bool {
        self >= required
    }
}

/// A protocol engine's attestation of what it implements.
///
/// Implement this by reading the code, not the roadmap. The value must be what
/// the protocol *does*; if a malicious upgrade is half-landed, the attestation
/// stays [`Security::SemiHonest`] until the checks are complete and tested.
///
/// Deliberately object-safe (a method, not an associated const) so engines can
/// be held behind `dyn` — the interpreter's `Backend` is used that way.
pub trait AttestedSecurity {
    fn attested_security(&self) -> Security;

    /// Human-readable reason the attestation is what it is. Shown in the error
    /// when a caller asks for more, so the message points at the actual gap
    /// rather than just saying "no".
    fn attestation_note(&self) -> &'static str {
        ""
    }
}

/// Refusal to run at a requested security level.
///
/// Separate from [`Abort`]: this is a *configuration* error discovered before
/// any protocol message is sent, whereas an abort is a run-time detection of
/// cheating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityGateError {
    pub engine: &'static str,
    pub required: Security,
    pub attested: Security,
    pub note: &'static str,
}

impl fmt::Display for SecurityGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} attests {} security but {} was required",
            self.engine, self.attested, self.required
        )?;
        if !self.note.is_empty() {
            write!(f, " ({})", self.note)?;
        }
        Ok(())
    }
}

impl std::error::Error for SecurityGateError {}

/// Gate a session on the engine's attested level.
///
/// Call this at session construction, before any protocol message is sent.
pub fn require_security<E: AttestedSecurity + ?Sized>(
    engine_name: &'static str,
    engine: &E,
    required: Security,
) -> Result<(), SecurityGateError> {
    let attested = engine.attested_security();
    if attested.satisfies(required) {
        Ok(())
    } else {
        Err(SecurityGateError {
            engine: engine_name,
            required,
            attested,
            note: engine.attestation_note(),
        })
    }
}

// ─── Abort ───────────────────────────────────────────────────────────────────

/// Where a consistency check failed. Carried by [`Abort`] so a failed run says
/// which check caught the deviation, which is what the cheating-detection tests
/// assert on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortReason {
    /// A party's broadcast reveal component disagreed with the copy the
    /// receiver already held under replicated sharing.
    RevealMismatch,
    /// A dealer sent inconsistent components of the same share to the two
    /// parties that are supposed to hold it.
    InputInconsistent,
    /// The batched multiplication sacrifice check did not open to zero.
    MultiplicationCheck,
    /// Two parties' accumulated transcripts disagreed at the deferred check.
    ViewMismatch,
    /// A garbled table or input label differed between the two garblers.
    GarbleMismatch,
    /// A partial decryption disagreed with its co-holder's recomputation.
    PartialDecryptionMismatch,
    /// A commitment did not match its later opening (coin flip).
    CommitmentMismatch,
}

impl fmt::Display for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AbortReason::RevealMismatch => "reveal component mismatch",
            AbortReason::InputInconsistent => "inconsistent input sharing",
            AbortReason::MultiplicationCheck => "multiplication check failed",
            AbortReason::ViewMismatch => "transcript view mismatch",
            AbortReason::GarbleMismatch => "garbled circuit mismatch",
            AbortReason::PartialDecryptionMismatch => "partial decryption mismatch",
            AbortReason::CommitmentMismatch => "commitment opening mismatch",
        };
        f.write_str(s)
    }
}

/// Cheating was detected; the honest parties must stop.
///
/// This is a `Result` error rather than a `panic!` on purpose. A malicious
/// abort is a *protocol outcome* — the benchmarks count them, the tests assert
/// on them, and a caller may legitimately want to retry with a different party
/// set. A panic would be indistinguishable from the transport timeouts that the
/// existing code panics on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Abort {
    pub reason: AbortReason,
    /// Which check site raised it, for debugging.
    pub site: &'static str,
}

impl Abort {
    pub fn new(reason: AbortReason, site: &'static str) -> Self {
        Abort { reason, site }
    }
}

impl fmt::Display for Abort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "protocol abort at {}: {}", self.site, self.reason)
    }
}

impl std::error::Error for Abort {}

/// Result of any operation that can detect cheating.
pub type Checked<T> = Result<T, Abort>;

#[cfg(test)]
mod tests {
    use super::*;

    struct Engine(Security, &'static str);
    impl AttestedSecurity for Engine {
        fn attested_security(&self) -> Security {
            self.0
        }
        fn attestation_note(&self) -> &'static str {
            self.1
        }
    }

    #[test]
    fn stronger_satisfies_weaker() {
        assert!(Security::Malicious.satisfies(Security::SemiHonest));
        assert!(Security::Malicious.satisfies(Security::Malicious));
        assert!(!Security::SemiHonest.satisfies(Security::Malicious));
        assert!(!Security::Covert.satisfies(Security::Malicious));
    }

    #[test]
    fn gate_refuses_an_overclaim_and_says_why() {
        let e = Engine(Security::SemiHonest, "no KOS consistency check");
        let err = require_security("iknp", &e, Security::Malicious).unwrap_err();
        assert_eq!(err.attested, Security::SemiHonest);
        let msg = err.to_string();
        assert!(msg.contains("iknp"), "{msg}");
        assert!(msg.contains("no KOS consistency check"), "{msg}");
    }

    #[test]
    fn gate_admits_an_engine_that_attests_enough() {
        let e = Engine(Security::Malicious, "");
        assert!(require_security("aby3", &e, Security::Malicious).is_ok());
        assert!(require_security("aby3", &e, Security::SemiHonest).is_ok());
    }

    #[test]
    fn parse_round_trips_display() {
        for s in [Security::SemiHonest, Security::Covert, Security::Malicious] {
            assert_eq!(Security::parse(&s.to_string()), Some(s));
        }
    }
}
