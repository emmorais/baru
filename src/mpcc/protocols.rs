//! The protocol registry — Baru's answer to MP-SPDZ's `*-party.x` matrix.
//!
//! Every protocol is described by a **capability** part (domain, party count,
//! corruption model) that decides *whether* it can run a given deployment, and a
//! **cost** part ([`CostParams`]) that decides *how expensive* it would be.
//! Protocols that Baru has not implemented yet still carry cost parameters, so
//! the analytic comparison in [`crate::mpcc::cost`] covers the whole matrix from
//! day one and each new implementation *validates* one row of it rather than
//! unlocking it.
//!
//! ## On the numbers
//!
//! The online parameters (elements per multiplication, rounds) are exact —
//! they come from the protocol definitions and are not tunable.  The **offline**
//! parameters and the local-work constants are order-of-magnitude estimates from
//! the literature, and are explicitly the thing that phase P1 of
//! `docs/mpc-compiler-roadmap.md` calibrates against measured runs.  Every
//! report that depends on them says so; do not quote them as measurements.

use std::fmt;

use serde::Serialize;

use super::ir::DomainKind;

/// Corruption threshold the protocol tolerates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Majority {
    /// Fewer than half the parties may be corrupt.
    Honest,
    /// Any number of parties may be corrupt (all but one).
    Dishonest,
    /// A trusted dealer supplies correlated randomness; not a real threat model,
    /// but the exact lower bound on online cost.
    Dealer,
}

impl fmt::Display for Majority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Majority::Honest => write!(f, "honest"),
            Majority::Dishonest => write!(f, "dishonest"),
            Majority::Dealer => write!(f, "dealer"),
        }
    }
}

/// Adversary behaviour the protocol is secure against.  Ordered: a protocol
/// secure against a stronger adversary also satisfies a weaker requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Security {
    SemiHonest,
    Covert,
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
}

/// Whether Baru can actually run this protocol today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Status {
    /// Implemented; the `&'static str` names the module.
    Implemented(&'static str),
    /// Not implemented; the `&'static str` names the roadmap phase.
    Planned(&'static str),
}

impl Status {
    pub fn is_implemented(&self) -> bool {
        matches!(self, Status::Implemented(_))
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Status::Implemented(m) => write!(f, "impl ({})", m),
            Status::Planned(p) => write!(f, "planned ({})", p),
        }
    }
}

/// Communication volume that may scale with the number of peers.
///
/// Replicated protocols send a fixed amount regardless of `n` (each party talks
/// to one neighbour); Shamir-style and additive-sharing protocols send to every
/// peer, so their volume is linear in `n − 1`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Elems {
    pub base: f64,
    pub per_peer: f64,
}

impl Elems {
    pub const fn new(base: f64, per_peer: f64) -> Self {
        Elems { base, per_peer }
    }

    /// Elements one party sends, for `parties` total participants.
    pub fn total(&self, parties: usize) -> f64 {
        self.base + self.per_peer * (parties.saturating_sub(1)) as f64
    }
}

/// Per-primitive costs, from which [`crate::mpcc::cost`] composes everything
/// else (comparisons, truncations, divisions, conversions).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CostParams {
    /// Rounds for a batch of secret×secret multiplications.
    pub mul_rounds: u32,
    /// Domain elements one party sends per multiplied lane.
    pub mul_elems: Elems,
    /// Rounds to open a batch of secrets to all parties.
    pub open_rounds: u32,
    pub open_elems: Elems,
    /// Beaver triples consumed per multiplication (0 for replicated protocols,
    /// which multiply from correlated randomness instead).
    pub triples_per_mul: f64,
    /// Domain elements of *offline* communication per triple produced.
    /// Zero means the protocol has no separate preprocessing phase.
    ///
    /// Used only when `offline_from_ot` is `None`; an OT-based protocol derives
    /// this from measured OT cost instead of restating a literature figure.
    pub offline_elems_per_triple: f64,
    /// Set when preprocessing is OT-based, in which case offline volume is
    /// computed from the measured OT cost in [`crate::mpcc::ot_cost`] rather
    /// than from `offline_elems_per_triple`.
    pub offline_from_ot: Option<crate::mpcc::ot_cost::OtTriples>,
    /// Rounds for a batch of `GF(2)` ANDs.
    pub and_rounds: u32,
    /// Bits one party sends per AND lane.  XOR is free everywhere.
    pub and_bits: Elems,
    /// edaBits consumed per bit of a bit-decomposition.
    pub edabits_per_bit: f64,
    /// daBits consumed per bit of an arithmetic↔binary conversion.
    pub dabits_per_bit: f64,
    /// Multiplicative overhead on all online volume from MAC/authentication
    /// material (2.0 = every share is accompanied by a MAC share).
    pub auth_overhead: f64,
    /// Constant-round in circuit depth: garbled circuits and BMR pay their cost
    /// in bandwidth and local crypto, not in latency.
    pub constant_round: bool,
    /// Total rounds for a constant-round protocol (input OT, garbling
    /// transfer, output).  Ignored unless `constant_round`.
    pub constant_rounds: u32,
    /// Size of the prime a field protocol works over.  Honest-majority
    /// protocols are happy with a ~61-bit Mersenne prime; SPDZ-family protocols
    /// need a large prime for MAC security.  Ignored by ring and binary
    /// protocols.
    pub field_bits: u32,
    /// Local cryptographic work per interactive lane, in nanoseconds.  Captures
    /// AES calls for garbling, MAC checks for authenticated protocols, and plain
    /// field arithmetic otherwise.  An estimate; calibrated in P1.
    pub local_ns_per_lane: f64,
}

/// A protocol Baru can compare against, whether or not it can run it.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ProtocolSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub domain: DomainKind,
    pub min_parties: usize,
    /// `None` = unbounded.
    pub max_parties: Option<usize>,
    pub majority: Majority,
    pub security: Security,
    pub status: Status,
    pub cost: CostParams,
    /// Citation for the protocol, so reports can point at the source.
    pub reference: &'static str,
}

impl ProtocolSpec {
    pub fn supports_parties(&self, n: usize) -> bool {
        n >= self.min_parties && self.max_parties.map(|m| n <= m).unwrap_or(true)
    }
}

// ─── The matrix ───────────────────────────────────────────────────────────────

const SEMI_RING_SHARED: CostParams = CostParams {
    mul_rounds: 1,
    mul_elems: Elems::new(1.0, 0.0),
    open_rounds: 1,
    open_elems: Elems::new(1.0, 0.0),
    triples_per_mul: 0.0,
    offline_elems_per_triple: 0.0,
    offline_from_ot: None,
    and_rounds: 1,
    and_bits: Elems::new(1.0, 0.0),
    edabits_per_bit: 1.0,
    dabits_per_bit: 1.0,
    auth_overhead: 1.0,
    constant_round: false,
    constant_rounds: 0,
    field_bits: 64,
    local_ns_per_lane: 5.0,
};

/// Every protocol Baru knows about, implemented or not.
pub const REGISTRY: &[ProtocolSpec] = &[
    // ── honest majority, replicated ───────────────────────────────────────
    ProtocolSpec {
        id: "aby3",
        name: "ABY3 / replicated 3-party",
        domain: DomainKind::Ring,
        min_parties: 3,
        max_parties: Some(3),
        majority: Majority::Honest,
        security: Security::SemiHonest,
        status: Status::Implemented("crypto::aby3_protocol"),
        cost: SEMI_RING_SHARED,
        reference: "Mohassel & Rindal, ABY3 (CCS'18)",
    },
    ProtocolSpec {
        id: "rep-field",
        name: "Replicated 3-party over a field",
        domain: DomainKind::Field,
        min_parties: 3,
        max_parties: Some(3),
        majority: Majority::Honest,
        security: Security::SemiHonest,
        status: Status::Planned("P2"),
        cost: SEMI_RING_SHARED,
        reference: "Araki et al. (CCS'16)",
    },
    ProtocolSpec {
        id: "rep4",
        name: "Rep4 — 4-party replicated, malicious",
        domain: DomainKind::Ring,
        min_parties: 4,
        max_parties: Some(4),
        majority: Majority::Honest,
        security: Security::Malicious,
        status: Status::Planned("P3"),
        cost: CostParams {
            mul_elems: Elems::new(2.0, 0.0),
            open_elems: Elems::new(2.0, 0.0),
            local_ns_per_lane: 10.0,
            ..SEMI_RING_SHARED
        },
        reference: "Dalskov, Escudero & Keller (USENIX'21)",
    },
    // ── honest majority, threshold sharing ────────────────────────────────
    ProtocolSpec {
        id: "shamir",
        name: "Shamir secret sharing",
        domain: DomainKind::Field,
        min_parties: 3,
        max_parties: None,
        majority: Majority::Honest,
        security: Security::SemiHonest,
        status: Status::Planned("P1"),
        cost: CostParams {
            mul_elems: Elems::new(0.0, 1.0),
            open_elems: Elems::new(0.0, 1.0),
            and_bits: Elems::new(0.0, 1.0),
            ..SEMI_RING_SHARED
        },
        reference: "Ben-Or, Goldwasser & Wigderson (STOC'88)",
    },
    ProtocolSpec {
        id: "atlas",
        name: "ATLAS — king-based honest majority",
        domain: DomainKind::Field,
        min_parties: 3,
        max_parties: None,
        majority: Majority::Honest,
        security: Security::SemiHonest,
        status: Status::Planned("P2"),
        cost: CostParams {
            // Two rounds through a "king" party, but constant volume per party
            // instead of linear in n.
            mul_rounds: 2,
            mul_elems: Elems::new(2.0, 0.0),
            ..SEMI_RING_SHARED
        },
        reference: "Goyal, Li, Ostrovsky, Polychroniadou & Song (CRYPTO'21)",
    },
    // ── dishonest majority, additive sharing ──────────────────────────────
    ProtocolSpec {
        id: "dealer",
        name: "Dealer model — trusted preprocessing",
        domain: DomainKind::Ring,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dealer,
        security: Security::SemiHonest,
        status: Status::Planned("P1"),
        cost: CostParams {
            // Beaver online: broadcast d and e.
            mul_elems: Elems::new(0.0, 2.0),
            open_elems: Elems::new(0.0, 1.0),
            triples_per_mul: 1.0,
            // The dealer just ships them: 3 elements, no interaction cost.
            offline_elems_per_triple: 3.0,
            and_bits: Elems::new(0.0, 2.0),
            ..SEMI_RING_SHARED
        },
        reference: "the online-cost lower bound for additive sharing",
    },
    ProtocolSpec {
        id: "semi2k",
        name: "Semi2k — additive mod 2^k, OT triples",
        domain: DomainKind::Ring,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dishonest,
        security: Security::SemiHonest,
        status: Status::Implemented("mpcc::semi2k"),
        cost: CostParams {
            mul_elems: Elems::new(0.0, 2.0),
            open_elems: Elems::new(0.0, 1.0),
            triples_per_mul: 1.0,
            // Offline volume comes from the *measured* OT cost, not a
            // literature figure: 2k correlated OTs per triple (Gilboa), each
            // priced by `baru bench-ot`.  See `crate::mpcc::ot_cost`.
            offline_from_ot: Some(crate::mpcc::ot_cost::OtTriples::PASSIVE),
            offline_elems_per_triple: 0.0,
            and_bits: Elems::new(0.0, 2.0),
            ..SEMI_RING_SHARED
        },
        reference: "Cramer, Damgård, Escudero, Scholl & Xing, SPDZ2k (CRYPTO'18), passive variant",
    },
    ProtocolSpec {
        id: "spdz2k",
        name: "SPDZ2k — authenticated mod 2^k",
        domain: DomainKind::Ring,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dishonest,
        security: Security::Malicious,
        status: Status::Planned("P2"),
        cost: CostParams {
            mul_elems: Elems::new(0.0, 2.0),
            open_elems: Elems::new(0.0, 1.0),
            triples_per_mul: 1.0,
            // Sacrifice doubles the OTs, and shares are k+s bits wide, which
            // widens every OT payload.  Derived from measured OT cost.
            offline_from_ot: Some(crate::mpcc::ot_cost::OtTriples {
                sacrifice: 2.0,
                extra_share_bits: 64,
            }),
            offline_elems_per_triple: 0.0,
            and_bits: Elems::new(0.0, 2.0),
            auth_overhead: 2.0,
            local_ns_per_lane: 20.0,
            ..SEMI_RING_SHARED
        },
        reference: "Cramer, Damgård, Escudero, Scholl & Xing (CRYPTO'18)",
    },
    ProtocolSpec {
        id: "mascot",
        name: "MASCOT — OT-based SPDZ over a field",
        domain: DomainKind::Field,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dishonest,
        security: Security::Malicious,
        status: Status::Planned("P3"),
        cost: CostParams {
            mul_elems: Elems::new(0.0, 2.0),
            open_elems: Elems::new(0.0, 1.0),
            triples_per_mul: 1.0,
            // MASCOT sacrifices a second triple to authenticate the first.
            offline_from_ot: Some(crate::mpcc::ot_cost::OtTriples {
                sacrifice: 2.0,
                extra_share_bits: 0,
            }),
            offline_elems_per_triple: 0.0,
            field_bits: 128,
            and_bits: Elems::new(0.0, 2.0),
            auth_overhead: 2.0,
            local_ns_per_lane: 20.0,
            ..SEMI_RING_SHARED
        },
        reference: "Keller, Orsini & Scholl (CCS'16)",
    },
    ProtocolSpec {
        id: "lowgear",
        name: "LowGear — SHE-based SPDZ preprocessing",
        domain: DomainKind::Field,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dishonest,
        // The implemented engine is SEMI-HONEST: it has no MACs and no
        // zero-knowledge proofs of plaintext knowledge, so it is the LowGear
        // analogue of Semi2k rather than of SPDZ. Overdrive's malicious
        // security needs ZKPoPK, which `mpcc::lowgear` deliberately omits.
        security: Security::SemiHonest,
        status: Status::Implemented("mpcc::lowgear"),
        cost: CostParams {
            mul_elems: Elems::new(0.0, 2.0),
            open_elems: Elems::new(0.0, 1.0),
            triples_per_mul: 1.0,
            // **Measured** by `baru bench-triples`, not assumed: 192 B/triple
            // at the implemented field below. One element is ceil(61/8) = 8
            // bytes, so 192 B is 24 elements.
            offline_elems_per_triple: 24.0,
            // The field Baru's engine ACTUALLY runs over, not Overdrive's.
            // `mpcc::lowgear`'s noise-flooding budget at q = 2^100 caps the
            // plaintext modulus at t = 114689 (~17 bits); the registry has to
            // describe what is built, or `calibrate` would lower programs for a
            // field the engine cannot evaluate and the comparison would be
            // meaningless. Overdrive uses a 128-bit field with 250-350 bit q at
            // n = 16384-32768 — out of reach while this crate holds ring
            // coefficients in u128.
            //
            // The byte-aligned WIRE width of one element, not the modulus
            // size — the same convention `shamir` uses. LowGear's `t` is a
            // 61-bit prime (`n = 8192`, `q = 2^190`), which occupies 8 bytes,
            // so 64. The cost model prices `field_bits/8` and the VM charges
            // `ceil(order_bits/8)`; declaring the byte-aligned width is what
            // makes those agree, and `mpcc::lowgear` has a test that fails if
            // they ever drift apart.
            field_bits: 64,
            and_bits: Elems::new(0.0, 2.0),
            // No MACs in the implemented engine (see `security` above), so
            // online shares carry no authentication material.
            auth_overhead: 1.0,
            local_ns_per_lane: 40.0,
            ..SEMI_RING_SHARED
        },
        reference: "Keller, Pastro & Rotaru, Overdrive (EUROCRYPT'18)",
    },
    // ── binary ────────────────────────────────────────────────────────────
    ProtocolSpec {
        id: "yao",
        name: "Yao garbled circuits (half-gates)",
        domain: DomainKind::Binary,
        min_parties: 2,
        max_parties: Some(2),
        majority: Majority::Dishonest,
        security: Security::SemiHonest,
        status: Status::Implemented("interpreter::gc_backend"),
        cost: CostParams {
            offline_from_ot: None,
            mul_rounds: 0,
            mul_elems: Elems::new(0.0, 0.0),
            open_rounds: 0,
            open_elems: Elems::new(0.0, 0.0),
            triples_per_mul: 0.0,
            offline_elems_per_triple: 0.0,
            and_rounds: 0,
            // Two 128-bit ciphertexts per AND; XOR is free.
            and_bits: Elems::new(256.0, 0.0),
            edabits_per_bit: 0.0,
            dabits_per_bit: 0.0,
            auth_overhead: 1.0,
            constant_round: true,
            // OT for the evaluator's inputs, table transfer, output decoding.
            constant_rounds: 3,
            field_bits: 0,
            // ~4 fixed-key AES calls to garble + 2 to evaluate one AND gate.
            local_ns_per_lane: 40.0,
        },
        reference: "Zahur, Rosulek & Evans, half-gates (EUROCRYPT'15)",
    },
    ProtocolSpec {
        id: "bmr",
        name: "BMR — constant-round multi-party garbling",
        domain: DomainKind::Binary,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dishonest,
        security: Security::SemiHonest,
        status: Status::Planned("P4"),
        cost: CostParams {
            constant_round: true,
            constant_rounds: 4,
            // Each party garbles its own key: table size grows with n.
            and_bits: Elems::new(0.0, 512.0),
            mul_rounds: 0,
            open_rounds: 0,
            and_rounds: 0,
            local_ns_per_lane: 80.0,
            ..SEMI_RING_SHARED
        },
        reference: "Beaver, Micali & Rogaway (STOC'90)",
    },
    ProtocolSpec {
        id: "semi-bin",
        name: "SemiBin — binary additive sharing",
        domain: DomainKind::Binary,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dishonest,
        security: Security::SemiHonest,
        status: Status::Planned("P4"),
        cost: CostParams {
            and_bits: Elems::new(0.0, 2.0),
            triples_per_mul: 1.0,
            offline_elems_per_triple: 128.0,
            ..SEMI_RING_SHARED
        },
        reference: "GMW with OT-extension triples",
    },
    ProtocolSpec {
        id: "tinier",
        name: "Tinier — authenticated binary sharing",
        domain: DomainKind::Binary,
        min_parties: 2,
        max_parties: None,
        majority: Majority::Dishonest,
        security: Security::Malicious,
        status: Status::Planned("P4"),
        cost: CostParams {
            and_bits: Elems::new(0.0, 2.0),
            triples_per_mul: 1.0,
            offline_elems_per_triple: 400.0,
            auth_overhead: 2.0,
            local_ns_per_lane: 20.0,
            ..SEMI_RING_SHARED
        },
        reference: "Frederiksen, Keller, Orsini & Scholl (ASIACRYPT'15)",
    },
];

/// Look a protocol up by its registry id.
pub fn find(id: &str) -> Option<&'static ProtocolSpec> {
    REGISTRY.iter().find(|p| p.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<&str> = REGISTRY.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate protocol id in REGISTRY");
    }

    /// `Implemented` means Baru can *run a program* under this protocol, which
    /// is exactly [`crate::mpcc::RUNNABLE`] minus the two simulation-only
    /// engines (`dealer` is a lower bound, not a protocol; `shamir` is not in
    /// the registry as an id). If the two lists drift apart, one of them is
    /// lying to a report.
    #[test]
    fn implemented_protocols_match_the_backends_baru_actually_has() {
        let implemented: Vec<&str> = REGISTRY
            .iter()
            .filter(|p| p.status.is_implemented())
            .map(|p| p.id)
            .collect();
        assert_eq!(implemented, vec!["aby3", "semi2k", "lowgear", "yao"]);
        // Everything claiming to be implemented must actually be executable.
        for id in &implemented {
            if *id == "aby3" || *id == "yao" {
                continue; // run through the interpreter, not the mpcc VM
            }
            assert!(
                crate::mpcc::RUNNABLE.contains(id),
                "{id} claims Implemented but mpcc::RUNNABLE cannot execute it"
            );
        }
    }

    #[test]
    fn party_bounds() {
        let aby3 = find("aby3").unwrap();
        assert!(aby3.supports_parties(3));
        assert!(!aby3.supports_parties(2));
        assert!(!aby3.supports_parties(4));
        let shamir = find("shamir").unwrap();
        assert!(shamir.supports_parties(7));
    }

    #[test]
    fn volume_scales_with_peers_only_where_the_protocol_says_so() {
        // Replicated: one neighbour, regardless of n.
        assert_eq!(find("aby3").unwrap().cost.mul_elems.total(3), 1.0);
        // Shamir: every peer.
        assert_eq!(find("shamir").unwrap().cost.mul_elems.total(5), 4.0);
    }

    #[test]
    fn security_is_ordered_so_stronger_satisfies_weaker() {
        assert!(Security::Malicious > Security::SemiHonest);
        assert_eq!(Security::parse("active"), Some(Security::Malicious));
    }
}
