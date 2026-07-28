//! **Private set intersection** — which patients appear at both hospitals,
//! without either hospital learning anything else.
//!
//! Stage 3 of the healthcare workflow in `docs/research-agenda.md`. Two
//! institutions hold patient identifier sets; they want `A ∩ B` and nothing
//! more. The obvious approaches are both wrong:
//!
//! * Exchange the raw sets — discloses everything.
//! * Exchange **hashes** of the sets — [`naive_hash`], and the one people
//!   actually ship. Patient identifiers are low-entropy (names, dates, national
//!   IDs with check digits), so the receiver just enumerates the space and
//!   inverts the hashes offline. It is not PSI; it is obfuscation.
//!
//! # What is implemented: DH-based PSI
//!
//! [`dh`] is the Meadows / Huberman–Franklin–Neuman construction over
//! Ristretto255. It needs no oblivious transfer, which is the point of building
//! it first: the research agenda assumed "PSI (from OT)", and this is the
//! baseline that assumption has to beat. Whether it does is a cost question, and
//! [`crate::bench::psi`] answers it with the measured OT numbers from
//! [`crate::ot`] rather than by assertion.
//!
//! # Security, stated precisely
//!
//! * **Semi-honest**, and one-sided: only the receiver learns the intersection.
//! * Rests on **DDH** in the random-oracle model — items are hashed to curve
//!   points, so the hash is doing real work and is not a formality.
//! * **Set sizes leak.** Both parties learn `|A|` and `|B|`. Padding to a public
//!   bound is the mitigation and is a deployment decision, not this module's.
//! * A malicious party can use inconsistent exponents across items, or feed a
//!   crafted set to probe membership. Nothing here detects that.
//! * The receiver learns the intersection *as a subset of its own items*, which
//!   is exactly the intended output, and the sender's set is shuffled so
//!   position carries no information.

use std::collections::HashSet;

pub mod dh;

/// What a PSI construction actually hides, so a comparison table cannot quietly
/// put a secure and an insecure protocol in the same column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Hiding {
    /// Hides everything but the set sizes and the intersection itself.
    NonSharedItems,
    /// Hides nothing useful: the digests invert under an offline dictionary
    /// attack, so the receiver recovers the sender's whole set.
    Nothing,
}

/// Bytes and rounds a run cost — measured, not estimated, because every
/// implementation builds explicit messages and sums their serialized sizes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Cost {
    pub bytes: usize,
    pub rounds: usize,
}

/// One protocol run.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// The receiver's view of `A ∩ B`, as a subset of its own items.
    pub intersection: Vec<Vec<u8>>,
    pub cost: Cost,
}

/// A private-set-intersection construction.
///
/// The point of the trait is that [`crate::bench::psi`] iterates over
/// implementations rather than naming one, so adding a construction — a
/// KKRT-style OT-based PSI, a circuit PSI — makes it appear in the comparison
/// without touching the benchmark. Mirrors the pattern the rest of the tree
/// already uses for backends ([`crate::fhe::FheBackend`],
/// [`crate::mpcc::vm::ShareEngine`], [`crate::db::ResultStore`]).
///
/// `intersect` runs both parties in process. That is deliberate: it keeps the
/// message sizes in [`Cost`] honest (they are serialized sizes of real messages)
/// without a transport in the way. A networked driver would implement the same
/// trait over the same message types.
pub trait PsiProtocol {
    fn name(&self) -> &'static str;

    /// What the construction hides. Stated in the interface so it travels with
    /// the numbers.
    fn hiding(&self) -> Hiding;

    /// Receiver-side output: which of `b`'s items are also in `a`.
    fn intersect(&self, a: &[Vec<u8>], b: &[Vec<u8>]) -> Result<Outcome, String>;
}

/// Every construction implemented today, in the order a comparison should read.
pub fn all() -> Vec<Box<dyn PsiProtocol>> {
    vec![Box::new(NaiveHash), Box::new(dh::DhPsi)]
}

/// The insecure baseline as a [`PsiProtocol`], so it appears in comparisons
/// with its `hiding()` attached rather than as a bare number.
pub struct NaiveHash;

impl PsiProtocol for NaiveHash {
    fn name(&self) -> &'static str {
        "naive-hash"
    }
    fn hiding(&self) -> Hiding {
        Hiding::Nothing
    }
    fn intersect(&self, a: &[Vec<u8>], b: &[Vec<u8>]) -> Result<Outcome, String> {
        Ok(Outcome {
            intersection: naive_hash(a, b),
            // One digest per item each way, one flight.
            cost: Cost {
                bytes: (a.len() + b.len()) * 32,
                rounds: 1,
            },
        })
    }
}

/// The insecure baseline: exchange `SHA-256(item)` and intersect.
///
/// Included because it is the comparison that matters. It is *cheap* — one hash
/// per item, 32 bytes per item on the wire, no public-key operations at all — so
/// any real PSI has to justify its cost against this, and the justification is
/// entirely about security rather than performance.
///
/// The break: the hash is unkeyed and deterministic, so a receiver holding the
/// digests can hash every candidate identifier offline and recover the sender's
/// whole set. For a 10-digit national ID that is 10¹⁰ hashes — minutes on a GPU.
/// The construction leaks the sender's *entire set*, not just the intersection.
pub fn naive_hash(a: &[Vec<u8>], b: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use sha2::{Digest, Sha256};
    let digest = |x: &Vec<u8>| -> [u8; 32] { Sha256::digest(x).into() };
    let set: HashSet<[u8; 32]> = a.iter().map(digest).collect();
    b.iter()
        .filter(|x| set.contains(&digest(x)))
        .cloned()
        .collect()
}

/// Plaintext intersection, preserving `b`'s order — the correctness oracle every
/// protocol here is checked against.
pub fn plain_intersect(a: &[Vec<u8>], b: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let set: HashSet<&[u8]> = a.iter().map(|x| x.as_slice()).collect();
    b.iter()
        .filter(|x| set.contains(x.as_slice()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(prefix: &str, range: std::ops::Range<usize>) -> Vec<Vec<u8>> {
        range.map(|i| format!("{prefix}{i}").into_bytes()).collect()
    }

    #[test]
    fn the_oracle_and_the_naive_baseline_agree() {
        let a = items("patient-", 0..20);
        let b = items("patient-", 10..30);
        let want = plain_intersect(&a, &b);
        assert_eq!(want.len(), 10);
        assert_eq!(naive_hash(&a, &b), want);
    }

    /// One property against every construction — including the insecure one,
    /// which must still be *correct*; its problem is what it leaks, not what it
    /// computes.
    #[test]
    fn every_construction_computes_the_intersection() {
        let a = items("patient-", 0..50);
        let b = items("patient-", 30..80);
        let want = plain_intersect(&a, &b);
        for proto in all() {
            let out = proto
                .intersect(&a, &b)
                .unwrap_or_else(|e| panic!("{}: {e}", proto.name()));
            assert_eq!(out.intersection, want, "{}", proto.name());
            assert!(
                out.cost.bytes > 0 && out.cost.rounds > 0,
                "{}",
                proto.name()
            );
        }
    }

    /// The security label has to travel with the numbers, or a comparison table
    /// silently puts a broken protocol in the same column as a real one.
    #[test]
    fn the_insecure_baseline_is_labelled_as_such() {
        let labels: Vec<(&str, Hiding)> = all().iter().map(|p| (p.name(), p.hiding())).collect();
        assert!(labels.contains(&("naive-hash", Hiding::Nothing)));
        assert!(labels.contains(&("dh-psi", Hiding::NonSharedItems)));
    }

    #[test]
    fn disjoint_and_identical_sets_are_handled() {
        let a = items("x", 0..5);
        let b = items("y", 0..5);
        assert!(plain_intersect(&a, &b).is_empty());
        assert_eq!(plain_intersect(&a, &a).len(), 5);
    }
}
