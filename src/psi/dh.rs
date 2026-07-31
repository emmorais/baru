//! Diffie–Hellman PSI over Ristretto255 — semi-honest, one-sided output.
//!
//! # The protocol
//!
//! `H` hashes an item to a curve point; `α`, `β` are the parties' secret
//! exponents. The sender's set is shuffled before it leaves.
//!
//! ```text
//! Sender    A  = shuffle { H(a)^α  : a ∈ A }              ──A──▶
//! Receiver  B  = { H(b)^β : b ∈ B }, in order             ◀──B──
//! Sender    AB = { p^α : p ∈ B }, in B's order            ──AB─▶
//! Receiver  computes { p^β : p ∈ A } and intersects it
//!           against AB, which is keyed to its own items
//! ```
//!
//! Both sides end up holding `H(x)^{αβ}` for their own items, and that value is
//! equal exactly when the items are equal — so the receiver can match without
//! either party ever seeing a value it could invert. Commutativity of
//! exponentiation is the whole mechanism.
//!
//! The receiver learns which of *its* items are shared. The sender learns
//! nothing: it only ever sees `H(b)^β`, uniform under DDH.
//!
//! # Cost shape
//!
//! Three flights of `n` compressed points, so **96 bytes per item** at 32 bytes
//! a point, and `2n` scalar multiplications per party. That second number is the
//! interesting one — a scalar multiplication is tens of microseconds, so DH-PSI
//! is *compute*-bound, where an OT-based PSI is symmetric-key throughout and
//! trades that for more bytes. Which wins is a property of the deployment, and
//! [`crate::bench::psi`] measures it against the OT figures in
//! [`crate::mpcc::ot_cost`].
//!
//! The final flights could be truncated to 16 bytes (a 128-bit collision bound
//! is ample for set sizes here) to cut a third off the traffic. Not done, so the
//! reported bytes are the untruncated construction rather than a best case.

use curve25519_dalek::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use rand::seq::SliceRandom;
use rand::Rng;
use sha2::{Digest, Sha512};
use std::collections::HashSet;

/// A compressed Ristretto point.
pub const POINT_BYTES: usize = 32;
const DOMAIN: &[u8] = b"baru/psi/dh/v1";

/// Hash an item to a curve point.
///
/// `from_uniform_bytes` on a 64-byte SHA-512 digest is exactly what
/// `RistrettoPoint::hash_from_bytes` does, spelled out so the `digest` feature is
/// not needed. It matters that this is a hash *to the group* and not a hash to a
/// scalar: hashing to a scalar and multiplying the basepoint would make
/// `H(x) = g^h(x)` with `h(x)` computable, and the receiver could then test
/// candidate items against `A` directly.
fn hash_to_point(item: &[u8]) -> RistrettoPoint {
    let mut h = Sha512::new();
    h.update(DOMAIN);
    h.update((item.len() as u64).to_le_bytes());
    h.update(item);
    let wide: [u8; 64] = h.finalize().into();
    RistrettoPoint::from_uniform_bytes(&wide)
}

fn random_scalar(rng: &mut impl Rng) -> Scalar {
    let mut wide = [0u8; 64];
    rng.fill_bytes(&mut wide);
    Scalar::from_bytes_mod_order_wide(&wide)
}

fn compress(points: &[RistrettoPoint]) -> Vec<[u8; POINT_BYTES]> {
    points.iter().map(|p| p.compress().to_bytes()).collect()
}

fn decompress(bytes: &[[u8; POINT_BYTES]]) -> Result<Vec<RistrettoPoint>, String> {
    bytes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            CompressedRistretto(*b)
                .decompress()
                .ok_or_else(|| format!("item {i} is not a valid Ristretto point"))
        })
        .collect()
}

/// One flight of points.
#[derive(Debug, Clone)]
pub struct Flight(pub Vec<[u8; POINT_BYTES]>);

impl Flight {
    pub fn bytes(&self) -> usize {
        self.0.len() * POINT_BYTES
    }
}

// ─── Proof of a consistent exponent ──────────────────────────────────────────

/// A batched Chaum–Pedersen proof that the sender raised **every** item of the
/// receiver's flight to the *same* exponent.
///
/// # What this does and does not defend against
///
/// DH-PSI here is one-sided: the receiver learns the intersection and the
/// sender learns nothing at all. So a deviating sender cannot break the
/// receiver's *privacy* — it can only corrupt the receiver's *output*, by
/// raising different items to different exponents so that chosen items fail to
/// match.
///
/// It is worth being precise about how much that matters, because it is easy to
/// overstate. A sender that uses inconsistent exponents on **its own** flight
/// `A` is doing something indistinguishable from submitting a smaller set, and
/// input substitution is unavoidable in any PSI — no proof can stop a party
/// lying about what it holds. That case therefore needs no defence.
///
/// The case that does need one is the **response**: `AB_i = B_i^α`. There the
/// sender is operating on the *receiver's* items, and inconsistency silently
/// removes chosen elements from the receiver's answer. This proof closes
/// exactly that, and the module claims nothing beyond it.
///
/// Cost is `O(1)` in set size — one commitment pair and one scalar for the
/// whole flight, not per item.
#[derive(Debug, Clone, Copy)]
pub struct ExponentProof {
    t1: [u8; POINT_BYTES],
    t2: [u8; POINT_BYTES],
    z: [u8; 32],
}

impl ExponentProof {
    pub fn bytes(&self) -> usize {
        2 * POINT_BYTES + 32
    }
}

/// Per-item weights for the batched check, bound to the whole transcript so the
/// sender cannot choose its deviation after seeing them.
fn batch_weights(b: &Flight, ab: &Flight) -> Vec<Scalar> {
    use sha2::{Digest, Sha256};
    let mut seed = Sha256::new();
    seed.update(b"baru/psi/dh/batch/v1");
    for p in &b.0 {
        seed.update(p);
    }
    for p in &ab.0 {
        seed.update(p);
    }
    let seed = seed.finalize();
    (0..b.0.len())
        .map(|i| {
            let mut h = Sha256::new();
            h.update(seed);
            h.update((i as u64).to_le_bytes());
            let d = h.finalize();
            let mut wide = [0u8; 64];
            wide[..32].copy_from_slice(&d);
            Scalar::from_bytes_mod_order_wide(&wide)
        })
        .collect()
}

fn dleq_challenge(
    s: &RistrettoPoint,
    p: &RistrettoPoint,
    q: &RistrettoPoint,
    t1: &RistrettoPoint,
    t2: &RistrettoPoint,
) -> Scalar {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"baru/psi/dh/dleq/v1");
    for pt in [s, p, q, t1, t2] {
        h.update(pt.compress().as_bytes());
    }
    let d = h.finalize();
    let mut wide = [0u8; 64];
    wide[..32].copy_from_slice(&d);
    Scalar::from_bytes_mod_order_wide(&wide)
}

/// Verify that `ab` is `b` raised uniformly to the exponent committed in `s`.
pub fn verify_exponent(
    s_commitment: &[u8; POINT_BYTES],
    b: &Flight,
    ab: &Flight,
    proof: &ExponentProof,
) -> Result<(), String> {
    if b.0.len() != ab.0.len() {
        return Err("response length does not match the challenge flight".to_string());
    }
    let s = CompressedRistretto(*s_commitment)
        .decompress()
        .ok_or("exponent commitment is not a valid point")?;
    let t1 = CompressedRistretto(proof.t1)
        .decompress()
        .ok_or("proof commitment t1 is invalid")?;
    let t2 = CompressedRistretto(proof.t2)
        .decompress()
        .ok_or("proof commitment t2 is invalid")?;
    let z = Option::<Scalar>::from(Scalar::from_canonical_bytes(proof.z))
        .ok_or("proof scalar is not canonical")?;

    let bs = decompress(&b.0)?;
    let abs = decompress(&ab.0)?;
    let w = batch_weights(b, ab);
    let p: RistrettoPoint = bs.iter().zip(&w).map(|(pt, wi)| pt * wi).sum();
    let q: RistrettoPoint = abs.iter().zip(&w).map(|(pt, wi)| pt * wi).sum();

    let e = dleq_challenge(&s, &p, &q, &t1, &t2);
    if RISTRETTO_BASEPOINT_TABLE * &z == t1 + s * e && p * z == t2 + q * e {
        Ok(())
    } else {
        Err("sender used inconsistent exponents across items".to_string())
    }
}

/// The party that holds `A` and learns nothing.
pub struct Sender {
    alpha: Scalar,
}

impl Sender {
    /// `A = shuffle { H(a)^α }`.
    pub fn new(items: &[Vec<u8>], rng: &mut impl Rng) -> (Self, Flight) {
        let alpha = random_scalar(rng);
        let mut blinded: Vec<RistrettoPoint> =
            items.iter().map(|a| hash_to_point(a) * alpha).collect();
        // Shuffle so the receiver cannot tie a position to one of our items.
        blinded.shuffle(rng);
        (Sender { alpha }, Flight(compress(&blinded)))
    }

    /// Raise the receiver's blinded items to `α`, preserving their order so the
    /// receiver can match them back to its own set.
    pub fn respond(&self, b: &Flight) -> Result<Flight, String> {
        let points = decompress(&b.0)?;
        let doubled: Vec<RistrettoPoint> = points.iter().map(|p| p * self.alpha).collect();
        Ok(Flight(compress(&doubled)))
    }

    /// Public commitment `S = G^α` to the exponent this sender uses. Ship it
    /// once, alongside the first flight; [`verify_exponent`] checks the
    /// response against it.
    pub fn exponent_commitment(&self) -> [u8; POINT_BYTES] {
        (RISTRETTO_BASEPOINT_TABLE * &self.alpha).compress().to_bytes()
    }

    /// [`Sender::respond`] plus a proof that every item used the same `α`.
    pub fn respond_proved(
        &self,
        b: &Flight,
        rng: &mut impl Rng,
    ) -> Result<(Flight, ExponentProof), String> {
        let ab = self.respond(b)?;
        let bs = decompress(&b.0)?;
        let abs = decompress(&ab.0)?;
        let w = batch_weights(b, &ab);
        let p: RistrettoPoint = bs.iter().zip(&w).map(|(pt, wi)| pt * wi).sum();
        let q: RistrettoPoint = abs.iter().zip(&w).map(|(pt, wi)| pt * wi).sum();

        let k = random_scalar(rng);
        let t1 = RISTRETTO_BASEPOINT_TABLE * &k;
        let t2 = p * k;
        let s = RISTRETTO_BASEPOINT_TABLE * &self.alpha;
        let e = dleq_challenge(&s, &p, &q, &t1, &t2);
        let z = k + e * self.alpha;
        Ok((
            ab,
            ExponentProof {
                t1: t1.compress().to_bytes(),
                t2: t2.compress().to_bytes(),
                z: z.to_bytes(),
            },
        ))
    }
}

/// The party that holds `B` and learns the intersection.
pub struct Receiver {
    beta: Scalar,
    items: Vec<Vec<u8>>,
}

impl Receiver {
    /// `B = { H(b)^β }`, in order.
    pub fn new(items: &[Vec<u8>], rng: &mut impl Rng) -> (Self, Flight) {
        let beta = random_scalar(rng);
        let blinded: Vec<RistrettoPoint> = items.iter().map(|b| hash_to_point(b) * beta).collect();
        (
            Receiver {
                beta,
                items: items.to_vec(),
            },
            Flight(compress(&blinded)),
        )
    }

    /// Intersect: raise the sender's set to `β` and look up our own
    /// doubly-blinded items in it.
    pub fn intersect(&self, a: &Flight, ab: &Flight) -> Result<Vec<Vec<u8>>, String> {
        if ab.0.len() != self.items.len() {
            return Err(format!(
                "response has {} points for {} items",
                ab.0.len(),
                self.items.len()
            ));
        }
        let theirs: HashSet<[u8; POINT_BYTES]> = decompress(&a.0)?
            .iter()
            .map(|p| (p * self.beta).compress().to_bytes())
            .collect();
        Ok(self
            .items
            .iter()
            .zip(&ab.0)
            .filter(|(_, mine)| theirs.contains(*mine))
            .map(|(item, _)| item.clone())
            .collect())
    }
}

#[cfg(test)]
mod exponent_proof_tests {
    use super::*;

    fn items(n: usize, tag: u8) -> Vec<Vec<u8>> {
        (0..n).map(|i| vec![tag, i as u8]).collect()
    }

    #[test]
    fn an_honest_response_proves_a_consistent_exponent() {
        let mut rng = rand::thread_rng();
        let (sender, _a) = Sender::new(&items(8, 1), &mut rng);
        let (_receiver, b) = Receiver::new(&items(6, 1), &mut rng);
        let (ab, proof) = sender.respond_proved(&b, &mut rng).expect("respond");
        assert!(verify_exponent(&sender.exponent_commitment(), &b, &ab, &proof).is_ok());
    }

    /// The attack the proof exists to stop: raise one of the receiver's items to
    /// a different exponent so it silently drops out of the intersection.
    #[test]
    fn a_single_inconsistent_item_is_caught() {
        let mut rng = rand::thread_rng();
        let (sender, _a) = Sender::new(&items(8, 1), &mut rng);
        let (_receiver, b) = Receiver::new(&items(6, 1), &mut rng);
        let (mut ab, proof) = sender.respond_proved(&b, &mut rng).expect("respond");

        // Replace one response point with the same item under a different
        // exponent — exactly "use α' for item 3".
        let rogue = random_scalar(&mut rng);
        let bs = decompress(&b.0).expect("decompress");
        ab.0[3] = (bs[3] * rogue).compress().to_bytes();

        assert!(
            verify_exponent(&sender.exponent_commitment(), &b, &ab, &proof).is_err(),
            "an inconsistent exponent must be caught"
        );
    }

    #[test]
    fn a_proof_does_not_transfer_to_a_different_commitment() {
        let mut rng = rand::thread_rng();
        let (sender, _a) = Sender::new(&items(4, 1), &mut rng);
        let (other, _a2) = Sender::new(&items(4, 2), &mut rng);
        let (_receiver, b) = Receiver::new(&items(4, 1), &mut rng);
        let (ab, proof) = sender.respond_proved(&b, &mut rng).expect("respond");
        assert!(verify_exponent(&other.exponent_commitment(), &b, &ab, &proof).is_err());
    }

    #[test]
    fn the_proved_response_still_computes_the_right_intersection() {
        let mut rng = rand::thread_rng();
        let a_items = items(8, 1);
        let b_items: Vec<Vec<u8>> = items(4, 1).into_iter().chain(items(3, 9)).collect();
        let (sender, a) = Sender::new(&a_items, &mut rng);
        let (receiver, b) = Receiver::new(&b_items, &mut rng);
        let (ab, proof) = sender.respond_proved(&b, &mut rng).expect("respond");
        verify_exponent(&sender.exponent_commitment(), &b, &ab, &proof).expect("proof");
        let got = receiver.intersect(&a, &ab).expect("intersect");
        assert_eq!(got, items(4, 1), "proving must not change the result");
    }
}

/// Bytes moved, by flight.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PsiTraffic {
    pub sender_set: usize,
    pub receiver_set: usize,
    pub response: usize,
    pub rounds: usize,
}

impl PsiTraffic {
    pub fn total(&self) -> usize {
        self.sender_set + self.receiver_set + self.response
    }
}

/// DH-PSI as a [`crate::psi::PsiProtocol`].
pub struct DhPsi;

impl crate::psi::PsiProtocol for DhPsi {
    fn name(&self) -> &'static str {
        "dh-psi"
    }
    fn hiding(&self) -> crate::psi::Hiding {
        crate::psi::Hiding::NonSharedItems
    }
    fn intersect(&self, a: &[Vec<u8>], b: &[Vec<u8>]) -> Result<crate::psi::Outcome, String> {
        let (intersection, traffic) = simulate(a, b)?;
        Ok(crate::psi::Outcome {
            intersection,
            cost: crate::psi::Cost {
                bytes: traffic.total(),
                rounds: traffic.rounds,
            },
        })
    }
}

/// Run the protocol in process. Returns the receiver's intersection.
pub fn simulate(a: &[Vec<u8>], b: &[Vec<u8>]) -> Result<(Vec<Vec<u8>>, PsiTraffic), String> {
    let mut rng = rand::thread_rng();
    let (sender, a_flight) = Sender::new(a, &mut rng);
    let (receiver, b_flight) = Receiver::new(b, &mut rng);
    let ab = sender.respond(&b_flight)?;
    let out = receiver.intersect(&a_flight, &ab)?;
    Ok((
        out,
        PsiTraffic {
            sender_set: a_flight.bytes(),
            receiver_set: b_flight.bytes(),
            response: ab.bytes(),
            rounds: 2,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psi::plain_intersect;

    fn items(prefix: &str, range: std::ops::Range<usize>) -> Vec<Vec<u8>> {
        range.map(|i| format!("{prefix}{i}").into_bytes()).collect()
    }

    /// The protocol must compute exactly the plaintext intersection.
    #[test]
    fn it_computes_the_intersection() {
        let a = items("patient-", 0..40);
        let b = items("patient-", 25..60);
        let (got, traffic) = simulate(&a, &b).unwrap();
        let want = plain_intersect(&a, &b);
        assert_eq!(got, want, "intersection mismatch");
        assert_eq!(got.len(), 15);
        // The sender's set costs |A| points; the receiver pays twice over, once
        // for its own flight and once for the response keyed to it.
        assert_eq!(traffic.sender_set, a.len() * POINT_BYTES);
        assert_eq!(traffic.receiver_set, b.len() * POINT_BYTES);
        assert_eq!(traffic.response, b.len() * POINT_BYTES);
        assert_eq!(traffic.total(), (a.len() + 2 * b.len()) * POINT_BYTES);
        assert_eq!(traffic.rounds, 2);
    }

    #[test]
    fn disjoint_sets_yield_nothing_and_identical_sets_yield_everything() {
        let a = items("a", 0..12);
        let b = items("b", 0..12);
        assert!(simulate(&a, &b).unwrap().0.is_empty());
        assert_eq!(simulate(&a, &a).unwrap().0.len(), 12);
    }

    #[test]
    fn asymmetric_set_sizes_work() {
        let a = items("p", 0..100);
        let b = items("p", 90..95);
        let (got, _) = simulate(&a, &b).unwrap();
        assert_eq!(got, plain_intersect(&a, &b));
        assert_eq!(got.len(), 5);
    }

    /// The sender's flight must be shuffled: position must not track its input
    /// order, or the receiver would learn which of the sender's items matched.
    #[test]
    fn the_senders_set_is_shuffled() {
        let a = items("p", 0..64);
        let mut rng = rand::thread_rng();
        // Compare against the unshuffled blinding under the same key by
        // rebuilding it: with 64 items, a chance-identical order is 1/64!.
        let (sender, flight) = Sender::new(&a, &mut rng);
        let in_order: Vec<[u8; POINT_BYTES]> = a
            .iter()
            .map(|x| (hash_to_point(x) * sender.alpha).compress().to_bytes())
            .collect();
        assert_ne!(flight.0, in_order, "the sender's set must be shuffled");
        // ...but it must be a permutation of it, not different values.
        let sent: HashSet<_> = flight.0.iter().collect();
        let expected: HashSet<_> = in_order.iter().collect();
        assert_eq!(sent, expected, "shuffling must not change the multiset");
    }

    /// Two runs must not produce equal flights, or a server watching repeated
    /// queries could link them even without inverting anything.
    #[test]
    fn blinding_is_fresh_per_run() {
        let a = items("p", 0..8);
        let mut rng = rand::thread_rng();
        let (_, f1) = Sender::new(&a, &mut rng);
        let (_, f2) = Sender::new(&a, &mut rng);
        let s1: HashSet<_> = f1.0.iter().collect();
        let s2: HashSet<_> = f2.0.iter().collect();
        assert_ne!(s1, s2, "each run must use a fresh exponent");
    }

    /// Hashing must land in the group, not on a discrete log the receiver could
    /// compute — different items must give unrelated points, and the same item
    /// must be stable.
    #[test]
    fn hash_to_point_is_deterministic_and_separating() {
        assert_eq!(hash_to_point(b"x"), hash_to_point(b"x"));
        assert_ne!(hash_to_point(b"x"), hash_to_point(b"y"));
        // Length-prefixed, so concatenation is not confusable.
        assert_ne!(hash_to_point(b"ab"), hash_to_point(b"a"));
    }

    #[test]
    fn malformed_flights_are_refused() {
        let a = items("p", 0..4);
        let mut rng = rand::thread_rng();
        let (sender, _) = Sender::new(&a, &mut rng);
        let bad = Flight(vec![[0xffu8; POINT_BYTES]]);
        assert!(sender.respond(&bad).is_err());
        let (receiver, _) = Receiver::new(&a, &mut rng);
        assert!(receiver
            .intersect(&bad, &Flight(vec![[0u8; POINT_BYTES]; 4]))
            .is_err());
    }
}
