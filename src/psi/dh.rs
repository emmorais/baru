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
