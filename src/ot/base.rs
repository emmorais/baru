//! Chou–Orlandi "simplest OT" over Ristretto255 — 1-out-of-2, semi-honest.
//!
//! One round each way, two scalar multiplications per party per OT, and no
//! trusted setup. It is the standard way to bootstrap [`super::iknp`], which
//! needs exactly `κ` of these and then never uses public-key crypto again.
//!
//! # The protocol
//!
//! With generator `G` and hash `H`:
//!
//! ```text
//! Sender    a ← Zq,  S = aG                              ──S──▶
//! Receiver  b ← Zq,  R = bG + c·S      (c = choice bit)  ◀──R──
//! Sender    k0 = H(aR),  k1 = H(a(R − S))
//! Receiver  kc = H(bS)
//! ```
//!
//! It works because `aR = a(bG + c·aG) = abG + c·a²G` while `bS = abG`, so at
//! `c = 0` the sender's `k0` and the receiver's `kc` are both `H(abG)`, and at
//! `c = 1` the `a(R − S) = abG` branch matches instead. The receiver cannot
//! compute the other key without `a`, and the sender cannot tell which branch
//! matched because `R` is uniform either way — `bG` masks `c·S` perfectly.
//!
//! Messages are then one-time padded: `e0 = m0 ⊕ k0`, `e1 = m1 ⊕ k1`.
//!
//! # Security notes, stated rather than implied
//!
//! * **Semi-honest.** The receiver's `R` is uniform regardless of `c`, so sender
//!   privacy of the *choice* holds even against a malicious receiver; but this
//!   implementation does no proof of knowledge, so a malicious sender can send a
//!   low-order or otherwise malformed `S`. Ristretto is the mitigation that
//!   matters here — its encoding is canonical and cofactor-free, so there are no
//!   small-subgroup points to send in the first place.
//! * **Randomness.** Scalars are drawn as 64 uniform bytes reduced mod `q`
//!   (`from_bytes_mod_order_wide`), which is unbiased, rather than by rejection
//!   sampling on 32.
//! * `a` is reused across a batch of OTs, which is what makes a batch cheap;
//!   the per-index hash domain separation in [`super::hash_block`] is what keeps
//!   the derived keys independent.

use curve25519_dalek::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use rand::Rng;

use super::{hash_block, xor_block, Block, OtTraffic};

const DOMAIN: &[u8] = b"baru/ot/base/v1";
/// A compressed Ristretto point.
pub const POINT_BYTES: usize = 32;

fn random_scalar(rng: &mut impl Rng) -> Scalar {
    let mut wide = [0u8; 64];
    rng.fill_bytes(&mut wide);
    Scalar::from_bytes_mod_order_wide(&wide)
}

/// The sender's first message: `S = aG`.
#[derive(Debug, Clone, Copy)]
pub struct Setup(pub [u8; POINT_BYTES]);

/// The receiver's reply: one point per OT.
#[derive(Debug, Clone)]
pub struct Choices(pub Vec<[u8; POINT_BYTES]>);

/// The sender's payload: the two padded messages per OT.
#[derive(Debug, Clone)]
pub struct Payload(pub Vec<(Block, Block)>);

impl Setup {
    pub fn bytes(&self) -> usize {
        POINT_BYTES
    }
}
impl Choices {
    pub fn bytes(&self) -> usize {
        self.0.len() * POINT_BYTES
    }
}
impl Payload {
    pub fn bytes(&self) -> usize {
        self.0.len() * 2 * super::KAPPA_BYTES
    }
}

/// Sender state between the two rounds.
pub struct Sender {
    a: Scalar,
    s_point: RistrettoPoint,
}

impl Sender {
    pub fn new(rng: &mut impl Rng) -> (Self, Setup) {
        let a = random_scalar(rng);
        let s_point = RISTRETTO_BASEPOINT_TABLE * &a;
        let setup = Setup(s_point.compress().to_bytes());
        (Sender { a, s_point }, setup)
    }

    /// Derive the two keys per OT from the receiver's points.
    pub fn keys(&self, choices: &Choices) -> Result<Vec<(Block, Block)>, String> {
        choices
            .0
            .iter()
            .enumerate()
            .map(|(i, r_bytes)| {
                let r = CompressedRistretto(*r_bytes).decompress().ok_or_else(|| {
                    format!("OT {i}: receiver point is not a valid Ristretto point")
                })?;
                let k0 = hash_block(DOMAIN, i, (self.a * r).compress().as_bytes());
                let k1 = hash_block(
                    DOMAIN,
                    i,
                    (self.a * (r - self.s_point)).compress().as_bytes(),
                );
                Ok((k0, k1))
            })
            .collect()
    }

    /// One-time-pad the message pairs under those keys.
    pub fn send(&self, choices: &Choices, messages: &[(Block, Block)]) -> Result<Payload, String> {
        let keys = self.keys(choices)?;
        if keys.len() != messages.len() {
            return Err(format!(
                "{} choice points for {} message pairs",
                keys.len(),
                messages.len()
            ));
        }
        Ok(Payload(
            keys.iter()
                .zip(messages)
                .map(|((k0, k1), (m0, m1))| (xor_block(k0, m0), xor_block(k1, m1)))
                .collect(),
        ))
    }
}

/// Receiver state between the two rounds.
pub struct Receiver {
    keys: Vec<Block>,
    choices: Vec<bool>,
}

impl Receiver {
    /// Pick `R = bG + c·S` per OT and remember `H(bS)`.
    pub fn new(
        setup: &Setup,
        choices: &[bool],
        rng: &mut impl Rng,
    ) -> Result<(Self, Choices), String> {
        let s_point = CompressedRistretto(setup.0)
            .decompress()
            .ok_or("sender setup is not a valid Ristretto point")?;
        let mut points = Vec::with_capacity(choices.len());
        let mut keys = Vec::with_capacity(choices.len());
        for (i, &c) in choices.iter().enumerate() {
            let b = random_scalar(rng);
            let r = if c {
                RISTRETTO_BASEPOINT_TABLE * &b + s_point
            } else {
                RISTRETTO_BASEPOINT_TABLE * &b
            };
            points.push(r.compress().to_bytes());
            keys.push(hash_block(DOMAIN, i, (b * s_point).compress().as_bytes()));
        }
        Ok((
            Receiver {
                keys,
                choices: choices.to_vec(),
            },
            Choices(points),
        ))
    }

    /// The keys the receiver holds — what [`super::iknp`] consumes directly,
    /// without ever padding a message.
    pub fn keys(&self) -> &[Block] {
        &self.keys
    }

    /// Unpad the chosen message from each pair.
    pub fn receive(&self, payload: &Payload) -> Result<Vec<Block>, String> {
        if payload.0.len() != self.keys.len() {
            return Err(format!(
                "payload has {} pairs, expected {}",
                payload.0.len(),
                self.keys.len()
            ));
        }
        Ok(payload
            .0
            .iter()
            .zip(&self.keys)
            .zip(&self.choices)
            .map(|(((e0, e1), k), &c)| xor_block(k, if c { e1 } else { e0 }))
            .collect())
    }
}

/// Run a batch of base OTs in process. Returns what the receiver got and the
/// traffic it took.
pub fn simulate(
    choices: &[bool],
    messages: &[(Block, Block)],
) -> Result<(Vec<Block>, OtTraffic), String> {
    let mut rng = rand::thread_rng();
    let (sender, setup) = Sender::new(&mut rng);
    let (receiver, chosen) = Receiver::new(&setup, choices, &mut rng)?;
    let payload = sender.send(&chosen, messages)?;
    let got = receiver.receive(&payload)?;
    Ok((
        got,
        OtTraffic {
            sender_to_receiver: setup.bytes() + payload.bytes(),
            receiver_to_sender: chosen.bytes(),
            // S, then R, then the payload: two round trips.
            rounds: 2,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ot::random_block;

    fn pairs(n: usize) -> Vec<(Block, Block)> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (random_block(&mut rng), random_block(&mut rng)))
            .collect()
    }

    /// The defining property: the receiver gets exactly the message it chose.
    #[test]
    fn the_receiver_gets_the_chosen_message() {
        let choices = vec![false, true, true, false, true];
        let msgs = pairs(choices.len());
        let (got, traffic) = simulate(&choices, &msgs).unwrap();
        for (i, (&c, m)) in choices.iter().zip(&msgs).enumerate() {
            let want = if c { m.1 } else { m.0 };
            assert_eq!(got[i], want, "OT {i} returned the wrong message");
        }
        assert_eq!(traffic.rounds, 2);
        assert!(traffic.total() > 0);
    }

    /// And it must learn *nothing* about the other one: the unchosen key is
    /// never derivable from what the receiver holds. The observable version of
    /// that claim is that the pad it cannot compute leaves the other ciphertext
    /// looking unrelated to the plaintext.
    #[test]
    fn the_unchosen_message_stays_padded() {
        let mut rng = rand::thread_rng();
        let (sender, setup) = Sender::new(&mut rng);
        let choices = vec![false];
        let (receiver, chosen) = Receiver::new(&setup, &choices, &mut rng).unwrap();
        let m = pairs(1);
        let payload = sender.send(&chosen, &m).unwrap();
        // The receiver's key opens e0 and not e1.
        let k = receiver.keys()[0];
        assert_eq!(xor_block(&k, &payload.0[0].0), m[0].0);
        assert_ne!(
            xor_block(&k, &payload.0[0].1),
            m[0].1,
            "the receiver's key must not open the unchosen message"
        );
    }

    #[test]
    fn sender_and_receiver_agree_on_the_chosen_key() {
        let mut rng = rand::thread_rng();
        let (sender, setup) = Sender::new(&mut rng);
        let choices = vec![false, true];
        let (receiver, chosen) = Receiver::new(&setup, &choices, &mut rng).unwrap();
        let keys = sender.keys(&chosen).unwrap();
        assert_eq!(keys[0].0, receiver.keys()[0], "c=0 must match k0");
        assert_eq!(keys[1].1, receiver.keys()[1], "c=1 must match k1");
        // ...and disagree on the other branch.
        assert_ne!(keys[0].1, receiver.keys()[0]);
        assert_ne!(keys[1].0, receiver.keys()[1]);
    }

    #[test]
    fn malformed_points_are_rejected_not_panicked_on() {
        let mut rng = rand::thread_rng();
        let (sender, _) = Sender::new(&mut rng);
        // 0xff.. is not a canonical Ristretto encoding.
        let bad = Choices(vec![[0xffu8; POINT_BYTES]]);
        assert!(sender.keys(&bad).is_err());
        assert!(Receiver::new(&Setup([0xffu8; POINT_BYTES]), &[false], &mut rng).is_err());
    }

    #[test]
    fn mismatched_batch_sizes_are_refused() {
        let mut rng = rand::thread_rng();
        let (sender, setup) = Sender::new(&mut rng);
        let (_, chosen) = Receiver::new(&setup, &[false, true], &mut rng).unwrap();
        assert!(sender.send(&chosen, &pairs(3)).is_err());
    }
}
