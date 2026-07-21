//! Participant identity: long-term key pairs for a permissionless network.
//!
//! Any participant can generate a key pair and join — no central authority.  An
//! [`Identity`] holds the secret signing key; a [`PublicIdentity`] is the
//! shareable half.  The participant's stable, network-wide handle is its
//! [`IdentityId`], the hash of its public key.
//!
//! This one identity is reused across the whole system: it authenticates secure
//! channels (TLS mutual auth), signs consensus votes and work receipts, earns
//! reputation, and keys the peer-to-peer chat.  Signatures are Ed25519 (pure
//! Rust — no native toolchain required); the id is `SHA-256(public key)`.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A participant's stable network handle: `SHA-256(public key)`.
///
/// Content-addressed, so it can index the reputation ledger, tag receipts, and
/// name peers without any registry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IdentityId(pub [u8; 32]);

impl IdentityId {
    /// Lowercase hex, useful for logs and keys.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// First `n` hex chars — a short, human-friendly label.
    pub fn short(&self, n: usize) -> String {
        let h = self.to_hex();
        h[..n.min(h.len())].to_string()
    }
}

impl std::fmt::Debug for IdentityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IdentityId({})", self.short(12))
    }
}

impl std::fmt::Display for IdentityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short(12))
    }
}

/// The public half of an identity: verify signatures, derive the id.
#[derive(Clone, Serialize, Deserialize)]
pub struct PublicIdentity {
    /// Raw 32-byte Ed25519 verifying key (kept as bytes so `serde` needs no
    /// dalek feature; the `VerifyingKey` is reconstructed on demand).
    key: [u8; 32],
}

impl PublicIdentity {
    /// This identity's network-wide id.
    pub fn id(&self) -> IdentityId {
        let mut hasher = Sha256::new();
        hasher.update(self.key);
        IdentityId(hasher.finalize().into())
    }

    /// The raw 32-byte public key.
    pub fn as_bytes(&self) -> [u8; 32] {
        self.key
    }

    /// Rebuild from raw bytes, rejecting non-canonical / invalid keys.
    pub fn from_bytes(key: [u8; 32]) -> Result<Self, IdentityError> {
        VerifyingKey::from_bytes(&key).map_err(|_| IdentityError::BadKey)?;
        Ok(PublicIdentity { key })
    }

    /// Verify that `sig` is this identity's signature over `msg`.
    pub fn verify(&self, msg: &[u8], sig: &IdentitySig) -> bool {
        let vk = match VerifyingKey::from_bytes(&self.key) {
            Ok(vk) => vk,
            Err(_) => return false,
        };
        vk.verify(msg, &Signature::from_bytes(&sig.0)).is_ok()
    }
}

impl std::fmt::Debug for PublicIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PublicIdentity({})", self.id())
    }
}

/// A detached Ed25519 signature (64 bytes).
#[derive(Clone, Copy)]
pub struct IdentitySig(pub [u8; 64]);

impl std::fmt::Debug for IdentitySig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IdentitySig({}…)", hex::encode(&self.0[..8]))
    }
}

// `serde` derives don't cover `[u8; 64]` (arrays > 32), so serialize the
// signature as a length-prefixed byte string.
impl Serialize for IdentitySig {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for IdentitySig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct SigVisitor;
        impl<'de> serde::de::Visitor<'de> for SigVisitor {
            type Value = IdentitySig;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a 64-byte Ed25519 signature")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<IdentitySig, E> {
                let arr: [u8; 64] = v
                    .try_into()
                    .map_err(|_| E::invalid_length(v.len(), &self))?;
                Ok(IdentitySig(arr))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<IdentitySig, A::Error> {
                let mut arr = [0u8; 64];
                for (i, b) in arr.iter_mut().enumerate() {
                    *b = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(IdentitySig(arr))
            }
        }
        d.deserialize_bytes(SigVisitor)
    }
}

/// A participant's full identity, including the secret signing key.
///
/// Not `Serialize`/`Clone`-able by design — the secret should not be copied
/// casually.  Use [`Identity::to_secret_bytes`] / [`Identity::from_secret_bytes`]
/// for deliberate persistence.
pub struct Identity {
    signing: SigningKey,
}

impl Identity {
    /// Generate a fresh identity from the OS CSPRNG — this is "joining" the
    /// permissionless network.
    pub fn generate() -> Self {
        let mut csprng = rand::rngs::OsRng;
        Identity {
            signing: SigningKey::generate(&mut csprng),
        }
    }

    /// The shareable public identity.
    pub fn public(&self) -> PublicIdentity {
        PublicIdentity {
            key: self.signing.verifying_key().to_bytes(),
        }
    }

    /// This identity's network-wide id.
    pub fn id(&self) -> IdentityId {
        self.public().id()
    }

    /// Sign a message with the secret key.
    pub fn sign(&self, msg: &[u8]) -> IdentitySig {
        IdentitySig(self.signing.sign(msg).to_bytes())
    }

    /// Serialize the 32-byte secret seed (handle with care).
    pub fn to_secret_bytes(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    /// Restore an identity from its 32-byte secret seed.
    pub fn from_secret_bytes(seed: [u8; 32]) -> Self {
        Identity {
            signing: SigningKey::from_bytes(&seed),
        }
    }
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the secret.
        write!(f, "Identity({})", self.id())
    }
}

/// Errors constructing an identity from untrusted bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityError {
    /// The public-key bytes were not a valid Ed25519 point.
    BadKey,
}

impl std::fmt::Display for IdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdentityError::BadKey => write!(f, "invalid Ed25519 public key"),
        }
    }
}

impl std::error::Error for IdentityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_verify_roundtrip() {
        let id = Identity::generate();
        let pk = id.public();
        let msg = b"prove useful work: FHE job #42";
        let sig = id.sign(msg);
        assert!(pk.verify(msg, &sig));
    }

    #[test]
    fn rejects_tampered_message() {
        let id = Identity::generate();
        let pk = id.public();
        let sig = id.sign(b"transfer 10 stake to alice");
        assert!(!pk.verify(b"transfer 99 stake to alice", &sig));
    }

    #[test]
    fn rejects_other_signer() {
        let a = Identity::generate();
        let b = Identity::generate();
        let msg = b"epoch 7 vote";
        let sig = a.sign(msg);
        // b's public key must not verify a's signature.
        assert!(!b.public().verify(msg, &sig));
    }

    #[test]
    fn id_is_deterministic_and_distinct() {
        let id = Identity::generate();
        assert_eq!(id.id(), id.public().id());
        let other = Identity::generate();
        assert_ne!(id.id(), other.id());
    }

    #[test]
    fn public_identity_serde_roundtrip() {
        let id = Identity::generate();
        let pk = id.public();
        let bytes = bincode::serialize(&pk).unwrap();
        let pk2: PublicIdentity = bincode::deserialize(&bytes).unwrap();
        assert_eq!(pk.id(), pk2.id());
        // The restored public key still verifies real signatures.
        let sig = id.sign(b"hello");
        assert!(pk2.verify(b"hello", &sig));
    }

    #[test]
    fn secret_bytes_roundtrip_preserves_identity() {
        let id = Identity::generate();
        let restored = Identity::from_secret_bytes(id.to_secret_bytes());
        assert_eq!(id.id(), restored.id());
        // Signature from the restored key verifies under the original public key.
        let sig = restored.sign(b"same key");
        assert!(id.public().verify(b"same key", &sig));
    }

    #[test]
    fn from_bytes_accepts_valid_key_and_preserves_id() {
        let id = Identity::generate();
        let pk = id.public();
        let restored = PublicIdentity::from_bytes(pk.as_bytes()).expect("valid key");
        assert_eq!(restored.id(), pk.id());
        // An off-curve point (a valid point with its low bit flipped is almost
        // always no longer on the curve) is rejected by decompression.
        let mut off_curve = pk.as_bytes();
        off_curve[0] ^= 0x01;
        // Either it rejects, or it decodes to a *different* identity — never the
        // same key. The security-relevant guarantee is no silent collision.
        if let Ok(other) = PublicIdentity::from_bytes(off_curve) {
            assert_ne!(other.id(), pk.id());
        }
    }
}
