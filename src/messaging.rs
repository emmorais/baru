//! Peer-to-peer chat and forum.
//!
//! Two out-of-band communication channels for participants, both authenticated
//! by the long-term Ed25519 [`Identity`]:
//!
//! * **Chat** — end-to-end encrypted direct messages.  The sender does an
//!   X25519 ECDH against the recipient's chat key, derives a ChaCha20-Poly1305
//!   key, and encrypts; the message is also signed with the sender's identity,
//!   so it is confidential *and* authenticated.  Confidentiality is end-to-end:
//!   it does not depend on the transport, so messages can be relayed or stored
//!   by others without exposure.
//! * **Forum** — an append-only, tamper-evident log of signed public posts,
//!   committed by a Merkle [`root`](Forum::root).  Every participant that sees
//!   the same posts computes the same root, so the forum state is agreeable in
//!   consensus and any tampering changes the root.
//!
//! Delivery rides on the existing [`Transport`](crate::crypto::net::Transport):
//! chat over unicast, forum over broadcast.  That wiring lives in the node
//! (`tcp_party`); this module is the message layer.
//!
//! Limitation: chat uses static-static ECDH, so it lacks forward secrecy.  An
//! ephemeral (Noise/Signal-style) ratchet is a follow-up.

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};

use crate::crypto::identity::{Identity, IdentityId, IdentitySig, PublicIdentity};
use crate::interpreter::merkle::merkle_root;

/// 32-byte commitment.
pub type Hash = [u8; 32];
/// An X25519 public key used for chat encryption.
pub type ChatPublicKey = [u8; 32];

// ─── Chat: end-to-end encrypted direct messages ────────────────────────────────

/// A participant's secret chat key (X25519).  Held alongside the [`Identity`];
/// its public half is published so others can encrypt to this participant.
pub struct ChatSecret {
    secret: StaticSecret,
}

impl ChatSecret {
    /// Generate a fresh chat key from the OS CSPRNG.
    pub fn generate() -> Self {
        ChatSecret {
            secret: StaticSecret::random_from_rng(rand::rngs::OsRng),
        }
    }

    /// The public chat key to publish.
    pub fn public(&self) -> ChatPublicKey {
        PublicKey::from(&self.secret).to_bytes()
    }

    /// Restore from raw secret bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        ChatSecret {
            secret: StaticSecret::from(bytes),
        }
    }
}

/// The public information needed to send someone an encrypted message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contact {
    pub identity: PublicIdentity,
    pub chat_pk: ChatPublicKey,
}

/// An end-to-end encrypted, identity-signed direct message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Sender's signing identity.
    pub from: PublicIdentity,
    /// Sender's chat public key (so the recipient can derive the shared secret
    /// and reply).
    pub from_chat_pk: ChatPublicKey,
    /// Intended recipient.
    pub to: IdentityId,
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
    /// Signature by `from` over `(to ‖ from_chat_pk ‖ nonce ‖ ciphertext)`.
    pub sig: IdentitySig,
}

/// Errors opening a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatError {
    /// The message was not addressed to this participant.
    WrongRecipient,
    /// The signature did not verify against the claimed sender.
    BadSignature,
    /// Decryption/authentication of the ciphertext failed.
    DecryptFailed,
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ChatError::WrongRecipient => "message not addressed to this participant",
            ChatError::BadSignature => "sender signature did not verify",
            ChatError::DecryptFailed => "decryption failed",
        };
        f.write_str(s)
    }
}

impl std::error::Error for ChatError {}

/// Derive the symmetric key from an ECDH shared secret.
fn chat_key(shared: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"baru-chat-key-v1");
    h.update(shared);
    h.finalize().into()
}

/// Bytes signed to authenticate a chat message.
fn chat_auth(to: &IdentityId, from_chat_pk: &ChatPublicKey, nonce: &[u8; 12], ct: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(32 + 32 + 12 + ct.len());
    v.extend_from_slice(&to.0);
    v.extend_from_slice(from_chat_pk);
    v.extend_from_slice(nonce);
    v.extend_from_slice(ct);
    v
}

/// Seal a message to `recipient`: encrypt end-to-end and sign with `sender`'s
/// identity.
pub fn seal(
    sender: &Identity,
    sender_chat: &ChatSecret,
    recipient: &Contact,
    plaintext: &[u8],
) -> ChatMessage {
    let shared = sender_chat
        .secret
        .diffie_hellman(&PublicKey::from(recipient.chat_pk));
    let key = chat_key(shared.as_bytes());
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key));

    let mut nonce = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .expect("chacha20poly1305 encrypt");

    let to = recipient.identity.id();
    let from_chat_pk = sender_chat.public();
    let sig = sender.sign(&chat_auth(&to, &from_chat_pk, &nonce, &ciphertext));
    ChatMessage {
        from: sender.public(),
        from_chat_pk,
        to,
        nonce,
        ciphertext,
        sig,
    }
}

/// Open a message addressed to us: check the recipient, verify the sender's
/// signature, then decrypt.  Returns the plaintext.
pub fn open(
    recipient: &Identity,
    recipient_chat: &ChatSecret,
    msg: &ChatMessage,
) -> Result<Vec<u8>, ChatError> {
    if msg.to != recipient.id() {
        return Err(ChatError::WrongRecipient);
    }
    let auth = chat_auth(&msg.to, &msg.from_chat_pk, &msg.nonce, &msg.ciphertext);
    if !msg.from.verify(&auth, &msg.sig) {
        return Err(ChatError::BadSignature);
    }
    let shared = recipient_chat
        .secret
        .diffie_hellman(&PublicKey::from(msg.from_chat_pk));
    let key = chat_key(shared.as_bytes());
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key));
    cipher
        .decrypt(Nonce::from_slice(&msg.nonce), msg.ciphertext.as_ref())
        .map_err(|_| ChatError::DecryptFailed)
}

// ─── Forum: append-only tamper-evident public log ──────────────────────────────

/// A public post, signed by its author.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForumPost {
    pub author: PublicIdentity,
    /// Per-author sequence number (monotonic), so an author's posts are ordered
    /// and a replay is detectable.
    pub seq: u64,
    pub body: String,
    pub sig: IdentitySig,
}

fn post_signing_bytes(seq: u64, body: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + body.len());
    v.extend_from_slice(&seq.to_le_bytes());
    v.extend_from_slice(body.as_bytes());
    v
}

impl ForumPost {
    /// Create a signed post.
    pub fn create(author: &Identity, seq: u64, body: &str) -> Self {
        let sig = author.sign(&post_signing_bytes(seq, body));
        ForumPost {
            author: author.public(),
            seq,
            body: body.to_string(),
            sig,
        }
    }

    /// Verify the post's signature binds this author to this content.
    pub fn verify(&self) -> bool {
        self.author
            .verify(&post_signing_bytes(self.seq, &self.body), &self.sig)
    }

    /// A leaf hash committing to `(author, seq, body)`.
    pub fn leaf(&self) -> Hash {
        let mut h = Sha256::new();
        h.update(self.author.id().0);
        h.update(self.seq.to_le_bytes());
        h.update((self.body.len() as u64).to_le_bytes());
        h.update(self.body.as_bytes());
        h.finalize().into()
    }
}

/// Errors appending to the forum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForumError {
    /// The post's signature did not verify.
    BadSignature,
    /// The author's sequence number did not increase (replay / reorder).
    StaleSeq,
}

/// An append-only log of public posts with a Merkle-root commitment.
#[derive(Default)]
pub struct Forum {
    posts: Vec<ForumPost>,
    /// Highest accepted sequence number per author, to reject replays.
    last_seq: std::collections::HashMap<IdentityId, u64>,
}

impl Forum {
    pub fn new() -> Self {
        Forum::default()
    }

    /// Append a post after verifying its signature and that the author's
    /// sequence advances.
    pub fn append(&mut self, post: ForumPost) -> Result<(), ForumError> {
        if !post.verify() {
            return Err(ForumError::BadSignature);
        }
        let author = post.author.id();
        if let Some(&last) = self.last_seq.get(&author) {
            if post.seq <= last {
                return Err(ForumError::StaleSeq);
            }
        }
        self.last_seq.insert(author, post.seq);
        self.posts.push(post);
        Ok(())
    }

    pub fn posts(&self) -> &[ForumPost] {
        &self.posts
    }

    pub fn len(&self) -> usize {
        self.posts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.posts.is_empty()
    }

    /// Tamper-evident commitment over the ordered post log.
    pub fn root(&self) -> Hash {
        let leaves: Vec<Hash> = self.posts.iter().map(ForumPost::leaf).collect();
        merkle_root(&leaves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Party {
        id: Identity,
        chat: ChatSecret,
    }

    impl Party {
        fn new() -> Self {
            Party {
                id: Identity::generate(),
                chat: ChatSecret::generate(),
            }
        }
        fn contact(&self) -> Contact {
            Contact {
                identity: self.id.public(),
                chat_pk: self.chat.public(),
            }
        }
    }

    #[test]
    fn chat_roundtrip() {
        let alice = Party::new();
        let bob = Party::new();
        let msg = seal(&alice.id, &alice.chat, &bob.contact(), b"meet at epoch 7");
        let got = open(&bob.id, &bob.chat, &msg).unwrap();
        assert_eq!(got, b"meet at epoch 7");
    }

    #[test]
    fn chat_reply_uses_senders_key() {
        let alice = Party::new();
        let bob = Party::new();
        // Bob can reply using the sender key carried in the message.
        let m1 = seal(&alice.id, &alice.chat, &bob.contact(), b"hi");
        assert_eq!(open(&bob.id, &bob.chat, &m1).unwrap(), b"hi");
        let reply = seal(&bob.id, &bob.chat, &alice.contact(), b"hey back");
        assert_eq!(open(&alice.id, &alice.chat, &reply).unwrap(), b"hey back");
    }

    #[test]
    fn third_party_cannot_decrypt() {
        let alice = Party::new();
        let bob = Party::new();
        let eve = Party::new();
        let msg = seal(&alice.id, &alice.chat, &bob.contact(), b"secret");
        // Eve is not the recipient.
        assert_eq!(
            open(&eve.id, &eve.chat, &msg),
            Err(ChatError::WrongRecipient)
        );
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let alice = Party::new();
        let bob = Party::new();
        let mut msg = seal(&alice.id, &alice.chat, &bob.contact(), b"transfer 10");
        msg.ciphertext[0] ^= 0xff;
        // The signature no longer matches the tampered ciphertext.
        assert_eq!(open(&bob.id, &bob.chat, &msg), Err(ChatError::BadSignature));
    }

    #[test]
    fn forged_sender_rejected() {
        let alice = Party::new();
        let bob = Party::new();
        let mallory = Party::new();
        let mut msg = seal(&alice.id, &alice.chat, &bob.contact(), b"hi");
        // Mallory swaps in her identity but can't produce Alice's signature.
        msg.from = mallory.id.public();
        assert_eq!(open(&bob.id, &bob.chat, &msg), Err(ChatError::BadSignature));
    }

    #[test]
    fn forum_post_verifies_and_tamper_is_detected() {
        let author = Identity::generate();
        let mut post = ForumPost::create(&author, 1, "hello world");
        assert!(post.verify());
        post.body = "goodbye world".to_string();
        assert!(!post.verify(), "tampered body must fail verification");
    }

    #[test]
    fn forum_rejects_bad_signature_and_stale_seq() {
        let author = Identity::generate();
        let other = Identity::generate();
        let mut forum = Forum::new();

        // A post signed by `author` but claiming `other` as author fails.
        let mut forged = ForumPost::create(&author, 1, "x");
        forged.author = other.public();
        assert_eq!(forum.append(forged), Err(ForumError::BadSignature));

        forum.append(ForumPost::create(&author, 1, "first")).unwrap();
        // Re-using or going backwards in seq is rejected.
        assert_eq!(
            forum.append(ForumPost::create(&author, 1, "replay")),
            Err(ForumError::StaleSeq)
        );
        forum.append(ForumPost::create(&author, 2, "second")).unwrap();
        assert_eq!(forum.len(), 2);
    }

    #[test]
    fn forum_root_agrees_and_reflects_tampering() {
        let a = Identity::generate();
        let b = Identity::generate();
        let posts = [
            ForumPost::create(&a, 1, "one"),
            ForumPost::create(&b, 1, "two"),
            ForumPost::create(&a, 2, "three"),
        ];

        // Two participants that see the same posts compute the same root.
        let mut f1 = Forum::new();
        let mut f2 = Forum::new();
        for p in &posts {
            f1.append(p.clone()).unwrap();
            f2.append(p.clone()).unwrap();
        }
        assert_eq!(f1.root(), f2.root());

        // A forum with a tampered post yields a different root.
        let mut f3 = Forum::new();
        f3.append(posts[0].clone()).unwrap();
        f3.append(ForumPost::create(&b, 1, "TWO")).unwrap();
        f3.append(posts[2].clone()).unwrap();
        assert_ne!(f1.root(), f3.root());
    }
}
