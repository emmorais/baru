//! A [`Backend`] that runs the maliciously-secure ABY3 protocol.
//!
//! Same interface as [`super::transport_backend::TransportBackend`], same wire
//! format, same three-party honest-majority deployment — but every operation
//! goes through the checked primitives in [`crate::crypto::aby3_verify`], so a
//! single actively corrupt party either fails to affect the result or is caught.
//!
//! # Using it
//!
//! ```ignore
//! let mut be = MaliciousBackend::new(me, transport);
//! let out = be.reveal(&result, None);
//! be.finish()?;            // ← the verdict. Do not trust `out` before this.
//! ```
//!
//! **The output of a run is meaningless until [`Self::finish`] returns `Ok`.**
//! That is inherent to security-with-abort, not an awkwardness of this
//! implementation: the checks are deferred precisely so that the decision to
//! abort cannot be timed to the data, which means the decision necessarily
//! arrives after the values do.
//!
//! # Why the trait signatures did not have to change
//!
//! Adding malicious security usually means threading a `Result` through every
//! operation, which here would have rippled through the whole evaluator. It is
//! unnecessary: because every check is **deferred to one point**, the individual
//! operations have nothing to report. Only [`Self::finish`] returns
//! [`Checked`]. The design property that makes the protocol secure against
//! selective failure is the same one that keeps the refactor small.
//!
//! # Cost relative to semi-honest
//!
//! Every wire carries a MAC, so multiplications send twice the bytes — but both
//! halves batch into the *same* round, so the round count is unchanged. Bit
//! decomposition additionally spends one batched round on a pool of random
//! pairs and one batched opening to attach MACs to its leaves. Given that this
//! repository's recurring finding is that round structure rather than
//! arithmetic dominates, the overhead is deliberately concentrated in the axis
//! that costs least. `bench-malicious` measures it rather than predicting it.

use std::collections::HashMap;

use crate::crypto::aby3_verify::{
    a2b_batch_checked, add, generate_mac_pairs, mac_existing_batch, mul_batch_checked,
    reveal_batch_checked, scalar_mul, share_public as mal_share_public, sub, MacPair, MalShare,
    Verifier,
};
use crate::crypto::net::{Mailbox, PartyId, PrfKeys, Transport};
use crate::crypto::security::{AttestedSecurity, Checked, Security};
use crate::crypto::types::Fr;

use super::backend::{Backend, ShareRef};

/// Ids reserved per bit-decomposition call. Wider than the semi-honest block
/// because the malicious path also spends ids on the random-pair pool and the
/// batched MAC opening, one per leaf.
const A2B_BLOCK: u64 = 1 << 20;

pub struct MaliciousBackend<T: Transport> {
    me: PartyId,
    mailbox: Mailbox<T>,
    keys: PrfKeys,
    verifier: Verifier,
    shares: HashMap<u64, MalShare>,
    next_id: u64,
    /// Unconsumed random pairs. Refilled in batches, since a refill costs a
    /// round however many pairs it produces.
    pool: Vec<MacPair>,
    interactive_calls: usize,
}

impl<T: Transport> MaliciousBackend<T> {
    pub fn new(me: PartyId, transport: T) -> Self {
        let mailbox = Mailbox::new(transport);
        let keys = mailbox.prf_keys(me);
        let verifier = Verifier::new(me, &keys);
        MaliciousBackend {
            me,
            mailbox,
            keys,
            verifier,
            shares: HashMap::new(),
            next_id: 1,
            pool: Vec::new(),
            interactive_calls: 0,
        }
    }

    /// Take the deferred verdict. Call once, at the end of the computation.
    ///
    /// Until this returns `Ok`, nothing the backend produced should be acted
    /// on — see the module header.
    pub fn finish(&mut self) -> Checked<()> {
        let op = self.alloc_block(4);
        crate::crypto::aby3_verify::check(self.me, &self.mailbox, op, &mut self.verifier)
    }

    /// Batched interactive protocol invocations, comparable with
    /// [`super::transport_backend::TransportBackend::interactive_calls`].
    pub fn interactive_calls(&self) -> usize {
        self.interactive_calls
    }

    /// Multiplications the verifier is covering — used by the coverage test.
    pub fn verified_products(&self) -> usize {
        self.verifier.product_count()
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn alloc_block(&mut self, n: u64) -> u64 {
        let id = self.next_id;
        self.next_id += n;
        id
    }

    fn store(&mut self, ms: MalShare, nbits: usize, owner: usize) -> ShareRef {
        let id = self.alloc_id();
        self.shares.insert(id, ms);
        ShareRef {
            id,
            nbits,
            owner,
            fixed: false,
        }
    }

    fn get(&self, sr: &ShareRef) -> MalShare {
        self.shares
            .get(&sr.id)
            .unwrap_or_else(|| panic!("MaliciousBackend: unknown share id {}", sr.id))
            .clone()
    }

    /// Hand out `n` random pairs, refilling in one batched round if short.
    fn take_pairs(&mut self, n: usize) -> Vec<MacPair> {
        if self.pool.len() < n {
            // Over-provision so a sequence of small requests does not pay a
            // round each.
            let want = (n - self.pool.len()).max(64);
            let op = self.alloc_block(2);
            self.interactive_calls += 1;
            let fresh = generate_mac_pairs(
                want,
                op,
                self.me,
                &self.mailbox,
                &self.keys,
                &mut self.verifier,
            );
            self.pool.extend(fresh);
        }
        self.pool.split_off(self.pool.len() - n)
    }

    /// Give already-shared values MACs, one batched opening for the whole slice.
    fn attach_macs(&mut self, vals: &[crate::crypto::aby3_protocol::PartyShare]) -> Vec<MalShare> {
        if vals.is_empty() {
            return Vec::new();
        }
        let pairs = self.take_pairs(vals.len());
        let base = self.alloc_block(vals.len() as u64);
        self.interactive_calls += 1;
        mac_existing_batch(
            vals,
            &pairs,
            base,
            self.me,
            &self.mailbox,
            &mut self.verifier,
        )
    }

    /// Batched [`Backend::share_secret`] — one round for the whole layer.
    pub fn share_secret_batch(
        &mut self,
        values: &[Fr],
        nbits: usize,
        owner: usize,
    ) -> Vec<ShareRef> {
        if values.is_empty() {
            return Vec::new();
        }
        self.interactive_calls += 1;
        let base = self.alloc_block(values.len() as u64);
        let raw = crate::crypto::aby3_protocol::share_input_batch(
            values,
            base,
            owner,
            self.me,
            &self.mailbox,
        );
        // A dealer can send inconsistent copies of the same component; this is
        // what catches it, at no communication cost.
        for (k, ps) in raw.iter().enumerate() {
            self.verifier.note_input(base + k as u64, ps);
        }
        let macd = self.attach_macs(&raw);
        macd.into_iter()
            .map(|ms| self.store(ms, nbits, owner))
            .collect()
    }

    /// Product of a slice of MAC'd shares, as a balanced tree.
    ///
    /// `log₂(n)` batched rounds rather than the `n` a sequential fold would
    /// cost — which matters because equality testing folds one term per bit.
    fn product_tree(&mut self, mut terms: Vec<MalShare>) -> MalShare {
        assert!(!terms.is_empty(), "product_tree needs at least one term");
        while terms.len() > 1 {
            let mut pairs = Vec::with_capacity(terms.len() / 2);
            let odd = if terms.len() % 2 == 1 {
                terms.pop()
            } else {
                None
            };
            for ch in terms.chunks_exact(2) {
                pairs.push((ch[0].clone(), ch[1].clone()));
            }
            let op = self.alloc_block(2);
            self.interactive_calls += 1;
            let mut next =
                mul_batch_checked(&pairs, op, self.me, &self.mailbox, &mut self.verifier);
            if let Some(o) = odd {
                next.push(o);
            }
            terms = next;
        }
        terms.pop().expect("one term remains")
    }

    fn a2b_many(&mut self, vals: &[MalShare], nbits: usize) -> Vec<Vec<MalShare>> {
        if vals.is_empty() {
            return Vec::new();
        }
        self.interactive_calls += 1;
        let op = self.alloc_block(A2B_BLOCK);
        a2b_batch_checked(
            vals,
            nbits,
            op,
            self.me,
            &self.mailbox,
            &self.keys,
            &mut self.verifier,
        )
    }

    fn one(&self) -> MalShare {
        mal_share_public(1, self.me, self.verifier.key())
    }
}

impl<T: Transport> AttestedSecurity for MaliciousBackend<T> {
    fn attested_security(&self) -> Security {
        Security::Malicious
    }
}

impl<T: Transport + 'static> Backend for MaliciousBackend<T> {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef {
        self.share_secret_batch(&[value], nbits, owner)
            .pop()
            .expect("one share")
    }

    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let (x, y) = (self.get(a), self.get(b));
        let nbits = a.nbits.max(b.nbits);
        self.store(add(&x, &y), nbits, a.owner)
    }

    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let (x, y) = (self.get(a), self.get(b));
        let nbits = a.nbits.max(b.nbits);
        self.store(sub(&x, &y), nbits, a.owner)
    }

    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        self.mul_batch(&[(a.clone(), b.clone())])
            .pop()
            .expect("one product")
    }

    fn mul_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        if pairs.is_empty() {
            return Vec::new();
        }
        self.interactive_calls += 1;
        let operands: Vec<(MalShare, MalShare)> = pairs
            .iter()
            .map(|(a, b)| (self.get(a), self.get(b)))
            .collect();
        let op = self.alloc_block(2);
        let out = mul_batch_checked(&operands, op, self.me, &self.mailbox, &mut self.verifier);
        out.into_iter()
            .zip(pairs)
            .map(|(ms, (a, b))| {
                let nbits = a.nbits.max(b.nbits);
                self.store(ms, nbits, a.owner)
            })
            .collect()
    }

    /// `a == b` as `Π (1 − bitᵢ(a−b))`, a balanced product tree.
    ///
    /// The product is 1 exactly when every bit of the difference is zero. This
    /// costs `log₂(nbits)` rounds after the decomposition, against `nbits`
    /// rounds for a sequential OR-fold.
    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let nbits = a.nbits.max(b.nbits);
        let diff = self.sub(a, b);
        let d = self.get(&diff);
        let bits = self.a2b_many(&[d], nbits).pop().expect("one lane");
        let one = self.one();
        let terms: Vec<MalShare> = bits.iter().map(|bit| sub(&one, bit)).collect();
        let ms = self.product_tree(terms);
        self.store(ms, 1, a.owner)
    }

    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        self.lt_batch(&[(a.clone(), b.clone())])
            .pop()
            .expect("one comparison")
    }

    fn lt_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        if pairs.is_empty() {
            return Vec::new();
        }
        let nbits = pairs
            .iter()
            .map(|(a, b)| a.nbits.max(b.nbits))
            .max()
            .expect("non-empty");
        let diffs: Vec<MalShare> = pairs
            .iter()
            .map(|(a, b)| {
                let (x, y) = (self.get(a), self.get(b));
                sub(&x, &y)
            })
            .collect();
        let owners: Vec<usize> = pairs.iter().map(|(a, _)| a.owner).collect();
        self.a2b_many(&diffs, nbits)
            .into_iter()
            .zip(owners)
            .map(|(mut bits, owner)| {
                let msb = bits.remove(nbits - 1);
                self.store(msb, 1, owner)
            })
            .collect()
    }

    /// `a ≤ b` as `1 − (b < a)` — one comparison, not a `lt` plus an `eq`.
    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let gt = self.lt(b, a);
        let g = self.get(&gt);
        let one = self.one();
        self.store(sub(&one, &g), 1, a.owner)
    }

    /// Reveal. Note the value is only trustworthy once [`Self::finish`] has
    /// returned `Ok` — an unverified opening is exactly what the adversary
    /// would like you to act on.
    fn reveal(&mut self, a: &ShareRef, _reveal_to: Option<usize>) -> Fr {
        self.interactive_calls += 1;
        let ms = self.get(a);
        let id = self.alloc_id();
        reveal_batch_checked(&[ms.val], id, self.me, &self.mailbox, &mut self.verifier)
            .pop()
            .expect("one value")
    }

    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef> {
        let ms = self.get(a);
        let bits = self.a2b_many(&[ms], nbits).pop().expect("one lane");
        let owner = a.owner;
        bits.into_iter().map(|b| self.store(b, 1, owner)).collect()
    }

    fn a2b_batch(&mut self, values: &[ShareRef], nbits: usize) -> Vec<Vec<ShareRef>> {
        if values.is_empty() {
            return Vec::new();
        }
        let vals: Vec<MalShare> = values.iter().map(|v| self.get(v)).collect();
        let owners: Vec<usize> = values.iter().map(|v| v.owner).collect();
        self.a2b_many(&vals, nbits)
            .into_iter()
            .zip(owners)
            .map(|(bits, owner)| bits.into_iter().map(|b| self.store(b, 1, owner)).collect())
            .collect()
    }

    /// `Σ 2ⁱ·bitᵢ` — local, no communication, and the MACs recompose with it.
    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        let owner = bits.first().map(|s| s.owner).unwrap_or(0);
        let mut acc = mal_share_public(0, self.me, self.verifier.key());
        for (i, br) in bits.iter().enumerate() {
            let b = self.get(br);
            acc = add(&acc, &scalar_mul(&b, 1u128 << i));
        }
        self.store(acc, bits.len(), owner)
    }

    /// Deliberately **not** a reset.
    ///
    /// The semi-honest backend clears its share table and rewinds its id
    /// counter between expressions. Rewinding here would be unsound twice over:
    /// operation ids seed the correlated randomness, so reuse would repeat a
    /// mask, and they tag the verification transcript, so reuse would let two
    /// different values collide in it. Ids stay monotonic and the verifier
    /// keeps accumulating until `finish`.
    fn reset_per_expression(&mut self) {
        self.shares.clear();
    }

    fn share_public(&mut self, value: Fr, nbits: usize) -> ShareRef {
        let ms = mal_share_public(value, self.me, self.verifier.key());
        self.store(ms, nbits, 0)
    }

    fn and(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let nbits = a.nbits.max(b.nbits);
        let bits = self.a2b_batch(&[a.clone(), b.clone()], nbits);
        let pairs: Vec<(ShareRef, ShareRef)> = bits[0]
            .iter()
            .cloned()
            .zip(bits[1].iter().cloned())
            .collect();
        let out = self.mul_batch(&pairs);
        self.b2a(&out)
    }

    fn or(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let nbits = a.nbits.max(b.nbits);
        let bits = self.a2b_batch(&[a.clone(), b.clone()], nbits);
        let pairs: Vec<(ShareRef, ShareRef)> = bits[0]
            .iter()
            .cloned()
            .zip(bits[1].iter().cloned())
            .collect();
        let prods = self.mul_batch(&pairs);
        // a ∨ b = a + b − ab, all local now that the products exist.
        let ors: Vec<ShareRef> = pairs
            .iter()
            .zip(&prods)
            .map(|((x, y), p)| {
                let s = self.add(x, y);
                self.sub(&s, p)
            })
            .collect();
        self.b2a(&ors)
    }

    /// Position of the highest set bit, as `Σ prefix-OR` from the top down.
    fn bitlen(&mut self, a: &ShareRef) -> ShareRef {
        let bits = self.a2b(a, a.nbits);
        // suffix[i] = OR of bits[i..], computed from the top so each step is
        // one multiplication on the running value.
        let mut suffix: Vec<ShareRef> = vec![bits[bits.len() - 1].clone()];
        for i in (0..bits.len() - 1).rev() {
            let prev = suffix.last().expect("non-empty").clone();
            let s = self.add(&bits[i], &prev);
            let p = self.mul(&bits[i], &prev);
            suffix.push(self.sub(&s, &p));
        }
        suffix.reverse();
        let mut acc = self.share_public(0, a.nbits);
        for s in &suffix {
            acc = self.add(&acc, s);
        }
        acc
    }

    fn shift_left(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        let x = self.get(a);
        let shifted = scalar_mul(&x, 1u128 << n);
        self.store(shifted, a.nbits, a.owner)
    }

    fn shift_right(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        let bits = self.a2b(a, a.nbits);
        if n >= bits.len() {
            return self.share_public(0, a.nbits);
        }
        self.b2a(&bits[n..])
    }
}
