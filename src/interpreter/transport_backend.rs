//! TransportBackend: distributed MPC backend using Transport for communication.
//!
//! This module implements the Backend trait for a real distributed MPC setting
//! where parties communicate with each other via a Transport abstraction. Each
//! operation (mul, a2b, reveal, …) runs the distributed protocol from
//! `crypto::aby3_protocol`.

use std::collections::HashMap;

use crate::crypto::aby3_protocol::PartyShare;
use crate::crypto::net::{Mailbox, PartyId, Transport};
use crate::crypto::types::Fr;

use super::backend::{Backend, ShareRef};

/// A `Backend` implementation that communicates with peer parties via a
/// `Transport`.  Each operation (mul, a2b, reveal, …) runs the distributed
/// protocol from `crypto::protocol`.
///
/// Share data is stored in a local `HashMap<share_id, PartyShare>`.  The
/// `ShareRef` handed back to the evaluator still carries a dummy
/// `local: None`; the actual per-party components live in `shares`.
pub struct TransportBackend<T: Transport> {
    me: PartyId,
    /// Buffering demultiplexer over the transport: receive helpers wait for a
    /// specific message while out-of-order arrivals are retained, not dropped.
    mailbox: Mailbox<T>,
    /// Map from share id to this party's two local components.
    shares: HashMap<u64, PartyShare>,
    /// Per-instance monotonic counter — all parties executing the same
    /// expression in lockstep will generate the same sequence of IDs.
    next_id: u64,
}

impl<T: Transport> TransportBackend<T> {
    pub fn new(me: PartyId, transport: T) -> Self {
        TransportBackend {
            me,
            mailbox: Mailbox::new(transport),
            shares: HashMap::new(),
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn store(&mut self, id: u64, ps: PartyShare, nbits: usize, owner: usize) -> ShareRef {
        self.shares.insert(id, ps);
        ShareRef {
            id,
            nbits,
            owner,
            fixed: false,
        }
    }

    fn get(&self, sr: &ShareRef) -> &PartyShare {
        self.shares
            .get(&sr.id)
            .unwrap_or_else(|| panic!("TransportBackend: unknown share id {}", sr.id))
    }

    fn get_cloned(&self, sr: &ShareRef) -> PartyShare {
        self.get(sr).clone()
    }

    /// This party's canonical additive component of `a`: in the replicated
    /// layout party i holds `(s_i, s_{i+1})`, so each party contributing its
    /// `left` yields the three distinct components whose wrapping sum is the
    /// secret — the per-party half of an MPC→FHE scheme switch
    /// (`crate::hybrid`), with no communication at all. This is
    /// deliberately NOT `Backend::export_additive`: that trait method's
    /// contract is "return ALL components" (satisfiable only by an
    /// in-process simulator); a real party can only ever produce its own.
    pub fn my_additive_component(&self, a: &ShareRef) -> Fr {
        self.get(a).left
    }

    fn store_public(&mut self, value: Fr, nbits: usize) -> ShareRef {
        let id = self.alloc_id();
        let ps = crate::crypto::aby3_protocol::share_public(value, self.me);
        self.store(id, ps, nbits, 0)
    }
}

impl<T: Transport + 'static> Backend for TransportBackend<T> {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef {
        let id = self.alloc_id();
        let ps =
            crate::crypto::aby3_protocol::share_input(value, id, owner, self.me, &self.mailbox);
        self.store(id, ps, nbits, owner)
    }

    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = self.alloc_id();
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = crate::crypto::aby3_protocol::add_party(&pa, &pb);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = self.alloc_id();
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = crate::crypto::aby3_protocol::sub_party(&pa, &pb);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = self.alloc_id();
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = crate::crypto::aby3_protocol::mul_party(&pa, &pb, id, self.me, &self.mailbox);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    /// Real batched multiplication: one communication round for the whole
    /// batch (`aby3_protocol::mul_batch_party`), instead of one round per
    /// pair via the trait's default loop-over-`mul` implementation.
    fn mul_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        if pairs.is_empty() {
            return Vec::new();
        }
        let base_op_id = self.alloc_id();
        let party_pairs: Vec<_> = pairs
            .iter()
            .map(|(a, b)| (self.get_cloned(a), self.get_cloned(b)))
            .collect();
        let results = crate::crypto::aby3_protocol::mul_batch_party(
            &party_pairs,
            base_op_id,
            self.me,
            &self.mailbox,
        );
        pairs
            .iter()
            .zip(results)
            .map(|((a, b), ps)| {
                let id = self.alloc_id();
                self.store(id, ps, a.nbits.max(b.nbits), a.owner)
            })
            .collect()
    }

    // divmod: previously attempted here via bitlen-normalization, which needs
    // `shift_left(b, diff)` for a *secret* `diff = bitlen(a)-bitlen(b)` —
    // `shift_left` only takes a public `usize` amount, so that approach was a
    // dead end. Uses the Backend trait's default restoring-division
    // implementation instead (composed from a2b/mul/add/sub/le/b2a below),
    // which only ever shifts by the public constant 1.

    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        println!(
            "TransportBackend: computing eq for share ids {} and {}",
            a.id, b.id
        );
        // eq(a,b) = NOT(OR(bits of a-b))
        let nbits = a.nbits.max(b.nbits);
        let diff = self.sub(a, b);
        let bit_refs = self.a2b(&diff, nbits);
        // OR all bits via a+b-a*b
        let zero = crate::crypto::aby3_protocol::share_public(0, self.me);
        let mut acc_id = self.alloc_id();
        println!("TransportBackend: inserting zero share id {}", acc_id);
        self.shares.insert(acc_id, zero);
        for br in &bit_refs {
            let pa = self.get_cloned(&ShareRef {
                id: acc_id,
                nbits: 1,
                owner: a.owner,
                fixed: false,
            });
            let pb = self.get_cloned(br);
            let sum = crate::crypto::aby3_protocol::add_party(&pa, &pb);
            let mul_id = self.alloc_id();
            let _mul_id_triple = self.alloc_id();
            let _mul_id_triple2 = self.alloc_id();
            let _mul_id_triple3 = self.alloc_id();
            println!("TransportBackend: computing mul for OR, mul_id={}", mul_id);
            let prod =
                crate::crypto::aby3_protocol::mul_party(&pa, &pb, mul_id, self.me, &self.mailbox);
            let or_ps = crate::crypto::aby3_protocol::sub_party(&sum, &prod);
            let new_id = self.alloc_id();
            println!("TransportBackend: inserting OR share id {}", new_id);
            self.shares.insert(new_id, or_ps);
            acc_id = new_id;
        }
        // NOT = 1 - acc
        let one = crate::crypto::aby3_protocol::share_public(1, self.me);
        let one_id = self.alloc_id();
        self.shares.insert(one_id, one);
        let acc_ps = self.get_cloned(&ShareRef {
            id: acc_id,
            nbits: 1,
            owner: a.owner,
            fixed: false,
        });
        let one_ps = self.get_cloned(&ShareRef {
            id: one_id,
            nbits: 1,
            owner: a.owner,
            fixed: false,
        });
        let not_ps = crate::crypto::aby3_protocol::sub_party(&one_ps, &acc_ps);
        let id = self.alloc_id();
        self.store(id, not_ps, 1, a.owner)
    }

    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        // lt(a,b) = MSB of (a - b) under two's-complement
        let nbits = a.nbits.max(b.nbits);
        let diff = self.sub(a, b);
        let bit_refs = self.a2b(&diff, nbits);
        bit_refs.last().cloned().expect("a2b produced no bits")
    }

    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let lt = self.lt(a, b);
        let eq = self.eq(a, b);
        // OR(lt, eq)
        let pa = self.get_cloned(&lt);
        let pb = self.get_cloned(&eq);
        let sum = crate::crypto::aby3_protocol::add_party(&pa, &pb);
        let mul_id = self.alloc_id();
        let prod =
            crate::crypto::aby3_protocol::mul_party(&pa, &pb, mul_id, self.me, &self.mailbox);
        let ps = crate::crypto::aby3_protocol::sub_party(&sum, &prod);
        let id = self.alloc_id();
        self.store(id, ps, 1, a.owner)
    }

    fn reveal(&mut self, a: &ShareRef, reveal_to: Option<usize>) -> Fr {
        let ps = self.get_cloned(a);
        crate::crypto::aby3_protocol::reveal(&ps, a.id, reveal_to, self.me, &self.mailbox)
    }

    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef> {
        let pa = self.get_cloned(a);
        let op_id = self.alloc_id();
        let bit_shares =
            crate::crypto::aby3_protocol::a2b_party(&pa, nbits, op_id, self.me, &self.mailbox);
        bit_shares
            .into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.shares.insert(id, ps);
                ShareRef {
                    id,
                    nbits: 1,
                    owner: a.owner,
                    fixed: false,
                }
            })
            .collect()
    }

    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        // b2a: sum_i bit_i * 2^i  (local, no communication needed)
        let id = self.alloc_id();
        let zero = crate::crypto::aby3_protocol::share_public(0, self.me);
        let mut acc = zero;
        for (i, br) in bits.iter().enumerate() {
            let bp = self.get_cloned(br);
            let term = crate::crypto::aby3_protocol::scalar_mul_party(&bp, 1u128 << i);
            acc = crate::crypto::aby3_protocol::add_party(&acc, &term);
        }
        self.store(
            id,
            acc,
            bits.first().map(|s| s.nbits).unwrap_or(0),
            bits.first().map(|s| s.owner).unwrap_or(0),
        )
    }

    fn reset_per_expression(&mut self) {
        self.shares.clear();
        self.next_id = 1;
    }

    fn share_public(&mut self, value: Fr, nbits: usize) -> ShareRef {
        self.store_public(value, nbits)
    }

    fn and(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let bit_refs_a = self.a2b(a, a.nbits);
        let bit_refs_b = self.a2b(b, b.nbits);
        let mut and_bits = Vec::new();
        for (bit_a, bit_b) in bit_refs_a.iter().zip(bit_refs_b.iter()) {
            let and_bit = self.mul(bit_a, bit_b);
            and_bits.push(and_bit);
        }
        self.b2a(&and_bits)
    }

    fn or(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let bit_refs_a = self.a2b(a, a.nbits);
        let bit_refs_b = self.a2b(b, b.nbits);
        let mut or_bits = Vec::new();
        for (bit_a, bit_b) in bit_refs_a.iter().zip(bit_refs_b.iter()) {
            let sum = self.add(bit_a, bit_b);
            let prod = self.mul(bit_a, bit_b);
            let ps = self.sub(&sum, &prod);
            or_bits.push(ps);
        }
        self.b2a(&or_bits)
    }

    fn bitlen(&mut self, a: &ShareRef) -> ShareRef {
        let zero: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            owner: 0,
            fixed: false,
        };
        let one: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            owner: 0,
            fixed: false,
        };
        let bit_refs_a = self.a2b(a, a.nbits);
        // Compute bit_refs_b such that bit_refs_b[i] = OR of bit_refs_a[i..]
        let mut bit_refs_b = Vec::new();
        for i in 0..bit_refs_a.len() {
            let or = bit_refs_a[i..]
                .iter()
                .cloned()
                .reduce(|a, b| self.or(&a, &b))
                .unwrap_or(zero.clone());
            bit_refs_b.push(or);
        }
        // Compute bitlen as count of 1s in (not(a_i) and not(b_i)) = 1 - OR(a_i, b_i), from MSB to LSB
        let mut bitlen: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            owner: 0,
            fixed: false,
        };
        for (a, b) in bit_refs_a.iter().zip(bit_refs_b.iter()) {
            let not_a = self.sub(&one, a);
            let not_b = self.sub(&one, b);
            let and = self.and(&not_a, &not_b);
            bitlen = self.add(&bitlen, &and);
        }
        self.b2a(&bit_refs_b)
    }

    fn shift_left(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        let mut result = a.clone();
        // shift left is easier, and more efficient,
        // since we just need to add the element to itself n times
        for _ in 0..n {
            result = self.add(&result, &result);
        }
        result
    }

    fn shift_right(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        // Initialize with zeros
        let mut result = Vec::new();
        for _ in 0..n {
            let zero_share = crate::crypto::aby3_protocol::share_public(0, self.me);
            let id = self.alloc_id();
            self.shares.insert(id, zero_share);
            result.push(ShareRef::new(id, 1, a.owner));
        }
        let bit_refs_a = self.a2b(a, a.nbits);
        // copy the bits shifted to the right
        for item in bit_refs_a.iter().take(bit_refs_a.len().saturating_sub(n)) {
            result.push(item.clone());
        }
        self.b2a(&result)
    }
}
