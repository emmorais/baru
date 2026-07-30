//! In-process **simulator** of the ABY3 protocol — correct on values, and
//! deliberately named so no call site can mistake it for the protocol.
//!
//! All three parties' shares live in one `HashMap` in one process, so the
//! simulator can take shortcuts the distributed protocol cannot: its share
//! multiplication **reconstructs and reshares locally**, which costs nothing and
//! reveals everything. That is fine for what it is for — a fast oracle that
//! checks a computation's *values* against a plaintext reference — and wrong for
//! anything else.
//!
//! **Never quote its timings or message counts as protocol costs.** It
//! understated one measured private layer by 2.7× (6.7 ms simulated against
//! 18 ms real). Anything claiming a cost must run on
//! [`crate::interpreter::transport_backend::TransportBackend`], which speaks the
//! genuine wire protocol between three parties; see `src/bench/` for the
//! three-thread pattern every benchmark here uses.
//!
//! It was called `Aby3Backend` until 2026-07-30, which read as a peer of
//! `TransportBackend` rather than as a simulation of it.

use std::collections::HashMap;

use crate::crypto::types::Fr;
use crate::crypto::{arith, bits, sim, Share3};

use super::backend::{Backend, ShareRef};

/// Simulates the ABY3 online protocol across three in-process parties. See the
/// module docs: correct on values, not a source of costs.
pub struct Aby3Simulator {
    /// Per-backend share storage: id -> Share3
    shares: HashMap<u64, Share3>,
    /// Next share ID to allocate
    next_id: u64,
}

impl Aby3Simulator {
    pub fn new() -> Self {
        Aby3Simulator {
            shares: HashMap::new(),
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn store(&mut self, s: Share3, nbits: usize, owner: usize) -> ShareRef {
        let id = self.alloc_id();
        self.shares.insert(id, s);
        ShareRef::new(id, nbits, owner)
    }

    fn get(&self, sr: &ShareRef) -> &Share3 {
        self.shares
            .get(&sr.id)
            .unwrap_or_else(|| panic!("Aby3Simulator: unknown share id {}", sr.id))
    }

    fn get_cloned(&self, sr: &ShareRef) -> Share3 {
        self.get(sr).clone()
    }
}

// Default implementation for Aby3Simulator
impl Default for Aby3Simulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for Aby3Simulator {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef {
        let s = arith::share_secret(value);
        self.store(s, nbits, owner)
    }

    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = self.get_cloned(a);
        let sb = self.get_cloned(b);
        let sum = arith::add_shares(&sa, &sb);
        self.store(sum, a.nbits.max(b.nbits), a.owner)
    }

    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = self.get_cloned(a);
        let sb = self.get_cloned(b);
        let out = arith::sub_shares(&sa, &sb);
        self.store(out, a.nbits.max(b.nbits), a.owner)
    }

    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = self.get_cloned(a);
        let sb = self.get_cloned(b);
        let prod = sim::mul_ab3_concurrent(&sa, &sb);
        self.store(prod, a.nbits.max(b.nbits), a.owner)
    }

    // divmod: uses the Backend trait's default restoring-division
    // implementation (composed from a2b/mul/add/sub/le/b2a below).

    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = self.get_cloned(a);
        let sb = self.get_cloned(b);
        let out = bits::eq_ab3(&sa, &sb, a.nbits.max(b.nbits));
        self.store(out, 1, a.owner)
    }

    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = self.get_cloned(a);
        let sb = self.get_cloned(b);
        let out = bits::lt_ab3_signed(&sa, &sb, a.nbits.max(b.nbits));
        self.store(out, 1, a.owner)
    }

    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = self.get_cloned(a);
        let sb = self.get_cloned(b);
        let out = bits::le_ab3_signed(&sa, &sb, a.nbits.max(b.nbits));
        self.store(out, 1, a.owner)
    }

    fn reveal(&mut self, a: &ShareRef, _reveal_to: Option<usize>) -> Fr {
        let s = self.get_cloned(a);
        arith::reconstruct(&s)
    }

    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef> {
        let sa = self.get_cloned(a);
        let bits = sim::a2b_concurrent(&sa, nbits);
        bits.into_iter()
            .map(|b| self.store(b, 1, a.owner))
            .collect()
    }

    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        let mut out = arith::share_secret(0);
        for (i, bit_ref) in bits.iter().enumerate() {
            let b = self.get_cloned(bit_ref);
            let term = crate::crypto::arith::scalar_mul(&b, 1u128 << i);
            out = crate::crypto::arith::add_shares(&out, &term);
        }
        self.store(
            out,
            bits.first().map(|s| s.nbits).unwrap_or(0),
            bits.first().map(|s| s.owner).unwrap_or(0),
        )
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
            let or_bit = self.add(bit_a, bit_b);
            or_bits.push(or_bit);
        }
        self.b2a(&or_bits)
    }

    fn bitlen(&mut self, a: &ShareRef) -> ShareRef {
        let bit_refs_a = self.a2b(a, a.nbits);
        // Compute bit_refs_b such that bit_refs_b[i] = OR of bit_refs_a[i..]
        let mut bit_refs_b = Vec::new();
        let zero_share = arith::share_secret(0);
        let zero_id = self.alloc_id();
        self.shares.insert(zero_id, zero_share);
        let mut init_false = ShareRef::new(zero_id, 1, a.owner);
        for i in (0..bit_refs_a.len()).rev() {
            let and = self.mul(&init_false, &bit_refs_a[i]);
            init_false = self.add(&init_false, &bit_refs_a[i]);
            init_false = self.sub(&init_false, &and);
            bit_refs_b.push(init_false.clone());
        }
        bit_refs_b.reverse();
        // Compute bitlen as count of 1s in (not(a_i) and not(b_i)) = 1 - OR(a_i, b_i), from MSB to LSB
        let zero_share2 = arith::share_secret(0);
        let zero_id2 = self.alloc_id();
        self.shares.insert(zero_id2, zero_share2);
        let mut bitlen = ShareRef::new(zero_id2, 1, a.owner);
        // compute popcount of bit_refs_b
        for b in bit_refs_b.iter() {
            bitlen = self.add(&bitlen, b);
        }
        bitlen
    }

    fn shift_left(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        let mut result = a.clone();
        // shift is easier, since we just need to add the element to itself n times
        for _ in 0..n {
            result = self.add(&result, &result);
        }
        result
    }

    fn shift_right(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        // Initialize with zeros
        let mut result = Vec::new();
        for _ in 0..a.nbits {
            let zero_share = arith::share_secret(0);
            result.push(self.store(zero_share, 1, a.owner));
        }
        let bit_refs_a = self.a2b(a, a.nbits);
        if bit_refs_a.len() > n {
            // copy the bits shifted to the right
            result[..(bit_refs_a.len() - n)].clone_from_slice(&bit_refs_a[n..]);
        }
        self.b2a(&result)
    }

    fn reset_per_expression(&mut self) {
        self.shares.clear();
        self.next_id = 1;
    }

    // ── Scheme switching (see `crate::hybrid`) ───────────────────────────────
    // The in-process simulator holds all three parties' components directly,
    // so export/import are trivial projections to/from `Share3`.

    fn share_public(&mut self, value: Fr, nbits: usize) -> ShareRef {
        // Whole value in the first component; the in-process reveal sums the
        // three components, so this reconstructs to `value`.
        self.store(Share3(value, 0, 0), nbits, 0)
    }

    fn export_additive(&mut self, a: &ShareRef) -> Vec<Fr> {
        let s = self.get_cloned(a);
        vec![s.0, s.1, s.2]
    }

    fn import_additive(&mut self, components: &[Fr], nbits: usize) -> ShareRef {
        // Fold any component count into three slots (wrapping addition
        // preserves the sum, which is all that defines the sharing).
        let mut c = [0u128; 3];
        for (i, v) in components.iter().enumerate() {
            c[i % 3] = c[i % 3].wrapping_add(*v);
        }
        self.store(Share3(c[0], c[1], c[2]), nbits, 0)
    }
}
