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
    /// Route batched comparisons through the **Yao leg** ([`Self::lt_gc_batch`])
    /// instead of the log-round bit decomposition. Off by default so the a2b
    /// path stays the control in every measurement; when on, every gadget that
    /// reduces to `lt_batch` — max-pooling, argmax, `max2_batch` — becomes
    /// constant-round without being rewritten. All parties must agree on the
    /// setting: it changes which protocol runs, hence the id-allocation
    /// sequence the parties execute in lockstep.
    gc_nonlin: bool,
}

impl<T: Transport> TransportBackend<T> {
    pub fn new(me: PartyId, transport: T) -> Self {
        TransportBackend {
            me,
            mailbox: Mailbox::new(transport),
            shares: HashMap::new(),
            next_id: 1,
            gc_nonlin: false,
        }
    }

    /// Enable/disable the garbled-circuit route for batched comparisons — see
    /// the `gc_nonlin` field. Set it identically on every party, before any
    /// operation that would diverge.
    pub fn set_gc_nonlinearities(&mut self, on: bool) {
        self.gc_nonlin = on;
    }

    /// Whether batched comparisons currently take the Yao leg.
    pub fn gc_nonlinearities(&self) -> bool {
        self.gc_nonlin
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Reserve `n` consecutive operation ids, returning the base. Sub-protocols
    /// that run several communication rounds derive their ids from this block
    /// so they cannot collide with other operations.
    fn alloc_block(&mut self, n: u64) -> u64 {
        let id = self.next_id;
        self.next_id += n;
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

    /// Batched [`Backend::share_secret`]: distribute `values`, all owned by
    /// `owner`, in one round instead of one per element. Ids are allocated as
    /// a consecutive block, so every party stays in lockstep exactly as the
    /// scalar path does.
    pub fn share_secret_batch(
        &mut self,
        values: &[Fr],
        nbits: usize,
        owner: usize,
    ) -> Vec<ShareRef> {
        if values.is_empty() {
            return Vec::new();
        }
        let base = self.alloc_block(values.len() as u64);
        let shares = crate::crypto::aby3_protocol::share_input_batch(
            values,
            base,
            owner,
            self.me,
            &self.mailbox,
        );
        shares
            .into_iter()
            .enumerate()
            .map(|(k, ps)| self.store(base + k as u64, ps, nbits, owner))
            .collect()
    }

    /// Garbled-circuit batched comparison — the **Yao leg** of ABY3
    /// (`crate::crypto::gc_protocol`). Computes `aᵢ < bᵢ` (signed, i.e.
    /// `MSB(aᵢ−bᵢ)`) for the whole batch in **one** garble round + one
    /// resharing round, regardless of bit width, versus the `~log(nbits)`
    /// rounds of the `a2b`-based [`Backend::lt_batch`]. Requires the 3-party
    /// honest-majority roles (garblers 0/1, evaluator 2); OT-free because every
    /// input wire is a garbler input. Returns RSS arithmetic shares of the 0/1
    /// results, drop-in interoperable with the rest of the pipeline.
    pub fn lt_gc_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        if pairs.is_empty() {
            return Vec::new();
        }
        let nbits = pairs.iter().map(|(a, b)| a.nbits.max(b.nbits)).max().unwrap();
        let n = pairs.len();
        let a: Vec<PartyShare> = pairs.iter().map(|(a, _)| self.get_cloned(a)).collect();
        let b: Vec<PartyShare> = pairs.iter().map(|(_, b)| self.get_cloned(b)).collect();
        // Reserve op_id (garble messages) + the Y2A id block, in lockstep.
        let op_id = self.alloc_block(1 + crate::crypto::gc_protocol::gc_id_block(n, 1));
        let out = crate::crypto::gc_protocol::gc_lt_batch(&a, &b, nbits, op_id, self.me, &self.mailbox);
        out.into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.store(id, ps, 1, 0)
            })
            .collect()
    }

    /// Garbled fused ReLU∘rescale for a batch — the whole nonlinearity in one
    /// garble round: `outᵢ = (vᵢ>=0) ? (vᵢ>>shift) : 0`. Same Yao-leg protocol
    /// and roles as [`Self::lt_gc_batch`]; returns RSS arithmetic shares of the
    /// activations (`nbits−shift`-bit results carried in `nbits`).
    pub fn relu_gc_batch(&mut self, vals: &[ShareRef], nbits: usize, shift: usize) -> Vec<ShareRef> {
        if vals.is_empty() {
            return Vec::new();
        }
        let n = vals.len();
        let width = nbits - shift;
        let vs: Vec<PartyShare> = vals.iter().map(|v| self.get_cloned(v)).collect();
        let op_id = self.alloc_block(1 + crate::crypto::gc_protocol::gc_id_block(n, width));
        let out = crate::crypto::gc_protocol::gc_relu_batch(&vs, nbits, shift, op_id, self.me, &self.mailbox);
        out.into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.store(id, ps, nbits, 0)
            })
            .collect()
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

    fn lt_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        if pairs.is_empty() {
            return Vec::new();
        }
        if self.gc_nonlin {
            return self.lt_gc_batch(pairs);
        }
        // Each comparison is the MSB of `a − b`; batch all the decompositions
        // into one `a2b_batch` so the whole set costs ~9 rounds, not `9·len`.
        let nbits = pairs.iter().map(|(a, b)| a.nbits.max(b.nbits)).max().unwrap();
        let diffs: Vec<ShareRef> = pairs.iter().map(|(a, b)| self.sub(a, b)).collect();
        self.a2b_batch(&diffs, nbits)
            .into_iter()
            .map(|bits| bits.into_iter().last().expect("a2b produced no bits"))
            .collect()
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
        // Reserve a block: the bit decomposition runs ~9 batched
        // multiplication rounds internally, each consuming two ids.
        let op_id = self.alloc_block(crate::crypto::aby3_protocol::A2B_ID_BLOCK);
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

    fn a2b_batch(&mut self, values: &[ShareRef], nbits: usize) -> Vec<Vec<ShareRef>> {
        if values.is_empty() {
            return Vec::new();
        }
        let party_vals: Vec<_> = values.iter().map(|v| self.get_cloned(v)).collect();
        // One id block for the whole batch: `a2b_batch_party` runs the same
        // ~9 rounds as a single decomposition, just wider.
        let op_id = self.alloc_block(crate::crypto::aby3_protocol::A2B_ID_BLOCK);
        let bit_sets = crate::crypto::aby3_protocol::a2b_batch_party(
            &party_vals,
            nbits,
            op_id,
            self.me,
            &self.mailbox,
        );
        values
            .iter()
            .zip(bit_sets)
            .map(|(v, bits)| {
                bits.into_iter()
                    .map(|ps| {
                        let id = self.alloc_id();
                        self.shares.insert(id, ps);
                        ShareRef { id, nbits: 1, owner: v.owner, fixed: false }
                    })
                    .collect()
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
