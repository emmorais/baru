use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::crypto::net::{PartyId, Transport};
use crate::crypto::aby3_protocol::{PartyShare};
use crate::crypto::types::Fr;
use crate::crypto::{arith, bits, sim, Share3};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct ShareRef {
    pub id: u64,
    pub nbits: usize,
    // backend-local representation when using SimBackend
    pub local: Option<crate::crypto::types::Share3>,
    // owner participant id for input shares
    pub owner: usize,
}

impl ShareRef {
    pub fn new_local(share: crate::crypto::types::Share3, nbits: usize, owner: usize) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        ShareRef {
            id,
            nbits,
            local: Some(share),
            owner,
        }
    }
    // zero
    pub fn zero() -> Self {
        ShareRef {
            id: 0,
            nbits: 0,
            local: Some(Share3::zero()),
            owner: 0,
        }
    }
}

/// Minimal synchronous backend trait for now.
pub trait Backend: Send + Sync {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef;
    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn divmod(&mut self, a: &ShareRef, b: &ShareRef) -> (ShareRef, ShareRef);
    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    /// Reveal a shared value to one or all participants. `reveal_to = None` -> reveal to all.
    fn reveal(&mut self, a: &ShareRef, reveal_to: Option<usize>) -> Fr;
    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef>;
    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef;
    /// Reset per-expression state (share map, ID counter) so the next expression
    /// starts with a clean slate.  Called by the REPL before each expression to
    /// keep all parties' ID counters in sync even after an error/abort.
    fn reset_per_expression(&mut self) {}
    fn and(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn or(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn bitlen(&mut self, a: &ShareRef) -> ShareRef;
    fn shift_right(&mut self, a: &ShareRef, n: usize) -> ShareRef;
    fn shift_left(&mut self, a: &ShareRef, n: usize) -> ShareRef;
}

/// ABY3 backend that runs the online protocol via explicit in-process message
/// exchanges between three simulated parties.
pub struct Aby3Backend {}

impl Aby3Backend {
    pub fn new() -> Self {
        Aby3Backend {}
    }
}

// Default implementation for Aby3Backend
impl Default for Aby3Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for Aby3Backend {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef {
        let s = arith::share_secret(value);
        ShareRef::new_local(s, nbits, owner)
    }

    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = a.local.clone().expect("expected local share");
        let sb = b.local.clone().expect("expected local share");
        let sum = arith::add_shares(&sa, &sb);
        ShareRef::new_local(sum, a.nbits.max(b.nbits), a.owner)
    }

    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = a.local.clone().expect("expected local share");
        let sb = b.local.clone().expect("expected local share");
        let out = arith::sub_shares(&sa, &sb);
        ShareRef::new_local(out, a.nbits.max(b.nbits), a.owner)
    }

    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = a.local.clone().expect("expected local share");
        let sb = b.local.clone().expect("expected local share");
        let prod = sim::mul_ab3_concurrent(&sa, &sb);
        ShareRef::new_local(prod, a.nbits.max(b.nbits), a.owner)
    }

    fn divmod(&mut self, _a: &ShareRef, _b: &ShareRef) -> (ShareRef, ShareRef) {
        unimplemented!("division not implemented yet")
    }

    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = a.local.clone().expect("expected local share");
        let sb = b.local.clone().expect("expected local share");
        let out = bits::eq_ab3(&sa, &sb, a.nbits.max(b.nbits));
        ShareRef::new_local(out, 1, a.owner)
    }

    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = a.local.clone().expect("expected local share");
        let sb = b.local.clone().expect("expected local share");
        let out = bits::lt_ab3_signed(&sa, &sb, a.nbits.max(b.nbits));
        ShareRef::new_local(out, 1, a.owner)
    }

    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let sa = a.local.clone().expect("expected local share");
        let sb = b.local.clone().expect("expected local share");
        let out = bits::le_ab3_signed(&sa, &sb, a.nbits.max(b.nbits));
        ShareRef::new_local(out, 1, a.owner)
    }

    fn reveal(&mut self, a: &ShareRef, _reveal_to: Option<usize>) -> Fr {
        let s = a.local.clone().expect("expected local share");
        arith::reconstruct(&s)
    }

    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef> {
        let sa = a.local.clone().expect("expected local share");
        let bits = sim::a2b_concurrent(&sa, nbits);
        bits.into_iter()
            .map(|b| ShareRef::new_local(b, 1, a.owner))
            .collect()
    }

    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        let mut out = arith::share_secret(0);
        for (i, bit_ref) in bits.iter().enumerate() {
            let b = bit_ref.local.clone().expect("expected local bit share");
            let term = crate::crypto::arith::scalar_mul(&b, 1u128 << i);
            out = crate::crypto::arith::add_shares(&out, &term);
        }
        ShareRef::new_local(
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
        let mut init_false = ShareRef::zero();
        for i in (0..bit_refs_a.len()).rev() {
            let and = self.mul(&init_false, &bit_refs_a[i]);
            init_false = self.add(&init_false, &bit_refs_a[i]);
            init_false = self.sub(&init_false, &and);
            bit_refs_b.push(init_false.clone());
        }
        bit_refs_b.reverse();
        // Compute bitlen as count of 1s in (not(a_i) and not(b_i)) = 1 - OR(a_i, b_i), from MSB to LSB
        let mut bitlen: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            local: Some(Share3::zero()),
            owner: 0,
        };
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
        let mut result = vec![ShareRef::new_local(Share3::zero(), 1, a.owner); a.nbits];
        let bit_refs_a = self.a2b(a, a.nbits);
        if bit_refs_a.len() > n {
            // copy the bits shifted to the right
            result[..(bit_refs_a.len() - n)].clone_from_slice(&bit_refs_a[n..]);
        }
        self.b2a(&result)
    }
}

// ---------------------------------------------------------------------------
// TransportBackend — real distributed backend using the Transport trait
// ---------------------------------------------------------------------------

/// A `Backend` implementation that communicates with peer parties via a
/// `Transport`.  Each operation (mul, a2b, reveal, …) runs the distributed
/// protocol from `crypto::protocol`.
///
/// Share data is stored in a local `HashMap<share_id, PartyShare>`.  The
/// `ShareRef` handed back to the evaluator still carries a dummy
/// `local: None`; the actual per-party components live in `shares`.
pub struct TransportBackend<T: Transport> {
    me: PartyId,
    transport: T,
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
            transport,
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
            local: None,
            owner,
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
}

impl<T: Transport + 'static> Backend for TransportBackend<T> {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef {
        let id = self.alloc_id();
        let ps = crate::crypto::aby3_protocol::share_input(value, id, owner, self.me, &self.transport);
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
        let ps = crate::crypto::aby3_protocol::mul_party(&pa, &pb, id, self.me, &self.transport);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    // assume a > b
    fn divmod(&mut self, _a: &ShareRef, _b: &ShareRef) -> (ShareRef, ShareRef) {
        // Compute the bitlen of a and b, multiply b by 2^(bitlen(a)-bitlen(b)),
        // then if the result is positive,
        // subtract b until the remainder is less than b, otherwise add b until the remainder is non-negative
        /*let bitlen_a = &self.bitlen(a);
        let bitlen_b = &self.bitlen(b);
        let mut diff = self.sub(bitlen_a, bitlen_b);
        // multiply b by 2^diff
        let mut rem = self.shift_left(&b, diff);

        if rem.nbits > 0 {
            // result is positive
            while rem.nbits > 0 {
                rem = self.sub(&rem, b);
            }
        } else {
            while rem.nbits < 0 {
                rem = self.add(&rem, b);
            }
        }
        (rem, b)*/
        unimplemented!("division not implemented yet")
    }

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
                local: None,
                owner: a.owner,
            });
            let pb = self.get_cloned(br);
            let sum = crate::crypto::aby3_protocol::add_party(&pa, &pb);
            let mul_id = self.alloc_id();
            let _mul_id_triple = self.alloc_id();
            let _mul_id_triple2 = self.alloc_id();
            let _mul_id_triple3 = self.alloc_id();
            println!("TransportBackend: computing mul for OR, mul_id={}", mul_id);
            let prod = crate::crypto::aby3_protocol::mul_party(&pa, &pb, mul_id, self.me, &self.transport);
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
            local: None,
            owner: a.owner,
        });
        let one_ps = self.get_cloned(&ShareRef {
            id: one_id,
            nbits: 1,
            local: None,
            owner: a.owner,
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
        let prod = crate::crypto::aby3_protocol::mul_party(&pa, &pb, mul_id, self.me, &self.transport);
        let ps = crate::crypto::aby3_protocol::sub_party(&sum, &prod);
        let id = self.alloc_id();
        self.store(id, ps, 1, a.owner)
    }

    fn reveal(&mut self, a: &ShareRef, reveal_to: Option<usize>) -> Fr {
        let ps = self.get_cloned(a);
        crate::crypto::aby3_protocol::reveal(&ps, a.id, reveal_to, self.me, &self.transport)
    }

    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef> {
        let pa = self.get_cloned(a);
        let op_id = self.alloc_id();
        let bit_shares = crate::crypto::aby3_protocol::a2b_party(&pa, nbits, op_id, self.me, &self.transport);
        bit_shares
            .into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.shares.insert(id, ps);
                ShareRef {
                    id,
                    nbits: 1,
                    local: None,
                    owner: a.owner,
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
            local: Some(Share3::zero()),
            owner: 0,
        };
        let one: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            local: Some(Share3::one()),
            owner: 0,
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
            local: None,
            owner: 0,
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
        let mut result = vec![ShareRef::new_local(Share3::zero(), 1, a.owner); n];
        let bit_refs_a = self.a2b(a, a.nbits);
        // copy the bits shifted to the right
        for item in bit_refs_a.iter().take(bit_refs_a.len().saturating_sub(n)) {
            result.push(item.clone());
        }
        self.b2a(&result)
    }
}
