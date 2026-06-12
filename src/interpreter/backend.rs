use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;

use crate::crypto::types::Fr;
use crate::crypto::{arith, bits, sim};
use crate::crypto::net::{Transport, PartyId};
use crate::crypto::protocol::{self, PartyShare};

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
        ShareRef { id, nbits, local: Some(share), owner }
    }
}

/// Minimal synchronous backend trait for now.
pub trait Backend: Send + Sync {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef;
    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    /// Reveal a shared value to one or all participants. `reveal_to = None` -> reveal to all.
    fn reveal(&mut self, a: &ShareRef, reveal_to: Option<usize>) -> Fr;
    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef>;
    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef;
}

/// Single-process simulation backend backed by current sim/inproc wrappers.
pub struct SimBackend {
    // no state for now; RNG could be added later
}

impl SimBackend {
    pub fn new() -> Self { SimBackend {} }
}

impl Backend for SimBackend {
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
        let prod = sim::mul_ab3_with_inproc(&sa, &sb);
        ShareRef::new_local(prod, a.nbits.max(b.nbits), a.owner)
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
        let bits = sim::a2b_with_inproc(&sa, nbits);
        bits.into_iter().map(|b| ShareRef::new_local(b, 1, a.owner)).collect()
    }
    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        // reconstruct bit values and assemble a shared number by scalar-multiplying
        // each bit share by 1<<i and summing
        let mut out = arith::share_secret(0);
        for (i, bit_ref) in bits.iter().enumerate() {
            let b = bit_ref.local.clone().expect("expected local bit share");
            let term = crate::crypto::arith::scalar_mul(&b, 1u128 << i);
            out = crate::crypto::arith::add_shares(&out, &term);
        }
        ShareRef::new_local(out, bits.get(0).map(|s| s.nbits).unwrap_or(0), bits.get(0).map(|s| s.owner).unwrap_or(0))
    }
}

/// ABY3 backend that runs the online protocol via explicit in-process message
/// exchanges between three simulated parties.
pub struct Aby3Backend {}

impl Aby3Backend {
    pub fn new() -> Self { Aby3Backend {} }
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
        bits.into_iter().map(|b| ShareRef::new_local(b, 1, a.owner)).collect()
    }

    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        let mut out = arith::share_secret(0);
        for (i, bit_ref) in bits.iter().enumerate() {
            let b = bit_ref.local.clone().expect("expected local bit share");
            let term = crate::crypto::arith::scalar_mul(&b, 1u128 << i);
            out = crate::crypto::arith::add_shares(&out, &term);
        }
        ShareRef::new_local(out, bits.get(0).map(|s| s.nbits).unwrap_or(0), bits.get(0).map(|s| s.owner).unwrap_or(0))
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
}

impl<T: Transport> TransportBackend<T> {
    pub fn new(me: PartyId, transport: T) -> Self {
        TransportBackend { me, transport, shares: HashMap::new() }
    }

    fn store(&mut self, id: u64, ps: PartyShare, nbits: usize, owner: usize) -> ShareRef {
        self.shares.insert(id, ps);
        ShareRef { id, nbits, local: None, owner }
    }

    fn get(&self, sr: &ShareRef) -> &PartyShare {
        self.shares.get(&sr.id).unwrap_or_else(|| panic!("TransportBackend: unknown share id {}", sr.id))
    }

    fn get_cloned(&self, sr: &ShareRef) -> PartyShare {
        self.get(sr).clone()
    }
}

impl<T: Transport + 'static> Backend for TransportBackend<T> {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let ps = protocol::share_input(value, id, owner, self.me, &self.transport);
        self.store(id, ps, nbits, owner)
    }

    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = protocol::add_party(&pa, &pb);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = protocol::sub_party(&pa, &pb);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = protocol::mul_party(&pa, &pb, self.me, &self.transport);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        // eq(a,b) = NOT(OR(bits of a-b))
        let nbits = a.nbits.max(b.nbits);
        let diff = self.sub(a, b);
        let bit_refs = self.a2b(&diff, nbits);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        // OR all bits
        let zero = protocol::share_public(0, self.me);
        let mut acc_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        self.shares.insert(acc_id, zero);
        for br in &bit_refs {
            let pa = self.get_cloned(&ShareRef { id: acc_id, nbits: 1, local: None, owner: a.owner });
            let pb = self.get_cloned(br);
            // OR via a+b-2ab (local, bits are public after reveal or just arithmetic)
            let two_ab = protocol::scalar_mul_party(&protocol::add_party(&pa, &pb), 0); // placeholder
            let _ = two_ab;
            // simpler: just OR arithmetically
            let or_ps = {
                let sum = protocol::add_party(&pa, &pb);
                let prod = protocol::mul_party(&pa, &pb, self.me, &self.transport);
                protocol::sub_party(&sum, &prod)
            };
            let new_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
            self.shares.insert(new_id, or_ps);
            acc_id = new_id;
        }
        // NOT = 1 - acc
        let one = protocol::share_public(1, self.me);
        let one_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        self.shares.insert(one_id, one);
        let acc_ps = self.get_cloned(&ShareRef { id: acc_id, nbits: 1, local: None, owner: a.owner });
        let one_ps = self.get_cloned(&ShareRef { id: one_id, nbits: 1, local: None, owner: a.owner });
        let not_ps = protocol::sub_party(&one_ps, &acc_ps);
        self.store(id, not_ps, 1, a.owner)
    }

    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        // lt(a,b) = MSB of (a - b) under two's-complement
        let nbits = a.nbits.max(b.nbits);
        let diff = self.sub(a, b);
        let bit_refs = self.a2b(&diff, nbits);
        let msb = bit_refs.last().cloned().expect("a2b produced no bits");
        msb
    }

    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let lt = self.lt(a, b);
        let eq = self.eq(a, b);
        // OR(lt, eq)
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let pa = self.get_cloned(&lt);
        let pb = self.get_cloned(&eq);
        let sum = protocol::add_party(&pa, &pb);
        let prod = protocol::mul_party(&pa, &pb, self.me, &self.transport);
        let ps = protocol::sub_party(&sum, &prod);
        self.store(id, ps, 1, a.owner)
    }

    fn reveal(&mut self, a: &ShareRef, reveal_to: Option<usize>) -> Fr {
        let ps = self.get_cloned(a);
        protocol::reveal(&ps, a.id, reveal_to, self.me, &self.transport)
    }

    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef> {
        let pa = self.get_cloned(a);
        let bit_shares = protocol::a2b_party(&pa, nbits, self.me, &self.transport);
        bit_shares.into_iter().map(|ps| {
            let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
            self.shares.insert(id, ps);
            ShareRef { id, nbits: 1, local: None, owner: a.owner }
        }).collect()
    }

    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        // b2a: sum_i bit_i * 2^i  (local, no communication needed)
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let zero = protocol::share_public(0, self.me);
        let mut acc = zero;
        for (i, br) in bits.iter().enumerate() {
            let bp = self.get_cloned(br);
            let term = protocol::scalar_mul_party(&bp, 1u128 << i);
            acc = protocol::add_party(&acc, &term);
        }
        self.store(id, acc, bits.get(0).map(|s| s.nbits).unwrap_or(0), bits.get(0).map(|s| s.owner).unwrap_or(0))
    }
}
