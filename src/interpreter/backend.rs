use std::sync::atomic::{AtomicU64, Ordering};

use crate::crypto::types::Fr;
use crate::crypto::{arith, sim};

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
