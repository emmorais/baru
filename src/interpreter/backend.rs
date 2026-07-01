use crate::crypto::types::Fr;


#[derive(Debug, Clone)]
pub struct ShareRef {
    pub id: u64,
    pub nbits: usize,
    pub owner: usize,
}

impl ShareRef {
    pub fn new(id: u64, nbits: usize, owner: usize) -> Self {
        ShareRef { id, nbits, owner }
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


