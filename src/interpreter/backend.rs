use crate::crypto::types::Fr;

#[derive(Debug, Clone)]
pub struct ShareRef {
    pub id: u64,
    pub nbits: usize,
    pub owner: usize,
    /// True when the shared value is a Q16.16 fixed-point number (the share
    /// holds the scaled integer).  Metadata only — backends ignore it; the
    /// interpreter uses it to insert rescales and coerce mixed operands.
    pub fixed: bool,
}

impl ShareRef {
    pub fn new(id: u64, nbits: usize, owner: usize) -> Self {
        ShareRef {
            id,
            nbits,
            owner,
            fixed: false,
        }
    }
}

/// Minimal synchronous backend trait for now.
pub trait Backend: Send + Sync {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef;
    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;

    /// Secret/secret division and modulo: `(a / b, a % b)`, both shared.
    ///
    /// Default implementation: a restoring binary long-division circuit,
    /// extended to Baru's signed two's-complement shares via a sign/magnitude
    /// reduction, so results match Rust's truncating `/` and `%` (quotient
    /// truncates toward zero; the remainder takes the dividend's sign, so
    /// `a == b*q + r` always holds).
    ///
    /// This composes entirely from other `Backend` primitives — `a2b`, `mul`,
    /// `add`, `sub`, `le`, `b2a` — so every backend gets a working `divmod`
    /// "for free" without needing its own implementation.
    ///
    /// In particular, this does **not** need to shift by a *shared* amount
    /// (no such primitive exists in this trait, and none is required): only
    /// the accumulating remainder is shifted, and always by the *public*
    /// constant 1 (a local self-add); the shared divisor itself is compared
    /// and conditionally subtracted at its own, unshifted magnitude. That is
    /// the standard MPC restoring-division building block, and it only needs
    /// primitives this trait already has.
    ///
    /// Division by a (secret) zero divisor is **not** checked: checking would
    /// require revealing whether `b == 0`, which this method does not do.
    /// Callers that need to guard against it should perform an explicit
    /// `eq`-based zero-check first.
    fn divmod(&mut self, a: &ShareRef, b: &ShareRef) -> (ShareRef, ShareRef) {
        let nbits = a.nbits.max(b.nbits);

        // Two's-complement sign bits (the top bit of a full decomposition).
        let a_bits = self.a2b(a, nbits);
        let b_bits = self.a2b(b, nbits);
        let sign_a = a_bits[nbits - 1].clone();
        let sign_b = b_bits[nbits - 1].clone();

        // abs(x) = x - 2*sign_x*x  (negates x exactly when sign_x == 1).
        let ta = self.mul(&sign_a, a);
        let two_ta = self.add(&ta, &ta);
        let abs_a = self.sub(a, &two_ta);

        let tb = self.mul(&sign_b, b);
        let two_tb = self.add(&tb, &tb);
        let abs_b = self.sub(b, &two_tb);

        // Unsigned restoring division of the magnitudes, MSB -> LSB.
        let abs_a_bits = self.a2b(&abs_a, nbits);
        let mut quotient_bits: Vec<ShareRef> = Vec::with_capacity(nbits);

        // Seed the remainder with the top bit (avoids needing a "zero" constant).
        let mut remainder = abs_a_bits[nbits - 1].clone();
        {
            let cond = self.le(&abs_b, &remainder); // 1 iff abs_b <= remainder
            let sub_amt = self.mul(&cond, &abs_b);
            remainder = self.sub(&remainder, &sub_amt);
            quotient_bits.push(cond); // bit (nbits-1)
        }
        for i in (0..nbits - 1).rev() {
            let doubled = self.add(&remainder, &remainder); // remainder <<= 1 (public shift by 1)
            let brought_down = self.add(&doubled, &abs_a_bits[i]);
            let cond = self.le(&abs_b, &brought_down);
            let sub_amt = self.mul(&cond, &abs_b);
            remainder = self.sub(&brought_down, &sub_amt);
            quotient_bits.push(cond); // bit i
        }
        // quotient_bits was pushed MSB -> LSB; b2a expects LSB-first (bit 0 first).
        quotient_bits.reverse();
        let abs_q = self.b2a(&quotient_bits);

        // Truncating semantics: quotient sign = sign_a XOR sign_b; remainder's
        // sign follows the dividend (sign_a) — together these guarantee
        // `a == b*quotient + remainder`, matching Rust's `/` and `%`.
        let sxa = self.mul(&sign_a, &sign_b);
        let sum_signs = self.add(&sign_a, &sign_b);
        let two_sxa = self.add(&sxa, &sxa);
        let sign_q = self.sub(&sum_signs, &two_sxa); // XOR(sign_a, sign_b) = a+b-2ab

        let tq = self.mul(&sign_q, &abs_q);
        let two_tq = self.add(&tq, &tq);
        let quotient = self.sub(&abs_q, &two_tq);

        let tr = self.mul(&sign_a, &remainder);
        let two_tr = self.add(&tr, &tr);
        let rem = self.sub(&remainder, &two_tr);

        (quotient, rem)
    }

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
