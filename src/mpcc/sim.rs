//! In-process protocol simulators: the first two rows of the registry that
//! Baru can actually *run* through the bytecode VM.
//!
//! * [`DealerEngine`] — `n`-party additive sharing over `Z_{2^k}` with Beaver
//!   triples.  The dealer model is not a threat model anyone deploys; it is the
//!   **online-cost lower bound** for additive sharing, and therefore the most
//!   useful single baseline a cost model can be calibrated against.  Semi2k and
//!   SPDZ2k have exactly this online phase — they differ only in where the
//!   triples come from and whether shares carry MACs.
//! * [`ShamirEngine`] — `n`-party threshold sharing over a prime field with
//!   GRR degree reduction.  Structurally different from the dealer in the way
//!   that matters for costing: multiplication sends *one* element per peer
//!   instead of two, and consumes no triples at all.
//!
//! Both hold every party's view in one process.  Sharings are genuinely split
//! and genuinely reconstructed, so a protocol bug is a wrong answer; what is
//! simulated rather than executed is preprocessing, which is drawn locally and
//! counted (see [`crate::mpcc::vm`] for exactly what that does and does not
//! make trustworthy).

use super::vm::{random_elem, Counters, Modulus, ShareEngine};

// ─── Dealer: additive sharing over Z_{2^k} ────────────────────────────────────

/// Additive `n`-out-of-`n` sharing: the components sum to the secret.
pub type Additive = Vec<u64>;

pub struct DealerEngine {
    m: Modulus,
    parties: usize,
    counters: Counters,
    trunc_pairs: bool,
}

impl DealerEngine {
    pub fn new(parties: usize, bits: u32) -> Result<Self, String> {
        if parties < 2 {
            return Err("dealer: need at least 2 parties".to_string());
        }
        Ok(DealerEngine {
            m: Modulus::Ring { k: bits },
            parties,
            counters: Counters::default(),
            trunc_pairs: false,
        })
    }

    /// Rescale with truncation pairs instead of a bit decomposition, so that the
    /// accuracy cost of stochastic rounding can be measured in-process across many
    /// inputs without paying for a distributed run each time.
    pub fn with_trunc_pairs(mut self, on: bool) -> Self {
        self.trunc_pairs = on;
        self
    }

    /// Bytes one party sends when every party sends `elems` elements to every
    /// peer.  This is the convention the cost model predicts in, so the two
    /// numbers can be compared directly.
    fn charge(&mut self, elems: u64) {
        self.counters.online_bytes += elems * self.m.element_bytes() * (self.parties as u64 - 1);
    }

    fn split(&self, value: u64) -> Additive {
        let mut parts: Vec<u64> = (0..self.parties - 1).map(|_| random_elem(self.m)).collect();
        let sum = parts.iter().fold(0u64, |acc, x| self.m.add(acc, *x));
        parts.push(self.m.sub(value, sum));
        parts
    }

    fn reconstruct(&self, s: &Additive) -> u64 {
        s.iter().fold(0u64, |acc, x| self.m.add(acc, *x))
    }

    /// A Beaver triple `(a, b, ab)`.  Drawn locally and counted — see the module
    /// docs on what "fake offline" does and does not measure.
    fn triple(&mut self) -> (Additive, Additive, Additive) {
        let a = random_elem(self.m);
        let b = random_elem(self.m);
        self.counters.triples += 1;
        (self.split(a), self.split(b), self.split(self.m.mul(a, b)))
    }
}

impl ShareEngine for DealerEngine {
    fn trunc_pair_mode(&self) -> bool {
        self.trunc_pairs
    }

    type Share = Additive;

    fn modulus(&self) -> Modulus {
        self.m
    }
    fn parties(&self) -> usize {
        self.parties
    }
    fn counters(&self) -> &Counters {
        &self.counters
    }
    fn counters_mut(&mut self) -> &mut Counters {
        &mut self.counters
    }

    fn input(&mut self, value: u64, _party: usize) -> Additive {
        // The input owner sends one component to each peer.
        self.charge(1);
        self.split(value)
    }

    fn constant(&mut self, value: u64) -> Additive {
        let mut s = vec![0u64; self.parties];
        s[0] = value;
        s
    }

    fn add(&mut self, a: &Additive, b: &Additive) -> Additive {
        a.iter().zip(b).map(|(x, y)| self.m.add(*x, *y)).collect()
    }

    fn sub(&mut self, a: &Additive, b: &Additive) -> Additive {
        a.iter().zip(b).map(|(x, y)| self.m.sub(*x, *y)).collect()
    }

    fn add_const(&mut self, a: &Additive, c: u64) -> Additive {
        // Only one party adds the constant, or the sum would be off by `n·c`.
        let mut out = a.clone();
        out[0] = self.m.add(out[0], c);
        out
    }

    fn mul_const(&mut self, a: &Additive, c: u64) -> Additive {
        a.iter().map(|x| self.m.mul(*x, c)).collect()
    }

    fn mul_batch(&mut self, pairs: &[(Additive, Additive)]) -> Vec<Additive> {
        // Beaver: open d = x − a and e = y − b, then z = ab + d·b + e·a + d·e.
        // Both openings ride in this one round, which is why the round is
        // charged as two elements per peer rather than one.
        self.charge(2 * pairs.len() as u64);
        self.counters.muls += pairs.len() as u64;

        let mut out = Vec::with_capacity(pairs.len());
        for (x, y) in pairs {
            let (a, b, ab) = self.triple();
            let d = self.reconstruct(&self.sub_pure(x, &a));
            let e = self.reconstruct(&self.sub_pure(y, &b));
            let db = self.mul_const_pure(&b, d);
            let ea = self.mul_const_pure(&a, e);
            let mut z = self.add_pure(&ab, &db);
            z = self.add_pure(&z, &ea);
            z[0] = self.m.add(z[0], self.m.mul(d, e));
            out.push(z);
        }
        out
    }

    fn open_batch(&mut self, values: &[Additive]) -> Vec<u64> {
        self.charge(values.len() as u64);
        self.counters.opens += values.len() as u64;
        values.iter().map(|s| self.reconstruct(s)).collect()
    }

    /// Drawn locally, and this is the *definition* rather than a simplification:
    /// a trusted dealer hands over correlated randomness for free, which is what
    /// makes this engine the online lower bound. Every other engine generates
    /// its bits through `vm::random_bits_batch`; the gap between them and this
    /// one is exactly the price of not having a dealer.
    fn random_bits(&mut self, n: usize) -> Vec<Additive> {
        self.counters.random_bits += n as u64;
        (0..n)
            .map(|_| self.split(u64::from(rand::random::<bool>())))
            .collect()
    }
}

impl DealerEngine {
    // Non-`&mut self` duplicates, so `mul_batch` can compose local operations
    // while it already holds a mutable borrow for the counters.
    fn add_pure(&self, a: &Additive, b: &Additive) -> Additive {
        a.iter().zip(b).map(|(x, y)| self.m.add(*x, *y)).collect()
    }
    fn sub_pure(&self, a: &Additive, b: &Additive) -> Additive {
        a.iter().zip(b).map(|(x, y)| self.m.sub(*x, *y)).collect()
    }
    fn mul_const_pure(&self, a: &Additive, c: u64) -> Additive {
        a.iter().map(|x| self.m.mul(*x, c)).collect()
    }
}

// ─── Shamir: threshold sharing over a prime field ─────────────────────────────

/// `shares[i]` is `f(i+1)` for a polynomial `f` with `f(0)` the secret.
pub type Shamir = Vec<u64>;

pub struct ShamirEngine {
    m: Modulus,
    parties: usize,
    /// Privacy threshold: any `t` parties learn nothing, `t+1` reconstruct.
    t: usize,
    /// Lagrange coefficients at 0 for the evaluation points `1..=n`.
    lambda: Vec<u64>,
    counters: Counters,
}

impl ShamirEngine {
    /// `t` defaults to the largest threshold an honest majority supports,
    /// `floor((n-1)/2)`, which is also what degree reduction needs
    /// (`n >= 2t + 1`).
    pub fn new(parties: usize) -> Result<Self, String> {
        if parties < 3 {
            return Err("shamir: need at least 3 parties for an honest majority".to_string());
        }
        let m = Modulus::MERSENNE61;
        let t = (parties - 1) / 2;
        let lambda = lagrange_at_zero(m, parties)?;
        Ok(ShamirEngine {
            m,
            parties,
            t,
            lambda,
            counters: Counters::default(),
        })
    }

    pub fn threshold(&self) -> usize {
        self.t
    }

    fn charge(&mut self, elems: u64) {
        self.counters.online_bytes += elems * self.m.element_bytes() * (self.parties as u64 - 1);
    }

    /// Share `value` with a random polynomial of the given degree.
    fn share_degree(&self, value: u64, degree: usize) -> Shamir {
        let coeffs: Vec<u64> = (0..degree).map(|_| random_elem(self.m)).collect();
        (1..=self.parties)
            .map(|x| {
                let x = x as u64;
                // Horner, from the top coefficient down to the secret.
                let mut acc = 0u64;
                for c in coeffs.iter().rev() {
                    acc = self.m.add(self.m.mul(acc, x), *c);
                }
                self.m.add(self.m.mul(acc, x), value)
            })
            .collect()
    }

    /// Interpolate `f(0)` from all `n` evaluations.  Correct for any `f` of
    /// degree `<= n - 1`, which covers both a normal sharing (degree `t`) and
    /// the degree-`2t` product before reduction.
    fn interpolate(&self, s: &Shamir) -> u64 {
        s.iter()
            .zip(&self.lambda)
            .fold(0u64, |acc, (y, l)| self.m.add(acc, self.m.mul(*y, *l)))
    }

    pub fn reconstruct(&self, s: &Shamir) -> u64 {
        self.interpolate(s)
    }
}

/// `lambda[i]` such that `f(0) = sum_i lambda[i] * f(i+1)` for `deg f < n`.
fn lagrange_at_zero(m: Modulus, n: usize) -> Result<Vec<u64>, String> {
    let xs: Vec<u64> = (1..=n as u64).collect();
    let mut out = Vec::with_capacity(n);
    for (i, xi) in xs.iter().enumerate() {
        let mut num = 1u64;
        let mut den = 1u64;
        for (j, xj) in xs.iter().enumerate() {
            if i == j {
                continue;
            }
            num = m.mul(num, *xj);
            den = m.mul(den, m.sub(*xj, *xi));
        }
        out.push(m.mul(num, m.inv(den)?));
    }
    Ok(out)
}

impl ShareEngine for ShamirEngine {
    type Share = Shamir;

    fn modulus(&self) -> Modulus {
        self.m
    }
    fn parties(&self) -> usize {
        self.parties
    }
    fn counters(&self) -> &Counters {
        &self.counters
    }
    fn counters_mut(&mut self) -> &mut Counters {
        &mut self.counters
    }

    fn input(&mut self, value: u64, _party: usize) -> Shamir {
        self.charge(1);
        self.share_degree(value, self.t)
    }

    fn constant(&mut self, value: u64) -> Shamir {
        // The constant polynomial is a valid degree-<=t sharing.
        vec![value; self.parties]
    }

    fn add(&mut self, a: &Shamir, b: &Shamir) -> Shamir {
        a.iter().zip(b).map(|(x, y)| self.m.add(*x, *y)).collect()
    }

    fn sub(&mut self, a: &Shamir, b: &Shamir) -> Shamir {
        a.iter().zip(b).map(|(x, y)| self.m.sub(*x, *y)).collect()
    }

    fn add_const(&mut self, a: &Shamir, c: u64) -> Shamir {
        // Unlike additive sharing, *every* party adds: it shifts f by c.
        a.iter().map(|x| self.m.add(*x, c)).collect()
    }

    fn mul_const(&mut self, a: &Shamir, c: u64) -> Shamir {
        a.iter().map(|x| self.m.mul(*x, c)).collect()
    }

    fn mul_batch(&mut self, pairs: &[(Shamir, Shamir)]) -> Vec<Shamir> {
        // GRR: the local product is a degree-2t sharing; each party re-shares
        // its own product at degree t and the parties take the Lagrange
        // combination.  One element per peer, and no triples at all — the
        // structural difference from Beaver-based protocols.
        self.charge(pairs.len() as u64);
        self.counters.muls += pairs.len() as u64;

        pairs
            .iter()
            .map(|(a, b)| {
                let reshares: Vec<Shamir> = (0..self.parties)
                    .map(|i| self.share_degree(self.m.mul(a[i], b[i]), self.t))
                    .collect();
                (0..self.parties)
                    .map(|j| {
                        (0..self.parties).fold(0u64, |acc, i| {
                            self.m.add(acc, self.m.mul(self.lambda[i], reshares[i][j]))
                        })
                    })
                    .collect()
            })
            .collect()
    }

    fn open_batch(&mut self, values: &[Shamir]) -> Vec<u64> {
        self.charge(values.len() as u64);
        self.counters.opens += values.len() as u64;
        values.iter().map(|s| self.interpolate(s)).collect()
    }

    /// Generated rather than dealt (`vm::random_bits_batch`) — Shamir has a real
    /// threat model, so its preprocessing has to cost something. XOR-reducing
    /// one private bit per party is `⌈log₂(parties)⌉` rounds, so unlike the
    /// two-party engines this is 2 rounds at `n = 3`.
    fn random_bits(&mut self, n: usize) -> Vec<Shamir> {
        super::vm::random_bits_batch(self, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dealer_shares_hide_the_secret_and_reconstruct_it() {
        let e = DealerEngine::new(3, 64).unwrap();
        let v = e.m.from_i64(-1234);
        let s = e.split(v);
        assert_eq!(s.len(), 3);
        assert!(
            s.iter().all(|c| *c != v),
            "no single component may equal the secret"
        );
        assert_eq!(e.reconstruct(&s), v);
    }

    #[test]
    fn dealer_multiplication_is_correct_over_signed_values() {
        let mut e = DealerEngine::new(3, 64).unwrap();
        for (a, b) in [(6i64, 7i64), (-6, 7), (-6, -7), (0, 5), (123456, -789)] {
            let (sa, sb) = (e.input(e.m.from_i64(a), 0), e.input(e.m.from_i64(b), 1));
            let prod = e.mul_batch(&[(sa, sb)]).remove(0);
            let got = e.open_batch(&[prod])[0];
            assert_eq!(e.m.to_i64(got, 64), a * b, "{} * {}", a, b);
        }
    }

    #[test]
    fn dealer_counts_two_elements_per_peer_for_a_multiplication() {
        let mut e = DealerEngine::new(3, 64).unwrap();
        let a = e.input(1, 0);
        let b = e.input(2, 1);
        let before = e.counters().online_bytes;
        e.mul_batch(&[(a, b)]);
        // 2 openings x 8 bytes x 2 peers.
        assert_eq!(e.counters().online_bytes - before, 2 * 8 * 2);
        assert_eq!(e.counters().triples, 1);
    }

    #[test]
    fn shamir_reconstructs_and_multiplies_at_several_party_counts() {
        for n in [3usize, 4, 5, 7] {
            let mut e = ShamirEngine::new(n).unwrap();
            assert_eq!(e.threshold(), (n - 1) / 2);
            for (a, b) in [(6i64, 7i64), (-6, 7), (-6, -7), (99999, -3)] {
                let sa = e.input(e.m.from_i64(a), 0);
                let sb = e.input(e.m.from_i64(b), 1);
                let sum = e.add(&sa, &sb);
                assert_eq!(e.m.to_i64(e.reconstruct(&sum), 32), a + b, "n={}", n);

                let prod = e.mul_batch(&[(sa, sb)]).remove(0);
                let got = e.open_batch(&[prod])[0];
                assert_eq!(e.m.to_i64(got, 32), a * b, "n={} {} * {}", n, a, b);
            }
        }
    }

    #[test]
    fn shamir_multiplication_needs_no_triples_and_sends_one_element_per_peer() {
        let mut e = ShamirEngine::new(3).unwrap();
        let a = e.input(1, 0);
        let b = e.input(2, 1);
        let before = e.counters().online_bytes;
        e.mul_batch(&[(a, b)]);
        assert_eq!(e.counters().online_bytes - before, 8 * 2);
        assert_eq!(e.counters().triples, 0);
    }

    #[test]
    fn a_threshold_subset_is_enough_but_the_degree_must_be_right() {
        // Degree-2t products cannot be opened from t+1 shares; the whole point
        // of GRR degree reduction is that the result can be.
        let mut e = ShamirEngine::new(5).unwrap();
        let a = e.input(e.m.from_i64(11), 0);
        let b = e.input(e.m.from_i64(13), 1);
        let reduced = e.mul_batch(&[(a, b)]).remove(0);
        // Interpolating the reduced sharing from any t+1 = 3 points agrees.
        let sub = lagrange_subset(e.m, &[1, 2, 3]);
        let got = (0..3).fold(0u64, |acc, i| e.m.add(acc, e.m.mul(sub[i], reduced[i])));
        assert_eq!(e.m.to_i64(got, 32), 143);
    }

    /// Lagrange coefficients at 0 for an arbitrary subset of evaluation points.
    fn lagrange_subset(m: Modulus, xs: &[u64]) -> Vec<u64> {
        xs.iter()
            .enumerate()
            .map(|(i, xi)| {
                let (mut num, mut den) = (1u64, 1u64);
                for (j, xj) in xs.iter().enumerate() {
                    if i != j {
                        num = m.mul(num, *xj);
                        den = m.mul(den, m.sub(*xj, *xi));
                    }
                }
                m.mul(num, m.inv(den).unwrap())
            })
            .collect()
    }
}
