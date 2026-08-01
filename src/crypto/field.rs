//! Arithmetic in a small prime field `Z_p`, and 3-party replicated sharing over it.
//!
//! # Why this exists
//!
//! Every other sharing in this crate lives in `Z_{2^k}`: [`crate::crypto::types::Fr`]
//! is a `u128` and the protocols use `wrapping_add`/`wrapping_mul`, so the
//! modulus is a power of two and reduction is free. That is the right choice
//! for fixed-point arithmetic and it is what makes the BFV rescale a shift.
//!
//! It is the wrong structure for one specific thing. Falcon's private-compare
//! (`Π_PC`, Algorithm 1 — see [falcon-reproduction.md](../../docs/falcon-reproduction.md))
//! forms a set of terms and asks whether **any of them is zero**. That question
//! is only meaningful without zero divisors, and `Z_{2^k}` is full of them:
//! `2^{k-1} · 2 = 0`. So the private compare — and therefore `wrap3`, and
//! therefore Falcon's DReLU — needs a genuine field. Falcon uses `p = 67`.
//!
//! This module is deliberately *not* generic over the ring. It is a small,
//! exhaustively testable field, kept separate from the `Z_{2^k}` path rather
//! than folded into it, because conflating the two is exactly how a zero-divisor
//! bug would get in.
//!
//! # On Galois rings
//!
//! `GR(2^k, d)` is the natural way to get an exceptional set over a
//! characteristic-`2^k` ring, and is what the large-scale iris work
//! (eprint 2024/705) uses for Shamir sharing. It is **not** a substitute for
//! this module: for `k > 1` a Galois ring still has zero divisors, so it cannot
//! host `Π_PC` either. Only the residue field `GR(2, d) = F_{2^d}` is a field,
//! and whether `Π_PC`'s algebra survives characteristic 2 is an open question
//! recorded in the plan, not an assumption made here.

use crate::crypto::types::Fr;

/// The private-compare modulus (Algorithm 1).
///
/// # Choosing it
///
/// `Π_PC` forms `c[i] = u[i] + 1 + Σ_{k>i} w[k]` with `u[i] ∈ {−1,0,1}` and at
/// most `ℓ` unit terms in the sum, so the terms live in roughly `[0, ℓ+2]`. The
/// zero test is only meaningful if none of them wraps, which fixes the
/// requirement: **`p > ℓ + 2`**.
///
/// So the modulus is a function of the *ring* width being compared, not a
/// universal constant. **Falcon uses `p = 37` with `ℓ = 32`** (§3.1: "we use
/// ℓ = 2^5, p = 37", with 13 bits of fixed-point precision); `p = 67` is
/// SecureNN's choice, sized for `ℓ = 64`.
///
/// `67` is kept here because it satisfies the bound for every `ℓ ≤ 64` and so
/// covers both parameter sets, and because it is small enough to test the whole
/// field exhaustively. **The faithful-reproduction benchmark must use Falcon's
/// own `ℓ = 32`**, since their communication advantage over prior work comes
/// largely from the smaller ring — see
/// [falcon-reproduction.md](../../docs/falcon-reproduction.md).
pub const P: u64 = 67;

/// Falcon's own private-compare modulus, for the faithful parameter set.
pub const P_FALCON: u64 = 37;

/// The largest ring width `p` can safely serve: `p > ℓ + 2`.
pub const fn max_ring_bits(p: u64) -> u64 {
    p - 3
}

/// An element of `Z_p`, always held reduced.
///
/// A newtype rather than a bare `u64` on purpose: the rest of the crate passes
/// `Fr = u128` ring elements around, and the one thing that must never happen is
/// a ring element being used where a field element is meant. The type system is
/// cheaper than the test that would otherwise have to catch it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fp(u64);

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);

    /// Reduce any integer into the field.
    pub fn new(v: u64) -> Fp {
        Fp(v % P)
    }

    /// Reduce a ring element into the field. Used where a `Z_{2^k}` share has
    /// to be reinterpreted, which is precisely the boundary worth naming.
    pub fn from_fr(v: Fr) -> Fp {
        Fp((v % P as Fr) as u64)
    }

    pub fn value(self) -> u64 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// `self^e` by square-and-multiply.
    pub fn pow(self, mut e: u64) -> Fp {
        let (mut base, mut acc) = (self, Fp::ONE);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }

    /// Multiplicative inverse, or `None` for zero.
    ///
    /// By Fermat: `a^(p−2) = a^{−1}` for `a ≠ 0`. Being a *field* is the whole
    /// reason this module exists, so the inverse is total on non-zero elements —
    /// that is the property `Z_{2^k}` cannot offer and the one `Π_PC` needs.
    pub fn inv(self) -> Option<Fp> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(P - 2))
        }
    }
}

impl std::ops::Add for Fp {
    type Output = Fp;
    fn add(self, o: Fp) -> Fp {
        let s = self.0 + o.0;
        Fp(if s >= P { s - P } else { s })
    }
}

impl std::ops::Sub for Fp {
    type Output = Fp;
    fn sub(self, o: Fp) -> Fp {
        Fp(if self.0 >= o.0 { self.0 - o.0 } else { self.0 + P - o.0 })
    }
}

impl std::ops::Neg for Fp {
    type Output = Fp;
    fn neg(self) -> Fp {
        Fp(if self.0 == 0 { 0 } else { P - self.0 })
    }
}

impl std::ops::Mul for Fp {
    type Output = Fp;
    /// `P` is far below `2^32`, so the product cannot overflow `u64`.
    fn mul(self, o: Fp) -> Fp {
        Fp((self.0 * o.0) % P)
    }
}

/// One party's replicated share of a field element: parties hold `(sᵢ, sᵢ₊₁)`
/// of a 3-out-of-3 additive sharing `x = s₀ + s₁ + s₂ (mod p)`, so any two
/// parties jointly hold all three summands — the same 2-out-of-3 redundancy
/// [`crate::crypto::aby3_protocol`] uses over `Z_{2^k}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FpShare {
    pub left: Fp,
    pub right: Fp,
}

impl FpShare {
    pub fn new(left: Fp, right: Fp) -> Self {
        FpShare { left, right }
    }

    /// Multiplication by a public constant — local, as in the ring case.
    pub fn scalar_mul(self, k: Fp) -> FpShare {
        FpShare { left: self.left * k, right: self.right * k }
    }
}

/// Addition of shares is local: communication-free, exactly as over `Z_{2^k}`.
impl std::ops::Add for FpShare {
    type Output = FpShare;
    fn add(self, o: FpShare) -> FpShare {
        FpShare { left: self.left + o.left, right: self.right + o.right }
    }
}

impl std::ops::Sub for FpShare {
    type Output = FpShare;
    fn sub(self, o: FpShare) -> FpShare {
        FpShare { left: self.left - o.left, right: self.right - o.right }
    }
}

impl std::ops::Neg for FpShare {
    type Output = FpShare;
    fn neg(self) -> FpShare {
        FpShare { left: -self.left, right: -self.right }
    }
}

/// Split `x` into a 3-party replicated sharing using the supplied randomness.
///
/// Returned in party order: party `i` gets `(sᵢ, sᵢ₊₁)`.
pub fn share(x: Fp, r0: Fp, r1: Fp) -> [FpShare; 3] {
    let s0 = r0;
    let s1 = r1;
    let s2 = x - s0 - s1;
    [
        FpShare::new(s0, s1),
        FpShare::new(s1, s2),
        FpShare::new(s2, s0),
    ]
}

/// Reconstruct from the three parties' shares.
///
/// Reads one summand per party rather than summing every held value: each
/// summand is held twice, so adding all six would double the secret. That is
/// the classic replicated-sharing mistake and the reason this is a named
/// function rather than an inline fold.
pub fn reconstruct(shares: &[FpShare; 3]) -> Fp {
    shares[0].left + shares[1].left + shares[2].left
}

/// Multiply two shared field elements — the `Z_p` analogue of
/// [`crate::crypto::aby3_protocol::mul_party`].
///
/// Each party forms the three cross terms it can see and re-randomises with a
/// zero-sum triple; summed over parties these are all nine products, i.e. `ab`.
/// The result is a 3-out-of-3 sharing that becomes replicated once each party
/// forwards its summand to one neighbour — one round, one field element per
/// party, exactly as over `Z_{2^k}`.
///
/// `zero` must sum to zero in the field; that is what keeps the output sharing
/// independent of the inputs. Passing a non-zero-sum triple silently corrupts
/// the result, so it is checked in debug builds.
pub fn mul_shares(a: &[FpShare; 3], b: &[FpShare; 3], zero: [Fp; 3]) -> [FpShare; 3] {
    debug_assert_eq!(zero[0] + zero[1] + zero[2], Fp::ZERO, "zero-sharing must sum to 0");
    let t: Vec<Fp> = (0..3)
        .map(|i| {
            a[i].left * b[i].left + a[i].left * b[i].right + a[i].right * b[i].left + zero[i]
        })
        .collect();
    [
        FpShare::new(t[0], t[1]),
        FpShare::new(t[1], t[2]),
        FpShare::new(t[2], t[0]),
    ]
}

/// Add a public field element to a shared one — local, and only party 0 adds it
/// so the sum is not tripled.
pub fn add_public(a: &[FpShare; 3], k: Fp) -> [FpShare; 3] {
    // Summand s₀ is held as `left` by party 0 and as `right` by party 2.
    [
        FpShare::new(a[0].left + k, a[0].right),
        a[1],
        FpShare::new(a[2].left, a[2].right + k),
    ]
}

/// Falcon's private compare, `Π_PC` (Algorithm 1).
///
/// Inputs are bit-shares of `x` over `Z_p` (most significant bit last, i.e.
/// `bits[i]` is bit `i`), a **public** ℓ-bit `r`, a random bit `beta` shared
/// over `Z_p`, and a random non-zero mask `m` shared over `Z_p`. The returned
/// value is `β' `, the public bit revealed in step 7; the protocol's actual
/// output is `β' ⊕ β` over `Z_2`, which the caller forms from its `⟦β⟧₂`.
///
/// The construction is: for each bit position, build `c[i]` so that exactly one
/// `c[i]` is zero precisely when the comparison goes one way. Multiplying all
/// the `c[i]` together and by a random non-zero `m` reveals *only* whether some
/// term was zero — the product is otherwise uniform, which is why `m` is needed
/// and why the field must have no zero divisors. In `Z_{2^k}` a product of
/// non-zero terms can vanish and this leaks nothing reliable; that is the whole
/// reason [`Fp`] exists.
///
/// # The functionality is `x ≥ r`, and **only for `x ≠ r`**
///
/// `β` selects which of two complementary predicates is computed — `β' = (x ≥ r)`
/// at `β = 0`, `β' = (r > x)` at `β = 1` — so `β' ⊕ β = (x ≥ r)` either way, and
/// the blinding hides which was evaluated.
///
/// That equivalence **breaks at `x = r`**. There every `w[k]` is zero and every
/// `u[i]` is zero, so every `c[i]` is 1, the product is non-zero, and `β' = 1`
/// *regardless of `β`* — leaving the output as `1 ⊕ β`, i.e. dependent on the
/// blinding bit rather than on the inputs. The protocol is therefore correct
/// only when `x ≠ r`.
///
/// This is not a defect to fix here: in the only caller that matters (`Π_WA`,
/// Algorithm 2) `x` is a freshly drawn random mask over `Z_L`, so `x = r` has
/// probability `2^-ℓ` — negligible at Falcon's ℓ of 32 or 64. **Any other caller
/// must establish the same thing**, which is why it is stated here rather than
/// left to be rediscovered. `private_compare_degenerates_when_x_equals_r` pins
/// the behaviour so it cannot change silently.
///
/// `zeros` supplies the zero-sharings consumed by the multiplications, in
/// order; `zeros.len()` must be at least `2·ℓ`.
pub fn private_compare(
    bits: &[[FpShare; 3]],
    r: u64,
    beta: &[FpShare; 3],
    m: &[FpShare; 3],
    zeros: &[[Fp; 3]],
) -> bool {
    let l = bits.len();
    assert!(zeros.len() >= 2 * l, "private_compare needs 2·ℓ zero-sharings");
    let rbit = |i: usize| (r >> i) & 1;
    let mut z = 0usize;

    // (−1)^β = 1 − 2β, computed on shares.
    let two = Fp::new(2);
    let sign: [FpShare; 3] = {
        let neg2b = [
            beta[0].scalar_mul(two),
            beta[1].scalar_mul(two),
            beta[2].scalar_mul(two),
        ];
        add_public(&[-neg2b[0], -neg2b[1], -neg2b[2]], Fp::ONE)
    };

    // w[i] = x[i] ⊕ r[i]; with r public this is x[i] or 1 − x[i], both local.
    let w: Vec<[FpShare; 3]> = (0..l)
        .map(|i| {
            if rbit(i) == 0 {
                bits[i]
            } else {
                add_public(&[-bits[i][0], -bits[i][1], -bits[i][2]], Fp::ONE)
            }
        })
        .collect();

    let mut prod: Option<[FpShare; 3]> = None;
    for i in (0..l).rev() {
        // u[i] = (−1)^β · (x[i] − r[i])
        let diff = add_public(&bits[i], -Fp::new(rbit(i)));
        let u = mul_shares(&sign, &diff, zeros[z]);
        z += 1;
        // c[i] = u[i] + 1 + Σ_{k>i} w[k]
        let mut c = add_public(&u, Fp::ONE);
        for wk in w.iter().skip(i + 1) {
            c = [c[0] + wk[0], c[1] + wk[1], c[2] + wk[2]];
        }
        prod = Some(match prod {
            None => c,
            Some(p) => {
                let out = mul_shares(&p, &c, zeros[z]);
                z += 1;
                out
            }
        });
    }

    // d = m · Π c[i], revealed. β' = 1 iff d ≠ 0.
    let d = mul_shares(&prod.expect("ℓ ≥ 1"), m, zeros[z]);
    !reconstruct(&d).is_zero()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The field axioms, checked exhaustively — `P` is 67, so "exhaustively"
    /// is 67² pairs and costs nothing. Anything less than exhaustive would be
    /// a strange choice at this size.
    #[test]
    fn field_axioms_hold_over_the_whole_field() {
        for a in 0..P {
            let a = Fp::new(a);
            assert_eq!(a + Fp::ZERO, a);
            assert_eq!(a * Fp::ONE, a);
            assert_eq!(a + (-a), Fp::ZERO);
            for b in 0..P {
                let b = Fp::new(b);
                assert_eq!(a + b, b + a, "add commutes");
                assert_eq!(a * b, b * a, "mul commutes");
                assert_eq!(a - b + b, a, "sub inverts add");
                for c in 0..P {
                    let c = Fp::new(c);
                    // Distributivity is the property Π_PC's term expansion uses.
                    assert_eq!(a * (b + c), a * b + a * c);
                }
            }
        }
    }

    /// The property that `Z_{2^k}` cannot provide and that `Π_PC` depends on:
    /// **no zero divisors**, hence every non-zero element inverts. If this ever
    /// fails, the private compare's "is any term zero" test is meaningless.
    #[test]
    fn there_are_no_zero_divisors_and_every_nonzero_inverts() {
        for a in 1..P {
            let a = Fp::new(a);
            let inv = a.inv().expect("non-zero elements invert");
            assert_eq!(a * inv, Fp::ONE, "a·a⁻¹ = 1 for a = {}", a.value());
            for b in 1..P {
                assert!(!(a * Fp::new(b)).is_zero(), "no zero divisors");
            }
        }
        assert_eq!(Fp::ZERO.inv(), None);
    }

    /// Contrast, kept as a test rather than a comment: the ring the rest of the
    /// crate uses *does* have zero divisors, which is why this module exists.
    #[test]
    fn the_ring_by_contrast_has_zero_divisors() {
        let nbits = 20u32;
        let mask: Fr = (1 << nbits) - 1;
        let half: Fr = 1 << (nbits - 1);
        assert_eq!(half.wrapping_mul(2) & mask, 0, "2^(k-1)·2 = 0 in Z_2^k");
    }

    /// Sharing must round-trip.
    #[test]
    fn replicated_sharing_round_trips() {
        for x in 0..P {
            let x = Fp::new(x);
            let sh = share(x, Fp::new(x.value() * 7 + 3), Fp::new(x.value() * 31 + 11));
            assert_eq!(reconstruct(&sh), x, "reconstruct(share(x)) = x");
        }
    }

    /// Perfect secrecy against one party, stated the way it is actually true.
    ///
    /// A party holds two of the three summands, so it is *not* enough to assert
    /// that its two shares fail to sum to the secret — they sometimes do, when
    /// the third summand happens to be zero, and that is a coincidence rather
    /// than a leak. The property that matters is that the third summand is
    /// unconstrained: holding any two summands leaves the secret **uniform over
    /// the whole field**. That is what this checks, and it is the reason a
    /// 2-out-of-3 replicated sharing tolerates one corruption.
    #[test]
    fn one_party_alone_learns_nothing() {
        for p in 0..3 {
            let view = FpShare::new(Fp::new(19), Fp::new(41));
            // Range over every value the unseen third summand could take and
            // collect the secrets consistent with this fixed view.
            let mut reachable = std::collections::HashSet::new();
            for missing in 0..P {
                let m = Fp::new(missing);
                // Whichever position the party occupies, the secret is the sum
                // of all three summands.
                let secret = view.left + view.right + m;
                reachable.insert(secret.value());
            }
            assert_eq!(
                reachable.len() as u64,
                P,
                "party {p}: view must leave the secret uniform over all {P} values"
            );
        }
    }

    /// Local operations must commute with reconstruction, or the protocol layer
    /// cannot use them communication-free.
    #[test]
    fn local_ops_commute_with_reconstruction() {
        for a in [0u64, 1, 5, 33, 66] {
            for b in [0u64, 2, 9, 40, 66] {
                let (fa, fb) = (Fp::new(a), Fp::new(b));
                let sa = share(fa, Fp::new(a * 3 + 1), Fp::new(b * 5 + 2));
                let sb = share(fb, Fp::new(b * 7 + 4), Fp::new(a * 11 + 6));
                let sum = [
                    sa[0] + sb[0],
                    sa[1] + sb[1],
                    sa[2] + sb[2],
                ];
                assert_eq!(reconstruct(&sum), fa + fb);
                let diff = [sa[0] - sb[0], sa[1] - sb[1], sa[2] - sb[2]];
                assert_eq!(reconstruct(&diff), fa - fb);
                let k = Fp::new(13);
                let scaled = [
                    sa[0].scalar_mul(k),
                    sa[1].scalar_mul(k),
                    sa[2].scalar_mul(k),
                ];
                assert_eq!(reconstruct(&scaled), fa * k);
            }
        }
    }

    /// Deterministic randomness for the tests: an LCG, so failures reproduce.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> Fp {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Fp::new(self.0 >> 33)
        }
        fn zeros(&mut self, n: usize) -> Vec<[Fp; 3]> {
            (0..n)
                .map(|_| {
                    let (a, b) = (self.next(), self.next());
                    [a, b, -(a + b)]
                })
                .collect()
        }
        fn share_of(&mut self, x: Fp) -> [FpShare; 3] {
            let (r0, r1) = (self.next(), self.next());
            share(x, r0, r1)
        }
    }

    /// The multiplication protocol must reconstruct the true product. Exhaustive
    /// over the field, which at `P = 67` is 4489 pairs.
    #[test]
    fn multiplication_reconstructs_the_product() {
        let mut rng = Rng(12345);
        for a in 0..P {
            for b in 0..P {
                let (fa, fb) = (Fp::new(a), Fp::new(b));
                let sa = rng.share_of(fa);
                let sb = rng.share_of(fb);
                let z = rng.zeros(1);
                let prod = mul_shares(&sa, &sb, z[0]);
                assert_eq!(reconstruct(&prod), fa * fb, "{a}·{b}");
            }
        }
    }

    /// Run `Π_PC` once and return `β' ⊕ β`, the protocol's stated output.
    fn pc_once(x: u64, r: u64, beta: u64, l: usize, rng: &mut Rng) -> u64 {
        let bits: Vec<[FpShare; 3]> =
            (0..l).map(|i| rng.share_of(Fp::new((x >> i) & 1))).collect();
        let sbeta = rng.share_of(Fp::new(beta));
        let mut mv = rng.next();
        if mv.is_zero() {
            mv = Fp::ONE; // the mask must be a unit
        }
        let sm = rng.share_of(mv);
        let z = rng.zeros(2 * l + 2);
        (private_compare(&bits, r, &sbeta, &sm, &z) as u64) ^ beta
    }

    /// `Π_PC` computes `x ≥ r`, independently of the blinding bit — exhaustively
    /// over every `x ≠ r` at ℓ = 5, for both `β`. Equality is excluded and
    /// covered by its own test; see the function's docs for why.
    #[test]
    fn private_compare_computes_ge_for_distinct_inputs() {
        const L: usize = 5;
        let mut rng = Rng(999);
        for x in 0..(1u64 << L) {
            for r in 0..(1u64 << L) {
                if x == r {
                    continue;
                }
                for beta in [0u64, 1] {
                    let out = pc_once(x, r, beta, L, &mut rng);
                    assert_eq!(
                        out,
                        (x >= r) as u64,
                        "x={x} r={r} β={beta}: Π_PC must give (x ≥ r)"
                    );
                }
            }
        }
    }

    /// The documented edge case, pinned so it cannot regress unnoticed.
    ///
    /// At `x = r` every `c[i]` is 1 whatever `β` is, so `β'` is always 1 and the
    /// output collapses to `1 ⊕ β` — a function of the blinding bit rather than
    /// the inputs. Callers must guarantee `x ≠ r`; `Π_WA` does so by masking
    /// with a fresh random value.
    #[test]
    fn private_compare_degenerates_when_x_equals_r() {
        const L: usize = 5;
        let mut rng = Rng(4242);
        for x in 0..(1u64 << L) {
            for beta in [0u64, 1] {
                let out = pc_once(x, x, beta, L, &mut rng);
                assert_eq!(out, 1 ^ beta, "x=r={x}: output is 1⊕β, not a predicate");
            }
        }
    }

    /// The modulus bound is `p > ℓ + 2`, and it is what ties `p` to the ring
    /// width rather than making it a universal constant.
    ///
    /// Both parameter sets are checked: Falcon's own `p = 37` must serve its
    /// `ℓ = 32`, and the `p = 67` used here must serve every `ℓ ≤ 64`. Getting
    /// this wrong is silent — the terms wrap and the zero test starts answering
    /// a different question.
    #[test]
    fn the_modulus_bound_holds_for_both_parameter_sets() {
        let is_prime = |p: u64| (2..p).all(|d| p % d != 0);
        assert!(is_prime(P), "P must be prime");
        assert!(is_prime(P_FALCON), "Falcon's p must be prime");
        assert!(P_FALCON > 32 + 2, "p = 37 must serve Falcon's ℓ = 32");
        assert!(P > 64 + 2, "p = 67 must serve ℓ = 64");
        assert_eq!(max_ring_bits(P_FALCON), 34, "p = 37 covers ℓ up to 34");
        assert_eq!(max_ring_bits(P), 64, "p = 67 covers ℓ up to 64");
    }
}
