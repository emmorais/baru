//! Multi-modular NTT for **exact** negacyclic polynomial products.
//!
//! # Why multi-modular, and not a plain NTT
//!
//! A textbook NTT needs a modulus with a primitive `2n`-th root of unity —
//! i.e. a prime `q ≡ 1 (mod 2n)`. Our BFV (and SI-NTRU) deliberately use a
//! **power-of-two `q`** (`2^100`), which has no such root: a direct NTT mod
//! `q` is impossible. That power-of-two modulus is load-bearing, though —
//! it makes the BFV `t/q` rescale a rounded *shift* and the ring arithmetic
//! native wrapping `u128`.
//!
//! So instead of changing the scheme, we use NTT-friendly primes purely as
//! a **computational device**: the quantity the schemes actually need is the
//! *exact integer* negacyclic product of centered coefficients,
//!
//! ```text
//! c[k] = Σ_{i+j=k} a[i]·b[j] − Σ_{i+j=k+n} a[i]·b[j]     (exact, in ℤ)
//! ```
//!
//! which we compute modulo several NTT-friendly primes (each in O(n log n))
//! and reconstruct exactly by CRT. Everything downstream — the rescale, the
//! mod-`q` reduction, decryption — is unchanged. This replaces an O(n²)
//! loop of software 256-bit multiply-accumulates with O(n log n) machine-word
//! modular arithmetic.
//!
//! Bound: coefficients are centered in `[−q/2, q/2)`, so
//! `|c[k]| ≤ n·(q/2)² = 2^12·2^198 = 2^210` at `n = 4096, q = 2^100`. Four
//! ~61-bit primes give `Π pᵢ > 2^244 > 2·2^210`, enough to represent the
//! signed result uniquely.
//!
//! Correctness is not argued, it is *tested*: [`exact_negacyclic_product`]
//! is differentially tested against the straightforward schoolbook
//! implementation it replaces (see the tests below and `fhe::bfv`).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Largest ring dimension supported; primes are chosen `≡ 1 (mod 2·MAX_N)`
/// so the same primes serve every power-of-two `n ≤ MAX_N` (a `2n`-th root
/// is derived from the stored `2·MAX_N`-th root).
pub const MAX_N: usize = 4096;
const ROOT_ORDER: u64 = (2 * MAX_N) as u64; // 8192
/// Four ~63-bit primes ⇒ product ≈ 2^252, which must both (a) exceed
/// `2·max|c[k]|` and (b) fit in [`I256`]. That covers BFV (`q=2^100`,
/// `n=4096` ⇒ `|c| ≤ 2^210`) *and* SI-NTRU (`q=2^120`, `n=2048` ⇒
/// `|c| ≤ 2^249`, needing > 2^250) — the widest case in this crate.
const NUM_PRIMES: usize = 4;

// ─── Modular arithmetic (primes < 2^62, so products fit u128) ────────────────

#[inline]
fn addmod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

#[inline]
fn submod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { a + p - b }
}

#[inline]
fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn powmod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    a %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    r
}

/// Modular inverse via Fermat (p prime).
fn invmod(a: u64, p: u64) -> u64 {
    powmod(a, p - 2, p)
}

/// Deterministic Miller–Rabin for 64-bit inputs.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for q in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(q) {
            return n == q;
        }
    }
    let mut d = n - 1;
    let mut r = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        r += 1;
    }
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..r {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// A primitive `ROOT_ORDER`-th root of unity mod `p` (requires
/// `p ≡ 1 mod ROOT_ORDER`).
fn primitive_root_of_order(p: u64, order: u64) -> u64 {
    debug_assert!((p - 1).is_multiple_of(order));
    let exp = (p - 1) / order;
    // Try successive candidates; g^exp has order dividing `order`, and has
    // order exactly `order` iff its `order/2` power is not 1 (order is a
    // power of two here).
    for cand in 2u64..10_000 {
        let h = powmod(cand, exp, p);
        if h != 0 && powmod(h, order / 2, p) != 1 {
            return h;
        }
    }
    panic!("no primitive root of order {order} found mod {p}");
}

/// The NTT-friendly primes and their `ROOT_ORDER`-th roots (computed once).
struct Primes {
    p: Vec<u64>,
    root: Vec<u64>,
}

fn primes() -> &'static Primes {
    static PRIMES: OnceLock<Primes> = OnceLock::new();
    PRIMES.get_or_init(|| {
        let mut p = Vec::with_capacity(NUM_PRIMES);
        let mut root = Vec::with_capacity(NUM_PRIMES);
        // Search downward from just under 2^63 in steps of ROOT_ORDER so
        // every prime satisfies p ≡ 1 (mod ROOT_ORDER). ~63-bit primes keep
        // `a+b` inside u64 and `a*b` inside u128, while four of them give a
        // 252-bit CRT modulus (fits I256, covers the SI-NTRU bound).
        let mut cand = ((1u64 << 63) / ROOT_ORDER) * ROOT_ORDER + 1;
        while p.len() < NUM_PRIMES {
            if is_prime(cand) {
                root.push(primitive_root_of_order(cand, ROOT_ORDER));
                p.push(cand);
            }
            cand -= ROOT_ORDER;
        }
        Primes { p, root }
    })
}

// ─── Negacyclic NTT plan (per prime, per ring dimension) ─────────────────────

struct Plan {
    p: u64,
    /// ψ^i and ψ^-i for the negacyclic twist (ψ a primitive 2n-th root).
    psi: Vec<u64>,
    psi_inv: Vec<u64>,
    /// ω = ψ² (primitive n-th root) and its inverse, plus n^{-1}.
    w: u64,
    w_inv: u64,
    n_inv: u64,
}

impl Plan {
    fn new(p: u64, root_max: u64, n: usize) -> Plan {
        assert!(n.is_power_of_two() && n <= MAX_N);
        // ψ = root^(ROOT_ORDER / 2n) is a primitive 2n-th root of unity.
        let psi_g = powmod(root_max, ROOT_ORDER / (2 * n as u64), p);
        let psi_inv_g = invmod(psi_g, p);
        let mut psi = Vec::with_capacity(n);
        let mut psi_inv = Vec::with_capacity(n);
        let (mut a, mut b) = (1u64, 1u64);
        for _ in 0..n {
            psi.push(a);
            psi_inv.push(b);
            a = mulmod(a, psi_g, p);
            b = mulmod(b, psi_inv_g, p);
        }
        let w = mulmod(psi_g, psi_g, p);
        Plan {
            p,
            psi,
            psi_inv,
            w,
            w_inv: invmod(w, p),
            n_inv: invmod(n as u64, p),
        }
    }

    /// In-place iterative Cooley–Tukey NTT with `root` a primitive n-th
    /// root of unity.
    fn ntt(&self, a: &mut [u64], root: u64) {
        let n = a.len();
        let p = self.p;
        // bit-reversal permutation
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2usize;
        while len <= n {
            let wlen = powmod(root, (n / len) as u64, p);
            for chunk in a.chunks_mut(len) {
                let mut wn = 1u64;
                let half = len / 2;
                for k in 0..half {
                    let u = chunk[k];
                    let v = mulmod(chunk[k + half], wn, p);
                    chunk[k] = addmod(u, v, p);
                    chunk[k + half] = submod(u, v, p);
                    wn = mulmod(wn, wlen, p);
                }
            }
            len <<= 1;
        }
    }

    /// Forward negacyclic transform: twist by ψ^i, then NTT.
    fn forward(&self, a: &mut [u64]) {
        for (ai, &psi) in a.iter_mut().zip(&self.psi) {
            *ai = mulmod(*ai, psi, self.p);
        }
        self.ntt(a, self.w);
    }

    /// Inverse negacyclic transform: inverse NTT, scale by n^{-1}, untwist.
    fn inverse(&self, a: &mut [u64]) {
        self.ntt(a, self.w_inv);
        for (ai, &psi_inv) in a.iter_mut().zip(&self.psi_inv) {
            *ai = mulmod(mulmod(*ai, self.n_inv, self.p), psi_inv, self.p);
        }
    }
}

fn plans_for(n: usize) -> &'static [Plan] {
    static CACHE: OnceLock<Mutex<HashMap<usize, &'static [Plan]>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    guard.entry(n).or_insert_with(|| {
        let pr = primes();
        let v: Vec<Plan> = (0..NUM_PRIMES)
            .map(|i| Plan::new(pr.p[i], pr.root[i], n))
            .collect();
        // Leaked once per ring dimension (a handful for the whole process).
        Box::leak(v.into_boxed_slice()) as &'static [Plan]
    })
}

// ─── Exact 256-bit signed integer (CRT accumulator) ──────────────────────────

/// Two's-complement 256-bit integer, enough for `|c[k]| ≤ 2^210` and for the
/// CRT modulus `Π pᵢ ≈ 2^244`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I256 {
    pub hi: u128,
    pub lo: u128,
}

impl I256 {
    pub const ZERO: I256 = I256 { hi: 0, lo: 0 };
    pub const ONE: I256 = I256 { hi: 0, lo: 1 };

    pub fn wrapping_add(self, o: I256) -> I256 {
        let (lo, carry) = self.lo.overflowing_add(o.lo);
        I256 { hi: self.hi.wrapping_add(o.hi).wrapping_add(carry as u128), lo }
    }
    pub fn wrapping_neg(self) -> I256 {
        let (lo, borrow) = 0u128.overflowing_sub(self.lo);
        I256 { hi: 0u128.wrapping_sub(self.hi).wrapping_sub(borrow as u128), lo }
    }
    pub fn wrapping_sub(self, o: I256) -> I256 {
        self.wrapping_add(o.wrapping_neg())
    }
    fn is_negative(self) -> bool {
        (self.hi >> 127) & 1 == 1
    }
    /// Unsigned comparison (used while the CRT value is known non-negative).
    fn ult(self, o: I256) -> bool {
        (self.hi, self.lo) < (o.hi, o.lo)
    }

    fn limbs(self) -> [u64; 4] {
        [
            (self.hi >> 64) as u64,
            self.hi as u64,
            (self.lo >> 64) as u64,
            self.lo as u64,
        ]
    }
    fn from_limbs(l: [u64; 4]) -> I256 {
        I256 {
            hi: ((l[0] as u128) << 64) | l[1] as u128,
            lo: ((l[2] as u128) << 64) | l[3] as u128,
        }
    }

    /// Multiply by a 64-bit scalar (truncating at 256 bits — callers keep
    /// the value below the CRT modulus).
    pub fn mul_u64(self, k: u64) -> I256 {
        let l = self.limbs();
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in (0..4).rev() {
            let prod = l[i] as u128 * k as u128 + carry;
            out[i] = prod as u64;
            carry = prod >> 64;
        }
        I256::from_limbs(out)
    }

    /// Reduce a **non-negative** value mod a prime `< 2^62`.
    fn mod_u64(self, p: u64) -> u64 {
        let mut r = 0u128;
        for l in self.limbs() {
            // r < p < 2^62 ⇒ (r<<64) < 2^126, plus a 64-bit limb: fits u128.
            r = ((r << 64) | l as u128) % p as u128;
        }
        r as u64
    }

    /// Arithmetic shift right by `k < 128`, rounding half away from zero.
    pub fn shr_round(self, k: u32) -> I256 {
        debug_assert!(k > 0 && k < 128);
        let negative = self.is_negative();
        let mag = if negative { self.wrapping_neg() } else { self };
        let rounded = mag.wrapping_add(I256 { hi: 0, lo: 1u128 << (k - 1) });
        let lo = (rounded.lo >> k) | (rounded.hi << (128 - k));
        let hi = rounded.hi >> k;
        let shifted = I256 { hi, lo };
        if negative { shifted.wrapping_neg() } else { shifted }
    }
}

// ─── The exact product ────────────────────────────────────────────────────────

/// Precomputed CRT constants (independent of the ring dimension).
struct CrtConst {
    ps: Vec<u64>,
    /// `prefix[i] = Π_{j<i} p_j`
    prefix: Vec<I256>,
    /// `inv(prefix[i] mod p_i) mod p_i`
    inv_prefix: Vec<u64>,
    modulus: I256,
    half: I256,
}

fn crt_consts() -> &'static CrtConst {
    static C: OnceLock<CrtConst> = OnceLock::new();
    C.get_or_init(|| {
        let ps = primes().p.clone();
        let mut modulus = I256::ONE;
        let mut prefix = Vec::with_capacity(NUM_PRIMES);
        for &p in &ps {
            prefix.push(modulus);
            modulus = modulus.mul_u64(p);
        }
        let half = I256 {
            hi: modulus.hi >> 1,
            lo: (modulus.lo >> 1) | (modulus.hi << 127),
        };
        let inv_prefix: Vec<u64> = (0..NUM_PRIMES)
            .map(|i| if i == 0 { 1 } else { invmod(prefix[i].mod_u64(ps[i]), ps[i]) })
            .collect();
        CrtConst { ps, prefix, inv_prefix, modulus, half }
    })
}

/// Garner CRT reconstruction of `n` coefficients from per-prime residues,
/// centered into `[−Π pᵢ/2, Π pᵢ/2)`.
fn crt_reconstruct(residues: &[Vec<u64>], n: usize) -> Vec<I256> {
    let c = crt_consts();
    (0..n)
        .map(|k| {
            let mut x = I256 { hi: 0, lo: residues[0][k] as u128 };
            for (i, res) in residues.iter().enumerate().skip(1) {
                let p = c.ps[i];
                let xi = x.mod_u64(p);
                let t = mulmod(submod(res[k] % p, xi, p), c.inv_prefix[i], p);
                x = x.wrapping_add(c.prefix[i].mul_u64(t));
            }
            if !x.ult(c.half) {
                x = x.wrapping_sub(c.modulus);
            }
            x
        })
        .collect()
}

#[inline]
fn reduce_mod(v: &[i128], p: u64) -> Vec<u64> {
    let pi = p as i128;
    v.iter().map(|&x| (((x % pi) + pi) % pi) as u64).collect()
}

/// Exact negacyclic product of two centered-coefficient polynomials, in ℤ:
/// `c[k] = Σ_{i+j=k} a[i]·b[j] − Σ_{i+j=k+n} a[i]·b[j]`.
///
/// Computed modulo [`NUM_PRIMES`] NTT-friendly primes in O(n log n) each and
/// reconstructed exactly by CRT (Garner), then centered. Requires
/// `|c[k]| < Π pᵢ / 2`.
pub fn exact_negacyclic_product(a: &[i128], b: &[i128], n: usize) -> Vec<I256> {
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);
    let mut residues: Vec<Vec<u64>> = Vec::with_capacity(NUM_PRIMES);
    for plan in plans_for(n) {
        let p = plan.p;
        let mut fa = reduce_mod(a, p);
        let mut fb = reduce_mod(b, p);
        plan.forward(&mut fa);
        plan.forward(&mut fb);
        for i in 0..n {
            fa[i] = mulmod(fa[i], fb[i], p);
        }
        plan.inverse(&mut fa);
        residues.push(fa);
    }
    crt_reconstruct(&residues, n)
}

/// A polynomial already in the NTT domain (per prime). Precomputing this
/// for an operand reused across several multiplications — e.g. the *input*
/// ciphertext of a chunked matvec, multiplied against many weight chunks —
/// saves its forward transform on every chunk after the first.
pub struct Transformed {
    n: usize,
    res: Vec<Vec<u64>>,
}

/// Forward-transform centered coefficients once, for reuse.
pub fn forward_transform(a: &[i128], n: usize) -> Transformed {
    let res = plans_for(n)
        .iter()
        .map(|plan| {
            let mut f = reduce_mod(a, plan.p);
            plan.forward(&mut f);
            f
        })
        .collect();
    Transformed { n, res }
}

/// The BFV multiply's three tensor components at once:
/// `(a₀b₀, a₀b₁ + a₁b₀, a₁b₁)`, exact in ℤ.
///
/// Sharing the forward transforms across the products costs **4 forward +
/// 3 inverse** NTTs per prime instead of the 12 that four independent
/// products would need — the main reason to have a BFV-specific entry
/// point rather than calling [`exact_negacyclic_product`] four times.
#[allow(clippy::type_complexity)]
pub fn exact_bfv_tensor(
    a0: &[i128],
    a1: &[i128],
    b0: &[i128],
    b1: &[i128],
    n: usize,
) -> (Vec<I256>, Vec<I256>, Vec<I256>) {
    let ta0 = forward_transform(a0, n);
    let ta1 = forward_transform(a1, n);
    exact_bfv_tensor_pre(&ta0, &ta1, b0, b1, n)
}

/// As [`exact_bfv_tensor`], but with the first operand's transform already
/// computed — **2 forward + 3 inverse** NTTs per prime per call instead of
/// 4 + 3, which is the saving that makes a multi-chunk matvec cheaper.
#[allow(clippy::type_complexity)]
pub fn exact_bfv_tensor_pre(
    ta0: &Transformed,
    ta1: &Transformed,
    b0: &[i128],
    b1: &[i128],
    n: usize,
) -> (Vec<I256>, Vec<I256>, Vec<I256>) {
    assert_eq!(ta0.n, n);
    assert_eq!(ta1.n, n);
    let (mut r0, mut r1, mut r2) = (Vec::new(), Vec::new(), Vec::new());
    for (k, plan) in plans_for(n).iter().enumerate() {
        let p = plan.p;
        let fa0 = &ta0.res[k];
        let fa1 = &ta1.res[k];
        let mut fb0 = reduce_mod(b0, p);
        let mut fb1 = reduce_mod(b1, p);
        plan.forward(&mut fb0);
        plan.forward(&mut fb1);
        let mut d0: Vec<u64> = (0..n).map(|i| mulmod(fa0[i], fb0[i], p)).collect();
        let mut d1: Vec<u64> = (0..n)
            .map(|i| addmod(mulmod(fa0[i], fb1[i], p), mulmod(fa1[i], fb0[i], p), p))
            .collect();
        let mut d2: Vec<u64> = (0..n).map(|i| mulmod(fa1[i], fb1[i], p)).collect();
        plan.inverse(&mut d0);
        plan.inverse(&mut d1);
        plan.inverse(&mut d2);
        r0.push(d0);
        r1.push(d1);
        r2.push(d2);
    }
    (
        crt_reconstruct(&r0, n),
        crt_reconstruct(&r1, n),
        crt_reconstruct(&r2, n),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference this replaces: straightforward O(n²) exact negacyclic
    /// product using 128-bit products widened to 256 bits.
    fn schoolbook(a: &[i128], b: &[i128], n: usize) -> Vec<I256> {
        fn mul_i128(a: i128, b: i128) -> I256 {
            let neg = (a < 0) != (b < 0);
            let (ua, ub) = (a.unsigned_abs(), b.unsigned_abs());
            let (a1, a0) = (ua >> 64, ua & 0xFFFF_FFFF_FFFF_FFFF);
            let (b1, b0) = (ub >> 64, ub & 0xFFFF_FFFF_FFFF_FFFF);
            let p00 = a0 * b0;
            let p01 = a0 * b1;
            let p10 = a1 * b0;
            let p11 = a1 * b1;
            let mid = (p00 >> 64) + (p01 & 0xFFFF_FFFF_FFFF_FFFF) + (p10 & 0xFFFF_FFFF_FFFF_FFFF);
            let lo = (p00 & 0xFFFF_FFFF_FFFF_FFFF) | (mid << 64);
            let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
            let m = I256 { hi, lo };
            if neg { m.wrapping_neg() } else { m }
        }
        let mut acc = vec![I256::ZERO; n];
        for (i, &ai) in a.iter().enumerate() {
            if ai == 0 {
                continue;
            }
            for (j, &bj) in b.iter().enumerate() {
                let prod = mul_i128(ai, bj);
                let idx = i + j;
                if idx < n {
                    acc[idx] = acc[idx].wrapping_add(prod);
                } else {
                    acc[idx - n] = acc[idx - n].wrapping_add(prod.wrapping_neg());
                }
            }
        }
        acc
    }

    fn pseudo(n: usize, seed: u64, bits: u32) -> Vec<i128> {
        // Deterministic pseudo-random centered coefficients of `bits` bits.
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let mag = ((s >> 3) as u128) & ((1u128 << bits) - 1);
                if s & 1 == 0 { mag as i128 } else { -(mag as i128) }
            })
            .collect()
    }

    #[test]
    fn primes_are_ntt_friendly() {
        let pr = primes();
        assert_eq!(pr.p.len(), NUM_PRIMES);
        for (&p, &r) in pr.p.iter().zip(&pr.root) {
            assert!(is_prime(p), "{p} must be prime");
            assert!((p - 1).is_multiple_of(ROOT_ORDER), "{p} must be ≡ 1 mod {ROOT_ORDER}");
            // r must have order exactly ROOT_ORDER.
            assert_eq!(powmod(r, ROOT_ORDER, p), 1);
            assert_ne!(powmod(r, ROOT_ORDER / 2, p), 1);
        }
    }

    #[test]
    fn ntt_matches_schoolbook_small() {
        for &n in &[8usize, 16, 64] {
            let a = pseudo(n, 1, 20);
            let b = pseudo(n, 2, 20);
            assert_eq!(exact_negacyclic_product(&a, &b, n), schoolbook(&a, &b, n), "n={n}");
        }
    }

    #[test]
    fn ntt_matches_schoolbook_at_full_coefficient_width() {
        // Worst case: coefficients near ±2^99 (centered mod q = 2^100), the
        // regime BFV actually uses.
        let n = 256;
        let a = pseudo(n, 7, 99);
        let b = pseudo(n, 9, 99);
        assert_eq!(exact_negacyclic_product(&a, &b, n), schoolbook(&a, &b, n));
    }

    #[test]
    fn ntt_matches_schoolbook_at_ring_dimension_4096() {
        let n = MAX_N;
        let a = pseudo(n, 11, 99);
        let b = pseudo(n, 13, 99);
        assert_eq!(exact_negacyclic_product(&a, &b, n), schoolbook(&a, &b, n));
    }

    #[test]
    fn bfv_tensor_matches_independent_products() {
        // The shared-forward-transform path must equal three independent
        // exact products (a0b0, a0b1+a1b0, a1b1).
        let n = 128;
        let (a0, a1) = (pseudo(n, 21, 99), pseudo(n, 22, 99));
        let (b0, b1) = (pseudo(n, 23, 99), pseudo(n, 24, 99));
        let (d0, d1, d2) = exact_bfv_tensor(&a0, &a1, &b0, &b1, n);
        assert_eq!(d0, schoolbook(&a0, &b0, n));
        assert_eq!(d2, schoolbook(&a1, &b1, n));
        let cross: Vec<I256> = schoolbook(&a0, &b1, n)
            .into_iter()
            .zip(schoolbook(&a1, &b0, n))
            .map(|(x, y)| x.wrapping_add(y))
            .collect();
        assert_eq!(d1, cross);
    }

    #[test]
    fn handles_zero_and_sign_edge_cases() {
        let n = 16;
        let zero = vec![0i128; n];
        let a = pseudo(n, 3, 60);
        assert_eq!(exact_negacyclic_product(&zero, &a, n), schoolbook(&zero, &a, n));
        let mut neg = vec![0i128; n];
        neg[0] = -(1i128 << 99);
        neg[1] = (1i128 << 99) - 1;
        assert_eq!(exact_negacyclic_product(&neg, &a, n), schoolbook(&neg, &a, n));
    }
}
