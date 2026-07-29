//! **CRT slot packing** — the encoding that lets one ciphertext carry `n`
//! *independent* values instead of one convolution.
//!
//! # Why this exists
//!
//! Everything else in this crate packs plaintexts into **coefficients**:
//! `m(X) = Σ xᵢ Xⁱ`. That is the right encoding for an inner product, because
//! a polynomial product *is* a convolution and the reversed-weight trick lands
//! each row's dot product on a known coefficient
//! ([`crate::fhe::bfv::pack_weights`]).
//!
//! It is the wrong encoding for anything elementwise. `m(X)²` has coefficient
//! `Σ_{i+j=k} xᵢxⱼ` — every output is polluted by cross terms, so `xᵢ²` is not
//! recoverable. Two things this repo wants need elementwise products:
//!
//! - **`mpcc::lowgear`**: an oblivious linear evaluation must produce `n`
//!   independent products `aᵢ·bᵢ` from one ciphertext, or SHE-based
//!   preprocessing costs *more* per triple than the oblivious transfer it
//!   replaces (~131 KB vs 3.1 KB — see that module).
//! - Polynomial approximation of a nonlinearity (sign/ReLU), which needs `x²`
//!   slotwise. Measured elsewhere to be a bad trade for other reasons, but it
//!   is blocked on this first.
//!
//! # The construction
//!
//! When `t` is a prime with `t ≡ 1 (mod 2n)`, the ring `Z_t[X]/(Xⁿ+1)` splits
//! completely: `Xⁿ+1 = Π_i (X − ψ^{2i+1})` where `ψ` is a primitive `2n`-th
//! root of unity mod `t`. The Chinese Remainder map
//!
//! ```text
//! m(X)  ↦  ( m(ψ¹), m(ψ³), …, m(ψ^{2n−1}) )
//! ```
//!
//! is a ring isomorphism onto `(Z_t)ⁿ`, so **polynomial multiplication becomes
//! componentwise multiplication**. Those `n` components are the "slots".
//!
//! That map is exactly the negacyclic NTT: twist by `ψ^j`, then a standard
//! NTT. [`decode`] is the forward direction (coefficients → slots) and
//! [`encode`] the inverse. Both are O(n log n) and run on *plaintexts* only,
//! so they cost nothing against a ciphertext operation.
//!
//! # Relationship to `fhe::ntt`
//!
//! [`crate::fhe::ntt`] also computes negacyclic NTTs, but for a different
//! purpose and modulus: it works over four fixed ~63-bit primes chosen as a
//! *computational device* to reconstruct an exact integer product, and its
//! plans are built for those primes only. Here the modulus is the **plaintext**
//! modulus `t`, chosen by the scheme's parameters, and the transform is part of
//! the encoding rather than an implementation detail. Sharing code would mean
//! parameterising `ntt.rs`'s cached plans over a runtime modulus, which buys
//! nothing: this transform runs on small `t` (< 2^32 in practice), is not on
//! the hot path, and is ~60 lines.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ─── Modular arithmetic (t < 2^62, so products fit u128) ─────────────────────

#[inline]
fn addmod(a: u64, b: u64, t: u64) -> u64 {
    let s = a + b;
    if s >= t { s - t } else { s }
}

#[inline]
fn submod(a: u64, b: u64, t: u64) -> u64 {
    if a >= b { a - b } else { a + t - b }
}

#[inline]
fn mulmod(a: u64, b: u64, t: u64) -> u64 {
    ((a as u128 * b as u128) % t as u128) as u64
}

fn powmod(mut a: u64, mut e: u64, t: u64) -> u64 {
    let mut r = 1u64;
    a %= t;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, t);
        }
        a = mulmod(a, a, t);
        e >>= 1;
    }
    r
}

fn invmod(a: u64, t: u64) -> u64 {
    powmod(a, t - 2, t)
}

/// Deterministic Miller–Rabin for 64-bit inputs.
pub fn is_prime(n: u64) -> bool {
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

/// The largest prime `t ≤ max_t` with `t ≡ 1 (mod 2n)` — i.e. the largest
/// plaintext modulus at most `max_t` that admits `n` slots at ring dimension
/// `n`. Returns `None` if there is none.
///
/// Callers pick `max_t` from a *noise* budget (see `mpcc::lowgear`), so this
/// answers "given how much room I have, what is the biggest usable field?".
pub fn largest_slot_prime(n: usize, max_t: u64) -> Option<u64> {
    let step = 2 * n as u64;
    let mut cand = (max_t / step) * step + 1;
    if cand > max_t {
        cand -= step;
    }
    while cand > 1 {
        if is_prime(cand) {
            return Some(cand);
        }
        cand = cand.checked_sub(step)?;
    }
    None
}

// ─── The slot transform ──────────────────────────────────────────────────────

/// Precomputed roots for one `(n, t)` pair.
struct Plan {
    t: u64,
    /// ψ^j and ψ^-j, ψ a primitive 2n-th root of unity mod t.
    psi: Vec<u64>,
    psi_inv: Vec<u64>,
    /// ω = ψ² (primitive n-th root) and its inverse, plus n^{-1} mod t.
    w: u64,
    w_inv: u64,
    n_inv: u64,
}

impl Plan {
    fn new(n: usize, t: u64) -> Result<Plan, String> {
        if !n.is_power_of_two() {
            return Err(format!("ring dimension {n} must be a power of two"));
        }
        if !is_prime(t) {
            return Err(format!("plaintext modulus {t} must be prime for slot packing"));
        }
        let order = 2 * n as u64;
        if !(t - 1).is_multiple_of(order) {
            return Err(format!(
                "slot packing needs t ≡ 1 (mod 2n): {t} mod {order} = {}",
                (t - 1) % order + 1
            ));
        }
        // g^((t−1)/2n) has order dividing 2n; it has order exactly 2n iff its
        // n-th power is not 1 (2n is a power of two here).
        let exp = (t - 1) / order;
        let mut psi_g = 0u64;
        for cand in 2u64..100_000 {
            let h = powmod(cand % t, exp, t);
            if h != 0 && powmod(h, order / 2, t) != 1 {
                psi_g = h;
                break;
            }
        }
        if psi_g == 0 {
            return Err(format!("no primitive {order}-th root of unity mod {t}"));
        }
        let psi_inv_g = invmod(psi_g, t);
        let mut psi = Vec::with_capacity(n);
        let mut psi_inv = Vec::with_capacity(n);
        let (mut a, mut b) = (1u64, 1u64);
        for _ in 0..n {
            psi.push(a);
            psi_inv.push(b);
            a = mulmod(a, psi_g, t);
            b = mulmod(b, psi_inv_g, t);
        }
        let w = mulmod(psi_g, psi_g, t);
        Ok(Plan {
            t,
            psi,
            psi_inv,
            w,
            w_inv: invmod(w, t),
            n_inv: invmod(n as u64, t),
        })
    }

    /// In-place iterative Cooley–Tukey NTT with `root` a primitive n-th root.
    fn ntt(&self, a: &mut [u64], root: u64) {
        let n = a.len();
        let t = self.t;
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
            let wlen = powmod(root, (n / len) as u64, t);
            for chunk in a.chunks_mut(len) {
                let mut wn = 1u64;
                let half = len / 2;
                for k in 0..half {
                    let u = chunk[k];
                    let v = mulmod(chunk[k + half], wn, t);
                    chunk[k] = addmod(u, v, t);
                    chunk[k + half] = submod(u, v, t);
                    wn = mulmod(wn, wlen, t);
                }
            }
            len <<= 1;
        }
    }

    /// coefficients → slots
    fn forward(&self, a: &mut [u64]) {
        for (ai, &p) in a.iter_mut().zip(&self.psi) {
            *ai = mulmod(*ai, p, self.t);
        }
        self.ntt(a, self.w);
    }

    /// slots → coefficients
    fn inverse(&self, a: &mut [u64]) {
        self.ntt(a, self.w_inv);
        for (ai, &p) in a.iter_mut().zip(&self.psi_inv) {
            *ai = mulmod(mulmod(*ai, self.n_inv, self.t), p, self.t);
        }
    }
}

fn plan_for(n: usize, t: u64) -> Result<&'static Plan, String> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, u64), &'static Plan>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = guard.get(&(n, t)) {
        return Ok(p);
    }
    // Leaked once per (n, t) — a handful for the whole process.
    let p: &'static Plan = Box::leak(Box::new(Plan::new(n, t)?));
    guard.insert((n, t), p);
    Ok(p)
}

/// Is `(n, t)` a usable slot-packing parameter pair?
pub fn supports_slots(n: usize, t: u64) -> bool {
    n.is_power_of_two() && t > 2 && is_prime(t) && (t - 1).is_multiple_of(2 * n as u64)
}

/// **Slots → coefficients.** The result is the plaintext polynomial to hand to
/// `encrypt_poly`; multiplying two such plaintexts in the ring multiplies the
/// slot vectors componentwise.
///
/// `slots.len()` must equal `n`; values must already be reduced mod `t`.
pub fn encode(slots: &[u64], n: usize, t: u64) -> Result<Vec<u64>, String> {
    if slots.len() != n {
        return Err(format!("expected {n} slots, got {}", slots.len()));
    }
    let plan = plan_for(n, t)?;
    let mut v: Vec<u64> = slots.iter().map(|&x| x % t).collect();
    plan.inverse(&mut v);
    Ok(v)
}

/// **Coefficients → slots**, the inverse of [`encode`].
pub fn decode(coeffs: &[u64], n: usize, t: u64) -> Result<Vec<u64>, String> {
    if coeffs.len() != n {
        return Err(format!("expected {n} coefficients, got {}", coeffs.len()));
    }
    let plan = plan_for(n, t)?;
    let mut v: Vec<u64> = coeffs.iter().map(|&x| x % t).collect();
    plan.forward(&mut v);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n=64 needs t ≡ 1 mod 128; 3329 = 26·128 + 1 (Kyber's prime).
    const N: usize = 64;
    const T: u64 = 3329;

    fn sample(n: usize, t: u64, seed: u64) -> Vec<u64> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) % t
            })
            .collect()
    }

    /// Textbook negacyclic product, the definition slot multiplication must
    /// agree with.
    fn negacyclic_mul(a: &[u64], b: &[u64], n: usize, t: u64) -> Vec<u64> {
        let mut out = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                let p = mulmod(a[i], b[j], t);
                let k = i + j;
                if k < n {
                    out[k] = addmod(out[k], p, t);
                } else {
                    out[k - n] = submod(out[k - n], p, t);
                }
            }
        }
        out
    }

    #[test]
    fn parameters_are_validated() {
        assert!(supports_slots(N, T));
        // t ≢ 1 mod 2n
        assert!(!supports_slots(N, 3331));
        // composite
        assert!(!supports_slots(N, 3329 * 2));
        assert!(encode(&vec![0; N], N, 3331).is_err());
    }

    #[test]
    fn encode_decode_round_trips() {
        let slots = sample(N, T, 7);
        let coeffs = encode(&slots, N, T).unwrap();
        assert_eq!(decode(&coeffs, N, T).unwrap(), slots);
    }

    #[test]
    fn wrong_lengths_are_rejected() {
        assert!(encode(&[1, 2, 3], N, T).is_err());
        assert!(decode(&[1, 2, 3], N, T).is_err());
    }

    /// **The property the whole module exists for**: a ring product of two
    /// encoded plaintexts decodes to the componentwise product of the slots.
    #[test]
    fn ring_multiplication_is_componentwise_on_slots() {
        let (x, y) = (sample(N, T, 1), sample(N, T, 2));
        let prod = negacyclic_mul(&encode(&x, N, T).unwrap(), &encode(&y, N, T).unwrap(), N, T);
        let got = decode(&prod, N, T).unwrap();
        let want: Vec<u64> = x.iter().zip(&y).map(|(a, b)| mulmod(*a, *b, T)).collect();
        assert_eq!(got, want);
    }

    /// Addition is componentwise in *both* encodings, so it must commute with
    /// the transform — this is what lets a masked ciphertext be added.
    #[test]
    fn ring_addition_is_componentwise_on_slots() {
        let (x, y) = (sample(N, T, 3), sample(N, T, 4));
        let (ex, ey) = (encode(&x, N, T).unwrap(), encode(&y, N, T).unwrap());
        let sum: Vec<u64> = ex.iter().zip(&ey).map(|(a, b)| addmod(*a, *b, T)).collect();
        let want: Vec<u64> = x.iter().zip(&y).map(|(a, b)| addmod(*a, *b, T)).collect();
        assert_eq!(decode(&sum, N, T).unwrap(), want);
    }

    /// The negative result that motivates the module: *coefficient* packing
    /// does not give componentwise products, so slots are not an optimisation.
    #[test]
    fn coefficient_packing_does_not_give_componentwise_products() {
        let (x, y) = (sample(N, T, 5), sample(N, T, 6));
        let prod = negacyclic_mul(&x, &y, N, T);
        let componentwise: Vec<u64> = x.iter().zip(&y).map(|(a, b)| mulmod(*a, *b, T)).collect();
        assert_ne!(prod, componentwise, "cross terms must pollute coefficient packing");
    }

    #[test]
    fn largest_slot_prime_respects_the_bound_and_the_congruence() {
        for n in [1024usize, 2048, 4096] {
            let t = largest_slot_prime(n, 1 << 17).expect("a prime exists under 2^17");
            assert!(t <= 1 << 17);
            assert!(is_prime(t));
            assert_eq!((t - 1) % (2 * n as u64), 0);
            assert!(supports_slots(n, t));
        }
    }

    /// The deployed pair (see `mpcc::lowgear`): n=4096 needs t ≡ 1 mod 8192.
    #[test]
    fn the_lowgear_parameters_support_slots() {
        let t = largest_slot_prime(4096, 1 << 17).unwrap();
        assert!(supports_slots(4096, t));
        let slots = sample(4096, t, 11);
        assert_eq!(decode(&encode(&slots, 4096, t).unwrap(), 4096, t).unwrap(), slots);
    }
}
