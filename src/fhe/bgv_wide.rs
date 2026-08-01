//! **BGV at wide parameters** — `n = 8192`, `q = 2^190`, prime `t ≈ 2^61`.
//!
//! # Why this is separate from `fhe::bgv`
//!
//! [`crate::fhe::bgv`] holds coefficients in `u128`, which caps `q` at `2^127`.
//! That is ample for the depth-1 inner product every other scheme here does,
//! and it keeps the hot path on native-width arithmetic.
//!
//! [`crate::mpcc::lowgear`] cannot live inside that cap. Its plaintext modulus
//! is pinned by the OLE's noise-flooding budget,
//!
//! ```text
//! t² · n² · 3 · 2^(σ−1) < q/2
//! ```
//!
//! so at `q = 2^100` the field is ~17 bits — too narrow to bit-decompose, hence
//! too narrow to run a comparison, hence too narrow to run a private layer. A
//! field wide enough for `bit_width + KAPPA` (≈ 58 bits) needs `q ≈ 2^190`, and
//! `q = 2^190` needs 256-bit coefficients and a 512-bit product accumulator
//! ([`crate::fhe::ntt::exact_negacyclic_product_wide`]).
//!
//! Making `bgv.rs` generic over coefficient width was the alternative. This
//! repo has already answered that question the other way: `bfv.rs` and
//! `bgv.rs` each carry their own `poly_add` / `center` / `sample_ternary`,
//! because self-contained schemes read better than a shared abstraction over
//! slightly different rings. A third instance at a third parameter regime
//! follows the house pattern, and — the deciding
//! argument — it cannot slow the narrow path, which was just tuned 4.65×.
//!
//! # What it implements, and what it does not
//!
//! LowGear's OLE is **ciphertext × plaintext**. So this module needs encryption,
//! a plaintext multiply, a plaintext add, noise flooding and decryption — and
//! it does **not** need the ct×ct tensor, relinearization, or any notion of key
//! power above one. That is why it is ~250 lines rather than a second `bgv.rs`.
//!
//! Everything else is the same scheme: message in the low bits
//! (`c(s) = m + t·e`), decryption is `[c(s)]_q mod t`, and `t` is free to be the
//! prime that CRT slot packing requires because there is no `Δ = q/t` to divide.

use rand::Rng;

use crate::fhe::bigint::{I256, I512};
use crate::fhe::ntt;

// ─── Parameters ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideParams {
    pub n: usize,
    /// log2 of the ciphertext modulus. Must be < 256 and leave room for the
    /// product accumulator: `n·(q/2)²` has to stay under the wide CRT modulus
    /// (~2^440).
    pub log2_q: u32,
    /// Prime plaintext modulus with `t ≡ 1 (mod 2n)`, so [`crate::fhe::slots`]
    /// can pack `n` independent values.
    pub t: u64,
}

impl WideParams {
    /// The deployed set. `n = 8192` and `log q = 190` sit inside the ~218-bit
    /// ceiling for 128-bit RLWE security at that dimension, and
    /// `|c| ≤ n·(q/2)² = 2^392` is inside the 7-prime CRT modulus.
    pub const DEFAULT: WideParams = WideParams {
        n: 8192,
        log2_q: 190,
        t: 0,
    };

    /// Small ring for fast tests. **Not secure.**
    pub const TEST_SMALL: WideParams = WideParams {
        n: 64,
        log2_q: 190,
        t: 3329,
    };

    pub fn q_mask(&self) -> I256 {
        I256::low_mask(self.log2_q)
    }

    /// `q/2`, the centering threshold.
    fn half_q(&self) -> I256 {
        I256::one().shl(self.log2_q - 1)
    }

    /// Serialized bytes of one ciphertext: 2 ring elements at `⌈log₂q/8⌉` bytes
    /// per coefficient — the wire figure, not the 32 bytes an `I256` occupies
    /// in memory.
    pub fn ct_wire_bytes(&self) -> usize {
        2 * self.n * self.log2_q.div_ceil(8) as usize
    }

    /// Reject parameters the arithmetic cannot actually carry.
    pub fn validate(&self) -> Result<(), String> {
        if self.log2_q >= 256 {
            return Err(format!(
                "log2_q = {} does not fit a 256-bit coefficient",
                self.log2_q
            ));
        }
        if !self.n.is_power_of_two() || self.n > ntt::MAX_N {
            return Err(format!(
                "n = {} must be a power of two ≤ {}",
                self.n,
                ntt::MAX_N
            ));
        }
        if !crate::fhe::slots::supports_slots(self.n, self.t) {
            return Err(format!(
                "t = {} must be prime and ≡ 1 (mod {}) to carry {} slots",
                self.t,
                2 * self.n,
                self.n
            ));
        }
        // |c[k]| ≤ n·(q/2)²  must stay inside the wide CRT modulus.
        let log2_bound = (self.n.trailing_zeros()) + 2 * (self.log2_q - 1);
        if log2_bound >= 440 {
            return Err(format!(
                "product bound 2^{log2_bound} exceeds the wide CRT modulus (~2^440)"
            ));
        }
        Ok(())
    }
}

// ─── Coefficient arithmetic (mod q, q a power of two) ────────────────────────

type Poly = Vec<I256>;

fn add_mod_q(a: I256, b: I256, mask: I256) -> I256 {
    a.wrapping_add(b).bitand(mask)
}

/// Centered representative in `[−q/2, q/2)`, the form the exact product wants.
fn center(c: I256, p: &WideParams) -> I256 {
    if c.ult(p.half_q()) {
        c
    } else {
        c.wrapping_sub(I256::one().shl(p.log2_q))
    }
}

fn poly_add(a: &[I256], b: &[I256], mask: I256) -> Poly {
    a.iter()
        .zip(b)
        .map(|(x, y)| add_mod_q(*x, *y, mask))
        .collect()
}

/// Negacyclic product mod `q`, via the wide exact product.
///
/// Operands are centered first (`a ≡ a_c mod q`, so the products agree mod `q`)
/// and the 512-bit exact result is truncated to the low `log2_q` bits — exact,
/// because `q` is a power of two.
fn poly_mul_mod(a: &[I256], b: &[I256], p: &WideParams) -> Poly {
    let mask = p.q_mask();
    let ac: Vec<I256> = a.iter().map(|&c| center(c, p)).collect();
    let bc: Vec<I256> = b.iter().map(|&c| center(c, p)).collect();
    ntt::exact_negacyclic_product_wide(&ac, &bc, p.n)
        .into_iter()
        .map(|c| reduce_acc(c, mask))
        .collect()
}

/// Reduce a 512-bit exact accumulator mod `q`.
fn reduce_acc(c: I512, mask: I256) -> I256 {
    c.resize::<4>().bitand(mask)
}

/// `x mod t` for a **signed** coefficient, `t` prime.
fn rem_t(x: I256, t: u64) -> u64 {
    if x.is_negative() {
        let m = x.wrapping_neg().mod_u64(t);
        if m == 0 {
            0
        } else {
            t - m
        }
    } else {
        x.mod_u64(t)
    }
}

fn sample_ternary(n: usize, mask: I256) -> Poly {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| match rng.gen_range(0u8..3) {
            0 => I256::ZERO,
            1 => I256::one(),
            // −1 mod q
            _ => mask,
        })
        .collect()
}

fn sample_uniform_q(n: usize, mask: I256) -> Poly {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let limbs = [
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ];
            I256 { limbs }.bitand(mask)
        })
        .collect()
}

/// Multiply each coefficient by the plaintext modulus, mod `q` — BGV scales its
/// errors by `t` so the message keeps the low bits to itself.
fn scale_by_t(e: &[I256], p: &WideParams) -> Poly {
    let mask = p.q_mask();
    e.iter().map(|&x| x.mul_u64(p.t).bitand(mask)).collect()
}

// ─── Keys and ciphertexts ────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct KeyBundle {
    s: Vec<I256>,
    b: Vec<I256>,
    a: Vec<I256>,
}

/// A wide BGV ciphertext: always two ring elements. There is no ct×ct multiply
/// here, so the degree never rises.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WideCt {
    pub c0: Vec<I256>,
    pub c1: Vec<I256>,
}

pub struct BgvWide {
    pub params: WideParams,
}

impl BgvWide {
    pub fn new(params: WideParams) -> Result<Self, String> {
        params.validate()?;
        Ok(BgvWide { params })
    }

    /// KeyGen: ternary `s`, public key `(b, a) = (−(a·s + t·e), a)`.
    pub fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        let p = &self.params;
        let mask = p.q_mask();
        let s = sample_ternary(p.n, mask);
        let a = sample_uniform_q(p.n, mask);
        let te = scale_by_t(&sample_ternary(p.n, mask), p);
        let a_s = poly_mul_mod(&a, &s, p);
        let b: Poly = (0..p.n)
            .map(|i| {
                I256::ZERO
                    .wrapping_sub(a_s[i])
                    .wrapping_sub(te[i])
                    .bitand(mask)
            })
            .collect();
        let client = bincode::serialize(&KeyBundle {
            s,
            b: b.clone(),
            a: a.clone(),
        })
        .unwrap();
        let server = bincode::serialize(&KeyBundle {
            s: Vec::new(),
            b,
            a,
        })
        .unwrap();
        (client, server)
    }

    fn bundle(&self, key_bytes: &[u8]) -> KeyBundle {
        bincode::deserialize(key_bytes).expect("wide BGV key bundle")
    }

    /// `c = (u·b + t·e₁ + m, u·a + t·e₂)`, so `c₀ + c₁·s = m + t·(…)`.
    pub fn encrypt_poly(&self, key_bytes: &[u8], m: &[u64]) -> WideCt {
        let p = &self.params;
        let mask = p.q_mask();
        let pk = self.bundle(key_bytes);
        let u = sample_ternary(p.n, mask);
        let te1 = scale_by_t(&sample_ternary(p.n, mask), p);
        let te2 = scale_by_t(&sample_ternary(p.n, mask), p);
        let ub = poly_mul_mod(&u, &pk.b, p);
        let ua = poly_mul_mod(&u, &pk.a, p);
        let c0 = (0..p.n)
            .map(|i| {
                let mi = I256::from_hi_lo(0, (*m.get(i).unwrap_or(&0) % p.t) as u128);
                ub[i].wrapping_add(te1[i]).wrapping_add(mi).bitand(mask)
            })
            .collect();
        let c1 = (0..p.n)
            .map(|i| ua[i].wrapping_add(te2[i]).bitand(mask))
            .collect();
        WideCt { c0, c1 }
    }

    /// `[c₀ + c₁·s]_q mod t`, on centered representatives.
    pub fn decrypt_poly(&self, ck_bytes: &[u8], ct: &WideCt) -> Vec<u64> {
        let p = &self.params;
        let s = self.bundle(ck_bytes).s;
        assert!(!s.is_empty(), "wide BGV decrypt needs the client key");
        let mask = p.q_mask();
        let c1s = poly_mul_mod(&ct.c1, &s, p);
        let acc = poly_add(&ct.c0, &c1s, mask);
        acc.iter().map(|&c| rem_t(center(c, p), p.t)).collect()
    }

    /// Add a plaintext polynomial into `c₀` — O(n), no ring multiply.
    pub fn add_plain_poly(&self, ct: &WideCt, m: &[u64]) -> WideCt {
        let p = &self.params;
        let mask = p.q_mask();
        let c0 = ct
            .c0
            .iter()
            .enumerate()
            .map(|(i, &slot)| {
                let mi = I256::from_hi_lo(0, (*m.get(i).unwrap_or(&0) % p.t) as u128);
                slot.wrapping_add(mi).bitand(mask)
            })
            .collect();
        WideCt {
            c0,
            c1: ct.c1.clone(),
        }
    }

    /// Multiply by a plaintext polynomial. **Depth 0** — the ciphertext degree
    /// is untouched, which is why LowGear's OLE needs no relinearization and no
    /// ct×ct multiply at all.
    ///
    /// The plaintext is centered into `(−t/2, t/2]` first, halving the
    /// coefficient magnitude the noise growth is proportional to.
    pub fn mul_plain_poly(&self, ct: &WideCt, m: &[u64]) -> WideCt {
        let p = &self.params;
        let half_t = p.t / 2;
        let mask = p.q_mask();
        let mp: Poly = (0..p.n)
            .map(|i| {
                let v = *m.get(i).unwrap_or(&0) % p.t;
                if v > half_t {
                    // v − t, as a centered negative value mod q
                    I256::ZERO
                        .wrapping_sub(I256::from_hi_lo(0, (p.t - v) as u128))
                        .bitand(mask)
                } else {
                    I256::from_hi_lo(0, v as u128)
                }
            })
            .collect();
        WideCt {
            c0: poly_mul_mod(&ct.c0, &mp, p),
            c1: poly_mul_mod(&ct.c1, &mp, p),
        }
    }

    /// Add flooding noise `t·e` with `‖e‖ ≤ 2^log2_bound` to `c₀`.
    ///
    /// The plaintext is untouched (the term is a multiple of `t`), but the
    /// ciphertext's noise is swamped — which is what stops the party holding the
    /// secret key from reading the responder's input off the returned noise.
    /// See [`crate::mpcc::lowgear`] for the budget that sizes `log2_bound`.
    pub fn add_flooding_noise(&self, ct: &WideCt, log2_bound: u32) -> WideCt {
        let p = &self.params;
        let mask = p.q_mask();
        let mut rng = rand::thread_rng();
        let c0 = ct
            .c0
            .iter()
            .map(|&slot| {
                // Uniform in (−2^bound, 2^bound), built limb-wise then masked.
                // The mask must be `log2_bound` bits, NOT `log2_bound + 1`: an
                // earlier version used the latter, doubling the noise and
                // pushing `t·|e|` past `q/2`, which silently corrupted every
                // triple at the deployed parameters.
                let mut e = I256 {
                    limbs: [
                        rng.gen::<u64>(),
                        rng.gen::<u64>(),
                        rng.gen::<u64>(),
                        rng.gen::<u64>(),
                    ],
                }
                .bitand(I256::low_mask(log2_bound));
                if rng.gen::<bool>() {
                    e = e.wrapping_neg();
                }
                slot.wrapping_add(e.mul_u64(p.t)).bitand(mask)
            })
            .collect();
        WideCt {
            c0,
            c1: ct.c1.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::slots;

    fn small() -> BgvWide {
        BgvWide::new(WideParams::TEST_SMALL).expect("test params validate")
    }

    #[test]
    fn parameters_are_validated() {
        assert!(WideParams::TEST_SMALL.validate().is_ok());
        // t not a slot prime
        assert!(WideParams {
            t: 3331,
            ..WideParams::TEST_SMALL
        }
        .validate()
        .is_err());
        // q too wide for a 256-bit coefficient
        assert!(WideParams {
            log2_q: 300,
            ..WideParams::TEST_SMALL
        }
        .validate()
        .is_err());
        // n above the NTT's ceiling
        assert!(WideParams {
            n: 1 << 20,
            ..WideParams::TEST_SMALL
        }
        .validate()
        .is_err());
    }

    #[test]
    fn encrypt_decrypt_round_trips() {
        let s = small();
        let (ck, pk) = s.keygen();
        let t = s.params.t;
        let m: Vec<u64> = (0..s.params.n as u64).map(|i| (i * 37 + 11) % t).collect();
        assert_eq!(s.decrypt_poly(&ck, &s.encrypt_poly(&pk, &m)), m);
    }

    #[test]
    fn plaintext_add_and_multiply_are_homomorphic() {
        let s = small();
        let (ck, pk) = s.keygen();
        let t = s.params.t;
        let a: Vec<u64> = (0..s.params.n as u64).map(|i| (i * 5 + 1) % t).collect();
        let b: Vec<u64> = (0..s.params.n as u64).map(|i| (i * 9 + 4) % t).collect();

        let ct = s.encrypt_poly(&pk, &slots::encode(&a, s.params.n, t).unwrap());
        let sum = s.add_plain_poly(&ct, &slots::encode(&b, s.params.n, t).unwrap());
        let got = slots::decode(&s.decrypt_poly(&ck, &sum), s.params.n, t).unwrap();
        let want: Vec<u64> = a.iter().zip(&b).map(|(x, y)| (x + y) % t).collect();
        assert_eq!(got, want, "slot-wise addition");

        let prod = s.mul_plain_poly(&ct, &slots::encode(&b, s.params.n, t).unwrap());
        let got = slots::decode(&s.decrypt_poly(&ck, &prod), s.params.n, t).unwrap();
        let want: Vec<u64> = a.iter().zip(&b).map(|(x, y)| (x * y) % t).collect();
        assert_eq!(got, want, "slot-wise multiplication");
    }

    /// Flooding must leave the plaintext untouched — the whole point is that it
    /// perturbs only the noise, which vanishes under the mod-`t` decryption.
    #[test]
    fn flooding_does_not_disturb_the_plaintext() {
        let s = small();
        let (ck, pk) = s.keygen();
        let t = s.params.t;
        let m: Vec<u64> = (0..s.params.n as u64).map(|i| (i * 13 + 2) % t).collect();
        let ct = s.encrypt_poly(&pk, &m);
        for bound in [10u32, 40, 80] {
            let flooded = s.add_flooding_noise(&ct, bound);
            assert_eq!(s.decrypt_poly(&ck, &flooded), m, "flooding at 2^{bound}");
        }
    }

    /// The OLE's defining relation, at wide parameters: `a∘b − r` plus `r`
    /// reconstructs the slot-wise product.
    #[test]
    fn masked_slot_product_reconstructs() {
        let s = small();
        let (ck, pk) = s.keygen();
        let (n, t) = (s.params.n, s.params.t);
        let a: Vec<u64> = (0..n as u64).map(|i| (i * 7 + 3) % t).collect();
        let b: Vec<u64> = (0..n as u64).map(|i| (i * 11 + 5) % t).collect();
        let r: Vec<u64> = (0..n as u64).map(|i| (i * 101 + 55) % t).collect();
        let neg_r: Vec<u64> = r.iter().map(|x| (t - x) % t).collect();

        let ct = s.encrypt_poly(&pk, &slots::encode(&a, n, t).unwrap());
        let masked = s.add_plain_poly(
            &s.mul_plain_poly(&ct, &slots::encode(&b, n, t).unwrap()),
            &slots::encode(&neg_r, n, t).unwrap(),
        );
        let share = slots::decode(&s.decrypt_poly(&ck, &masked), n, t).unwrap();
        for i in 0..n {
            assert_eq!((share[i] + r[i]) % t, (a[i] * b[i]) % t, "slot {i}");
        }
    }

    #[test]
    fn centering_round_trips_across_the_limb_boundary() {
        let p = WideParams::TEST_SMALL;
        let mask = p.q_mask();
        // q/2 − 1 stays positive; q/2 wraps to negative.
        let just_under = p.half_q().wrapping_sub(I256::one());
        assert!(!center(just_under, &p).is_negative());
        assert!(center(p.half_q(), &p).is_negative());
        // Centering then re-reducing is the identity mod q.
        for v in [I256::ZERO, I256::one(), just_under, p.half_q(), mask] {
            assert_eq!(center(v, &p).bitand(mask), v);
        }
    }

    #[test]
    fn rem_t_handles_negative_coefficients() {
        let t = 3329u64;
        assert_eq!(rem_t(I256::from_i128(7), t), 7);
        assert_eq!(rem_t(I256::from_i128(-7), t), t - 7);
        assert_eq!(rem_t(I256::from_i128(0), t), 0);
        assert_eq!(rem_t(I256::from_i128(-(t as i128)), t), 0);
    }
}
