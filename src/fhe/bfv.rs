//! BFV — the securely-parameterized RLWE scheme Baru's FHE pipelines run on:
//! the **minimal-FHE inner product**, i.e. one ciphertext×ciphertext
//! multiplication, **no relinearization keys**, **no bootstrapping**, decrypt
//! the degree-2 product directly under `s²`.
//!
//! # Parameters, and why they are secure
//!
//! The only constraint is the standard RLWE bound `log q ≤ (limit for n)`. At
//! `n = 4096` the HomomorphicEncryption.org standard allows `log q ≤ 109`
//! at 128-bit security; a single multiply with a usable `t = 2^20` needs
//! `q ≈ 2^80..2^100`, which fits. So `n = 4096, q = 2^100, t = 2^20` is
//! **secure and depth-1-capable** — both at once, which is what the whole
//! construction needs.
//!
//! # The construction
//!
//! - keygen: secret `s` (ternary), public key `pk = (-(a·s + e), a)`.
//!   **No relinearization or Galois keys.**
//! - encrypt: `c = (u·b + e₁ + Δ·m, u·a + e₂)`; decrypts as
//!   `c₀ + c₁·s = Δ·m + small`.
//! - **inner product** via reversed-coefficient packing
//!   ([`Bfv::pack_input`] / [`Bfv::pack_weights`]): a single ct×ct multiply
//!   places row `r`'s dot product in coefficient `r·d + d − 1`, so one
//!   multiplication computes a whole matvec.
//! - multiply (no relin): the BFV "tensor" `(d₀,d₁,d₂)` with each
//!   component rescaled by `t/q` (a rounded shift by `log q − log t`). It
//!   decrypts under `(1, s, s²)`:
//!   `round((d₀ + d₁·s + d₂·s²)/Δ) mod t = m·m' mod t`.
//!
//! Because the only consumer of the product is a *decryption* — the masked
//! decryption of an FHE→MPC scheme switch ([`crate::hybrid`]) — the degree-2
//! key never has to be switched back down. That is what deletes the
//! relinearization keys, and with depth capped at 1 there is nothing to
//! bootstrap either. [`crate::fhe::bgv`] instantiates the same construction on
//! the other classic RLWE encoding, as a check that the encoding is not where
//! the cost lives.
//!
//! Parameters use a power-of-two `q`, so mod-q arithmetic is native wrapping
//! `u128`, the `t/q` rescale is an arithmetic shift, and `t | q` makes
//! plaintext arithmetic natively mod-t (share-sum carries in scheme switching
//! vanish by construction).

use rand::Rng;

use crate::fhe::ntt::I256;
use crate::fhe::poly;

// ─── Parameters ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BfvParams {
    pub n: usize,
    pub log2_q: u32,
    pub log2_t: u32,
}

impl BfvParams {
    /// Secure depth-1 default: n=4096, q=2^100 (≤ the 128-bit RLWE bound of
    /// ~109 for n=4096), t=2^20.
    pub const DEFAULT: BfvParams = BfvParams { n: 4096, log2_q: 100, log2_t: 20 };
    /// Small ring for fast unit tests (NOT secure — testing only).
    pub const TEST_SMALL: BfvParams = BfvParams { n: 64, log2_q: 100, log2_t: 20 };

    fn q_mask(&self) -> u128 {
        (1u128 << self.log2_q) - 1
    }
    pub fn t(&self) -> u64 {
        1u64 << self.log2_t
    }
    fn delta_shift(&self) -> u32 {
        self.log2_q - self.log2_t
    }
    fn half_q(&self) -> u128 {
        1u128 << (self.log2_q - 1)
    }
}

// ─── Polynomial helpers (coeffs mod q, power-of-two q) ────────────────────────

type Poly = Vec<u128>;

fn poly_add(a: &[u128], b: &[u128], mask: u128) -> Poly {
    a.iter().zip(b).map(|(x, y)| x.wrapping_add(*y) & mask).collect()
}

/// Negacyclic schoolbook product mod q (wrapping u128 is exact since q | 2^128).
fn poly_mul_mod(a: &[u128], b: &[u128], p: &BfvParams) -> Poly {
    // Exact negacyclic product via the multi-modular NTT, then reduced mod
    // q. Centering the operands first is sound because a ≡ a_c (mod q), so
    // the product agrees mod q; the exact result's low bits are the answer.
    let mask = p.q_mask();
    let ac: Vec<i128> = a.iter().map(|&c| center(c, p)).collect();
    let bc: Vec<i128> = b.iter().map(|&c| center(c, p)).collect();
    poly::negacyclic_product(&ac, &bc, p.n)
        .into_iter()
        .map(|c| c.low_u128() & mask)
        .collect()
}

fn center(c: u128, p: &BfvParams) -> i128 {
    if c >= p.half_q() {
        c as i128 - (1i128 << p.log2_q)
    } else {
        c as i128
    }
}

/// Rescale exact accumulators by `t/q` (= shift right by `log q − log t`,
/// rounded) and reduce mod q.
fn rescale(acc: Vec<I256>, p: &BfvParams) -> Poly {
    let mask = p.q_mask();
    let shift = p.delta_shift();
    acc.into_iter().map(|c| c.shr_round(shift).low_u128() & mask).collect()
}

fn sample_ternary(n: usize) -> Poly {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| match rng.gen_range(0u8..3) {
            0 => 0u128,
            1 => 1u128,
            _ => u128::MAX, // −1, masked on use
        })
        .collect()
}

fn sample_uniform_q(n: usize, p: &BfvParams) -> Poly {
    let mut rng = rand::thread_rng();
    let mask = p.q_mask();
    (0..n).map(|_| rng.gen::<u128>() & mask).collect()
}

// ─── Keys & ciphertexts ───────────────────────────────────────────────────────

/// A key bundle. The client bundle carries the secret `s` *and* the public
/// key `(b, a)` so a single key handle can both encrypt (needs `pk`) and
/// decrypt (needs `s`) — matching `FheAdapter`'s one-key interface. The
/// server bundle has `s` empty (public key only).
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyBundle {
    s: Vec<u128>,
    b: Vec<u128>,
    a: Vec<u128>,
}

/// A BFV ciphertext: a vector of ring elements (2 fresh, 3 after one
/// multiply). `key_pow` = the highest power of `s` it decrypts under
/// (`len − 1`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BfvCt {
    pub c: Vec<Vec<u128>>,
}

impl BfvCt {
    pub fn key_pow(&self) -> usize {
        self.c.len() - 1
    }
}

/// A fresh ciphertext held in the NTT domain, ready to be multiplied
/// against many others without re-transforming.
pub struct BfvCtNtt {
    pre: poly::PreparedPair,
}

pub struct Bfv {
    pub params: BfvParams,
}

impl Bfv {
    pub fn new(params: BfvParams) -> Self {
        Bfv { params }
    }

    /// KeyGen: ternary secret `s`, public key `(b, a) = (-(a·s + e), a)`.
    /// Returns `(client_key_bytes, public_key_bytes)`. No eval keys.
    pub fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        let p = &self.params;
        let mask = p.q_mask();
        let s = sample_ternary(p.n);
        let a = sample_uniform_q(p.n, p);
        let e = sample_ternary(p.n);
        let a_s = poly_mul_mod(&a, &s, p);
        // b = -(a·s + e)
        let b: Poly = (0..p.n)
            .map(|i| 0u128.wrapping_sub(a_s[i]).wrapping_sub(e[i]) & mask)
            .collect();
        let client = bincode::serialize(&KeyBundle { s: s.clone(), b: b.clone(), a: a.clone() }).unwrap();
        let server = bincode::serialize(&KeyBundle { s: Vec::new(), b, a }).unwrap();
        (client, server)
    }

    fn bundle(&self, key_bytes: &[u8]) -> KeyBundle {
        bincode::deserialize(key_bytes).expect("BFV key bundle")
    }

    /// Encrypt a message polynomial (coeffs mod t) with the public key:
    /// `c = (u·b + e₁ + Δ·m, u·a + e₂)`, `u` ternary.
    pub fn encrypt_poly(&self, key_bytes: &[u8], m: &[u64]) -> BfvCt {
        let p = &self.params;
        let mask = p.q_mask();
        let pk = self.bundle(key_bytes);
        let u = sample_ternary(p.n);
        let e1 = sample_ternary(p.n);
        let e2 = sample_ternary(p.n);
        let ub = poly_mul_mod(&u, &pk.b, p);
        let ua = poly_mul_mod(&u, &pk.a, p);
        let shift = p.delta_shift();
        let t_mask = (p.t() - 1) as u128;
        let c0: Poly = (0..p.n)
            .map(|i| {
                let mi = (*m.get(i).unwrap_or(&0) as u128) & t_mask;
                ub[i].wrapping_add(e1[i]).wrapping_add(mi << shift) & mask
            })
            .collect();
        let c1: Poly = (0..p.n).map(|i| ua[i].wrapping_add(e2[i]) & mask).collect();
        BfvCt { c: vec![c0, c1] }
    }

    pub fn encrypt_scalar(&self, pk_bytes: &[u8], value: u64) -> BfvCt {
        let mut m = vec![0u64; self.params.n];
        m[0] = value & (self.params.t() - 1);
        self.encrypt_poly(pk_bytes, &m)
    }

    /// Decrypt under `s^{key_pow}`: compute `Σ_i c_i · s^i`, then
    /// `round(·/Δ) mod t`.
    pub fn decrypt_poly(&self, ck_bytes: &[u8], ct: &BfvCt) -> Vec<u64> {
        let p = &self.params;
        let s = self.bundle(ck_bytes).s;
        assert!(!s.is_empty(), "BFV decrypt needs the client key (with s)");
        // Σ cᵢ·sⁱ by **Horner**: c₀ + s·(c₁ + s·c₂). That is one ring
        // multiply per extra key power (2 for a degree-2 product) instead
        // of the 6 an explicit `sⁱ` ladder costs — the ring multiply is the
        // dominant term, so this roughly triples decryption speed.
        let mask = p.q_mask();
        let mut acc = ct.c.last().expect("ciphertext has components").clone();
        for ci in ct.c.iter().rev().skip(1) {
            acc = poly_mul_mod(&acc, &s, p);
            acc = poly_add(&acc, ci, mask);
        }
        let shift = p.delta_shift();
        let t_mask = p.t() - 1;
        acc.iter()
            .map(|&c| {
                let centered = center(c, p);
                let rounded = (centered + (1i128 << (shift - 1))) >> shift;
                (rounded as u64) & t_mask
            })
            .collect()
    }

    pub fn decrypt_scalar(&self, ck_bytes: &[u8], ct: &BfvCt) -> u64 {
        self.decrypt_poly(ck_bytes, ct)[0]
    }

    /// Add a **plaintext** polynomial into a ciphertext (`c₀ += Δ·m`).
    ///
    /// O(n) — no ring multiplication, so it is essentially free next to an
    /// encryption. This is what makes the offline/online split pay: a party
    /// can precompute `Enc(ρ)` before the query arrives and, online, publish
    /// only the masked correction `δ = c − ρ` (uniform, hence leaking
    /// nothing), leaving the evaluator a plaintext addition instead of a
    /// fresh encryption.
    pub fn add_plain_poly(&self, ct: &BfvCt, m: &[u64]) -> BfvCt {
        let p = &self.params;
        let mask = p.q_mask();
        let shift = p.delta_shift();
        let t_mask = (p.t() - 1) as u128;
        let mut c = ct.c.clone();
        for (i, slot) in c[0].iter_mut().enumerate() {
            let mi = (*m.get(i).unwrap_or(&0) as u128) & t_mask;
            *slot = slot.wrapping_add(mi << shift) & mask;
        }
        BfvCt { c }
    }

    /// Add a plaintext scalar into a **single coefficient** (`c₀[i] += Δ·v`) —
    /// the sparse case of [`Bfv::add_plain_poly`], used to fold a bias
    /// directly onto a packed matvec's output position.
    ///
    /// Works on a degree-2 product as well as a fresh ciphertext: decryption
    /// computes `round(t/q · Σ cᵢ sⁱ)`, so a `Δ·v` term in `c₀` contributes
    /// `v` whatever the key power is.
    pub fn add_plain_at(&self, ct: &BfvCt, position: usize, value: u64) -> BfvCt {
        let p = &self.params;
        let mut c = ct.c.clone();
        let v = (value & (p.t() - 1)) as u128;
        c[0][position] = c[0][position].wrapping_add(v << p.delta_shift()) & p.q_mask();
        BfvCt { c }
    }

    /// Multiply a ciphertext by a **plaintext** polynomial: every component
    /// times `m`, no rescaling and no relinearization.
    ///
    /// Decryption computes `round(t/q · Σ cᵢ sⁱ)`, so scaling the whole
    /// ciphertext by an unscaled `m` scales the message: `Δ·m₀ · m ⇒ m₀·m`.
    /// The ciphertext degree is untouched, so unlike [`Bfv::mul`] this does not
    /// consume the one multiplication the depth-1 parameters allow — it is what
    /// makes single-server PIR ([`crate::pir`]) a depth-0 operation.
    ///
    /// The plaintext is **centered** into `(−t/2, t/2]` first. That is exact
    /// rather than merely convenient: `q = Δ·t` here, so `Δ·m₀·t ≡ 0 (mod q)`
    /// and subtracting `t` from a coefficient changes nothing mod `q` — while
    /// halving the coefficient magnitude that noise growth is proportional to.
    pub fn mul_plain_poly(&self, ct: &BfvCt, m: &[u64]) -> BfvCt {
        let p = &self.params;
        let mask = p.q_mask();
        let t = p.t();
        let half_t = (t >> 1) as i128;
        let mp: Poly = (0..p.n)
            .map(|i| {
                let v = (*m.get(i).unwrap_or(&0) & (t - 1)) as i128;
                let centered = if v > half_t { v - t as i128 } else { v };
                (centered as u128) & mask
            })
            .collect();
        BfvCt {
            c: ct.c.iter().map(|ci| poly_mul_mod(ci, &mp, p)).collect(),
        }
    }

    pub fn add(&self, a: &BfvCt, b: &BfvCt) -> BfvCt {
        let mask = self.params.q_mask();
        let n = a.c.len().max(b.c.len());
        let zero = vec![0u128; self.params.n];
        let c: Vec<Vec<u128>> = (0..n)
            .map(|i| poly_add(a.c.get(i).unwrap_or(&zero), b.c.get(i).unwrap_or(&zero), mask))
            .collect();
        BfvCt { c }
    }

    /// The minimal multiply: ct×ct → degree-2 ciphertext `(d₀,d₁,d₂)`, the
    /// BFV tensor product each component rescaled by `t/q`. **No
    /// relinearization** — the result decrypts under `s²`.
    ///
    /// The exact integer tensor is computed by a **multi-modular NTT**
    /// (`crate::fhe::ntt`) in O(n log n), sharing forward transforms across
    /// the three components; a power-of-two `q` admits no NTT directly, so
    /// NTT-friendly primes are used as a computational device and the exact
    /// result is CRT-reconstructed before the (unchanged) rounded-shift
    /// rescale.
    pub fn mul(&self, a: &BfvCt, b: &BfvCt) -> BfvCt {
        self.mul_pre(&self.transform(a), b)
    }

    /// Pre-transform a fresh ciphertext for reuse across several multiplies
    /// (e.g. the input of a chunked matvec against many weight chunks).
    pub fn transform(&self, ct: &BfvCt) -> BfvCtNtt {
        assert_eq!(ct.c.len(), 2, "minimal mul: fresh (degree-1) operands only");
        let p = &self.params;
        let cen = |v: &[u128]| -> Vec<i128> { v.iter().map(|&c| center(c, p)).collect() };
        BfvCtNtt {
            pre: poly::prepare_pair(&cen(&ct.c[0]), &cen(&ct.c[1]), p.n),
        }
    }

    /// Multiply a **pre-transformed** ciphertext by a fresh one, saving the
    /// first operand's forward NTTs on every call after the transform.
    pub fn mul_pre(&self, a: &BfvCtNtt, b: &BfvCt) -> BfvCt {
        assert_eq!(b.c.len(), 2);
        let p = &self.params;
        let cen = |v: &[u128]| -> Vec<i128> { v.iter().map(|&c| center(c, p)).collect() };
        let (b0, b1) = (cen(&b.c[0]), cen(&b.c[1]));
        let (t0, t1, t2) = poly::rlwe_tensor_pre(&a.pre, &b0, &b1, p.n);
        BfvCt {
            c: vec![rescale(t0, p), rescale(t1, p), rescale(t2, p)],
        }
    }

    // ── Packing (shared with `bgv` — the construction, not the scheme) ──

    pub fn pack_input(&self, x: &[i64]) -> Vec<u64> {
        assert!(x.len() <= self.params.n);
        let t_mask = self.params.t() - 1;
        let mut m = vec![0u64; self.params.n];
        for (i, &v) in x.iter().enumerate() {
            m[i] = (v as u64) & t_mask;
        }
        m
    }

    pub fn pack_weights(&self, w: &[Vec<i64>]) -> Vec<u64> {
        let d = w.first().map(|r| r.len()).unwrap_or(0);
        let rows = w.len();
        assert!(
            (rows + 1) * d <= self.params.n,
            "BFV packing: (rows+1)*in_dim = {} exceeds ring dimension {}",
            (rows + 1) * d,
            self.params.n
        );
        let t_mask = self.params.t() - 1;
        let mut m = vec![0u64; self.params.n];
        for (r, row) in w.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                m[r * d + (d - 1 - i)] = (v as u64) & t_mask;
            }
        }
        m
    }

    pub fn matvec_output_position(&self, in_dim: usize, row: usize) -> usize {
        row * in_dim + in_dim - 1
    }

    pub fn to_signed(&self, v: u64) -> i64 {
        let t = self.params.t();
        if v >= t / 2 {
            (v as i64) - (t as i64)
        } else {
            v as i64
        }
    }

    /// Serialized ciphertext size in bytes (for the E9 comparison).
    pub fn ct_bytes(&self, ct: &BfvCt) -> usize {
        ct.c.len() * self.params.n * 16
    }
}

/// [`crate::hybrid::FheAdapter`] for BFV — what lets this scheme drive the
/// generic MPC↔FHE scheme switch and the whole hybrid pipeline. The client
/// key bundle carries `pk` too, so one handle both encrypts and decrypts.
pub struct BfvAdapter<'a>(pub &'a Bfv);

impl crate::hybrid::FheAdapter for BfvAdapter<'_> {
    type Ct = BfvCt;
    fn encrypt(&self, client_key: &[u8], value: u64) -> BfvCt {
        self.0.encrypt_scalar(client_key, value)
    }
    fn decrypt(&self, client_key: &[u8], ct: &BfvCt) -> u64 {
        self.0.decrypt_scalar(client_key, ct)
    }
    fn add(&self, a: &BfvCt, b: &BfvCt) -> BfvCt {
        self.0.add(a, b)
    }
    fn backend_id(&self) -> &'static str {
        "bfv"
    }
    fn ct_bytes(&self, ct: &BfvCt) -> usize {
        self.0.ct_bytes(ct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheme() -> Bfv {
        Bfv::new(BfvParams::TEST_SMALL)
    }

    #[test]
    fn roundtrip_scalar_and_poly() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        for &v in &[0u64, 1, 42, (1 << 20) - 1, 1 << 19] {
            let ct = s.encrypt_scalar(&pk, v);
            assert_eq!(ct.key_pow(), 1);
            assert_eq!(s.decrypt_scalar(&ck, &ct), v, "scalar {v}");
        }
        let m: Vec<u64> = (0..s.params.n as u64).map(|i| i % (1 << 20)).collect();
        let ct = s.encrypt_poly(&pk, &m);
        assert_eq!(s.decrypt_poly(&ck, &ct), m);
    }

    #[test]
    fn addition_is_homomorphic() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        let a = s.encrypt_scalar(&pk, 100);
        let b = s.encrypt_scalar(&pk, 250);
        assert_eq!(s.decrypt_scalar(&ck, &s.add(&a, &b)), 350);
    }

    #[test]
    fn multiply_decrypts_under_s_squared_no_relin() {
        // The headline: ct×ct with NO relinearization, degree-2 result
        // decrypted under s².
        let s = scheme();
        let (ck, pk) = s.keygen();
        for &(a, b) in &[(3u64, 5u64), (123, 456), (1000, 1000)] {
            let ca = s.encrypt_scalar(&pk, a);
            let cb = s.encrypt_scalar(&pk, b);
            let cm = s.mul(&ca, &cb);
            assert_eq!(cm.key_pow(), 2, "product must be degree-2 (no relin)");
            assert_eq!(s.decrypt_scalar(&ck, &cm), (a * b) % (1 << 20), "{a}*{b} under s²");
        }
    }

    #[test]
    fn packed_matvec_all_rows_in_one_multiplication() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        let x = vec![3i64, -4, 5, -1];
        let w = vec![vec![2i64, 3, -1, 4], vec![-5i64, 1, 0, 2], vec![7i64, -7, 2, 2]];
        let cx = s.encrypt_poly(&pk, &s.pack_input(&x));
        let cw = s.encrypt_poly(&pk, &s.pack_weights(&w));
        let prod = s.mul(&cx, &cw); // the WHOLE matvec: one ct×ct, no relin
        let out = s.decrypt_poly(&ck, &prod);
        for (r, row) in w.iter().enumerate() {
            let expected: i64 = x.iter().zip(row).map(|(a, b)| a * b).sum();
            let pos = s.matvec_output_position(x.len(), r);
            assert_eq!(s.to_signed(out[pos]), expected, "row {r}");
        }
    }

    #[test]
    #[ignore] // n=4096 schoolbook ring — slow in debug, fast in release
    fn full_secure_params_matvec_roundtrip() {
        let s = Bfv::new(BfvParams::DEFAULT);
        let (ck, pk) = s.keygen();
        let x: Vec<i64> = (0..8).map(|i| (i * 7 % 101) - 50).collect();
        let w: Vec<Vec<i64>> = (0..2)
            .map(|r| (0..8).map(|i| ((r * 13 + i * 7) % 101) - 50).collect())
            .collect();
        let cx = s.encrypt_poly(&pk, &s.pack_input(&x));
        let cw = s.encrypt_poly(&pk, &s.pack_weights(&w));
        let prod = s.mul(&cx, &cw);
        let out = s.decrypt_poly(&ck, &prod);
        for (r, row) in w.iter().enumerate() {
            let expected: i64 = x.iter().zip(row).map(|(a, b)| a * b).sum();
            assert_eq!(s.to_signed(out[s.matvec_output_position(x.len(), r)]), expected, "row {r}");
        }
    }

    /// Measure the ACTUAL post-multiply noise budget of a depth-1 ct×ct
    /// product — the number that gates threshold decryption with smudging
    /// (`docs/malicious-hybrid.md` §6). Prints, does not assert a bound, since
    /// the point is to read the measured value into the parameter analysis.
    #[test]
    fn measures_post_multiply_noise_budget() {
        for (log2_t, wmag) in [(20u32, 8i64), (24, 8), (20, 64)] {
            let p = BfvParams { n: 4096, log2_q: 100, log2_t };
            let s = Bfv::new(p);
            let (ck, pk) = s.keygen();
            let t = p.t();
            let mut rng = rand::thread_rng();
            // One operand full-range (packed activations near t), the other a
            // small weight window — the pipeline's worst case.
            let m1: Vec<u64> = (0..p.n).map(|_| rng.gen::<u64>() % t).collect();
            let m2: Vec<u64> = (0..p.n).map(|_| (rng.gen_range(-wmag..=wmag)).rem_euclid(t as i64) as u64).collect();
            let c = s.mul(&s.encrypt_poly(&pk, &m1), &s.encrypt_poly(&pk, &m2));

            // acc = Σ cᵢ·sⁱ (the pre-round decrypt inner value).
            let sk = s.bundle(&ck).s;
            let mask = p.q_mask();
            let mut acc = c.c.last().unwrap().clone();
            for ci in c.c.iter().rev().skip(1) {
                acc = poly_mul_mod(&acc, &sk, &p);
                acc = poly_add(&acc, ci, mask);
            }
            // The decrypt is correct, so its output IS the true plaintext m;
            // the noise is acc − Δ·m (mod q, centered).
            let m_true = s.decrypt_poly(&ck, &c);
            let delta = 1u128 << p.delta_shift();
            let mut max_noise: i128 = 0;
            for (&accv, &mv) in acc.iter().zip(&m_true) {
                let dm = delta.wrapping_mul(mv as u128) & mask;
                let noise = center(accv.wrapping_sub(dm) & mask, &p).unsigned_abs() as i128;
                max_noise = max_noise.max(noise);
            }
            let noise_bits = (max_noise as f64).log2();
            let correctness_bound = p.log2_q as f64 - 1.0 - log2_t as f64; // log2(q/2t)
            println!(
                "n=4096 q=2^100 t=2^{log2_t} wmag={wmag}: |e_mult|≈2^{:.1}, correctness bound q/2t=2^{:.0}, budget left ≈ {:.1} bits",
                noise_bits, correctness_bound, correctness_bound - noise_bits
            );
        }
    }
}
