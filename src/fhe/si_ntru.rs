//! Scale-invariant NTRU-based homomorphic encryption (LTV family), ported
//! from the C++ reference in Inversed-Tech's `hom-ham` repository
//! (`si-ntru/`, the `lheIrisTest` path: `Mult` + `DecryptMul`).
//!
//! # The scheme
//!
//! Ring `R_q = Z_q[X]/(X^n + 1)` (negacyclic), plaintext modulus `t`,
//! `Δ = q/t`:
//!
//! - **KeyGen**: `f = t·f' + 1` with `f'` small (ternary here; Gaussian
//!   with σ≈1 in the reference — same support in practice), retried until
//!   `f` is invertible; `g` small; public key `h = t·g·f⁻¹ mod q`.
//! - **Encrypt**: `c = e₁ + h·e₂ + Δ·m` (message in the *top* bits — the
//!   scale-invariant form, reference `ltv_base::Encrypt`).
//! - **Decrypt**: `m = round(t·(f·c mod q)/q) mod t` (reference `Decrypt`).
//! - **Add/Sub**: coefficient-wise mod q.
//! - **Mult** (scale-invariant, reference `ltv::Mult`): lift the two
//!   ciphertexts out of `R_q`, take the *exact* integer product in
//!   `Z[X]/(X^n+1)`, rescale each coefficient by `t/q` with rounding,
//!   reduce mod q. **No relinearization/key-switch**: the product simply
//!   decrypts under `f²` (reference `DecryptMul`), and after `k`
//!   multiplications under `f^(k+1)`-ish powers — [`SiNtru::decrypt`]
//!   takes the ciphertext's tracked key power and applies `f` that many
//!   times. This is the whole point: for low-depth circuits (inner
//!   products = depth 1) the expensive machinery of FHE — bootstrapping,
//!   relinearization keys, galois keys — is *deleted*, not optimized.
//!
//! # Why this is fast for inner products
//!
//! Two tricks compose (both from the reference):
//! 1. **Reversed packing** (`SampleFeature`): put `x` in coefficients
//!    `0..d` of one message polynomial and each weight row *reversed* into
//!    a block of another; then ONE ciphertext×ciphertext multiplication
//!    computes every row's inner product simultaneously — row `r`'s dot
//!    product appears in coefficient `r·d + d − 1` of the product (cross
//!    terms provably cannot land on those coefficients while
//!    `(rows+1)·d ≤ n`; asserted). The reference's iris test does exactly
//!    this for 2000-bit Hamming distances.
//! 2. **Decrypt-with-`f²`** instead of relinearizing: the only consumer of
//!    the product is a *decryption* (in Baru's case: the masked decryption
//!    of an FHE→MPC scheme switch, `crate::hybrid`), so the degree-2 key
//!    never needs to be switched back down.
//!
//! So a whole private matvec (weights AND input both encrypted) is: one
//! ring multiplication, plus one masked decryption. No PBS (TFHE spends
//! seconds per gate), no deep modulus chain (CKKS's polynomial ReLU needs
//! degree-32768 contexts), no rotation keys (CKKS's `dot_plain`).
//!
//! # Parameter choices in this port (vs the reference)
//!
//! The reference uses a random 101-bit prime `q`, n = 2048, t = 2^11. This
//! port keeps the scheme but picks `q = 2^120`, `t = 2^20`, `n = 2048`:
//! - `t | q` exactly, so `Δ·t ≡ 0 (mod q)`: plaintext arithmetic is
//!   *natively* mod-t (share-sum carries in scheme switching vanish by
//!   construction), and the `q mod t` noise term of prime-q LTV
//!   disappears.
//! - Power-of-two `q` makes mod-q arithmetic native wrapping `u128` ops
//!   (mod 2^120 is a quotient ring of mod 2^128, so wrapping arithmetic
//!   is exact until a final mask), the `t/q` rescale an arithmetic shift,
//!   and `f⁻¹` a 3-step Hensel lift — no big-integer division anywhere.
//! - The scale-invariant multiplication is done on *centered*
//!   representatives with exact 256-bit intermediates ([`I256`]) — the
//!   textbook-correct YASHE'-style product.
//!
//! Noise budget at these parameters (ternary errors): fresh noise
//! ‖v‖ ≲ n·t ≈ 2^31; after one SI-mult ≲ n·t·v + n·‖f²‖ ≈ 2^63 — against
//! a decryption threshold of Δ/2 = 2^99. Depth 1 has ~36 bits of slack
//! (enough for thousands of additions). Depth 2 does **not** fit at
//! t = 2^20: decryption under `f⁴` multiplies the rescaling roundoff by
//! ‖f⁴‖ ≈ n³·t⁴ ≈ 2^113 > Δ/2 — measured by a failing assertion before
//! this comment was written, not just derived. Going deeper requires
//! re-balancing parameters (the depth-2 test below passes with t = 2^8,
//! where ‖f⁴‖·n/2 ≈ 2^75 ≪ Δ/2 = 2^111) — exactly the "parameters
//! adapted accordingly, decryption with a different power of f" trade-off
//! this scheme family exposes explicitly instead of hiding behind
//! bootstrapping.
//!
//! # Security status — read this
//!
//! LTV/YASHE-family NTRU at *these* parameter shapes (large q relative to
//! n) sits in the **overstretched NTRU regime** and is subject to
//! subfield/dense-sublattice lattice attacks (Albrecht–Bai–Ducas,
//! CRYPTO 2016; Kirchner–Fouque 2017; Ducas–van Woerden's "NTRU fatigue"
//! puts the onset near q ≈ n^2.484, far below both this port's 2^120 and
//! the reference's 2^101 at n = 2048). That is *why* the YASHE line was
//! abandoned for production FHE. This backend exists for what Baru uses it
//! for: research/benchmarking of the depth-1 no-bootstrapping design point
//! and its scheme-switching pipeline — not production key material. Same
//! honesty rule as the trusted-dealer caveat in the MPC stack.

use rand::Rng;

// ─── Parameters ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiNtruParams {
    /// Ring dimension (power of two; ring is X^n + 1).
    pub n: usize,
    /// log2 of the ciphertext modulus q (≤ 126; q = 2^log2_q).
    pub log2_q: u32,
    /// log2 of the plaintext modulus t (t = 2^log2_t, t | q).
    pub log2_t: u32,
}

impl SiNtruParams {
    /// The default parameter set: n matches the reference (2048); q/t are
    /// powers of two per the module docs.
    pub const DEFAULT: SiNtruParams = SiNtruParams {
        n: 2048,
        log2_q: 120,
        log2_t: 20,
    };

    /// Small-ring parameters for fast unit tests (NOT for benchmarks).
    pub const TEST_SMALL: SiNtruParams = SiNtruParams {
        n: 64,
        log2_q: 120,
        log2_t: 20,
    };

    pub fn q_mask(&self) -> u128 {
        if self.log2_q >= 128 {
            u128::MAX
        } else {
            (1u128 << self.log2_q) - 1
        }
    }
    pub fn t(&self) -> u64 {
        1u64 << self.log2_t
    }
    /// Δ = q/t = 2^(log2_q − log2_t).
    pub fn delta_shift(&self) -> u32 {
        self.log2_q - self.log2_t
    }
    pub fn half_q(&self) -> u128 {
        1u128 << (self.log2_q - 1)
    }
}

// ─── Exact 256-bit signed accumulator (for the scale-invariant rescale) ──────

/// Two's-complement 256-bit integer: just enough surface for the exact
/// centered product Σ ±aᵢbⱼ (|coef| ≤ n·(q/2)² ≤ 2^249 at defaults) and the
/// rounded arithmetic right-shift that implements the t/q rescale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct I256 {
    hi: u128,
    lo: u128,
}

impl I256 {
    const ZERO: I256 = I256 { hi: 0, lo: 0 };

    fn wrapping_add(self, o: I256) -> I256 {
        let (lo, carry) = self.lo.overflowing_add(o.lo);
        I256 {
            hi: self.hi.wrapping_add(o.hi).wrapping_add(carry as u128),
            lo,
        }
    }

    fn wrapping_neg(self) -> I256 {
        let (lo, borrow) = 0u128.overflowing_sub(self.lo);
        I256 {
            hi: 0u128.wrapping_sub(self.hi).wrapping_sub(borrow as u128),
            lo,
        }
    }

    /// Exact product of two signed 128-bit values.
    fn from_mul_i128(a: i128, b: i128) -> I256 {
        let neg = (a < 0) != (b < 0);
        let ua = a.unsigned_abs();
        let ub = b.unsigned_abs();
        // 128×128 → 256 unsigned via 64-bit limbs.
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
        if neg {
            m.wrapping_neg()
        } else {
            m
        }
    }

    /// Arithmetic shift right by `k < 128` bits, rounding to nearest
    /// (round-half-away-from-zero matches the reference's
    /// `(aux ± q/2)/q`).
    fn shr_round(self, k: u32) -> I256 {
        debug_assert!(k > 0 && k < 128);
        let negative = (self.hi >> 127) & 1 == 1;
        let mag = if negative { self.wrapping_neg() } else { self };
        // mag + 2^(k-1), then logical shift right by k.
        let rounded = mag.wrapping_add(I256 {
            hi: 0,
            lo: 1u128 << (k - 1),
        });
        let lo = (rounded.lo >> k) | (rounded.hi << (128 - k));
        let hi = rounded.hi >> k;
        let shifted = I256 { hi, lo };
        if negative {
            shifted.wrapping_neg()
        } else {
            shifted
        }
    }
}

// ─── Polynomial helpers (coefficients mod q, stored masked) ──────────────────

type Poly = Vec<u128>;

fn poly_add(a: &[u128], b: &[u128], mask: u128) -> Poly {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.wrapping_add(*y) & mask)
        .collect()
}

fn poly_sub(a: &[u128], b: &[u128], mask: u128) -> Poly {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.wrapping_sub(*y) & mask)
        .collect()
}

/// Negacyclic schoolbook product mod q (wrapping u128 is exact for the
/// quotient ring since q | 2^128).
fn poly_mul_mod(a: &[u128], b: &[u128], params: &SiNtruParams) -> Poly {
    let n = params.n;
    let mut out = vec![0u128; n];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            let p = ai.wrapping_mul(bj);
            let idx = i + j;
            if idx < n {
                out[idx] = out[idx].wrapping_add(p);
            } else {
                out[idx - n] = out[idx - n].wrapping_sub(p);
            }
        }
    }
    let mask = params.q_mask();
    for c in &mut out {
        *c &= mask;
    }
    out
}

fn center(c: u128, params: &SiNtruParams) -> i128 {
    if c >= params.half_q() {
        c as i128 - (1i128 << params.log2_q)
    } else {
        c as i128
    }
}

/// The scale-invariant multiplication (reference `ltv::Mult`): exact
/// centered integer product in Z[X]/(X^n+1), each coefficient rescaled by
/// t/q (an arithmetic shift by Δ's bit width, rounded), reduced mod q.
fn poly_mul_rescale(a: &[u128], b: &[u128], params: &SiNtruParams) -> Poly {
    let n = params.n;
    let ac: Vec<i128> = a.iter().map(|&c| center(c, params)).collect();
    let bc: Vec<i128> = b.iter().map(|&c| center(c, params)).collect();
    let mut acc = vec![I256::ZERO; n];
    for (i, &ai) in ac.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in bc.iter().enumerate() {
            let p = I256::from_mul_i128(ai, bj);
            let idx = i + j;
            if idx < n {
                acc[idx] = acc[idx].wrapping_add(p);
            } else {
                acc[idx - n] = acc[idx - n].wrapping_add(p.wrapping_neg());
            }
        }
    }
    let mask = params.q_mask();
    let shift = params.delta_shift();
    acc.into_iter()
        .map(|c| c.shr_round(shift).lo & mask)
        .collect()
}

fn sample_ternary(n: usize) -> Poly {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| match rng.gen_range(0u8..3) {
            0 => 0u128,
            1 => 1u128,
            _ => u128::MAX, // −1; masked on use
        })
        .collect()
}

// ─── Keys & ciphertexts ───────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct KeyBundle {
    /// Secret key f (present only in the client bundle).
    f: Vec<u128>,
    /// Public key h.
    h: Vec<u128>,
}

/// A ciphertext plus the secret-key power it decrypts under (1 = fresh;
/// each scale-invariant multiplication adds the operands' powers).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NtruCt {
    pub poly: Vec<u128>,
    pub key_pow: u32,
}

pub struct SiNtru {
    pub params: SiNtruParams,
}

impl SiNtru {
    pub fn new(params: SiNtruParams) -> Self {
        SiNtru { params }
    }

    /// KeyGen (reference `ltv_base::KeyGen`): f = t·f' + 1 (always odd, so
    /// always invertible mod 2^log2_q — no retry loop needed, unlike the
    /// prime-q reference), h = t·g·f⁻¹.
    pub fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        let p = &self.params;
        let mask = p.q_mask();
        let t = p.t() as u128;

        let fp = sample_ternary(p.n);
        let mut f: Poly = fp.iter().map(|&c| c.wrapping_mul(t) & mask).collect();
        f[0] = f[0].wrapping_add(1) & mask;

        let f_inv = self.invert(&f);
        // sanity: f · f⁻¹ ≡ 1
        let check = poly_mul_mod(&f, &f_inv, p);
        debug_assert!(check[0] == 1 && check[1..].iter().all(|&c| c == 0));

        let g = sample_ternary(p.n);
        let tg: Poly = g.iter().map(|&c| c.wrapping_mul(t) & mask).collect();
        let h = poly_mul_mod(&tg, &f_inv, p);

        let client = bincode::serialize(&KeyBundle { f: f.clone(), h: h.clone() }).unwrap();
        let server = bincode::serialize(&KeyBundle { f: Vec::new(), h }).unwrap();
        (client, server)
    }

    /// Hensel-lift inversion of an odd polynomial mod 2^log2_q: f ≡ 1
    /// (mod t) makes v = 1 correct to log2_t bits; each v ← v(2 − f·v)
    /// doubles the correct bit count.
    fn invert(&self, f: &[u128]) -> Poly {
        let p = &self.params;
        let mask = p.q_mask();
        let mut v: Poly = vec![0; p.n];
        v[0] = 1;
        let mut correct_bits = p.log2_t;
        while correct_bits < p.log2_q {
            let fv = poly_mul_mod(f, &v, p);
            let mut two_minus = vec![0u128; p.n];
            two_minus[0] = 2u128.wrapping_sub(fv[0]) & mask;
            for i in 1..p.n {
                two_minus[i] = fv[i].wrapping_neg() & mask;
            }
            v = poly_mul_mod(&v, &two_minus, p);
            correct_bits *= 2;
        }
        v
    }

    fn keys(&self, key_bytes: &[u8]) -> KeyBundle {
        bincode::deserialize(key_bytes).expect("SiNtru: key bundle")
    }

    /// Encrypt a message polynomial (coefficients mod t):
    /// c = e₁ + h·e₂ + Δ·m (reference `ltv_base::Encrypt`).
    pub fn encrypt_poly(&self, key_bytes: &[u8], m: &[u64]) -> NtruCt {
        let p = &self.params;
        let mask = p.q_mask();
        let h = self.keys(key_bytes).h;
        let e1 = sample_ternary(p.n);
        let e2 = sample_ternary(p.n);
        let he2 = poly_mul_mod(&h, &e2, p);
        let shift = p.delta_shift();
        let t_mask = (p.t() - 1) as u128;
        let poly: Poly = (0..p.n)
            .map(|i| {
                let mi = (*m.get(i).unwrap_or(&0) as u128) & t_mask;
                (e1[i].wrapping_add(he2[i]).wrapping_add(mi << shift)) & mask
            })
            .collect();
        NtruCt { poly, key_pow: 1 }
    }

    /// Encrypt a scalar (message in coefficient 0).
    pub fn encrypt_scalar(&self, key_bytes: &[u8], value: u64) -> NtruCt {
        let mut m = vec![0u64; self.params.n];
        m[0] = value & (self.params.t() - 1);
        self.encrypt_poly(key_bytes, &m)
    }

    /// Decrypt the full message polynomial, applying f as many times as
    /// the ciphertext's key power (reference `Decrypt` for power 1,
    /// `DecryptMul` for power 2, generalized).
    pub fn decrypt_poly(&self, client_key_bytes: &[u8], ct: &NtruCt) -> Vec<u64> {
        let p = &self.params;
        let f = self.keys(client_key_bytes).f;
        assert!(!f.is_empty(), "SiNtru: decrypting requires the client key");
        let mut y = ct.poly.clone();
        for _ in 0..ct.key_pow {
            y = poly_mul_mod(&y, &f, p);
        }
        let shift = p.delta_shift();
        let t_mask = p.t() - 1;
        y.iter()
            .map(|&c| {
                let centered = center(c, p);
                // round(t·y/q) mod t = round(y / Δ) mod t
                let rounded = (centered + (1i128 << (shift - 1))) >> shift;
                (rounded as u64) & t_mask
            })
            .collect()
    }

    pub fn decrypt_scalar(&self, client_key_bytes: &[u8], ct: &NtruCt) -> u64 {
        self.decrypt_poly(client_key_bytes, ct)[0]
    }

    /// Homomorphic addition. Mixed key powers are fine: a power-j
    /// ciphertext also decrypts under f^k for k ≥ j (its noise grows by
    /// ‖f‖·n per extra power — budgeted in the module docs).
    pub fn add(&self, a: &NtruCt, b: &NtruCt) -> NtruCt {
        NtruCt {
            poly: poly_add(&a.poly, &b.poly, self.params.q_mask()),
            key_pow: a.key_pow.max(b.key_pow),
        }
    }

    pub fn sub(&self, a: &NtruCt, b: &NtruCt) -> NtruCt {
        NtruCt {
            poly: poly_sub(&a.poly, &b.poly, self.params.q_mask()),
            key_pow: a.key_pow.max(b.key_pow),
        }
    }

    /// Add a plaintext polynomial (Δ·m) — free of encryption noise.
    pub fn add_plain_at(&self, a: &NtruCt, position: usize, value: u64) -> NtruCt {
        let p = &self.params;
        let mut poly = a.poly.clone();
        let v = (value & (p.t() - 1)) as u128;
        poly[position] = poly[position].wrapping_add(v << p.delta_shift()) & p.q_mask();
        NtruCt {
            poly,
            key_pow: a.key_pow,
        }
    }

    /// The scale-invariant multiplication (reference `ltv::Mult`); result
    /// decrypts under the summed key power — no relinearization.
    pub fn mul(&self, a: &NtruCt, b: &NtruCt) -> NtruCt {
        NtruCt {
            poly: poly_mul_rescale(&a.poly, &b.poly, &self.params),
            key_pow: a.key_pow + b.key_pow,
        }
    }

    // ── Packed inner products / matvec (reference `SampleFeature(s)`) ────

    /// Pack a vector into coefficients 0..d of a message polynomial.
    pub fn pack_input(&self, x: &[i64]) -> Vec<u64> {
        assert!(x.len() <= self.params.n);
        let t_mask = self.params.t() - 1;
        let mut m = vec![0u64; self.params.n];
        for (i, &v) in x.iter().enumerate() {
            m[i] = (v as u64) & t_mask;
        }
        m
    }

    /// Pack a weight matrix, each row *reversed*, at stride `in_dim`:
    /// row r occupies coefficients [r·d, r·d+d). After one multiplication
    /// with a packed input, row r's inner product is coefficient
    /// r·d + d − 1 of the product.
    pub fn pack_weights(&self, w: &[Vec<i64>]) -> Vec<u64> {
        let d = w.first().map(|r| r.len()).unwrap_or(0);
        let rows = w.len();
        assert!(
            (rows + 1) * d <= self.params.n,
            "SiNtru packing: (rows+1)*in_dim = {} exceeds ring dimension {} — cross terms \
             would wrap negacyclically into the result coefficients",
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

    /// Coefficient index holding row r's inner product after a packed
    /// multiplication.
    pub fn matvec_output_position(&self, in_dim: usize, row: usize) -> usize {
        row * in_dim + in_dim - 1
    }

    /// Sign-extend a decrypted mod-t coefficient to i64.
    pub fn to_signed(&self, v: u64) -> i64 {
        let t = self.params.t();
        if v >= t / 2 {
            (v as i64) - (t as i64)
        } else {
            v as i64
        }
    }
}

// ─── Scheme-switching adapter ─────────────────────────────────────────────────

/// [`crate::hybrid::FheAdapter`] for SI-NTRU scalars (value in coefficient
/// 0). The generic mask-and-open conversion works *unchanged* at
/// `nbits = log2_t`: masks are residues mod t, and since Δ·t ≡ 0 (mod q),
/// mod-t wraparound of the masked sum is native — there are no share-sum
/// carries at all (cleaner than the integer-FHE backends).
pub struct SiNtruAdapter<'a>(pub &'a SiNtru);

impl crate::hybrid::FheAdapter for SiNtruAdapter<'_> {
    type Ct = NtruCt;
    fn encrypt(&self, client_key: &[u8], value: u64) -> NtruCt {
        self.0.encrypt_scalar(client_key, value)
    }
    fn decrypt(&self, client_key: &[u8], ct: &NtruCt) -> u64 {
        self.0.decrypt_scalar(client_key, ct)
    }
    fn add(&self, a: &NtruCt, b: &NtruCt) -> NtruCt {
        self.0.add(a, b)
    }
    fn backend_id(&self) -> &'static str {
        "si-ntru"
    }
}

/// Polynomial-level FHE→MPC conversion: ONE mask-and-open turns every
/// requested coefficient of `ct` into an MPC share simultaneously (vs one
/// conversion round per scalar) — this is where the packed matvec pays a
/// second time: the whole layer output crosses into MPC for the price of
/// one masked decryption. The mask polynomial is uniform mod t in *every*
/// coefficient, so the published (decrypted) polynomial reveals nothing.
pub fn ntru_poly_to_mpc_shares(
    scheme: &SiNtru,
    client_key: &[u8],
    ct: &NtruCt,
    positions: &[usize],
    n_parties: usize,
    be: &mut dyn crate::interpreter::backend::Backend,
) -> Vec<crate::interpreter::backend::ShareRef> {
    assert!(n_parties >= 2);
    let mut rng = rand::thread_rng();
    let t_mask = scheme.params.t() - 1;
    let nbits = scheme.params.log2_t as usize;

    // Each party contributes an encrypted uniform mask polynomial.
    let masks: Vec<Vec<u64>> = (0..n_parties)
        .map(|_| (0..scheme.params.n).map(|_| rng.gen::<u64>() & t_mask).collect())
        .collect();
    let mut masked = ct.clone();
    for m in &masks {
        let enc = scheme.encrypt_poly(client_key, m);
        masked = scheme.add(&masked, &enc);
    }
    // Open the masked polynomial (uniform mod t — safe to publish).
    let y = scheme.decrypt_poly(client_key, &masked);

    positions
        .iter()
        .map(|&pos| {
            let mut components: Vec<u128> = Vec::with_capacity(n_parties);
            components
                .push((y[pos] as u128).wrapping_sub(masks[0][pos] as u128));
            for m in &masks[1..] {
                components.push(0u128.wrapping_sub(m[pos] as u128));
            }
            be.import_additive(&components, nbits)
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::{fhe_to_mpc_share, mpc_share_to_fhe, FheAdapter};
    use crate::interpreter::aby3_backend::Aby3Backend;
    use crate::interpreter::backend::Backend;

    fn scheme() -> SiNtru {
        SiNtru::new(SiNtruParams::TEST_SMALL)
    }

    #[test]
    fn i256_mul_matches_i128_for_small_values() {
        for &(a, b) in &[
            (3i128, 5i128),
            (-3, 5),
            (3, -5),
            (-3, -5),
            (1i128 << 100, 1i128 << 20),
            (-(1i128 << 100), (1i128 << 20) + 7),
        ] {
            let p = I256::from_mul_i128(a, b);
            let expected = a * b; // fits i128 for these cases
            if expected >= 0 {
                assert_eq!(p.hi, 0, "hi for {a}*{b}");
            } else {
                assert_eq!(p.hi, u128::MAX, "hi for {a}*{b}");
            }
            assert_eq!(p.lo, expected as u128);
        }
    }

    #[test]
    fn i256_shr_round_rounds_to_nearest() {
        // 5 >> 1 rounds to 3 (half away from zero), −5 >> 1 to −3.
        let five = I256 { hi: 0, lo: 5 };
        assert_eq!(five.shr_round(1).lo, 3);
        let neg5 = five.wrapping_neg();
        let r = neg5.shr_round(1);
        assert_eq!(r.lo as i128, -3);
    }

    #[test]
    fn roundtrip_scalar_and_poly() {
        let s = scheme();
        let (ck, _) = s.keygen();
        for &v in &[0u64, 1, 42, (1 << 20) - 1, 1 << 19] {
            let ct = s.encrypt_scalar(&ck, v);
            assert_eq!(s.decrypt_scalar(&ck, &ct), v, "scalar {v}");
        }
        let m: Vec<u64> = (0..s.params.n as u64).map(|i| i % (1 << 20)).collect();
        let ct = s.encrypt_poly(&ck, &m);
        assert_eq!(s.decrypt_poly(&ck, &ct), m);
    }

    #[test]
    fn addition_is_homomorphic_mod_t() {
        let s = scheme();
        let (ck, _) = s.keygen();
        let a = s.encrypt_scalar(&ck, 900_000);
        let b = s.encrypt_scalar(&ck, 300_000);
        let sum = s.decrypt_scalar(&ck, &s.add(&a, &b));
        assert_eq!(sum, (900_000 + 300_000) % (1 << 20));
        let diff = s.decrypt_scalar(&ck, &s.sub(&a, &b));
        assert_eq!(diff, 600_000);
    }

    #[test]
    fn multiplication_decrypts_under_f_squared() {
        // The headline property (reference DecryptMul): no key switch, the
        // product ciphertext (key_pow 2) decrypts with f².
        let s = scheme();
        let (ck, _) = s.keygen();
        for &(a, b) in &[(3u64, 5u64), (123, 456), (1000, 1000)] {
            let ca = s.encrypt_scalar(&ck, a);
            let cb = s.encrypt_scalar(&ck, b);
            let cm = s.mul(&ca, &cb);
            assert_eq!(cm.key_pow, 2);
            assert_eq!(
                s.decrypt_scalar(&ck, &cm),
                (a * b) % (1 << 20),
                "{a}*{b} under f²"
            );
        }
    }

    #[test]
    fn depth_two_decrypts_under_f_fourth_with_adapted_parameters() {
        // "The same technique for more than one multiplication, but
        // parameters would have to be adapted accordingly": at the default
        // t = 2^20 the f⁴·roundoff term exceeds Δ/2 and depth 2 fails
        // (measured, see module docs) — with t re-balanced down to 2^8 the
        // same two sequential SI-mults decrypt correctly under f⁴.
        let s = SiNtru::new(SiNtruParams {
            n: 64,
            log2_q: 120,
            log2_t: 8,
        });
        let (ck, _) = s.keygen();
        let c2 = s.encrypt_scalar(&ck, 2);
        let c3 = s.encrypt_scalar(&ck, 3);
        let c5 = s.encrypt_scalar(&ck, 5);
        let c7 = s.encrypt_scalar(&ck, 7);
        let m23 = s.mul(&c2, &c3);
        let m57 = s.mul(&c5, &c7);
        let m = s.mul(&m23, &m57);
        assert_eq!(m.key_pow, 4);
        assert_eq!(s.decrypt_scalar(&ck, &m), 2 * 3 * 5 * 7); // 210 < 2^8
    }

    #[test]
    fn mixed_key_power_addition_works() {
        // fresh (f) + product (f²) decrypts under f² — needed by the
        // masked-decrypt conversion.
        let s = scheme();
        let (ck, _) = s.keygen();
        let ca = s.encrypt_scalar(&ck, 10);
        let cb = s.encrypt_scalar(&ck, 20);
        let cm = s.mul(&ca, &cb); // 200, pow 2
        let cf = s.encrypt_scalar(&ck, 33); // pow 1
        let sum = s.add(&cm, &cf);
        assert_eq!(sum.key_pow, 2);
        assert_eq!(s.decrypt_scalar(&ck, &sum), 233);
    }

    #[test]
    fn packed_inner_product_in_one_multiplication() {
        let s = scheme();
        let (ck, _) = s.keygen();
        let x = vec![3i64, -4, 5, 0, -2, 7];
        let w = vec![vec![2i64, 3, -1, 4, 0, -5]];
        let expected: i64 = x.iter().zip(&w[0]).map(|(a, b)| a * b).sum();

        let cx = s.encrypt_poly(&ck, &s.pack_input(&x));
        let cw = s.encrypt_poly(&ck, &s.pack_weights(&w));
        let prod = s.mul(&cx, &cw);
        let out = s.decrypt_poly(&ck, &prod);
        let pos = s.matvec_output_position(x.len(), 0);
        assert_eq!(s.to_signed(out[pos]), expected);
    }

    #[test]
    fn packed_matvec_all_rows_in_one_multiplication() {
        let s = scheme();
        let (ck, _) = s.keygen();
        let x = vec![3i64, -4, 5, -1];
        let w = vec![
            vec![2i64, 3, -1, 4],
            vec![-5i64, 1, 0, 2],
            vec![7i64, -7, 2, 2],
        ];
        let cx = s.encrypt_poly(&ck, &s.pack_input(&x));
        let cw = s.encrypt_poly(&ck, &s.pack_weights(&w));
        let prod = s.mul(&cx, &cw); // the WHOLE matvec: one ct×ct mult
        let out = s.decrypt_poly(&ck, &prod);
        for (r, row) in w.iter().enumerate() {
            let expected: i64 = x.iter().zip(row).map(|(a, b)| a * b).sum();
            let pos = s.matvec_output_position(x.len(), r);
            assert_eq!(s.to_signed(out[pos]), expected, "row {r}");
        }
    }

    #[test]
    fn scheme_switch_roundtrip_via_generic_adapter() {
        // The generic hybrid conversions, unchanged, at nbits = 20.
        let s = scheme();
        let (ck, _) = s.keygen();
        let a = SiNtruAdapter(&s);
        let nbits = s.params.log2_t as usize;
        let mut be = Aby3Backend::new();
        for &v in &[7i64, -9, 20050, -20050] {
            let residue = (v as i128).rem_euclid(1i128 << nbits) as u64;
            let ct = a.encrypt(&ck, residue);
            let sref = fhe_to_mpc_share(&a, &ck, &ct, &mut be, nbits);
            let raw = be.reveal(&sref, None);
            let got = s.to_signed((raw as u64) & (s.params.t() - 1));
            assert_eq!(got, v, "fhe->share for {v}");

            let mut be2 = Aby3Backend::new();
            let sref2 = be2.share_secret(residue as u128, nbits, 0);
            let ct2 = mpc_share_to_fhe(&mut be2, &sref2, &a, &ck);
            let got2 = s.to_signed(a.decrypt(&ck, &ct2) & (s.params.t() - 1));
            assert_eq!(got2, v, "share->fhe for {v}");
        }
    }

    #[test]
    fn poly_level_conversion_switches_all_outputs_at_once() {
        // Packed matvec output -> ONE masked decrypt -> shares -> MPC ReLU.
        let s = scheme();
        let (ck, _) = s.keygen();
        let x = vec![3i64, -4, 5, -1];
        let w = vec![vec![2i64, 3, -1, 4], vec![-50i64, 1, 0, 2]];
        let expected: Vec<i64> = w
            .iter()
            .map(|row| x.iter().zip(row).map(|(a, b)| a * b).sum::<i64>().max(0))
            .collect();

        let cx = s.encrypt_poly(&ck, &s.pack_input(&x));
        let cw = s.encrypt_poly(&ck, &s.pack_weights(&w));
        let prod = s.mul(&cx, &cw);

        let positions: Vec<usize> = (0..w.len())
            .map(|r| s.matvec_output_position(x.len(), r))
            .collect();
        let mut be = Aby3Backend::new();
        let shares = ntru_poly_to_mpc_shares(&s, &ck, &prod, &positions, 3, &mut be);
        let got: Vec<i64> = shares
            .iter()
            .map(|z| {
                let zero = be.sub(z, z);
                let mask = be.lt(&zero, z);
                let relu = be.mul(z, &mask);
                let raw = be.reveal(&relu, None) as u64;
                s.to_signed(raw & (s.params.t() - 1))
            })
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    #[ignore] // slow in debug: full n=2048 ring, schoolbook O(n²)
    fn full_params_matvec_roundtrip() {
        let s = SiNtru::new(SiNtruParams::DEFAULT);
        let (ck, _) = s.keygen();
        let x: Vec<i64> = (0..8).map(|i| (i * 7 % 101) - 50).collect();
        let w: Vec<Vec<i64>> = (0..2)
            .map(|r| (0..8).map(|i| ((r * 13 + i * 7) % 101) - 50).collect())
            .collect();
        let cx = s.encrypt_poly(&ck, &s.pack_input(&x));
        let cw = s.encrypt_poly(&ck, &s.pack_weights(&w));
        let prod = s.mul(&cx, &cw);
        let out = s.decrypt_poly(&ck, &prod);
        for (r, row) in w.iter().enumerate() {
            let expected: i64 = x.iter().zip(row).map(|(a, b)| a * b).sum();
            let pos = s.matvec_output_position(x.len(), r);
            assert_eq!(s.to_signed(out[pos]), expected, "row {r}");
        }
    }
}
