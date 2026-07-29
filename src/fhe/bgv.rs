//! **BGV** — Brakerski–Gentry–Vaikuntanathan, instantiated with exactly the
//! tricks that make [`crate::fhe::si_ntru`] cheap, on an RLWE lattice that is
//! actually secure.
//!
//! # The point of this module
//!
//! `si_ntru` established the *construction*: reversed-coefficient packing so
//! one ciphertext×ciphertext multiply computes a whole matvec, decryption
//! under the **squared key** so relinearization is deleted, and depth 1 so
//! bootstrapping never happens. That construction is scheme-agnostic; SI-NTRU
//! is just the cheapest arithmetic to run it on, and it is **overstretched and
//! insecure** (see its docs). [`crate::fhe::bfv`] instantiates the same
//! construction securely. This module is the third instantiation, and it
//! exists to answer a specific question: *how much of BFV's cost is RLWE, and
//! how much is BFV's particular encoding?*
//!
//! # BGV vs BFV: one structural difference
//!
//! Both are RLWE. They differ in where the message lives:
//!
//! ```text
//! BFV:  c(s) = Δ·m + e          message in the HIGH bits (Δ = q/t)
//! BGV:  c(s) = m + t·e          message in the LOW bits
//! ```
//!
//! Everything follows from that line:
//!
//! - **Multiplication.** BFV's tensor product carries a `Δ²` that must be
//!   scaled back to `Δ` — so every one of the `3n` exact 256-bit accumulator
//!   coefficients gets a rounded shift (`I256::shr_round`) before reduction.
//!   In BGV, `(m + t·e)(m' + t·e') = m·m' + t·(m·e' + m'·e + t·e·e')` is
//!   *already* in the right form: the tensor is reduced mod `q` and nothing
//!   else. **BGV's multiply is BFV's multiply minus the rescale.**
//! - **Decryption.** BFV rounds `t/q · c(s)`; BGV takes `[c(s)]_q mod t`,
//!   which for a power-of-two `t` is a mask.
//! - **Noise.** BFV's post-multiply noise is driven by `Δ` and the rescale
//!   roundoff. BGV's is driven by the *messages*: the `t·e·e'` term dominates,
//!   so noise growth depends on `t`, not on the plaintext magnitudes. Budget
//!   at the default parameters is worked out below and **measured** by
//!   [`tests::measures_post_multiply_noise_budget`].
//!
//! What is *not* different: the ring, the ring dimension, the ciphertext
//! shape (2 elements fresh, 3 after a multiply), and therefore the number of
//! polynomial multiplications per ct×ct. That last point is the one to hold
//! on to when reading the benchmark — see "What to expect" below.
//!
//! # Parameters
//!
//! `n = 4096`, `q = 2^100`, `t = 2^20` — **identical to BFV's defaults on
//! purpose**, so `baru bench-fhe-innerprod` compares encodings rather than
//! parameter choices. A power-of-two `q` is not a problem for BGV here: the
//! textbook reason to want `q ≡ 1 (mod t)` is modulus switching, which this
//! depth-1 instantiation does not do (there is nothing to switch down to).
//! Decryption only needs the centered lift of `c(s)` to equal `m + t·e`
//! exactly over ℤ, i.e. `|m + t·e| < q/2`.
//!
//! Noise budget (ternary secret and errors, `‖·‖` = max coefficient):
//!
//! ```text
//! fresh:      c(s) = m + t·v,   v = e₁ + e₂·s − u·e,   ‖v‖ ≲ 3n ≈ 2^13.6
//! after mul:  m·m' + t·E,       ‖E‖ ≲ n·(‖m‖‖v'‖ + ‖m'‖‖v‖ + t‖v‖‖v'‖)
//!             worst case ⇒ ‖m·m' + t·E‖ ≲ 2^79   against q/2 = 2^99
//! ```
//!
//! That is the worst-case bound. **Measured** (`measures_post_multiply_noise_budget`,
//! `--ignored`), the quantity that actually has to stay under `q/2` is
//! `2^60.2` — **38.8 bits of slack**, because the ternary convolutions
//! concentrate around `√n` rather than the `n` the bound assumes. The same
//! test sweeps the plaintext magnitude (`wmag` 8 → 64) and the noise moves by
//! 0.1 bits, confirming the `t·v·v'` term dominates and that the budget is
//! message-independent — the practical difference from BFV, whose post-multiply
//! noise does track the plaintext.
//!
//! Depth 2 does not fit (a second multiply squares the noise past `q/2`), and
//! that is by design — the same depth-1 contract as the other two schemes.
//!
//! # Security
//!
//! RLWE at `n = 4096`, `q = 2^100` — inside the ~109-bit `log q` ceiling for
//! 128-bit security at that dimension, and with **no NTRU lattice**, so none
//! of the overstretched/fatigue analysis that sinks SI-NTRU applies. Same
//! standing as BFV; this is a deployable instantiation, not a reference.
//!
//! # What it costs, measured
//!
//! `baru bench-fhe-innerprod --poly all --reps 15`, packed 8×4 matvec, both
//! operands encrypted (this machine):
//!
//! | scheme | n | multiply (NTT) | multiply (NTL) | ct bytes |
//! |---|---|---|---|---|
//! | BFV | 4096 | 5.4 ms | 11.2 ms | 131072 |
//! | **BGV** | 4096 | **5.3 ms** | **11.1 ms** | 131072 |
//! | SI-NTRU | 2048 | 1.0 ms | 1.7 ms | 32772 |
//!
//! **BGV's multiply is indistinguishable from BFV's.** Across repeated runs
//! the gap is +0.5%, +0.5%, −0.1% — inside the ~1–2% run-to-run spread. The
//! rescale really is removed, but it is `3n = 12288` shifts against 28
//! size-4096 transforms, so it does not survive contact with the measurement.
//! (These were 20.7 ms until `ntt.rs`'s butterfly moved to Shoup-preconditioned
//! multiplication; the ratio between the two schemes did not move.)
//! *The encoding is not where the money is*, and this module is the evidence
//! for that rather than a speedup.
//!
//! **BGV is ~5× SI-NTRU, and that gap is structural**, not something a better
//! RLWE encoding can close:
//!
//! | | ring dim | ring mults per ct×ct | transforms per prime |
//! |---|---|---|---|
//! | SI-NTRU | 2048 | 1 (ct is **one** ring element) | 2 fwd + 1 inv |
//! | BFV / BGV | 4096 | 3 (ct is **two** ring elements) | 4 fwd + 3 inv |
//!
//! ~2× from the ring dimension and ~2.3× from the tensor ⇒ ≈5×, which is what
//! the table shows. That factor is the price of *not* being an overstretched
//! NTRU lattice — SI-NTRU's single-ring-element ciphertext is exactly the
//! structure that makes it insecure. So the honest reading is that RLWE costs
//! ~5× SI-NTRU no matter which RLWE encoding is chosen, and the way to make
//! the depth-1 product cheaper is faster ring arithmetic
//! ([`crate::fhe::poly`]), not a different scheme.

use rand::Rng;

use crate::fhe::ntt::I256;
use crate::fhe::poly;

// ─── Parameters ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BgvParams {
    pub n: usize,
    pub log2_q: u32,
    /// Plaintext modulus. **Any** modulus ≥ 2: a power of two for the
    /// coefficient-packed inner product (the default), or a prime with
    /// `t ≡ 1 (mod 2n)` to unlock CRT slot packing ([`crate::fhe::slots`]),
    /// which is what [`crate::mpcc::lowgear`] needs for elementwise products.
    ///
    /// BGV does not care which: unlike BFV there is no `Δ = q/t` that wants
    /// `t | q`, because the message lives in the low bits and decryption is a
    /// reduction mod `t` rather than a rounded division.
    pub t: u64,
}

impl BgvParams {
    /// Secure depth-1 default — deliberately identical to [`crate::fhe::bfv`]'s
    /// so the two encodings can be compared at equal parameters.
    pub const DEFAULT: BgvParams = BgvParams { n: 4096, log2_q: 100, t: 1 << 20 };
    /// Small ring for fast unit tests (NOT secure — testing only).
    pub const TEST_SMALL: BgvParams = BgvParams { n: 64, log2_q: 100, t: 1 << 20 };

    fn q_mask(&self) -> u128 {
        (1u128 << self.log2_q) - 1
    }
    pub fn t(&self) -> u64 {
        self.t
    }
    /// Does this parameter set admit CRT slot packing?
    pub fn supports_slots(&self) -> bool {
        crate::fhe::slots::supports_slots(self.n, self.t)
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

fn center(c: u128, p: &BgvParams) -> i128 {
    if c >= p.half_q() {
        c as i128 - (1i128 << p.log2_q)
    } else {
        c as i128
    }
}

/// Negacyclic product mod `q`, via the exact multi-modular NTT.
///
/// Operands are centered first: `a ≡ a_c (mod q)`, so the products agree mod
/// `q`, and centering keeps coefficients inside `i128` for the transform.
fn poly_mul_mod(a: &[u128], b: &[u128], p: &BgvParams) -> Poly {
    let mask = p.q_mask();
    let ac: Vec<i128> = a.iter().map(|&c| center(c, p)).collect();
    let bc: Vec<i128> = b.iter().map(|&c| center(c, p)).collect();
    poly::negacyclic_product(&ac, &bc, p.n)
        .into_iter()
        .map(|c| c.lo & mask)
        .collect()
}

/// Reduce an exact 256-bit accumulator mod `q`. **This is the whole
/// difference from BFV's `rescale`** — a mask where BFV needs a rounded
/// 256-bit shift, because BGV's message is already at the bottom of the
/// ciphertext rather than scaled by `Δ`.
fn reduce_q(acc: Vec<I256>, p: &BgvParams) -> Poly {
    let mask = p.q_mask();
    acc.into_iter().map(|c| c.lo & mask).collect()
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

fn sample_uniform_q(n: usize, p: &BgvParams) -> Poly {
    let mut rng = rand::thread_rng();
    let mask = p.q_mask();
    (0..n).map(|_| rng.gen::<u128>() & mask).collect()
}

/// Multiply a ternary-ish polynomial by `t` (mod q) — BGV scales its errors
/// by the plaintext modulus so the message occupies the low bits.
fn scale_by_t(e: &[u128], p: &BgvParams) -> Poly {
    let mask = p.q_mask();
    // A multiply rather than a shift, so a prime `t` works too. For a ternary
    // `x` held as `0 / 1 / u128::MAX`, wrapping multiplication gives `−t mod q`
    // in the `−1` case, which is what the mask then selects.
    e.iter().map(|&x| x.wrapping_mul(p.t as u128) & mask).collect()
}

// ─── Keys & ciphertexts ───────────────────────────────────────────────────────

/// Key bundle, same shape as BFV's: the client bundle carries the secret `s`
/// *and* the public key so one handle both encrypts and decrypts; the server
/// bundle has `s` empty.
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyBundle {
    s: Vec<u128>,
    b: Vec<u128>,
    a: Vec<u128>,
}

/// A BGV ciphertext: 2 ring elements fresh, 3 after one multiply.
/// `key_pow` = the highest power of `s` it decrypts under (`len − 1`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BgvCt {
    pub c: Vec<Vec<u128>>,
}

impl BgvCt {
    pub fn key_pow(&self) -> usize {
        self.c.len() - 1
    }
}

/// A fresh ciphertext held in the NTT domain, ready to be multiplied against
/// many others without re-transforming (the chunked-matvec saving).
pub struct BgvCtNtt {
    pre: poly::PreparedPair,
}

pub struct Bgv {
    pub params: BgvParams,
}

impl Bgv {
    pub fn new(params: BgvParams) -> Self {
        Bgv { params }
    }

    /// KeyGen: ternary secret `s`, public key `(b, a) = (−(a·s + t·e), a)`.
    /// The `t·` is the BGV difference — it keeps `b + a·s ≡ 0 (mod t)` so a
    /// message added to `c₀` survives the mod-`t` decryption untouched.
    /// No evaluation keys of any kind.
    pub fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        let p = &self.params;
        let mask = p.q_mask();
        let s = sample_ternary(p.n);
        let a = sample_uniform_q(p.n, p);
        let te = scale_by_t(&sample_ternary(p.n), p);
        let a_s = poly_mul_mod(&a, &s, p);
        let b: Poly = (0..p.n)
            .map(|i| 0u128.wrapping_sub(a_s[i]).wrapping_sub(te[i]) & mask)
            .collect();
        let client = bincode::serialize(&KeyBundle { s: s.clone(), b: b.clone(), a: a.clone() }).unwrap();
        let server = bincode::serialize(&KeyBundle { s: Vec::new(), b, a }).unwrap();
        (client, server)
    }

    fn bundle(&self, key_bytes: &[u8]) -> KeyBundle {
        bincode::deserialize(key_bytes).expect("BGV key bundle")
    }

    /// Encrypt a message polynomial (coeffs mod `t`) under the public key:
    /// `c = (u·b + t·e₁ + m, u·a + t·e₂)`, `u` ternary.
    ///
    /// Then `c₀ + c₁·s = m + t·(e₁ + e₂·s − u·e)` — the message sits in the
    /// low bits with everything else a multiple of `t`.
    pub fn encrypt_poly(&self, key_bytes: &[u8], m: &[u64]) -> BgvCt {
        let p = &self.params;
        let mask = p.q_mask();
        let pk = self.bundle(key_bytes);
        let u = sample_ternary(p.n);
        let te1 = scale_by_t(&sample_ternary(p.n), p);
        let te2 = scale_by_t(&sample_ternary(p.n), p);
        let ub = poly_mul_mod(&u, &pk.b, p);
        let ua = poly_mul_mod(&u, &pk.a, p);
        let t = p.t() as u128;
        let c0: Poly = (0..p.n)
            .map(|i| {
                let mi = (*m.get(i).unwrap_or(&0) as u128) % t;
                ub[i].wrapping_add(te1[i]).wrapping_add(mi) & mask
            })
            .collect();
        let c1: Poly = (0..p.n).map(|i| ua[i].wrapping_add(te2[i]) & mask).collect();
        BgvCt { c: vec![c0, c1] }
    }

    pub fn encrypt_scalar(&self, pk_bytes: &[u8], value: u64) -> BgvCt {
        let mut m = vec![0u64; self.params.n];
        m[0] = value % self.params.t();
        self.encrypt_poly(pk_bytes, &m)
    }

    /// Decrypt under `s^{key_pow}`: `Σ cᵢ·sⁱ` by Horner, centered, then
    /// `mod t`.
    ///
    /// Horner (`c₀ + s·(c₁ + s·c₂)`) costs one ring multiply per extra key
    /// power instead of the six an explicit `sⁱ` ladder needs — the same
    /// saving BFV takes. The final step is a **mask**, not BFV's rounded
    /// division by `Δ`, because the message never left the low bits.
    pub fn decrypt_poly(&self, ck_bytes: &[u8], ct: &BgvCt) -> Vec<u64> {
        let p = &self.params;
        let s = self.bundle(ck_bytes).s;
        assert!(!s.is_empty(), "BGV decrypt needs the client key (with s)");
        let mask = p.q_mask();
        let mut acc = ct.c.last().expect("ciphertext has components").clone();
        for ci in ct.c.iter().rev().skip(1) {
            acc = poly_mul_mod(&acc, &s, p);
            acc = poly_add(&acc, ci, mask);
        }
        let t = p.t();
        acc.iter()
            .map(|&c| {
                // Centered lift into (−q/2, q/2] first: `m + t·e` is a signed
                // quantity, and `(-5) mod t` must be `t-5`, which taking the
                // low bits of the *unsigned* representative would get wrong
                // whenever the noise is negative.
                let centered = center(c, p);
                centered.rem_euclid(t as i128) as u64
            })
            .collect()
    }

    pub fn decrypt_scalar(&self, ck_bytes: &[u8], ct: &BgvCt) -> u64 {
        self.decrypt_poly(ck_bytes, ct)[0]
    }

    /// Add a **plaintext** polynomial into a ciphertext (`c₀ += m`).
    ///
    /// O(n), no ring multiplication. Cheaper to state than BFV's version,
    /// which has to scale by `Δ` first. This is what the offline/online split
    /// in `crate::hybrid` uses: precompute `Enc(ρ)`, then online publish only
    /// the uniform correction `δ = c − ρ` and fold it in with this.
    pub fn add_plain_poly(&self, ct: &BgvCt, m: &[u64]) -> BgvCt {
        let p = &self.params;
        let mask = p.q_mask();
        let t = p.t() as u128;
        let mut c = ct.c.clone();
        for (i, slot) in c[0].iter_mut().enumerate() {
            let mi = (*m.get(i).unwrap_or(&0) as u128) % t;
            *slot = slot.wrapping_add(mi) & mask;
        }
        BgvCt { c }
    }

    /// Multiply a ciphertext by a **plaintext** polynomial — every component
    /// times `m`. No rescaling, no relinearization, and the ciphertext degree
    /// is untouched, so this does not consume the one multiplication the
    /// depth-1 parameters allow.
    ///
    /// The plaintext is centered into `(−t/2, t/2]` to halve the coefficient
    /// magnitude that noise growth is proportional to. Unlike BFV this is a
    /// genuine approximation choice rather than a free one: in BGV the message
    /// is not a multiple of `Δ`, so subtracting `t` from a coefficient changes
    /// the ciphertext by `t·(...)`, which lands in the noise term and cancels
    /// under the final `mod t`. Correct, but it *does* add to the noise, which
    /// is why the centering is worth taking.
    pub fn mul_plain_poly(&self, ct: &BgvCt, m: &[u64]) -> BgvCt {
        let p = &self.params;
        let mask = p.q_mask();
        let t = p.t();
        let half_t = (t >> 1) as i128;
        let mp: Poly = (0..p.n)
            .map(|i| {
                let v = (*m.get(i).unwrap_or(&0) % t) as i128;
                let centered = if v > half_t { v - t as i128 } else { v };
                (centered as u128) & mask
            })
            .collect();
        BgvCt {
            c: ct.c.iter().map(|ci| poly_mul_mod(ci, &mp, p)).collect(),
        }
    }

    /// Add **flooding noise** `t·e` with `‖e‖ ≤ 2^log2_bound` to `c₀`.
    ///
    /// The plaintext is untouched — the added term is a multiple of `t`, so it
    /// vanishes under the mod-`t` decryption — but the ciphertext's *noise* is
    /// swamped. That is the point: in an OLE the returned ciphertext's noise
    /// depends on the responder's private input, and the party who decrypts
    /// holds the secret key and can therefore read that noise off. Flooding
    /// (smudging) makes the noise distribution statistically independent of it:
    /// if the input-dependent noise is bounded by `B` and the added term is
    /// uniform over `[−2^σ·B, 2^σ·B]`, the two distributions are within `2^−σ`.
    ///
    /// The caller owns the budget check — `log2_bound` must leave
    /// `t · 2^log2_bound < q/2` room, or decryption breaks. See
    /// [`crate::mpcc::lowgear::LowGearParams::validate`], which derives it.
    pub fn add_flooding_noise(&self, ct: &BgvCt, log2_bound: u32) -> BgvCt {
        let p = &self.params;
        let mask = p.q_mask();
        let mut rng = rand::thread_rng();
        let bound = 1i128 << log2_bound;
        let mut c = ct.c.clone();
        for slot in c[0].iter_mut() {
            let e = rng.gen_range(-bound..=bound);
            let te = (e as u128).wrapping_mul(p.t as u128);
            *slot = slot.wrapping_add(te) & mask;
        }
        BgvCt { c }
    }

    pub fn add(&self, a: &BgvCt, b: &BgvCt) -> BgvCt {
        let mask = self.params.q_mask();
        let n = a.c.len().max(b.c.len());
        let zero = vec![0u128; self.params.n];
        let c: Vec<Vec<u128>> = (0..n)
            .map(|i| poly_add(a.c.get(i).unwrap_or(&zero), b.c.get(i).unwrap_or(&zero), mask))
            .collect();
        BgvCt { c }
    }

    /// The minimal multiply: ct×ct → degree-2 ciphertext `(d₀,d₁,d₂)`.
    /// **No relinearization** — the result decrypts under `s²`.
    ///
    /// The exact integer tensor comes from the multi-modular NTT with forward
    /// transforms shared across the three components (4 forward + 3 inverse
    /// per prime, not 12). Where BFV then rescales every coefficient by `t/q`,
    /// BGV reduces mod `q` and stops: `(m + t·e)(m' + t·e')` is already
    /// `m·m' + t·(...)`, the form decryption expects.
    pub fn mul(&self, a: &BgvCt, b: &BgvCt) -> BgvCt {
        self.mul_pre(&self.transform(a), b)
    }

    /// Pre-transform a fresh ciphertext for reuse across several multiplies.
    pub fn transform(&self, ct: &BgvCt) -> BgvCtNtt {
        assert_eq!(ct.c.len(), 2, "minimal mul: fresh (degree-1) operands only");
        let p = &self.params;
        let cen = |v: &[u128]| -> Vec<i128> { v.iter().map(|&c| center(c, p)).collect() };
        BgvCtNtt {
            pre: poly::prepare_pair(&cen(&ct.c[0]), &cen(&ct.c[1]), p.n),
        }
    }

    /// Multiply a **pre-transformed** ciphertext by a fresh one, saving the
    /// first operand's forward NTTs on every call after the transform.
    pub fn mul_pre(&self, a: &BgvCtNtt, b: &BgvCt) -> BgvCt {
        assert_eq!(b.c.len(), 2);
        let p = &self.params;
        let cen = |v: &[u128]| -> Vec<i128> { v.iter().map(|&c| center(c, p)).collect() };
        let (b0, b1) = (cen(&b.c[0]), cen(&b.c[1]));
        let (t0, t1, t2) = poly::rlwe_tensor_pre(&a.pre, &b0, &b1, p.n);
        BgvCt {
            c: vec![reduce_q(t0, p), reduce_q(t1, p), reduce_q(t2, p)],
        }
    }

    // ── Packing (identical to SI-NTRU and BFV — the shared construction) ──

    pub fn pack_input(&self, x: &[i64]) -> Vec<u64> {
        assert!(x.len() <= self.params.n);
        let t = self.params.t();
        let mut m = vec![0u64; self.params.n];
        for (i, &v) in x.iter().enumerate() {
            m[i] = v.rem_euclid(t as i64) as u64;
        }
        m
    }

    pub fn pack_weights(&self, w: &[Vec<i64>]) -> Vec<u64> {
        let d = w.first().map(|r| r.len()).unwrap_or(0);
        let rows = w.len();
        assert!(
            (rows + 1) * d <= self.params.n,
            "BGV packing: (rows+1)*in_dim = {} exceeds ring dimension {}",
            (rows + 1) * d,
            self.params.n
        );
        let t = self.params.t();
        let mut m = vec![0u64; self.params.n];
        for (r, row) in w.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                m[r * d + (d - 1 - i)] = v.rem_euclid(t as i64) as u64;
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
    pub fn ct_bytes(&self, ct: &BgvCt) -> usize {
        ct.c.len() * self.params.n * 16
    }
}

/// [`crate::hybrid::FheAdapter`] for BGV — lets it drive the generic MPC↔FHE
/// scheme switch and the whole hybrid pipeline, exactly as `BfvAdapter` and
/// `SiNtruAdapter` do.
pub struct BgvAdapter<'a>(pub &'a Bgv);

impl crate::hybrid::FheAdapter for BgvAdapter<'_> {
    type Ct = BgvCt;
    fn encrypt(&self, client_key: &[u8], value: u64) -> BgvCt {
        self.0.encrypt_scalar(client_key, value)
    }
    fn decrypt(&self, client_key: &[u8], ct: &BgvCt) -> u64 {
        self.0.decrypt_scalar(client_key, ct)
    }
    fn add(&self, a: &BgvCt, b: &BgvCt) -> BgvCt {
        self.0.add(a, b)
    }
    fn backend_id(&self) -> &'static str {
        "bgv"
    }
    fn ct_bytes(&self, ct: &BgvCt) -> usize {
        self.0.ct_bytes(ct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheme() -> Bgv {
        Bgv::new(BgvParams::TEST_SMALL)
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

    /// The headline: ct×ct with NO relinearization, degree-2 result decrypted
    /// under `s²` — the SI-NTRU construction on an RLWE lattice.
    #[test]
    fn multiply_decrypts_under_s_squared_no_relin() {
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

    /// Negative plaintexts exercise the centered lift in `decrypt_poly` — the
    /// one place BGV's low-bit encoding is easy to get wrong.
    #[test]
    fn negative_values_survive_the_centered_lift() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        let t = s.params.t();
        for v in [-1i64, -7, -1000, -(1 << 18)] {
            let ct = s.encrypt_scalar(&pk, (v.rem_euclid(t as i64)) as u64);
            assert_eq!(s.to_signed(s.decrypt_scalar(&ck, &ct)), v, "encrypt/decrypt {v}");
        }
        // and through a multiplication
        let ca = s.encrypt_scalar(&pk, (-6i64).rem_euclid(t as i64) as u64);
        let cb = s.encrypt_scalar(&pk, 7);
        assert_eq!(s.to_signed(s.decrypt_scalar(&ck, &s.mul(&ca, &cb))), -42);
    }

    /// The packing trick: one ciphertext×ciphertext multiply computes every
    /// row's inner product at once, both operands encrypted.
    #[test]
    fn packed_matvec_all_rows_in_one_multiplication() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        let x = vec![3i64, -4, 5, -1];
        let w = vec![vec![2i64, 3, -1, 4], vec![-5i64, 1, 0, 2], vec![7i64, -7, 2, 2]];
        let cx = s.encrypt_poly(&pk, &s.pack_input(&x));
        let cw = s.encrypt_poly(&pk, &s.pack_weights(&w));
        let prod = s.mul(&cx, &cw);
        let out = s.decrypt_poly(&ck, &prod);
        for (r, row) in w.iter().enumerate() {
            let expected: i64 = x.iter().zip(row).map(|(a, b)| a * b).sum();
            let pos = s.matvec_output_position(x.len(), r);
            assert_eq!(s.to_signed(out[pos]), expected, "row {r}");
        }
    }

    #[test]
    fn plaintext_multiply_keeps_the_degree() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        let ct = s.encrypt_scalar(&pk, 21);
        let mut m = vec![0u64; s.params.n];
        m[0] = 3;
        let out = s.mul_plain_poly(&ct, &m);
        assert_eq!(out.key_pow(), 1, "plaintext multiply must not raise the degree");
        assert_eq!(s.decrypt_scalar(&ck, &out), 63);
    }

    #[test]
    fn plain_addition_matches_ciphertext_addition() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        let ct = s.encrypt_scalar(&pk, 500);
        let mut m = vec![0u64; s.params.n];
        m[0] = 44;
        assert_eq!(s.decrypt_scalar(&ck, &s.add_plain_poly(&ct, &m)), 544);
    }

    /// `mul_pre` must agree with `mul` — it is the same product with the first
    /// operand's transforms hoisted, and the chunked matvec depends on it.
    #[test]
    fn pre_transformed_multiply_matches_the_direct_one() {
        let s = scheme();
        let (ck, pk) = s.keygen();
        let ca = s.encrypt_scalar(&pk, 111);
        let cb = s.encrypt_scalar(&pk, 9);
        let direct = s.decrypt_scalar(&ck, &s.mul(&ca, &cb));
        let pre = s.decrypt_scalar(&ck, &s.mul_pre(&s.transform(&ca), &cb));
        assert_eq!(direct, 999);
        assert_eq!(pre, 999);
    }

    /// A **prime** plaintext modulus works exactly like a power-of-two one —
    /// BGV has no `Δ = q/t` to divide, so nothing in the scheme depends on `t`
    /// factoring nicely. This is what unlocks CRT slot packing.
    #[test]
    fn prime_plaintext_modulus_round_trips() {
        // n=64 needs t ≡ 1 mod 128; 3329 is prime and 26·128 + 1.
        let s = Bgv::new(BgvParams { n: 64, log2_q: 100, t: 3329 });
        assert!(s.params.supports_slots());
        let (ck, pk) = s.keygen();
        for v in [0u64, 1, 17, 3328] {
            assert_eq!(s.decrypt_scalar(&ck, &s.encrypt_scalar(&pk, v)), v, "scalar {v}");
        }
        let m: Vec<u64> = (0..s.params.n as u64).map(|i| i % 3329).collect();
        assert_eq!(s.decrypt_poly(&ck, &s.encrypt_poly(&pk, &m)), m);
    }

    /// **Slot packing end to end**: encode two slot vectors, multiply the
    /// ciphertext by a plaintext, decrypt, decode — and get the componentwise
    /// product. This is the operation `mpcc::lowgear`'s OLE is built from, and
    /// the one coefficient packing cannot express.
    #[test]
    fn slot_packed_plaintext_multiply_is_componentwise() {
        use crate::fhe::slots;
        let (n, t) = (64usize, 3329u64);
        let s = Bgv::new(BgvParams { n, log2_q: 100, t });
        let (ck, pk) = s.keygen();

        let a: Vec<u64> = (0..n as u64).map(|i| (i * 37 + 5) % t).collect();
        let b: Vec<u64> = (0..n as u64).map(|i| (i * 11 + 2) % t).collect();

        let ct = s.encrypt_poly(&pk, &slots::encode(&a, n, t).unwrap());
        let prod = s.mul_plain_poly(&ct, &slots::encode(&b, n, t).unwrap());
        // A plaintext multiply does not raise the degree, so no depth is spent.
        assert_eq!(prod.key_pow(), 1);

        let got = slots::decode(&s.decrypt_poly(&ck, &prod), n, t).unwrap();
        let want: Vec<u64> = a.iter().zip(&b).map(|(x, y)| (x * y) % t).collect();
        assert_eq!(got, want, "slot-wise product");
    }

    /// The masked form the OLE actually sends back: `a∘b − r`, where the mask
    /// is folded in with a *plaintext* addition (O(n), no ring multiply).
    #[test]
    fn slot_packed_masked_product_matches_the_protocol_form() {
        use crate::fhe::slots;
        let (n, t) = (64usize, 3329u64);
        let s = Bgv::new(BgvParams { n, log2_q: 100, t });
        let (ck, pk) = s.keygen();

        let a: Vec<u64> = (0..n as u64).map(|i| (i * 7 + 1) % t).collect();
        let b: Vec<u64> = (0..n as u64).map(|i| (i * 13 + 3) % t).collect();
        let r: Vec<u64> = (0..n as u64).map(|i| (i * 101 + 55) % t).collect();
        let neg_r: Vec<u64> = r.iter().map(|x| (t - x) % t).collect();

        let ct = s.encrypt_poly(&pk, &slots::encode(&a, n, t).unwrap());
        let masked = s.add_plain_poly(
            &s.mul_plain_poly(&ct, &slots::encode(&b, n, t).unwrap()),
            &slots::encode(&neg_r, n, t).unwrap(),
        );
        let share0 = slots::decode(&s.decrypt_poly(&ck, &masked), n, t).unwrap();
        // share0 + r ≡ a∘b (mod t) — the OLE's defining relation.
        for i in 0..n {
            assert_eq!((share0[i] + r[i]) % t, (a[i] * b[i]) % t, "slot {i}");
        }
    }

    #[test]
    #[ignore] // n=4096 ring — slow in debug, fast in release
    fn full_secure_params_matvec_roundtrip() {
        let s = Bgv::new(BgvParams::DEFAULT);
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

    /// Measure the ACTUAL post-multiply noise, against the `q/2` correctness
    /// bound — the BGV analogue of BFV's budget test. Prints rather than
    /// asserting a bound, so the measured value can be read into the parameter
    /// analysis in the module docs.
    ///
    /// The interesting BGV-specific check is the `wmag` sweep: because the
    /// dominant noise term is `t·e·e'` and not a message term, the budget
    /// should barely move with plaintext magnitude.
    #[test]
    #[ignore] // n=4096 ring — slow in debug, fast in release
    fn measures_post_multiply_noise_budget() {
        for (log2_t, wmag) in [(20u32, 8i64), (24, 8), (20, 64)] {
            let p = BgvParams { n: 4096, log2_q: 100, t: 1u64 << log2_t };
            let s = Bgv::new(p);
            let (ck, pk) = s.keygen();
            let t = p.t();
            let mut rng = rand::thread_rng();
            let m1: Vec<u64> = (0..p.n).map(|_| rng.gen::<u64>() % t).collect();
            let m2: Vec<u64> = (0..p.n)
                .map(|_| (rng.gen_range(-wmag..=wmag)).rem_euclid(t as i64) as u64)
                .collect();
            let c = s.mul(&s.encrypt_poly(&pk, &m1), &s.encrypt_poly(&pk, &m2));

            // acc = Σ cᵢ·sⁱ, the pre-reduction decrypt value = m·m' + t·E.
            let sk = s.bundle(&ck).s;
            let mask = p.q_mask();
            let mut acc = c.c.last().unwrap().clone();
            for ci in c.c.iter().rev().skip(1) {
                acc = poly_mul_mod(&acc, &sk, &p);
                acc = poly_add(&acc, ci, mask);
            }
            // |acc| centered IS |m·m' + t·E|; that is the quantity that must
            // stay under q/2, so report it directly.
            let max_mag = acc
                .iter()
                .map(|&v| center(v, &p).unsigned_abs())
                .max()
                .unwrap_or(0);
            let bits = (max_mag as f64).log2();
            println!(
                "BGV n=4096 q=2^100 t=2^{log2_t} wmag={wmag}: |m·m' + t·E| ≈ 2^{:.1}, bound q/2 = 2^99, budget left ≈ {:.1} bits",
                bits,
                99.0 - bits
            );
            assert!(bits < 99.0, "noise exceeded the correctness bound");
        }
    }
}
