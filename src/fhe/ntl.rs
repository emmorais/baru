//! Exact negacyclic polynomial products via **NTL + GMP** — a second,
//! independent implementation of what [`crate::fhe::ntt`] computes.
//!
//! # Why a second implementation
//!
//! `ntt.rs` computes `c = a·b mod (X^n + 1)` exactly in ℤ by reducing modulo
//! four NTT-friendly primes, transforming each in O(n log n) machine-word
//! arithmetic, and reconstructing by CRT. That is a hand-written multi-modular
//! FFT: about 200 lines of modular arithmetic we maintain ourselves, with a
//! coefficient bound (`|c[k]| < Π pᵢ / 2 ≈ 2^251`) that every caller must
//! respect and that is checked only by tests.
//!
//! NTL's `ZZX` multiplication answers the same question with no bound to
//! respect and no primes to choose: coefficients are arbitrary-precision GMP
//! integers, and NTL selects its own strategy by operand size (plain,
//! Karatsuba, or homomorphic imaging / Schönhage–Strassen). It is a mature,
//! independently tuned implementation by people who do this for a living.
//!
//! Having both buys two things:
//!
//! 1. **A differential oracle.** The tests below check `ntt.rs` against NTL at
//!    full coefficient width and full ring dimension — a far stronger check
//!    than the O(n²) schoolbook reference, which is too slow to run at
//!    `n = 4096` with 100-bit coefficients.
//! 2. **A measured answer to "is the hand-written NTT worth keeping?"** — see
//!    `baru bench-polymul`, which times the two head to head.
//!
//! # What it costs
//!
//! This path is **not free of overhead**: every coefficient crosses the FFI as
//! limbs and is rebuilt as a heap-allocated GMP integer, so the conversion is
//! O(n) allocations per operand on top of the multiplication itself. At the
//! sizes this crate uses (`n ≤ 4096`, ~100-bit coefficients) that conversion is
//! a real fraction of the time, not a rounding error. The benchmark reports
//! total wall clock, conversion included, because that is what a caller pays.
//!
//! # Availability
//!
//! Behind the `ntl` cargo feature, which needs a C++ toolchain plus NTL and
//! GMP (`brew install ntl gmp`, `apt install libntl-dev libgmp-dev`). The
//! default build does not compile or link any of it; nothing in the crate
//! depends on this module.
//!
//! ```text
//! cargo test  --features ntl fhe::ntl
//! cargo run   --features ntl --release -- bench-polymul
//! ```

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use super::ntt::I256;

// ─── FFI (see cpp/ntl_polymul.cpp for the contract) ──────────────────────────

extern "C" {
    fn baru_ntl_negacyclic_mul(a: *const u64, b: *const u64, n: usize, out: *mut u64) -> c_int;
    fn baru_ntl_bfv_tensor(
        a0: *const u64,
        a1: *const u64,
        b0: *const u64,
        b1: *const u64,
        n: usize,
        d0: *mut u64,
        d1: *mut u64,
        d2: *mut u64,
    ) -> c_int;
    fn baru_ntl_version() -> *const c_char;
}

const RC_OK: c_int = 0;
const RC_OVERFLOW: c_int = 1;
const RC_BAD_ARG: c_int = 2;

/// Why an NTL product could not be returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NtlError {
    /// A result coefficient needs more than 255 magnitude bits, so it does not
    /// fit the [`I256`] the Rust side speaks. The product itself was correct —
    /// only the return representation is too narrow.
    Overflow,
    /// Degenerate arguments (`n == 0`), or lengths that disagree with `n`.
    BadArgument(String),
    /// A return code the C++ side is not documented to produce.
    Unknown(i32),
}

impl std::fmt::Display for NtlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NtlError::Overflow => write!(
                f,
                "NTL product coefficient exceeds 256 signed bits (I256 cannot represent it)"
            ),
            NtlError::BadArgument(m) => write!(f, "bad argument: {m}"),
            NtlError::Unknown(c) => write!(f, "unexpected NTL shim return code {c}"),
        }
    }
}

impl std::error::Error for NtlError {}

// ─── Limb marshalling ────────────────────────────────────────────────────────

/// `n` signed 128-bit coefficients → `2n` little-endian limbs.
fn to_limbs(v: &[i128]) -> Vec<u64> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        let u = x as u128;
        out.push(u as u64);
        out.push((u >> 64) as u64);
    }
    out
}

/// `4n` little-endian limbs → `n` two's-complement [`I256`].
fn from_limbs(limbs: &[u64], n: usize) -> Vec<I256> {
    (0..n)
        .map(|k| {
            let w = &limbs[4 * k..4 * k + 4];
            I256 {
                lo: (w[0] as u128) | ((w[1] as u128) << 64),
                hi: (w[2] as u128) | ((w[3] as u128) << 64),
            }
        })
        .collect()
}

fn check(rc: c_int) -> Result<(), NtlError> {
    match rc {
        RC_OK => Ok(()),
        RC_OVERFLOW => Err(NtlError::Overflow),
        RC_BAD_ARG => Err(NtlError::BadArgument("null pointer or n == 0".into())),
        other => Err(NtlError::Unknown(other)),
    }
}

fn check_lengths(parts: &[(&str, usize)], n: usize) -> Result<(), NtlError> {
    if n == 0 {
        return Err(NtlError::BadArgument("n must be non-zero".into()));
    }
    for &(name, len) in parts {
        if len != n {
            return Err(NtlError::BadArgument(format!(
                "{name} has length {len}, expected n = {n}"
            )));
        }
    }
    Ok(())
}

// ─── Public API (mirrors `fhe::ntt`) ─────────────────────────────────────────

/// Exact negacyclic product `c[k] = Σ_{i+j=k} a[i]·b[j] − Σ_{i+j=k+n} a[i]·b[j]`,
/// computed by NTL over GMP integers.
///
/// Drop-in for [`crate::fhe::ntt::exact_negacyclic_product`] — same inputs, same
/// outputs, different algorithm. Returns [`NtlError::Overflow`] if a result
/// coefficient does not fit `I256`; the NTT version has the same limitation but
/// wraps silently instead of reporting it.
pub fn try_exact_negacyclic_product(a: &[i128], b: &[i128], n: usize) -> Result<Vec<I256>, NtlError> {
    check_lengths(&[("a", a.len()), ("b", b.len())], n)?;
    let (la, lb) = (to_limbs(a), to_limbs(b));
    let mut out = vec![0u64; 4 * n];
    // SAFETY: `la`/`lb` hold exactly 2n limbs and `out` exactly 4n, which is
    // what the shim reads and writes for this `n`; all three outlive the call.
    let rc = unsafe { baru_ntl_negacyclic_mul(la.as_ptr(), lb.as_ptr(), n, out.as_mut_ptr()) };
    check(rc)?;
    Ok(from_limbs(&out, n))
}

/// [`try_exact_negacyclic_product`], panicking on error — the signature that
/// matches [`crate::fhe::ntt::exact_negacyclic_product`] exactly, for use as a
/// drop-in replacement and in benchmarks.
///
/// # Panics
/// If a result coefficient exceeds 256 signed bits, or lengths disagree with `n`.
pub fn exact_negacyclic_product(a: &[i128], b: &[i128], n: usize) -> Vec<I256> {
    try_exact_negacyclic_product(a, b, n).expect("NTL negacyclic product")
}

/// The BFV multiply's three tensor components — `(a₀b₀, a₀b₁ + a₁b₀, a₁b₁)` —
/// exact in ℤ, mirroring [`crate::fhe::ntt::exact_bfv_tensor`].
///
/// One FFI crossing instead of four, and the cross terms are summed as `ZZX`
/// before the negacyclic fold. There is no shared-transform saving to exploit
/// (NTL owns its transforms internally), so unlike the NTT version this is not
/// cheaper per component than four separate products — only cheaper in
/// marshalling.
#[allow(clippy::type_complexity)]
pub fn try_exact_bfv_tensor(
    a0: &[i128],
    a1: &[i128],
    b0: &[i128],
    b1: &[i128],
    n: usize,
) -> Result<(Vec<I256>, Vec<I256>, Vec<I256>), NtlError> {
    check_lengths(
        &[("a0", a0.len()), ("a1", a1.len()), ("b0", b0.len()), ("b1", b1.len())],
        n,
    )?;
    let (la0, la1) = (to_limbs(a0), to_limbs(a1));
    let (lb0, lb1) = (to_limbs(b0), to_limbs(b1));
    let mut d0 = vec![0u64; 4 * n];
    let mut d1 = vec![0u64; 4 * n];
    let mut d2 = vec![0u64; 4 * n];
    // SAFETY: as above — four 2n-limb inputs, three 4n-limb outputs, all live
    // for the duration of the call.
    let rc = unsafe {
        baru_ntl_bfv_tensor(
            la0.as_ptr(),
            la1.as_ptr(),
            lb0.as_ptr(),
            lb1.as_ptr(),
            n,
            d0.as_mut_ptr(),
            d1.as_mut_ptr(),
            d2.as_mut_ptr(),
        )
    };
    check(rc)?;
    Ok((from_limbs(&d0, n), from_limbs(&d1, n), from_limbs(&d2, n)))
}

/// [`try_exact_bfv_tensor`], panicking on error.
///
/// # Panics
/// If a result coefficient exceeds 256 signed bits, or lengths disagree with `n`.
#[allow(clippy::type_complexity)]
pub fn exact_bfv_tensor(
    a0: &[i128],
    a1: &[i128],
    b0: &[i128],
    b1: &[i128],
    n: usize,
) -> (Vec<I256>, Vec<I256>, Vec<I256>) {
    try_exact_bfv_tensor(a0, a1, b0, b1, n).expect("NTL BFV tensor")
}

/// The linked NTL and GMP versions, e.g. `"NTL 11.5.1 / GMP 6.3.0"` — recorded
/// alongside benchmark numbers so a timing can be attributed to a build.
pub fn version() -> String {
    // SAFETY: the shim returns a pointer to a NUL-terminated static buffer that
    // is never freed; we copy out of it immediately.
    unsafe { CStr::from_ptr(baru_ntl_version()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::ntt;

    /// Deterministic pseudo-random coefficients, centered in `[−2^bits, 2^bits)`.
    fn sample(n: usize, bits: u32, seed: u64) -> Vec<i128> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let hi = (s >> 1) as u128;
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let lo = (s >> 1) as u128;
                let mag = (((hi << 64) | lo) % (1u128 << bits)) as i128;
                if s & 1 == 0 { mag } else { -mag }
            })
            .collect()
    }

    /// O(n²) reference, independent of both implementations under test.
    /// Coefficients are kept small enough that `i128` products cannot overflow.
    fn schoolbook_small(a: &[i128], b: &[i128], n: usize) -> Vec<I256> {
        let mut acc = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let k = i + j;
                let p = a[i] * b[j];
                if k < n {
                    acc[k] += p;
                } else {
                    acc[k - n] -= p;
                }
            }
        }
        acc.iter()
            .map(|&x| {
                let u = x as u128;
                I256 { lo: u, hi: if x < 0 { u128::MAX } else { 0 } }
            })
            .collect()
    }

    #[test]
    fn matches_schoolbook_on_small_inputs() {
        for n in [1usize, 2, 4, 8, 16, 32] {
            let a = sample(n, 20, n as u64 * 7 + 1);
            let b = sample(n, 20, n as u64 * 13 + 5);
            assert_eq!(exact_negacyclic_product(&a, &b, n), schoolbook_small(&a, &b, n), "n={n}");
        }
    }

    /// The point of having two implementations: they must agree at the widths
    /// and dimensions the FHE schemes actually use, where the O(n²) reference
    /// is too slow to serve as the oracle.
    #[test]
    fn agrees_with_multimodular_ntt_at_full_width() {
        for (n, bits) in [(256usize, 99u32), (1024, 99), (4096, 99)] {
            let a = sample(n, bits, 0xA11CE);
            let b = sample(n, bits, 0xB0B);
            assert_eq!(
                exact_negacyclic_product(&a, &b, n),
                ntt::exact_negacyclic_product(&a, &b, n),
                "n={n}, {bits}-bit coefficients"
            );
        }
    }

    /// SI-NTRU's corner: `n = 2048`, `q = 2^120`, where `|c[k]|` reaches ~2^249
    /// — the widest case the CRT modulus in `ntt.rs` can still represent.
    #[test]
    fn agrees_with_multimodular_ntt_at_si_ntru_width() {
        let (n, bits) = (2048usize, 119u32);
        let a = sample(n, bits, 0xDEAD);
        let b = sample(n, bits, 0xBEEF);
        assert_eq!(
            exact_negacyclic_product(&a, &b, n),
            ntt::exact_negacyclic_product(&a, &b, n)
        );
    }

    #[test]
    fn negacyclic_wraparound_has_the_right_sign() {
        // X^(n-1) · X = X^n = −1 in ℤ[X]/(X^n+1).
        let n = 8;
        let mut a = vec![0i128; n];
        a[n - 1] = 1;
        let mut b = vec![0i128; n];
        b[1] = 1;
        let c = exact_negacyclic_product(&a, &b, n);
        assert_eq!(c[0], I256::ZERO.wrapping_sub(I256::ONE));
        assert!(c[1..].iter().all(|&x| x == I256::ZERO));
    }

    #[test]
    fn bfv_tensor_matches_the_ntt_tensor() {
        let n = 1024;
        let (a0, a1) = (sample(n, 99, 1), sample(n, 99, 2));
        let (b0, b1) = (sample(n, 99, 3), sample(n, 99, 4));
        assert_eq!(
            exact_bfv_tensor(&a0, &a1, &b0, &b1, n),
            ntt::exact_bfv_tensor(&a0, &a1, &b0, &b1, n)
        );
    }

    #[test]
    fn tensor_components_are_the_individual_products() {
        let n = 64;
        let (a0, a1) = (sample(n, 40, 11), sample(n, 40, 12));
        let (b0, b1) = (sample(n, 40, 13), sample(n, 40, 14));
        let (d0, d1, d2) = exact_bfv_tensor(&a0, &a1, &b0, &b1, n);
        assert_eq!(d0, exact_negacyclic_product(&a0, &b0, n));
        assert_eq!(d2, exact_negacyclic_product(&a1, &b1, n));
        let cross: Vec<I256> = exact_negacyclic_product(&a0, &b1, n)
            .into_iter()
            .zip(exact_negacyclic_product(&a1, &b0, n))
            .map(|(x, y)| x.wrapping_add(y))
            .collect();
        assert_eq!(d1, cross);
    }

    #[test]
    fn zero_and_negation_behave() {
        let n = 32;
        let a = sample(n, 60, 99);
        let zero = vec![0i128; n];
        assert!(exact_negacyclic_product(&zero, &a, n)
            .iter()
            .all(|&x| x == I256::ZERO));
        let neg: Vec<i128> = a.iter().map(|x| -x).collect();
        let pos = exact_negacyclic_product(&a, &a, n);
        let mixed = exact_negacyclic_product(&neg, &a, n);
        for (p, m) in pos.iter().zip(&mixed) {
            assert_eq!(*m, p.wrapping_neg());
        }
    }

    /// Overflow is *reported*, not silently truncated — the property the NTT
    /// path cannot offer.
    #[test]
    fn oversized_coefficients_report_overflow() {
        // Signs chosen so nothing cancels: with M = i128::MAX,
        //   c[0] = a₀b₀ − (a₁b₃ + a₂b₂ + a₃b₁) = M² + 3M² = 4M² ≈ 2^256.
        let n = 4;
        let m = i128::MAX;
        let a = vec![m; n];
        let b = vec![m, -m, -m, -m];
        assert_eq!(try_exact_negacyclic_product(&a, &b, n), Err(NtlError::Overflow));
    }

    #[test]
    fn length_mismatches_are_rejected() {
        let e = try_exact_negacyclic_product(&[1, 2, 3], &[1, 2, 3], 4).unwrap_err();
        assert!(matches!(e, NtlError::BadArgument(_)));
        let e = try_exact_negacyclic_product(&[], &[], 0).unwrap_err();
        assert!(matches!(e, NtlError::BadArgument(_)));
    }

    #[test]
    fn version_string_names_both_libraries() {
        let v = version();
        assert!(v.contains("NTL") && v.contains("GMP"), "got {v:?}");
    }
}
