//! The one place every FHE scheme in this crate asks for a polynomial
//! product — and the switch that decides *who computes it*.
//!
//! `bfv`, `bgv`, `bgv_wide` and `pir` all bottom out in the same operation: the
//! exact negacyclic product of centered coefficients in `ℤ[X]/(X^n + 1)`. Two
//! implementations exist:
//!
//! - [`Backend::Ntt`] — [`crate::fhe::ntt`], the hand-written multi-modular
//!   NTT. Always available, pure Rust, no toolchain.
//! - [`Backend::Ntl`] — [`crate::fhe::ntl`], NTL over GMP. Requires
//!   `--features ntl`; ~2.5× faster (`baru bench-polymul`).
//!
//! The choice is a **process-global runtime switch**, not a compile-time one,
//! so a single benchmark binary can measure the same scheme both ways without
//! rebuilding. That is the entire reason this module exists; a `cfg` would
//! have been simpler and would have made the comparison impossible.
//!
//! ```no_run
//! use baru::fhe::poly::{self, Backend};
//! // Fastest available, whatever the build has.
//! poly::use_fastest();
//! assert!(poly::backend() == Backend::Ntl || poly::backend() == Backend::Ntt);
//! ```
//!
//! # Thread-safety and scope of the switch
//!
//! The backend is a single `AtomicU8` read once per product. Setting it is not
//! scoped and not per-scheme: it changes the arithmetic under *every* FHE
//! scheme in the process, immediately. That is fine for benchmarks and for a
//! process that picks one backend at startup, and it is the wrong tool for
//! running two backends concurrently in different threads. Both implementations
//! are pure functions with identical outputs, so a racing switch can only
//! change which one runs, never the result.
//!
//! # Why the results are interchangeable
//!
//! Both compute the same exact integer. `ntl.rs`'s tests assert this at
//! `n = 4096` with ±2⁹⁹ coefficients, and [`tests::backends_agree`] below
//! re-checks it through this dispatch layer. So switching backends is a pure
//! performance decision — no scheme's correctness, noise budget or security
//! depends on it.

use std::sync::atomic::{AtomicU8, Ordering};

use super::ntt::{self, I256};

/// Which implementation computes the ring products.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The hand-written multi-modular NTT (`fhe::ntt`). Always available.
    Ntt,
    /// NTL over GMP (`fhe::ntl`). Only with `--features ntl`.
    Ntl,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Ntt => "ntt",
            Backend::Ntl => "ntl",
        }
    }

    /// Parse a CLI value (`"ntt"` / `"ntl"`).
    pub fn parse(s: &str) -> Result<Backend, String> {
        match s {
            "ntt" => Ok(Backend::Ntt),
            "ntl" => Ok(Backend::Ntl),
            other => Err(format!("unknown polynomial backend {other:?} (want \"ntt\" or \"ntl\")")),
        }
    }
}

const NTT: u8 = 0;
const NTL: u8 = 1;

static BACKEND: AtomicU8 = AtomicU8::new(NTT);

/// Is `b` compiled into this build?
pub fn is_available(b: Backend) -> bool {
    match b {
        Backend::Ntt => true,
        Backend::Ntl => cfg!(feature = "ntl"),
    }
}

/// Every backend this build can actually run.
pub fn available() -> Vec<Backend> {
    let mut v = vec![Backend::Ntt];
    if is_available(Backend::Ntl) {
        v.push(Backend::Ntl);
    }
    v
}

/// The backend in force.
pub fn backend() -> Backend {
    match BACKEND.load(Ordering::Relaxed) {
        NTL => Backend::Ntl,
        _ => Backend::Ntt,
    }
}

/// Select the backend for the whole process.
///
/// Returns an error rather than falling back silently if the build lacks it —
/// a benchmark that quietly measured the NTT while reporting "ntl" would be
/// worse than one that refuses to run.
pub fn set_backend(b: Backend) -> Result<(), String> {
    if !is_available(b) {
        return Err(format!(
            "polynomial backend {:?} is not in this build (rebuild with --features ntl)",
            b.name()
        ));
    }
    BACKEND.store(match b { Backend::Ntl => NTL, Backend::Ntt => NTT }, Ordering::Relaxed);
    Ok(())
}

/// Select the fastest backend, and report it.
///
/// That is now **always [`Backend::Ntt`]**. It used to be NTL by ~2.5×, and
/// this function used to prefer it; then `ntt.rs`'s butterfly was rewritten to
/// use Shoup-preconditioned multiplication (the thing profiling NTL revealed it
/// was doing and we were not), and the hand-written transform became ~1.8×
/// *faster* than NTL — it keeps this crate's cheap fixed-width CRT conversion
/// **and** has the fast butterfly, where NTL pays generic GMP bignum costs for
/// the conversion. `baru bench-polymul` is the measurement.
///
/// The NTL backend is kept because it is a genuinely independent implementation
/// and therefore a differential oracle for the one we rely on — see
/// [`tests::backends_agree`] and `fhe::ntl`. It is no longer a performance
/// option.
pub fn use_fastest() -> Backend {
    set_backend(Backend::Ntt).expect("the NTT backend is always available");
    Backend::Ntt
}

// ─── Dispatched operations ───────────────────────────────────────────────────

/// Exact negacyclic product `c = a·b` in `ℤ[X]/(X^n + 1)`.
pub fn negacyclic_product(a: &[i128], b: &[i128], n: usize) -> Vec<I256> {
    match backend() {
        Backend::Ntt => ntt::exact_negacyclic_product(a, b, n),
        #[cfg(feature = "ntl")]
        Backend::Ntl => super::ntl::exact_negacyclic_product(a, b, n),
        #[cfg(not(feature = "ntl"))]
        Backend::Ntl => unreachable!("set_backend rejects unavailable backends"),
    }
}

/// A single polynomial prepared for repeated multiplication — the decryption
/// key's forward transform, hoisted out of the per-ciphertext loop.
///
/// Under NTL this keeps the raw coefficients (that backend owns its transforms
/// and exposes no handle), so the saving is NTT-only — the same asymmetry
/// [`prepare_pair`] documents.
pub enum PreparedPoly {
    Ntt(Box<ntt::Transformed>),
    Raw(Vec<i128>),
}

/// Prepare the reusable operand of a repeated single product.
pub fn prepare_poly(a: &[i128], n: usize) -> PreparedPoly {
    match backend() {
        Backend::Ntt => PreparedPoly::Ntt(Box::new(ntt::forward_transform(a, n))),
        Backend::Ntl => PreparedPoly::Raw(a.to_vec()),
    }
}

/// The exact negacyclic product with the first operand already prepared.
pub fn negacyclic_product_pre(a: &PreparedPoly, b: &[i128], n: usize) -> Vec<I256> {
    match a {
        PreparedPoly::Ntt(t) => ntt::exact_negacyclic_product_pre(t, b, n),
        PreparedPoly::Raw(a) => negacyclic_product(a, b, n),
    }
}

/// A ciphertext pair prepared for repeated multiplication.
///
/// Under the NTT backend this holds the two forward transforms, so a chunked
/// matvec pays them once rather than per weight chunk. NTL owns its transforms
/// internally and exposes no such handle, so under that backend this just
/// keeps the coefficients and the saving does not exist — a real asymmetry,
/// and the reason [`prepare_pair`] is a backend-dispatched type rather than a
/// plain alias for `ntt::Transformed`.
pub enum PreparedPair {
    Ntt(Box<ntt::Transformed>, Box<ntt::Transformed>),
    Raw(Vec<i128>, Vec<i128>),
}

/// Prepare the reusable operand of a repeated multiplication.
pub fn prepare_pair(a0: &[i128], a1: &[i128], n: usize) -> PreparedPair {
    match backend() {
        Backend::Ntt => PreparedPair::Ntt(
            Box::new(ntt::forward_transform(a0, n)),
            Box::new(ntt::forward_transform(a1, n)),
        ),
        Backend::Ntl => PreparedPair::Raw(a0.to_vec(), a1.to_vec()),
    }
}

/// The RLWE multiply's three tensor components — `(a₀b₀, a₀b₁ + a₁b₀, a₁b₁)` —
/// exact in ℤ, with the first operand already prepared.
///
/// Shared by BFV and BGV: they differ in what they do with the result (a
/// `t/q` rescale versus a mask), not in the product itself.
#[allow(clippy::type_complexity)]
pub fn rlwe_tensor_pre(
    a: &PreparedPair,
    b0: &[i128],
    b1: &[i128],
    n: usize,
) -> (Vec<I256>, Vec<I256>, Vec<I256>) {
    match a {
        PreparedPair::Ntt(t0, t1) => ntt::exact_bfv_tensor_pre(t0, t1, b0, b1, n),
        PreparedPair::Raw(a0, a1) => rlwe_tensor(a0, a1, b0, b1, n),
    }
}

/// The RLWE tensor with **both** operands prepared — see
/// [`crate::fhe::ntt::exact_bfv_tensor_pre_pre`]. Under the NTL backend
/// `PreparedPair` carries raw coefficients, so this degrades to the unprepared
/// path and the saving does not exist there; that asymmetry is the same one
/// [`prepare_pair`] documents.
#[allow(clippy::type_complexity)]
pub fn rlwe_tensor_pre_pre(
    a: &PreparedPair,
    b: &PreparedPair,
    n: usize,
) -> (Vec<I256>, Vec<I256>, Vec<I256>) {
    match (a, b) {
        (PreparedPair::Ntt(a0, a1), PreparedPair::Ntt(b0, b1)) => {
            ntt::exact_bfv_tensor_pre_pre(a0, a1, b0, b1, n)
        }
        (PreparedPair::Ntt(a0, a1), PreparedPair::Raw(b0, b1)) => {
            ntt::exact_bfv_tensor_pre(a0, a1, b0, b1, n)
        }
        (PreparedPair::Raw(..), PreparedPair::Ntt(..)) => {
            // One backend is live at a time, so the two operands are always
            // prepared the same way. If that ever stops holding, fail loudly:
            // there is no correct product to compute from a raw first operand
            // and a transformed second one, and a quiet wrong answer here would
            // surface as a misclassification several layers away.
            panic!(
                "rlwe_tensor_pre_pre: operands prepared under different backends \
                 (raw × transformed) — the polynomial backend changed between \
                 preparing the weights and using them"
            )
        }
        (PreparedPair::Raw(a0, a1), PreparedPair::Raw(b0, b1)) => rlwe_tensor(a0, a1, b0, b1, n),
    }
}

/// The three tensor components from raw coefficients.
#[allow(clippy::type_complexity)]
pub fn rlwe_tensor(
    a0: &[i128],
    a1: &[i128],
    b0: &[i128],
    b1: &[i128],
    n: usize,
) -> (Vec<I256>, Vec<I256>, Vec<I256>) {
    match backend() {
        Backend::Ntt => ntt::exact_bfv_tensor(a0, a1, b0, b1, n),
        #[cfg(feature = "ntl")]
        Backend::Ntl => super::ntl::exact_bfv_tensor(a0, a1, b0, b1, n),
        #[cfg(not(feature = "ntl"))]
        Backend::Ntl => unreachable!("set_backend rejects unavailable backends"),
    }
}

/// Run `f` with `b` selected, restoring the previous backend afterwards.
///
/// Restores on unwind too, so a panicking benchmark case cannot leave the
/// process measuring the wrong backend.
pub fn with_backend<T>(b: Backend, f: impl FnOnce() -> T) -> Result<T, String> {
    struct Restore(Backend);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = set_backend(self.0);
        }
    }
    let previous = backend();
    set_backend(b)?;
    let _restore = Restore(previous);
    Ok(f())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn ntt_is_always_available_and_is_the_default() {
        assert!(is_available(Backend::Ntt));
        assert!(available().contains(&Backend::Ntt));
        set_backend(Backend::Ntt).unwrap();
        assert_eq!(backend(), Backend::Ntt);
    }

    #[test]
    fn unavailable_backends_are_rejected_not_silently_downgraded() {
        if !is_available(Backend::Ntl) {
            assert!(set_backend(Backend::Ntl).is_err());
            assert_eq!(backend(), Backend::Ntt, "a rejected set must not change the backend");
        }
    }

    #[test]
    fn parse_round_trips_and_rejects_junk() {
        for b in [Backend::Ntt, Backend::Ntl] {
            assert_eq!(Backend::parse(b.name()).unwrap(), b);
        }
        assert!(Backend::parse("fft").is_err());
    }

    /// The property the whole switch rests on: whichever backend runs, the
    /// answer is the same.
    #[test]
    fn backends_agree() {
        let n = 1024;
        let (a0, a1) = (sample(n, 99, 1), sample(n, 99, 2));
        let (b0, b1) = (sample(n, 99, 3), sample(n, 99, 4));

        let mut products = Vec::new();
        let mut tensors = Vec::new();
        for b in available() {
            with_backend(b, || {
                products.push(negacyclic_product(&a0, &b0, n));
                tensors.push(rlwe_tensor(&a0, &a1, &b0, &b1, n));
                // ...and via the prepared path, which must match the direct one.
                let pre = prepare_pair(&a0, &a1, n);
                assert_eq!(rlwe_tensor_pre(&pre, &b0, &b1, n), *tensors.last().unwrap());
            })
            .unwrap();
        }
        assert!(products.windows(2).all(|w| w[0] == w[1]), "backends disagree on the product");
        assert!(tensors.windows(2).all(|w| w[0] == w[1]), "backends disagree on the tensor");
    }

    #[test]
    fn with_backend_restores_the_previous_choice() {
        set_backend(Backend::Ntt).unwrap();
        with_backend(Backend::Ntt, || assert_eq!(backend(), Backend::Ntt)).unwrap();
        assert_eq!(backend(), Backend::Ntt);
    }
}
