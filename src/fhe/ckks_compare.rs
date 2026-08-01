//! Approximate comparison, ReLU, and max for CKKS — the missing piece
//! `src/fhe/backend.rs`'s `RealFheBackend` doc explicitly calls out:
//! "Comparisons... are impossible on approximate ciphertexts." That's true
//! for *exact* comparison, but the CKKS/HEAAN literature has a well-established
//! answer for *approximate* comparison: don't compare — approximate `sign(x)`
//! with a polynomial, then everything built from `sign` (comparison, ReLU,
//! max, mux) falls out arithmetically:
//!
//!   `sign(x) ≈ P(x)`  for some polynomial `P`
//!   `(a > b)  ≈ (sign(a - b) + 1) / 2`
//!   `relu(x)  ≈ x · (sign(x) + 1) / 2`
//!
//! # The polynomial: Newton–Schulz iteration for `sign`
//!
//! This module uses the classical Newton's-method iteration for the scalar
//! sign function (a numerical-analysis staple predating homomorphic
//! encryption by decades — see e.g. Roberts, "Linear model reduction and
//! solution of the algebraic Riccati equation by use of the sign function",
//! 1980) — and, not coincidentally, exactly the "f" polynomial family Cheon,
//! Kim, Kim and Song introduced for *homomorphic* comparison:
//!
//!   `f(x) = x·(3 − x²) / 2`
//!
//! `f` has fixed points at `-1, 0, 1`. `f'(0) = 1.5` (repelling — iterating
//! pushes any `x ≠ 0` away from the discontinuity), `f'(±1) = 0`
//! (superattracting — quadratic convergence once near ±1). So for
//! `x₀ ∈ (-√3, √3) \ {0}`, iterating `f` converges to `sign(x₀)`: slowly at
//! first (escaping the neighbourhood of 0), then quadratically fast.
//!
//! # What this implementation is, and isn't
//!
//! Cheon–Kim–Kim–Song (Asiacrypt 2019, "Numerical Method for Comparison on
//! Homomorphically Encrypted Numbers") introduce a *second* polynomial
//! family `g`, optimized for fast convergence once already close to ±1, and
//! show that composing a few rounds of `f` (to gain distance from 0) with a
//! few rounds of `g` (to sharpen precision) gives asymptotically optimal
//! depth for a target `(α, ε)` accuracy — refined further by Cheon, Kim,
//! Kim, Lee, Song (Asiacrypt 2020) and Lee, Lee, Kim, Cheon et al.
//! ("Minimax Approximation of Sign Function by Composite Polynomial for
//! Homomorphic Comparison", <https://eprint.iacr.org/2020/834>) into a
//! depth-minimal composite of *minimax* polynomials chosen by dynamic
//! programming.
//!
//! This module implements only the `f`-iteration — the foundational,
//! easy-to-verify piece of that line of work, not the depth-optimized
//! composite. It is mathematically correct and does converge (tested below
//! against the exact `sign` function), but for a given target precision it
//! costs more multiplicative depth than the optimal method — measured
//! precisely in `bench-nn`'s CKKS-compare workload rather than assumed.
//! Implementing the optimal `f`+`g` composite (which needs the `g` family's
//! coefficients and the paper's dynamic-programming depth-allocation step)
//! is future work.
//!
//! # Domain
//!
//! `f` only converges for `|x| < √3`. Callers must supply a `bound: f64`
//! (a known or estimated max absolute pre-activation value — e.g. from
//! `crate::ml::QuantizationReport` on real sample data) and evaluate
//! `sign(x/bound)` instead of `sign(x)` directly; `sign` is scale-invariant
//! for positive `bound`, so this changes nothing about *what* is computed,
//! only keeps the iteration inside its convergence domain. A `bound` that's
//! too small (true `|x|` exceeds it) diverges silently to nonsense — exactly
//! the kind of "assumed, not measured" failure mode this crate's `ml`
//! quantization report is designed to catch for the *linear* part of a
//! network; the same discipline applies here.

/// One Newton–Schulz step: `f(x) = x·(3 − x²) / 2`.
pub fn newton_sign_step(x: f64) -> f64 {
    0.5 * x * (3.0 - x * x)
}

/// Iterate [`newton_sign_step`] `iterations` times. Converges to `sign(x)`
/// for `x ∈ (-√3, √3)`; diverges outside that range. Callers should pass
/// `x / bound` for a `bound` that is a genuine bound on `|x|` (see module
/// docs) so this is always evaluated inside its domain of convergence.
pub fn sign_approx_plain(x: f64, iterations: u32) -> f64 {
    let mut cur = x;
    for _ in 0..iterations {
        cur = newton_sign_step(cur);
    }
    cur
}

/// `relu(x) = x · (sign(x) + 1) / 2`, using [`sign_approx_plain`] scaled by
/// `bound` (see module docs — `bound` must satisfy `|x| <= bound`).
pub fn relu_approx_plain(x: f64, bound: f64, iterations: u32) -> f64 {
    let s = sign_approx_plain(x / bound, iterations);
    x * (s + 1.0) / 2.0
}

/// Approximate `(a > b)` as a real close to 0.0 or 1.0, using
/// `sign_approx_plain((a - b) / bound)`. `bound` must satisfy
/// `|a - b| <= bound`.
pub fn cmpgt_approx_plain(a: f64, b: f64, bound: f64, iterations: u32) -> f64 {
    let s = sign_approx_plain((a - b) / bound, iterations);
    (s + 1.0) / 2.0
}

// ─── Real SEAL backend: a deeper CKKS context that can afford this ───────────
//
// `CkksBackend` (ckks_backend.rs) deliberately keeps every ciphertext at the
// top modulus level and tracks scale via a `depth` counter — cheap and
// exactly right for the shallow (depth <= 2) linear workloads it targets,
// but each multiplication on that scheme *compounds* the required modulus
// budget multiplicatively (squaring an already-scaled ciphertext squares the
// scale again), so it cannot survive the several sequential multiplications
// [`sign_approx_plain`] needs. `CkksCompareBackend` below is a separate,
// independent context that does what real CKKS deployments do to support
// depth: an actual `rescale_to_next` after every multiplication, which
// resets the scale to a nominal ~2^40 and drops one prime from the modulus
// chain instead of letting the scale compound — so depth cost is linear in
// the number of primes (levels), not exponential. The cost: a much bigger
// ring dimension (to fit enough primes) and correspondingly much slower
// arithmetic — precisely the "different performance profile" trade-off
// this module exists to measure rather than hand-wave.

use sealy::{
    CKKSEncoder, CKKSEncryptionParametersBuilder, CKKSEvaluator, Ciphertext,
    CoefficientModulusFactory, Context, Decryptor, DegreeType, Encryptor, Evaluator, FromBytes,
    KeyGenerator, Plaintext, PublicKey, RelinearizationKey, SecretKey, SecurityLevel, ToBytes,
};

const COMPARE_SCALE_BITS: i32 = 40;
const COMPARE_SPECIAL_PRIME_BITS: i32 = 60;

/// A CKKS ciphertext at a specific modulus-chain level (0 = freshly
/// encrypted, all primes present; each real multiplication increments this
/// by 1 and drops one prime via `rescale_to_next`). Unlike `CkksCt`'s
/// `depth` (a scale exponent), `level` here is a genuine SEAL modulus-chain
/// index — the quantity actually bounded by the number of primes.
#[derive(Clone)]
pub struct DeepCt {
    pub ct: Ciphertext,
    pub level: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DeepCtWire {
    level: u32,
    bytes: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ClientKeys {
    secret: Vec<u8>,
    public: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ServerKeysWire {
    relin: Vec<u8>,
}

struct ServerKeys {
    relin: RelinearizationKey,
}

/// A deeper sibling of [`crate::fhe::ckks_backend::CkksBackend`], built with
/// enough real (rescale-consuming) levels to evaluate [`sign_approx_plain`]
/// homomorphically. See the module docs for why this needs its own,
/// independent parameter set and evaluation convention.
pub struct CkksCompareBackend {
    ctx: Context,
    evaluator: CKKSEvaluator,
    encoder: CKKSEncoder,
    server: Option<ServerKeys>,
    degree: u64,
}

impl CkksCompareBackend {
    /// How many real (rescale-consuming) multiplications
    /// [`Self::sign_approx`] needs for `iterations` Newton steps, including
    /// the initial `x / bound` scaling multiply.
    pub fn levels_for_sign(iterations: u32) -> u32 {
        1 + 3 * iterations
    }

    /// Levels needed by [`Self::relu_approx`]: sign_approx's levels, plus 2
    /// more for `x * (sign + 1) / 2` (one multiply to fold in the 0.5
    /// scalar, one to multiply by `x`).
    pub fn levels_for_relu(iterations: u32) -> u32 {
        Self::levels_for_sign(iterations) + 2
    }

    /// Levels needed by [`Self::cmpgt_approx`]: sign_approx's levels, plus 1
    /// more to fold `(sign + 1) / 2` into a single multiply-then-add.
    pub fn levels_for_cmpgt(iterations: u32) -> u32 {
        Self::levels_for_sign(iterations) + 1
    }

    /// Build a context with exactly `num_levels` usable (40-bit) primes
    /// bracketed by two 60-bit "special" primes, at the smallest supported
    /// ring dimension (8192 / 16384 / 32768) whose
    /// `CoefficientModulusFactory::max_bit_count` (the real,
    /// HomomorphicEncryption.org-standard security budget for that degree —
    /// looked up from SEAL itself, not assumed) can fit the requested
    /// chain at 128-bit security. Panics if not even the largest supported
    /// degree fits (i.e. `num_levels` is too large to be practical here).
    pub fn new(num_levels: u32) -> Self {
        let needed_bits = 2 * COMPARE_SPECIAL_PRIME_BITS + (num_levels as i32) * COMPARE_SCALE_BITS;
        let degree = [DegreeType::D8192, DegreeType::D16384, DegreeType::D32768]
            .into_iter()
            .find(|&d| {
                CoefficientModulusFactory::max_bit_count(d.into(), SecurityLevel::TC128) as i32
                    >= needed_bits
            })
            .unwrap_or_else(|| {
                panic!(
                    "CkksCompareBackend: {num_levels} levels needs {needed_bits} bits of \
                     coefficient modulus, more than D32768 supports at 128-bit security \
                     ({} bits) — reduce the iteration count or accept lower security",
                    CoefficientModulusFactory::max_bit_count(32768, SecurityLevel::TC128)
                )
            });
        Self::with_degree(degree, num_levels)
    }

    /// Same as [`Self::new`] but with an explicit ring dimension (for
    /// benchmarking the degree/precision/speed trade-off directly).
    pub fn with_degree(degree: DegreeType, num_levels: u32) -> Self {
        let mut bit_sizes = vec![COMPARE_SPECIAL_PRIME_BITS];
        bit_sizes.extend(std::iter::repeat_n(COMPARE_SCALE_BITS, num_levels as usize));
        bit_sizes.push(COMPARE_SPECIAL_PRIME_BITS);
        let params = CKKSEncryptionParametersBuilder::new()
            .set_poly_modulus_degree(degree)
            .set_coefficient_modulus(
                CoefficientModulusFactory::build(degree, &bit_sizes)
                    .expect("CkksCompareBackend: modulus chain"),
            )
            .build()
            .expect("CkksCompareBackend: parameters");
        let ctx = Context::new(&params, true, SecurityLevel::TC128)
            .expect("CkksCompareBackend: context (requested chain likely exceeds the security budget for this degree)");
        let evaluator = CKKSEvaluator::new(&ctx).expect("CkksCompareBackend: evaluator");
        let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(COMPARE_SCALE_BITS))
            .expect("CkksCompareBackend: encoder");
        CkksCompareBackend {
            ctx,
            evaluator,
            encoder,
            server: None,
            degree: degree.into(),
        }
    }

    /// The ring dimension (poly modulus degree) this instance was actually
    /// built with — [`Self::new`] picks the smallest supported one that
    /// fits the requested level count, so this is worth reporting alongside
    /// any timing (bigger degree means much slower ops).
    pub fn degree(&self) -> u64 {
        self.degree
    }

    pub fn with_server_keys(num_levels: u32, server_key_bytes: &[u8]) -> Self {
        let mut be = Self::new(num_levels);
        be.install_server_keys(server_key_bytes);
        be
    }

    pub fn with_degree_and_server_keys(
        degree: DegreeType,
        num_levels: u32,
        server_key_bytes: &[u8],
    ) -> Self {
        let mut be = Self::with_degree(degree, num_levels);
        be.install_server_keys(server_key_bytes);
        be
    }

    fn install_server_keys(&mut self, server_key_bytes: &[u8]) {
        let wire: ServerKeysWire =
            bincode::deserialize(server_key_bytes).expect("CkksCompareBackend: server key bundle");
        self.server = Some(ServerKeys {
            relin: RelinearizationKey::from_bytes(&self.ctx, &wire.relin)
                .expect("CkksCompareBackend: relin key"),
        });
    }

    fn server(&self) -> &ServerKeys {
        self.server.as_ref().expect(
            "CkksCompareBackend has no evaluation keys — construct it with \
             with_server_keys/with_degree_and_server_keys",
        )
    }

    pub fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        let keygen = KeyGenerator::new(&self.ctx).expect("CkksCompareBackend: keygen");
        let secret = keygen.secret_key();
        let public = keygen.create_public_key();
        let relin = keygen
            .create_relinearization_keys()
            .expect("CkksCompareBackend: relin keygen");
        let client = bincode::serialize(&ClientKeys {
            secret: secret.as_bytes().expect("sk ser"),
            public: public.as_bytes().expect("pk ser"),
        })
        .unwrap();
        let server = bincode::serialize(&ServerKeysWire {
            relin: relin.as_bytes().expect("relin ser"),
        })
        .unwrap();
        (client, server)
    }

    fn client_keys(&self, client_key_bytes: &[u8]) -> ClientKeys {
        bincode::deserialize(client_key_bytes).expect("CkksCompareBackend: client key bundle")
    }

    pub fn encrypt(&self, client_key_bytes: &[u8], value: f64) -> DeepCt {
        let keys = self.client_keys(client_key_bytes);
        let public = PublicKey::from_bytes(&self.ctx, &keys.public).expect("public key");
        let encryptor = Encryptor::with_public_key(&self.ctx, &public).expect("encryptor");
        let pt = self
            .encoder
            .encode_f64(&[value])
            .expect("CkksCompareBackend: encode");
        DeepCt {
            ct: encryptor.encrypt(&pt).expect("CkksCompareBackend: encrypt"),
            level: 0,
        }
    }

    pub fn decrypt(&self, client_key_bytes: &[u8], ct: &DeepCt) -> f64 {
        let keys = self.client_keys(client_key_bytes);
        let secret = SecretKey::from_bytes(&self.ctx, &keys.secret).expect("secret key");
        let decryptor = Decryptor::new(&self.ctx, &secret).expect("decryptor");
        let pt = decryptor
            .decrypt(&ct.ct)
            .expect("CkksCompareBackend: decrypt");
        self.encoder
            .decode_f64(&pt)
            .expect("CkksCompareBackend: decode")[0]
    }

    pub fn serialize_ct(&self, ct: &DeepCt) -> Vec<u8> {
        bincode::serialize(&DeepCtWire {
            level: ct.level,
            bytes: ct.ct.as_bytes().expect("ct ser"),
        })
        .unwrap()
    }

    pub fn deserialize_ct(&self, bytes: &[u8]) -> DeepCt {
        let wire: DeepCtWire = bincode::deserialize(bytes).expect("ct wire");
        DeepCt {
            ct: Ciphertext::from_bytes(&self.ctx, &wire.bytes).expect("ct deser"),
            level: wire.level,
        }
    }

    /// Drop primes (no rescale — scale is untouched) until `ct` is at
    /// `target_level`. Cheap relative to an actual multiply: no noise
    /// growth beyond what mod-switching itself costs, no scale change.
    fn mod_switch_to(&self, ct: &DeepCt, target_level: u32) -> DeepCt {
        assert!(
            ct.level <= target_level,
            "cannot mod-switch up: {} -> {}",
            ct.level,
            target_level
        );
        let mut cur = ct.ct.clone();
        for _ in ct.level..target_level {
            cur = self
                .evaluator
                .mod_switch_to_next(&cur)
                .expect("CkksCompareBackend: mod_switch ciphertext");
        }
        DeepCt {
            ct: cur,
            level: target_level,
        }
    }

    fn mod_switch_plaintext_to(
        &self,
        pt: &Plaintext,
        from_level: u32,
        target_level: u32,
    ) -> Plaintext {
        assert!(from_level <= target_level);
        let mut cur = pt.clone();
        for _ in from_level..target_level {
            cur = self
                .evaluator
                .mod_switch_to_next_plaintext(&cur)
                .expect("CkksCompareBackend: mod_switch plaintext");
        }
        cur
    }

    /// Encode a plaintext scalar (broadcast to every slot) at exactly
    /// `target`'s current scale, then mod-switch it down to `target`'s
    /// level. `add_plain`/`multiply_plain` require the plaintext and
    /// ciphertext scales to match near-exactly, and a real
    /// `rescale_to_next` perturbs the scale away from a round power of two
    /// (dividing by a prime only *close to* 2^40) — so encoding at a fixed
    /// nominal scale like 2^40 works for level-0 ciphertexts but fails
    /// (`InvalidArgument`) at any level reached via a real rescale. Building
    /// a fresh encoder at `target`'s exact scale for every op is the
    /// standard way real CKKS applications handle this — not the fixed
    /// per-depth encoder cache `CkksBackend::with_encoder` uses, because
    /// that trick only works with the "never rescale, track depth as a
    /// scale exponent" convention this backend deliberately does not use.
    fn encode_at(&self, value: f64, target: &DeepCt) -> Plaintext {
        let encoder = CKKSEncoder::new(&self.ctx, target.ct.scale())
            .expect("CkksCompareBackend: per-op encoder");
        let pt = encoder
            .encode_f64(&[value])
            .expect("CkksCompareBackend: encode");
        self.mod_switch_plaintext_to(&pt, 0, target.level)
    }

    pub fn add(&self, a: &DeepCt, b: &DeepCt) -> DeepCt {
        let level = a.level.max(b.level);
        let a = self.mod_switch_to(a, level);
        let b = self.mod_switch_to(b, level);
        DeepCt {
            ct: self.evaluator.add(&a.ct, &b.ct).expect("add"),
            level,
        }
    }

    pub fn sub(&self, a: &DeepCt, b: &DeepCt) -> DeepCt {
        let level = a.level.max(b.level);
        let a = self.mod_switch_to(a, level);
        let b = self.mod_switch_to(b, level);
        DeepCt {
            ct: self.evaluator.sub(&a.ct, &b.ct).expect("sub"),
            level,
        }
    }

    /// Ciphertext-ciphertext multiply: align levels, multiply, relinearize,
    /// then a *real* `rescale_to_next` — the whole point of this backend —
    /// bringing the scale back to ~2^40 and consuming exactly one prime,
    /// so `k` sequential multiplications cost `k` levels, not `k`
    /// exponentiated ones.
    pub fn mul(&self, a: &DeepCt, b: &DeepCt) -> DeepCt {
        let level = a.level.max(b.level);
        let a = self.mod_switch_to(a, level);
        let b = self.mod_switch_to(b, level);
        let prod = self.evaluator.multiply(&a.ct, &b.ct).expect("mul");
        let prod = self
            .evaluator
            .relinearize(&prod, &self.server().relin)
            .expect("relinearize");
        let prod = self
            .evaluator
            .rescale_to_next(&prod)
            .expect("CkksCompareBackend: rescale");
        DeepCt {
            ct: prod,
            level: level + 1,
        }
    }

    pub fn mul_plain(&self, a: &DeepCt, b: f64) -> DeepCt {
        let pt = self.encode_at(b, a);
        let prod = self
            .evaluator
            .multiply_plain(&a.ct, &pt)
            .expect("mul_plain");
        let prod = self
            .evaluator
            .rescale_to_next(&prod)
            .expect("CkksCompareBackend: rescale (plain)");
        DeepCt {
            ct: prod,
            level: a.level + 1,
        }
    }

    pub fn add_plain(&self, a: &DeepCt, b: f64) -> DeepCt {
        let pt = self.encode_at(b, a);
        DeepCt {
            ct: self.evaluator.add_plain(&a.ct, &pt).expect("add_plain"),
            level: a.level,
        }
    }

    /// Homomorphic [`sign_approx_plain`]: `iterations` Newton–Schulz steps
    /// on `x / bound`. Needs [`Self::levels_for_sign`] levels of budget.
    pub fn sign_approx(&self, x: &DeepCt, bound: f64, iterations: u32) -> DeepCt {
        let mut cur = self.mul_plain(x, 1.0 / bound);
        for _ in 0..iterations {
            // f(v) = v * (1.5 - 0.5*v^2)
            let v2 = self.mul(&cur, &cur);
            let inner = self.mul_plain(&v2, -0.5);
            let inner = self.add_plain(&inner, 1.5);
            let v_switched = self.mod_switch_to(&cur, inner.level);
            cur = self.mul(&v_switched, &inner);
        }
        cur
    }

    /// Homomorphic [`relu_approx_plain`]: `x * (sign_approx(x/bound) + 1) / 2`.
    /// Needs [`Self::levels_for_relu`] levels of budget.
    pub fn relu_approx(&self, x: &DeepCt, bound: f64, iterations: u32) -> DeepCt {
        let s = self.sign_approx(x, bound, iterations);
        let half = self.mul_plain(&s, 0.5);
        let half = self.add_plain(&half, 0.5);
        let x_switched = self.mod_switch_to(x, half.level);
        self.mul(&x_switched, &half)
    }

    /// Homomorphic [`cmpgt_approx_plain`]: `(sign_approx((a - b)/bound) + 1) / 2`.
    /// Needs [`Self::levels_for_cmpgt`] levels of budget.
    pub fn cmpgt_approx(&self, a: &DeepCt, b: &DeepCt, bound: f64, iterations: u32) -> DeepCt {
        let diff = self.sub(a, b);
        let s = self.sign_approx(&diff, bound, iterations);
        let half = self.mul_plain(&s, 0.5);
        self.add_plain(&half, 0.5)
    }
}

#[cfg(test)]
mod real_seal_tests {
    use super::*;

    #[test]
    fn levels_formulas_are_internally_consistent() {
        assert_eq!(CkksCompareBackend::levels_for_sign(3), 10);
        assert_eq!(CkksCompareBackend::levels_for_relu(3), 12);
        assert_eq!(CkksCompareBackend::levels_for_cmpgt(3), 11);
    }

    #[test]
    #[ignore] // slow: real SEAL keygen + a deep modulus chain
    fn real_seal_sign_approx_matches_plain_within_tolerance() {
        let iterations = 3;
        let levels = CkksCompareBackend::levels_for_sign(iterations);
        let (ck, sk) = CkksCompareBackend::new(levels).keygen();
        let be = CkksCompareBackend::with_server_keys(levels, &sk);
        let bound = 5.0;
        for &x in &[3.0, -3.0, 0.5, -0.5, 1.0, -1.0] {
            let ct = be.encrypt(&ck, x);
            let out = be.sign_approx(&ct, bound, iterations);
            let got = be.decrypt(&ck, &out);
            let expected = sign_approx_plain(x / bound, iterations);
            assert!(
                (got - expected).abs() < 1e-2,
                "sign_approx mismatch at x={x}: ckks={got} plain={expected}"
            );
        }
    }

    #[test]
    #[ignore] // slow: real SEAL keygen + a deep modulus chain
    fn real_seal_relu_approx_matches_plain_within_tolerance() {
        // Compare against `relu_approx_plain` at the *same* iteration count,
        // not the exact ReLU: at only 3 iterations, x=0.5 with bound=5.0
        // lands at ratio 0.1 — still close enough to the unstable fixed
        // point at 0 that 3 iterations alone genuinely don't converge to
        // ~1.0 yet (see `sign_approx_needs_more_iterations_closer_to_zero`).
        // That's the documented "slow near zero" behavior of the plain
        // approximation itself, not a CKKS-arithmetic bug — so the real-SEAL
        // path must match the plain math's *actual* output at this
        // iteration count, not the ground truth it hasn't converged to yet.
        let iterations = 3;
        let levels = CkksCompareBackend::levels_for_relu(iterations);
        let (ck, sk) = CkksCompareBackend::new(levels).keygen();
        let be = CkksCompareBackend::with_server_keys(levels, &sk);
        let bound = 5.0;
        for &x in &[3.0, -3.0, 0.5, -0.5, 0.0] {
            let ct = be.encrypt(&ck, x);
            let out = be.relu_approx(&ct, bound, iterations);
            let got = be.decrypt(&ck, &out);
            let expected = relu_approx_plain(x, bound, iterations);
            assert!(
                (got - expected).abs() < 1e-2,
                "relu_approx mismatch at x={x}: ckks={got} plain(approx)={expected}"
            );
        }
    }

    #[test]
    #[ignore] // slow: real SEAL keygen + a deep modulus chain
    fn real_seal_cmpgt_approx_matches_plain_within_tolerance() {
        let iterations = 3;
        let levels = CkksCompareBackend::levels_for_cmpgt(iterations);
        let (ck, sk) = CkksCompareBackend::new(levels).keygen();
        let be = CkksCompareBackend::with_server_keys(levels, &sk);
        let bound = 5.0;
        for &(a, b) in &[(3.0, 1.0), (1.0, 3.0), (-2.0, -4.0)] {
            let ca = be.encrypt(&ck, a);
            let cb = be.encrypt(&ck, b);
            let out = be.cmpgt_approx(&ca, &cb, bound, iterations);
            let got = be.decrypt(&ck, &out);
            let expected = if a > b { 1.0 } else { 0.0 };
            assert!(
                (got - expected).abs() < 1e-1,
                "cmpgt_approx mismatch at a={a} b={b}: ckks={got} expected={expected}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_sign_step_fixed_points() {
        assert_eq!(newton_sign_step(0.0), 0.0);
        assert_eq!(newton_sign_step(1.0), 1.0);
        assert_eq!(newton_sign_step(-1.0), -1.0);
    }

    #[test]
    fn sign_approx_converges_for_values_well_inside_domain() {
        // Comfortably inside (-sqrt(3), sqrt(3)) and not too close to 0;
        // a handful of iterations should already be extremely close to
        // the exact sign.
        for &x in &[0.9, 0.5, 0.1, -0.1, -0.5, -0.9, 1.2, -1.2] {
            let got = sign_approx_plain(x, 12);
            let expected = if x > 0.0 { 1.0 } else { -1.0 };
            assert!(
                (got - expected).abs() < 1e-6,
                "sign_approx_plain({x}, 12) = {got}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn sign_approx_accuracy_improves_monotonically_with_iterations() {
        // More iterations must not make things worse, for a fixed input
        // away from the unstable fixed point at 0.
        let x = 0.3;
        let errors: Vec<f64> = (1..=10)
            .map(|k| (sign_approx_plain(x, k) - 1.0).abs())
            .collect();
        for w in errors.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "error should not increase with more iterations: {errors:?}"
            );
        }
        assert!(*errors.last().unwrap() < 1e-6, "errors: {errors:?}");
    }

    #[test]
    fn sign_approx_needs_more_iterations_closer_to_zero() {
        // The "slow near zero" property documented above: matching accuracy
        // near the boundary costs strictly more iterations than far from it.
        fn iters_to_reach(x: f64, tol: f64) -> u32 {
            let target = if x > 0.0 { 1.0 } else { -1.0 };
            for k in 1..200 {
                if (sign_approx_plain(x, k) - target).abs() < tol {
                    return k;
                }
            }
            panic!("did not converge for x={x} within 200 iterations");
        }
        let near = iters_to_reach(0.01, 1e-3);
        let far = iters_to_reach(0.5, 1e-3);
        assert!(
            near > far,
            "expected more iterations near 0 (got {near}) than far from it (got {far})"
        );
    }

    #[test]
    fn relu_approx_matches_plain_relu_within_tolerance() {
        let bound = 10.0;
        for &x in &[-8.0, -2.0, -0.1, 0.0, 0.1, 3.0, 9.5] {
            let got = relu_approx_plain(x, bound, 16);
            let expected = x.max(0.0);
            assert!(
                (got - expected).abs() < 1e-3,
                "relu_approx_plain({x}, {bound}, 16) = {got}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn cmpgt_approx_matches_plain_comparison_within_tolerance() {
        let bound = 5.0;
        for &(a, b) in &[
            (3.0, 1.0),
            (1.0, 3.0),
            (0.0, 0.0),
            (-2.0, -4.0),
            (-4.0, -2.0),
        ] {
            let got = cmpgt_approx_plain(a, b, bound, 16);
            let expected = if a > b { 1.0 } else { 0.0 };
            assert!(
                (got - expected).abs() < 1e-3 || (a == b && (got - 0.5).abs() < 1e-3),
                "cmpgt_approx_plain({a}, {b}, {bound}, 16) = {got}, expected ~{expected}"
            );
        }
    }
}
