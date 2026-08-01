//! The FHE **input boundary**: what a proof of well-formedness has to achieve.
//!
//! This is the last thing between the hybrid pipeline and a malicious
//! attestation. [`crate::fhe::threshold`] makes every *evaluation* step
//! checkable by replication, but replication provably cannot cover a
//! ciphertext arriving from outside: both honest co-holders compute on a
//! malformed input faithfully and **agree**. So each externally supplied
//! ciphertext must carry a zero-knowledge proof that it is a well-formed
//! encryption with bounded noise (`docs/malicious-hybrid.md` §4.4).
//!
//! That document calls the proof "the top open item" and says the noise bound
//! "needs to be made rigorous". This module makes it rigorous — not by
//! implementing the proof, but by computing **the bound the proof must hit**,
//! which is the thing that decides whether any construction is viable.
//!
//! # The relation a proof must establish
//!
//! For public key `(a, b)` and ciphertext `(c₀, c₁)`, knowledge of
//! `(m, u, e₁, e₂)` with
//!
//! ```text
//! c₀ = u·b + e₁ + Δ·m   (mod q)
//! c₁ = u·a + e₂         (mod q)
//! ‖u‖∞, ‖e₁‖∞, ‖e₂‖∞ ≤ B
//! ```
//!
//! Honest encryption uses ternary randomness, so `B = 1`.
//!
//! # The finding: soundness slack is paid for out of the smudging budget
//!
//! Every practical lattice Σ-protocol is **relaxed**: the extractor recovers a
//! witness whose norm is larger than the honest one by a slack factor, coming
//! from the challenge difference and from rejection sampling. So a verifier
//! that accepts the proof is accepting inputs with noise up to `2^slack · B`,
//! not `B`.
//!
//! That slack does not stay at the boundary. Ciphertext noise grows through the
//! depth-1 multiply, and threshold decryption then has to hide the result under
//! smudging noise that still fits the correctness budget. The consequence,
//! computed by [`achievable_lambda`]:
//!
//! > **Each bit of soundness slack in the boundary proof costs one bit of
//! > statistical security in the threshold decryption.**
//!
//! §6a of the design note measured ~40 bits of headroom at `n = 4096` and
//! concluded the parameters were fine. That analysis assumed **honest inputs**.
//! Once the boundary proof is included, a Σ-protocol with a typical slack of
//! 2^10–2^20 leaves 20–30 bits — below a conservative λ. The honest reading is
//! that `n = 4096` is enough for λ = 40 *only with an unusually tight proof*,
//! and that `n = 8192` is what makes the combination comfortable.
//!
//! This is a parameter conclusion the earlier analysis could not have reached,
//! because it treated the two components separately.

use super::bfv::BfvParams;

/// The norm bound a boundary proof actually establishes, expressed as the
/// `log2` slack over the honest witness. Produced by
/// [`super::boundary_proof::proven_bound`] and consumed by
/// [`achievable_lambda`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessBound {
    pub slack_bits: u32,
}

/// Parties contributing a smudging term during threshold decryption. Each adds
/// its own, so their sum is what the budget must accommodate.
pub const SMUDGING_PARTIES: u32 = 3;

/// `log2(q / 2t)` — the magnitude a decryption's total noise must stay under
/// for the rounding to recover the plaintext.
pub fn correctness_ceiling_bits(p: &BfvParams) -> u32 {
    p.log2_q.saturating_sub(p.log2_t).saturating_sub(1)
}

/// Statistical security achievable for smudging, given the noise that has to be
/// hidden.
///
/// Hiding a magnitude `2^noise` at security `λ` needs flooding noise of
/// `2^(noise+λ)`; correctness needs the flooded total to stay under the ceiling.
/// Three parties each contribute one term, costing `log2(3) ≈ 1.6` bits, which
/// is rounded up to 2 rather than dropped.
pub fn smudging_lambda(p: &BfvParams, log2_noise: u32) -> u32 {
    let ceiling = correctness_ceiling_bits(p);
    let parties_overhead = 2; // ceil(log2(3))
    ceiling
        .saturating_sub(log2_noise)
        .saturating_sub(parties_overhead)
}

/// Statistical security still achievable **once the boundary proof's soundness
/// slack is charged**.
///
/// `slack_bits` is `log2` of the factor by which the proof's extracted witness
/// norm may exceed the honest one. An input whose noise is `2^slack` times
/// larger produces a product whose noise is larger by the same factor — the
/// multiply is linear in each operand's noise — so the slack passes straight
/// through to what smudging must hide.
///
/// Returns 0 when the slack consumes the entire budget, which is the honest
/// answer rather than a small positive number: at that point threshold
/// decryption is not statistically hiding at all.
pub fn achievable_lambda(p: &BfvParams, log2_honest_mult_noise: u32, slack_bits: u32) -> u32 {
    smudging_lambda(p, log2_honest_mult_noise.saturating_add(slack_bits))
}

/// The largest soundness slack a boundary proof may have while still leaving
/// `target_lambda` bits of statistical security.
///
/// This is the number to hand to whoever designs the proof: it is the design
/// constraint, expressed in the only unit that matters.
pub fn max_tolerable_slack_bits(
    p: &BfvParams,
    log2_honest_mult_noise: u32,
    target_lambda: u32,
) -> u32 {
    smudging_lambda(p, log2_honest_mult_noise).saturating_sub(target_lambda)
}

/// Largest `log2 q` this BFV implementation can represent.
///
/// The modulus is a power of two held in a `u128` (`BfvParams::q_mask` is
/// `(1 << log2_q) - 1`), so anything above 127 overflows. **This, not the ring
/// dimension, is what binds the parameter search.** An earlier version of
/// [`smallest_ring_for`] proposed `n = 8192, q = 2^190` — cryptographically
/// sensible, and impossible to instantiate here. Reaching further needs a
/// multi-precision or RNS modulus, which is an implementation change rather
/// than a parameter choice.
pub const MAX_REPRESENTABLE_LOG2_Q: u32 = 127;

/// Smallest representable parameter set that leaves `target_lambda` bits given
/// a proof with `slack_bits` of slack, or `None` when none exists.
///
/// Candidates are capped at [`MAX_REPRESENTABLE_LOG2_Q`]. Returning `None` is a
/// real answer — it means the target is unreachable without changing the
/// modulus representation — and callers must not read it as "use the largest
/// entry".
pub fn smallest_ring_for(
    log2_t: u32,
    log2_honest_mult_noise: u32,
    slack_bits: u32,
    target_lambda: u32,
) -> Option<BfvParams> {
    for (n, log2_q) in [(4096usize, 100u32), (8192, 120), (16384, 127)] {
        debug_assert!(log2_q <= MAX_REPRESENTABLE_LOG2_Q);
        let p = BfvParams { n, log2_q, log2_t };
        if achievable_lambda(&p, log2_honest_mult_noise, slack_bits) >= target_lambda {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §6a of `docs/malicious-hybrid.md` measured `|e_mult| ≈ 2^38.6` at
    /// n=4096, q=2^100, t=2^20 and reported ~40 bits of headroom. Reproduce
    /// that from the model, so the two cannot drift apart unnoticed.
    #[test]
    fn the_model_reproduces_the_measured_headroom() {
        let p = BfvParams { n: 4096, log2_q: 100, log2_t: 20 };
        assert_eq!(correctness_ceiling_bits(&p), 79);
        // Measured 2^38.6, rounded up to 39.
        let lambda = smudging_lambda(&p, 39);
        assert!(
            (38..=40).contains(&lambda),
            "model says {lambda} bits, §6a measured ~40"
        );

        // t = 2^24 is about 4 bits worse on the ceiling and ~4 on the noise.
        let p24 = BfvParams { n: 4096, log2_q: 100, log2_t: 24 };
        assert_eq!(correctness_ceiling_bits(&p24), 75);
    }

    /// **The finding.** Boundary-proof slack is subtracted from the smudging
    /// budget bit for bit — the two analyses cannot be done independently.
    #[test]
    fn soundness_slack_costs_statistical_security_one_for_one() {
        let p = BfvParams { n: 4096, log2_q: 100, log2_t: 20 };
        let honest = smudging_lambda(&p, 39);
        for slack in [0u32, 1, 5, 10, 20] {
            let got = achievable_lambda(&p, 39, slack);
            assert_eq!(
                got,
                honest.saturating_sub(slack),
                "slack of {slack} bits must cost exactly {slack} bits of lambda"
            );
        }
    }

    /// The practical consequence: a Σ-protocol with realistic slack does **not**
    /// leave a conservative λ at n=4096, which §6a's honest-input analysis could
    /// not have shown.
    #[test]
    fn a_realistic_proof_does_not_fit_at_n_4096() {
        let p = BfvParams { n: 4096, log2_q: 100, log2_t: 20 };
        // Typical relaxed-Σ slack from challenge difference + rejection
        // sampling is well above 2^10.
        assert!(
            achievable_lambda(&p, 39, 10) < 40,
            "a 2^10-slack proof must not still claim 40 bits at n=4096"
        );
        assert!(
            achievable_lambda(&p, 39, 20) < 20,
            "a 2^20-slack proof leaves very little at n=4096"
        );
    }

    /// A larger ring helps, but only up to the modulus the implementation can
    /// represent — and that ceiling is what actually binds.
    #[test]
    fn a_larger_ring_helps_but_the_modulus_ceiling_binds() {
        // A modest slack is absorbed by moving to a bigger representable ring.
        let bigger = smallest_ring_for(20, 39, 10, 40).expect("10 bits of slack fits");
        assert!(bigger.n > 4096, "got n={}", bigger.n);
        assert!(bigger.log2_q <= MAX_REPRESENTABLE_LOG2_Q);

        // The *real* proof's slack does not fit at lambda=40 anywhere
        // representable. `None` is the honest answer, and the measured run
        // (`bfv::tests::measures_post_multiply_noise_at_n8192`) confirms it:
        // lambda tops out around 30.
        assert_eq!(
            smallest_ring_for(20, 41, 26, 40),
            None,
            "a 26-bit-slack proof cannot reach lambda=40 with a u128 modulus"
        );
        // It does reach a still-useful lambda=30.
        let p = smallest_ring_for(20, 41, 26, 30).expect("lambda=30 is reachable");
        assert!(achievable_lambda(&p, 41, 26) >= 30);
    }

    /// The design constraint, stated in the unit a proof designer needs.
    #[test]
    fn the_tolerable_slack_is_a_usable_number() {
        let p = BfvParams { n: 4096, log2_q: 100, log2_t: 20 };
        let budget = max_tolerable_slack_bits(&p, 39, 30);
        assert!(
            (5..=10).contains(&budget),
            "at n=4096 targeting lambda=30, a proof may have ~{budget} bits of slack"
        );
        // Targeting more than the honest headroom leaves nothing at all.
        assert_eq!(max_tolerable_slack_bits(&p, 39, 40), 0);
    }

    /// Zero must mean zero: a slack that consumes the budget is reported as no
    /// security, not as a small comfortable number.
    #[test]
    fn an_exhausted_budget_reports_zero() {
        let p = BfvParams { n: 4096, log2_q: 100, log2_t: 20 };
        assert_eq!(achievable_lambda(&p, 39, 100), 0);
    }
}
