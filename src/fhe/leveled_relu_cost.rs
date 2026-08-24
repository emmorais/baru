//! What a **levelled** FHE scheme spends to approximate a comparison — the
//! surviving quantitative result of the CKKS experiment.
//!
//! # Why this module exists after CKKS was removed
//!
//! Baru's headline claim is that the MPC scheme switch *is* an interactive
//! bootstrap, so a network of any depth runs with **zero** FHE bootstraps
//! (`bench-deep-nn`, `docs/experiments.md` §III.2). That claim is only
//! interesting against a baseline: a levelled scheme has to bootstrap every few
//! layers or exhaust its modulus chain.
//!
//! Quantifying that baseline needs one number — how many levels a single
//! nonlinearity costs — and it must not be invented. It was **measured** on a
//! real SEAL CKKS context (`bench-ckks-compare`, §II.4): approximate `sign` via
//! Newton iteration on `f(v) = v(1.5 − 0.5v²)`, then `relu(x) = x(sign+1)/2`.
//!
//! The CKKS backend itself was removed on 2026-08-03 — approximate comparison
//! costs seconds per operation with real error, and it carried 3.4 MB of
//! vendored C++ plus the entire Windows cmake/MSVC build story. The *level
//! arithmetic* below is what the experiment produced, it is pure integer
//! arithmetic with no dependency, and it is what the competitor model consumes.
//! Deleting it would have left the flagship comparison ungrounded, so this is
//! the experiment's result kept without the experiment's cost.
//!
//! **What this is not.** It is not a claim about any *particular* library's
//! optimum. It counts rescale-consuming multiplications in the specific
//! approximation Baru measured; a different polynomial or a scheme with cheaper
//! bootstrapping would give a different figure. `bench-deep-nn` reports the
//! per-bootstrap level budget it assumes alongside the result for that reason.

/// Rescale-consuming multiplications in `sign_approx` for `iterations` Newton
/// steps, including the initial `x / bound` scaling multiply.
///
/// Each step is `v · (1.5 − 0.5v²)`: one squaring, one scalar fold, one
/// multiply — three levels.
pub fn levels_for_sign(iterations: u32) -> u32 {
    1 + 3 * iterations
}

/// Levels for `relu_approx`: [`levels_for_sign`] plus 2 — one multiply to fold
/// in the `0.5` scalar, one to multiply by `x`.
pub fn levels_for_relu(iterations: u32) -> u32 {
    levels_for_sign(iterations) + 2
}

/// Levels for `cmpgt_approx`: [`levels_for_sign`] plus 1, folding `(sign+1)/2`
/// into a single multiply-then-add.
pub fn levels_for_cmpgt(iterations: u32) -> u32 {
    levels_for_sign(iterations) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The figures the competitor model in `bench-deep-nn` is built on. Pinned
    /// because they are the only surviving record of a measurement whose
    /// apparatus has been deleted — if these drift, the flagship comparison
    /// silently changes meaning.
    #[test]
    fn the_measured_level_costs_are_pinned() {
        assert_eq!(levels_for_sign(0), 1, "just the bound scaling");
        assert_eq!(levels_for_sign(2), 7);
        assert_eq!(levels_for_relu(2), 9, "the default in bench-deep-nn");
        assert_eq!(levels_for_cmpgt(2), 8);
    }

    /// Monotone in iterations, and relu always costs more than the sign it is
    /// built from. Cheap invariants, but they catch an inverted edit.
    #[test]
    fn costs_grow_with_iterations_and_relu_exceeds_sign() {
        for i in 0..6 {
            assert!(levels_for_sign(i) < levels_for_sign(i + 1));
            assert!(levels_for_relu(i) > levels_for_sign(i));
            assert!(levels_for_cmpgt(i) > levels_for_sign(i));
            assert!(levels_for_relu(i) > levels_for_cmpgt(i));
        }
    }
}
