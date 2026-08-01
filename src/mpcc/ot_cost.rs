//! Offline cost of OT-based preprocessing, derived from **measured** OT cost.
//!
//! This closes the loop the research agenda called the reason to build OT first:
//! it pays twice, once for PSI and once here. Semi2k, SPDZ2k and MASCOT all get
//! their Beaver triples from oblivious transfer, and until now their offline
//! volume in [`super::protocols::REGISTRY`] was a literature estimate — flagged
//! as such by `estimate`'s own output, and by `docs/mpc-compiler-roadmap.md`,
//! which lists calibrating it as P1.
//!
//! # What is measured and what is not
//!
//! Being precise about this matters more than the number, because the whole
//! point of the cost model is that its inputs are traceable.
//!
//! * **Measured** (`baru bench-ot`, [`crate::bench::ot`]): the bytes one OT
//!   costs. [`BYTES_PER_OT`] is the large-batch figure, where the `κ` base OTs
//!   have amortized away and the marginal cost is the protocol's own `κ` bits
//!   per OT.
//! * **Analytic, from the protocol**: how many OTs a triple needs. Gilboa's OLE
//!   computes `x·y` over `Z_2^k` from `k` correlated OTs of `k`-bit strings, and
//!   a Beaver triple needs the two cross terms, so `2k` OTs. This is a count of
//!   protocol steps, not a guess.
//! * **Still an estimate**: local CPU cost of preprocessing, and LowGear's
//!   BFV-based route, which is compute-bound rather than bandwidth-bound and so
//!   is not an OT count at all.
//!
//! So the offline *volume* for the OT-based protocols is now measurement ×
//! protocol structure, and `estimate` says so in its notes instead of
//! disclaiming it. The offline *time* is still volume ÷ bandwidth, which ignores
//! that OT generation is also CPU work — an honest remaining gap, and the reason
//! [`US_PER_OT`] is recorded here even though the cost model does not yet use it.
//!
//! # This is the naive route, so treat it as an upper bound
//!
//! Two ways a real implementation does better than the count here, both worth
//! naming so the figure is not mistaken for a floor:
//!
//! * **Vector OLE.** Modern preprocessing generates many products at once with
//!   sublinear communication per product, rather than `k` OTs per product.
//! * **Batched sacrifice.** MASCOT and SPDZ2k do not literally burn one triple
//!   per triple; the check amortizes over a bucket. `sacrifice: 2.0` is the
//!   pessimistic 1:1 reading.
//!
//! What replacing the literature constants with this derivation did change is a
//! *ranking*, and in the direction the literature would predict: LowGear's
//! BFV-based preprocessing moves ~384 B per triple against 9–49 KiB for the
//! OT-based protocols on the same program, a gap the previous hand-entered
//! constants understated by an order of magnitude. That is LowGear's entire
//! selling point — it trades bandwidth for CPU — and the model now shows it.
//! The CPU half of that trade is still missing, so LowGear's *total* remains
//! flattering; the note `estimate` emits for it says exactly that.

/// Bytes per OT at large batch, measured by `baru bench-ot` on 2026-07-28
/// (M-series, release, `--sizes 8192`).
///
/// The marginal cost is `κ/8 = 16` B — one column bit per output OT — and the
/// measured 16.5 is that floor plus the amortized base phase.
pub const BYTES_PER_OT: f64 = 16.5;

/// Microseconds per OT at the same batch size. Recorded for the CPU term the
/// offline model does *not* yet include.
pub const US_PER_OT: f64 = 4.67;

/// Batch size the figures above were measured at.
pub const MEASURED_AT_BATCH: usize = 8192;

/// How many correlated OTs Gilboa's OLE needs per multiplied bit, per cross
/// term. A Beaver triple needs both cross terms, hence the 2 in [`OtTriples::triple_ots`].
pub const OTS_PER_BIT_PER_CROSS_TERM: f64 = 1.0;

/// OT-based triple generation for one protocol.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct OtTriples {
    /// Triples produced per triple *used*: 1 for a passive protocol, 2 where a
    /// second triple is sacrificed to check the first (MASCOT, SPDZ2k).
    pub sacrifice: f64,
    /// Extra bits each share carries beyond the ring width — SPDZ2k's `s`
    /// statistical-security bits, which widen every OT payload.
    pub extra_share_bits: u32,
}

use serde::Serialize;

impl OtTriples {
    /// Passive, no sacrifice, no share widening.
    pub const PASSIVE: OtTriples = OtTriples {
        sacrifice: 1.0,
        extra_share_bits: 0,
    };

    /// Correlated OTs consumed per triple actually used.
    pub fn triple_ots(&self, bit_width: u32) -> f64 {
        2.0 * OTS_PER_BIT_PER_CROSS_TERM * f64::from(bit_width) * self.sacrifice
    }

    /// Bytes one of those OTs moves: the measured cost of the OT itself, plus
    /// the payload correction a *correlated* OT sends on top of the extension's
    /// column (the string being transferred is `k + extra` bits wide).
    pub fn bytes_per_ot(&self, bit_width: u32) -> f64 {
        BYTES_PER_OT + f64::from(bit_width + self.extra_share_bits) / 8.0
    }

    /// Offline bytes per triple.
    pub fn triple_bytes(&self, bit_width: u32) -> f64 {
        self.triple_ots(bit_width) * self.bytes_per_ot(bit_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derivation, spelled out at k = 64 so a change to either the measured
    /// constant or the protocol count is visible as a test diff rather than as a
    /// silently different ranking.
    #[test]
    fn the_derivation_is_what_the_docs_say() {
        let p = OtTriples::PASSIVE;
        // 2 cross terms × 64 bits = 128 correlated OTs per triple.
        assert_eq!(p.triple_ots(64), 128.0);
        // Each moves the measured 16.5 B plus a 64-bit (8 B) payload.
        assert!((p.bytes_per_ot(64) - 24.5).abs() < 1e-9);
        assert!((p.triple_bytes(64) - 3136.0).abs() < 1e-6);
    }

    /// Sacrificing doubles the OTs; widening the share widens every payload.
    #[test]
    fn sacrifice_and_widening_both_raise_the_cost() {
        let passive = OtTriples::PASSIVE;
        let active = OtTriples {
            sacrifice: 2.0,
            extra_share_bits: 64,
        };
        assert_eq!(active.triple_ots(64), 2.0 * passive.triple_ots(64));
        assert!(active.bytes_per_ot(64) > passive.bytes_per_ot(64));
        assert!(active.triple_bytes(64) > 2.0 * passive.triple_bytes(64));
    }

    /// Cost must grow with the ring width — quadratically, since both the OT
    /// count and each payload scale with `k`. That is the shape that makes
    /// 64-bit rings expensive to preprocess and is worth pinning.
    #[test]
    fn cost_is_quadratic_in_the_ring_width() {
        let p = OtTriples::PASSIVE;
        let (a, b) = (p.triple_bytes(32), p.triple_bytes(64));
        let ratio = b / a;
        assert!(
            (2.0..=4.0).contains(&ratio),
            "doubling k should cost between 2x and 4x, got {ratio:.2}x"
        );
    }
}
