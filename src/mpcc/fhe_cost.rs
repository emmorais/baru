//! FHE as an **assignment target**, so the cost model spans MPC *and* FHE.
//!
//! This is the piece that makes Baru's comparison something HyCC, Silph and SING
//! structurally cannot do (`docs/related-work.md` §1): those systems assign
//! operations among *MPC* sharings. Baru assigns among MPC sharings **and** FHE,
//! so it can answer the question the research agenda says nobody else can ask —
//! *should this part be an FHE computation at all?*
//!
//! # The three structural facts that decide every assignment
//!
//! 1. **FHE evaluation costs no rounds.** It is non-interactive, so a WAN's
//!    latency term vanishes. This is FHE's entire advantage and it is why the
//!    answer changes with the network profile.
//! 2. **FHE cannot compare.** BFV and CKKS are arithmetic-only — no comparison,
//!    no mux, no argmax. TFHE can, at roughly a third of a second *per gate*,
//!    which for a 32-bit comparison is ~11 s against MPC's ~10 ms. That is the
//!    three-to-four orders of magnitude `src/hybrid/` already documents.
//! 3. **FHE has a depth budget.** At the no-relinearization depth-1 parameters
//!    (`src/fhe/bfv.rs`), a value may pass through **one** ciphertext×ciphertext
//!    multiplication before it must be refreshed. Exceeding that is not
//!    expensive, it is *infeasible*.
//!
//! Put those together and the optimal assignment for a neural network is forced:
//! linear layers on FHE (no rounds), nonlinearities on MPC (FHE cannot do them),
//! and the switch between them refreshes the ciphertext — so depth never
//! accumulates. **That is the switch-as-bootstrap thesis of
//! `docs/paper-outline.md`, derived from the cost model rather than asserted.**
//!
//! # Provenance of the numbers
//!
//! Same discipline as `docs/gadget-costs.md`: each figure says where it came
//! from, because a cost model whose inputs cannot be traced is not a measurement.
//!
//! * **measured** — `baru bench-fhe-innerprod` and `baru bench`, this machine,
//!   release build.
//! * **structural** — a property of the construction, not a timing (depth budget,
//!   whether comparison exists at all).
//! * **assumed** — stated where a term was not measured, and why it is safe to
//!   treat it as negligible.
//!
//! One comparability caveat, and it is the benchmark's own: Baru's BFV uses a
//! schoolbook Rust NTT while its CKKS is SEAL's optimized C++. `bench-fhe-innerprod`
//! says outright that the two are *not timing-comparable*, so **CKKS is excluded
//! from assignment** rather than entered with a number that would flatter it. The
//! consequence is that the FHE side here is BFV's speed, not the best achievable
//! — the ranking is conservative *against* FHE.

use serde::Serialize;

/// Whether a scheme can evaluate a comparison at all, and at what price.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Comparison {
    /// Arithmetic-only: no comparison, no mux, no argmax. BFV, CKKS.
    Unsupported,
    /// Bootstrapped boolean gates — possible, and catastrophically slow.
    PerGateNs(f64),
}

/// An FHE scheme as a target the assignment can place operations on.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct FheSpec {
    pub id: &'static str,
    /// Ciphertext×ciphertext multiplications a value may pass through before it
    /// must be refreshed. **Structural**, from the construction's parameters.
    pub depth_budget: u32,
    /// Local time for one ct×ct multiply. **Measured.**
    pub mul_ns: f64,
    /// Local time for a ciphertext addition. **Assumed** negligible: an `O(n)`
    /// coefficient-wise add against the `O(n log n)` NTT product above, so it
    /// cannot change an assignment decision at this granularity.
    pub add_ns: f64,
    /// Local time for ct×plaintext. **Assumed** to be a small fraction of a
    /// ct×ct product; it skips one transform.
    pub mul_plain_ns: f64,
    /// Serialized ciphertext size — the unit a scheme switch moves. **Measured.**
    pub ct_bytes: f64,
    pub comparison: Comparison,
    /// Width of the **plaintext modulus** these parameters provide, in bits.
    ///
    /// The constraint this exists for is easy to miss and fails *silently*.
    /// Switching a **secret-shared** value into FHE does not hand the engine
    /// the value: it hands it the un-reduced sum of the parties' share
    /// components, i.e. `a + k·2^bits` for `k < parties`. A ciphertext ×
    /// ciphertext product of two such values is congruent to `a·b` modulo
    /// `2^bits` — which is why the construction is correct — but as an integer
    /// it is roughly `(parties·2^bits)^2`, and a layer sums many of them. If
    /// the plaintext modulus cannot hold that, the result wraps and the
    /// answer is wrong rather than slow.
    ///
    /// Measured envelope for `BFV_DEPTH1` (n=4096, q=2^100): 10-bit shares
    /// work at a 256-wide layer; 12-bit shares already fail at 64-wide,
    /// because past `t ≈ 2^32` the depth-1 noise budget is exhausted and
    /// `fhe::ntt` caps the ring at `MAX_N = 4096`, so there is nowhere to
    /// grow. See `baru bench-mixed-layer --deployed --share-bits W`.
    ///
    /// **This does not bite when the weights are plaintext.** A
    /// ciphertext × plaintext product does not square a share magnitude, which
    /// is why Gazelle-style hybrids — plaintext weights, only activations
    /// crossing — are unaffected. It is encrypting *both* operands that is
    /// expensive here.
    pub log2_plaintext: u32,
    /// Plaintext slots one ciphertext holds, i.e. how many independent scalar
    /// operations pack into a single homomorphic operation.
    ///
    /// This is the term without which FHE can never win, and leaving it out was a
    /// real modelling error: a first version of this model charged FHE per scalar
    /// operation and concluded "never use FHE", which contradicts this repo's own
    /// `bench-cnn` measurement of FHE at ~5% of a hybrid CNN's runtime. Baru's
    /// `src/fhe/conv_pack.rs` packs a whole convolution channel into **one**
    /// ct×ct multiply, and the CKKS batched form puts a whole 784-pixel image in
    /// one ciphertext — so per-lane charging is simply the wrong unit.
    pub slots: f64,
    /// One-line provenance, printed with any table this appears in.
    pub note: &'static str,
}

/// BFV at the secure depth-1 parameters the hybrid pipeline uses.
///
/// `multiply 20.7ms`, `ct-bytes 131072` measured by `baru bench-fhe-innerprod`
/// (n=4096, log q=100, at or below the ~109-bit RLWE ceiling for 128-bit
/// security). Depth 1 is the no-relinearization construction: multiply once, then
/// decrypt the degree-2 result under `s²`.
pub const BFV_DEPTH1: FheSpec = FheSpec {
    id: "bfv",
    depth_budget: 1,
    mul_ns: 20_700_000.0,
    add_ns: 100_000.0,
    mul_plain_ns: 5_000_000.0,
    ct_bytes: 131_072.0,
    comparison: Comparison::Unsupported,
    // Measured ceiling, not a nominal parameter: past t ≈ 2^32 the depth-1
    // noise budget at q=2^100 is gone and the product decrypts wrongly.
    log2_plaintext: 32,
    // n = 4096 coefficient slots.  `conv_pack` needs 2·in_c·H·W ≲ n, which is
    // what makes a 28×28 image fit with room to spare.
    slots: 4096.0,
    note: "measured bench-fhe-innerprod; arithmetic-only, depth 1 (no relin)",
};

/// TFHE — the scheme that *can* compare, priced so the model shows why you would
/// not want it to.
///
/// `baru bench` measures 0.2–0.5 s per gate; 0.35 s is the midpoint, and the
/// conclusion (comparisons belong on MPC) is insensitive to which end is used.
pub const TFHE: FheSpec = FheSpec {
    id: "tfhe",
    // Programmable bootstrapping every gate, so depth is unbounded — that is
    // exactly what makes each gate cost a third of a second.
    depth_budget: u32::MAX,
    mul_ns: 350_000_000.0,
    add_ns: 350_000_000.0,
    mul_plain_ns: 350_000_000.0,
    ct_bytes: 515_000.0,
    comparison: Comparison::PerGateNs(350_000_000.0),
    // Boolean TFHE carries one bit per ciphertext; the plaintext-modulus
    // constraint above is an arithmetic-scheme concern and does not apply.
    log2_plaintext: u32::MAX,
    // Boolean TFHE evaluates one gate per ciphertext — no packing.  That is a
    // second reason it loses to MPC on nonlinearities, independent of the
    // per-gate cost.
    slots: 1.0,
    note: "measured baru bench (0.2-0.5 s/gate, midpoint); bootstraps every gate",
};

/// The schemes assignment may place operations on.
///
/// CKKS is deliberately absent — see the module docs: its SEAL implementation is
/// not timing-comparable with Baru's schoolbook-NTT BFV, and entering it anyway
/// would put an implementation-quality difference into a protocol comparison.
pub const REGISTRY: &[FheSpec] = &[BFV_DEPTH1, TFHE];

pub fn find(id: &str) -> Option<&'static FheSpec> {
    REGISTRY.iter().find(|s| s.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two facts every assignment decision turns on.
    #[test]
    fn the_registry_encodes_the_structural_limits() {
        let bfv = find("bfv").unwrap();
        assert_eq!(
            bfv.comparison,
            Comparison::Unsupported,
            "BFV is arithmetic-only"
        );
        assert_eq!(bfv.depth_budget, 1, "no-relin construction is depth 1");

        let tfhe = find("tfhe").unwrap();
        assert!(matches!(tfhe.comparison, Comparison::PerGateNs(_)));
        // Per *operation* TFHE is only ~17x a BFV product; the gap that matters
        // is per *lane*, because TFHE cannot pack and BFV packs 4096 slots.
        let per_lane = |s: &FheSpec| s.mul_ns / s.slots;
        assert!(
            per_lane(tfhe) > 1000.0 * per_lane(bfv),
            "per lane TFHE should be orders off BFV ({:.0} vs {:.3} ns/lane) — that \
             gap, not the per-op one, is why nonlinearities leave FHE",
            per_lane(tfhe),
            per_lane(bfv)
        );
    }

    /// A 32-bit comparison on TFHE against MPC's ~10 ms, which is the ratio
    /// `src/hybrid/` cites as three to four orders of magnitude.
    #[test]
    fn tfhe_comparison_is_orders_of_magnitude_off_mpc() {
        let Comparison::PerGateNs(ns) = find("tfhe").unwrap().comparison else {
            panic!("TFHE must support comparison");
        };
        let cmp_secs = ns * 32.0 / 1e9;
        assert!(
            cmp_secs > 100.0 * 0.010,
            "a 32-bit TFHE comparison is {cmp_secs:.1} s, which should be >100x MPC's 10 ms"
        );
    }

    /// Packing is what makes FHE viable, so the registry must say that boolean
    /// TFHE has none and an arithmetic scheme has a lot.
    #[test]
    fn packing_distinguishes_the_schemes() {
        assert_eq!(find("tfhe").unwrap().slots, 1.0, "boolean TFHE cannot pack");
        assert!(
            find("bfv").unwrap().slots >= 1024.0,
            "an arithmetic scheme packs a whole layer into one ciphertext"
        );
    }

    #[test]
    fn ckks_is_excluded_on_purpose() {
        assert!(
            find("ckks").is_none(),
            "CKKS must stay out of assignment while its implementation is not \
             timing-comparable with BFV's — see the module docs"
        );
    }
}
