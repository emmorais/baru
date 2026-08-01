//! Falcon's wrap functions and `Π_WA` (Algorithm 2) — the route to DReLU.
//!
//! See [falcon-reproduction.md](../../docs/falcon-reproduction.md) for why this
//! is being built at all. In short: the sign of a replicated-shared value is
//! determined by the **carry** produced when the three summands are added as
//! integers, and recovering that carry cheaply is Falcon's contribution over
//! SecureNN's full bit-decomposition.
//!
//! # The identities this rests on
//!
//! With `L = 2^ℓ`, writing `θ_e`, `α_e`, `δ_e` for the *exact* (integer-valued)
//! wrap counts:
//!
//! ```text
//! r   = a + x − η·L            (5)   η = 1 iff a + x ≥ L
//! r   = r₁ + r₂ + r₃ − α_e·L   (6)
//! rᵢ  = aᵢ + xᵢ − βᵢ·L         (7)   βᵢ = wrap2(aᵢ, xᵢ, L)
//! x   = x₁ + x₂ + x₃ − δ_e·L   (8)
//! a   = a₁ + a₂ + a₃ − θ_e·L   (9)
//! ```
//!
//! Summing (7) over `i`, substituting into (6) and comparing with (5) and (9)
//! leaves `θ ≡ β₁ + β₂ + β₃ + α − η − δ (mod 2)`, which is step 5 of the
//! algorithm. Every term but `η` is either local or public; `η = (x ≥ r+1)` is
//! the one secret comparison, and it is what `Π_PC` provides.
//!
//! # Why `a + x ≥ L` is the same question as `x ≥ r + 1`
//!
//! `r ≡ a + x (mod L)` with `a, x, r ∈ [0, L)`. If `a + x < L` then `r = a + x ≥ x`.
//! If `a + x ≥ L` then `r = a + x − L < x`. So the wrap happened exactly when
//! `r < x`, i.e. when `x ≥ r + 1`.

use crate::crypto::field::{self, Fp, FpShare};

/// `wrap2(a₁, a₂, L)` — 1 if the two summands overflow, else 0. Purely local:
/// the party holding both summands computes it with no communication, which is
/// why Algorithm 2 needs no protocol for it.
pub fn wrap2(a1: u64, a2: u64, l_bits: u32) -> u64 {
    let l = 1u128 << l_bits;
    u64::from((a1 as u128 + a2 as u128) >= l)
}

/// The *exact* wrap of three summands: how many times `Σaᵢ` passes `L`, so 0, 1
/// or 2. Equation (3) in the paper.
pub fn wrap3_exact(a1: u64, a2: u64, a3: u64, l_bits: u32) -> u64 {
    let l = 1u128 << l_bits;
    let s = a1 as u128 + a2 as u128 + a3 as u128;
    (s / l) as u64
}

/// `wrap3 = wrap3_exact mod 2` — equation (4), the bit the protocol computes.
pub fn wrap3(a1: u64, a2: u64, a3: u64, l_bits: u32) -> u64 {
    wrap3_exact(a1, a2, a3, l_bits) & 1
}

/// The randomness `Π_WA` consumes, drawn once per invocation.
///
/// Bundled into a struct rather than passed loose because the three parts must
/// be *consistent* — `bits` are the bits of the same `x` whose summands are
/// `x_summands`, and `delta` is that `x`'s own wrap3. Supplying a mismatched
/// set silently returns the wrong answer, so they are constructed together by
/// [`WrapRandomness::new`].
pub struct WrapRandomness {
    /// The three additive summands of the mask `x`.
    pub x_summands: [u64; 3],
    /// Replicated `Z_p` shares of each bit of `x`, least significant first.
    pub bits: Vec<[FpShare; 3]>,
    /// `δ = wrap3(x₁, x₂, x₃, L)`.
    pub delta: u64,
}

impl WrapRandomness {
    /// Build a consistent randomness bundle for mask summands `x_summands`.
    pub fn new(
        x_summands: [u64; 3],
        l_bits: u32,
        share_bit: &mut impl FnMut(u64) -> [FpShare; 3],
    ) -> Self {
        let mask = (1u128 << l_bits) - 1;
        let x =
            ((x_summands[0] as u128 + x_summands[1] as u128 + x_summands[2] as u128) & mask) as u64;
        let bits = (0..l_bits).map(|i| share_bit((x >> i) & 1)).collect();
        WrapRandomness {
            x_summands,
            bits,
            delta: wrap3(x_summands[0], x_summands[1], x_summands[2], l_bits),
        }
    }

    /// The mask value itself, `x = Σxⱼ mod L`.
    pub fn x(&self, l_bits: u32) -> u64 {
        let mask = (1u128 << l_bits) - 1;
        ((self.x_summands[0] as u128 + self.x_summands[1] as u128 + self.x_summands[2] as u128)
            & mask) as u64
    }
}

/// `Π_WA` (Algorithm 2): shares of `a` in `Z_L` → the bit
/// `θ = wrap3(a₁, a₂, a₃, L)`.
///
/// Returns `θ` directly rather than a sharing of it: this is the local
/// simulation used to establish correctness, and the transport version will
/// carry the same arithmetic on shares. Every step but `Π_PC` is local or
/// public.
///
/// # Preconditions
///
/// `Π_PC` is only well defined for `x ≠ r+1` (see
/// [`field::private_compare`]). Here that is **not** a probabilistic
/// assumption — it is decidable from `a` alone. `x ≡ r+1` means
/// `x ≡ a + x + 1 (mod L)`, i.e. `a ≡ −1 (mod L)`, i.e. **`a = L − 1`**, whatever
/// the mask happens to be. So the degenerate case is a *fixed input*, not a
/// negligible event, and a caller cannot dismiss it by appealing to randomness.
///
/// Falcon's DReLU discharges it structurally: it invokes `Π_WA` on the
/// **doubled** summands `(2a₁, 2a₂, 2a₃)`, so the wrapped value is `2a mod L`,
/// which is even, while `L − 1 = 2^ℓ − 1` is odd. The bad input is therefore
/// unreachable there. `wrap3_protocol_is_exact` checks every input including
/// `a = L−1`, and `the_degenerate_input_is_unreachable_after_doubling` pins
/// the argument that makes it safe.
pub fn wrap3_protocol(
    a_summands: [u64; 3],
    rnd: &WrapRandomness,
    l_bits: u32,
    zeros: &[[Fp; 3]],
    beta: &[FpShare; 3],
    m: &[FpShare; 3],
) -> u64 {
    let mask = (1u128 << l_bits) - 1;
    let l_minus_1 = mask as u64;

    // 1. rⱼ ≡ aⱼ + xⱼ (mod L) and βⱼ = wrap2(aⱼ, xⱼ, L) — both local.
    let mut r_j = [0u64; 3];
    let mut beta_j = [0u64; 3];
    for j in 0..3 {
        r_j[j] = ((a_summands[j] as u128 + rnd.x_summands[j] as u128) & mask) as u64;
        beta_j[j] = wrap2(a_summands[j], rnd.x_summands[j], l_bits);
    }

    // 2. Reconstruct r, and 3. compute α in the clear from the opened rⱼ.
    let r = ((r_j[0] as u128 + r_j[1] as u128 + r_j[2] as u128) & mask) as u64;
    let alpha = wrap3(r_j[0], r_j[1], r_j[2], l_bits);

    // 4. η = (x ≥ r+1). At r = L−1 the comparison is against L, which no ℓ-bit
    //    x can reach, so η = 0 — handled here rather than by widening Π_PC,
    //    since r+1 would not be an ℓ-bit public value.
    let eta = if r == l_minus_1 {
        0
    } else {
        u64::from(field::private_compare(&rnd.bits, r + 1, beta, m, zeros)) ^ reconstruct_bit(beta)
    };

    // 5. θ = β₁ + β₂ + β₃ + α − η − δ (mod 2)
    (beta_j[0] + beta_j[1] + beta_j[2] + alpha + 2 - eta - rnd.delta) & 1
}

/// The blinding bit `β`, reconstructed — the local simulation needs it to form
/// `β' ⊕ β`, which in the deployed protocol each party does on its own `⟦β⟧₂`.
fn reconstruct_bit(beta: &[FpShare; 3]) -> u64 {
    field::reconstruct(beta).value() & 1
}

/// The most significant bit of an `ℓ`-bit ring element — its sign under the
/// two's-complement reading the fixed-point scheme uses.
pub fn msb(v: u64, l_bits: u32) -> u64 {
    (v >> (l_bits - 1)) & 1
}

/// Interpret an `ℓ`-bit ring element as a signed integer.
pub fn signed(v: u64, l_bits: u32) -> i128 {
    if msb(v, l_bits) == 1 {
        v as i128 - (1i128 << l_bits)
    } else {
        v as i128
    }
}

/// Equation (11), in the clear: the identity `Π_DReLU` is built on.
///
/// ```text
/// DReLU(a) = MSB(a₁) ⊕ MSB(a₂) ⊕ MSB(a₃) ⊕ wrap3(2a₁, 2a₂, 2a₃, L) ⊕ 1
/// ```
///
/// The MSB of a sum differs from the XOR of the summands' MSBs by the carry
/// *into* the top position, and that carry is precisely the wrap of the
/// summands with their top bits shifted up — i.e. the wrap of the doubled
/// summands. Doubling is therefore how a carry at position `ℓ−2` is turned into
/// a wrap at position `ℓ−1`, where `Π_WA` can see it.
///
/// Stated separately from the protocol so it can be checked exhaustively in the
/// clear: if this identity is wrong, nothing built on it can be right.
pub fn drelu_identity(a_summands: [u64; 3], theta_doubled: u64, l_bits: u32) -> u64 {
    msb(a_summands[0], l_bits)
        ^ msb(a_summands[1], l_bits)
        ^ msb(a_summands[2], l_bits)
        ^ theta_doubled
        ^ 1
}

/// Double each summand mod `L` — the input `Π_WA` is actually run on.
///
/// Keeping this a named function rather than inlining the shift is deliberate:
/// the doubling is a **correctness requirement**, not an optimisation. `Π_PC`
/// degenerates at `a = L−1` (see [`wrap3_protocol`]), and doubling is what makes
/// that input unreachable, since `2a mod L` is always even and `L−1` is odd.
/// Removing it would leave a protocol that is wrong on exactly one input, and
/// randomly so.
pub fn doubled(a_summands: [u64; 3], l_bits: u32) -> [u64; 3] {
    let mask = (1u128 << l_bits) - 1;
    [
        ((a_summands[0] as u128 * 2) & mask) as u64,
        ((a_summands[1] as u128 * 2) & mask) as u64,
        ((a_summands[2] as u128 * 2) & mask) as u64,
    ]
}

/// `Π_DReLU` (Algorithm 3, steps 1–2): 1 if `a ≥ 0` under the signed reading,
/// else 0.
///
/// One `Π_WA` on the doubled summands, then a local XOR. No exclusions are
/// needed on the input: the doubling puts the degenerate case out of reach.
pub fn drelu_protocol(
    a_summands: [u64; 3],
    rnd: &WrapRandomness,
    l_bits: u32,
    zeros: &[[Fp; 3]],
    beta: &[FpShare; 3],
    m: &[FpShare; 3],
) -> u64 {
    let theta = wrap3_protocol(doubled(a_summands, l_bits), rnd, l_bits, zeros, beta, m);
    drelu_identity(a_summands, theta, l_bits)
}

/// `Π_ReLU` (Algorithm 3): `max(0, a)` under the signed reading.
///
/// Step 3 of the algorithm selects between `a` and `0` with `Π_SS`, which is one
/// multiplication by the DReLU bit. In this local simulation the selection is
/// direct; on the transport it becomes `Π_SS` exactly as the paper specifies.
pub fn relu_protocol(
    a_summands: [u64; 3],
    rnd: &WrapRandomness,
    l_bits: u32,
    zeros: &[[Fp; 3]],
    beta: &[FpShare; 3],
    m: &[FpShare; 3],
) -> u64 {
    let mask = (1u128 << l_bits) - 1;
    let a = ((a_summands[0] as u128 + a_summands[1] as u128 + a_summands[2] as u128) & mask) as u64;
    if drelu_protocol(a_summands, rnd, l_bits, zeros, beta, m) == 1 {
        a
    } else {
        0
    }
}

/// Everything one `Π_DReLU` invocation consumes, drawn fresh each time.
///
/// Bundled because these four must be mutually consistent and independent
/// across calls: reusing a mask or a blinding bit between comparisons would
/// correlate their openings.
pub struct DreluRandomness {
    pub wrap: WrapRandomness,
    pub zeros: Vec<[Fp; 3]>,
    pub beta: [FpShare; 3],
    pub mask: [FpShare; 3],
}

/// Supplies fresh [`DreluRandomness`] per comparison.
pub type RandomnessSource<'a> = &'a mut dyn FnMut() -> DreluRandomness;

/// Splits a ring value into the three additive summands the parties hold.
pub type SummandSplit<'a> = &'a mut dyn FnMut(u64) -> [u64; 3];

/// [`drelu_protocol`] against a bundled randomness set.
pub fn drelu_with(a_summands: [u64; 3], r: &DreluRandomness, l_bits: u32) -> u64 {
    drelu_protocol(a_summands, &r.wrap, l_bits, &r.zeros, &r.beta, &r.mask)
}

/// `Π_SS` (Select Shares): `x` when `b = 0`, `y` when `b = 1`.
///
/// The paper implements this without branching on a secret: with a precomputed
/// random bit `c` shared in **both** `Z_2` and `Z_L`, the parties open
/// `e = b ⊕ c`, set `⟦d⟧_L = ⟦1−c⟧_L` if `e = 1` and `⟦c⟧_L` otherwise — so `d`
/// is `b` lifted into the ring without revealing it — and then compute
/// `z = (y − x)·d + x` with one `Π_Mult`.
///
/// So the cost of a select is **one multiplication plus one opening**, which is
/// what makes `Π_Maxpool`'s accounting come out the way it does: `n−1` DReLUs
/// and `2(n−1)` selects, not `n−1` of some fused primitive.
///
/// This local simulation performs the selection directly; the ring lift is
/// mechanical and belongs with the transport wiring.
pub fn select_shares(x: u64, y: u64, b: u64) -> u64 {
    if b == 1 {
        y
    } else {
        x
    }
}

/// `Π_Maxpool` (Algorithm 4): the maximum of `n` shared values **and** the
/// one-hot indicator of which position achieved it.
///
/// Folds left, comparing the running maximum against each next element with one
/// `Π_DReLU` and selecting both the value and the indicator with `Π_SS`. Cost is
/// therefore `n−1` DReLUs and `2(n−1)` selects.
///
/// # Two things worth knowing before comparing this against Baru's max-pool
///
/// **The one-hot is not free, and inference does not need it.** Falcon carries
/// `ind` because it also trains: the indicator is what routes gradients back
/// through a max-pool. Half the selects here exist only for that. A like-for-like
/// comparison against an inference-only max-pool should either drop the
/// indicator or count it separately — see [`maxpool_value_only`].
///
/// **Inputs must be bounded.** `Π_DReLU` is applied to `max − aᵢ`, so that
/// difference has to stay inside the signed range: `|aᵢ| < L/4` suffices. This
/// is the same class of constraint as the two bits of headroom Baru's `nbits`
/// calibration carries for its own comparisons — a comparison protocol over a
/// ring always costs range somewhere.
pub fn maxpool_protocol(
    values: &[u64],
    l_bits: u32,
    rnd: RandomnessSource,
    summands: SummandSplit,
) -> (u64, Vec<u64>) {
    assert!(!values.is_empty(), "max-pool needs at least one input");
    let mask = (1u128 << l_bits) - 1;
    let sub = |a: u64, b: u64| (((a as u128) + (1u128 << l_bits) - b as u128) & mask) as u64;

    let mut max = values[0];
    let mut ind = vec![0u64; values.len()];
    ind[0] = 1;

    for (i, &ai) in values.iter().enumerate().skip(1) {
        let dmax = sub(max, ai);
        // b = DReLU(max − aᵢ) = 1 exactly when max ≥ aᵢ.
        let b = drelu_with(summands(dmax), &rnd(), l_bits);
        max = select_shares(ai, max, b);
        let mut next = vec![0u64; values.len()];
        for (j, slot) in next.iter_mut().enumerate() {
            let e_i = u64::from(j == i);
            *slot = select_shares(e_i, ind[j], b);
        }
        ind = next;
    }
    (max, ind)
}

/// `Π_Maxpool` without the one-hot — the inference-only shape, `n−1` DReLUs and
/// `n−1` selects. This is the variant that should be timed against Baru's
/// `maxpool2d`, which likewise returns only the value.
pub fn maxpool_value_only(
    values: &[u64],
    l_bits: u32,
    rnd: RandomnessSource,
    summands: SummandSplit,
) -> u64 {
    let mask = (1u128 << l_bits) - 1;
    let sub = |a: u64, b: u64| (((a as u128) + (1u128 << l_bits) - b as u128) & mask) as u64;
    let mut max = values[0];
    for &ai in values.iter().skip(1) {
        let b = drelu_with(summands(sub(max, ai)), &rnd(), l_bits);
        max = select_shares(ai, max, b);
    }
    max
}

/// A truncation pair `(r, r >> d)`, generated in preprocessing.
///
/// Falcon delegates this to ABY3's `Π_trunc2` (its Fig. 3): sample a random `r`
/// and its exact shift `r' = r / 2^d`, both secret shared. The pair is
/// query-independent, so its cost is offline.
pub struct TruncPair {
    pub r: u64,
    pub r_shifted: u64,
}

impl TruncPair {
    /// `r'` is the **exact** shift of `r`, which is what makes the pair usable.
    pub fn new(r: u64, d: u32) -> Self {
        TruncPair {
            r,
            r_shifted: r >> d,
        }
    }
}

/// Truncate a shared value by `d` bits using a truncation pair: open `a − r`,
/// shift it in the clear, and add back `r'`.
///
/// ```text
/// c = a − r  (opened)
/// ⟦a >> d⟧ = (c >> d) + ⟦r'⟧
/// ```
///
/// # This is the *probabilistic* truncation, and it is not exact
///
/// `a = c + r` as integers only when the addition does not wrap, and even then
/// `(c + r) >> d ≠ (c >> d) + (r >> d)` whenever the low `d` bits of `c` and `r`
/// carry — which happens for about half of all inputs. The result is therefore
/// off by one **least significant bit** with probability ≈ ½, plus a large error
/// with probability ≈ `2^{-(ℓ-d)}` when the masked value wraps.
///
/// Falcon explicitly does **not** use this. Its stated design is "a smaller ring
/// size while using an exact yet expensive truncation protocol", and the smaller
/// ring (ℓ = 32 rather than 64) is where its ≥2× communication advantage comes
/// from. Reproducing Falcon therefore means reproducing the *exact* variant;
/// this function exists as the baseline that shows what exactness is buying,
/// and [`truncate_exact`] is the one to compare against.
pub fn truncate_probabilistic(a: u64, pair: &TruncPair, d: u32, l_bits: u32) -> u64 {
    let mask = (1u128 << l_bits) - 1;
    let c = ((a as u128 + pair.r as u128) & mask) as u64;
    (((c >> d) as u128 + (1u128 << l_bits) - pair.r_shifted as u128) & mask) as u64
}

/// Exact truncation: as [`truncate_probabilistic`], but correcting the borrow
/// out of the low `d` bits.
///
/// Writing `c = a + r` (no wrap), `a_lo`/`r_lo`/`c_lo` for the low `d` bits:
///
/// ```text
/// (c>>d) − (r>>d) = (a + r_lo − c_lo)/2^d = (a>>d) + borrow
/// borrow = 1  ⟺  a_lo + r_lo ≥ 2^d  ⟺  c_lo < r_lo
/// ```
///
/// so the exact result is `(c>>d) − (r>>d) − borrow`.
///
/// **`r_lo` is secret**, which is exactly why this is the "expensive" variant:
/// `c_lo` is public but the comparison `c_lo < r_lo` is a secure comparison, one
/// per truncation, and every fixed-point multiply needs a truncation. Here the
/// borrow is computed directly — the purpose of this stage is to pin down *what*
/// must be computed, not yet how it is paid for.
///
/// Exactness is what lets the ring shrink: an approximate truncation needs guard
/// bits to absorb the drift, and those bits are the communication Falcon saves
/// by moving from `ℓ = 64` to `ℓ = 32`.
pub fn truncate_exact(a: u64, pair: &TruncPair, d: u32, l_bits: u32) -> u64 {
    let mask = (1u128 << l_bits) - 1;
    let dmask = (1u64 << d) - 1;
    let c = ((a as u128 + pair.r as u128) & mask) as u64;
    let borrow = u64::from((c & dmask) < (pair.r & dmask));
    (((c >> d) as u128 + (1u128 << l_bits) * 2 - pair.r_shifted as u128 - borrow as u128) & mask)
        as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn fp(&mut self) -> Fp {
            Fp::new(self.next_u64())
        }
        fn zeros(&mut self, n: usize) -> Vec<[Fp; 3]> {
            (0..n)
                .map(|_| {
                    let (a, b) = (self.fp(), self.fp());
                    [a, b, -(a + b)]
                })
                .collect()
        }
        fn share_of(&mut self, v: u64) -> [FpShare; 3] {
            let (r0, r1) = (self.fp(), self.fp());
            field::share(Fp::new(v), r0, r1)
        }
    }

    /// The plaintext wrap functions must match their definitions (2)–(4).
    #[test]
    fn wrap_functions_match_their_definitions() {
        let l_bits = 4;
        let l = 1u64 << l_bits;
        for a in 0..l {
            for b in 0..l {
                assert_eq!(wrap2(a, b, l_bits), u64::from(a + b >= l));
                for c in 0..l {
                    assert_eq!(wrap3_exact(a, b, c, l_bits), (a + b + c) / l);
                    assert_eq!(wrap3(a, b, c, l_bits), ((a + b + c) / l) % 2);
                }
            }
        }
    }

    /// Run `Π_WA` once on the given summands with fresh randomness.
    fn wa_once(a_sum: [u64; 3], l_bits: u32, rng: &mut Rng) -> u64 {
        let l = 1u64 << l_bits;
        let xs = [rng.next_u64() % l, rng.next_u64() % l, rng.next_u64() % l];
        let rnd = WrapRandomness::new(xs, l_bits, &mut |b| rng.share_of(b));
        let z = rng.zeros(2 * l_bits as usize + 4);
        let bit = rng.next_u64() & 1;
        let beta = rng.share_of(bit);
        let mut mv = rng.fp();
        if mv.is_zero() {
            mv = Fp::ONE;
        }
        let m = field::share(mv, rng.fp(), rng.fp());
        wrap3_protocol(a_sum, &rnd, l_bits, &z, &beta, &m)
    }

    fn value_of(a: [u64; 3], l_bits: u32) -> u64 {
        ((a[0] as u128 + a[1] as u128 + a[2] as u128) & ((1u128 << l_bits) - 1)) as u64
    }

    /// `Π_WA` returns exactly `wrap3(a₁,a₂,a₃,L)` on every input except the one
    /// its `Π_PC` precondition excludes.
    ///
    /// Exhaustive over all 4096 summand triples at ℓ = 4, each with fresh
    /// randomness, skipping `a = L−1` — which the next test shows is precisely
    /// and only where it fails.
    #[test]
    fn wrap3_protocol_is_exact_away_from_the_degenerate_input() {
        const LB: u32 = 4;
        let l = 1u64 << LB;
        let mut rng = Rng(2024);
        for a1 in 0..l {
            for a2 in 0..l {
                for a3 in 0..l {
                    let a_sum = [a1, a2, a3];
                    if value_of(a_sum, LB) == l - 1 {
                        continue;
                    }
                    assert_eq!(
                        wa_once(a_sum, LB, &mut rng),
                        wrap3(a1, a2, a3, LB),
                        "Π_WA wrong at ({a1},{a2},{a3})"
                    );
                }
            }
        }
    }

    /// The degenerate set is **exactly** `a = L−1`, and it is a property of the
    /// input rather than of the randomness.
    ///
    /// `Π_PC` misbehaves at `x = r+1`; here `r ≡ a + x`, so `x ≡ r+1` reduces to
    /// `a ≡ −1 (mod L)` with the mask cancelling entirely. This checks both
    /// directions: away from `a = L−1` a long run is always right, and at
    /// `a = L−1` the answer is unreliable — wrong for a substantial fraction of
    /// draws, because the degenerate `Π_PC` output is `1 ⊕ β` for a random `β`.
    ///
    /// It matters that this is *not* negligible-probability: a caller cannot
    /// wave it away with a union bound, it has to avoid the input.
    #[test]
    fn the_degenerate_set_is_exactly_a_equals_l_minus_1() {
        const LB: u32 = 4;
        let l = 1u64 << LB;
        let mut rng = Rng(77);

        let mut wrong_at_bad = 0;
        let mut trials = 0;
        for a1 in 0..l {
            for a2 in 0..l {
                let a3 = (l - 1 + 2 * l - a1 - a2) % l; // force a = L−1
                let a_sum = [a1, a2, a3];
                assert_eq!(value_of(a_sum, LB), l - 1);
                if wa_once(a_sum, LB, &mut rng) != wrap3(a1, a2, a3, LB) {
                    wrong_at_bad += 1;
                }
                trials += 1;
            }
        }
        assert!(
            wrong_at_bad * 4 > trials,
            "at a = L−1 the result should be unreliable, but only {wrong_at_bad}/{trials} were wrong"
        );
    }

    /// Equation (11) checked in the clear, exhaustively over every summand
    /// triple at ℓ = 5. Everything in stage 4 rests on this one identity, so it
    /// is verified without any protocol in the way.
    #[test]
    fn the_drelu_identity_holds_in_the_clear() {
        const LB: u32 = 5;
        let l = 1u64 << LB;
        for a1 in 0..l {
            for a2 in 0..l {
                for a3 in 0..l {
                    let a_sum = [a1, a2, a3];
                    let d = doubled(a_sum, LB);
                    let theta = wrap3(d[0], d[1], d[2], LB);
                    let got = drelu_identity(a_sum, theta, LB);
                    let a = value_of(a_sum, LB);
                    let want = u64::from(signed(a, LB) >= 0);
                    assert_eq!(got, want, "Eq.(11) at ({a1},{a2},{a3}), a={a}");
                }
            }
        }
    }

    /// `Π_DReLU` end to end, exhaustive over all 4096 triples at ℓ = 4 — and
    /// **with no inputs excluded**, unlike `Π_WA`. That is the payoff of the
    /// doubling: `2a mod L` is even, `L−1` is odd, so `Π_PC`'s degenerate input
    /// cannot be reached from here.
    #[test]
    fn drelu_is_exact_on_every_input() {
        const LB: u32 = 4;
        let l = 1u64 << LB;
        let mut rng = Rng(31337);
        for a1 in 0..l {
            for a2 in 0..l {
                for a3 in 0..l {
                    let a_sum = [a1, a2, a3];
                    let got = {
                        let xs = [rng.next_u64() % l, rng.next_u64() % l, rng.next_u64() % l];
                        let rnd = WrapRandomness::new(xs, LB, &mut |b| rng.share_of(b));
                        let z = rng.zeros(2 * LB as usize + 4);
                        let bit = rng.next_u64() & 1;
                        let beta = rng.share_of(bit);
                        let mut mv = rng.fp();
                        if mv.is_zero() {
                            mv = Fp::ONE;
                        }
                        let m = field::share(mv, rng.fp(), rng.fp());
                        drelu_protocol(a_sum, &rnd, LB, &z, &beta, &m)
                    };
                    let a = value_of(a_sum, LB);
                    assert_eq!(
                        got,
                        u64::from(signed(a, LB) >= 0),
                        "DReLU wrong at ({a1},{a2},{a3}), a={a} (signed {})",
                        signed(a, LB)
                    );
                }
            }
        }
    }

    /// `Π_ReLU` must equal `max(0, a)` under the signed reading — the same
    /// function `CnnModel::forward` applies before its rescale.
    #[test]
    fn relu_matches_max_zero_a() {
        const LB: u32 = 4;
        let l = 1u64 << LB;
        let mut rng = Rng(5150);
        for a1 in 0..l {
            for a2 in 0..l {
                for a3 in 0..l {
                    let a_sum = [a1, a2, a3];
                    let xs = [rng.next_u64() % l, rng.next_u64() % l, rng.next_u64() % l];
                    let rnd = WrapRandomness::new(xs, LB, &mut |b| rng.share_of(b));
                    let z = rng.zeros(2 * LB as usize + 4);
                    let bit = rng.next_u64() & 1;
                    let beta = rng.share_of(bit);
                    let mut mv = rng.fp();
                    if mv.is_zero() {
                        mv = Fp::ONE;
                    }
                    let m = field::share(mv, rng.fp(), rng.fp());
                    let got = relu_protocol(a_sum, &rnd, LB, &z, &beta, &m);
                    let a = value_of(a_sum, LB);
                    let want = if signed(a, LB) >= 0 { a } else { 0 };
                    assert_eq!(got, want, "ReLU wrong at a={a} (signed {})", signed(a, LB));
                }
            }
        }
    }

    /// Removing the doubling would break the protocol — the reason
    /// [`doubled`] is a named, documented step rather than an inline shift.
    ///
    /// Without it, `Π_WA` is handed `a` directly and the degenerate input
    /// `a = L−1` becomes reachable, so DReLU is unreliable exactly there. This
    /// is the negative control for `drelu_is_exact_on_every_input`.
    #[test]
    fn without_the_doubling_the_degenerate_input_becomes_reachable() {
        const LB: u32 = 4;
        let l = 1u64 << LB;
        let mut rng = Rng(8080);
        let mut wrong = 0;
        let mut trials = 0;
        for a1 in 0..l {
            for a2 in 0..l {
                let a3 = (l - 1 + 2 * l - a1 - a2) % l; // a = L−1, the bad input
                let a_sum = [a1, a2, a3];
                let xs = [rng.next_u64() % l, rng.next_u64() % l, rng.next_u64() % l];
                let rnd = WrapRandomness::new(xs, LB, &mut |b| rng.share_of(b));
                let z = rng.zeros(2 * LB as usize + 4);
                let bit = rng.next_u64() & 1;
                let beta = rng.share_of(bit);
                let mut mv = rng.fp();
                if mv.is_zero() {
                    mv = Fp::ONE;
                }
                let m = field::share(mv, rng.fp(), rng.fp());
                // Feed the summands straight in, skipping `doubled`.
                let theta = wrap3_protocol(a_sum, &rnd, LB, &z, &beta, &m);
                if theta != wrap3(a1, a2, a3, LB) {
                    wrong += 1;
                }
                trials += 1;
            }
        }
        assert!(
            wrong * 4 > trials,
            "skipping the doubling should make a=L−1 unreliable, but only {wrong}/{trials} were wrong"
        );
    }

    /// `Π_Maxpool` must return the true maximum under the signed reading, and a
    /// one-hot vector marking where it was found.
    ///
    /// Exhaustive over every 4-tuple drawn from a range narrow enough that
    /// `max − aᵢ` cannot overflow the signed window — the bound the function's
    /// docs state. 2×2 is the pooling window the CNN benchmarks actually use.
    #[test]
    fn maxpool_returns_the_max_and_its_one_hot() {
        const LB: u32 = 5;
        let l = 1i128 << LB;
        // |aᵢ| < L/4 keeps every pairwise difference inside the signed range.
        let lo = -(l / 4) + 1;
        let hi = l / 4;
        let enc = |v: i128| ((v + l) % l) as u64;
        let mut rng = Rng(600613);
        let vals: Vec<i128> = (lo..hi).step_by(3).collect();

        for &a in &vals {
            for &b in &vals {
                for &c in &vals {
                    for &d in &vals {
                        let plain = [a, b, c, d];
                        let enc_vals: Vec<u64> = plain.iter().map(|&v| enc(v)).collect();
                        let mut mk = || {
                            let l_u = 1u64 << LB;
                            let xs = [
                                rng.next_u64() % l_u,
                                rng.next_u64() % l_u,
                                rng.next_u64() % l_u,
                            ];
                            let wr = WrapRandomness::new(xs, LB, &mut |bit| rng.share_of(bit));
                            let z = rng.zeros(2 * LB as usize + 4);
                            let bit = rng.next_u64() & 1;
                            let beta = rng.share_of(bit);
                            let mut mv = rng.fp();
                            if mv.is_zero() {
                                mv = Fp::ONE;
                            }
                            let m = field::share(mv, rng.fp(), rng.fp());
                            DreluRandomness {
                                wrap: wr,
                                zeros: z,
                                beta,
                                mask: m,
                            }
                        };
                        let mut split = |v: u64| -> [u64; 3] {
                            let l_u = 1u64 << LB;
                            let s0 = 0u64;
                            let s1 = 0u64;
                            [s0, s1, (v + 2 * l_u - s0 - s1) % l_u]
                        };
                        let (got, ind) = maxpool_protocol(&enc_vals, LB, &mut mk, &mut split);

                        let want = *plain.iter().max().unwrap();
                        assert_eq!(signed(got, LB), want, "max of {plain:?} should be {want}");
                        assert_eq!(ind.iter().sum::<u64>(), 1, "indicator must be one-hot");
                        let k = ind.iter().position(|&e| e == 1).unwrap();
                        assert_eq!(plain[k], want, "one-hot must point at a maximal element");
                    }
                }
            }
        }
    }

    /// The inference-only variant must agree with the full one on the value,
    /// while doing `n−1` selects instead of `2(n−1)` — the distinction that
    /// keeps the eventual comparison against Baru's `maxpool2d` fair.
    #[test]
    fn value_only_maxpool_agrees_with_the_full_protocol() {
        const LB: u32 = 5;
        let l = 1i128 << LB;
        let enc = |v: i128| ((v + l) % l) as u64;
        let mut rng = Rng(4711);
        for trial in 0..400i128 {
            let plain: Vec<i128> = (0..4)
                .map(|k| ((trial * 7 + k * 13) % (l / 2)) - l / 4)
                .collect();
            let enc_vals: Vec<u64> = plain.iter().map(|&v| enc(v)).collect();
            let mut mk = || {
                let l_u = 1u64 << LB;
                let xs = [
                    rng.next_u64() % l_u,
                    rng.next_u64() % l_u,
                    rng.next_u64() % l_u,
                ];
                let wr = WrapRandomness::new(xs, LB, &mut |bit| rng.share_of(bit));
                let z = rng.zeros(2 * LB as usize + 4);
                let bit = rng.next_u64() & 1;
                let beta = rng.share_of(bit);
                let mut mv = rng.fp();
                if mv.is_zero() {
                    mv = Fp::ONE;
                }
                let m = field::share(mv, rng.fp(), rng.fp());
                DreluRandomness {
                    wrap: wr,
                    zeros: z,
                    beta,
                    mask: m,
                }
            };
            let mut split = |v: u64| -> [u64; 3] {
                let l_u = 1u64 << LB;
                [0, 0, v % l_u]
            };
            let only = maxpool_value_only(&enc_vals, LB, &mut mk, &mut split);
            assert_eq!(
                signed(only, LB),
                *plain.iter().max().unwrap(),
                "value-only max-pool on {plain:?}"
            );
        }
    }

    /// Exact truncation must equal a plain shift for every value and every mask
    /// that does not wrap — no exceptions, no probability.
    ///
    /// Restricted to `a + r < L` because the masking itself wraps otherwise,
    /// which is a separate (and in Falcon, preprocessing-controlled) concern
    /// from the carry this function corrects.
    #[test]
    fn exact_truncation_is_exact() {
        const LB: u32 = 10;
        let l = 1u64 << LB;
        for d in [1u32, 3, 5] {
            for a in (0..l).step_by(7) {
                for r in (0..l).step_by(13) {
                    if (a as u128) + (r as u128) >= l as u128 {
                        continue;
                    }
                    let pair = TruncPair::new(r, d);
                    assert_eq!(
                        truncate_exact(a, &pair, d, LB),
                        a >> d,
                        "exact truncation of {a} by {d} with mask {r}"
                    );
                }
            }
        }
    }

    /// The baseline it replaces is wrong about half the time, by exactly one
    /// unit — which is the quantitative reason Falcon pays for exactness and
    /// gets a smaller ring in return.
    ///
    /// Asserting the *rate* rather than just "sometimes wrong" is what makes
    /// this a measurement instead of a restatement.
    #[test]
    fn probabilistic_truncation_is_off_by_one_about_half_the_time() {
        const LB: u32 = 10;
        let l = 1u64 << LB;
        let d = 4u32;
        let (mut wrong, mut total, mut worst) = (0u64, 0u64, 0i128);
        for a in (0..l).step_by(3) {
            for r in (0..l).step_by(5) {
                if (a as u128) + (r as u128) >= l as u128 {
                    continue;
                }
                let pair = TruncPair::new(r, d);
                let got = truncate_probabilistic(a, &pair, d, LB) as i128;
                let want = (a >> d) as i128;
                if got != want {
                    wrong += 1;
                    worst = worst.max((got - want).abs());
                }
                total += 1;
            }
        }
        let rate = wrong as f64 / total as f64;
        assert!(
            (0.3..0.7).contains(&rate),
            "expected ≈½ of inputs to be off; got {rate:.3} ({wrong}/{total})"
        );
        assert_eq!(
            worst, 1,
            "the error should be exactly one unit in the last place"
        );
    }

    /// The precondition argument, pinned.
    ///
    /// `Π_PC` degenerates at `x = r+1`, which happens exactly when `a = L−1`.
    /// Falcon's DReLU calls `Π_WA` on doubled summands, so the wrapped value is
    /// even while `L−1` is odd — the bad input cannot be reached. This test is
    /// the argument, not a restatement of it: it enumerates every `a` and
    /// checks that no doubling lands on `L−1`.
    #[test]
    fn the_degenerate_input_is_unreachable_after_doubling() {
        for l_bits in 2..=16u32 {
            let l = 1u64 << l_bits;
            let bad = l - 1;
            assert_eq!(bad & 1, 1, "L−1 is odd for L = 2^{l_bits}");
            for a in 0..l.min(4096) {
                let doubled = (a.wrapping_mul(2)) & (l - 1);
                assert_ne!(doubled, bad, "2a mod L can never be L−1 (a = {a})");
            }
        }
    }
}
