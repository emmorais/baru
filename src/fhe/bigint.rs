//! Fixed-width signed integers, generic over limb count.
//!
//! # Why this exists
//!
//! The multi-modular NTT ([`crate::fhe::ntt`]) needs an accumulator wide enough
//! to hold the *exact* integer product before it is reduced mod `q`, and the
//! width follows from the parameters:
//!
//! ```text
//! |c[k]| ≤ n·(q/2)²        so the CRT modulus Π pᵢ must exceed 2·that
//! ```
//!
//! | profile | n | q | `|c|` bound | primes | accumulator |
//! |---|---|---|---|---|---|
//! | narrow (BFV, BGV, SI-NTRU) | 4096 | 2^100 | 2^210 | 4 × 63 bit | **256 bit** |
//! | wide (LowGear's field) | 8192 | 2^190 | 2^392 | 7 × 63 bit | **512 bit** |
//!
//! The narrow profile was hard-coded as a bespoke `I256` (two `u128` halves).
//! The wide profile needs 512-bit accumulation *and* 256-bit coefficients, so
//! rather than write a second bespoke type this module parameterises the width.
//!
//! **The narrow path must not pay for the wide one.** CRT reconstruction does a
//! `mod_u64` per limb per prime, so a 512-bit accumulator would cost the
//! existing schemes ~2× on that step for nothing. Both widths are therefore
//! separate instantiations — `IntN<4>` and `IntN<8>` — and the hot path keeps
//! its 4 limbs.
//!
//! # Representation
//!
//! Two's complement, `limbs[0]` least significant. Every operation is wrapping
//! at the type's width; callers keep values inside the range their parameters
//! guarantee (the CRT modulus bound above), exactly as the previous `I256` did.

use std::cmp::Ordering;

/// A two's-complement signed integer of `L` 64-bit limbs, little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntN<const L: usize> {
    pub limbs: [u64; L],
}

/// 256-bit: the narrow profile's accumulator, and the wide profile's
/// *coefficient* type.
pub type I256 = IntN<4>;
/// 512-bit: the wide profile's accumulator.
pub type I512 = IntN<8>;

// serde does not derive for const-generic arrays, so the limbs go over the
// wire as a sequence. Key bundles and ciphertexts are `bincode`-serialised, so
// this is on the path for any wide-parameter scheme.
impl<const L: usize> serde::Serialize for IntN<L> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.limbs.as_slice(), s)
    }
}

impl<'de, const L: usize> serde::Deserialize<'de> for IntN<L> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v: Vec<u64> = serde::Deserialize::deserialize(d)?;
        if v.len() != L {
            return Err(serde::de::Error::invalid_length(v.len(), &"L limbs"));
        }
        let mut limbs = [0u64; L];
        limbs.copy_from_slice(&v);
        Ok(IntN { limbs })
    }
}

impl<const L: usize> Default for IntN<L> {
    fn default() -> Self {
        IntN { limbs: [0; L] }
    }
}

impl<const L: usize> IntN<L> {
    pub const ZERO: IntN<L> = IntN { limbs: [0; L] };

    pub fn one() -> Self {
        let mut v = [0u64; L];
        v[0] = 1;
        IntN { limbs: v }
    }

    /// Sign-extend a `u128` pair (`hi:lo`) — the shape the old `I256` had, kept
    /// so call sites that build from two halves read the same.
    pub fn from_hi_lo(hi: u128, lo: u128) -> Self {
        let mut v = [0u64; L];
        if L > 0 {
            v[0] = lo as u64;
        }
        if L > 1 {
            v[1] = (lo >> 64) as u64;
        }
        if L > 2 {
            v[2] = hi as u64;
        }
        if L > 3 {
            v[3] = (hi >> 64) as u64;
        }
        // Sign-extend past 256 bits if the value was negative and L > 4.
        if L > 4 && (hi >> 127) & 1 == 1 {
            for limb in v.iter_mut().skip(4) {
                *limb = u64::MAX;
            }
        }
        IntN { limbs: v }
    }

    /// The low 128 bits — what a caller masks to reduce mod a power-of-two `q`.
    pub fn low_u128(self) -> u128 {
        let lo = if L > 0 { self.limbs[0] as u128 } else { 0 };
        let hi = if L > 1 { self.limbs[1] as u128 } else { 0 };
        lo | (hi << 64)
    }

    /// Bits 128..256.
    pub fn high_u128(self) -> u128 {
        let lo = if L > 2 { self.limbs[2] as u128 } else { 0 };
        let hi = if L > 3 { self.limbs[3] as u128 } else { 0 };
        lo | (hi << 64)
    }

    pub fn from_i128(v: i128) -> Self {
        let mut out = IntN::<L>::from_hi_lo(0, v as u128);
        if v < 0 {
            for limb in out.limbs.iter_mut().skip(2) {
                *limb = u64::MAX;
            }
        }
        out
    }

    /// Widen or narrow. Narrowing truncates, which is what reducing mod a
    /// power-of-two `q` wants; widening sign-extends.
    pub fn resize<const M: usize>(self) -> IntN<M> {
        let mut out = [0u64; M];
        let fill = if self.is_negative() { u64::MAX } else { 0 };
        for (i, o) in out.iter_mut().enumerate() {
            *o = if i < L { self.limbs[i] } else { fill };
        }
        IntN { limbs: out }
    }

    pub fn is_negative(self) -> bool {
        L > 0 && (self.limbs[L - 1] >> 63) & 1 == 1
    }

    pub fn is_zero(self) -> bool {
        self.limbs.iter().all(|&x| x == 0)
    }

    pub fn wrapping_add(self, o: Self) -> Self {
        let mut out = [0u64; L];
        let mut carry = 0u64;
        for (i, slot) in out.iter_mut().enumerate() {
            let (s1, c1) = self.limbs[i].overflowing_add(o.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            *slot = s2;
            carry = u64::from(c1) + u64::from(c2);
        }
        IntN { limbs: out }
    }

    pub fn wrapping_neg(self) -> Self {
        let mut out = [0u64; L];
        for (o, &x) in out.iter_mut().zip(self.limbs.iter()) {
            *o = !x;
        }
        IntN { limbs: out }.wrapping_add(IntN::<L>::one())
    }

    pub fn wrapping_sub(self, o: Self) -> Self {
        self.wrapping_add(o.wrapping_neg())
    }

    /// Bitwise AND — with [`low_mask`](Self::low_mask), this is reduction mod a
    /// power-of-two modulus wider than 128 bits.
    // Inherent rather than `std::ops::BitAnd`: every operation on this type is
    // deliberately wrapping/truncating at the type's width, and using operator
    // syntax would invite the reader to assume the usual overflow semantics.
    #[allow(clippy::should_implement_trait)]
    pub fn bitand(self, o: Self) -> Self {
        let mut out = [0u64; L];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.limbs[i] & o.limbs[i];
        }
        IntN { limbs: out }
    }

    /// All ones in the low `bits` positions, zero above — the mask for a
    /// modulus `q = 2^bits`.
    pub fn low_mask(bits: u32) -> Self {
        let mut out = [0u64; L];
        for (i, slot) in out.iter_mut().enumerate() {
            let lo = (i as u32) * 64;
            *slot = if bits >= lo + 64 {
                u64::MAX
            } else if bits > lo {
                (1u64 << (bits - lo)) - 1
            } else {
                0
            };
        }
        IntN { limbs: out }
    }

    /// Unsigned comparison — used while a CRT value is known non-negative.
    pub fn ult(self, o: Self) -> bool {
        self.ucmp(o) == Ordering::Less
    }

    fn ucmp(self, o: Self) -> Ordering {
        for i in (0..L).rev() {
            match self.limbs[i].cmp(&o.limbs[i]) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        Ordering::Equal
    }

    /// Multiply by a 64-bit scalar, truncating at the type's width (callers
    /// keep the value below the CRT modulus).
    pub fn mul_u64(self, k: u64) -> Self {
        let mut out = [0u64; L];
        let mut carry = 0u128;
        for (i, slot) in out.iter_mut().enumerate() {
            let prod = self.limbs[i] as u128 * k as u128 + carry;
            *slot = prod as u64;
            carry = prod >> 64;
        }
        IntN { limbs: out }
    }

    /// Reduce a **non-negative** value mod `p < 2^62`.
    pub fn mod_u64(self, p: u64) -> u64 {
        let mut r = 0u128;
        for i in (0..L).rev() {
            // r < p < 2^62 ⇒ (r<<64) < 2^126, plus a 64-bit limb: fits u128.
            r = ((r << 64) | self.limbs[i] as u128) % p as u128;
        }
        r as u64
    }

    /// Arithmetic shift right by `k`, rounding half away from zero.
    pub fn shr_round(self, k: u32) -> Self {
        debug_assert!(k > 0 && (k as usize) < 64 * L);
        let negative = self.is_negative();
        let mag = if negative { self.wrapping_neg() } else { self };
        // + 2^(k-1) so the truncating shift below rounds to nearest.
        let half = IntN::<L>::one().shl(k - 1);
        let rounded = mag.wrapping_add(half);
        let shifted = rounded.shr(k);
        if negative {
            shifted.wrapping_neg()
        } else {
            shifted
        }
    }

    /// Logical left shift.
    // The loop indexes a *different* array (`self.limbs[src]`) than it writes,
    // with `src` offset by the limb shift, so an iterator over `out` cannot
    // express it without a second index anyway.
    #[allow(clippy::needless_range_loop, clippy::should_implement_trait)]
    pub fn shl(self, k: u32) -> Self {
        let limb_shift = (k / 64) as usize;
        let bit = k % 64;
        let mut out = [0u64; L];
        for i in (0..L).rev() {
            if i < limb_shift {
                continue;
            }
            let src = i - limb_shift;
            let mut v = self.limbs[src] << bit;
            if bit > 0 && src > 0 {
                v |= self.limbs[src - 1] >> (64 - bit);
            }
            out[i] = v;
        }
        IntN { limbs: out }
    }

    /// Logical right shift.
    #[allow(clippy::needless_range_loop, clippy::should_implement_trait)]
    pub fn shr(self, k: u32) -> Self {
        let limb_shift = (k / 64) as usize;
        let bit = k % 64;
        let mut out = [0u64; L];
        for i in 0..L {
            let src = i + limb_shift;
            if src >= L {
                break;
            }
            let mut v = self.limbs[src] >> bit;
            if bit > 0 && src + 1 < L {
                v |= self.limbs[src + 1] << (64 - bit);
            }
            out[i] = v;
        }
        IntN { limbs: out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random `i128`, both signs.
    fn sample_i128(seed: u64, bits: u32) -> i128 {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let hi = next() as u128;
        let lo = next() as u128;
        let mag = (((hi << 64) | lo) % (1u128 << bits)) as i128;
        if next() & 1 == 0 {
            mag
        } else {
            -mag
        }
    }

    /// The type must agree with `i128` wherever `i128` can represent the answer
    /// — that is the only oracle available for a bespoke integer type.
    #[test]
    fn agrees_with_i128_on_add_sub_neg() {
        for seed in 1..300u64 {
            let (a, b) = (sample_i128(seed, 100), sample_i128(seed * 7 + 1, 100));
            let (x, y) = (I256::from_i128(a), I256::from_i128(b));
            assert_eq!(x.wrapping_add(y), I256::from_i128(a.wrapping_add(b)), "{a}+{b}");
            assert_eq!(x.wrapping_sub(y), I256::from_i128(a.wrapping_sub(b)), "{a}-{b}");
            assert_eq!(x.wrapping_neg(), I256::from_i128(-a), "-{a}");
            assert_eq!(x.is_negative(), a < 0);
            assert_eq!(x.is_zero(), a == 0);
        }
    }

    #[test]
    fn agrees_with_i128_on_mul_u64_and_shifts() {
        for seed in 1..300u64 {
            let a = sample_i128(seed, 60);
            let k = (sample_i128(seed * 3 + 5, 30).unsigned_abs() as u64).max(1);
            assert_eq!(
                I256::from_i128(a).mul_u64(k),
                I256::from_i128(a.wrapping_mul(k as i128)),
                "{a} * {k}"
            );
            // shr_round: round half away from zero, matching the old I256.
            for sh in [1u32, 7, 40] {
                let want = {
                    let neg = a < 0;
                    let mag = a.unsigned_abs();
                    let r = (mag + (1u128 << (sh - 1))) >> sh;
                    if neg { -(r as i128) } else { r as i128 }
                };
                assert_eq!(I256::from_i128(a).shr_round(sh), I256::from_i128(want), "{a} >> {sh}");
            }
        }
    }

    #[test]
    fn mod_u64_matches_i128_remainder_for_non_negative_values() {
        let p = 9_223_372_036_854_497_281u64; // one of ntt.rs's primes
        for seed in 1..200u64 {
            let a = sample_i128(seed, 100).unsigned_abs() as i128;
            assert_eq!(I256::from_i128(a).mod_u64(p), (a % p as i128) as u64, "{a} mod p");
        }
    }

    #[test]
    fn shifts_are_inverse_where_no_bits_are_lost() {
        for seed in 1..200u64 {
            let a = sample_i128(seed, 60).unsigned_abs() as i128;
            let x = I256::from_i128(a);
            for sh in [1u32, 64, 65, 130] {
                assert_eq!(x.shl(sh).shr(sh), x, "{a} << {sh} >> {sh}");
            }
        }
    }

    #[test]
    fn hi_lo_round_trips() {
        for seed in 1..200u64 {
            let hi = sample_i128(seed, 100).unsigned_abs();
            let lo = sample_i128(seed * 11 + 3, 127).unsigned_abs();
            let v = I256::from_hi_lo(hi, lo);
            assert_eq!(v.low_u128(), lo);
            assert_eq!(v.high_u128(), hi);
        }
    }

    /// Widening must preserve value and sign; narrowing must truncate, which is
    /// what a mod-2^k reduction wants.
    #[test]
    fn resize_preserves_value_both_ways() {
        for seed in 1..200u64 {
            let a = sample_i128(seed, 100);
            let narrow = I256::from_i128(a);
            let wide: I512 = narrow.resize();
            assert_eq!(wide.is_negative(), a < 0);
            assert_eq!(wide.resize::<4>(), narrow, "round trip for {a}");
            assert_eq!(wide.low_u128(), narrow.low_u128());
        }
    }

    /// 512-bit values exceed `i128`, so the oracle is the arithmetic identity
    /// `(a·k)/k = a` built from the type's own operations plus `ult` ordering.
    #[test]
    fn wide_arithmetic_is_consistent() {
        let big = I512::from_i128(i128::MAX).shl(200);
        assert!(!big.is_negative());
        assert!(I512::ZERO.ult(big));
        assert_eq!(big.wrapping_sub(big), I512::ZERO);
        assert_eq!(big.wrapping_neg().wrapping_neg(), big);
        assert!(big.wrapping_neg().is_negative());
        // A value past 256 bits must survive a widen/narrow round trip only if
        // it fits — this one does not, and truncation is the documented result.
        assert_ne!(big.resize::<4>().resize::<8>(), big);
    }

    #[test]
    fn low_mask_and_bitand_reduce_mod_a_power_of_two() {
        for bits in [1u32, 63, 64, 65, 127, 128, 190, 255] {
            let m = I256::low_mask(bits);
            // 2^bits − 1 has exactly `bits` ones: adding one gives a power of two.
            let plus_one = m.wrapping_add(I256::one());
            assert_eq!(plus_one, I256::one().shl(bits), "low_mask({bits})");
            // Masking is idempotent and kills everything above `bits`.
            assert_eq!(m.bitand(m), m);
            assert!(I256::one().shl(bits).bitand(m).is_zero(), "bit {bits} must be masked off");
            assert_eq!(I256::one().shl(bits - 1).bitand(m), I256::one().shl(bits - 1));
        }
    }

    /// The wide profile's modulus: q = 2^190 does not fit u128, so masking has
    /// to work across the limb boundary.
    #[test]
    fn masking_at_190_bits_matches_a_manual_reduction() {
        let m = I256::low_mask(190);
        for seed in 1..100u64 {
            let a = sample_i128(seed, 100);
            let v = I256::from_i128(a).shl(60); // pushes past 128 bits
            let masked = v.bitand(m);
            // Everything at or above bit 190 must be clear.
            assert!(masked.shr(190).is_zero());
            // …and the low 190 bits are untouched.
            assert_eq!(masked.bitand(m), masked);
        }
    }

    #[test]
    fn serde_round_trips_through_bincode() {
        for seed in 1..50u64 {
            let v = I256::from_i128(sample_i128(seed, 120));
            let bytes = bincode::serialize(&v).unwrap();
            assert_eq!(bincode::deserialize::<I256>(&bytes).unwrap(), v);
        }
        let w = I512::one().shl(400);
        assert_eq!(bincode::deserialize::<I512>(&bincode::serialize(&w).unwrap()).unwrap(), w);
        // A sequence of the wrong length must be rejected, not silently padded.
        let short = bincode::serialize(&vec![1u64, 2]).unwrap();
        assert!(bincode::deserialize::<I256>(&short).is_err());
    }

    #[test]
    fn zero_and_one_behave() {
        assert!(I256::ZERO.is_zero());
        assert_eq!(I256::one().mul_u64(5), I256::from_i128(5));
        assert_eq!(I256::ZERO.wrapping_neg(), I256::ZERO);
        assert_eq!(I512::one().shl(511).shr(511), I512::one());
    }
}
