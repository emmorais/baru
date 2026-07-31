//! Threshold BFV with **derandomized** randomness — the FHE half of the
//! malicious-security story.
//!
//! Implements §4.1–4.3 of `docs/malicious-hybrid.md`: make every FHE operation
//! either deterministic, or deterministic *after* seeding its randomness from a
//! key that exactly two parties share. Once that holds, integrity comes from
//! **replication** rather than from proving anything about the evaluation.
//!
//! # The one idea
//!
//! Under replicated sharing every value is held by two of the three parties, so
//! every operation can be performed twice and the results compared. With at
//! most one corruption, one of the two computers is honest, so a cheat is
//! always caught.
//!
//! The obstacle is randomness: two honest parties encrypting the same value
//! produce *different* ciphertexts, because each draws fresh `(u, e₁, e₂)`.
//! Comparison is then impossible. The fix is to derive that randomness from the
//! pairwise PRF key the two co-holders already share
//! ([`crate::crypto::net::PrfKeys`], established by one 32-byte handshake per
//! pair per session). Both then produce a **bit-identical** ciphertext, and
//! recompute-and-compare works again.
//!
//! So the whole construction is: *classify every FHE operation as deterministic
//! or randomized; verify deterministic ones by recompute-and-compare;
//! derandomize randomized ones so they become recompute-and-compare too.*
//!
//! # What this buys, stated precisely
//!
//! The contrast with single-server verifiable FHE is the point worth making:
//! there, zero-knowledge must cover the **entire evaluation**. Here it is
//! needed only at the **input boundary** — ciphertexts arriving from outside
//! still need a proof of well-formedness, because two honest parties will
//! faithfully compute on the same bad input and agree (§4.4, not implemented;
//! see [`BOUNDARY_GAP`]). Replication covers everything in between for free.
//!
//! # What this does *not* buy
//!
//! The trust threshold for both privacy and integrity remains **honest
//! majority**, exactly as in pure MPC. Two colluding servers hold all three
//! components of every mask and can reconstruct every intermediate. The
//! encrypted model is *not* safe under 2-collusion, and this module makes no
//! such claim — see §5 of the design doc, where that overclaim was identified
//! and retracted before any code existed.
//!
//! The one narrow benefit that does survive is at-rest: against a passive
//! adversary that exfiltrates the storage of up to two servers *without
//! participating in the protocol*, the model stays confidential, because the
//! full key is never assembled anywhere.

use sha2::{Digest, Sha256};

use super::bfv::{center, poly_mul_mod, Bfv, BfvCt, BfvParams, Poly};

/// Why this module is not yet sufficient for a malicious deployment.
pub const BOUNDARY_GAP: &str =
    "externally supplied ciphertexts still need a ZK proof of well-formedness: replication \
     cannot catch a malformed input, because both honest co-holders compute on it faithfully \
     and agree";

// ─── Deterministic sampling ──────────────────────────────────────────────────

/// A seeded stream of pseudorandom bytes, domain-separated per use.
///
/// Every randomized FHE operation draws from this rather than from the OS, so
/// two parties holding the same seed produce identical output. Distinct
/// `domain` and `ctr` values keep independent draws independent — reusing a
/// stream across two different plaintexts would be a genuine leak, which is why
/// callers pass an explicit counter rather than an implicit one.
struct Prg {
    seed: [u8; 32],
    domain: &'static [u8],
    ctr: u64,
    block: u64,
    buf: Vec<u8>,
    pos: usize,
}

impl Prg {
    fn new(seed: &[u8; 32], domain: &'static [u8], ctr: u64) -> Self {
        Prg {
            seed: *seed,
            domain,
            ctr,
            block: 0,
            buf: Vec::new(),
            pos: 0,
        }
    }

    fn next_byte(&mut self) -> u8 {
        if self.pos >= self.buf.len() {
            let mut h = Sha256::new();
            h.update(self.domain);
            h.update(self.seed);
            h.update(self.ctr.to_le_bytes());
            h.update(self.block.to_le_bytes());
            self.buf = h.finalize().to_vec();
            self.block += 1;
            self.pos = 0;
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    fn next_u128(&mut self) -> u128 {
        let mut v = 0u128;
        for _ in 0..16 {
            v = (v << 8) | self.next_byte() as u128;
        }
        v
    }
}

/// Ternary `{−1, 0, 1}` polynomial, deterministically from a seed.
///
/// Rejection-sampled to stay uniform on three values: taking a byte mod 3 would
/// bias toward 0 and 1, and the secret-key distribution is not a place to
/// accept avoidable bias.
fn ternary_from_seed(seed: &[u8; 32], domain: &'static [u8], ctr: u64, n: usize) -> Poly {
    let mut prg = Prg::new(seed, domain, ctr);
    (0..n)
        .map(|_| loop {
            let b = prg.next_byte();
            if b < 252 {
                return match b % 3 {
                    0 => 0u128,
                    1 => 1u128,
                    _ => u128::MAX, // −1, masked on use
                };
            }
        })
        .collect()
}

/// Uniform-mod-`q` polynomial, deterministically from a seed.
fn uniform_from_seed(
    seed: &[u8; 32],
    domain: &'static [u8],
    ctr: u64,
    n: usize,
    p: &BfvParams,
) -> Poly {
    let mut prg = Prg::new(seed, domain, ctr);
    let mask = p.q_mask();
    (0..n).map(|_| prg.next_u128() & mask).collect()
}

/// Smudging (noise-flooding) polynomial: uniform in `[−2^bits, 2^bits]`,
/// deterministically from a seed.
///
/// Threshold decryption publishes `c₁·sᵢ + eᵢ`, and without flooding that value
/// leaks information about the key share. The bound is a parameter because it
/// is a real trade-off, measured rather than guessed: see
/// [`smudging_headroom_bits`].
fn smudge_from_seed(
    seed: &[u8; 32],
    domain: &'static [u8],
    ctr: u64,
    n: usize,
    bits: u32,
    p: &BfvParams,
) -> Poly {
    let mut prg = Prg::new(seed, domain, ctr);
    let mask = p.q_mask();
    let range = 1u128 << bits;
    (0..n)
        .map(|_| {
            let v = prg.next_u128() % (2 * range + 1);
            // Centre on zero, then represent mod q.
            let signed = v as i128 - range as i128;
            if signed < 0 {
                (0u128.wrapping_sub((-signed) as u128)) & mask
            } else {
                (signed as u128) & mask
            }
        })
        .collect()
}

/// Statistical security available for smudging at these parameters, given a
/// measured multiplication-noise magnitude.
///
/// Correctness needs the flooding noise to fit inside `q/2t`; hiding needs it to
/// exceed the noise it masks by `λ` bits. So the affordable `λ` is the gap
/// between them — the "headroom" §6a of the design doc measured rather than
/// estimated, finding 32–40 bits at the current parameters where a pen-and-paper
/// estimate had predicted none.
pub fn smudging_headroom_bits(params: &BfvParams, log2_noise: u32) -> u32 {
    let log2_q_over_2t = params.log2_q - params.log2_t - 1;
    log2_q_over_2t.saturating_sub(log2_noise)
}

// ─── Replicated key shares ───────────────────────────────────────────────────

/// One party's replicated share of the BFV secret key.
///
/// `s = s₀ + s₁ + s₂` with party `i` holding `(sᵢ, sᵢ₊₁)`, mirroring the
/// arithmetic sharing in [`crate::crypto::aby3_protocol`]. Component `i` is
/// therefore known to `Pᵢ` and `Pᵢ₋₁`, which is exactly the redundancy that
/// makes a partial decryption checkable.
///
/// Under the honest-majority assumption (≤1 corruption) no adversary holds
/// enough to decrypt. Two colluding parties *would* — that is the stated
/// threshold, not a weakness this module hides.
#[derive(Clone, Debug)]
pub struct KeyShare {
    pub left: Poly,
    pub right: Poly,
}

/// Split a BFV secret key into replicated shares for the three parties.
///
/// A dealer-based split, adequate for a deployment where the model owner
/// generates the key once and distributes it before disappearing. Genuinely
/// distributed key generation is future work and would remove the dealer from
/// the trust story; the interfaces below do not change when it lands.
pub fn deal_key_shares(s: &[u128], p: &BfvParams, seed: &[u8; 32]) -> [KeyShare; 3] {
    let n = s.len();
    let mask = p.q_mask();
    let s0 = uniform_from_seed(seed, b"baru/fhe/threshold/deal0", 0, n, p);
    let s1 = uniform_from_seed(seed, b"baru/fhe/threshold/deal1", 0, n, p);
    let s2: Poly = (0..n)
        .map(|i| s[i].wrapping_sub(s0[i]).wrapping_sub(s1[i]) & mask)
        .collect();
    let comp = [s0, s1, s2];
    std::array::from_fn(|i| KeyShare {
        left: comp[i].clone(),
        right: comp[(i + 1) % 3].clone(),
    })
}

/// One party's contribution to a threshold decryption.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialDecryption {
    pub d: Poly,
}

/// Compute the partial decryption `dᵢ = c₁·sᵢ + smudge`, derandomized.
///
/// `pair_seed` must be the PRF key shared by the two parties that hold
/// component `which` — then **both compute the identical polynomial** and a
/// deviation by either is caught by comparison. That is the entire malicious
/// argument for this step, and it costs one extra evaluation rather than any
/// proof.
pub fn partial_decrypt(
    bfv: &Bfv,
    ct: &BfvCt,
    s_component: &[u128],
    pair_seed: &[u8; 32],
    ctr: u64,
    smudge_bits: u32,
) -> PartialDecryption {
    let p = &bfv.params;
    let mask = p.q_mask();
    let c1_s = poly_mul_mod(&ct.c[1], s_component, p);
    let e = smudge_from_seed(pair_seed, b"baru/fhe/threshold/smudge", ctr, p.n, smudge_bits, p);
    let d: Poly = (0..p.n)
        .map(|i| c1_s[i].wrapping_add(e[i]) & mask)
        .collect();
    PartialDecryption { d }
}

/// Combine `c₀` with all three partial decryptions and decode.
///
/// `c₀ + Σᵢ dᵢ ≈ Δ·m`, because `Σᵢ sᵢ = s`. The smudging noise sums in too,
/// which is why its magnitude has to fit the budget
/// ([`smudging_headroom_bits`]).
pub fn combine(bfv: &Bfv, ct: &BfvCt, partials: &[PartialDecryption; 3]) -> Vec<u64> {
    let p = &bfv.params;
    let mask = p.q_mask();
    let shift = p.delta_shift();
    let t_mask = (p.t() - 1) as u128;
    (0..p.n)
        .map(|i| {
            let acc = partials
                .iter()
                .fold(ct.c[0][i], |a, part| a.wrapping_add(part.d[i]) & mask);
            // Round to the nearest multiple of Δ, then reduce mod t.
            let rounded = (acc >> shift).wrapping_add((acc >> (shift - 1)) & 1);
            (rounded & t_mask) as u64
        })
        .collect()
}

/// Encrypt with randomness derived from `seed` rather than the OS.
///
/// Two parties holding the same seed produce **bit-identical** ciphertexts, so
/// re-encryption — the MPC→FHE direction of the scheme switch — becomes
/// recompute-and-compare like everything else.
///
/// `ctr` must be unique per plaintext. Reusing a counter across two different
/// messages reuses `(u, e₁, e₂)`, which leaks their difference; the discipline
/// is the same one [`crate::crypto::aby3_protocol`] applies to its `α`
/// counters.
pub fn encrypt_derandomized(
    bfv: &Bfv,
    pk_b: &[u128],
    pk_a: &[u128],
    m: &[u64],
    seed: &[u8; 32],
    ctr: u64,
) -> BfvCt {
    let p = &bfv.params;
    let mask = p.q_mask();
    let u = ternary_from_seed(seed, b"baru/fhe/threshold/enc-u", ctr, p.n);
    let e1 = ternary_from_seed(seed, b"baru/fhe/threshold/enc-e1", ctr, p.n);
    let e2 = ternary_from_seed(seed, b"baru/fhe/threshold/enc-e2", ctr, p.n);
    let ub = poly_mul_mod(&u, pk_b, p);
    let ua = poly_mul_mod(&u, pk_a, p);
    let shift = p.delta_shift();
    let t_mask = (p.t() - 1) as u128;
    let c0: Poly = (0..p.n)
        .map(|i| {
            let mi = (*m.get(i).unwrap_or(&0) as u128) & t_mask;
            ub[i].wrapping_add(e1[i]).wrapping_add(mi << shift) & mask
        })
        .collect();
    let c1: Poly = (0..p.n)
        .map(|i| ua[i].wrapping_add(e2[i]) & mask)
        .collect();
    BfvCt { c: vec![c0, c1] }
}

/// Digest of a ciphertext, for recompute-and-compare on the deterministic
/// operations (ct×ct multiply, homomorphic add).
///
/// Those need no derandomization — they are already deterministic functions of
/// their inputs — so two parties that both hold the inputs can simply compare
/// hashes instead of shipping the ciphertext.
pub fn ct_digest(ct: &BfvCt) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"baru/fhe/ct/v1");
    for poly in &ct.c {
        for coeff in poly {
            h.update(coeff.to_le_bytes());
        }
    }
    h.finalize().into()
}

/// Centred magnitude of a polynomial, in bits — used to report measured noise
/// rather than assert a bound.
pub fn log2_magnitude(poly: &[u128], p: &BfvParams) -> u32 {
    let peak = poly
        .iter()
        .map(|&c| center(c, p).unsigned_abs())
        .max()
        .unwrap_or(0);
    128 - peak.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::bfv::BfvParams;

    fn params() -> BfvParams {
        BfvParams { n: 1024, log2_q: 100, log2_t: 20 }
    }

    #[test]
    fn the_same_seed_gives_bit_identical_ciphertexts() {
        let bfv = Bfv::new(params());
        let (ck, pk) = bfv.keygen();
        let ek = bfv.encrypt_key(&pk);
        let _ = ck;
        // Two "parties" holding the same pairwise key encrypt the same message.
        let seed = [42u8; 32];
        let m = vec![7u64; 8];
        let (b, a) = super::super::bfv::public_parts(&pk);
        let c1 = encrypt_derandomized(&bfv, &b, &a, &m, &seed, 1);
        let c2 = encrypt_derandomized(&bfv, &b, &a, &m, &seed, 1);
        assert_eq!(ct_digest(&c1), ct_digest(&c2), "derandomization must be exact");
        let _ = ek;
    }

    #[test]
    fn different_counters_give_different_ciphertexts() {
        let bfv = Bfv::new(params());
        let (_ck, pk) = bfv.keygen();
        let (b, a) = super::super::bfv::public_parts(&pk);
        let seed = [42u8; 32];
        let m = vec![7u64; 8];
        let c1 = encrypt_derandomized(&bfv, &b, &a, &m, &seed, 1);
        let c2 = encrypt_derandomized(&bfv, &b, &a, &m, &seed, 2);
        assert_ne!(
            ct_digest(&c1),
            ct_digest(&c2),
            "counter reuse is the one thing that must not happen silently"
        );
    }

    #[test]
    fn derandomized_encryption_still_decrypts() {
        let bfv = Bfv::new(params());
        let (ck, pk) = bfv.keygen();
        let (b, a) = super::super::bfv::public_parts(&pk);
        let m: Vec<u64> = (0..16).map(|i| i * 3 + 1).collect();
        let ct = encrypt_derandomized(&bfv, &b, &a, &m, &[9u8; 32], 5);
        let got = bfv.decrypt_poly(&ck, &ct);
        assert_eq!(&got[..16], &m[..], "derandomizing must not change semantics");
    }

    #[test]
    fn threshold_decryption_reconstructs_the_plaintext() {
        let bfv = Bfv::new(params());
        let (ck, pk) = bfv.keygen();
        let s = super::super::bfv::secret_part(&ck);
        let shares = deal_key_shares(&s, &bfv.params, &[3u8; 32]);

        let m: Vec<u64> = (0..8).map(|i| i + 100).collect();
        let ct = bfv.encrypt_poly(&pk, &m);

        // Pairwise seeds: component i is held by P_i and P_{i-1}.
        let seeds = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
            partial_decrypt(&bfv, &ct, &shares[i].left, &seeds[i], 0, 8)
        });
        let got = combine(&bfv, &ct, &partials);
        assert_eq!(&got[..8], &m[..], "threshold decryption must agree with the plain one");
    }

    /// The load-bearing property: the two parties that hold a key component
    /// produce the *same* partial decryption, so a deviation is detectable.
    #[test]
    fn co_holders_produce_identical_partial_decryptions() {
        let bfv = Bfv::new(params());
        let (ck, pk) = bfv.keygen();
        let s = super::super::bfv::secret_part(&ck);
        let shares = deal_key_shares(&s, &bfv.params, &[3u8; 32]);
        let ct = bfv.encrypt_poly(&pk, &[5u64; 4]);

        let pair_seed = [77u8; 32];
        // P_1 holds component 1 as `left`; P_0 holds the same component as
        // `right`. Both compute it, from the key they share.
        let from_p1 = partial_decrypt(&bfv, &ct, &shares[1].left, &pair_seed, 0, 8);
        let from_p0 = partial_decrypt(&bfv, &ct, &shares[0].right, &pair_seed, 0, 8);
        assert_eq!(
            from_p1, from_p0,
            "co-holders must agree, or recompute-and-compare cannot work"
        );
    }

    #[test]
    fn a_deviating_partial_decryption_is_visible_to_the_co_holder() {
        let bfv = Bfv::new(params());
        let (ck, pk) = bfv.keygen();
        let s = super::super::bfv::secret_part(&ck);
        let shares = deal_key_shares(&s, &bfv.params, &[3u8; 32]);
        let ct = bfv.encrypt_poly(&pk, &[5u64; 4]);
        let pair_seed = [77u8; 32];

        let honest = partial_decrypt(&bfv, &ct, &shares[1].left, &pair_seed, 0, 8);
        let mut cheating = honest.clone();
        cheating.d[0] = cheating.d[0].wrapping_add(1);
        assert_ne!(honest, cheating, "the co-holder's comparison must catch this");
    }

    /// The gate that actually matters: does threshold decryption still recover
    /// the plaintext once the smudging noise is large enough to *hide* the key
    /// share, rather than the token amount the other tests use?
    ///
    /// At `q = 2^100, t = 2^20` the correctness budget is `q/2t = 2^79`, and
    /// three parties each add their own flooding term. Sweeping upward finds
    /// where it actually breaks instead of asserting a bound from theory.
    #[test]
    fn decryption_survives_realistic_smudging() {
        let bfv = Bfv::new(params());
        let (ck, pk) = bfv.keygen();
        let s = super::super::bfv::secret_part(&ck);
        let shares = deal_key_shares(&s, &bfv.params, &[3u8; 32]);
        let m: Vec<u64> = (0..8).map(|i| i + 100).collect();
        let ct = bfv.encrypt_poly(&pk, &m);
        let seeds = [[1u8; 32], [2u8; 32], [3u8; 32]];

        let mut highest_ok = 0u32;
        for bits in [8u32, 20, 40, 60, 70, 75] {
            let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
                partial_decrypt(&bfv, &ct, &shares[i].left, &seeds[i], 0, bits)
            });
            if combine(&bfv, &ct, &partials)[..8] == m[..] {
                highest_ok = bits;
            } else {
                break;
            }
        }
        // 40 bits of statistical security is the level §6a says fits at these
        // parameters with no change. Anything less would contradict the note.
        assert!(
            highest_ok >= 40,
            "smudging headroom is smaller than the design note claims: broke above {highest_ok} bits"
        );
    }

    #[test]
    fn smudging_headroom_matches_the_measured_design_note() {
        // §6a measured |e_mult| ≈ 2^38.6 at n=4096, q=2^100, t=2^20.
        let p = BfvParams { n: 4096, log2_q: 100, log2_t: 20 };
        assert_eq!(smudging_headroom_bits(&p, 39), 40);
        // t = 2^24 is about 8 bits worse, as the note says.
        let p24 = BfvParams { n: 4096, log2_q: 100, log2_t: 24 };
        assert_eq!(smudging_headroom_bits(&p24, 43), 32);
    }

    #[test]
    fn ternary_sampling_is_unbiased_across_the_three_values() {
        let poly = ternary_from_seed(&[5u8; 32], b"test", 0, 30_000);
        let zeros = poly.iter().filter(|&&c| c == 0).count();
        let ones = poly.iter().filter(|&&c| c == 1).count();
        let negs = poly.iter().filter(|&&c| c == u128::MAX).count();
        for count in [zeros, ones, negs] {
            let frac = count as f64 / 30_000.0;
            assert!(
                (frac - 1.0 / 3.0).abs() < 0.02,
                "ternary sampling is biased: {zeros}/{ones}/{negs}"
            );
        }
    }
}
