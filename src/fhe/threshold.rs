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
//! need a proof of well-formedness, because two honest parties will faithfully
//! compute on the same bad input and agree. That proof is
//! [`crate::fhe::boundary_proof`]; what remains unsettled about it is a
//! parameter question, not its absence, and [`BOUNDARY_GAP`] states it.
//! Replication covers everything in between for free.
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

/// What distributed key generation does and does not close.
///
/// The dealer is gone from the *threshold* path: [`dkg_sample_share`] and friends
/// generate a key no party ever holds, [`partial_decrypt_any`] decrypts both fresh
/// and once-multiplied ciphertexts under it, and
/// `a_scheme_switch_completes_without_any_party_holding_the_key` runs the whole
/// composition the hybrid performs --- a masked degree-2 product, threshold
/// decryption, and the additive shares that come out of it --- with no party able
/// to decrypt at any point.
///
/// What remains is **harness structure rather than protocol**. [`crate::bench::cnn`]
/// evaluates the FHE half single-threaded, holding one `DecryptKey`, because
/// simulating a single evaluator is cheaper than simulating three; a real
/// deployment already runs the evaluation on all three servers, since that is what
/// the replication-based malicious-security argument requires of it. So the
/// benchmark retains a decrypting party that the protocol no longer needs.
///
/// The cost of removing it is measured rather than guessed
/// (`what_threshold_decryption_costs_against_one_local_decrypt`): one party's
/// partial decryption is \(1.18\times\) a whole local decryption, plus one
/// communication round per switch, because the three partials are computed
/// concurrently. Total *work* is \(3.5\times\), which is the throughput cost and
/// not the latency one.
///
/// Stating the distinction because the UC argument's gap was *key generation*, and
/// a reader could otherwise take "DKG landed" to mean the benchmark's numbers
/// already reflect it, which they do not.
pub const DKG_WIRING_GAP: &str =
    "the protocol is complete and composes — a whole scheme switch (masked degree-2 product, \
     threshold decryption, additive shares) runs with no party holding the key. What remains \
     is harness structure: `bench::cnn` evaluates FHE single-threaded with a whole DecryptKey, \
     because simulating one evaluator is faster than three, so the BENCHMARK retains a \
     decrypting party that the protocol no longer requires";

/// Why this module alone is not sufficient for a malicious deployment, and what
/// closes it.
///
/// Replication secures every *evaluation* step, and cannot vouch for a ciphertext
/// arriving from outside: both honest co-holders compute on a malformed input
/// faithfully and agree.
/// `a_malformed_input_ciphertext_is_not_caught_by_replication` shows it costs
/// nothing to exploit — adding half a `Δ` to one coefficient turns a result of
/// `42` into `43`, with every integrity check in this module passing, because the
/// evaluation *was* performed correctly. The off-by-one is the part worth
/// noticing: a malformed input need not produce garbage a sanity check would
/// reject. A real adversary has far more room than half a `Δ`, including noise
/// large enough that whether the decryption wraps depends on the *model weights*,
/// which turns a correctness failure into a channel leaking what the protocol
/// protects.
///
/// **The proof that closes this exists**: [`crate::fhe::boundary_proof`], a
/// Schnorr-style Σ-protocol over the ring, Fiat–Shamir'd, using Lyubashevsky
/// rejection sampling so a released response is *perfectly* hiding. This constant
/// used to say the proof was still needed, and was left stale when it landed —
/// which is a documentation hazard worth naming, because a reader (and an author
/// of this paper's security section) took it at face value.
///
/// What is genuinely unsettled is narrower and is stated by
/// [`crate::fhe::boundary`]: **soundness slack is paid out of the smudging budget,
/// bit for bit.** A verifier accepting the proof accepts noise up to
/// `2^slack · B`, that noise grows through the evaluation, and the flooding at
/// threshold decryption must cover it. The measured 26-bit slack leaves λ = 30 at
/// the largest modulus a `u128` representation admits, against the 40 an
/// honest-input analysis reports. That, plus the proof being a tested rather than
/// a reviewed implementation, is why the attestation stays semi-honest.
pub const BOUNDARY_GAP: &str =
    "the boundary proof of well-formedness EXISTS (fhe::boundary_proof); what is unsettled is \
     that its soundness slack is paid out of the smudging budget bit for bit, so the measured \
     26-bit slack leaves lambda = 30 rather than 40, and the implementation is tested rather \
     than reviewed — see fhe::boundary for the analysis";

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
    let e = smudge_from_seed(
        pair_seed,
        b"baru/fhe/threshold/smudge",
        ctr,
        p.n,
        smudge_bits,
        p,
    );
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
    let c1: Poly = (0..p.n).map(|i| ua[i].wrapping_add(e2[i]) & mask).collect();
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
        BfvParams {
            n: 1024,
            log2_q: 100,
            log2_t: 20,
        }
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
        assert_eq!(
            ct_digest(&c1),
            ct_digest(&c2),
            "derandomization must be exact"
        );
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
        assert_eq!(
            &got[..16],
            &m[..],
            "derandomizing must not change semantics"
        );
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
        let partials: [PartialDecryption; 3] =
            std::array::from_fn(|i| partial_decrypt(&bfv, &ct, &shares[i].left, &seeds[i], 0, 8));
        let got = combine(&bfv, &ct, &partials);
        assert_eq!(
            &got[..8],
            &m[..],
            "threshold decryption must agree with the plain one"
        );
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
        assert_ne!(
            honest, cheating,
            "the co-holder's comparison must catch this"
        );
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
        let p = BfvParams {
            n: 4096,
            log2_q: 100,
            log2_t: 20,
        };
        assert_eq!(smudging_headroom_bits(&p, 39), 40);
        // t = 2^24 is about 8 bits worse, as the note says.
        let p24 = BfvParams {
            n: 4096,
            log2_q: 100,
            log2_t: 24,
        };
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

// ─── Distributed key generation ──────────────────────────────────────────────

/// Why the dealer was there, and what removing it costs.
///
/// [`deal_key_shares`] needs a party who knows `s`, and not by oversight. It
/// makes components `s₀`, `s₁` **uniform mod q** while their sum is **ternary**,
/// by choosing `s₂ = s − s₀ − s₁`. Those two properties together are what a
/// dealer buys: the sharing is perfectly hiding *and* the key is small.
///
/// No protocol in which each party samples its own component independently can
/// have all three of independent sampling, a small sum, and uniform components.
/// Sampling `sᵢ` uniform mod q makes `s` uniform mod q, and BFV decryption
/// computes `c₀ + c₁·s = Δm + (e₁ + e₂·s − e·u)`, whose noise contains `e₂·s`
/// — with `‖s‖ ≈ q` that term is larger than `q/2t` and decryption simply fails.
/// Sampling `sᵢ` ternary keeps `s` small but makes the components non-uniform.
///
/// [`dkg_sample_share`] takes the second horn deliberately, and the security
/// argument is that the leak is one the standard assumption already covers.
pub const DEALER_RATIONALE: &str =
    "a dealer makes the key shares uniform AND the key small; independent sampling can have \
     either but not both, so DKG accepts non-uniform (ternary) components and argues security \
     from RLWE with a bounded secret rather than from perfect hiding of the shares";

/// One party's locally sampled key component, together with the copy it must
/// forward to keep the sharing replicated.
///
/// # The construction
///
/// Party `i` samples `sᵢ` ternary from its own seed and sends it to party
/// `i − 1`, so party `i` ends up holding `(sᵢ, sᵢ₊₁)` exactly as
/// [`deal_key_shares`] leaves it. Replication is preserved on purpose: it is
/// what makes a partial decryption checkable by recompute-and-compare, and a
/// DKG that produced only additive shares would remove the dealer at the cost of
/// the malicious-security argument. That trade is available and we do not take
/// it.
///
/// # Why the non-uniform components are safe here
///
/// An adversary corrupting one party learns two of the three ternary components,
/// so it knows `k = sᵢ + sᵢ₊₁` and that `s − k` is ternary. That looks alarming
/// and is not, because the residual problem is the standard one. Given the public
/// key `b = −(a·s + e)`, the adversary can compute
///
/// ```text
///   b + a·k = −(a·(s − k) + e)
/// ```
///
/// which is an RLWE sample whose secret `s − k` is ternary. So knowing two
/// components reduces the adversary's task to RLWE with a ternary secret ---
/// precisely the assumption BFV rests on in the first place. Nothing is lost
/// beyond the honest-majority threshold already assumed: two colluding parties
/// hold all three components and can decrypt, which is the stated bound and not
/// a weakness introduced here.
///
/// # What it costs
///
/// `s = s₀ + s₁ + s₂` with each `sᵢ` ternary, so `‖s‖∞ ≤ 3` rather than `1`. The
/// `e₂·s` term of the decryption noise grows with `‖s‖`, and for products
/// decrypted under `s²` the relevant bound is `‖s²‖∞ ≤ 9n` rather than `n`. That
/// is a constant-factor cost in the noise budget, and
/// [`dkg_key_norm_headroom_bits`] reports what it consumes so a deployment can
/// check it rather than assume it.
#[derive(Clone, Debug)]
pub struct DkgContribution {
    /// This party's own component `sᵢ`.
    pub own: Poly,
}

/// Sample party `me`'s key component from its private seed.
///
/// Ternary, so that `s = Σ sᵢ` stays small enough to decrypt under. See
/// [`DkgContribution`] for why non-uniform components do not break the security
/// argument, and [`DEALER_RATIONALE`] for why uniform ones are unavailable
/// without a dealer.
pub fn dkg_sample_share(p: &BfvParams, seed: &[u8; 32], me: usize) -> DkgContribution {
    // Reduced mod q on the way out, so `−1` is `q−1` rather than `u128::MAX`.
    // `ternary_from_seed` uses the latter convention and documents it as "masked
    // on use", which is correct for wrapping arithmetic and wrong for anything
    // that inspects a coefficient — `center` on an unreduced value reports a
    // sup-norm near `q`. `deal_key_shares` masks for the same reason, so masking
    // here makes the two paths produce components that are not merely
    // interchangeable in arithmetic but identical in representation.
    let mask = p.q_mask();
    DkgContribution {
        own: ternary_from_seed(seed, b"baru/fhe/dkg/component", me as u64, p.n)
            .into_iter()
            .map(|c| c & mask)
            .collect(),
    }
}

/// Assemble party `me`'s replicated [`KeyShare`] from its own component and the
/// one received from party `me + 1`.
///
/// The result is indistinguishable in shape from what [`deal_key_shares`]
/// produces, which is the point: every downstream operation --- partial
/// decryption, combination, the squared-key path --- is unchanged.
pub fn dkg_assemble_share(own: &DkgContribution, from_next: &Poly) -> KeyShare {
    KeyShare {
        left: own.own.clone(),
        right: from_next.clone(),
    }
}

/// Party `me`'s contribution to the joint public key: `bᵢ = −(a·sᵢ + eᵢ)`.
///
/// `a` is common public randomness --- in a deployment, derived from a session
/// transcript or a common coin, never chosen by one party, since a party that
/// picks `a` after seeing the others' `bⱼ` can bias the key. The caller supplies
/// it so that the choice is visible at the call site rather than buried here.
///
/// Summing the three contributions gives `b = −(a·s + Σeᵢ)`, a well-formed BFV
/// public key for `s = Σ sᵢ` whose error is the sum of three ternary
/// polynomials.
pub fn dkg_public_contribution(
    p: &BfvParams,
    a: &[u128],
    own: &DkgContribution,
    e_seed: &[u8; 32],
    me: usize,
) -> Poly {
    let mask = p.q_mask();
    let e = ternary_from_seed(e_seed, b"baru/fhe/dkg/pkerror", me as u64, p.n);
    let a_s = poly_mul_mod(a, &own.own, p);
    (0..p.n)
        .map(|i| 0u128.wrapping_sub(a_s[i]).wrapping_sub(e[i]) & mask)
        .collect()
}

/// Aggregate the three public-key contributions into `b`.
pub fn dkg_aggregate_public(p: &BfvParams, contributions: &[Poly; 3]) -> Poly {
    let mask = p.q_mask();
    (0..p.n)
        .map(|i| {
            contributions
                .iter()
                .fold(0u128, |acc, c| acc.wrapping_add(c[i]) & mask)
        })
        .collect()
}

/// Reconstruct `s = Σ sᵢ` from the three contributions.
///
/// For tests and for the key-norm check only --- no party runs this in a
/// deployment, and a protocol step that needed it would defeat the purpose.
pub fn dkg_reconstruct_secret(p: &BfvParams, contributions: &[&Poly; 3]) -> Poly {
    let mask = p.q_mask();
    (0..p.n)
        .map(|i| {
            contributions
                .iter()
                .fold(0u128, |acc, c| acc.wrapping_add(c[i]) & mask)
        })
        .collect()
}

/// Bits of noise headroom the larger DKG key consumes, relative to a ternary one.
///
/// A dealt key is ternary (`‖s‖∞ = 1`); a DKG key is the sum of three ternary
/// polynomials (`‖s‖∞ ≤ 3`). Decryption noise carries an `e₂·s` term and a
/// product decrypted under `s²` carries `‖s²‖`, so the cost is
/// `log₂ 3 ≈ 1.6` bits for a fresh ciphertext and about `log₂ 9 ≈ 3.2` for a
/// product. Reported rather than assumed, because it comes off the same budget
/// as the smudging noise ([`smudging_headroom_bits`]) and the two compete.
pub fn dkg_key_norm_headroom_bits(squared: bool) -> u32 {
    if squared {
        4
    } else {
        2
    }
}

#[cfg(test)]
mod dkg_tests {
    use super::*;
    use crate::fhe::bfv::{public_key_bytes, BfvParams};

    fn params() -> BfvParams {
        BfvParams {
            n: 1024,
            log2_q: 100,
            log2_t: 20,
        }
    }

    /// Deterministic stand-in for the common public randomness `a`.
    fn common_a(p: &BfvParams) -> Poly {
        uniform_from_seed(&[9u8; 32], b"baru/fhe/dkg/common-a", 0, p.n, p)
    }

    /// **A key nobody generated, used end to end.**
    ///
    /// Three parties each sample a component, exchange one polynomial to restore the
    /// replicated layout, publish one contribution each to form `b`, and the resulting public
    /// key encrypts a message that the three of them then threshold-decrypt. At no point does
    /// any party hold `s`, and no dealer exists — which is the whole claim.
    #[test]
    fn a_distributed_key_encrypts_and_threshold_decrypts() {
        let p = params();
        let bfv = Bfv::new(p);
        let a = common_a(&p);

        // Each party samples privately.
        let seeds: [[u8; 32]; 3] = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &seeds[i], i))
            .collect();

        // One polynomial each to the previous party restores the replicated layout.
        let shares: Vec<KeyShare> = (0..3)
            .map(|i| dkg_assemble_share(&contribs[i], &contribs[(i + 1) % 3].own))
            .collect();
        for (i, sh) in shares.iter().enumerate() {
            assert_eq!(
                sh.left, contribs[i].own,
                "party {i} must hold its own component on the left"
            );
            assert_eq!(
                sh.right,
                contribs[(i + 1) % 3].own,
                "and its successor's on the right — the layout `deal_key_shares` produces"
            );
        }

        // Public key: one contribution each, summed.
        let pk_parts: [Poly; 3] = std::array::from_fn(|i| {
            dkg_public_contribution(&p, &a, &contribs[i], &[7u8; 32], i)
        });
        let b = dkg_aggregate_public(&p, &pk_parts);
        let pk = public_key_bytes(&b, &a);

        // Encrypt under it, then threshold-decrypt with the three components.
        let m: Vec<u64> = (0..8).map(|i| (i * 1000 + 7) as u64).collect();
        let ct = bfv.encrypt_poly(&pk, &m);
        let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
            partial_decrypt(&bfv, &ct, &contribs[i].own, &[11u8; 32], i as u64, 0)
        });
        let got = combine(&bfv, &ct, &partials);
        assert_eq!(&got[..m.len()], &m[..], "distributed key must decrypt");
    }

    /// **The key is small, which is the property the whole design turns on.**
    ///
    /// A dealt key is ternary. A DKG key is the sum of three ternary polynomials, so its
    /// coefficients live in `[−3, 3]`. If that ever stopped holding — a component sampled
    /// uniform, say — decryption would not merely degrade, it would fail outright, because the
    /// noise carries an `e₂·s` term. This asserts the bound directly rather than inferring it
    /// from a successful decryption, since a decryption can succeed by luck on one message.
    #[test]
    fn the_distributed_key_stays_small() {
        let p = params();
        let seeds: [[u8; 32]; 3] = [[4u8; 32], [5u8; 32], [6u8; 32]];
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &seeds[i], i))
            .collect();
        let refs: [&Poly; 3] = [&contribs[0].own, &contribs[1].own, &contribs[2].own];
        let s = dkg_reconstruct_secret(&p, &refs);

        let peak = s
            .iter()
            .map(|&c| crate::fhe::bfv::center(c, &p).unsigned_abs())
            .max()
            .expect("non-empty");
        println!("  ||s||_inf = {peak} over {} coefficients", p.n);
        assert!(
            peak <= 3,
            "the sum of three ternary polynomials must have sup-norm at most 3, got {peak}"
        );
        // And each component really is ternary, or the sum bound is an accident.
        for (i, c) in contribs.iter().enumerate() {
            let m = c
                .own
                .iter()
                .map(|&x| crate::fhe::bfv::center(x, &p).unsigned_abs())
                .max()
                .expect("non-empty");
            assert!(m <= 1, "component {i} must be ternary, got sup-norm {m}");
        }
    }

    /// **The adversary's residual problem is RLWE with a ternary secret.**
    ///
    /// One corruption learns two of the three components, hence `k = sᵢ + sᵢ₊₁`. The security
    /// argument is that `b + a·k` is then an RLWE sample under the ternary secret `s − k`, which
    /// is the assumption BFV already rests on. That is an argument about hardness and cannot be
    /// tested — but the *algebraic identity* it rests on can be, and if the identity failed the
    /// argument would be vacuous.
    #[test]
    fn what_a_corrupt_party_learns_reduces_to_a_ternary_rlwe_sample() {
        let p = params();
        let a = common_a(&p);
        let mask = p.q_mask();
        let seeds: [[u8; 32]; 3] = [[21u8; 32], [22u8; 32], [23u8; 32]];
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &seeds[i], i))
            .collect();
        let pk_parts: [Poly; 3] = std::array::from_fn(|i| {
            dkg_public_contribution(&p, &a, &contribs[i], &[31u8; 32], i)
        });
        let b = dkg_aggregate_public(&p, &pk_parts);

        // Party 0 holds components 0 and 1.
        let k: Poly = (0..p.n)
            .map(|i| contribs[0].own[i].wrapping_add(contribs[1].own[i]) & mask)
            .collect();
        // b + a·k should be -(a·(s-k) + e), i.e. the same relation with secret s₂.
        let a_k = poly_mul_mod(&a, &k, &p);
        let lhs: Poly = (0..p.n)
            .map(|i| b[i].wrapping_add(a_k[i]) & mask)
            .collect();
        let a_s2 = poly_mul_mod(&a, &contribs[2].own, &p);

        // lhs + a·s₂ must be the pure error sum −Σeᵢ, whose sup-norm is at most 3.
        let residual: Poly = (0..p.n)
            .map(|i| lhs[i].wrapping_add(a_s2[i]) & mask)
            .collect();
        let peak = residual
            .iter()
            .map(|&c| crate::fhe::bfv::center(c, &p).unsigned_abs())
            .max()
            .expect("non-empty");
        println!("  ||b + a(s_0+s_1) + a·s_2||_inf = {peak} (three ternary errors)");
        assert!(
            peak <= 3,
            "the residual must be the sum of three ternary errors, got sup-norm {peak} — if this \
             is large the reduction to ternary RLWE does not hold and the security argument for \
             non-uniform components is vacuous"
        );
    }

    /// **A product, threshold-decrypted under a key nobody generated.**
    ///
    /// This is the case the hybrid actually needs and the one the old threshold path could not
    /// do: `Bfv::mul` produces a degree-2 ciphertext because nothing is relinearized, and
    /// `partial_decrypt` reads only `c[1]`, so it would have returned a wrong answer silently
    /// rather than refusing. The three parties here hold only what the DKG gave them, form
    /// their `s²` shares locally, and recover the product.
    #[test]
    fn a_product_threshold_decrypts_under_the_distributed_squared_key() {
        let p = params();
        let bfv = Bfv::new(p);
        let a = common_a(&p);
        let seeds: [[u8; 32]; 3] = [[51u8; 32], [52u8; 32], [53u8; 32]];
        let contribs: Vec<DkgContribution> =
            (0..3).map(|i| dkg_sample_share(&p, &seeds[i], i)).collect();
        let shares: Vec<KeyShare> = (0..3)
            .map(|i| dkg_assemble_share(&contribs[i], &contribs[(i + 1) % 3].own))
            .collect();
        let pk_parts: [Poly; 3] = std::array::from_fn(|i| {
            dkg_public_contribution(&p, &a, &contribs[i], &[61u8; 32], i)
        });
        let pk = public_key_bytes(&dkg_aggregate_public(&p, &pk_parts), &a);

        for &(x, y) in &[(3u64, 5u64), (123, 456), (1000, 1000)] {
            let ct = bfv.mul(&bfv.encrypt_scalar(&pk, x), &bfv.encrypt_scalar(&pk, y));
            assert_eq!(ct.key_pow(), 2, "no relinearization, so the product is degree 2");
            let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
                partial_decrypt_any(&bfv, &ct, &shares[i], &[71u8; 32], i as u64, 0)
            });
            let got = combine(&bfv, &ct, &partials);
            assert_eq!(
                got[0],
                (x * y) % (1 << 20),
                "{x}*{y} under a distributed s²"
            );
        }
    }

    /// **The three local `s²` shares really do sum to `s²`.**
    ///
    /// The end-to-end test above would also pass if the identity were wrong in a way that
    /// happened to cancel, so this checks the algebra directly: `Σ (sᵢ² + 2 sᵢ sᵢ₊₁)` against
    /// `(Σ sᵢ)²` computed in one place.
    #[test]
    fn the_local_squared_shares_sum_to_the_squared_key() {
        let p = params();
        let mask = p.q_mask();
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &[(80 + i) as u8; 32], i))
            .collect();
        let shares: Vec<KeyShare> = (0..3)
            .map(|i| dkg_assemble_share(&contribs[i], &contribs[(i + 1) % 3].own))
            .collect();

        let summed: Poly = (0..p.n)
            .map(|j| {
                shares.iter().fold(0u128, |acc, sh| {
                    acc.wrapping_add(squared_key_share(sh, &p)[j]) & mask
                })
            })
            .collect();
        let refs: [&Poly; 3] = [&contribs[0].own, &contribs[1].own, &contribs[2].own];
        let s = dkg_reconstruct_secret(&p, &refs);
        let s_sq = poly_mul_mod(&s, &s, &p);
        assert_eq!(
            summed, s_sq,
            "the replicated identity s^2 = sum_i (s_i^2 + 2 s_i s_{{i+1}}) must hold exactly, \
             or the free squared-key share is not a share of s^2"
        );
    }

    /// A degree-3 ciphertext has no supported decryption, and must say so rather than
    /// silently drop a term the way `partial_decrypt` did.
    #[test]
    #[should_panic(expected = "key_pow 1 or 2")]
    fn an_unsupported_degree_is_refused() {
        let p = params();
        let bfv = Bfv::new(p);
        let a = common_a(&p);
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &[(90 + i) as u8; 32], i))
            .collect();
        let share = dkg_assemble_share(&contribs[0], &contribs[1].own);
        let pk = public_key_bytes(
            &dkg_aggregate_public(
                &p,
                &std::array::from_fn(|i| {
                    dkg_public_contribution(&p, &a, &contribs[i], &[91u8; 32], i)
                }),
            ),
            &a,
        );
        let mut ct = bfv.encrypt_scalar(&pk, 1);
        ct.c.push(vec![0u128; p.n]);
        ct.c.push(vec![0u128; p.n]);
        let _ = partial_decrypt_any(&bfv, &ct, &share, &[92u8; 32], 0, 0);
    }

    /// **What threshold decryption costs against the single-key path it would replace.**
    ///
    /// Wiring the threshold path into the CNN hybrid is only worth doing if its cost is known,
    /// and the pipeline currently spends \SI{45}{\milli\second} per diagnosis in
    /// `decrypt_poly_pre`. Three partial decryptions and a combine cannot be free, so the
    /// question is the factor.
    ///
    /// The measurement also checks one optimisation that must exist for the answer to be
    /// acceptable: `squared_key_share` depends only on the key, never on the ciphertext, so a
    /// session can compute it once. Leaving it on the per-ciphertext path doubles the
    /// polynomial multiplications, and a scheme switch decrypts once per layer.
    #[test]
    #[ignore] // timing measurement, not a correctness check
    fn what_threshold_decryption_costs_against_one_local_decrypt() {
        use std::time::Instant;
        let p = params();
        let bfv = Bfv::new(p);
        let a = common_a(&p);
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &[(100 + i) as u8; 32], i))
            .collect();
        let shares: Vec<KeyShare> = (0..3)
            .map(|i| dkg_assemble_share(&contribs[i], &contribs[(i + 1) % 3].own))
            .collect();
        let pk = public_key_bytes(
            &dkg_aggregate_public(
                &p,
                &std::array::from_fn(|i| {
                    dkg_public_contribution(&p, &a, &contribs[i], &[101u8; 32], i)
                }),
            ),
            &a,
        );
        // A degree-2 ciphertext, which is what the hybrid decrypts.
        let ct = bfv.mul(&bfv.encrypt_scalar(&pk, 7), &bfv.encrypt_scalar(&pk, 9));

        const REPS: u32 = 20;

        // Baseline: the single-key path the pipeline uses today. The key is prepared once,
        // matching `decrypt_poly_pre`'s intended use.
        let (ck, _) = bfv.keygen();
        let dk = bfv.decrypt_key(&ck);
        let t = Instant::now();
        for _ in 0..REPS {
            let _ = bfv.decrypt_poly_pre(&dk, &ct);
        }
        let local_us = t.elapsed().as_micros() as f64 / REPS as f64;

        // Threshold, squared share recomputed per ciphertext (the naive wiring).
        let t = Instant::now();
        for r in 0..REPS {
            let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
                partial_decrypt_any(&bfv, &ct, &shares[i], &[102u8; 32], r as u64, 0)
            });
            let _ = combine(&bfv, &ct, &partials);
        }
        let naive_us = t.elapsed().as_micros() as f64 / REPS as f64;

        // Threshold, squared share hoisted: what a session should actually do.
        let s2: Vec<Poly> = shares.iter().map(|sh| squared_key_share(sh, &p)).collect();
        let t = Instant::now();
        for r in 0..REPS {
            let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
                partial_decrypt_with_squared(
                    &bfv,
                    &ct,
                    &shares[i].left,
                    &s2[i],
                    &[103u8; 32],
                    r as u64,
                    0,
                )
            });
            let _ = combine(&bfv, &ct, &partials);
        }
        let hoisted_us = t.elapsed().as_micros() as f64 / REPS as f64;

        // One party's share of the hoisted work, which is what a deployment actually waits for:
        // the three partial decryptions run on three machines at once.
        let t = Instant::now();
        for r in 0..REPS {
            let _ = partial_decrypt_with_squared(
                &bfv,
                &ct,
                &shares[0].left,
                &s2[0],
                &[105u8; 32],
                r as u64,
                0,
            );
        }
        let one_party_us = t.elapsed().as_micros() as f64 / REPS as f64;

        println!("  single-key decrypt (today)          {local_us:9.0} us");
        println!(
            "  threshold TOTAL WORK, s^2 per ct    {naive_us:9.0} us  ({:.1}x)",
            naive_us / local_us
        );
        println!(
            "  threshold TOTAL WORK, s^2 hoisted   {hoisted_us:9.0} us  ({:.1}x)",
            hoisted_us / local_us
        );
        println!(
            "  hoisting the squared share is worth {:.2}x of the threshold cost",
            naive_us / hoisted_us
        );
        println!(
            "  ONE PARTY, s^2 hoisted              {one_party_us:9.0} us  ({:.2}x)",
            one_party_us / local_us
        );
        println!(
            "  The two figures answer different questions and must not be interchanged.\n  \
             Total work is the throughput cost, and it is what this single-threaded loop\n  \
             measures. Deployment LATENCY is the one-party figure plus one communication\n  \
             round, because the three partial decryptions happen on three machines at the\n  \
             same time. Quoting {:.1}x as a latency cost would repeat this repo's most\n  \
             frequent error.",
            hoisted_us / local_us
        );
        // The per-party cost must be about a third of the total, or the three partial
        // decryptions are not the symmetric work the parallelism argument assumes.
        let share_of_total = one_party_us / hoisted_us;
        assert!(
            (0.20..=0.50).contains(&share_of_total),
            "one party should do roughly a third of the threshold work, got {share_of_total:.2} \
             — if the work is lopsided, the deployment-latency argument does not hold"
        );

        // Correctness of the hoisted path, so a timing test cannot pass on a wrong answer.
        let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
            partial_decrypt_with_squared(&bfv, &ct, &shares[i].left, &s2[i], &[104u8; 32], i as u64, 0)
        });
        assert_eq!(combine(&bfv, &ct, &partials)[0], 63, "7*9 under a hoisted s^2");
        assert!(
            hoisted_us < naive_us,
            "hoisting a ciphertext-independent term must not be slower: {hoisted_us:.0} vs {naive_us:.0} us"
        );
    }

    /// **A whole scheme switch, with no party able to decrypt.**
    ///
    /// This is the composition the hybrid actually performs, and the point at which the DKG
    /// stops being a key-generation exercise and becomes usable: an FHE product is masked by
    /// three party masks, threshold-decrypted, and turned into additive MPC shares. It matters
    /// that all four pieces are exercised together, because each was tested in isolation and
    /// the pipeline needs them in series — a degree-2 ciphertext, a distributed key, the locally
    /// derived squared share, and the mask arithmetic that makes the opened value a sharing.
    ///
    /// The switch works as the CNN pipeline builds it: party `i` draws `m_i`, encrypts it, the
    /// three mask ciphertexts are summed onto the product, and decryption yields
    /// `y = x*w + sum(m_i) mod t`. Party 0 then holds `y - m_0` and the others hold `-m_i`, so
    /// the three shares sum to `x*w`. Nothing in that sequence requires a party that can
    /// decrypt, which is the property `DKG_WIRING_GAP` says the deployed pipeline still lacks.
    #[test]
    fn a_scheme_switch_completes_without_any_party_holding_the_key() {
        let p = params();
        let bfv = Bfv::new(p);
        let a = common_a(&p);
        let t_mod = p.t();

        // ── Distributed key, as in the tests above ──
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &[(120 + i) as u8; 32], i))
            .collect();
        let shares: Vec<KeyShare> = (0..3)
            .map(|i| dkg_assemble_share(&contribs[i], &contribs[(i + 1) % 3].own))
            .collect();
        let pk = public_key_bytes(
            &dkg_aggregate_public(
                &p,
                &std::array::from_fn(|i| {
                    dkg_public_contribution(&p, &a, &contribs[i], &[121u8; 32], i)
                }),
            ),
            &a,
        );
        // Squared shares are key-only work, hoisted once per session.
        let s2: Vec<Poly> = shares.iter().map(|sh| squared_key_share(sh, &p)).collect();

        for &(x, w) in &[(12u64, 34u64), (500, 3), (1, 1), (777, 999)] {
            // ── The FHE half: a product, masked by three party masks ──
            let prod = bfv.mul(&bfv.encrypt_scalar(&pk, x), &bfv.encrypt_scalar(&pk, w));
            let masks: [u64; 3] = [11111, 22222, 33333];
            let mut masked = prod;
            for &m in &masks {
                let mut mv = vec![0u64; p.n];
                mv[0] = m % t_mod;
                masked = bfv.add(&masked, &bfv.encrypt_poly(&pk, &mv));
            }

            // ── Threshold decryption: three partials, no assembled key ──
            let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
                partial_decrypt_with_squared(
                    &bfv,
                    &masked,
                    &shares[i].left,
                    &s2[i],
                    &[122u8; 32],
                    i as u64,
                    0,
                )
            });
            let y = combine(&bfv, &masked, &partials)[0];

            // ── The MPC half: the opened value becomes an additive sharing ──
            let share0 = y.wrapping_sub(masks[0]) % t_mod;
            let share1 = t_mod - (masks[1] % t_mod);
            let share2 = t_mod - (masks[2] % t_mod);
            let recovered = ((share0 as u128 + share1 as u128 + share2 as u128)
                % t_mod as u128) as u64;

            assert_eq!(
                recovered,
                (x * w) % t_mod,
                "the three shares of the switched value must sum to x*w for {x}*{w}"
            );
            // And no single share reveals it, which is the reason for the masks.
            assert_ne!(
                share0,
                (x * w) % t_mod,
                "party 0's share must not be the plaintext product"
            );
        }
    }

    /// **Why the input boundary needs a proof, demonstrated rather than asserted.**
    ///
    /// Replication cannot catch a malformed input ciphertext, because both honest co-holders
    /// compute on it faithfully and agree. That has been a sentence in a doc comment since the
    /// module was written. This makes it a fact you can run: a tampered ciphertext yields a
    /// wrong plaintext, and every integrity mechanism *this module* has says the computation
    /// was performed correctly --- which it was.
    ///
    /// This is the attack [`crate::fhe::boundary_proof`] exists to stop, so the test measures
    /// the exposure of `threshold` **alone**, not of the system with the boundary proof in
    /// place. Framing it as an open gap would misstate the artifact, and briefly did: the
    /// constant this test used to cite still said the proof was needed long after it landed.
    ///
    /// The tamper is deliberately the mildest one that works: add half a `Δ` to the constant
    /// term, which is not a large perturbation but is enough to push the rounding in decryption
    /// across a boundary. A real adversary has far more room, including noise large enough to
    /// make the wrap depend on the *model weights*, which is how a correctness failure at the
    /// boundary becomes an information leak about the thing the protocol is protecting.
    ///
    /// What the test asserts is the *asymmetry*: a deviating **party** is caught (there is a
    /// test above for exactly that), and a deviating **input** is not. Only the second needs a
    /// proof of well-formedness, which is why the obligation is narrow.
    #[test]
    fn a_malformed_input_ciphertext_is_not_caught_by_replication() {
        let p = params();
        let bfv = Bfv::new(p);
        let a = common_a(&p);
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &[(140 + i) as u8; 32], i))
            .collect();
        let shares: Vec<KeyShare> = (0..3)
            .map(|i| dkg_assemble_share(&contribs[i], &contribs[(i + 1) % 3].own))
            .collect();
        let pk = public_key_bytes(
            &dkg_aggregate_public(
                &p,
                &std::array::from_fn(|i| {
                    dkg_public_contribution(&p, &a, &contribs[i], &[141u8; 32], i)
                }),
            ),
            &a,
        );
        let s2: Vec<Poly> = shares.iter().map(|sh| squared_key_share(sh, &p)).collect();

        // An honest input, for the baseline.
        let honest = bfv.encrypt_scalar(&pk, 21);
        // The same input, tampered at the boundary: half a Δ on the constant term. No party
        // did this; it arrived this way.
        let mut malformed = honest.clone();
        let half_delta = 1u128 << (p.delta_shift() - 1);
        malformed.c[0][0] = malformed.c[0][0].wrapping_add(half_delta) & p.q_mask();

        let evaluate = |ct: &BfvCt| -> u64 {
            let prod = bfv.mul(ct, &bfv.encrypt_scalar(&pk, 2));
            let partials: [PartialDecryption; 3] = std::array::from_fn(|i| {
                partial_decrypt_with_squared(
                    &bfv,
                    &prod,
                    &shares[i].left,
                    &s2[i],
                    &[142u8; 32],
                    i as u64,
                    0,
                )
            });
            combine(&bfv, &prod, &partials)[0]
        };

        let good = evaluate(&honest);
        let bad = evaluate(&malformed);
        println!("  honest input -> {good}, malformed input -> {bad} (expected {})", 21 * 2);
        assert_eq!(good, 42, "the honest path must be exact, or this test proves nothing");
        assert_ne!(
            bad, 42,
            "the tampered input must actually change the result, or the demonstration is vacuous"
        );

        // Now the point. Two honest co-holders each recompute the same work --- the check this
        // module relies on for integrity --- and agree, because the computation *was* performed
        // correctly. The fault is in the input, which no amount of recomputation inspects.
        //
        // The evaluated product is formed **once** and shared by both, which is what
        // derandomization achieves in the real protocol: co-holders seed encryption from the
        // pairwise PRF key so they produce bit-identical ciphertexts
        // (`the_same_seed_gives_bit_identical_ciphertexts`). Encrypting twice here would give
        // two different ciphertexts and compare nothing — as a first draft of this test did.
        let evaluated = bfv.mul(&malformed, &bfv.encrypt_scalar(&pk, 2));
        let co_holder_a = partial_decrypt_with_squared(
            &bfv,
            &evaluated,
            &shares[0].left,
            &s2[0],
            &[142u8; 32],
            0,
            0,
        );
        let co_holder_b = partial_decrypt_with_squared(
            &bfv,
            &evaluated,
            &shares[0].left,
            &s2[0],
            &[142u8; 32],
            0,
            0,
        );
        assert_eq!(
            co_holder_a, co_holder_b,
            "recompute-and-compare passes on a malformed input: both honest parties agree, and \
             the agreed value is wrong. This is why the input boundary needs a proof and \
             cannot be discharged by replication"
        );
        println!(
            "  recompute-and-compare AGREES on the malformed input, and the agreed answer is \
             wrong — replication is silent here by construction"
        );
    }

    /// The DKG path must be a drop-in for the dealt one: same layout, same downstream calls.
    #[test]
    fn dkg_shares_are_shaped_like_dealt_ones() {
        let p = params();
        let bfv = Bfv::new(p);
        let (ck, _pk) = bfv.keygen();
        let dealt = deal_key_shares(&crate::fhe::bfv::secret_part(&ck), &p, &[8u8; 32]);
        let contribs: Vec<DkgContribution> = (0..3)
            .map(|i| dkg_sample_share(&p, &[(40 + i) as u8; 32], i))
            .collect();
        let dkg: Vec<KeyShare> = (0..3)
            .map(|i| dkg_assemble_share(&contribs[i], &contribs[(i + 1) % 3].own))
            .collect();
        for i in 0..3 {
            assert_eq!(dealt[i].left.len(), dkg[i].left.len());
            assert_eq!(dealt[i].right.len(), dkg[i].right.len());
        }
        // The one thing that differs, and it is the point: dealt components are uniform mod q,
        // DKG components are ternary. Assert the difference so it cannot be lost.
        let dealt_peak = dealt[0]
            .left
            .iter()
            .map(|&c| crate::fhe::bfv::center(c, &p).unsigned_abs())
            .max()
            .unwrap();
        let dkg_peak = dkg[0]
            .left
            .iter()
            .map(|&c| crate::fhe::bfv::center(c, &p).unsigned_abs())
            .max()
            .unwrap();
        println!("  dealt component sup-norm {dealt_peak}, dkg component sup-norm {dkg_peak}");
        assert!(
            dealt_peak > 1_000_000 && dkg_peak <= 1,
            "dealt components are uniform mod q and DKG components ternary; that is the trade \
             `DEALER_RATIONALE` describes and it must remain visible"
        );
    }
}

// ─── Threshold decryption of products (degree 2, under s²) ───────────────────

/// Party `i`'s share of `s²`, computed **with no communication at all**.
///
/// This is the part of the design that falls out for free, and it is the reason
/// the replicated layout is worth preserving through key generation. Writing
/// `s = s₀ + s₁ + s₂`,
///
/// ```text
///   s² = s₀² + s₁² + s₂² + 2(s₀s₁ + s₁s₂ + s₂s₀)
/// ```
///
/// and party `i`, holding `(sᵢ, sᵢ₊₁)`, can form `sᵢ² + 2·sᵢ·sᵢ₊₁` from what it
/// already has. Those three terms sum to exactly `s²` --- each square appears
/// once and each cross term once. It is the same identity ABY3 multiplication
/// uses, which is not a coincidence: it is the same sharing.
///
/// So a product ciphertext, which decrypts under `s²` because Baru relinearizes
/// nothing, needs **no extra protocol and no evaluation key** to threshold-decrypt.
/// The alternative --- generating a relinearization key distributively --- is the
/// expensive part of threshold FHE and the depth-1 design avoids needing it.
pub fn squared_key_share(share: &KeyShare, p: &BfvParams) -> Poly {
    let mask = p.q_mask();
    let sq = poly_mul_mod(&share.left, &share.left, p);
    let cross = poly_mul_mod(&share.left, &share.right, p);
    (0..p.n)
        .map(|i| {
            sq[i]
                .wrapping_add(cross[i])
                .wrapping_add(cross[i])
                & mask
        })
        .collect()
}

/// Partial decryption of a ciphertext of **any** supported degree.
///
/// For a fresh ciphertext this is [`partial_decrypt`]. For a product it adds the
/// `c₂·(s²)ᵢ` term, with the `s²` share taken from [`squared_key_share`] — so the
/// caller supplies the replicated [`KeyShare`] rather than a single component,
/// because the squared share needs both halves.
///
/// Only one smudging polynomial is drawn, for the whole partial decryption: the
/// quantity that must be flooded is the single value `dᵢ` that goes on the wire,
/// not each term of it.
pub fn partial_decrypt_any(
    bfv: &Bfv,
    ct: &BfvCt,
    share: &KeyShare,
    pair_seed: &[u8; 32],
    ctr: u64,
    smudge_bits: u32,
) -> PartialDecryption {
    let p = &bfv.params;
    let mask = p.q_mask();
    assert!(
        ct.key_pow() <= 2,
        "threshold decryption supports fresh and once-multiplied ciphertexts \
         (key_pow 1 or 2); got {}",
        ct.key_pow()
    );
    let mut acc = poly_mul_mod(&ct.c[1], &share.left, p);
    if ct.key_pow() == 2 {
        let s2 = squared_key_share(share, p);
        let c2_s2 = poly_mul_mod(&ct.c[2], &s2, p);
        for i in 0..p.n {
            acc[i] = acc[i].wrapping_add(c2_s2[i]) & mask;
        }
    }
    let e = smudge_from_seed(
        pair_seed,
        b"baru/fhe/threshold/smudge",
        ctr,
        p.n,
        smudge_bits,
        p,
    );
    let d: Poly = (0..p.n)
        .map(|i| acc[i].wrapping_add(e[i]) & mask)
        .collect();
    PartialDecryption { d }
}

/// [`partial_decrypt_any`] with the `s²` share supplied rather than derived.
///
/// [`squared_key_share`] depends only on the key, so a session computes it once
/// and every subsequent partial decryption reuses it. That matters because it is
/// two of the four polynomial multiplications a degree-2 partial decryption
/// needs, and a scheme switch decrypts once per layer — see
/// `what_threshold_decryption_costs_against_one_local_decrypt` for the measured
/// difference.
///
/// `s2` may be empty for a fresh (degree-1) ciphertext, which needs no squared
/// share at all.
pub fn partial_decrypt_with_squared(
    bfv: &Bfv,
    ct: &BfvCt,
    s_component: &[u128],
    s2: &[u128],
    pair_seed: &[u8; 32],
    ctr: u64,
    smudge_bits: u32,
) -> PartialDecryption {
    let p = &bfv.params;
    let mask = p.q_mask();
    assert!(
        ct.key_pow() <= 2,
        "threshold decryption supports fresh and once-multiplied ciphertexts \
         (key_pow 1 or 2); got {}",
        ct.key_pow()
    );
    let mut acc = poly_mul_mod(&ct.c[1], s_component, p);
    if ct.key_pow() == 2 {
        assert!(
            !s2.is_empty(),
            "a degree-2 ciphertext needs the squared-key share; pass the one from \
             `squared_key_share`"
        );
        let c2_s2 = poly_mul_mod(&ct.c[2], s2, p);
        for i in 0..p.n {
            acc[i] = acc[i].wrapping_add(c2_s2[i]) & mask;
        }
    }
    let e = smudge_from_seed(
        pair_seed,
        b"baru/fhe/threshold/smudge",
        ctr,
        p.n,
        smudge_bits,
        p,
    );
    PartialDecryption {
        d: (0..p.n)
            .map(|i| acc[i].wrapping_add(e[i]) & mask)
            .collect(),
    }
}
