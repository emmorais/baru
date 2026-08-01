//! The input-boundary proof: a zero-knowledge proof that a ciphertext is a
//! **well-formed encryption with bounded noise**.
//!
//! This is the component [`super::boundary`] specifies and
//! `docs/malicious-hybrid.md` §4.4 calls the only place zero knowledge is
//! needed. Replication secures every *evaluation* step
//! ([`super::threshold`]), but it cannot vouch for a ciphertext arriving from
//! outside: both honest co-holders compute on a malformed input faithfully and
//! agree. A proof is the only way to close that.
//!
//! # The protocol
//!
//! A Schnorr-style Σ-protocol over the ring, made non-interactive by
//! Fiat–Shamir, with **rejection sampling** for zero knowledge — the standard
//! Lyubashevsky shape.
//!
//! For public key `(a, b)` and ciphertext `(c₀, c₁)`, the prover knows
//! `(m, u, e₁, e₂)` with `‖u‖∞, ‖e₁‖∞, ‖e₂‖∞ ≤ B` and
//!
//! ```text
//! c₀ = u·b + e₁ + Δ·m   (mod q)
//! c₁ = u·a + e₂         (mod q)
//! ```
//!
//! Per repetition: sample masks `(yᵤ, y₁, y₂, y_m)` with coefficients uniform
//! in `[−M, M]`, commit to `t₀ = yᵤ·b + y₁ + Δ·y_m` and `t₁ = yᵤ·a + y₂`,
//! derive a challenge bit from `H(pk ‖ ct ‖ all commitments)`, and respond
//! `z = y + c·w`.
//!
//! # Why the responses are rejected rather than flooded
//!
//! A response is released only if `‖z‖∞ ≤ M − B`. Conditioned on that, `z` is
//! uniform on `[−(M−B), M−B]` **whatever the witness was** — for `c = 0`
//! trivially, and for `c = 1` because the conditioning window sits strictly
//! inside the shifted support. That gives *perfect* zero knowledge for the
//! released transcript, with no statistical slack.
//!
//! The alternative — masking with noise superpolynomially larger than the
//! witness — would also hide it, but at a cost this construction cannot pay:
//! the mask width sets the extractor's slack, and [`super::boundary`] shows
//! **each bit of slack is subtracted from the threshold decryption's
//! statistical security**. Flooding would blow the budget outright. Rejection
//! sampling keeps `M ≈ 3n·B`, so the slack stays around
//! `log₂(2M)` bits, which is what makes the whole thing fit.
//!
//! # Soundness
//!
//! Two accepting transcripts that share commitments but differ in a challenge
//! bit yield, by subtraction, a witness satisfying the relation with norms
//! bounded by `2(M − B)`. So this proves knowledge of an opening whose noise is
//! at most that — a **relaxed** proof, and the relaxation is exactly the slack
//! [`super::boundary::achievable_lambda`] charges against the smudging budget.
//! A single binary challenge is guessable with probability ½, so repetitions
//! are run in parallel; `reps` sets the soundness error at `2^-reps`.

use sha2::{Digest, Sha256};

use super::bfv::{center, poly_mul_mod, BfvParams, BfvCt, Poly};
use super::boundary::WitnessBound;

/// Honest encryption randomness is ternary, so every witness coefficient is in
/// `{−1, 0, 1}`.
pub const HONEST_BOUND: i128 = 1;

/// Minimum repetitions, and **not** merely a soundness-error parameter.
///
/// With binary challenges the naive reading is "soundness error `2^-reps`, so
/// pick `reps` to taste". That is wrong here, and the way it is wrong is worth
/// stating because an implementation passes every other test while having it:
///
/// **The prover's retry-on-rejection loop is a Fiat–Shamir grinding oracle.**
/// A rejected transcript is discarded and the masks resampled, which resamples
/// the commitments and therefore the challenges. If the challenge vector ever
/// comes out all-zero, every response is a pure mask — no witness is ever
/// added, nothing is rejected, and the transcript verifies while proving
/// nothing at all. An honest prover reaches that state *by accident* in about
/// `2^reps` attempts, and a malicious one aims for it.
///
/// At `reps = 8` this was reproducible in seconds: the loop found an all-zero
/// challenge and happily "proved" a ciphertext whose noise was `2^40` times the
/// permitted bound.
///
/// So `reps` is a **grinding-resistance** parameter, and must be set to the
/// security level rather than to a comfortable soundness error. 128 makes the
/// attack cost `2^128` while an honest prover still succeeds on its first or
/// second attempt.
pub const MIN_REPS: usize = 128;

/// Mask half-width factor, per coefficient *per repetition*.
///
/// Sizing this correctly is the difference between a proof that terminates and
/// one that does not. Rejection is per coefficient with probability ≈ `B/M`,
/// and a transcript is released only if **every** coefficient of **every**
/// repetition survives — the Fiat–Shamir challenges depend on all commitments
/// at once, so a rejection anywhere forces the whole proof to restart. Expected
/// rejections are therefore `3n · reps · B / M`, and a first attempt at
/// `M = 3n·B` gave an acceptance rate of `0.37^reps`: for 8 repetitions, about
/// one attempt in three thousand.
///
/// `M = 30·n·reps·B` keeps overall acceptance near 90%, at the cost of a slack
/// of roughly `log₂(2M)` bits. That trade is the one
/// [`super::boundary::achievable_lambda`] prices.
pub const MASK_WIDTH_FACTOR: i128 = 30;

/// The witness a prover holds. Never serialized — it is the secret.
pub struct Opening {
    pub m: Vec<u64>,
    pub u: Poly,
    pub e1: Poly,
    pub e2: Poly,
}

/// A non-interactive proof of well-formedness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WellFormedProof {
    /// Challenge bits, one per repetition. Fiat–Shamir binds them to the
    /// commitments, so the verifier recomputes and compares rather than
    /// trusting them.
    pub challenges: Vec<bool>,
    /// `(zᵤ, z₁, z₂, z_m)` per repetition, centred.
    pub responses: Vec<[Vec<i128>; 4]>,
}

impl WellFormedProof {
    /// Serialized size, for reporting the boundary's cost honestly.
    pub fn bytes(&self) -> usize {
        self.challenges.len().div_ceil(8)
            + self
                .responses
                .iter()
                .map(|r| r.iter().map(|v| v.len() * 8).sum::<usize>())
                .sum::<usize>()
    }

    /// Extractor slack in bits: the factor by which a proven opening's norm may
    /// exceed the honest one. Feed this to
    /// [`super::boundary::achievable_lambda`].
    pub fn slack_bits(n: usize, bound: i128, reps: usize) -> u32 {
        let m = mask_width(n, bound, reps);
        let extracted = 2 * (m - bound);
        (128 - (extracted.max(1) as u128).leading_zeros()).max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    /// A response exceeded the norm bound — the proof does not establish a
    /// bounded-noise opening.
    ResponseOutOfRange { repetition: usize },
    /// The recomputed commitments do not hash to the challenges, so the
    /// transcript was not produced by this statement.
    ChallengeMismatch,
    Malformed(&'static str),
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofError::ResponseOutOfRange { repetition } => {
                write!(f, "response {repetition} exceeds the norm bound")
            }
            ProofError::ChallengeMismatch => f.write_str("Fiat-Shamir challenge mismatch"),
            ProofError::Malformed(m) => write!(f, "malformed proof: {m}"),
        }
    }
}

impl std::error::Error for ProofError {}

fn mask_width(n: usize, bound: i128, reps: usize) -> i128 {
    MASK_WIDTH_FACTOR * n as i128 * reps.max(1) as i128 * bound
}

/// Reduce a centred integer into `[0, q)`.
fn to_q(v: i128, mask: u128) -> u128 {
    if v < 0 {
        (0u128.wrapping_sub((-v) as u128)) & mask
    } else {
        (v as u128) & mask
    }
}

fn centred(p: &[u128], params: &BfvParams) -> Vec<i128> {
    p.iter().map(|&c| center(c, params)).collect()
}

fn from_centred(v: &[i128], mask: u128) -> Poly {
    v.iter().map(|&c| to_q(c, mask)).collect()
}

/// Absorb the statement and the commitments, then squeeze one challenge bit per
/// repetition.
fn challenges_from(
    pk_b: &[u128],
    pk_a: &[u128],
    ct: &BfvCt,
    commitments: &[(Poly, Poly)],
    reps: usize,
) -> Vec<bool> {
    let mut h = Sha256::new();
    h.update(b"baru/fhe/boundary/v1");
    for poly in [pk_b, pk_a] {
        for c in poly {
            h.update(c.to_le_bytes());
        }
    }
    for poly in &ct.c {
        for c in poly {
            h.update(c.to_le_bytes());
        }
    }
    for (t0, t1) in commitments {
        for poly in [t0, t1] {
            for c in poly {
                h.update(c.to_le_bytes());
            }
        }
    }
    let mut out = Vec::with_capacity(reps);
    let mut counter = 0u64;
    while out.len() < reps {
        let mut hh = h.clone();
        hh.update(counter.to_le_bytes());
        for byte in hh.finalize() {
            for bit in 0..8 {
                if out.len() == reps {
                    break;
                }
                out.push((byte >> bit) & 1 == 1);
            }
        }
        counter += 1;
    }
    out
}

/// Recompute the commitment pair a response implies:
/// `t₀ = zᵤ·b + z₁ + Δ·z_m − c·c₀`, `t₁ = zᵤ·a + z₂ − c·c₁`.
///
/// Shared by prover and verifier so the two cannot drift apart.
fn commitment_from_response(
    params: &BfvParams,
    pk_b: &[u128],
    pk_a: &[u128],
    ct: &BfvCt,
    z: &[Vec<i128>; 4],
    challenge: bool,
    delta: u128,
) -> (Poly, Poly) {
    let mask = params.q_mask();
    let zu = from_centred(&z[0], mask);
    let z1 = from_centred(&z[1], mask);
    let z2 = from_centred(&z[2], mask);
    let zm = from_centred(&z[3], mask);

    let zub = poly_mul_mod(&zu, pk_b, params);
    let zua = poly_mul_mod(&zu, pk_a, params);

    let t0: Poly = (0..params.n)
        .map(|i| {
            let mut v = zub[i]
                .wrapping_add(z1[i])
                .wrapping_add(zm[i].wrapping_mul(delta));
            if challenge {
                v = v.wrapping_sub(ct.c[0][i]);
            }
            v & mask
        })
        .collect();
    let t1: Poly = (0..params.n)
        .map(|i| {
            let mut v = zua[i].wrapping_add(z2[i]);
            if challenge {
                v = v.wrapping_sub(ct.c[1][i]);
            }
            v & mask
        })
        .collect();
    (t0, t1)
}

/// Produce a proof that `ct` is a well-formed encryption under `(pk_b, pk_a)`.
///
/// Retries internally when rejection sampling discards a transcript. The retry
/// count depends only on public norms, not on the witness, so it leaks nothing.
pub fn prove(
    params: &BfvParams,
    pk_b: &[u128],
    pk_a: &[u128],
    ct: &BfvCt,
    opening: &Opening,
    reps: usize,
    rng: &mut impl rand::Rng,
) -> WellFormedProof {
    assert!(
        reps >= MIN_REPS,
        "boundary proof needs at least {MIN_REPS} repetitions: fewer lets the \
         retry-on-rejection loop grind the Fiat-Shamir challenge to all-zero, \
         which verifies without proving anything (see MIN_REPS)"
    );
    let mask = params.q_mask();
    let delta = 1u128 << params.delta_shift();
    let n = params.n;
    let bound = HONEST_BOUND;
    let m_width = mask_width(n, bound, reps);
    let accept = m_width - bound;

    let wu = centred(&opening.u, params);
    let w1 = centred(&opening.e1, params);
    let w2 = centred(&opening.e2, params);
    let wm: Vec<i128> = (0..n)
        .map(|i| *opening.m.get(i).unwrap_or(&0) as i128)
        .collect();

    // Retry the whole proof on rejection: a fresh Fiat-Shamir transcript is
    // simpler to argue about than per-repetition resampling, and the expected
    // number of attempts is small.
    loop {
        let mut masks: Vec<[Vec<i128>; 4]> = Vec::with_capacity(reps);
        let mut commitments: Vec<(Poly, Poly)> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let draw = |rng: &mut dyn rand::RngCore| -> Vec<i128> {
                (0..n)
                    .map(|_| rand::Rng::gen_range(rng, -m_width..=m_width))
                    .collect()
            };
            let yu = draw(rng);
            let y1 = draw(rng);
            let y2 = draw(rng);
            let ym = draw(rng);

            let yu_q = from_centred(&yu, mask);
            let yub = poly_mul_mod(&yu_q, pk_b, params);
            let yua = poly_mul_mod(&yu_q, pk_a, params);
            let y1_q = from_centred(&y1, mask);
            let y2_q = from_centred(&y2, mask);
            let ym_q = from_centred(&ym, mask);

            let t0: Poly = (0..n)
                .map(|i| {
                    yub[i]
                        .wrapping_add(y1_q[i])
                        .wrapping_add(ym_q[i].wrapping_mul(delta))
                        & mask
                })
                .collect();
            let t1: Poly = (0..n)
                .map(|i| yua[i].wrapping_add(y2_q[i]) & mask)
                .collect();
            commitments.push((t0, t1));
            masks.push([yu, y1, y2, ym]);
        }

        let challenges = challenges_from(pk_b, pk_a, ct, &commitments, reps);

        let mut responses = Vec::with_capacity(reps);
        let mut rejected = false;
        for (j, c) in challenges.iter().enumerate() {
            let add = |y: &Vec<i128>, w: &Vec<i128>| -> Vec<i128> {
                if *c {
                    y.iter().zip(w).map(|(a, b)| a + b).collect()
                } else {
                    y.clone()
                }
            };
            let zu = add(&masks[j][0], &wu);
            let z1 = add(&masks[j][1], &w1);
            let z2 = add(&masks[j][2], &w2);
            let zm = add(&masks[j][3], &wm);

            // Release only inside the window that makes `z` uniform regardless
            // of the witness. `z_m` is unconstrained: the message is not a
            // small value, and bounding it is not what the relation claims.
            if [&zu, &z1, &z2]
                .iter()
                .any(|v| v.iter().any(|c| c.abs() > accept))
            {
                rejected = true;
                break;
            }
            responses.push([zu, z1, z2, zm]);
        }
        if !rejected {
            return WellFormedProof {
                challenges,
                responses,
            };
        }
    }
}

/// Verify a proof of well-formedness.
///
/// On success the ciphertext is guaranteed to have an opening with noise at
/// most `2(M − B)` — see [`WellFormedProof::slack_bits`], which is the number
/// the parameter analysis in [`super::boundary`] consumes.
pub fn verify(
    params: &BfvParams,
    pk_b: &[u128],
    pk_a: &[u128],
    ct: &BfvCt,
    proof: &WellFormedProof,
) -> Result<(), ProofError> {
    let reps = proof.challenges.len();
    if reps < MIN_REPS || proof.responses.len() != reps {
        // Refusing short proofs is a *verifier* obligation, not just a prover
        // convention: a short proof is grindable, so accepting one would accept
        // a forgery.
        return Err(ProofError::Malformed(
            "too few repetitions, or challenge/response count mismatch",
        ));
    }
    let n = params.n;
    let accept = mask_width(n, HONEST_BOUND, reps) - HONEST_BOUND;
    let delta = 1u128 << params.delta_shift();

    // Norm check first: this is the part that makes the proof mean "bounded
    // noise" rather than merely "some opening exists".
    for (j, z) in proof.responses.iter().enumerate() {
        if z.iter().take(3).any(|v| v.len() != n) || z[3].len() != n {
            return Err(ProofError::Malformed("response length"));
        }
        if z.iter()
            .take(3)
            .any(|v| v.iter().any(|c| c.abs() > accept))
        {
            return Err(ProofError::ResponseOutOfRange { repetition: j });
        }
    }

    let commitments: Vec<(Poly, Poly)> = proof
        .responses
        .iter()
        .zip(&proof.challenges)
        .map(|(z, c)| commitment_from_response(params, pk_b, pk_a, ct, z, *c, delta))
        .collect();

    // Fiat–Shamir: the challenges must be the ones these commitments determine.
    // This is what binds the proof to the statement — without it a prover could
    // pick the challenge after seeing the commitments.
    if challenges_from(pk_b, pk_a, ct, &commitments, reps) != proof.challenges {
        return Err(ProofError::ChallengeMismatch);
    }
    Ok(())
}

/// The bound this construction actually proves, for feeding to
/// [`super::boundary`].
pub fn proven_bound(params: &BfvParams, reps: usize) -> WitnessBound {
    WitnessBound {
        slack_bits: WellFormedProof::slack_bits(params.n, HONEST_BOUND, reps),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::bfv::{public_parts, Bfv};

    fn params() -> BfvParams {
        // Small ring keeps the tests fast; the construction does not depend on
        // the dimension beyond the mask width.
        BfvParams { n: 256, log2_q: 100, log2_t: 20 }
    }

    /// Encrypt while retaining the randomness, so the test holds a real
    /// witness rather than a synthetic one.
    fn encrypt_with_opening(
        bfv: &Bfv,
        pk: &[u8],
        m: &[u64],
        rng: &mut impl rand::Rng,
    ) -> (BfvCt, Opening) {
        let p = &bfv.params;
        let mask = p.q_mask();
        let (b, a) = public_parts(pk);
        let ternary = |rng: &mut dyn rand::RngCore| -> Poly {
            (0..p.n)
                .map(|_| match rand::Rng::gen_range(rng, 0u8..3) {
                    0 => 0u128,
                    1 => 1u128,
                    _ => mask, // -1
                })
                .collect()
        };
        let u = ternary(rng);
        let e1 = ternary(rng);
        let e2 = ternary(rng);
        let ub = poly_mul_mod(&u, &b, p);
        let ua = poly_mul_mod(&u, &a, p);
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
        (
            BfvCt { c: vec![c0, c1] },
            Opening {
                m: m.to_vec(),
                u,
                e1,
                e2,
            },
        )
    }

    #[test]
    fn an_honest_ciphertext_proves_and_verifies() {
        let bfv = Bfv::new(params());
        let (_ck, pk) = bfv.keygen();
        let (b, a) = public_parts(&pk);
        let mut rng = rand::thread_rng();
        let m: Vec<u64> = (0..params().n).map(|i| (i as u64) % 1024).collect();
        let (ct, opening) = encrypt_with_opening(&bfv, &pk, &m, &mut rng);

        let proof = prove(&params(), &b, &a, &ct, &opening, MIN_REPS, &mut rng);
        verify(&params(), &b, &a, &ct, &proof).expect("honest proof must verify");
    }

    /// The proof must be bound to its statement: it must not verify against a
    /// different ciphertext.
    #[test]
    fn a_proof_does_not_transfer_to_another_ciphertext() {
        let bfv = Bfv::new(params());
        let (_ck, pk) = bfv.keygen();
        let (b, a) = public_parts(&pk);
        let mut rng = rand::thread_rng();
        let m1: Vec<u64> = vec![7; params().n];
        let m2: Vec<u64> = vec![9; params().n];
        let (ct1, op1) = encrypt_with_opening(&bfv, &pk, &m1, &mut rng);
        let (ct2, _op2) = encrypt_with_opening(&bfv, &pk, &m2, &mut rng);

        let proof = prove(&params(), &b, &a, &ct1, &op1, MIN_REPS, &mut rng);
        assert_eq!(
            verify(&params(), &b, &a, &ct2, &proof),
            Err(ProofError::ChallengeMismatch),
            "a proof for one ciphertext must not verify for another"
        );
    }

    /// **The case the boundary exists for.** A ciphertext whose noise exceeds
    /// the permitted bound — the thing replication cannot catch — must not be
    /// provable.
    ///
    /// The bad ciphertext is *built from* the inflated opening, so the algebraic
    /// relation holds exactly and only the norm is wrong. That matters: a test
    /// that merely perturbed the ciphertext would be rejected for the wrong
    /// reason and would not exercise the norm check at all.
    #[test]
    fn a_high_noise_ciphertext_cannot_be_proven() {
        let p = params();
        let bfv = Bfv::new(p);
        let (_ck, pk) = bfv.keygen();
        let (b, a) = public_parts(&pk);
        let mask = p.q_mask();
        let mut rng = rand::thread_rng();

        // An opening with one enormous noise coefficient.
        let big: i128 = 1 << 40;
        let m: Vec<u64> = vec![3; p.n];
        let (_honest_ct, mut opening) = encrypt_with_opening(&bfv, &pk, &m, &mut rng);
        opening.e1[0] = to_q(big, mask);

        // Rebuild the ciphertext so c0 = u*b + e1 + delta*m holds exactly.
        let ub = poly_mul_mod(&opening.u, &b, &p);
        let ua = poly_mul_mod(&opening.u, &a, &p);
        let shift = p.delta_shift();
        let t_mask = (p.t() - 1) as u128;
        let c0: Poly = (0..p.n)
            .map(|i| {
                let mi = (m[i] as u128) & t_mask;
                ub[i]
                    .wrapping_add(opening.e1[i])
                    .wrapping_add(mi << shift)
                    & mask
            })
            .collect();
        let c1: Poly = (0..p.n)
            .map(|i| ua[i].wrapping_add(opening.e2[i]) & mask)
            .collect();
        let bad = BfvCt { c: vec![c0, c1] };

        // The prover cannot release a transcript: every challenge bit that is 1
        // pushes the response out of the window, and with MIN_REPS repetitions
        // an all-zero challenge vector is not reachable. Run it under a cap so
        // a regression shows up as a failure rather than a hang.
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let produced = std::thread::spawn(move || {
            let p = params();
            let bfv = Bfv::new(p);
            let _ = &bfv;
            let mut rng = rand::thread_rng();
            let _ = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            prove(&p, &b, &a, &bad, &opening, MIN_REPS, &mut rng)
        });
        // A correct implementation never finishes this; give it a bounded
        // window and treat completion as the failure it would be.
        std::thread::sleep(std::time::Duration::from_secs(3));
        assert!(
            !produced.is_finished(),
            "a ciphertext with out-of-bound noise must not be provable, but the \
             prover produced a transcript — the norm check or the grinding \
             resistance has regressed"
        );
    }

    /// A tampered response must be caught.
    #[test]
    fn a_tampered_response_is_rejected() {
        let bfv = Bfv::new(params());
        let (_ck, pk) = bfv.keygen();
        let (b, a) = public_parts(&pk);
        let mut rng = rand::thread_rng();
        let m: Vec<u64> = vec![11; params().n];
        let (ct, opening) = encrypt_with_opening(&bfv, &pk, &m, &mut rng);
        let mut proof = prove(&params(), &b, &a, &ct, &opening, MIN_REPS, &mut rng);
        proof.responses[0][0][0] += 1;
        assert!(verify(&params(), &b, &a, &ct, &proof).is_err());
    }

    /// Flipping a challenge bit must invalidate the proof — otherwise the
    /// Fiat–Shamir binding would be decorative.
    #[test]
    fn a_flipped_challenge_is_rejected() {
        let bfv = Bfv::new(params());
        let (_ck, pk) = bfv.keygen();
        let (b, a) = public_parts(&pk);
        let mut rng = rand::thread_rng();
        let m: Vec<u64> = vec![5; params().n];
        let (ct, opening) = encrypt_with_opening(&bfv, &pk, &m, &mut rng);
        let mut proof = prove(&params(), &b, &a, &ct, &opening, MIN_REPS, &mut rng);
        proof.challenges[0] = !proof.challenges[0];
        assert_eq!(
            verify(&params(), &b, &a, &ct, &proof),
            Err(ProofError::ChallengeMismatch)
        );
    }

    /// Every released response must lie in the window that makes it uniform
    /// regardless of the witness — the zero-knowledge property, checked on the
    /// transcript the verifier actually sees.
    #[test]
    fn released_responses_are_inside_the_zero_knowledge_window() {
        let bfv = Bfv::new(params());
        let (_ck, pk) = bfv.keygen();
        let (b, a) = public_parts(&pk);
        let mut rng = rand::thread_rng();
        let m: Vec<u64> = vec![1; params().n];
        let (ct, opening) = encrypt_with_opening(&bfv, &pk, &m, &mut rng);
        let proof = prove(&params(), &b, &a, &ct, &opening, MIN_REPS, &mut rng);
        let accept = mask_width(params().n, HONEST_BOUND, MIN_REPS) - HONEST_BOUND;
        for z in &proof.responses {
            for v in z.iter().take(3) {
                assert!(
                    v.iter().all(|c| c.abs() <= accept),
                    "a released response left the zero-knowledge window"
                );
            }
        }
    }

    /// The slack this construction achieves must be small enough to be useful,
    /// and it must be the number the parameter analysis consumes.
    #[test]
    fn the_slack_is_small_enough_to_budget_against() {
        use crate::fhe::boundary::{achievable_lambda, smallest_ring_for};
        let slack = WellFormedProof::slack_bits(4096, HONEST_BOUND, MIN_REPS);
        // ~2*3*4096 = 2^14.6, so around 15 bits.
        assert!(
            (20..=30).contains(&slack),
            "rejection sampling should keep slack in the twenties, got {slack}"
        );

        // At n=4096 that slack eats most of the budget — the finding from
        // `boundary`, now instantiated with a real construction's number.
        assert!(achievable_lambda(&params(), 39, slack) < 40);

        // And a larger ring absorbs it, which is the parameter recommendation.
        let ring = smallest_ring_for(20, 39, slack, 40).expect("some ring works");
        assert!(ring.n >= 8192, "expected n>=8192, got {}", ring.n);
    }
}
