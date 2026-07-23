//! Scheme switching: convert values between MPC secret shares and FHE
//! ciphertexts — in both directions, generically over backends — so a
//! computation can run each of its phases on whichever cryptosystem is
//! cheapest there, Lego-style.
//!
//! # Why (the neural-network motivation)
//!
//! This repo's own measurements make the case concretely:
//! - MPC (`bench-nn`): comparisons/ReLU are *cheap* (~10 ms/element,
//!   real replicated-secret-sharing), but every multiplication costs
//!   communication, and a naive matvec burned tens of thousands of messages
//!   before the batched precompile.
//! - CKKS (`bench-nn`, `bench-ckks-compare`): linear algebra over a whole
//!   vector is *one* SIMD ciphertext op — but a single approximate ReLU costs
//!   seconds in a deep, ring-dimension-inflated context, three to four
//!   orders of magnitude behind MPC.
//!
//! So the obviously-right private-NN pipeline is *heterogeneous*: FHE for the
//! linear layers, MPC for the nonlinearities — which is exactly the
//! architecture of Gazelle (Juvekar, Vaikuntanathan, Chandrakasan, USENIX
//! Security 2018) and its successors (Delphi, Cheetah, …), and the
//! cross-scheme conversion below is the glue those systems are built on.
//! The general "convert between FHE schemes / sharings instead of paying the
//! worst scheme's cost everywhere" idea is CHIMERA (Boura–Gama–Georgieva–
//! Jetchev, <https://eprint.iacr.org/2018/758>); evaluating under a
//! secret-shared FHE key is threshold FHE (Asharov et al., EUROCRYPT 2012,
//! <https://eprint.iacr.org/2011/613>).
//!
//! # The two conversions
//!
//! **MPC → FHE** ([`shares_to_fhe`]): each party encrypts its own additive
//! share component under the FHE public key; the (untrusted) evaluator
//! homomorphically sums the ciphertexts. Since the components' wrapping sum
//! *is* the secret (mod `2^nbits`), the homomorphic sum is an encryption of
//! the secret — and no party ever saw it in the clear. Needs only
//! homomorphic *addition*: no relinearization/rotation keys, works on every
//! backend in this crate including CKKS.
//!
//! **FHE → MPC** ([`fhe_to_shares`]): mask-and-open. Each party `i` samples
//! a uniform mask `r_i (mod 2^nbits)` and submits `Enc(r_i)`; the evaluator
//! computes `Enc(m + Σ rᵢ)` and that *masked* ciphertext is decrypted and
//! published. The published value `y = m + Σ rᵢ (mod 2^nbits)` is uniformly
//! random — it reveals nothing about `m`. Shares: party 0 takes `y − r₀`,
//! every other party takes `−rᵢ`; the wrapping sum telescopes back to `m`.
//!
//! **Re-encryption instead of bootstrapping** ([`mpc_reencrypt`]): compose
//! the two. FHE → shares → fresh FHE ciphertext yields an encryption of the
//! same value with *fresh* noise (and, for levelled CKKS, a restored
//! level/depth budget) — the interactive alternative to bootstrapping.
//! Trade-off, stated honestly: bootstrapping is non-interactive but
//! enormously expensive; this path is one round of interaction plus cheap
//! masking, but it moves trust from "the FHE scheme alone" to "the MPC
//! parties don't collude" (same threshold trust model as the sharing
//! itself, and as Baru's existing trusted-triple-dealer known limitation).
//!
//! # Trust model of this in-process implementation
//!
//! These functions *simulate all parties in one process*, mirroring how
//! `Aby3Backend` simulates the 3-party MPC protocol in-process: the masking
//! and reconstruction data flow is exactly the real protocol's, but the
//! final masked decryption uses the single client key rather than a genuine
//! threshold decryption (where the FHE secret key itself would be
//! secret-shared and each party would contribute a partial decryption).
//! Distributed key generation / partial decryption is future work; the
//! *interfaces* below don't change when it lands, because the masked
//! ciphertext being decrypted is already safe to open publicly.
//!
//! # Generic interfaces (new FHE backend × new MPC backend)
//!
//! The conversion is written against two small capability traits, so any
//! future pairing works without touching this module:
//! - FHE side: [`FheAdapter`] — encrypt/decrypt/add over `u64` residues.
//!   [`IntFhe`] blankets every integer [`FheBackend`] (mock, TFHE, future);
//!   [`RealFhe`] adapts any [`RealFheBackend`] (CKKS real + mock), where
//!   residues cross the boundary as exactly-representable `f64`s.
//! - MPC side: `Backend::export_additive` / `Backend::import_additive`
//!   (see `crate::interpreter::backend`) — raw additive components whose
//!   wrapping sum is the secret. Implemented for `Aby3Backend`; a
//!   per-party transport backend needs a round to agree which replicated
//!   component each party contributes (future work).
//!
//! # Residue-class semantics (caveat)
//!
//! MPC shares live mod `2^nbits`; FHE integer ciphertexts live mod `2^64`.
//! A converted ciphertext encrypts `m + k·2^nbits` for a small carry `k <
//! n_parties` — the *residue class* of `m`, not its canonical
//! representative. `+`, `−`, `×` all commute with the residue map, so
//! arithmetic keeps working, and every decrypt path here reduces mod
//! `2^nbits` — but an operation that inspects the raw representative
//! (comparison, PBS lookup) would see the carry. Normalising inside FHE
//! would itself cost a comparison, so it is deliberately *not* done;
//! convert back to MPC if you need comparisons — that's the whole point.

use rand::Rng;

use crate::crypto::types::Fr;
use crate::fhe::backend::{FheBackend, RealFheBackend};
use crate::interpreter::backend::{Backend, ShareRef};

// ─── FHE capability trait ─────────────────────────────────────────────────────

/// The minimal FHE surface scheme switching needs: encrypt/decrypt `u64`
/// residues and add ciphertexts. Deliberately tiny so that *every* FHE
/// backend — integer or approximate-real — can offer it.
pub trait FheAdapter {
    type Ct: Clone;
    fn encrypt(&self, client_key: &[u8], value: u64) -> Self::Ct;
    /// Decrypt to a `u64`. May carry share-sum carries above `2^nbits`;
    /// callers reduce (see module docs on residue-class semantics).
    fn decrypt(&self, client_key: &[u8], ct: &Self::Ct) -> u64;
    fn add(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct;
    fn backend_id(&self) -> &'static str;
    /// Serialized size of one ciphertext — what a conversion actually puts
    /// on the wire (used by benchmark link-cost models).
    fn ct_bytes(&self, ct: &Self::Ct) -> usize;
}

/// Adapter over any integer [`FheBackend`] (mock, TFHE, future backends).
pub struct IntFhe<'a, F: FheBackend>(pub &'a F);

impl<F: FheBackend> FheAdapter for IntFhe<'_, F> {
    type Ct = F::Ct;
    fn encrypt(&self, client_key: &[u8], value: u64) -> F::Ct {
        self.0.encrypt(client_key, value)
    }
    fn decrypt(&self, client_key: &[u8], ct: &F::Ct) -> u64 {
        self.0.decrypt(client_key, ct)
    }
    fn add(&self, a: &F::Ct, b: &F::Ct) -> F::Ct {
        self.0.add(a, b)
    }
    fn backend_id(&self) -> &'static str {
        self.0.backend_id()
    }
    fn ct_bytes(&self, ct: &F::Ct) -> usize {
        self.0.serialize_ct(ct).len()
    }
}

/// Adapter over any approximate-real [`RealFheBackend`] (CKKS + its mock).
///
/// Residues cross the boundary as `f64`s: components are `< 2^nbits ≤ 2^32`
/// and at most a handful are summed, so every intermediate stays far below
/// `2^53` and is exactly representable; CKKS's own noise (≪ 0.5 at these
/// magnitudes under addition-only circuits) is killed by the final
/// `.round()`. Conversion needs no relin/galois keys — additions only.
///
/// Caveat: *multiplying* converted residues under a real-arithmetic backend
/// is only exact while magnitudes stay ≪ 2^53 — a negative value's residue
/// is near `2^nbits`, so its square is near `2^64` and loses low bits in
/// `f64` (and drowns in CKKS noise). Additions are always fine; for
/// products of possibly-negative values, either keep them in an integer
/// backend or convert back to MPC — which is the point of this module.
pub struct RealFhe<'a, B: RealFheBackend>(pub &'a B);

impl<B: RealFheBackend> FheAdapter for RealFhe<'_, B> {
    type Ct = B::Ct;
    fn encrypt(&self, client_key: &[u8], value: u64) -> B::Ct {
        self.0.encrypt(client_key, value as f64)
    }
    fn decrypt(&self, client_key: &[u8], ct: &B::Ct) -> u64 {
        self.0.decrypt(client_key, ct).round() as u64
    }
    fn add(&self, a: &B::Ct, b: &B::Ct) -> B::Ct {
        self.0.add(a, b)
    }
    fn backend_id(&self) -> &'static str {
        self.0.backend_id()
    }
    fn ct_bytes(&self, ct: &B::Ct) -> usize {
        self.0.serialize_ct(ct).len()
    }
}

// ─── Residue helpers ──────────────────────────────────────────────────────────

fn reduce(v: u128, nbits: usize) -> u64 {
    if nbits >= 64 {
        v as u64
    } else {
        (v & ((1u128 << nbits) - 1)) as u64
    }
}

fn reduce64(v: u64, nbits: usize) -> u64 {
    if nbits >= 64 {
        v
    } else {
        v & ((1u64 << nbits) - 1)
    }
}

// ─── Core conversions ─────────────────────────────────────────────────────────

/// MPC → FHE: each party encrypts its additive component; homomorphically
/// sum. Returns an encryption of the secret's residue class mod `2^nbits`
/// (representative may carry `+ k·2^nbits`, `k < components.len()` — see
/// module docs). No party's component leaves its "hands" unencrypted.
pub fn shares_to_fhe<A: FheAdapter>(
    fhe: &A,
    client_key: &[u8],
    components: &[Fr],
    nbits: usize,
) -> A::Ct {
    assert!(!components.is_empty(), "shares_to_fhe: no components");
    let mut cts = components
        .iter()
        .map(|&c| fhe.encrypt(client_key, reduce(c, nbits)));
    let first = cts.next().unwrap();
    cts.fold(first, |acc, ct| fhe.add(&acc, &ct))
}

/// FHE → MPC: mask-and-open. Returns `n_parties` additive components whose
/// wrapping sum ≡ the plaintext mod `2^nbits`. The only value ever
/// decrypted is the *masked* ciphertext, whose plaintext is uniformly
/// random mod `2^nbits` — safe to publish (in a real deployment this is the
/// value a threshold decryption would open publicly; see module docs for
/// the trust model of the single-key stand-in used here).
pub fn fhe_to_shares<A: FheAdapter>(
    fhe: &A,
    client_key: &[u8],
    ct: &A::Ct,
    n_parties: usize,
    nbits: usize,
) -> Vec<Fr> {
    assert!(n_parties >= 2, "fhe_to_shares: need at least 2 parties");
    let mut rng = rand::thread_rng();
    // Party i samples r_i and contributes Enc(r_i).
    let masks: Vec<u64> = (0..n_parties)
        .map(|_| reduce64(rng.gen::<u64>(), nbits))
        .collect();
    let mut masked = ct.clone();
    for &r in &masks {
        let enc_r = fhe.encrypt(client_key, r);
        masked = fhe.add(&masked, &enc_r);
    }
    // Open the masked value (uniform mod 2^nbits — reveals nothing).
    let y = reduce64(fhe.decrypt(client_key, &masked), nbits);
    // Party 0: y - r_0.  Party i>0: -r_i.  Wrapping sum telescopes to m.
    let mut components: Vec<Fr> = Vec::with_capacity(n_parties);
    components.push((y as u128).wrapping_sub(masks[0] as u128));
    for &r in &masks[1..] {
        components.push(0u128.wrapping_sub(r as u128));
    }
    components
}

/// Refresh a ciphertext *without bootstrapping*: FHE → shares → fresh FHE.
/// The output encrypts the same residue class with fresh encryption noise
/// (and, for levelled schemes like CKKS, a full level budget again) at the
/// cost of one interactive mask-and-open round. See module docs for the
/// bootstrapping-vs-interaction trade-off and trust model.
pub fn mpc_reencrypt<A: FheAdapter>(
    fhe: &A,
    client_key: &[u8],
    ct: &A::Ct,
    n_parties: usize,
    nbits: usize,
) -> A::Ct {
    let components = fhe_to_shares(fhe, client_key, ct, n_parties, nbits);
    shares_to_fhe(fhe, client_key, &components, nbits)
}

// ─── Bridges to a live MPC backend ────────────────────────────────────────────

/// Convert a live MPC share into an FHE ciphertext of the same value
/// (mod `2^s.nbits`). The share stays secret throughout: only its encrypted
/// components cross the boundary.
pub fn mpc_share_to_fhe<A: FheAdapter>(
    be: &mut dyn Backend,
    s: &ShareRef,
    fhe: &A,
    client_key: &[u8],
) -> A::Ct {
    let components = be.export_additive(s);
    shares_to_fhe(fhe, client_key, &components, s.nbits)
}

/// Convert an FHE ciphertext into a live MPC share of the same value
/// (mod `2^nbits`), via mask-and-open.
pub fn fhe_to_mpc_share<A: FheAdapter>(
    fhe: &A,
    client_key: &[u8],
    ct: &A::Ct,
    be: &mut dyn Backend,
    nbits: usize,
) -> ShareRef {
    let components = fhe_to_shares(fhe, client_key, ct, 3, nbits);
    be.import_additive(&components, nbits)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::{MockCkksBackend, MockFheBackend};
    use crate::interpreter::aby3_backend::Aby3Backend;

    const NBITS: usize = 32;

    fn signed_residue(v: i64) -> u128 {
        (v as i128).rem_euclid(1i128 << NBITS) as u128
    }

    fn to_signed(v: u64) -> i64 {
        (v as u32) as i32 as i64
    }

    #[test]
    fn mpc_share_roundtrips_through_fhe_mock() {
        let fhe = MockFheBackend;
        let adapter = IntFhe(&fhe);
        let (ck, _) = fhe.keygen();
        for &v in &[0i64, 1, 42, -1, -12345, i32::MAX as i64, i32::MIN as i64] {
            let mut be = Aby3Backend::new();
            let s = be.share_secret(signed_residue(v), NBITS, 0);
            let ct = mpc_share_to_fhe(&mut be, &s, &adapter, &ck);
            let got = to_signed(reduce64(adapter.decrypt(&ck, &ct), NBITS));
            assert_eq!(got, v, "share->fhe->decrypt for {v}");
        }
    }

    #[test]
    fn fhe_ciphertext_roundtrips_through_mpc_shares() {
        let fhe = MockFheBackend;
        let adapter = IntFhe(&fhe);
        let (ck, _) = fhe.keygen();
        for &v in &[0i64, 7, -7, 99999, -99999] {
            let mut be = Aby3Backend::new();
            let ct = adapter.encrypt(&ck, signed_residue(v) as u64);
            let s = fhe_to_mpc_share(&adapter, &ck, &ct, &mut be, NBITS);
            let got = to_signed(reduce64(be.reveal(&s, None) as u64, NBITS));
            assert_eq!(got, v, "fhe->share->reveal for {v}");
        }
    }

    #[test]
    fn masked_opening_hides_the_plaintext() {
        // The only decrypted value in fhe_to_shares is the masked one; the
        // components handed to each party must not individually equal the
        // secret, and two runs must produce different (random) components.
        let fhe = MockFheBackend;
        let adapter = IntFhe(&fhe);
        let (ck, _) = fhe.keygen();
        let secret = 424242u64;
        let ct = adapter.encrypt(&ck, secret);
        let a = fhe_to_shares(&adapter, &ck, &ct, 3, NBITS);
        let b = fhe_to_shares(&adapter, &ck, &ct, 3, NBITS);
        assert_ne!(a, b, "two conversions must use fresh masks");
        // Sum reconstructs; no single component is the secret (overwhelmingly
        // likely with uniform masks — a flaky failure here means the masking
        // is broken, not bad luck).
        let sum = a.iter().fold(0u128, |acc, &c| acc.wrapping_add(c));
        assert_eq!(reduce(sum, NBITS), secret);
        for c in &a {
            assert_ne!(reduce(*c, NBITS), secret);
        }
    }

    #[test]
    fn arithmetic_still_works_after_conversion_mod_2_nbits() {
        // The residue-class claim from the module docs: +, ×, − on a
        // converted ciphertext (which may carry k·2^nbits) stay correct
        // modulo 2^nbits.
        let fhe = MockFheBackend;
        let adapter = IntFhe(&fhe);
        let (ck, _) = fhe.keygen();
        let mut be = Aby3Backend::new();
        let s = be.share_secret(signed_residue(-5), NBITS, 0);
        let ct = mpc_share_to_fhe(&mut be, &s, &adapter, &ck);
        // (-5 * 3) + 7 = -8, computed on the carried representative.
        let prod = fhe.mul_plain(&ct, 3);
        let out = fhe.add_plain(&prod, 7);
        let got = to_signed(reduce64(adapter.decrypt(&ck, &out), NBITS));
        assert_eq!(got, -8);
    }

    #[test]
    fn mpc_reencrypt_preserves_the_value() {
        let fhe = MockFheBackend;
        let adapter = IntFhe(&fhe);
        let (ck, _) = fhe.keygen();
        let ct = adapter.encrypt(&ck, signed_residue(-1234) as u64);
        let fresh = mpc_reencrypt(&adapter, &ck, &ct, 3, NBITS);
        assert_eq!(
            to_signed(reduce64(adapter.decrypt(&ck, &fresh), NBITS)),
            -1234
        );
    }

    #[test]
    fn ckks_mock_shares_roundtrip() {
        // The RealFhe adapter: same conversions, approximate-real backend.
        let fhe = MockCkksBackend;
        let adapter = RealFhe(&fhe);
        let (ck, _) = RealFheBackend::keygen(&fhe);
        let mut be = Aby3Backend::new();
        let s = be.share_secret(signed_residue(31337), NBITS, 0);
        let ct = mpc_share_to_fhe(&mut be, &s, &adapter, &ck);
        assert_eq!(reduce64(adapter.decrypt(&ck, &ct), NBITS), 31337);

        let mut be2 = Aby3Backend::new();
        let s2 = fhe_to_mpc_share(&adapter, &ck, &ct, &mut be2, NBITS);
        assert_eq!(reduce64(be2.reveal(&s2, None) as u64, NBITS), 31337);
    }

    #[test]
    fn gazelle_style_pipeline_fhe_linear_then_mpc_relu() {
        // The heterogeneous private-NN pattern this module exists for
        // (Gazelle, USENIX Security 2018): the *linear* layer runs under FHE
        // (public weights, mul_plain/add_plain — cheap there), then the
        // result switches schemes and the *nonlinearity* runs under MPC
        // (~10ms/element vs seconds under CKKS, per this repo's own
        // bench-nn / bench-ckks-compare measurements).
        let fhe = MockFheBackend;
        let adapter = IntFhe(&fhe);
        let (ck, _) = fhe.keygen();

        let weights: [i64; 3] = [2, -3, 1];
        let inputs: [i64; 3] = [4, 5, -6];
        let bias = 1i64;
        // plain: 8 - 15 - 6 + 1 = -12 -> relu = 0
        let expected_pre = weights
            .iter()
            .zip(&inputs)
            .map(|(w, x)| w * x)
            .sum::<i64>()
            + bias;
        let expected = expected_pre.max(0);

        // Phase 1 — FHE linear layer over the encrypted input vector.
        let cts: Vec<u64> = inputs
            .iter()
            .map(|&x| adapter.encrypt(&ck, signed_residue(x) as u64))
            .collect();
        let mut acc = fhe.mul_plain(&cts[0], weights[0]);
        for (ct, &w) in cts[1..].iter().zip(&weights[1..]) {
            acc = fhe.add(&acc, &fhe.mul_plain(ct, w));
        }
        acc = fhe.add_plain(&acc, bias);

        // Phase 2 — switch schemes: FHE ciphertext -> MPC share.
        let mut be = Aby3Backend::new();
        let z = fhe_to_mpc_share(&adapter, &ck, &acc, &mut be, NBITS);

        // Phase 3 — MPC ReLU (mux style, same as bench-nn's Layer arm).
        let zero = be.sub(&z, &z);
        let mask = be.lt(&zero, &z);
        let relu = be.mul(&z, &mask);

        // Phase 4 — and back to FHE, e.g. for the next linear layer.
        let ct_out = mpc_share_to_fhe(&mut be, &relu, &adapter, &ck);
        let got = to_signed(reduce64(adapter.decrypt(&ck, &ct_out), NBITS));
        assert_eq!(got, expected, "pre-activation {expected_pre}");
    }
}
