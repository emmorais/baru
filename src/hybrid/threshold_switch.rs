//! The scheme switch under a **threshold** key, with every randomized step
//! derandomized so it can be checked by replication.
//!
//! [`super`]'s switch is correct but simulates all parties in one process and
//! decrypts with a single client key. This module replaces both halves with the
//! real thing: the FHE secret key is replicated-shared across the three parties,
//! decryption is a sum of partial decryptions, and every value that used to be
//! sampled freshly is now derived from a pairwise PRF key.
//!
//! # Which seed, and why it has to be that one
//!
//! Replicated sharing puts component `j` in the hands of exactly `P_j` (as
//! `left`) and `P_{j−1}` (as `right`). Those two parties — and only those two —
//! share the pairwise PRF key `k_{j−1,j}`. So the rule is forced:
//!
//! ```text
//! component me      → the key shared with my predecessor   (PrfKeys::prev)
//! component me + 1  → the key shared with my successor      (PrfKeys::mine)
//! ```
//!
//! Both holders then derive **identical** randomness for that component, so
//! their ciphertexts and partial decryptions are bit-identical and any
//! deviation shows up as a mismatch. The third party holds neither and learns
//! nothing — which is what keeps the derandomization private rather than merely
//! reproducible.
//!
//! # What is checked, and what still is not
//!
//! Checked here: encryption of each share component (MPC→FHE), each mask
//! contribution, and each partial decryption (FHE→MPC). The homomorphic sum in
//! between is deterministic, so any party holding the inputs recomputes it and
//! compares a digest.
//!
//! Not checked, and not checkable this way: a **malformed ciphertext arriving
//! from outside**. Both honest co-holders will compute on it faithfully and
//! agree, so replication cannot see the problem. That is the boundary proof
//! obligation recorded in [`crate::fhe::threshold::BOUNDARY_GAP`], and it is
//! the reason nothing here attests malicious security yet.

use crate::crypto::net::PrfKeys;
use crate::crypto::types::Fr;
use crate::fhe::bfv::{Bfv, BfvCt};
use crate::fhe::threshold::{
    combine, ct_digest, encrypt_derandomized, partial_decrypt, KeyShare, PartialDecryption,
};

const N: usize = 3;

/// One party's state for the threshold scheme switch.
pub struct SwitchParty {
    pub me: usize,
    bfv: Bfv,
    pk_b: Vec<u128>,
    pk_a: Vec<u128>,
    key: KeyShare,
    keys: PrfKeys,
    smudge_bits: u32,
}

impl SwitchParty {
    pub fn new(
        me: usize,
        bfv: Bfv,
        pk_bytes: &[u8],
        key: KeyShare,
        keys: PrfKeys,
        smudge_bits: u32,
    ) -> Self {
        let (pk_b, pk_a) = crate::fhe::bfv::public_parts(pk_bytes);
        SwitchParty {
            me,
            bfv,
            pk_b,
            pk_a,
            key,
            keys,
            smudge_bits,
        }
    }

    /// The pairwise seed for additive component `j`, if this party holds it.
    ///
    /// Returns `None` for the one component this party does not hold — which is
    /// exactly the component it must not be able to derandomize, since seeing
    /// that randomness would let it strip the other parties' masking.
    pub fn seed_for(&self, j: usize) -> Option<[u8; 32]> {
        if j == self.me {
            Some(self.keys.prev) // shared with my predecessor, the co-holder
        } else if j == (self.me + 1) % N {
            Some(self.keys.mine) // shared with my successor
        } else {
            None
        }
    }

    /// Which components this party holds, in `(index, seed)` form.
    pub fn held(&self) -> [(usize, [u8; 32]); 2] {
        [
            (self.me, self.keys.prev),
            ((self.me + 1) % N, self.keys.mine),
        ]
    }

    /// MPC→FHE, one component: encrypt with randomness both co-holders can
    /// reproduce, so the two ciphertexts are bit-identical and comparable.
    pub fn encrypt_component(&self, j: usize, value: Fr, nbits: usize, ctr: u64) -> Option<BfvCt> {
        let seed = self.seed_for(j)?;
        let t_mask = (self.bfv.params.t() - 1) as u128;
        let reduced = if nbits >= 128 {
            value
        } else {
            value & ((1u128 << nbits) - 1)
        };
        let mut m = vec![0u64; self.bfv.params.n];
        m[0] = (reduced & t_mask) as u64;
        Some(encrypt_derandomized(
            &self.bfv,
            &self.pk_b,
            &self.pk_a,
            &m,
            &seed,
            component_ctr(ctr, j),
        ))
    }

    /// This party's mask contribution for component `j`, derived from the same
    /// pairwise key — so the co-holder computes the identical mask and a wrong
    /// one is caught.
    pub fn mask_component(&self, j: usize, nbits: usize, ctr: u64) -> Option<u64> {
        let seed = self.seed_for(j)?;
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"baru/hybrid/threshold/mask");
        h.update(seed);
        h.update(component_ctr(ctr, j).to_le_bytes());
        let d = h.finalize();
        let mut b = [0u8; 8];
        b.copy_from_slice(&d[..8]);
        let raw = u64::from_le_bytes(b);
        Some(if nbits >= 64 {
            raw
        } else {
            raw & ((1u64 << nbits) - 1)
        })
    }

    /// Partial decryption under key component `j`, derandomized so the
    /// co-holder produces the identical polynomial.
    pub fn partial_for(&self, j: usize, ct: &BfvCt, ctr: u64) -> Option<PartialDecryption> {
        let seed = self.seed_for(j)?;
        let s_component = if j == self.me {
            &self.key.left
        } else {
            &self.key.right
        };
        Some(partial_decrypt(
            &self.bfv,
            ct,
            s_component,
            &seed,
            component_ctr(ctr, j),
            self.smudge_bits,
        ))
    }

    pub fn bfv(&self) -> &Bfv {
        &self.bfv
    }
}

/// Counters are namespaced by component so two components never draw the same
/// randomness from different pairwise keys at the same logical step.
fn component_ctr(ctr: u64, j: usize) -> u64 {
    ctr.wrapping_mul(N as u64).wrapping_add(j as u64)
}

/// A deviation detected while cross-checking co-holders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchMismatch {
    Encryption { component: usize },
    Mask { component: usize },
    PartialDecryption { component: usize },
}

/// Cross-check every component of an MPC→FHE conversion and return the summed
/// ciphertext.
///
/// Each component is encrypted by both of its holders; the digests must agree.
/// The homomorphic sum afterwards is deterministic, so it needs no
/// derandomization — any party can recompute it.
pub fn checked_shares_to_fhe(
    parties: &[SwitchParty; N],
    components: &[Fr; N],
    nbits: usize,
    ctr: u64,
) -> Result<BfvCt, SwitchMismatch> {
    let mut summed: Option<BfvCt> = None;
    for j in 0..N {
        let mut seen: Vec<BfvCt> = Vec::new();
        for p in parties {
            if let Some(ct) = p.encrypt_component(j, components[j], nbits, ctr) {
                seen.push(ct);
            }
        }
        debug_assert_eq!(seen.len(), 2, "each component has exactly two holders");
        if ct_digest(&seen[0]) != ct_digest(&seen[1]) {
            return Err(SwitchMismatch::Encryption { component: j });
        }
        summed = Some(match summed {
            None => seen[0].clone(),
            Some(acc) => parties[0].bfv().add(&acc, &seen[0]),
        });
    }
    Ok(summed.expect("three components"))
}

/// Cross-check and run an FHE→MPC conversion: mask, threshold-decrypt, share.
///
/// Returns the additive components whose wrapping sum is the plaintext. The
/// only value opened is the *masked* one, which is uniform and therefore safe
/// to publish.
pub fn checked_fhe_to_shares(
    parties: &[SwitchParty; N],
    ct: &BfvCt,
    nbits: usize,
    ctr: u64,
) -> Result<Vec<Fr>, SwitchMismatch> {
    // ── masks: both holders of each component must agree ──
    let mut masks = [0u64; N];
    for j in 0..N {
        let seen: Vec<u64> = parties
            .iter()
            .filter_map(|p| p.mask_component(j, nbits, ctr))
            .collect();
        debug_assert_eq!(seen.len(), 2);
        if seen[0] != seen[1] {
            return Err(SwitchMismatch::Mask { component: j });
        }
        masks[j] = seen[0];
    }

    // ── mask the ciphertext (deterministic: recomputable by anyone) ──
    let bfv = parties[0].bfv();
    let total_mask: u64 = masks
        .iter()
        .fold(0u64, |a, &m| a.wrapping_add(m) & mask_of(nbits));
    let masked = bfv.add_plain_at(ct, 0, total_mask);

    // ── partial decryptions: both holders of each key component must agree ──
    let mut partials: Vec<PartialDecryption> = Vec::with_capacity(N);
    for j in 0..N {
        let seen: Vec<PartialDecryption> = parties
            .iter()
            .filter_map(|p| p.partial_for(j, &masked, ctr))
            .collect();
        debug_assert_eq!(seen.len(), 2);
        if seen[0] != seen[1] {
            return Err(SwitchMismatch::PartialDecryption { component: j });
        }
        partials.push(seen[0].clone());
    }
    let opened = combine(
        bfv,
        &masked,
        &[
            partials[0].clone(),
            partials[1].clone(),
            partials[2].clone(),
        ],
    );
    let y = opened[0] & mask_of(nbits);

    // Party 0 takes y − r₀; the others take −rⱼ. The wrapping sum telescopes.
    let mut out: Vec<Fr> = Vec::with_capacity(N);
    out.push((y as u128).wrapping_sub(masks[0] as u128));
    for &r in &masks[1..] {
        out.push(0u128.wrapping_sub(r as u128));
    }
    Ok(out)
}

fn mask_of(nbits: usize) -> u64 {
    if nbits >= 64 {
        u64::MAX
    } else {
        (1u64 << nbits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::bfv::BfvParams;
    use crate::fhe::threshold::deal_key_shares;

    const NBITS: usize = 16;

    fn params() -> BfvParams {
        BfvParams {
            n: 1024,
            log2_q: 100,
            log2_t: 20,
        }
    }

    /// Build three parties sharing a consistent ring of pairwise keys.
    ///
    /// `k[i]` is the key shared by `P_i` and `P_{i+1}`, so `P_i.mine = k[i]`
    /// and `P_i.prev = k[i−1]` — the same ring the MPC layer establishes with
    /// one 32-byte handshake per pair.
    fn three_parties(smudge_bits: u32) -> ([SwitchParty; 3], Vec<u8>, Vec<u8>) {
        let bfv = Bfv::new(params());
        let (ck, pk) = bfv.keygen();
        let s = crate::fhe::bfv::secret_part(&ck);
        let shares = deal_key_shares(&s, &bfv.params, &[3u8; 32]);
        let k = [[11u8; 32], [22u8; 32], [33u8; 32]];
        let parties = std::array::from_fn(|i| {
            SwitchParty::new(
                i,
                Bfv::new(params()),
                &pk,
                shares[i].clone(),
                PrfKeys {
                    mine: k[i],
                    prev: k[(i + 2) % 3],
                },
                smudge_bits,
            )
        });
        (parties, ck, pk)
    }

    #[test]
    fn each_component_has_exactly_two_holders_with_a_shared_seed() {
        let (parties, _ck, _pk) = three_parties(8);
        for j in 0..3 {
            let seeds: Vec<[u8; 32]> = parties.iter().filter_map(|p| p.seed_for(j)).collect();
            assert_eq!(seeds.len(), 2, "component {j} must have two holders");
            assert_eq!(
                seeds[0], seeds[1],
                "the two holders of component {j} must share a key"
            );
        }
    }

    /// The third party must not be able to derive the randomness protecting a
    /// component it does not hold — otherwise derandomization would destroy the
    /// privacy it is supposed to preserve.
    #[test]
    fn the_non_holder_cannot_derandomize_a_component() {
        let (parties, _ck, _pk) = three_parties(8);
        for j in 0..3 {
            let non_holders: Vec<usize> = (0..3)
                .filter(|&i| parties[i].seed_for(j).is_none())
                .collect();
            assert_eq!(non_holders, vec![(j + 1) % 3], "exactly one non-holder");
        }
    }

    #[test]
    fn mpc_to_fhe_round_trips_through_a_threshold_key() {
        let (parties, ck, _pk) = three_parties(8);
        // A secret split into three additive components mod 2^NBITS.
        let secret: u64 = 1234;
        let comps: [Fr; 3] = [700, 500, (secret as u128).wrapping_sub(1200)];
        let ct = checked_shares_to_fhe(&parties, &comps, NBITS, 7).expect("no mismatch");
        let got = parties[0].bfv().decrypt_poly(&ck, &ct)[0] & ((1 << NBITS) - 1);
        assert_eq!(got, secret, "the summed ciphertext must encrypt the secret");
    }

    #[test]
    fn fhe_to_mpc_round_trips_and_the_shares_reconstruct() {
        let (parties, _ck, pk) = three_parties(8);
        let secret: u64 = 4321;
        let bfv = parties[0].bfv();
        let mut m = vec![0u64; params().n];
        m[0] = secret;
        let ct = bfv.encrypt_poly(&pk, &m);

        let comps = checked_fhe_to_shares(&parties, &ct, NBITS, 11).expect("no mismatch");
        let sum = comps.iter().fold(0u128, |a, &c| a.wrapping_add(c)) & ((1u128 << NBITS) - 1);
        assert_eq!(sum as u64, secret, "shares must reconstruct the plaintext");
    }

    /// Full round trip: the re-encryption that replaces bootstrapping.
    #[test]
    fn re_encryption_preserves_the_value() {
        let (parties, ck, pk) = three_parties(8);
        let secret: u64 = 999;
        let bfv = parties[0].bfv();
        let mut m = vec![0u64; params().n];
        m[0] = secret;
        let ct = bfv.encrypt_poly(&pk, &m);

        let comps = checked_fhe_to_shares(&parties, &ct, NBITS, 3).expect("switch out");
        let arr: [Fr; 3] = [comps[0], comps[1], comps[2]];
        let fresh = checked_shares_to_fhe(&parties, &arr, NBITS, 4).expect("switch back");
        let got = bfv.decrypt_poly(&ck, &fresh)[0] & ((1 << NBITS) - 1);
        assert_eq!(got, secret, "re-encryption must preserve the value");
    }

    /// A party that deviates on a component it co-holds is caught, because its
    /// co-holder computes the same thing from the same seed.
    #[test]
    fn a_deviating_co_holder_is_caught() {
        let (mut parties, _ck, _pk) = three_parties(8);
        // Give party 1 a different pairwise key than its co-holder believes —
        // the effect of any deviation in the derandomized derivation.
        parties[1] = SwitchParty::new(
            1,
            Bfv::new(params()),
            &_pk,
            {
                let s = crate::fhe::bfv::secret_part(&_ck);
                deal_key_shares(&s, &params(), &[3u8; 32])[1].clone()
            },
            PrfKeys {
                mine: [99u8; 32], // wrong: should be k[1]
                prev: [11u8; 32],
            },
            8,
        );
        let comps: [Fr; 3] = [1, 2, 3];
        let err = checked_shares_to_fhe(&parties, &comps, NBITS, 7)
            .expect_err("a wrong seed must be caught");
        assert_eq!(err, SwitchMismatch::Encryption { component: 2 });
    }

    #[test]
    fn threshold_switch_still_works_at_realistic_smudging() {
        let (parties, _ck, pk) = three_parties(40);
        let secret: u64 = 777;
        let bfv = parties[0].bfv();
        let mut m = vec![0u64; params().n];
        m[0] = secret;
        let ct = bfv.encrypt_poly(&pk, &m);
        let comps = checked_fhe_to_shares(&parties, &ct, NBITS, 5).expect("no mismatch");
        let sum = comps.iter().fold(0u128, |a, &c| a.wrapping_add(c)) & ((1u128 << NBITS) - 1);
        assert_eq!(
            sum as u64, secret,
            "40 bits of flooding must not break correctness"
        );
    }
}
