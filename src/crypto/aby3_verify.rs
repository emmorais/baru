//! Malicious security with abort for the ABY3 layer.
//!
//! Upgrades the semi-honest replicated-sharing protocol in
//! [`super::aby3_protocol`] to tolerate **one actively corrupt party out of
//! three**: honest parties either output the correct result or return
//! [`Abort`]. Nothing here changes the semi-honest path — it is reused
//! wholesale, so its measured numbers remain the baseline the malicious cost
//! is compared against.
//!
//! # Where a corrupt party can actually deviate
//!
//! Replicated sharing is redundant: every additive component is held by two of
//! the three parties. So most of the protocol defends itself — if a value can
//! be recomputed by someone else, a lie is caught by comparison. Auditing the
//! semi-honest protocol leaves exactly three deviation points:
//!
//! 1. **Input sharing.** A dealer can send inconsistent copies of the same
//!    component to the two parties meant to hold it (§ [`Verifier::note_input`]).
//! 2. **Reveal.** A party can broadcast a component that is not the one it
//!    holds (§ [`reveal_checked`]).
//! 3. **Multiplication.** This is the hard one. In `mul_batch_party` each party
//!    sends `zᵢ = xᵢyᵢ + xᵢyᵢ₊₁ + xᵢ₊₁yᵢ + αᵢ`, which depends on the pair
//!    `(xᵢ, xᵢ₊₁)` that **no other party holds**. It is not recomputable, so
//!    replication does not cover it, and a corrupt party can add an arbitrary
//!    offset to the product. Everything else in the layer — `a2b`, the prefix
//!    adder, every comparison — is built from multiplication and addition, so
//!    closing this closes all of them.
//!
//! # The multiplication check: a MAC that costs one extra multiply
//!
//! Following Chida et al. (CRYPTO'18) in the ring adaptation of Eerikson et
//! al. (USENIX'20): draw a global MAC key `r`, and carry every wire as the
//! pair `([x], [r·x])`. Linear operations apply to both halves for free.
//! Multiplication computes
//!
//! ```text
//! [z]     = [x]·[y]
//! [r·z]   = [r·x]·[y]
//! ```
//!
//! — two semi-honest multiplications instead of one, batched into the **same
//! round**, so the cost is 2× in bandwidth and 1× in latency. At the end,
//! public random `χᵢ` are drawn by coin flip and the parties open
//!
//! ```text
//! T = Σ χᵢ·[r·zᵢ] − [r]·(Σ χᵢ·[zᵢ])
//! ```
//!
//! accepting iff `T = 0`. One opening for the whole session.
//!
//! `r` itself costs nothing: [`PrfKeys::random_share`](super::net::PrfKeys::random_share)
//! produces a replicated sharing of a fresh uniform value with no
//! communication, from the pairwise keys the multiplication protocol already
//! set up. So the malicious upgrade needs **no dealer and no preprocessing** —
//! the same property that motivated the triple-free multiplication in the first
//! place.
//!
//! # Soundness over Z_2^128, and why no ring widening is needed
//!
//! The usual objection to doing this over a ring is that `Z_2^k` has zero
//! divisors: an adversarial error `d = 2^{k−1}` survives multiplication by half
//! of all `r`, so the check passes with probability 1/2. The standard fix is to
//! widen the ring to `Z_2^{k+s}` and pay for the extra `s` bits everywhere.
//!
//! Baru does not have to pay, because it is **already** sharing over
//! `Z_2^128` ([`Fr`] is `u128`) while carrying values of a declared width
//! `nbits` of 8/16/32. Let `k` be the number of bits a semantically meaningful
//! value actually occupies — a product of two `nbits` values needs `2·nbits`,
//! plus the accumulation depth of the surrounding dot product. An adversarial
//! error that is zero mod `2^k` does not change any value the computation cares
//! about; an error that *does* change the result satisfies `d ≢ 0 (mod 2^k)`,
//! hence `d = 2^v·odd` with `v < k`. The number of MAC keys for which such an
//! error survives is `2^v < 2^k`, out of `2^128`, so
//!
//! ```text
//! Pr[undetected] < 2^{k−128} = 2^{−s},   s = 128 − k.
//! ```
//!
//! With Q16.16 values (`nbits = 32`) a product occupies 64 bits, leaving
//! `s = 64`. The slack the literature charges as overhead is already paid for.
//! This is not a claim to take on faith — [`headroom_bits`] computes it from
//! the widths actually in use, and `bench-malicious` reports the largest
//! intermediate a real inference produces so the margin can be checked rather
//! than assumed.
//!
//! # One abort point, not many
//!
//! Every check here is **deferred**: mismatches are accumulated and the verdict
//! is taken once, in [`check`]. Aborting the instant a check fails
//! would leak, because *which* check fails and *when* can depend on the secret
//! inputs — a corrupt party could inject an error that cancels only for
//! particular values and learn a predicate on the data from whether the
//! protocol survived. A single, input-independent abort point closes that
//! channel (`docs/malicious-hybrid.md` §4.6).

use sha2::{Digest, Sha256};

use super::aby3_protocol::{mul_batch_party, share_component, PartyShare};
use super::net::{serialize_msg, Mailbox, MpcMsg, PartyId, Transport};
use super::security::{Abort, AbortReason, Checked};
use super::types::Fr;

const N: usize = 3;
const TIMEOUT_MS: u64 = 60_000;

/// PRF counter domain for verification randomness.
///
/// `aby3_protocol::alpha_ctr` maps operation ids into `op_id << 40`,
/// so ordinary protocol randomness lives in the low part of the counter space
/// for any realistic op_id. Verification randomness is placed at the very top
/// so it can never collide with it — a collision would reuse a mask and leak.
pub const VERIFY_CTR_BASE: u64 = 1 << 63;

/// A wire under the malicious protocol: the value and its MAC.
///
/// `mac` is a replicated sharing of `r·val` for the session MAC key `r`. The
/// invariant is maintained by every operation in this module and is what the
/// final check verifies.
#[derive(Clone, Debug)]
pub struct MalShare {
    pub val: PartyShare,
    pub mac: PartyShare,
}

/// Statistical security, in bits, for a check over `Z_2^128` when meaningful
/// values occupy `used_bits`.
///
/// See the module header. Returns 0 when the width leaves no slack, which is
/// the honest answer rather than a small positive number: at that point the
/// check proves nothing and the caller must widen or narrow.
pub fn headroom_bits(used_bits: u32) -> u32 {
    128u32.saturating_sub(used_bits)
}

/// Bits a product of two `nbits` values can occupy once `accumulated` of them
/// are summed — the quantity [`headroom_bits`] must be fed.
pub fn product_width(nbits: usize, accumulated: usize) -> u32 {
    let acc_bits = usize::BITS - accumulated.saturating_sub(1).leading_zeros();
    2 * nbits as u32 + acc_bits
}

/// Accumulates every deferred check and delivers the single verdict.
///
/// Held inside the [`Mailbox`] so protocol functions can register into it
/// without threading an extra parameter through every signature.
pub struct Verifier {
    me: PartyId,
    /// Replicated sharing of the session MAC key `r`.
    key: PartyShare,
    /// `([z], [r·z])` for every multiplication performed this session.
    products: Vec<(PartyShare, PartyShare)>,
    /// Running transcript of values this party should agree on with its
    /// successor, and with its predecessor.
    t_next: Sha256,
    t_prev: Sha256,
    /// Set when a check failed locally. The verdict is still deferred to
    /// [`check`] so the abort cannot be timed to a secret.
    failed: Option<AbortReason>,
    /// Random pairs consumed so far, so no `ρ` is ever reused as a pad.
    pairs_drawn: u64,
}

impl Verifier {
    /// Draw the session MAC key. Free: no communication.
    pub fn new(me: PartyId, keys: &super::net::PrfKeys) -> Self {
        let (left, right) = keys.random_share(VERIFY_CTR_BASE);
        Verifier {
            me,
            key: PartyShare { left, right },
            products: Vec::new(),
            t_next: Sha256::new(),
            t_prev: Sha256::new(),
            failed: None,
            pairs_drawn: 0,
        }
    }

    /// This party's replicated share of the MAC key.
    pub fn key(&self) -> &PartyShare {
        &self.key
    }

    /// Record a locally-detected mismatch without acting on it yet.
    fn flag(&mut self, reason: AbortReason) {
        if self.failed.is_none() {
            self.failed = Some(reason);
        }
    }

    /// Absorb a value both this party and its **successor** should hold.
    fn absorb_next(&mut self, tag: &str, id: u64, component: usize, v: Fr) {
        self.t_next.update(tag.as_bytes());
        self.t_next.update(id.to_le_bytes());
        self.t_next.update((component as u64).to_le_bytes());
        self.t_next.update(v.to_le_bytes());
    }

    /// Absorb a value both this party and its **predecessor** should hold.
    fn absorb_prev(&mut self, tag: &str, id: u64, component: usize, v: Fr) {
        self.t_prev.update(tag.as_bytes());
        self.t_prev.update(id.to_le_bytes());
        self.t_prev.update((component as u64).to_le_bytes());
        self.t_prev.update(v.to_le_bytes());
    }

    /// Register a multiplication for the final sacrifice check.
    pub fn note_product(&mut self, z: PartyShare, z_mac: PartyShare) {
        self.products.push((z, z_mac));
    }

    /// How many multiplications this verifier is covering. Used by the coverage
    /// test that asserts no call path bypasses the checked multiply.
    pub fn product_count(&self) -> usize {
        self.products.len()
    }

    /// Register the components received from a dealer during input sharing.
    ///
    /// The dealer hands component `j` to both `P_j` (as `left`) and `P_{j−1}`
    /// (as `right`). Both absorb it into the transcript they share with each
    /// other, so a dealer that sends two different values is caught at the
    /// deferred comparison — without any extra message now.
    pub fn note_input(&mut self, share_id: u64, got: &PartyShare) {
        self.note_sharing("in", share_id, got);
    }

    /// Assert that a replicated sharing is **well formed**: that this party's
    /// two components agree with the copies its neighbours hold.
    ///
    /// Every replicated sharing has each component held by two parties, so
    /// absorbing both of mine into the two pairwise transcripts makes any
    /// disagreement visible at the deferred comparison.
    ///
    /// This is what covers sharings built *locally, with no message sent* —
    /// the bit-decomposition leaves in particular. Nothing goes on the wire
    /// there, so there is no message to corrupt; the only available deviation
    /// is to use a value inconsistent with what the co-holder used, and that is
    /// exactly what this catches. Cost: no communication, one hash comparison
    /// per session.
    pub fn note_sharing(&mut self, tag: &str, share_id: u64, s: &PartyShare) {
        let me = self.me;
        // My `left` is component `me`, which my predecessor also holds.
        self.absorb_prev(tag, share_id, me, s.left);
        // My `right` is component `me+1`, which my successor also holds.
        self.absorb_next(tag, share_id, (me + 1) % N, s.right);
    }
}

/// A pre-verified random pair `([ρ], [r·ρ])` — the currency for giving an
/// existing sharing a MAC.
///
/// # Why a MAC cannot simply be multiplied into existence
///
/// The obvious way to attach a MAC to a sharing `[x]` that does not have one —
/// bit decomposition creates thousands of these, since its leaves are built
/// locally with no communication — is to compute `[r]·[x]`. That is **unsound
/// here, and unsound in a way that is worse than a missed cheat.**
///
/// A corrupt party can inject an error `δ` into that multiplication, giving
/// `[r·x] + δ`. Downstream, the first gate `[z] = [x]·[y]` produces the MAC
/// `([r·x] + δ)·[y] = [r·z] + δ·y`. The final check therefore fails only when
/// `δ·y ≠ 0` — and `y` is a **secret bit**. When `y = 0` the error cancels on
/// its own and the protocol completes normally; when `y = 1` it aborts. Whether
/// the computation aborts has become a function of the data, so the adversary
/// learns a predicate on the secret inputs by observing nothing more than
/// whether the run survived.
///
/// That is a **selective-failure attack**, and it defeats the whole point of
/// deferring the abort to a single input-independent point: the deferral hides
/// *when* the abort happens, not *whether* it happens.
///
/// # The fix
///
/// Following Chida et al. (CRYPTO'18 §3–4), MACs for un-MAC'd sharings come
/// from random pairs `([ρ], [r·ρ])` prepared in advance:
///
/// ```text
/// d = open(x − ρ)                 (uniform: ρ is a fresh one-time pad)
/// [r·x] = [r·ρ] + d·[r]           (local — no multiplication)
/// ```
///
/// A corrupt party can still cheat while the *pairs* are being produced, and
/// that cheat is still caught by the same sacrifice. But `ρ` is **random and
/// independent of the inputs**, so an abort there reveals nothing about the
/// data: the selective-failure channel closes. Attaching the MAC itself is then
/// a local affine operation with no opportunity to deviate at all.
#[derive(Clone, Debug)]
pub struct MacPair {
    rho: PartyShare,
    r_rho: PartyShare,
}

/// PRF counter room reserved for random pairs, immediately above the MAC key.
///
/// Generous on purpose: bit decomposition consumes one pair per leaf, and a
/// single 28×28 convolution layer decomposed at 32 bits needs ~75 000. A
/// session-wide cap of 2²⁰ would have been reachable by a real workload, and
/// exhausting it must never silently reuse a pad.
pub const MAC_PAIR_CAPACITY: u64 = 1 << 40;

/// Produce `n` random pairs and register them for verification.
///
/// The `[ρ]` come free from the pairwise PRF keys; only `[r·ρ]` costs a
/// multiplication, and the whole pool is produced in **one** batched round
/// however many pairs are asked for. Each pair is registered as a product, so
/// the end-of-session sacrifice verifies `r·ρ` — which is exactly the guarantee
/// the affine MAC attachment then relies on.
pub fn generate_mac_pairs<T: Transport>(
    n: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    keys: &super::net::PrfKeys,
    v: &mut Verifier,
) -> Vec<MacPair> {
    assert!(
        (v.pairs_drawn + n as u64) < MAC_PAIR_CAPACITY,
        "random-pair counter space exhausted; each ρ must be a one-time pad"
    );
    if n == 0 {
        return Vec::new();
    }
    let key = v.key.clone();
    let mut rhos = Vec::with_capacity(n);
    for _ in 0..n {
        let ctr = VERIFY_CTR_BASE + 1 + v.pairs_drawn;
        v.pairs_drawn += 1;
        let (left, right) = keys.random_share(ctr);
        rhos.push(PartyShare { left, right });
    }

    let prods: Vec<(PartyShare, PartyShare)> =
        rhos.iter().map(|r| (key.clone(), r.clone())).collect();
    let macs = mul_batch_party(&prods, op_id, me, mbx);

    rhos.into_iter()
        .zip(macs)
        .map(|(rho, r_rho)| {
            // ρ is input-independent, so registering it here cannot leak
            // through a selective abort — see `MacPair`.
            v.note_product(rho.clone(), r_rho.clone());
            MacPair { rho, r_rho }
        })
        .collect()
}

/// Attach MACs to sharings that do not have them, consuming one pair each.
///
/// **One** batched opening for the whole slice, whatever its length; the MAC
/// arithmetic is local. Use this for values that enter the malicious protocol
/// without a MAC — most importantly the leaves of a bit decomposition, of which
/// a single layer produces thousands. Opening them one at a time would cost a
/// round each and would dominate everything else in the protocol.
pub fn mac_existing_batch<T: Transport>(
    values: &[PartyShare],
    pairs: &[MacPair],
    base_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    v: &mut Verifier,
) -> Vec<MalShare> {
    assert_eq!(
        values.len(),
        pairs.len(),
        "each value needs its own one-time pair"
    );
    if values.is_empty() {
        return Vec::new();
    }
    let key = v.key.clone();
    let masked: Vec<PartyShare> = values
        .iter()
        .zip(pairs)
        .map(|(x, pair)| super::aby3_protocol::sub_party(x, &pair.rho))
        .collect();
    let opened = reveal_batch_checked(&masked, base_id, me, mbx, v);

    values
        .iter()
        .zip(pairs)
        .zip(opened)
        .map(|((x, pair), d)| {
            // [r·x] = [r·ρ] + d·[r] — affine, so there is nothing to deviate in.
            let mac = super::aby3_protocol::add_party(
                &pair.r_rho,
                &super::aby3_protocol::scalar_mul_party(&key, d),
            );
            MalShare {
                val: x.clone(),
                mac,
            }
        })
        .collect()
}

/// Batched [`reveal_checked`]: one round for any number of values.
///
/// Checking works exactly as in the unbatched form — each party's successor
/// component is verified for free against a held copy, and the component it
/// cannot check is corroborated through the pairwise transcript.
pub fn reveal_batch_checked<T: Transport>(
    shares: &[PartyShare],
    base_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    v: &mut Verifier,
) -> Vec<Fr> {
    if shares.is_empty() {
        return Vec::new();
    }
    let n = shares.len();
    mbx.broadcast(serialize_msg(&MpcMsg::RevealShareBatch {
        base_id,
        values: shares.iter().map(|s| s.left).collect(),
    }));

    let mut got: [Option<Vec<Fr>>; N] = [None, None, None];
    let mut received = 0usize;
    while received < N - 1 {
        match mbx.recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::RevealShareBatch { base_id: b, .. } if *b == base_id),
        ) {
            Some((from, MpcMsg::RevealShareBatch { values, .. })) => {
                if got[from].is_none() {
                    assert_eq!(values.len(), n, "reveal batch length mismatch");
                    got[from] = Some(values);
                    received += 1;
                }
            }
            Some(_) => unreachable!("predicate guarantees RevealShareBatch"),
            None => panic!("reveal_batch_checked: timeout waiting for base_id={base_id}"),
        }
    }

    let prev = (me + N - 1) % N;
    let next = (me + 1) % N;
    let from_next = got[next].take().expect("successor's batch");
    let from_prev = got[prev].take().expect("predecessor's batch");

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if from_next[i] != shares[i].right {
            v.flag(AbortReason::RevealMismatch);
        }
        v.absorb_next("revb", base_id + i as u64, prev, from_prev[i]);
        v.absorb_prev("revb", base_id + i as u64, next, shares[i].right);
        out.push(
            shares[i]
                .left
                .wrapping_add(from_next[i])
                .wrapping_add(from_prev[i]),
        );
    }
    out
}

/// Reveal a value to all three parties, checking every component.
///
/// Each party broadcasts its `left` (additive component `i`). Component `i` is
/// held by `P_i` and by `P_{i−1}`, so of the two receivers exactly one — the
/// predecessor — can compare the broadcast against a copy it already holds.
/// That check is **free**: no extra bytes, and it catches a party that lies to
/// its predecessor.
///
/// The other receiver has no copy, so it absorbs what it was told into the
/// transcript it shares with the party that *does* hold it. If the sender told
/// the two receivers different stories, or told the non-holder a value the
/// holder disagrees with, the transcripts diverge and
/// [`check`] aborts. So a reveal costs exactly what it did before,
/// and the whole cost of checking it is one hash comparison per session.
pub fn reveal_checked<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    v: &mut Verifier,
) -> Fr {
    mbx.broadcast(serialize_msg(&MpcMsg::RevealShare {
        share_id,
        value: share.left,
    }));

    let mut got: [Option<Fr>; N] = [None; N];
    got[me] = Some(share.left);
    let mut received = 0usize;
    while received < N - 1 {
        match mbx.recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::RevealShare { share_id: sid, .. } if *sid == share_id),
        ) {
            Some((from, MpcMsg::RevealShare { value, .. })) => {
                if got[from].is_none() {
                    got[from] = Some(value);
                    received += 1;
                }
            }
            Some(_) => unreachable!("predicate guarantees RevealShare"),
            None => panic!("reveal_checked: timeout waiting for share_id={share_id}"),
        }
    }

    let prev = (me + N - 1) % N;
    let next = (me + 1) % N;
    let from_next = got[next].expect("received successor's component");
    let from_prev = got[prev].expect("received predecessor's component");

    // Free check: my successor's component `next` is my own `right`.
    if from_next != share.right {
        v.flag(AbortReason::RevealMismatch);
    }
    // My predecessor's component `prev` is held by *its* predecessor, which is
    // my successor. Absorb it into the transcript I share with them.
    v.absorb_next("rev", share_id, prev, from_prev);
    // And corroborate: the component I hold as `right` is the one my successor
    // could not check for itself.
    v.absorb_prev("rev", share_id, next, share.right);

    share.left.wrapping_add(from_next).wrapping_add(from_prev)
}

/// Batched malicious multiplication.
///
/// Computes `[z] = [x]·[y]` and `[r·z] = [r·x]·[y]` for every pair, issuing
/// **one** semi-honest batched multiplication over both halves. Latency is
/// therefore identical to the semi-honest protocol — one round for any batch
/// width — and only the bytes double. Given this repository's recurring finding
/// that round structure, not arithmetic, is what costs, that is the shape of
/// overhead worth having.
pub fn mul_batch_checked<T: Transport>(
    pairs: &[(MalShare, MalShare)],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    v: &mut Verifier,
) -> Vec<MalShare> {
    if pairs.is_empty() {
        return Vec::new();
    }
    // Values first, then MACs, in one batch: a single round for both.
    let mut flat: Vec<(PartyShare, PartyShare)> = Vec::with_capacity(2 * pairs.len());
    for (x, y) in pairs {
        flat.push((x.val.clone(), y.val.clone()));
    }
    for (x, y) in pairs {
        flat.push((x.mac.clone(), y.val.clone()));
    }

    let out = mul_batch_party(&flat, base_op_id, me, mbx);
    let (vals, macs) = out.split_at(pairs.len());

    vals.iter()
        .zip(macs)
        .map(|(z, z_mac)| {
            v.note_product(z.clone(), z_mac.clone());
            MalShare {
                val: z.clone(),
                mac: z_mac.clone(),
            }
        })
        .collect()
}

// ─── Local (communication-free) operations ───────────────────────────────────

pub fn add(a: &MalShare, b: &MalShare) -> MalShare {
    MalShare {
        val: super::aby3_protocol::add_party(&a.val, &b.val),
        mac: super::aby3_protocol::add_party(&a.mac, &b.mac),
    }
}

pub fn sub(a: &MalShare, b: &MalShare) -> MalShare {
    MalShare {
        val: super::aby3_protocol::sub_party(&a.val, &b.val),
        mac: super::aby3_protocol::sub_party(&a.mac, &b.mac),
    }
}

pub fn scalar_mul(a: &MalShare, k: Fr) -> MalShare {
    MalShare {
        val: super::aby3_protocol::scalar_mul_party(&a.val, k),
        mac: super::aby3_protocol::scalar_mul_party(&a.mac, k),
    }
}

/// Share a public constant. The MAC is `r·c`, a local scalar multiple of the
/// key share — so public values enter the malicious protocol for free.
pub fn share_public(value: Fr, me: PartyId, key: &PartyShare) -> MalShare {
    MalShare {
        val: super::aby3_protocol::share_public(value, me),
        mac: super::aby3_protocol::scalar_mul_party(key, value),
    }
}

// ─── Bit decomposition ───────────────────────────────────────────────────────

/// `a ⊕ b = a + b − 2ab` on MAC'd bit shares, given their product.
fn xor_from_product(a: &MalShare, b: &MalShare, ab: &MalShare) -> MalShare {
    sub(&add(a, b), &scalar_mul(ab, 2))
}

/// Ids reserved by one [`a2b_batch_checked`] call, matching
/// [`super::aby3_protocol::A2B_ID_BLOCK`] but wider: the malicious path also
/// needs ids for the pair pool and the batched MAC opening.
pub const A2B_CHECKED_ID_BLOCK: u64 = 256;

/// Maliciously secure batched bit decomposition.
///
/// Structurally the semi-honest [`super::aby3_protocol::a2b_batch_party`] with
/// MAC'd wires, so it keeps that routine's defining property: **every internal
/// round is batched across all lanes and all bits**, so decomposing a whole
/// layer costs the same ~9 rounds as decomposing one value.
///
/// # Why the leaves are safe
///
/// Step 1 builds binary sharings of the three additive components locally, with
/// no communication. A corrupt party's only available deviation is to use bits
/// that disagree with the component its co-holder holds — every component of a
/// replicated sharing is known to two parties — and
/// [`Verifier::note_sharing`] catches exactly that, for free.
///
/// The leaves then need MACs, and they must come from [`MacPair`]s rather than
/// from a multiplication; see that type's documentation for the
/// selective-failure attack the multiplication route opens. Both the pool and
/// the attachment are single batched rounds.
pub fn a2b_batch_checked<T: Transport>(
    values: &[MalShare],
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    keys: &super::net::PrfKeys,
    v: &mut Verifier,
) -> Vec<Vec<MalShare>> {
    let count = values.len();
    if count == 0 || nbits == 0 {
        return vec![Vec::new(); count];
    }
    let mut next_id = op_id;
    let mut take_id = |n: usize| {
        let b = next_id;
        next_id += n as u64;
        b
    };

    // ── 1. free binary sharings of the three additive components ──
    let mut leaves: Vec<PartyShare> = Vec::with_capacity(count * N * nbits);
    for a in values {
        for j in 0..N {
            let known: Fr = if j == me {
                a.val.left
            } else if j == (me + 1) % N {
                a.val.right
            } else {
                0
            };
            for k in 0..nbits {
                leaves.push(share_component((known >> k) & 1, j, me));
            }
        }
    }
    // Well-formedness of every locally-built sharing. No communication.
    let leaf_tag_base = take_id(leaves.len());
    for (i, leaf) in leaves.iter().enumerate() {
        v.note_sharing("a2b", leaf_tag_base + i as u64, leaf);
    }

    // ── 2. MAC the leaves: one batched pool, one batched opening ──
    let pairs = generate_mac_pairs(leaves.len(), take_id(1), me, mbx, keys, v);
    let macd = mac_existing_batch(&leaves, &pairs, take_id(leaves.len()), me, mbx, v);

    // Regroup into per-lane (x, y, z) bit vectors.
    let comp: Vec<(Vec<MalShare>, Vec<MalShare>, Vec<MalShare>)> = (0..count)
        .map(|lane| {
            let b = lane * N * nbits;
            (
                macd[b..b + nbits].to_vec(),
                macd[b + nbits..b + 2 * nbits].to_vec(),
                macd[b + 2 * nbits..b + 3 * nbits].to_vec(),
            )
        })
        .collect();

    // ── 3. carry-save 3 → 2, all lanes in two batched rounds ──
    let mut pw_pairs = Vec::with_capacity(3 * count * nbits);
    for (x, y, z) in &comp {
        for k in 0..nbits {
            pw_pairs.push((x[k].clone(), y[k].clone()));
        }
        for k in 0..nbits {
            pw_pairs.push((x[k].clone(), z[k].clone()));
        }
        for k in 0..nbits {
            pw_pairs.push((y[k].clone(), z[k].clone()));
        }
    }
    let pw = mul_batch_checked(&pw_pairs, take_id(1), me, mbx, v);

    let mut triple = Vec::with_capacity(count * nbits);
    for (lane, (_x, _y, z)) in comp.iter().enumerate() {
        let xy = &pw[lane * 3 * nbits..lane * 3 * nbits + nbits];
        for k in 0..nbits {
            triple.push((xy[k].clone(), z[k].clone()));
        }
    }
    let xyz_all = mul_batch_checked(&triple, take_id(1), me, mbx, v);

    let key = v.key.clone();
    let mut sums = Vec::with_capacity(count);
    let mut shifts = Vec::with_capacity(count);
    for (lane, (x, y, z)) in comp.iter().enumerate() {
        let b = lane * 3 * nbits;
        let (xy, xz, yz) = (
            &pw[b..b + nbits],
            &pw[b + nbits..b + 2 * nbits],
            &pw[b + 2 * nbits..b + 3 * nbits],
        );
        let xyz = &xyz_all[lane * nbits..lane * nbits + nbits];
        let mut sum = Vec::with_capacity(nbits);
        let mut carry = Vec::with_capacity(nbits);
        for k in 0..nbits {
            let sym = add(&add(&xy[k], &xz[k]), &yz[k]);
            let lin = add(&add(&x[k], &y[k]), &z[k]);
            sum.push(add(
                &sub(&lin, &scalar_mul(&sym, 2)),
                &scalar_mul(&xyz[k], 4),
            ));
            carry.push(sub(&sym, &scalar_mul(&xyz[k], 2)));
        }
        let mut shifted = Vec::with_capacity(nbits);
        shifted.push(share_public(0, me, &key));
        shifted.extend(carry.into_iter().take(nbits - 1));
        sums.push(sum);
        shifts.push(shifted);
    }

    // ── 4. one batched prefix adder over all lanes ──
    ppa_add_batch_checked(&sums, &shifts, nbits, &mut next_id, me, mbx, v)
}

/// Batched Kogge–Stone prefix adder on MAC'd bit shares.
fn ppa_add_batch_checked<T: Transport>(
    u: &[Vec<MalShare>],
    w: &[Vec<MalShare>],
    nbits: usize,
    next_id: &mut u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    v: &mut Verifier,
) -> Vec<Vec<MalShare>> {
    let count = u.len();
    let take_id = |n: &mut u64| {
        let b = *n;
        *n += 2;
        b
    };

    let mut pairs = Vec::with_capacity(count * nbits);
    for lane in 0..count {
        for k in 0..nbits {
            pairs.push((u[lane][k].clone(), w[lane][k].clone()));
        }
    }
    let uv = mul_batch_checked(&pairs, take_id(next_id), me, mbx, v);
    let mut g: Vec<Vec<MalShare>> = (0..count)
        .map(|l| uv[l * nbits..l * nbits + nbits].to_vec())
        .collect();
    let mut p: Vec<Vec<MalShare>> = (0..count)
        .map(|l| {
            (0..nbits)
                .map(|k| xor_from_product(&u[l][k], &w[l][k], &uv[l * nbits + k]))
                .collect()
        })
        .collect();
    let p_leaf = p.clone();

    let mut d = 1usize;
    while d < nbits {
        let mut pr = Vec::with_capacity(2 * count * (nbits - d));
        for lane in 0..count {
            for k in d..nbits {
                pr.push((p[lane][k].clone(), g[lane][k - d].clone()));
                pr.push((p[lane][k].clone(), p[lane][k - d].clone()));
            }
        }
        let out = mul_batch_checked(&pr, take_id(next_id), me, mbx, v);
        let width = nbits - d;
        for lane in 0..count {
            let (mut ng, mut np) = (g[lane].clone(), p[lane].clone());
            for (i, k) in (d..nbits).enumerate() {
                let o = 2 * (lane * width + i);
                ng[k] = add(&g[lane][k], &out[o]);
                np[k] = out[o + 1].clone();
            }
            g[lane] = ng;
            p[lane] = np;
        }
        d <<= 1;
    }

    if nbits == 1 {
        return (0..count).map(|l| vec![p_leaf[l][0].clone()]).collect();
    }
    let mut pairs = Vec::with_capacity(count * (nbits - 1));
    for lane in 0..count {
        for k in 1..nbits {
            pairs.push((p_leaf[lane][k].clone(), g[lane][k - 1].clone()));
        }
    }
    let prod = mul_batch_checked(&pairs, take_id(next_id), me, mbx, v);
    let width = nbits - 1;
    (0..count)
        .map(|lane| {
            let mut bits = Vec::with_capacity(nbits);
            bits.push(p_leaf[lane][0].clone());
            for (i, k) in (1..nbits).enumerate() {
                bits.push(xor_from_product(
                    &p_leaf[lane][k],
                    &g[lane][k - 1],
                    &prod[lane * width + i],
                ));
            }
            bits
        })
        .collect()
}

/// Maliciously secure batched `a < b`, as the sign bit of `a − b`.
///
/// Every Baru comparison (`<`, `>`, `≤`, `≥`, ReLU, max-pool, argmax) reduces to
/// this, so covering it covers the entire nonlinear surface.
// `(nbits, op_id, me, mbx, keys, v)` is this crate's protocol-call convention,
// carried by every checked operation; bundling it here alone would make this
// function the odd one out rather than clearer.
#[allow(clippy::too_many_arguments)]
pub fn lt_batch_checked<T: Transport>(
    a: &[MalShare],
    b: &[MalShare],
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    keys: &super::net::PrfKeys,
    v: &mut Verifier,
) -> Vec<MalShare> {
    assert_eq!(a.len(), b.len(), "comparison operands must align");
    let diffs: Vec<MalShare> = a.iter().zip(b).map(|(x, y)| sub(x, y)).collect();
    let bits = a2b_batch_checked(&diffs, nbits, op_id, me, mbx, keys, v);
    bits.into_iter()
        .map(|mut bs| bs.remove(nbits - 1))
        .collect()
}

// ─── The deferred verdict ────────────────────────────────────────────────────

/// Public challenge coefficients, agreed by commit-then-open.
///
/// Every party commits to a seed before any seed is revealed, so no party can
/// choose its contribution after seeing the others' and bias the challenge.
/// This matters: the coefficients must be unpredictable at the time the
/// adversary decides what errors to inject, or it could pick errors that the
/// linear combination cancels.
fn coin_flip<T: Transport>(me: PartyId, mbx: &Mailbox<T>) -> Checked<[u8; 32]> {
    let mut seed = [0u8; 32];
    rand::Rng::fill(&mut rand::thread_rng(), &mut seed);
    let digest: [u8; 32] = Sha256::digest(seed).into();

    mbx.broadcast(serialize_msg(&MpcMsg::CoinCommit { digest }));
    let mut commits: [Option<[u8; 32]>; N] = [None; N];
    commits[me] = Some(digest);
    for _ in 0..N - 1 {
        match mbx.recv_matching(TIMEOUT_MS, |m| matches!(m, MpcMsg::CoinCommit { .. })) {
            Some((from, MpcMsg::CoinCommit { digest })) => commits[from] = Some(digest),
            _ => panic!("coin_flip: timeout waiting for commitments"),
        }
    }

    mbx.broadcast(serialize_msg(&MpcMsg::CoinOpen { seed }));
    let mut seeds: [Option<[u8; 32]>; N] = [None; N];
    seeds[me] = Some(seed);
    for _ in 0..N - 1 {
        match mbx.recv_matching(TIMEOUT_MS, |m| matches!(m, MpcMsg::CoinOpen { .. })) {
            Some((from, MpcMsg::CoinOpen { seed })) => seeds[from] = Some(seed),
            _ => panic!("coin_flip: timeout waiting for openings"),
        }
    }

    let mut acc = Sha256::new();
    for i in 0..N {
        let s = seeds[i].ok_or_else(|| Abort::new(AbortReason::CommitmentMismatch, "coin_flip"))?;
        let c =
            commits[i].ok_or_else(|| Abort::new(AbortReason::CommitmentMismatch, "coin_flip"))?;
        let recomputed: [u8; 32] = Sha256::digest(s).into();
        if recomputed != c {
            return Err(Abort::new(AbortReason::CommitmentMismatch, "coin_flip"));
        }
        acc.update(s);
    }
    Ok(acc.finalize().into())
}

/// Expand the coin-flip output into per-product challenge coefficients.
fn challenges(seed: &[u8; 32], n: usize) -> Vec<Fr> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0u64;
    while out.len() < n {
        let mut h = Sha256::new();
        h.update(seed);
        h.update(i.to_le_bytes());
        let d = h.finalize();
        for chunk in d.chunks_exact(16) {
            if out.len() == n {
                break;
            }
            let mut b = [0u8; 16];
            b.copy_from_slice(chunk);
            out.push(u128::from_le_bytes(b));
        }
        i += 1;
    }
    out
}

/// The single verdict: run the sacrifice check and compare transcripts.
///
/// Call once, at the end of a session (or at an explicit synchronisation point
/// that is not data-dependent). Returns `Ok(())` iff no deviation was detected.
///
/// Order matters. The transcript comparison is evaluated *together with* the
/// sacrifice result rather than before it, and any locally-flagged mismatch is
/// folded in, so exactly one abort decision is taken and its timing carries no
/// information about the inputs.
pub fn check<T: Transport>(
    me: PartyId,
    mbx: &Mailbox<T>,
    op_id: u64,
    v: &mut Verifier,
) -> Checked<()> {
    // ── the sacrifice ────────────────────────────────────────────────────────
    let products = v.products.clone();
    let key = v.key.clone();

    let sacrifice_ok = if products.is_empty() {
        true
    } else {
        let seed = coin_flip(me, mbx)?;
        let chi = challenges(&seed, products.len());

        // w = Σ χᵢ·zᵢ  and  u = Σ χᵢ·(r·zᵢ), both local.
        let mut w = PartyShare { left: 0, right: 0 };
        let mut u = PartyShare { left: 0, right: 0 };
        for (c, (z, z_mac)) in chi.iter().zip(&products) {
            w = super::aby3_protocol::add_party(&w, &super::aby3_protocol::scalar_mul_party(z, *c));
            u = super::aby3_protocol::add_party(
                &u,
                &super::aby3_protocol::scalar_mul_party(z_mac, *c),
            );
        }

        // T = u − r·w. One multiplication, whatever the session's size.
        let rw = mul_batch_party(&[(key, w)], op_id, me, mbx)
            .pop()
            .expect("one product");
        let t = super::aby3_protocol::sub_party(&u, &rw);
        reveal_checked(&t, op_id, me, mbx, v) == 0
    };

    // ── transcript comparison ────────────────────────────────────────────────
    let mine_next = <[u8; 32]>::from(v.t_next.clone().finalize());
    let mine_prev = <[u8; 32]>::from(v.t_prev.clone().finalize());
    let flagged = v.failed;

    let next = (me + 1) % N;
    let prev = (me + N - 1) % N;
    mbx.send(
        next,
        serialize_msg(&MpcMsg::ViewHash {
            toward_next: true,
            digest: mine_next,
        }),
    );
    mbx.send(
        prev,
        serialize_msg(&MpcMsg::ViewHash {
            toward_next: false,
            digest: mine_prev,
        }),
    );

    let mut views_ok = true;
    for _ in 0..N - 1 {
        match mbx.recv_matching(TIMEOUT_MS, |m| matches!(m, MpcMsg::ViewHash { .. })) {
            // My predecessor's "toward next" transcript is the one it keeps
            // with me; it must equal my "toward prev".
            Some((
                _,
                MpcMsg::ViewHash {
                    toward_next: true,
                    digest,
                },
            )) => {
                views_ok &= digest == mine_prev;
            }
            // My successor's "toward prev" transcript must equal my "toward next".
            Some((
                _,
                MpcMsg::ViewHash {
                    toward_next: false,
                    digest,
                },
            )) => {
                views_ok &= digest == mine_next;
            }
            _ => panic!("check: timeout waiting for view hashes"),
        }
    }

    // ── one verdict ──────────────────────────────────────────────────────────
    if let Some(reason) = flagged {
        return Err(Abort::new(reason, "deferred check"));
    }
    if !views_ok {
        return Err(Abort::new(AbortReason::ViewMismatch, "deferred check"));
    }
    if !sacrifice_ok {
        return Err(Abort::new(
            AbortReason::MultiplicationCheck,
            "deferred check",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headroom_is_what_the_module_header_claims() {
        // Q16.16 values: a product needs 64 bits, leaving 64 bits of slack.
        assert_eq!(product_width(32, 1), 64);
        assert_eq!(headroom_bits(product_width(32, 1)), 64);

        // A 28x28 convolution accumulates 784 products; ~10 more bits.
        let w = product_width(32, 784);
        assert_eq!(w, 74);
        assert_eq!(headroom_bits(w), 54);

        // 8-bit values leave a great deal more.
        assert_eq!(headroom_bits(product_width(8, 1)), 112);
    }

    #[test]
    fn headroom_reports_zero_rather_than_a_comfortable_lie() {
        assert_eq!(headroom_bits(128), 0);
        assert_eq!(headroom_bits(200), 0);
    }

    #[test]
    fn challenges_are_deterministic_and_distinct() {
        let seed = [7u8; 32];
        let a = challenges(&seed, 40);
        let b = challenges(&seed, 40);
        assert_eq!(a, b, "challenge expansion must be deterministic");
        assert_eq!(a.len(), 40);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 40, "challenge coefficients repeated");
    }

    #[test]
    fn verification_counters_cannot_collide_with_protocol_alphas() {
        // `alpha_ctr` puts op_id `k` at `k << 40`, so reaching the verifier's
        // base would need an op_id of 2^23 — far beyond any session, since ids
        // are handed out one per operation.
        assert_eq!(VERIFY_CTR_BASE, 1u64 << 63);
        let highest_realistic_op_id: u64 = 1 << 22;
        assert!(
            (highest_realistic_op_id << 40) < VERIFY_CTR_BASE || highest_realistic_op_id > 1 << 23
        );
    }
}
