//! ABY3-specific protocol helpers for additive 3-party secret sharing.
//!
//! This module contains the ABY3 protocol primitives: PartyShare representation,
//! share input/output, multiplication via Beaver triples, a2b decomposition,
//! and related operations. These are specific to the ABY3 scheme and should not
//! be used directly by the evaluator or generic backends.

use rand::Rng;

use crate::crypto::net::{serialize_msg, Mailbox, MpcMsg, PartyId, Transport};
use crate::crypto::types::Fr;

const N: usize = 3;
/// Wall-clock cap on waiting for a peer's message. This is a liveness
/// backstop for a genuinely hung/failed party, not a performance knob: a
/// correct run never approaches it. It is generous because `cargo test` runs
/// many 3-party simulations concurrently, and a large batched-resharing
/// message can sit behind CPU-starved peer threads far longer than any real
/// deadline — a lower value made the heavy end-to-end tests flake under load
/// while passing in isolation.
const TIMEOUT_MS: u64 = 60_000;

/// The two field elements that party `id` holds for a given shared value.
/// In ABY3, party i holds (s_i, s_{(i+1) mod 3}).
///
/// This is ABY3-specific and should not be exposed to the interpreter.
#[derive(Clone, Debug)]
pub struct PartyShare {
    pub left: Fr,  // s_i
    pub right: Fr, // s_{(i+1) mod 3}
}

/// Distribute a secret input `value` to all three parties.
///
/// - Party `owner` calls this with the actual `value`.
/// - All other parties call this with `value = 0` (ignored).
///
/// Returns the `PartyShare` for the *calling* party.
pub fn share_input<T: Transport>(
    value: Fr,
    share_id: u64,
    owner: PartyId,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> PartyShare {
    if me == owner {
        let mut rng = rand::thread_rng();
        let s: [Fr; 3] = {
            let s0: Fr = rng.gen();
            let s1: Fr = rng.gen();
            let s2 = value.wrapping_sub(s0).wrapping_sub(s1);
            [s0, s1, s2]
        };
        for i in 0..N {
            let right_idx = (i + 1) % N;
            if i != me {
                mbx.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[i],
                    }),
                );
                mbx.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[right_idx],
                    }),
                );
            }
        }
        PartyShare {
            left: s[me],
            right: s[(me + 1) % N],
        }
    } else {
        let mut components: Vec<Fr> = Vec::new();
        while components.len() < 2 {
            if let Some((_from, m)) = mbx.recv_matching(
                TIMEOUT_MS,
                |m| matches!(m, MpcMsg::InputShare { share_id: sid, .. } if *sid == share_id),
            ) {
                if let MpcMsg::InputShare { value: v, .. } = m {
                    components.push(v);
                }
            } else {
                panic!(
                    "share_input: timeout waiting for InputShare share_id={}",
                    share_id
                );
            }
        }
        PartyShare {
            left: components[0],
            right: components[1],
        }
    }
}

/// Distribute a whole batch of secret inputs owned by the *same* party — the
/// batched form of [`share_input`], with identical semantics per element.
///
/// Share `k` of the batch takes id `base_id + k`, so ids stay in lockstep with
/// the unbatched path. The owner sends **one** message per peer for the entire
/// batch instead of two per element, and each non-owner performs **one**
/// blocking receive instead of `n`.
///
/// This matters because the FHE→MPC scheme switch imports an entire layer of
/// activations at once: at one round trip per element a $28\times28$
/// convolution's output spent seconds in `recv` alone, dwarfing the
/// nonlinearity it was importing values for.
pub fn share_input_batch<T: Transport>(
    values: &[Fr],
    base_id: u64,
    owner: PartyId,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    if me == owner {
        let mut rng = rand::thread_rng();
        // Per-peer buffers: peer `i` receives its own component and the
        // component of `i+1`, exactly as in the unbatched protocol.
        let mut lefts = [
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        ];
        let mut rights = [
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        ];
        let mut mine = Vec::with_capacity(n);
        for &value in values {
            let s0: Fr = rng.gen();
            let s1: Fr = rng.gen();
            let s2 = value.wrapping_sub(s0).wrapping_sub(s1);
            let s = [s0, s1, s2];
            for i in 0..N {
                lefts[i].push(s[i]);
                rights[i].push(s[(i + 1) % N]);
            }
            mine.push(PartyShare {
                left: s[me],
                right: s[(me + 1) % N],
            });
        }
        for i in 0..N {
            if i != me {
                mbx.send(
                    i,
                    serialize_msg(&MpcMsg::InputShareBatch {
                        base_id,
                        lefts: std::mem::take(&mut lefts[i]),
                        rights: std::mem::take(&mut rights[i]),
                    }),
                );
            }
        }
        mine
    } else {
        let (_from, m) = mbx
            .recv_matching(
                TIMEOUT_MS,
                |m| matches!(m, MpcMsg::InputShareBatch { base_id: b, .. } if *b == base_id),
            )
            .unwrap_or_else(|| {
                panic!("share_input_batch: timeout waiting for InputShareBatch base_id={base_id}")
            });
        let MpcMsg::InputShareBatch { lefts, rights, .. } = m else {
            unreachable!("predicate guarantees InputShareBatch")
        };
        assert_eq!(
            lefts.len(),
            n,
            "share_input_batch: owner sent a batch of the wrong length"
        );
        assert_eq!(
            rights.len(),
            n,
            "share_input_batch: owner sent a batch of the wrong length"
        );
        lefts
            .into_iter()
            .zip(rights)
            .map(|(left, right)| PartyShare { left, right })
            .collect()
    }
}

/// Share a public constant locally (no communication).  Party 0 gets the
/// value in its left component; others get 0 so that reconstruction = value.
///
/// The replicated layout must be consistent: party `i` holds
/// `(s_i, s_{(i+1) mod 3})` for the additive sharing `s = (value, 0, 0)`.
/// Party 2's *right* component is therefore `s_0 = value`, not 0 — an
/// earlier version zeroed every right component, which reconstructs fine
/// (reveal sums only the lefts) but corrupts any *multiplication* of the
/// public value, since the multiplication protocol relies on the replicated
/// pairing. (Latent until a public nonzero constant was fed into a
/// product — e.g. `crate::gadgets` argmax indices / ReLU `keep` mask.)
pub fn share_public(value: Fr, me: PartyId) -> PartyShare {
    let s = |i: usize| if i == 0 { value } else { 0 };
    PartyShare {
        left: s(me),
        right: s((me + 1) % 3),
    }
}

/// Add two party shares locally (no communication required).
pub fn add_party(a: &PartyShare, b: &PartyShare) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_add(b.left),
        right: a.right.wrapping_add(b.right),
    }
}

/// Subtract two party shares locally.
pub fn sub_party(a: &PartyShare, b: &PartyShare) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_sub(b.left),
        right: a.right.wrapping_sub(b.right),
    }
}

/// Multiply a party share by a public scalar locally.
pub fn scalar_mul_party(a: &PartyShare, k: Fr) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_mul(k),
        right: a.right.wrapping_mul(k),
    }
}

/// Reveal a shared value to `reveal_to`.
///
/// - If `reveal_to == Some(p)`, all parties send their left component to p.
/// - If `reveal_to == None`, all parties broadcast; everyone reconstructs.
pub fn reveal<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    reveal_to: Option<PartyId>,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Fr {
    // Collect `N - 1` RevealShare contributions for `share_id`, summing them
    // into `start`.  Buffering means peer contributions can arrive in any order.
    fn collect<T: Transport>(mbx: &Mailbox<T>, share_id: u64, start: Fr) -> Fr {
        let mut sum = start;
        let mut received = 0usize;
        while received < N - 1 {
            if let Some((_from, m)) = mbx.recv_matching(
                TIMEOUT_MS,
                |m| matches!(m, MpcMsg::RevealShare { share_id: sid, .. } if *sid == share_id),
            ) {
                if let MpcMsg::RevealShare { value: v, .. } = m {
                    sum = sum.wrapping_add(v);
                    received += 1;
                }
            } else {
                panic!(
                    "reveal: timeout waiting for RevealShare share_id={}",
                    share_id
                );
            }
        }
        sum
    }

    match reveal_to {
        Some(target) => {
            if me != target {
                mbx.send(
                    target,
                    serialize_msg(&MpcMsg::RevealShare {
                        share_id,
                        value: share.left,
                    }),
                );
                return 0;
            }
            collect(mbx, share_id, share.left)
        }
        None => {
            mbx.broadcast(serialize_msg(&MpcMsg::RevealShare {
                share_id,
                value: share.left,
            }));
            collect(mbx, share_id, share.left)
        }
    }
}

/// ABY3 multiplication — **no Beaver triples**.
///
/// With replicated sharing `x = x₀+x₁+x₂` (party `i` holds `xᵢ, xᵢ₊₁`), each
/// party can compute three of the nine cross terms locally:
///
/// ```text
/// zᵢ = xᵢyᵢ + xᵢyᵢ₊₁ + xᵢ₊₁yᵢ + αᵢ
/// ```
///
/// Summed over the three parties these cover all nine terms exactly once, so
/// `Σzᵢ = (Σxᵢ)(Σyᵢ) = xy`. The `αᵢ` are zero-sum correlated randomness from
/// pairwise PRF keys — they cancel in the sum but re-randomise each party's
/// component, which is what stops `zᵢ` leaking. Generating them costs *no*
/// communication, so the whole multiplication is **one ring element sent to
/// one neighbour**.
///
/// This replaces a Beaver-triple protocol that had party 0 deal 6 ring
/// elements per multiplication and then opened 2 more. Triples are not
/// merely moved offline here, they are eliminated: there is nothing left to
/// precompute or distribute.
/// Open a whole batch of ring shares to every party in **one** directed message
/// each — the semi-honest counterpart of
/// [`crate::crypto::aby3_verify::reveal_batch_checked`], which broadcasts and
/// cross-checks instead.
///
/// Party `i` holds `(sᵢ, sᵢ₊₁)` and is missing `sᵢ₊₂`, which party `i+2` holds
/// as its `left`. So every party sends its `left` to party `i+1` and receives
/// the one summand it lacks. One round, one message per party, whatever the
/// batch size.
///
/// The batching is the point: [`reveal`] costs `N−1` messages *per value*, so
/// opening a convolution layer element by element would spend tens of thousands
/// of round trips on what is structurally a single exchange. `Π_WA` opens one
/// masked value per DReLU, so without this the wrap protocol's cost would be
/// dominated by an artefact of the API rather than by the protocol.
pub fn reveal_batch_party<T: Transport>(
    shares: &[PartyShare],
    base_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<Fr> {
    if shares.is_empty() {
        return Vec::new();
    }
    let mine: Vec<Fr> = shares.iter().map(|s| s.left).collect();
    mbx.send(
        (me + 1) % N,
        serialize_msg(&MpcMsg::RevealShareBatch {
            base_id,
            values: mine,
        }),
    );
    let (_, msg) = mbx
        .recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::RevealShareBatch { base_id: b, .. } if *b == base_id),
        )
        .unwrap_or_else(|| panic!("reveal_batch_party: timeout for base_id={base_id}"));
    let missing = match msg {
        MpcMsg::RevealShareBatch { values, .. } => values,
        _ => unreachable!("predicate guarantees RevealShareBatch"),
    };
    assert_eq!(missing.len(), shares.len(), "reveal batch length mismatch");
    shares
        .iter()
        .zip(missing)
        .map(|(s, m)| s.left.wrapping_add(s.right).wrapping_add(m))
        .collect()
}

/// Open a batch of ring shares and return the three **additive components** of
/// each, in canonical party order, rather than only their sum.
///
/// Same single directed message as [`reveal_batch_party`] — after the exchange
/// every party already holds all three summands (`left`, `right`, and the one
/// received), so the components cost nothing extra. What differs is only what is
/// returned.
///
/// `Π_WA` needs exactly this: its step 3 computes `α = wrap3(r₁, r₂, r₃, L)` "in
/// the clear", which is a function of the individual summands and *not*
/// recoverable from `r = Σrⱼ`. A reveal that returns only the value would make
/// the wrap protocol unimplementable without a second exchange.
/// Pack ring values at `bits` each, little-endian within the stream.
///
/// Public so that tests which tamper with a resharing message can express the
/// tamper on *values* rather than on the encoding.
pub fn pack_ring(values: &[Fr], bits: u32) -> Vec<u8> {
    let mut out = vec![0u8; (values.len() * bits as usize).div_ceil(8)];
    for (i, v) in values.iter().enumerate() {
        let base = i * bits as usize;
        for b in 0..bits as usize {
            if (v >> b) & 1 == 1 {
                out[(base + b) / 8] |= 1 << ((base + b) % 8);
            }
        }
    }
    out
}

pub fn unpack_ring(data: &[u8], n: usize, bits: u32) -> Vec<Fr> {
    (0..n)
        .map(|i| {
            let base = i * bits as usize;
            let mut v: Fr = 0;
            for b in 0..bits as usize {
                let idx = base + b;
                if data[idx / 8] >> (idx % 8) & 1 == 1 {
                    v |= 1 << b;
                }
            }
            v
        })
        .collect()
}

/// [`reveal_components_batch_party`] with the payload packed at `bits`.
///
/// Same protocol, honest width. Use this wherever the reveal traffic is a
/// material share of the total; `RevealShareBatch`'s `u128` payload is 75–84%
/// padding at the widths this crate actually uses.
pub fn reveal_components_packed_party<T: Transport>(
    shares: &[PartyShare],
    bits: u32,
    base_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<[Fr; 3]> {
    if shares.is_empty() {
        return Vec::new();
    }
    let mine: Vec<Fr> = shares.iter().map(|s| s.left).collect();
    mbx.send(
        (me + 1) % N,
        serialize_msg(&MpcMsg::RevealPackedBatch {
            base_id,
            bits,
            data: pack_ring(&mine, bits),
        }),
    );
    let (_, msg) = mbx
        .recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::RevealPackedBatch { base_id: b, .. } if *b == base_id),
        )
        .unwrap_or_else(|| panic!("reveal_components_packed_party: timeout id={base_id}"));
    let received = match msg {
        MpcMsg::RevealPackedBatch { data, .. } => unpack_ring(&data, shares.len(), bits),
        _ => unreachable!("predicate guarantees RevealPackedBatch"),
    };
    shares
        .iter()
        .zip(received)
        .map(|(s, got)| {
            let mut c = [0 as Fr; 3];
            c[me % N] = s.left;
            c[(me + 1) % N] = s.right;
            c[(me + 2) % N] = got;
            c
        })
        .collect()
}

pub fn reveal_components_batch_party<T: Transport>(
    shares: &[PartyShare],
    base_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<[Fr; 3]> {
    if shares.is_empty() {
        return Vec::new();
    }
    let mine: Vec<Fr> = shares.iter().map(|s| s.left).collect();
    mbx.send(
        (me + 1) % N,
        serialize_msg(&MpcMsg::RevealShareBatch {
            base_id,
            values: mine,
        }),
    );
    let (_, msg) = mbx
        .recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::RevealShareBatch { base_id: b, .. } if *b == base_id),
        )
        .unwrap_or_else(|| panic!("reveal_components_batch_party: timeout id={base_id}"));
    let received = match msg {
        MpcMsg::RevealShareBatch { values, .. } => values,
        _ => unreachable!("predicate guarantees RevealShareBatch"),
    };
    assert_eq!(received.len(), shares.len(), "reveal batch length mismatch");
    shares
        .iter()
        .zip(received)
        .map(|(s, got)| {
            // This party's `left` is component `me`, its `right` component
            // `me+1`; the value it just received is component `me+2`. Placing
            // them by index makes every party agree on the ordering, which is
            // what lets `wrap3` be computed identically everywhere.
            let mut c = [0 as Fr; 3];
            c[me % N] = s.left;
            c[(me + 1) % N] = s.right;
            c[(me + 2) % N] = got;
            c
        })
        .collect()
}

pub fn mul_party<T: Transport>(
    a: &PartyShare,
    b: &PartyShare,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> PartyShare {
    mul_batch_party(&[(a.clone(), b.clone())], op_id, me, mbx)
        .pop()
        .expect("one product")
}
/// Batched [`mul_party`]: one message per party for the whole batch.
///
/// Batches wider than this cannot be given collision-free α counters under
/// the [`alpha_ctr`] encoding. It is astronomically larger than any real
/// batch (a full network is ~10⁵ multiplications), so the cap only guards a
/// programming error.
pub const MAX_BATCH: usize = 1 << 40;

/// Globally-unique PRF counter for pair `i` of the multiplication batch
/// based at `op_id`. Operation ids are handed out uniquely (the backend's
/// `alloc_id`/`alloc_block`, and `take_id` inside the bit protocols), so
/// giving each op_id its own disjoint `2^40`-wide slot makes every single
/// multiplication's α distinct — even a batch of 10⁵ products cannot reach
/// into the next op_id's slot. This is what lets one op_id cover an
/// arbitrarily wide batch without the old `op_id + i` ranges overlapping the
/// next round.
fn alpha_ctr(op_id: u64, i: usize) -> u64 {
    debug_assert!(i < MAX_BATCH, "batch exceeds the α-counter slot width");
    (op_id << 40).wrapping_add(i as u64)
}

/// Each product draws its own `α` from a **globally distinct** counter, so
/// no two multiplications — within this batch or across calls — are ever
/// masked by the same randomness. Reusing an α would be harmless for
/// correctness (it always cancels in the sum) but a privacy leak: two
/// products sharing an α expose their difference on the wire. See
/// [`alpha_ctr`] for how collisions are ruled out.
/// A batch of **dot products**, each resharing **once**.
///
/// # Why this is not `mul_batch_party` plus local sums
///
/// It computes the same values, and the products are already summed locally in
/// both — but the resharing granularity differs, and that is the whole cost.
/// Resharing per product sends one ring element per *multiplication*; resharing
/// per dot product sends one per *output*. The cross-terms can be accumulated
/// before the message because summation is local:
///
/// ```text
/// Σₖ (aₖ,ᵢ·bₖ,ᵢ + aₖ,ᵢ·bₖ,ᵢ₊₁ + aₖ,ᵢ₊₁·bₖ,ᵢ) + αᵢ
/// ```
///
/// summed over the three parties is `Σₖ aₖ·bₖ`, with the α telescoping to zero
/// exactly as for a single product.
///
/// This is what Falcon's cost table means by `MatMul (x×y)(y×z) = k·x·z` — `x·z`
/// is the number of *outputs*, not of products. On the MedMNIST CNN the two
/// differ by **32×**: 239 520 products against 7 474 outputs.
///
/// One α per dot product, not per term: each opened value must be independently
/// masked, and reusing an α across dots would expose their difference.
pub fn dot_batch_party<T: Transport>(
    dots: &[&[(PartyShare, PartyShare)]],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    dot_batch_party_at(dots, 128, base_op_id, me, mbx)
}

/// [`dot_batch_party`] with the resharing carried at `bits`.
///
/// Same precondition as [`mul_batch_party_at`]: every consumer of the result
/// must work at `bits` or below. True throughout `falcon_mpc`, which is
/// `l_bits`-consistent end to end; not true of the generic backend path, where
/// widths mix.
pub fn dot_batch_party_at<T: Transport>(
    dots: &[&[(PartyShare, PartyShare)]],
    bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    if dots.is_empty() {
        return Vec::new();
    }
    let wmask: Fr = if bits >= 128 {
        Fr::MAX
    } else {
        (1u128 << bits) - 1
    };
    let keys = mbx.prf_keys(me);
    let z_mine: Vec<u128> = dots
        .iter()
        .enumerate()
        .map(|(i, terms)| {
            let acc = terms.iter().fold(0u128, |acc, (a, b)| {
                acc.wrapping_add(a.left.wrapping_mul(b.left))
                    .wrapping_add(a.left.wrapping_mul(b.right))
                    .wrapping_add(a.right.wrapping_mul(b.left))
            });
            acc.wrapping_add(keys.alpha(alpha_ctr(base_op_id, i))) & wmask
        })
        .collect();

    mbx.send(
        (me + N - 1) % N,
        serialize_msg(&MpcMsg::TaggedZBatch {
            base_op_id,
            bits,
            data: pack_ring(&z_mine, bits),
        }),
    );
    let (_, msg) = mbx
        .recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::TaggedZBatch { base_op_id: b, .. } if *b == base_op_id),
        )
        .unwrap_or_else(|| panic!("dot_batch_party: timeout on z for op_id={base_op_id}"));
    let z_next = match msg {
        MpcMsg::TaggedZBatch { data, bits, .. } => unpack_ring(&data, z_mine.len(), bits),
        _ => unreachable!("predicate guarantees TaggedZBatch"),
    };
    z_mine
        .into_iter()
        .zip(z_next)
        .map(|(left, right)| PartyShare { left, right })
        .collect()
}

pub fn mul_batch_party<T: Transport>(
    pairs: &[(PartyShare, PartyShare)],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    mul_batch_party_at(pairs, 128, base_op_id, me, mbx)
}

/// [`mul_batch_party`] with the sharing reduced to `bits` before it goes on the
/// wire.
///
/// # Why the width has to be given, and why it is safe
///
/// The resharing message carried a `u128` per product regardless of how wide
/// the shares actually were. For the bit-level multiplications inside `a2b` —
/// which is what every comparison and ReLU in this crate bottoms out in — that
/// is **128 bits on the wire per 1 bit of content**. Measured on sign
/// extraction: 449 elements per operation, 7 184 wire bytes, ~56 bytes of
/// content.
///
/// Reducing to `bits` is sound because the whole sharing is modulo a power of
/// two: `2^bits | 2^128`, so a share reduced mod `2^bits` still reconstructs the
/// value mod `2^bits`, and the α mask stays zero-sum (`F(kᵢ) − F(kᵢ₋₁)` telescopes
/// under any modulus). Products of reduced shares remain correct mod `2^bits`,
/// which is all any caller interpreting at that width can observe.
///
/// # ⚠️ Not currently usable below 128, and why
///
/// Reducing genuinely saves wire bytes — switching `a2b`'s ten internal calls to
/// `nbits = 32` took sign extraction from 7 184 to 1 796 B, a clean 4×. It also
/// **breaks the crate**, and the reason is worth recording.
///
/// Shares here are full-width `u128` *everywhere*; `nbits` is an
/// interpretation attached to a `ShareRef`, never enforced on the sharing
/// itself. Reducing a share's components mod `2^bits` leaves them summing to
/// `v + k·2^bits` rather than `v` over `Z_{2^128}`. Every operation that stays
/// within that width is fine — but a *downstream* consumer working at full
/// width, such as the multiply inside `max2`, then computes on `v + k·2^bits`.
/// Observed directly: `max2(9, 6)` returned 22 020 105.
///
/// So the 4× is real but is not a local change. It needs the sharing to be
/// consistently modulo `2^nbits` — reduction at every operation and a `reveal`
/// that masks — which is a crate-wide refactor, not a packing tweak. The
/// parameter and the packed wire format are kept here so that work has
/// somewhere to land; **every caller passes 128 today.**
pub fn mul_batch_party_at<T: Transport>(
    pairs: &[(PartyShare, PartyShare)],
    bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    if pairs.is_empty() {
        return Vec::new();
    }
    assert!(
        pairs.len() <= MAX_BATCH,
        "multiplication batch too large for α counters"
    );
    let keys = mbx.prf_keys(me);
    let wmask: Fr = if bits >= 128 {
        Fr::MAX
    } else {
        (1u128 << bits) - 1
    };
    // Local: three of the nine cross terms, re-randomised by a zero-sum α, then
    // reduced to the width actually in use.
    let z_mine: Vec<u128> = pairs
        .iter()
        .enumerate()
        .map(|(i, (a, b))| {
            a.left
                .wrapping_mul(b.left)
                .wrapping_add(a.left.wrapping_mul(b.right))
                .wrapping_add(a.right.wrapping_mul(b.left))
                .wrapping_add(keys.alpha(alpha_ctr(base_op_id, i)))
                & wmask
        })
        .collect();

    // Resharing: party i needs z_{i+1} to hold (z_i, z_{i+1}), so each party
    // sends its components to its predecessor. One directed message, not a
    // broadcast.
    mbx.send(
        (me + N - 1) % N,
        serialize_msg(&MpcMsg::TaggedZBatch {
            base_op_id,
            bits,
            data: pack_ring(&z_mine, bits),
        }),
    );
    let (_, msg) = mbx
        .recv_matching(
            TIMEOUT_MS,
            |m| matches!(m, MpcMsg::TaggedZBatch { base_op_id: b, .. } if *b == base_op_id),
        )
        .unwrap_or_else(|| panic!("mul_batch_party: timeout on z for op_id={base_op_id}"));
    let z_next = match msg {
        MpcMsg::TaggedZBatch { data, bits: b, .. } => unpack_ring(&data, z_mine.len(), b),
        _ => unreachable!("predicate guarantees TaggedZBatch"),
    };
    assert_eq!(
        z_next.len(),
        z_mine.len(),
        "batch length mismatch in resharing"
    );

    z_mine
        .into_iter()
        .zip(z_next)
        .map(|(left, right)| PartyShare { left, right })
        .collect()
}
/// Operation ids reserved by one `a2b_party` call. Each internal
/// `mul_batch` round derives its correlated randomness from a distinct id
/// range; the block gives generous headroom over the ~9 rounds used.
pub const A2B_ID_BLOCK: u64 = 64;

/// Replicated sharing of a value held by the two parties that own additive
/// component `owner` — **no communication**.
///
/// A replicated sharing puts value `v` in one additive component and zero in
/// the others. Component `owner` is held as `left` by party `owner` and as
/// `right` by party `owner−1`, so exactly those two parties supply `v` and
/// the third contributes nothing and learns nothing. This is what makes the
/// first step of bit decomposition free.
pub(crate) fn share_component(value: Fr, owner: usize, me: PartyId) -> PartyShare {
    PartyShare {
        left: if me == owner { value } else { 0 },
        right: if (me + 1) % N == owner { value } else { 0 },
    }
}

/// XOR of two bit-shares in the arithmetic encoding: `a ⊕ b = a + b − 2ab`,
/// given their already-computed product.
fn xor_from_product(a: &PartyShare, b: &PartyShare, ab: &PartyShare) -> PartyShare {
    sub_party(&add_party(a, b), &scalar_mul_party(ab, 2))
}

/// Kogge–Stone parallel-prefix adder on bit-shares: the low `nbits` bits of
/// `u + v`. Costs `2 + ⌈log₂ nbits⌉` communication rounds.
fn ppa_add<T: Transport>(
    u: &[PartyShare],
    v: &[PartyShare],
    nbits: usize,
    next_id: &mut u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let take_id = |n: &mut u64| {
        let b = *n;
        *n += 2;
        b
    };
    // generate g = u∧v and propagate p = u⊕v — one product serves both.
    let pairs: Vec<_> = (0..nbits).map(|k| (u[k].clone(), v[k].clone())).collect();
    let base = take_id(next_id);
    let uv = mul_batch_party(&pairs, base, me, mbx);
    let mut g = uv.clone();
    let mut p: Vec<PartyShare> = (0..nbits)
        .map(|k| xor_from_product(&u[k], &v[k], &uv[k]))
        .collect();
    let p_leaf = p.clone();

    let mut d = 1usize;
    while d < nbits {
        let mut pr = Vec::with_capacity(2 * (nbits - d));
        for k in d..nbits {
            pr.push((p[k].clone(), g[k - d].clone()));
            pr.push((p[k].clone(), p[k - d].clone()));
        }
        let base = take_id(next_id);
        let out = mul_batch_party(&pr, base, me, mbx);
        let (mut ng, mut np) = (g.clone(), p.clone());
        for (i, k) in (d..nbits).enumerate() {
            // A group cannot both generate and propagate, so `g` and
            // `p ∧ g'` are mutually exclusive and their XOR is plain
            // addition — no communication for the combine.
            ng[k] = add_party(&g[k], &out[2 * i]);
            np[k] = out[2 * i + 1].clone();
        }
        g = ng;
        p = np;
        d <<= 1;
    }

    // carry into position k is the group generate over 0..k−1
    if nbits == 1 {
        return vec![p_leaf[0].clone()];
    }
    let pairs: Vec<_> = (1..nbits)
        .map(|k| (p_leaf[k].clone(), g[k - 1].clone()))
        .collect();
    let base = take_id(next_id);
    let prod = mul_batch_party(&pairs, base, me, mbx);
    let mut out = Vec::with_capacity(nbits);
    out.push(p_leaf[0].clone()); // nothing carries into the LSB
    for k in 1..nbits {
        out.push(xor_from_product(&p_leaf[k], &g[k - 1], &prod[k - 1]));
    }
    out
}

/// Arithmetic-to-bit decomposition (ABY3 bit decomposition).
///
/// `op_id` must be the base of a reserved block of [`A2B_ID_BLOCK`] ids.
///
/// # Protocol
///
/// 1. **Binary-share the three additive components** — free, via
///    [`share_component`]: `a = a₀ + a₁ + a₂` and each `aⱼ` is already known
///    to exactly the two parties a replicated sharing of it requires.
/// 2. **Carry-save reduce 3 → 2.** In this arithmetic bit encoding
///    `x⊕y⊕z = x+y+z − 2(xy+xz+yz) + 4xyz` and
///    `maj(x,y,z) = xy+xz+yz − 2xyz`, so both the sum and carry vectors fall
///    out of the *same* four products — two rounds, not four.
/// 3. **Add the remaining two** with a Kogge–Stone prefix adder.
///
/// This replaces an earlier version that masked with a value derived from
/// `op_id` alone and opened `a − r`. Because `op_id` is a public lockstep
/// counter that mask carried no secret entropy, so the input was recoverable
/// by anyone — that version was not private. Nothing here is ever opened:
/// every product goes through Beaver triples, which is why the carry chain
/// now costs communication instead of being "free".
pub fn a2b_party<T: Transport>(
    a: &PartyShare,
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let mut next_id = op_id;
    let take_id = |n: &mut u64| {
        let b = *n;
        *n += 2;
        b
    };

    // ── 1. free binary sharings of the three additive components ──
    let comp: Vec<Vec<PartyShare>> = (0..N)
        .map(|j| {
            let known: Fr = if j == me {
                a.left
            } else if j == (me + 1) % N {
                a.right
            } else {
                0
            };
            (0..nbits)
                .map(|k| share_component((known >> k) & 1, j, me))
                .collect()
        })
        .collect();
    let (x, y, z) = (&comp[0], &comp[1], &comp[2]);

    // ── 2. carry-save 3 → 2 (four products, depth 2) ──
    let mut pairs = Vec::with_capacity(3 * nbits);
    for k in 0..nbits {
        pairs.push((x[k].clone(), y[k].clone()));
    }
    for k in 0..nbits {
        pairs.push((x[k].clone(), z[k].clone()));
    }
    for k in 0..nbits {
        pairs.push((y[k].clone(), z[k].clone()));
    }
    let base = take_id(&mut next_id);
    let pw = mul_batch_party(&pairs, base, me, mbx);
    let (xy, xz, yz) = (
        &pw[0..nbits],
        &pw[nbits..2 * nbits],
        &pw[2 * nbits..3 * nbits],
    );

    let triple: Vec<_> = (0..nbits).map(|k| (xy[k].clone(), z[k].clone())).collect();
    let base = take_id(&mut next_id);
    let xyz = mul_batch_party(&triple, base, me, mbx);

    let mut sum = Vec::with_capacity(nbits);
    let mut carry = Vec::with_capacity(nbits);
    for k in 0..nbits {
        let sym = add_party(&add_party(&xy[k], &xz[k]), &yz[k]); // xy+xz+yz
        let lin = add_party(&add_party(&x[k], &y[k]), &z[k]); // x+y+z
        sum.push(add_party(
            &sub_party(&lin, &scalar_mul_party(&sym, 2)),
            &scalar_mul_party(&xyz[k], 4),
        ));
        carry.push(sub_party(&sym, &scalar_mul_party(&xyz[k], 2)));
    }

    // ── 3. a = sum + (carry ≪ 1), mod 2^nbits ──
    let mut shifted = Vec::with_capacity(nbits);
    shifted.push(share_public(0, me));
    shifted.extend(carry.iter().take(nbits - 1).cloned());
    ppa_add(&sum, &shifted, nbits, &mut next_id, me, mbx)
}

/// Batched [`a2b_party`]: decompose `count` shared values at once.
///
/// Every internal multiplication round is batched **across all lanes**, so
/// the whole batch costs the same ~9 communication rounds as a single
/// decomposition — versus `count · 9` when the values are decomposed one at a
/// time in a loop. This is the difference between ~10 rounds and ~12 000 for
/// a CNN's ReLU layer, and it is where most of the MPC latency lives.
///
/// Returns one bit-vector (LSB first, length `nbits`) per input value.
pub fn a2b_batch_party<T: Transport>(
    values: &[PartyShare],
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<Vec<PartyShare>> {
    // Full width: the resulting bit-shares may be consumed by *any* downstream
    // arithmetic, including multiplication against wide operands.
    a2b_batch_party_at(values, nbits, 128, op_id, me, mbx)
}

/// [`a2b_batch_party`] with the bit-shares carried at `wire_bits` on the wire.
///
/// # When this is safe, and it is a real precondition
///
/// A bit-share here is an *arithmetic* sharing of a 0/1 value (`a ⊕ b = a+b−2ab`),
/// so its components are full-width ring elements even though the value is one
/// bit. Narrowing them to `wire_bits` keeps the sharing valid **modulo
/// `2^wire_bits`** and no wider. Every consumer must therefore work at that
/// width or below.
///
/// That is false for the general `lt`/`max2` path — its output bit is
/// multiplied by a 20-bit operand, and narrowing broke it (`max2(9,6)` returned
/// 22 020 105). It is **true** for a caller that reads the result purely as a
/// bit, which is what `falcon_mpc::truncate_batch` does with the borrow.
///
/// `wire_bits` must be at least 2: the XOR encoding evaluates `a+b−2ab`, so one
/// bit is not enough room even though the value is.
///
/// Worth the trouble because this dominates: the borrow's 9-bit `a2b` is ~108
/// ring elements, and at `u128` that is 1 728 B against 54 B of content — 96%
/// of an all-RSS network's traffic went here.
pub fn a2b_batch_party_at<T: Transport>(
    values: &[PartyShare],
    nbits: usize,
    wire_bits: u32,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<Vec<PartyShare>> {
    assert!(
        wire_bits >= 2,
        "the XOR encoding needs at least two bits of room"
    );
    let count = values.len();
    if count == 0 {
        return Vec::new();
    }
    let mut next_id = op_id;
    let take_id = |n: &mut u64| {
        let b = *n;
        *n += 2;
        b
    };

    // ── 1. free binary sharings of the three additive components, per lane ──
    // comp[lane] = (x, y, z), each an nbits vector.
    let comp: Vec<(Vec<PartyShare>, Vec<PartyShare>, Vec<PartyShare>)> = values
        .iter()
        .map(|a| {
            let per_owner: Vec<Vec<PartyShare>> = (0..N)
                .map(|j| {
                    let known: Fr = if j == me {
                        a.left
                    } else if j == (me + 1) % N {
                        a.right
                    } else {
                        0
                    };
                    (0..nbits)
                        .map(|k| share_component((known >> k) & 1, j, me))
                        .collect()
                })
                .collect();
            let mut it = per_owner.into_iter();
            (it.next().unwrap(), it.next().unwrap(), it.next().unwrap())
        })
        .collect();

    // ── 2. carry-save 3 → 2, all lanes in two batched rounds ──
    // Round A: xy, xz, yz for every lane and bit.
    let mut pairs = Vec::with_capacity(3 * count * nbits);
    for (x, y, z) in &comp {
        for k in 0..nbits {
            pairs.push((x[k].clone(), y[k].clone()));
        }
        for k in 0..nbits {
            pairs.push((x[k].clone(), z[k].clone()));
        }
        for k in 0..nbits {
            pairs.push((y[k].clone(), z[k].clone()));
        }
    }
    let base = take_id(&mut next_id);
    let pw = mul_batch_party_at(&pairs, wire_bits, base, me, mbx);

    // Round B: xyz for every lane and bit.
    let mut triple = Vec::with_capacity(count * nbits);
    for (lane, (_x, _y, z)) in comp.iter().enumerate() {
        let xy = &pw[lane * 3 * nbits..lane * 3 * nbits + nbits];
        for k in 0..nbits {
            triple.push((xy[k].clone(), z[k].clone()));
        }
    }
    let base = take_id(&mut next_id);
    let xyz_all = mul_batch_party_at(&triple, wire_bits, base, me, mbx);

    // Per-lane sum and shifted-carry, all local.
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
            let sym = add_party(&add_party(&xy[k], &xz[k]), &yz[k]);
            let lin = add_party(&add_party(&x[k], &y[k]), &z[k]);
            sum.push(add_party(
                &sub_party(&lin, &scalar_mul_party(&sym, 2)),
                &scalar_mul_party(&xyz[k], 4),
            ));
            carry.push(sub_party(&sym, &scalar_mul_party(&xyz[k], 2)));
        }
        let mut shifted = Vec::with_capacity(nbits);
        shifted.push(share_public(0, me));
        shifted.extend(carry.into_iter().take(nbits - 1));
        sums.push(sum);
        shifts.push(shifted);
    }

    // ── 3. one batched prefix-adder over all lanes ──
    ppa_add_batch(&sums, &shifts, nbits, wire_bits, &mut next_id, me, mbx)
}

/// Batched [`ppa_add`]: `count` independent `nbits`-wide Kogge–Stone adders,
/// every layer's products flattened into one multiplication round.
#[allow(clippy::too_many_arguments)]
fn ppa_add_batch<T: Transport>(
    u: &[Vec<PartyShare>],
    v: &[Vec<PartyShare>],
    nbits: usize,
    wire_bits: u32,
    next_id: &mut u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<Vec<PartyShare>> {
    let count = u.len();
    let take_id = |n: &mut u64| {
        let b = *n;
        *n += 2;
        b
    };

    // init g = u∧v, p = u⊕v — one product per (lane, bit).
    let mut pairs = Vec::with_capacity(count * nbits);
    for lane in 0..count {
        for k in 0..nbits {
            pairs.push((u[lane][k].clone(), v[lane][k].clone()));
        }
    }
    let base = take_id(next_id);
    let uv = mul_batch_party_at(&pairs, wire_bits, base, me, mbx);
    let mut g: Vec<Vec<PartyShare>> = (0..count)
        .map(|l| uv[l * nbits..l * nbits + nbits].to_vec())
        .collect();
    let mut p: Vec<Vec<PartyShare>> = (0..count)
        .map(|l| {
            (0..nbits)
                .map(|k| xor_from_product(&u[l][k], &v[l][k], &uv[l * nbits + k]))
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
        let base = take_id(next_id);
        let out = mul_batch_party_at(&pr, wire_bits, base, me, mbx);
        let width = nbits - d;
        for lane in 0..count {
            let (mut ng, mut np) = (g[lane].clone(), p[lane].clone());
            for (i, k) in (d..nbits).enumerate() {
                let o = 2 * (lane * width + i);
                ng[k] = add_party(&g[lane][k], &out[o]);
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
    let base = take_id(next_id);
    let prod = mul_batch_party_at(&pairs, wire_bits, base, me, mbx);
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

/// Reveal a bit share to all parties.
pub fn reveal_bit_to_all<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> bool {
    let v = reveal(share, share_id, None, me, mbx);
    v % 2 != 0
}

#[cfg(test)]
fn party_pairs(s: &crate::crypto::types::Share3) -> Vec<PartyShare> {
    vec![
        PartyShare {
            left: s.0,
            right: s.1,
        },
        PartyShare {
            left: s.1,
            right: s.2,
        },
        PartyShare {
            left: s.2,
            right: s.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;

    /// The batched reveal must open the same values as the per-share [`reveal`],
    /// and must do it in **one** message per party rather than `N−1` per value.
    ///
    /// The message count is half the point: `Π_WA` opens one masked value per
    /// DReLU, so an unbatched reveal would make the wrap protocol's cost an
    /// artefact of the API.
    #[test]
    #[ignore] // real 3-party transport
    fn batched_reveal_matches_the_scalar_one_in_a_single_round() {
        use crate::bench::comm::CountingTransport;
        const N_VALS: usize = 512;
        let secrets: Vec<Fr> = (0..N_VALS as u128)
            .map(|i| i.wrapping_mul(0x9E37_79B9))
            .collect();
        let expected = secrets.clone();

        let handles = InprocHandle::create(3);
        let mut comm = None;
        let mut joins = Vec::new();
        for h in handles {
            let counted = CountingTransport::new(h, 3);
            if comm.is_none() {
                comm = Some(counted.handle());
            }
            let secrets = secrets.clone();
            joins.push(std::thread::spawn(move || {
                let mbx = Mailbox::new(counted);
                let me = mbx.id();
                // Owner 0 supplies each value; sharing a held value is free.
                let shares: Vec<PartyShare> =
                    secrets.iter().map(|&v| share_component(v, 0, me)).collect();
                reveal_batch_party(&shares, 7, me, &mbx)
            }));
        }
        let outs: Vec<Vec<Fr>> = joins
            .into_iter()
            .map(|j| j.join().expect("panic"))
            .collect();
        for o in &outs {
            assert_eq!(o, &expected, "every party must open the same values");
        }
        let stats = comm.unwrap().stats();
        assert_eq!(
            stats.messages, 1,
            "a batched reveal is one directed message per party, got {}",
            stats.messages
        );
    }

    /// Component sharings must be constructible with no communication and
    /// must reconstruct the original additive components. This is what makes
    /// the first phase of bit decomposition free, and it is also where a
    /// party could wrongly place a value it does not own — worth pinning.
    #[test]
    fn component_sharings_are_free_and_consistent() {
        let secret: Fr = 0xDEAD_BEEF;
        let shares = crate::crypto::arith::share_secret(secret);
        let parties = party_pairs(&shares);

        for j in 0..N {
            // Each party supplies only what it already holds.
            let built: Vec<PartyShare> = (0..N)
                .map(|me| {
                    let known: Fr = if j == me {
                        parties[me].left
                    } else if j == (me + 1) % N {
                        parties[me].right
                    } else {
                        0
                    };
                    share_component(known, j, me)
                })
                .collect();
            // The party that owns neither end of component j contributes
            // nothing — it never has to know the value.
            let idle = (j + 1) % N;
            assert_eq!(built[idle].left, 0);
            assert_eq!(built[idle].right, 0);
            // And the sharing reconstructs that component.
            let total = built.iter().fold(0u128, |acc, p| acc.wrapping_add(p.left));
            assert_eq!(total, parties[j].left, "component {j} must reconstruct");
        }
    }

    /// Run `f` (given `(me, mailbox)`) concurrently across 3 in-process
    /// parties and collect their results in party-index order.
    fn run_parties<R: Send + 'static>(
        f: impl Fn(PartyId, Mailbox<InprocHandle>) -> R + Send + Sync + Clone + 'static,
    ) -> Vec<R> {
        let handles = InprocHandle::create(3);
        let joins: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(me, h)| {
                let f = f.clone();
                std::thread::spawn(move || f(me, Mailbox::new(h)))
            })
            .collect();
        joins.into_iter().map(|j| j.join().unwrap()).collect()
    }

    /// `share_input_batch` must be indistinguishable from a run of
    /// `share_input`, element by element — it is the same protocol with the
    /// round structure collapsed, and the only reason to prefer it is cost.
    /// If it ever diverged, the scheme switch would import wrong activations
    /// and every downstream exactness test would fail for a reason far from
    /// the cause, so the equivalence is pinned here at the protocol level.
    #[test]
    fn batched_input_sharing_matches_the_scalar_protocol() {
        let secrets: Vec<Fr> = vec![0, 1, 42, 0xDEAD_BEEF, Fr::MAX, 7];
        let n = secrets.len();

        for owner in 0..N {
            let ss = secrets.clone();
            let batched = run_parties(move |me, mbx| {
                let vals: Vec<Fr> = if me == owner {
                    ss.clone()
                } else {
                    vec![0; ss.len()]
                };
                let shares = share_input_batch(&vals, 0, owner, me, &mbx);
                (0..shares.len())
                    .map(|k| reveal(&shares[k], 1000 + k as u64, None, me, &mbx))
                    .collect::<Vec<Fr>>()
            });
            let ss = secrets.clone();
            let scalar = run_parties(move |me, mbx| {
                (0..ss.len())
                    .map(|k| {
                        let v = if me == owner { ss[k] } else { 0 };
                        let sh = share_input(v, k as u64, owner, me, &mbx);
                        reveal(&sh, 1000 + k as u64, None, me, &mbx)
                    })
                    .collect::<Vec<Fr>>()
            });
            assert_eq!(
                batched[0], secrets,
                "batched sharing must reconstruct the inputs"
            );
            assert_eq!(
                batched[0], scalar[0],
                "batched and scalar input sharing must agree (owner {owner})"
            );
            assert_eq!(batched[0].len(), n);
        }
    }

    /// Triple-free multiplication is correct *and* leaks nothing on the wire.
    /// Correctness: `Σzᵢ = xy`. Privacy: a single party's outbound message
    /// (`z_i`) is masked by `αᵢ`, which depends on a PRF key that party does
    /// not hold both halves of — so its value is not a function of the
    /// inputs alone. We check the weaker observable consequence that the
    /// three `α` cancel (`Σαᵢ = 0`), which is what makes the masking free,
    /// and that flipping a hidden input does not change what one victim
    /// party can compute from its own view — i.e. no `op_id`-only recovery
    /// like the old `a2b` had.
    #[test]
    fn triple_free_mul_is_correct_and_reshares_are_masked() {
        // Correctness across the transport.
        let out = run_parties(|me, mbx| {
            let a = share_input(if me == 0 { 7 } else { 0 }, 0, 0, me, &mbx);
            let b = share_input(if me == 0 { 6 } else { 0 }, 1, 0, me, &mbx);
            let p = mul_party(&a, &b, 100, me, &mbx);
            reveal(&p, 200, None, me, &mbx)
        });
        assert_eq!(out[0], 42, "7*6 must reveal 42");

        // Privacy of the reshared component: recompute exactly what party 1
        // sends to party 0 (its z_1) for two different secret inputs. If z_1
        // were a public function of op_id (the old bug's shape), it would be
        // identical regardless of the secret; the α mask must make it differ.
        use crate::crypto::net::PrfKeys;
        let keys = PrfKeys {
            mine: [1u8; 32],
            prev: [2u8; 32],
        };
        let z1 = |xr: u128| {
            // party 1 holds (x1,x2),(y1,y2); vary its x-share.
            let a = PartyShare { left: xr, right: 3 };
            let b = PartyShare { left: 4, right: 5 };
            a.left
                .wrapping_mul(b.left)
                .wrapping_add(a.left.wrapping_mul(b.right))
                .wrapping_add(a.right.wrapping_mul(b.left))
                .wrapping_add(keys.alpha(100))
        };
        assert_ne!(
            z1(11),
            z1(12),
            "reshared component must depend on the secret share"
        );
        // And the α that masks it is genuinely zero-sum across a key ring.
        let k0 = PrfKeys {
            mine: [1u8; 32],
            prev: [3u8; 32],
        };
        let k1 = PrfKeys {
            mine: [2u8; 32],
            prev: [1u8; 32],
        };
        let k2 = PrfKeys {
            mine: [3u8; 32],
            prev: [2u8; 32],
        };
        assert_eq!(
            k0.alpha(100)
                .wrapping_add(k1.alpha(100))
                .wrapping_add(k2.alpha(100)),
            0,
            "the three parties' α must cancel"
        );
    }

    /// Batched decomposition must reproduce plaintext binary expansion for
    /// every lane, including two's-complement negatives — i.e. flattening the
    /// per-lane products into shared rounds changes nothing but the round
    /// count.
    #[test]
    fn a2b_batch_matches_plaintext() {
        let nbits = 16;
        let vals: Vec<i64> = vec![0, 3, 255, -1, 1024, -300, 4095, -2048];
        let expected: Vec<u128> = vals
            .iter()
            .map(|&v| (v as i128).rem_euclid(1i128 << nbits) as u128)
            .collect();
        let got = run_parties(move |me, mbx| {
            let shares: Vec<PartyShare> = vals
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let fr: Fr = if me == 0 {
                        (v as i128).rem_euclid(1i128 << nbits) as u128
                    } else {
                        0
                    };
                    share_input(fr, i as u64, 0, me, &mbx)
                })
                .collect();
            let bit_sets = a2b_batch_party(&shares, nbits, 1_000, me, &mbx);
            bit_sets
                .iter()
                .enumerate()
                .map(|(i, bits)| {
                    bits.iter().enumerate().fold(0u128, |acc, (k, b)| {
                        acc | ((reveal(b, 20_000 + (i * nbits + k) as u64, None, me, &mbx) & 1)
                            << k)
                    })
                })
                .collect::<Vec<u128>>()
        });
        assert_eq!(got[0], expected, "batched a2b must match plaintext");
    }

    /// The rebuilt bit decomposition must agree with plaintext binary
    /// expansion, including two's-complement negatives (the MSB is what
    /// `lt` returns, so sign handling is load-bearing).
    #[test]
    fn a2b_matches_plaintext_binary_expansion() {
        let nbits = 12;
        let cases: Vec<i64> = vec![0, 1, 2, 5, 255, 1024, 4095, -1, -2, -7, -2048];
        for &v in &cases {
            let out = run_parties(move |me, mbx| {
                let fr: Fr = if me == 0 {
                    (v as i128).rem_euclid(1i128 << nbits) as u128
                } else {
                    0
                };
                let sh = share_input(fr, 0, me, 0, &mbx);
                let bits = a2b_party(&sh, nbits, 1_000, me, &mbx);
                bits.iter()
                    .enumerate()
                    .map(|(i, b)| reveal(b, 5_000 + i as u64, None, me, &mbx) & 1)
                    .collect::<Vec<u128>>()
            });
            let want = (v as i128).rem_euclid(1i128 << nbits) as u128;
            let got = out[0]
                .iter()
                .enumerate()
                .fold(0u128, |acc, (i, b)| acc | (b << i));
            assert_eq!(got, want, "a2b({v}) gave {got:#x}, want {want:#x}");
        }
    }

    /// Secret-share `values[i]` (owned by party 0) for i in 0..values.len(),
    /// returning this party's `PartyShare` for each.
    fn share_all<T: Transport>(values: &[i64], me: PartyId, mbx: &Mailbox<T>) -> Vec<PartyShare> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let fr = if me == 0 { v as u128 } else { 0 };
                share_input(fr, i as u64 * 10, 0, me, mbx)
            })
            .collect()
    }

    #[test]
    fn mul_batch_matches_sequential_mul_party() {
        let a_vals = [3i64, -5, 7, 0, 12];
        let b_vals = [4i64, 6, -2, 9, 11];
        let expected: Vec<i64> = a_vals.iter().zip(&b_vals).map(|(a, b)| a * b).collect();

        let results = run_parties(move |me, mbx| {
            let a_shares = share_all(&a_vals, me, &mbx);
            let b_shares = share_all(&b_vals, me, &mbx);
            let pairs: Vec<_> = a_shares.into_iter().zip(b_shares).collect();

            let out = mul_batch_party(&pairs, 1000, me, &mbx);

            // Reveal every product to party 0.
            out.into_iter()
                .enumerate()
                .map(|(i, ps)| reveal(&ps, 2000 + i as u64, Some(0), me, &mbx))
                .collect::<Vec<Fr>>()
        });

        // Only party 0 (the reveal target) has the real sums; the others'
        // `reveal` calls return 0 by design (see `reveal`'s doc).
        let got: Vec<i64> = results[0].iter().map(|&v| v as i128 as i64).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn mul_batch_matches_sequential_mul_party_for_single_pair() {
        // count=1 must behave identically to a plain `mul_party` call.
        let results = run_parties(|me, mbx| {
            let a = share_input(if me == 0 { 6 } else { 0 }, 0, 0, me, &mbx);
            let b = share_input(if me == 0 { 7 } else { 0 }, 1, 0, me, &mbx);
            let batched = mul_batch_party(&[(a.clone(), b.clone())], 100, me, &mbx)
                .into_iter()
                .next()
                .unwrap();
            reveal(&batched, 200, Some(0), me, &mbx) as i128 as i64
        });
        assert_eq!(results[0], 42);
    }

    #[test]
    fn mul_batch_uses_one_round_instead_of_n() {
        use crate::bench::comm::CountingTransport;

        let n = 8usize;
        let a_vals: Vec<i64> = (0..n as i64).collect();
        let b_vals: Vec<i64> = (0..n as i64).map(|i| i + 1).collect();

        let handles = InprocHandle::create(3);
        let joins: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(me, h)| {
                let a_vals = a_vals.clone();
                let b_vals = b_vals.clone();
                std::thread::spawn(move || {
                    let counted = CountingTransport::new(h, 3);
                    let handle = counted.handle();
                    let mbx = Mailbox::new(counted);

                    let a_shares = share_all(&a_vals, me, &mbx);
                    let b_shares = share_all(&b_vals, me, &mbx);
                    let pairs: Vec<_> = a_shares.into_iter().zip(b_shares).collect();

                    let batched_stats_before = handle.stats();
                    let _ = mul_batch_party(&pairs, 1000, me, &mbx);
                    let batched_stats = handle.stats().saturating_sub(&batched_stats_before);

                    // Sequential baseline: n separate mul_party calls on fresh shares.
                    let a2 = share_all(&a_vals, me, &mbx);
                    let b2 = share_all(&b_vals, me, &mbx);
                    let seq_before = handle.stats();
                    for (i, (a, b)) in a2.iter().zip(&b2).enumerate() {
                        let _ = mul_party(a, b, 5000 + i as u64 * 10, me, &mbx);
                    }
                    let seq_stats = handle.stats().saturating_sub(&seq_before);

                    (batched_stats, seq_stats)
                })
            })
            .collect();
        let results: Vec<_> = joins.into_iter().map(|j| j.join().unwrap()).collect();

        let (batched, seq) = results[0];
        assert!(
            batched.messages < seq.messages,
            "batched ({}) should use far fewer messages than sequential ({}) for n={n}",
            batched.messages,
            seq.messages
        );
        // Sequential does n independent rounds -> message count scales with n;
        // batched does it in one round -> message count is O(1) in n.
        assert!(
            seq.messages >= batched.messages * (n as u64) / 2,
            "expected roughly n-fold more messages sequentially: batched={} seq={} n={n}",
            batched.messages,
            seq.messages
        );
    }
}
