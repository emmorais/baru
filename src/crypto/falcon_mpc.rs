//! Falcon's protocols on the **real transport** — the versions whose messages
//! are counted.
//!
//! [`crate::crypto::falcon`] establishes that the arithmetic is right, by local
//! simulation over plaintext summands and exhaustive tests. This module is the
//! other half: the same protocols expressed over
//! [`crate::crypto::fp_protocol`] and the 3-party mailbox, so a run produces
//! message and byte counts comparable to Falcon's published figures and to the
//! repo's own routes.
//!
//! # Rounds, and why the shape matters here
//!
//! `Π_PC` written literally from Algorithm 1 is a loop of `ℓ` iterations each
//! doing a multiplication, then a running product — `2ℓ` sequential rounds at
//! ℓ = 32, which is 64 round trips for one comparison. That is not what Falcon
//! costs, because the dependence structure does not require it:
//!
//! - the `ℓ` products `u[i] = (−1)^β·(x[i] − r[i])` are mutually independent, so
//!   they are **one** batched multiplication;
//! - `Π c[i]` is a reduction, so it is a **tree** of depth `⌈log₂ ℓ⌉`, not a
//!   left fold of depth `ℓ`.
//!
//! Total: `1 + ⌈log₂ ℓ⌉ + 1 + 1` rounds — 8 at ℓ = 32, against 64 for the naive
//! shape. This is exactly the transformation `mpcc`'s ASAP scheduler and
//! `ReassociateReduction` rule perform automatically, which is the point of the
//! eventual comparison: here it is done **by hand**, as Falcon's authors did it,
//! so that "compiled vs hand-written" is a fair fight rather than a comparison
//! against a straw baseline.

use crate::crypto::aby3_protocol::{reveal_components_packed_party, PartyShare};
use crate::crypto::falcon;
use crate::crypto::field::{Fp, FpShare};
use crate::crypto::fp_protocol::{mul_batch, reveal_batch, share_component};
use crate::crypto::net::{Mailbox, PartyId, Transport};
use crate::crypto::types::Fr;

/// Op-id slots reserved by one `private_compare` call: one per batched
/// multiplication round, plus the reveal, with headroom.
pub const PC_ID_BLOCK: u64 = 64;

/// Add a public field element to a shared one — local, and placed in a single
/// additive component so it is not added three times.
fn add_public(s: FpShare, k: Fp, me: PartyId) -> FpShare {
    s + share_component(k, 0, me)
}

/// `Π_PC` (Algorithm 1) over the transport. Returns the revealed bit `β'`; the
/// protocol's output is `β' ⊕ β`, formed by the caller from its `⟦β⟧₂`.
///
/// `bits` are `Z_p` shares of the bits of `x`, least significant first; `r` is
/// public. See [`crate::crypto::field::private_compare`] for the functionality
/// (`x ≥ r`) and its `x ≠ r` precondition.
pub fn private_compare<T: Transport>(
    bits: &[FpShare],
    r: u64,
    beta: &FpShare,
    m: &FpShare,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> bool {
    let l = bits.len();
    assert!(l > 0, "private_compare needs at least one bit");
    let rbit = |i: usize| (r >> i) & 1;
    let mut op = base_op_id;

    // (−1)^β = 1 − 2β, local.
    let sign = add_public(-(*beta + *beta), Fp::ONE, me);

    // w[i] = x[i] ⊕ r[i]; r public, so local either way.
    let w: Vec<FpShare> = (0..l)
        .map(|i| {
            if rbit(i) == 0 {
                bits[i]
            } else {
                add_public(-bits[i], Fp::ONE, me)
            }
        })
        .collect();

    // Round 1: all ℓ of the u[i] at once — they share the `sign` operand but
    // depend on nothing of each other.
    let u_pairs: Vec<(FpShare, FpShare)> = (0..l)
        .map(|i| (sign, add_public(bits[i], -Fp::new(rbit(i)), me)))
        .collect();
    let u = mul_batch(&u_pairs, op, me, mbx);
    op += 1;

    // c[i] = u[i] + 1 + Σ_{k>i} w[k]; suffix sums make this local and linear.
    let mut suffix = vec![FpShare::new(Fp::ZERO, Fp::ZERO); l + 1];
    for i in (0..l).rev() {
        suffix[i] = suffix[i + 1] + w[i];
    }
    let mut layer: Vec<FpShare> = (0..l)
        .map(|i| add_public(u[i] + suffix[i + 1], Fp::ONE, me))
        .collect();

    // Rounds 2..: multiply the c[i] together as a balanced tree.
    while layer.len() > 1 {
        let pairs: Vec<(FpShare, FpShare)> = layer
            .chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| (c[0], c[1]))
            .collect();
        let mut next = mul_batch(&pairs, op, me, mbx);
        op += 1;
        if layer.len() % 2 == 1 {
            next.push(layer[layer.len() - 1]);
        }
        layer = next;
    }

    // Mask by a unit and open: the product is otherwise uniform, so only
    // "was some term zero" escapes.
    let d = mul_batch(&[(layer[0], *m)], op, me, mbx);
    op += 1;
    let opened = reveal_batch(&d, op, me, mbx);
    !opened[0].is_zero()
}

/// `Π_PC` over a whole vector of comparisons at once.
///
/// This — not [`private_compare`] — is what a deployment runs. A convolution
/// layer needs thousands of comparisons that do not depend on one another, so
/// they proceed in lockstep: the round count stays at
/// [`private_compare_rounds`] for the *entire layer*, and each round's message
/// simply carries `n` times as many field elements.
///
/// The distinction dominates the cost. Per-element message framing amortises to
/// nothing, so the marginal cost of a comparison falls to its field elements
/// alone — which is the figure that can be set against Falcon's published
/// communication.
///
/// Every item must share the same bit width.
pub fn private_compare_batch<T: Transport>(
    items: &[(Vec<FpShare>, u64)],
    betas: &[FpShare],
    masks: &[FpShare],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<bool> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    let l = items[0].0.len();
    assert!(
        items.iter().all(|(b, _)| b.len() == l),
        "batch must share one bit width"
    );
    assert_eq!(betas.len(), n, "one blinding bit per item");
    assert_eq!(masks.len(), n, "one mask per item");
    let mut op = base_op_id;

    let signs: Vec<FpShare> = betas
        .iter()
        .map(|b| add_public(-(*b + *b), Fp::ONE, me))
        .collect();

    // One batched round for every u[i] of every item.
    let mut u_pairs = Vec::with_capacity(n * l);
    for (k, (bits, r)) in items.iter().enumerate() {
        for (i, bit) in bits.iter().enumerate() {
            let rbit = Fp::new((r >> i) & 1);
            u_pairs.push((signs[k], add_public(*bit, -rbit, me)));
        }
    }
    let u = mul_batch(&u_pairs, op, me, mbx);
    op += 1;

    // c[i] per item, all local.
    let mut layers: Vec<Vec<FpShare>> = Vec::with_capacity(n);
    for (k, (bits, r)) in items.iter().enumerate() {
        let w: Vec<FpShare> = (0..l)
            .map(|i| {
                if (r >> i) & 1 == 0 {
                    bits[i]
                } else {
                    add_public(-bits[i], Fp::ONE, me)
                }
            })
            .collect();
        let mut suffix = vec![FpShare::new(Fp::ZERO, Fp::ZERO); l + 1];
        for i in (0..l).rev() {
            suffix[i] = suffix[i + 1] + w[i];
        }
        layers.push(
            (0..l)
                .map(|i| add_public(u[k * l + i] + suffix[i + 1], Fp::ONE, me))
                .collect(),
        );
    }

    // Product trees advance in lockstep, so the whole batch costs ⌈log₂ ℓ⌉
    // rounds rather than n·⌈log₂ ℓ⌉.
    while layers[0].len() > 1 {
        let mut pairs = Vec::new();
        for layer in &layers {
            for c in layer.chunks(2).filter(|c| c.len() == 2) {
                pairs.push((c[0], c[1]));
            }
        }
        let prods = mul_batch(&pairs, op, me, mbx);
        op += 1;
        let mut cursor = 0;
        for layer in layers.iter_mut() {
            let full = layer.len() / 2;
            let mut next: Vec<FpShare> = prods[cursor..cursor + full].to_vec();
            cursor += full;
            if layer.len() % 2 == 1 {
                next.push(layer[layer.len() - 1]);
            }
            *layer = next;
        }
    }

    let mask_pairs: Vec<(FpShare, FpShare)> =
        layers.iter().zip(masks).map(|(l, m)| (l[0], *m)).collect();
    let d = mul_batch(&mask_pairs, op, me, mbx);
    op += 1;
    reveal_batch(&d, op, me, mbx)
        .into_iter()
        .map(|v| !v.is_zero())
        .collect()
}

/// What preprocessing supplies for one `Π_WA`, per element.
///
/// All of it is query-independent, which is why Falcon accounts for it offline:
/// a random mask `x` shared over the ring, the same `x`'s bits shared over
/// `Z_p`, and `x`'s own wrap `δ` as a shared bit. The blinding bit and unit mask
/// for the inner `Π_PC` come with it.
///
/// `delta` cannot be computed by any party from its own view — it is
/// `wrap3(x₁,x₂,x₃)` and every party is missing a summand — so it genuinely has
/// to arrive from preprocessing rather than being derived locally.
pub struct WrapPreproc {
    /// Ring shares of the mask `x`.
    pub x: PartyShare,
    /// `Z_p` shares of each bit of `x`, least significant first.
    pub bits: Vec<FpShare>,
    /// `δ = wrap3(x₁,x₂,x₃,L)`, as a replicated bit (0/1 per component).
    pub delta: [u64; 2],
    /// `Π_PC`'s blinding bit, over `Z_p` and as a replicated bit.
    pub beta_fp: FpShare,
    pub beta_bit: [u64; 2],
    /// A unit mask for `Π_PC`'s opening.
    pub m: FpShare,
}

/// `Π_WA` (Algorithm 2) over the transport, batched.
///
/// Returns each `θ = wrap3(a₁,a₂,a₃,L)` as a **replicated bit** `(left, right)`,
/// which is what the protocol produces and what `Π_DReLU` consumes.
///
/// Cost: one batched ring reveal plus one batched `Π_PC`. Everything else —
/// `wrap2` per component, the suffix sums, the final parity — is local, exactly
/// as the paper says.
///
/// See [`crate::crypto::falcon::wrap3_protocol`] for the correctness argument
/// and, importantly, the `a ≠ L−1` precondition that `Π_DReLU`'s doubling
/// discharges.
pub fn wrap3_batch<T: Transport>(
    a: &[PartyShare],
    pre: &[WrapPreproc],
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<[u64; 2]> {
    let n = a.len();
    assert_eq!(pre.len(), n, "one preprocessing bundle per element");
    if n == 0 {
        return Vec::new();
    }
    let mask: Fr = (1u128 << l_bits) - 1;
    let l_minus_1 = mask as u64;

    // 1. rⱼ = aⱼ + xⱼ and βⱼ = wrap2(aⱼ, xⱼ) — local, for both components this
    //    party holds.
    let r_shares: Vec<PartyShare> = a
        .iter()
        .zip(pre)
        .map(|(av, p)| PartyShare {
            left: av.left.wrapping_add(p.x.left) & mask,
            right: av.right.wrapping_add(p.x.right) & mask,
        })
        .collect();
    let beta_local: Vec<[u64; 2]> = a
        .iter()
        .zip(pre)
        .map(|(av, p)| {
            [
                falcon::wrap2(av.left as u64, p.x.left as u64, l_bits),
                falcon::wrap2(av.right as u64, p.x.right as u64, l_bits),
            ]
        })
        .collect();

    // 2–3. Open the rⱼ and compute α in the clear. The components, not just the
    //      sum: α is a function of the summands.
    // Packed at the ring width: this reveal is one of only two messages a
    // DReLU sends, so u128 padding would be ~19% of the whole protocol.
    let comps = reveal_components_packed_party(&r_shares, l_bits, base_op_id, me, mbx);

    // 4. η = (x ≥ r+1) via Π_PC, batched across the whole layer. At r = L−1 the
    //    comparison is against L, which no ℓ-bit x reaches, so η = 0 and no
    //    compare is needed — the same special case the local simulation makes.
    let mut pc_items = Vec::new();
    let mut pc_slots = Vec::new();
    let mut r_pub = Vec::with_capacity(n);
    for (i, c) in comps.iter().enumerate() {
        let r = (c[0].wrapping_add(c[1]).wrapping_add(c[2]) & mask) as u64;
        r_pub.push(r);
        if r != l_minus_1 {
            pc_items.push((pre[i].bits.clone(), r + 1));
            pc_slots.push(i);
        }
    }
    let betas: Vec<FpShare> = pc_slots.iter().map(|&i| pre[i].beta_fp).collect();
    let masks: Vec<FpShare> = pc_slots.iter().map(|&i| pre[i].m).collect();
    let bprimes = private_compare_batch(&pc_items, &betas, &masks, base_op_id + 1, me, mbx);

    // 5. θ = β₁+β₂+β₃ + α − η − δ (mod 2), all local on replicated bits.
    let mut out = Vec::with_capacity(n);
    let mut k = 0;
    for i in 0..n {
        let c = &comps[i];
        let alpha = falcon::wrap3(c[0] as u64, c[1] as u64, c[2] as u64, l_bits);
        // η as a replicated bit: β' is public, so XOR it into one component.
        let eta: [u64; 2] = if r_pub[i] == l_minus_1 {
            [0, 0]
        } else {
            let bp = u64::from(bprimes[k]);
            k += 1;
            [
                pre[i].beta_bit[0] ^ if me == 0 { bp } else { 0 },
                pre[i].beta_bit[1] ^ if me == 2 { bp } else { 0 },
            ]
        };
        // Public α likewise goes into component 0 only, so it is counted once.
        let a0 = if me == 0 { alpha } else { 0 };
        let a1 = if me == 2 { alpha } else { 0 };
        out.push([
            (beta_local[i][0] ^ a0 ^ eta[0] ^ pre[i].delta[0]) & 1,
            (beta_local[i][1] ^ a1 ^ eta[1] ^ pre[i].delta[1]) & 1,
        ]);
    }
    out
}

/// `Π_DReLU` over the transport: 1 when `a ≥ 0` under the signed reading.
///
/// One `Π_WA` on the **doubled** shares, then a local XOR of the summands' MSBs
/// (equation 11). The doubling is not an optimisation — it is what puts
/// `Π_PC`'s degenerate input `a = L−1` out of reach; see
/// [`crate::crypto::falcon::doubled`].
pub fn drelu_batch<T: Transport>(
    a: &[PartyShare],
    pre: &[WrapPreproc],
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<[u64; 2]> {
    let mask: Fr = (1u128 << l_bits) - 1;
    let doubled: Vec<PartyShare> = a
        .iter()
        .map(|s| PartyShare {
            left: s.left.wrapping_mul(2) & mask,
            right: s.right.wrapping_mul(2) & mask,
        })
        .collect();
    let theta = wrap3_batch(&doubled, pre, l_bits, base_op_id, me, mbx);
    a.iter()
        .zip(theta)
        .map(|(s, t)| {
            let m0 = falcon::msb(s.left as u64, l_bits);
            let m1 = falcon::msb(s.right as u64, l_bits);
            // The trailing ⊕1 of equation (11) is public, so one component only.
            [
                (t[0] ^ m0 ^ u64::from(me == 0)) & 1,
                (t[1] ^ m1 ^ u64::from(me == 2)) & 1,
            ]
        })
        .collect()
}

/// Preprocessing for one `Π_SS`: a random bit `c` shared in **both** rings.
///
/// The double sharing is the trick. `b` lives in `Z_2` but the select needs it
/// as a ring value to multiply by; opening `b ⊕ c` reveals nothing (c is
/// uniform) yet lets each party pick between `⟦c⟧_L` and `⟦1−c⟧_L` locally,
/// which *is* `⟦b⟧_L`. Lifting a bit into the ring would otherwise cost a
/// conversion protocol.
pub struct SelectPreproc {
    /// `c` as a replicated bit over `Z_2`.
    pub c_bit: [u64; 2],
    /// The same `c` as a replicated ring element.
    pub c_ring: PartyShare,
}

/// `Π_SS` (Select Shares), batched: `x` where `b = 0`, `y` where `b = 1`.
///
/// One bit opening plus one ring multiplication — two rounds, whatever the batch
/// size.
#[allow(clippy::too_many_arguments)]
pub fn select_batch<T: Transport>(
    x: &[PartyShare],
    y: &[PartyShare],
    b: &[[u64; 2]],
    pre: &[SelectPreproc],
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let n = x.len();
    assert!(
        y.len() == n && b.len() == n && pre.len() == n,
        "select_batch length mismatch"
    );
    if n == 0 {
        return Vec::new();
    }
    let mask: Fr = (1u128 << l_bits) - 1;

    // 1. e = b ⊕ c, opened. One bit per element on the wire.
    let masked: Vec<PartyShare> = b
        .iter()
        .zip(pre)
        .map(|(bi, p)| PartyShare {
            left: (bi[0] ^ p.c_bit[0]) as Fr,
            right: (bi[1] ^ p.c_bit[1]) as Fr,
        })
        .collect();
    let comps = reveal_components_packed_party(&masked, 1, base_op_id, me, mbx);
    let e: Vec<u64> = comps
        .iter()
        .map(|c| ((c[0] ^ c[1] ^ c[2]) & 1) as u64)
        .collect();

    // 2. d = ⟦b⟧_L, chosen locally now that e is public.
    let d: Vec<PartyShare> = e
        .iter()
        .zip(pre)
        .map(|(&ei, p)| {
            if ei == 1 {
                // 1 − c, with the public 1 placed in component 0 only.
                PartyShare {
                    left: (Fr::from(me == 0).wrapping_sub(p.c_ring.left)) & mask,
                    right: (Fr::from(me == 2).wrapping_sub(p.c_ring.right)) & mask,
                }
            } else {
                p.c_ring.clone()
            }
        })
        .collect();

    // 3. z = (y − x)·d + x, one batched ring multiply.
    let diffs: Vec<PartyShare> = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| PartyShare {
            left: yi.left.wrapping_sub(xi.left) & mask,
            right: yi.right.wrapping_sub(xi.right) & mask,
        })
        .collect();
    let pairs: Vec<(PartyShare, PartyShare)> = diffs.into_iter().zip(d).collect();
    let prod =
        crate::crypto::aby3_protocol::mul_batch_party_at(&pairs, l_bits, base_op_id + 1, me, mbx);
    prod.into_iter()
        .zip(x)
        .map(|(pz, xi)| PartyShare {
            left: pz.left.wrapping_add(xi.left) & mask,
            right: pz.right.wrapping_add(xi.right) & mask,
        })
        .collect()
}

/// `Π_ReLU` (Algorithm 3) over the transport: `max(0, a)` under the signed
/// reading.
///
/// `Π_DReLU` for the sign bit, then `Π_SS` selecting between `0` and `a`. This
/// completes Falcon's inference nonlinearity, and is the `4kn` row of their
/// table rather than the `3kn` one.
#[allow(clippy::too_many_arguments)]
pub fn relu_batch<T: Transport>(
    a: &[PartyShare],
    wrap_pre: &[WrapPreproc],
    sel_pre: &[SelectPreproc],
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let bits = drelu_batch(a, wrap_pre, l_bits, base_op_id, me, mbx);
    let zeros = vec![PartyShare { left: 0, right: 0 }; a.len()];
    // b = 1 means a ≥ 0, so the selected value is `a`; b = 0 selects zero.
    select_batch(&zeros, a, &bits, sel_pre, l_bits, base_op_id + 32, me, mbx)
}

/// `Π_Maxpool` (Algorithm 4) over the transport, **value only**.
///
/// Folds left over each window: one `Π_DReLU` on the running difference, then
/// one `Π_SS` to keep the larger. `n−1` of each per window, with every window's
/// step `k` batched together — so a whole pooling layer costs `n−1` DReLU
/// rounds, not `windows × (n−1)`.
///
/// The one-hot `e_k` Falcon also returns is omitted: it exists to route
/// gradients in training, and charging inference for it would overstate the
/// baseline. See [`crate::crypto::falcon::maxpool_value_only`].
///
/// # Input bound
///
/// `Π_DReLU` is applied to `max − aᵢ`, so `|aᵢ| < L/4` keeps that difference
/// inside the signed window — the same headroom Baru's own `nbits` calibration
/// carries for its comparisons.
#[allow(clippy::too_many_arguments)]
pub fn maxpool_batch<T: Transport>(
    windows: &[Vec<PartyShare>],
    wrap_pre: &[Vec<WrapPreproc>],
    sel_pre: &[Vec<SelectPreproc>],
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let w = windows.len();
    if w == 0 {
        return Vec::new();
    }
    let arity = windows[0].len();
    assert!(arity >= 1, "a pooling window needs at least one element");
    assert!(
        windows.iter().all(|win| win.len() == arity),
        "all windows must have the same arity so their folds stay in lockstep"
    );
    let mask: Fr = (1u128 << l_bits) - 1;

    let mut running: Vec<PartyShare> = windows.iter().map(|win| win[0].clone()).collect();
    let mut op = base_op_id;
    for k in 1..arity {
        // Step k of every window at once — this is what keeps a pooling layer's
        // round count independent of how many windows it has.
        let diffs: Vec<PartyShare> = running
            .iter()
            .zip(windows)
            .map(|(m, win)| PartyShare {
                left: m.left.wrapping_sub(win[k].left) & mask,
                right: m.right.wrapping_sub(win[k].right) & mask,
            })
            .collect();
        let wp: Vec<WrapPreproc> = (0..w).map(|i| clone_wrap(&wrap_pre[i][k - 1])).collect();
        let bits = drelu_batch(&diffs, &wp, l_bits, op, me, mbx);
        op += PC_ID_BLOCK;

        let candidates: Vec<PartyShare> = windows.iter().map(|win| win[k].clone()).collect();
        let sp: Vec<SelectPreproc> = (0..w)
            .map(|i| SelectPreproc {
                c_bit: sel_pre[i][k - 1].c_bit,
                c_ring: sel_pre[i][k - 1].c_ring.clone(),
            })
            .collect();
        // b = 1 means running ≥ candidate, so keep the running max.
        running = select_batch(&candidates, &running, &bits, &sp, l_bits, op, me, mbx);
        op += PC_ID_BLOCK;
    }
    running
}

fn clone_wrap(p: &WrapPreproc) -> WrapPreproc {
    WrapPreproc {
        x: p.x.clone(),
        bits: p.bits.clone(),
        delta: p.delta,
        beta_fp: p.beta_fp,
        beta_bit: p.beta_bit,
        m: p.m,
    }
}

/// Panics unless the truncation mask covers the whole ring.
///
/// A separate function because it is the one line standing between the protocol and a
/// silent 0.72-bit-per-call leak, and a caller can satisfy [`truncate_batch`]'s *other*
/// precondition (`a + r` must not wrap) by violating this one. Named so it can be tested
/// without a transport, and so the reason appears at the failure rather than in a doc.
pub fn assert_full_ring_mask(r_bits: u32, l_bits: u32) {
    assert_eq!(
        r_bits, l_bits,
        "the mask must cover the whole ring, or opening `a + r` leaks about `a` \
         (~0.72 bits per truncation) — see `truncate_batch`"
    );
}

/// Id space one [`truncate_batch`] call needs: two narrowed `a2b`s (the borrow and the
/// wrap) and two `Π_SS` lifts, each on disjoint ids.
pub const TRUNC_ID_BLOCK: u64 = 4 * (crate::crypto::aby3_protocol::A2B_ID_BLOCK + PC_ID_BLOCK) + 16;

/// Preprocessing for one exact truncation by `d` bits.
///
/// A truncation pair `(r, r>>d)` — ABY3's `Π_trunc2` — plus the bits of `r mod
/// 2^d` over `Z_p`, which is what makes the borrow computable.
pub struct TruncPreproc {
    /// `r`, uniform over the **whole** ring `Z_{2^l_bits}`, shared valid mod
    /// `2^(l_bits+1)`.
    ///
    /// Uniform over the whole ring is what makes the opened `c = a + r` uniform too, and
    /// therefore *perfectly* independent of `a`. A mask restricted to the no-wrap range
    /// would keep `a + r` inside the ring — which is why an earlier version chose one —
    /// but it also leaves `c` correlated with `a`, leaking ≈0.72 bits per truncation.
    /// [`truncate_batch`] pays for the wrap instead, with a bit it computes.
    ///
    /// The extra bit of width is for that computation: the wrap bit is the sign of
    /// `c − r`, which needs `l_bits + 1` bits to hold a difference of two `l_bits` values.
    pub r: PartyShare,
    pub r_shifted: PartyShare,
    /// Ring shares of `r mod 2^d` — the low bits the borrow compares against.
    ///
    /// A *ring* sharing, not `Z_p` bit-shares, because the borrow cannot use
    /// `Π_PC`: see [`truncate_batch`].
    pub r_lo: PartyShare,
    /// `Π_SS` selector lifting the **borrow** bit into the ring.
    pub sel: SelectPreproc,
    /// `Π_SS` selector lifting the **wrap** bit. A second one, because each `Π_SS` lift
    /// consumes its own daBit — sharing one between the two bits would correlate them.
    pub sel_wrap: SelectPreproc,
}

/// Exact truncation by `d` bits, batched — Falcon's "exact yet expensive"
/// variant, on the transport.
///
/// ```text
/// c = a + r  (opened, mod 2^l_bits)
/// a>>d = (c>>d) − (r>>d) − borrow + wrap·2^(l_bits−d)
///   borrow = 1 ⟺ c_lo < r_lo
///   wrap   = 1 ⟺ c   < r        (equivalently, a + r overflowed the ring)
/// ```
///
/// `c_lo` is public once `c` is opened but **`r_lo` is secret**, so the borrow
/// is a secure comparison — one per truncation, and every fixed-point multiply
/// needs a truncation. That is the entire reason Falcon calls this expensive,
/// and the reason it buys `ℓ = 32` instead of 64.
///
/// # Why the borrow does **not** use `Π_PC`
///
/// It looks like it should: `borrow ⟺ r_lo > c_lo ⟺ r_lo ≥ c_lo + 1`, which is
/// `Π_PC`'s exact shape — secret `r_lo`, public `c_lo + 1`. But `Π_PC` is
/// undefined when its secret equals its public argument
/// ([`crate::crypto::field::private_compare`]), and here that is `r_lo = c_lo+1`
/// — **precisely the boundary where the borrow flips**. The degenerate input is
/// not rare and not avoidable by masking: it is the answer.
///
/// Caught by `transport_truncation_is_exact` at the first value where it bites
/// (`a = 15`, `d = 4`, `r_lo = 10`, `c_lo + 1 = 10`).
///
/// So the borrow is taken as the sign of `c_lo − r_lo` via bit decomposition,
/// which is exact at equality. Worth noting for the reproduction: Falcon's own
/// stack needs a comparison primitive that `Π_PC` does not provide.
///
/// The resulting bit is lifted into the ring with `Π_SS`, which is how a `Z_2`
/// value becomes subtractable from a `Z_L` one without a conversion protocol.
///
/// # Why the wrap term exists: `r` is the mask, so it must cover the whole ring
///
/// Step 1 **opens** `c`, so `r` is the only thing hiding `a`. That forces `r` uniform
/// over the *whole* ring — then `c` is uniform and perfectly independent of `a`.
///
/// The first version of this protocol instead required `a + r` not to wrap and kept `r`
/// below `L − max|a|` to guarantee it. That is correct, and it leaks: with `r` no wider
/// than `a`, the posterior on `a` given `c` is uniform on a support of size
/// `min(c, A) − max(0, c − A)` — **1** when `c = 0` — for an average leak of
/// `1 / (2 ln 2) ≈ 0.72` bits per truncation. Every correctness test passed throughout,
/// because leakage is not a wrong answer.
///
/// Covering the ring means `a + r` *does* wrap, about half the time, so the wrap has to
/// be corrected rather than avoided. The correction is one bit, and it has the same shape
/// as the borrow already being computed — public `c`, secret `r` — so it costs a second
/// narrowed `a2b` and a second `Π_SS` lift, and **the ring stays narrow**. That last part
/// is the reason to prefer this over widening `r` by a statistical parameter `κ`: the
/// alternative needs an `nbits + κ + 1` bit ring, which is incompatible with the narrowed
/// wire encoding and measured ~1.9× *dearer* than `a2b` in bytes. This is cheaper and
/// hides perfectly rather than statistically.
///
/// The identity is verified exhaustively over all `(a, r)` at `l_bits = 10` by
/// `the_wrap_correction_is_exact_over_every_mask`.
#[allow(clippy::too_many_arguments)]
pub fn truncate_batch<T: Transport>(
    a: &[PartyShare],
    pre: &[TruncPreproc],
    d: u32,
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let n = a.len();
    assert_eq!(pre.len(), n, "one truncation pair per value");
    if n == 0 {
        return Vec::new();
    }
    let mask: Fr = (1u128 << l_bits) - 1;
    let dmask: u64 = (1u64 << d) - 1;

    // 1. Open c = a + r.
    let masked: Vec<PartyShare> = a
        .iter()
        .zip(pre)
        .map(|(av, p)| PartyShare {
            left: av.left.wrapping_add(p.r.left) & mask,
            right: av.right.wrapping_add(p.r.right) & mask,
        })
        .collect();
    let comps = reveal_components_packed_party(&masked, l_bits, base_op_id, me, mbx);
    let c: Vec<u64> = comps
        .iter()
        .map(|k| ((k[0].wrapping_add(k[1]).wrapping_add(k[2])) & mask) as u64)
        .collect();

    // 2. The two sign bits.
    //
    //    borrow = sign of (c_lo − r_lo)      1 ⟺ c_lo < r_lo
    //    wrap   = sign of (c    − r)         1 ⟺ a + r overflowed the ring
    //
    // Signs rather than `Π_PC` because `Π_PC` is undefined when its secret equals its public
    // argument, and for both bits that is exactly where the answer flips: `r_lo = c_lo + 1`
    // for the borrow, `a = 0` for the wrap. Neither is rare and neither can be masked away.
    //
    // Narrow to 4 columns: each is read as `& 1` and then fed to `Π_SS` as a `Z₂` value,
    // never multiplied by a wide operand, so the bit-shares only have to be valid mod 2^4.
    // That precondition is what makes narrowing safe here and not in the general `lt` path —
    // and it is worth 32×, because this `a2b` was 96% of an all-RSS network.
    //
    // # Two calls, at their own widths — a batched single call was tried and withdrawn
    //
    // The two bits are mutually independent (both are functions of the already-opened `c` and
    // the preprocessed `r`), so they *can* share one `a2b` over the concatenation, and doing
    // so cut the truncation's online rounds 22 → 12. But one call needs one width, so the
    // borrow's `d+1 = 9` columns widen to the wrap's `l_bits+1 = 21`, and the online bytes
    // rose 49 616 → 72 160 (+45%) for `n = 256`.
    //
    // That trade lost. At the pipeline it is roughly −10% messages for +5% bytes, worth ~2.7%
    // at RTT 20 / 30 Mbps and ~0.8% *worse* at RTT 0 / 10 Mbps — both inside the ±13%
    // run-to-run noise. Worse, it made Falcon's ReLU **dearer than `a2b` in bytes** (1.05×
    // cheaper → 0.93×), so it spent the term Falcon wins on to improve the term it loses on,
    // without changing which side of garbled circuits it lands. Trading away the reason a
    // route exists for an unmeasurable gain is not an optimisation.
    //
    // The `Π_SS` lifts below *are* batched, because both run at `l_bits` already — no
    // widening, so no byte cost. That is the part of the idea that survived, and it is worth
    // being precise about how little it buys: the truncation's online rounds go 22 → 20
    // (−9%), its bytes are unchanged, and the whole ReLU's messages move 576 → 574. Kept
    // because it is free and the batched form is no more code than two calls, **not** because
    // it is a speedup worth quoting.
    let w = d as usize + 1;
    let wmask_lo: Fr = (1u128 << w) - 1;
    let diffs: Vec<PartyShare> = c
        .iter()
        .zip(pre)
        .map(|(&ci, p)| {
            let lo = (ci & dmask) as Fr;
            let pub0 = if me == 0 { lo } else { 0 };
            let pub1 = if me == 2 { lo } else { 0 };
            PartyShare {
                left: pub0.wrapping_sub(p.r_lo.left) & wmask_lo,
                right: pub1.wrapping_sub(p.r_lo.right) & wmask_lo,
            }
        })
        .collect();
    let bits =
        crate::crypto::aby3_protocol::a2b_batch_party_at(&diffs, w, 4, base_op_id + 1, me, mbx);

    // `r`'s shares are valid mod `2^(l_bits+1)`, which is what lets this difference be formed
    // without the subtraction itself wrapping.
    let ww = l_bits as usize + 1;
    let wmask: Fr = (1u128 << ww) - 1;
    let wdiffs: Vec<PartyShare> = c
        .iter()
        .zip(pre)
        .map(|(&ci, p)| {
            let cv = ci as Fr;
            let pub0 = if me == 0 { cv } else { 0 };
            let pub1 = if me == 2 { cv } else { 0 };
            PartyShare {
                left: pub0.wrapping_sub(p.r.left) & wmask,
                right: pub1.wrapping_sub(p.r.right) & wmask,
            }
        })
        .collect();
    let wbits = crate::crypto::aby3_protocol::a2b_batch_party_at(
        &wdiffs,
        ww,
        4,
        base_op_id + 1 + crate::crypto::aby3_protocol::A2B_ID_BLOCK,
        me,
        mbx,
    );
    let msb_of = |bits: &[Vec<PartyShare>], w: usize| -> Vec<[u64; 2]> {
        bits.iter()
            .map(|bv| {
                let msb = &bv[w - 1];
                [(msb.left & 1) as u64, (msb.right & 1) as u64]
            })
            .collect()
    };
    let borrow_bits = msb_of(&bits, w);
    let wrap_bits = msb_of(&wbits, ww);

    // 3. Lift both bits into the ring with Π_SS: select between 0 and 1.
    let sel: Vec<SelectPreproc> = pre
        .iter()
        .map(|p| SelectPreproc {
            c_bit: p.sel.c_bit,
            c_ring: p.sel.c_ring.clone(),
        })
        .collect();
    // One `Π_SS` for both lifts, for the same reason the `a2b` above is one call: the borrow
    // and the wrap are independent, so the two selects are one batched select over the
    // concatenation. Each bit keeps its **own** selector daBit — `sel` for the borrow,
    // `sel_wrap` for the wrap — because sharing one would correlate the two lifted values.
    // Batching changes when they are sent, not what randomness they use.
    let mut both_bits: Vec<[u64; 2]> = Vec::with_capacity(2 * n);
    both_bits.extend_from_slice(&borrow_bits);
    both_bits.extend_from_slice(&wrap_bits);
    let mut both_sel: Vec<SelectPreproc> = Vec::with_capacity(2 * n);
    both_sel.extend(sel);
    both_sel.extend(pre.iter().map(|p| SelectPreproc {
        c_bit: p.sel_wrap.c_bit,
        c_ring: p.sel_wrap.c_ring.clone(),
    }));
    let lifted = select_batch(
        &vec![PartyShare { left: 0, right: 0 }; 2 * n],
        &vec![
            PartyShare {
                left: Fr::from(me == 0),
                right: Fr::from(me == 2),
            };
            2 * n
        ],
        &both_bits,
        &both_sel,
        l_bits,
        base_op_id + 2 + PC_ID_BLOCK + 2 * crate::crypto::aby3_protocol::A2B_ID_BLOCK,
        me,
        mbx,
    );
    let (borrow, wrap) = (&lifted[..n], &lifted[n..]);

    // 4. out = (c>>d) − (r>>d) − borrow + wrap·2^(l_bits−d), public term in component 0.
    //    The wrap term restores the `2^l_bits` the opening lost, divided by `2^d`
    //    because it is being added in the shifted domain.
    let wrap_weight: Fr = 1u128 << (l_bits - d);
    (0..n)
        .map(|i| {
            let hi = (c[i] >> d) as Fr;
            let pub0 = if me == 0 { hi } else { 0 };
            let pub1 = if me == 2 { hi } else { 0 };
            PartyShare {
                left: pub0
                    .wrapping_sub(pre[i].r_shifted.left)
                    .wrapping_sub(borrow[i].left)
                    .wrapping_add(wrap[i].left.wrapping_mul(wrap_weight))
                    & mask,
                right: pub1
                    .wrapping_sub(pre[i].r_shifted.right)
                    .wrapping_sub(borrow[i].right)
                    .wrapping_add(wrap[i].right.wrapping_mul(wrap_weight))
                    & mask,
            }
        })
        .collect()
}

/// A convolution with **secret** kernels, as one batched dot product per output.
///
/// `im2col` in shares: the patch gather is pure indexing and costs nothing, so a
/// conv layer is a matmul whose output count is `out_c · oh · ow` and whose
/// inner length is `in_c · K · K`. One `dot_batch_party` round, one ring element
/// per output — the same accounting Falcon's `MatMul = k·x·z` gives.
///
/// `input` is `[in_c][in_h][in_w]` row-major; `kernels[oc]` is
/// `[in_c · K · K]` in the same `(ic, ky, kx)` order the plaintext reference
/// uses, so the two cannot disagree about the flatten.
#[allow(clippy::too_many_arguments)]
pub fn conv_batch<T: Transport>(
    input: &[PartyShare],
    kernels: &[Vec<PartyShare>],
    in_c: usize,
    in_h: usize,
    in_w: usize,
    k: usize,
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let (oh, ow) = (in_h - k + 1, in_w - k + 1);
    let mut dots: Vec<Vec<(PartyShare, PartyShare)>> = Vec::with_capacity(kernels.len() * oh * ow);
    for kern in kernels {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut terms = Vec::with_capacity(in_c * k * k);
                for ic in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let w = &kern[(ic * k + ky) * k + kx];
                            let x = &input[(ic * in_h + oy + ky) * in_w + ox + kx];
                            terms.push((w.clone(), x.clone()));
                        }
                    }
                }
                dots.push(terms);
            }
        }
    }
    let refs: Vec<&[(PartyShare, PartyShare)]> = dots.iter().map(|d| d.as_slice()).collect();
    crate::crypto::aby3_protocol::dot_batch_party_at(&refs, l_bits, base_op_id, me, mbx)
}

/// Everything one dense→truncate→ReLU layer consumes from preprocessing.
pub struct LayerPreproc {
    pub trunc: Vec<TruncPreproc>,
    pub wrap: Vec<WrapPreproc>,
    pub sel: Vec<SelectPreproc>,
}

/// One fixed-point layer, the whole of Falcon's inference block:
/// `ReLU(truncate(W·x, d))`.
///
/// The truncation is where Falcon puts it — after the multiplication, not fused
/// into the nonlinearity — which is why a like-for-like comparison against
/// Baru's fused `relu∘rescale` has to be made at the *layer*, not the operation.
///
/// Rounds: 1 (matmul) + truncation + `5 + log₂ ℓ` (ReLU). All batched across the
/// layer's width, so the count is independent of how wide it is.
#[allow(clippy::too_many_arguments)]
pub fn layer_batch<T: Transport>(
    x: &[PartyShare],
    w: &[Vec<PartyShare>],
    pre: &LayerPreproc,
    d: u32,
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let z = dense_batch(x, w, l_bits, base_op_id, me, mbx);
    let t = truncate_batch(
        &z,
        &pre.trunc,
        d,
        l_bits,
        base_op_id + 4 * PC_ID_BLOCK,
        me,
        mbx,
    );
    relu_batch(
        &t,
        &pre.wrap,
        &pre.sel,
        l_bits,
        base_op_id + 8 * PC_ID_BLOCK,
        me,
        mbx,
    )
}

/// A dense layer with **secret** weights: `out[j] = Σ_k w[j][k] · x[k]`.
///
/// This is Falcon's `Π_MatMul` and Baru's `gadgets::dense` — the *same* RSS
/// protocol, which is why the linear layers were never the difference between
/// the two systems. Every product is secret×secret, so the whole layer is one
/// batched multiplication round followed by local sums: `out·in` elements on
/// the wire, one round, whatever the shape.
///
/// The bias is added locally by the caller if wanted; a public constant costs
/// nothing.
pub fn dense_batch<T: Transport>(
    x: &[PartyShare],
    w: &[Vec<PartyShare>],
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let inn = x.len();
    assert!(
        w.iter().all(|row| row.len() == inn),
        "weight rows must match the input width"
    );
    if w.is_empty() {
        return Vec::new();
    }
    let mask: Fr = (1u128 << l_bits) - 1;
    let mut pairs = Vec::with_capacity(w.len() * inn);
    for row in w {
        for (wk, xk) in row.iter().zip(x) {
            pairs.push((wk.clone(), xk.clone()));
        }
    }
    let prods =
        crate::crypto::aby3_protocol::mul_batch_party_at(&pairs, l_bits, base_op_id, me, mbx);
    w.iter()
        .enumerate()
        .map(|(j, _)| {
            let mut acc = PartyShare { left: 0, right: 0 };
            for k in 0..inn {
                let p = &prods[j * inn + k];
                acc = PartyShare {
                    left: acc.left.wrapping_add(p.left) & mask,
                    right: acc.right.wrapping_add(p.right) & mask,
                };
            }
            acc
        })
        .collect()
}

/// Rounds one [`private_compare`] costs at bit-width `ℓ`: one batched multiply
/// for the `u[i]`, `⌈log₂ ℓ⌉` for the product tree, one for the mask, one to
/// open.
///
/// Stated as a function so the benchmark can assert the measured round count
/// against it rather than quoting it — a documented cost that nothing checks is
/// how a claim goes stale.
pub fn private_compare_rounds(l: usize) -> u32 {
    let tree = (usize::BITS - (l - 1).leading_zeros()).max(1);
    1 + tree + 2
}

/// Field elements sent **per party** by one [`private_compare`]: `ℓ` for the
/// batched `u[i]`, `ℓ−1` across the whole product tree, 1 for the mask, 1 for
/// the opening.
pub fn private_compare_elements(l: usize) -> usize {
    l + (l - 1) + 1 + 1
}

// ─── The offline phase ───────────────────────────────────────────────────────
//
// Everything `Π_WA` and `Π_SS` consume is query-independent, which is why Falcon
// accounts for it offline. It must still be *generated*, and generating it is
// where a stand-in is easy and wrong: a value derived from a seed every party can
// compute is not a secret, and preprocessing whose secrecy fails takes the online
// protocol's secrecy with it.
//
// # The one free primitive, and the one that costs a round
//
// A replicated sharing of a uniformly random value is **free**. Party `i` holds
// the pair `(vᵢ, vᵢ₊₁)`, and component `vᵢ` is derived from the PRF key shared by
// exactly the two parties that hold it. Nobody predicts a component they do not
// hold, and nobody holds all three, so the value is secret with no communication
// (`PrfKeys::random_share_mod`). At modulus 2 that is a uniformly random *bit*
// shared over `Z₂`, since addition mod 2 is XOR.
//
// What is *not* free is the **same** bit shared in a second structure — a daBit.
// `Π_PC` needs its blinding bit over `Z_p`, `Π_SS` needs its selector over the
// ring, and both need it to be the bit they already hold over `Z₂`.
//
// The replicated layout makes this cheap, for a reason worth stating precisely:
// **each additive component `bᵢ` is already known to exactly the two parties that
// must hold it** in a replicated sharing of `bᵢ` — party `i` holds it as `left`,
// party `i−1` as `right`. So each `bᵢ` crosses into `Z_p` (or the ring) with *no
// communication at all*, via `share_component`. Only the XOR needs a protocol:
//
// ```text
// b = b₀ ⊕ b₁ ⊕ b₂
//   = c ⊕ b₂           where c = b₀ ⊕ b₁ = b₀ + b₁ − 2·b₀·b₁
//   = c + b₂ − 2·c·b₂
// ```
//
// Two multiplications, both batched across the whole request, and no input
// sharing. **No party learns `b`**: each holds two of the three components and
// never the third, and the multiplication protocol reveals nothing.
//
// The tempting shortcut — "party 0 holds `(b₀,b₁)` so let it input `c` directly"
// — does not work, and failing at it is instructive. A *replicated* sharing of
// `c` needs `c` at two parties; party 2 holds `(b₂,b₀)` and cannot compute it.
// Sharing a one-party value takes a real input protocol and a message; sharing
// each `bᵢ` takes neither.

/// A batch of random bits, shared over `Z₂` **and** over `Z_p`, consistently.
///
/// The `Z₂` sharing is free; the `Z_p` sharing costs one batched multiplication
/// for the whole batch. See the module note above for why the party roles are
/// fixed rather than symmetric.
pub fn random_dabits_fp<T: Transport>(
    n: usize,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> (Vec<[u64; 2]>, Vec<FpShare>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let z2 = random_bits_z2(n, base_op_id, me, mbx);
    let fp = fp_image_of(
        &z2,
        base_op_id.wrapping_add(n as u64),
        base_op_id.wrapping_add(n as u64 + 1),
        me,
        mbx,
    );
    (z2, fp)
}

/// Operation ids one [`random_dabits_fp`] or [`random_dabits_ring`] call
/// consumes, starting at its `base_op_id`.
///
/// `n` of them go to the `Z₂` draw — one PRF counter per bit, since
/// [`crate::crypto::net::PrfKeys::random_share_mod`] takes a scalar counter —
/// and one to each of the two XOR multiplications, which need only a single id
/// apiece however wide the batch (`mul_batch` tags the whole batch with one id
/// and derives per-element randomness inside its own `2^40` slot).
///
/// Callers **must** reserve this much. The earlier code hard-coded the offline
/// phase at `base + 1_000`, `base + 2_000`, `base + 3_000`, which silently
/// overlapped once a batch passed ~333 elements — and a CNN ReLU layer is far
/// larger than that. Overlap means two protocols deriving masks from the same
/// PRF counter, so it is a privacy bug, not just a mixed-up tag.
pub fn dabit_id_span(n: usize) -> u64 {
    n as u64 + 2
}

/// The free half of every daBit: one replicated `Z₂` bit per element, straight
/// from the pairwise PRFs. No communication — each party derives both of its
/// components locally.
fn random_bits_z2<T: Transport>(
    n: usize,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<[u64; 2]> {
    let keys = mbx.prf_keys(me);
    (0..n)
        .map(|i| {
            let (l, r) = keys.random_share_mod(base_op_id.wrapping_add(i as u64), 2);
            [l & 1, r & 1]
        })
        .collect()
}

/// Carry `Z₂` bits into `Z_p` by XORing their three components there.
///
/// Each component crosses for free: component `j` is this party's `left` when
/// `me == j` and its `right` when `me == j−1`, which is exactly who
/// `share_component` expects to supply it. Only the two XORs cost a round each.
fn fp_image_of<T: Transport>(
    z2: &[[u64; 2]],
    mul_id_a: u64,
    mul_id_b: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<FpShare> {
    let comp = |b: &[u64; 2], j: usize| -> FpShare {
        let v = if me == j {
            b[0]
        } else if me == (j + 2) % 3 {
            b[1]
        } else {
            0
        };
        share_component(Fp::new(v), j, me)
    };
    let b0: Vec<FpShare> = z2.iter().map(|b| comp(b, 0)).collect();
    let b1: Vec<FpShare> = z2.iter().map(|b| comp(b, 1)).collect();
    let b2: Vec<FpShare> = z2.iter().map(|b| comp(b, 2)).collect();

    let two = Fp::new(2);
    // c = b₀ ⊕ b₁
    let p01: Vec<(FpShare, FpShare)> = b0.iter().copied().zip(b1.iter().copied()).collect();
    let prod01 = mul_batch(&p01, mul_id_a, me, mbx);
    let cs: Vec<FpShare> = b0
        .iter()
        .zip(&b1)
        .zip(&prod01)
        .map(|((x, y), xy)| *x + *y - xy.scalar_mul(two))
        .collect();
    // b = c ⊕ b₂
    let p2: Vec<(FpShare, FpShare)> = cs.iter().copied().zip(b2.iter().copied()).collect();
    let prod2 = mul_batch(&p2, mul_id_b, me, mbx);
    cs.iter()
        .zip(&b2)
        .zip(&prod2)
        .map(|((c, y), cy)| *c + *y - cy.scalar_mul(two))
        .collect()
}

/// [`fp_image_of`] for the ring: same free component crossing, same two XORs.
fn ring_image_of<T: Transport>(
    z2: &[[u64; 2]],
    nbits: usize,
    mul_id_a: u64,
    mul_id_b: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    use crate::crypto::aby3_protocol::{mul_batch_party_at, share_component as ring_component};
    let mask: Fr = if nbits >= 128 {
        Fr::MAX
    } else {
        (1u128 << nbits) - 1
    };
    let comp = |b: &[u64; 2], j: usize| -> PartyShare {
        let v = if me == j {
            b[0]
        } else if me == (j + 2) % 3 {
            b[1]
        } else {
            0
        };
        ring_component(Fr::from(v), j, me)
    };
    let xor = |x: &PartyShare, y: &PartyShare, xy: &PartyShare| PartyShare {
        left: x
            .left
            .wrapping_add(y.left)
            .wrapping_sub(xy.left.wrapping_mul(2))
            & mask,
        right: x
            .right
            .wrapping_add(y.right)
            .wrapping_sub(xy.right.wrapping_mul(2))
            & mask,
    };
    let b0: Vec<PartyShare> = z2.iter().map(|b| comp(b, 0)).collect();
    let b1: Vec<PartyShare> = z2.iter().map(|b| comp(b, 1)).collect();
    let b2: Vec<PartyShare> = z2.iter().map(|b| comp(b, 2)).collect();

    let p01: Vec<(PartyShare, PartyShare)> = b0.iter().cloned().zip(b1.iter().cloned()).collect();
    let prod01 = mul_batch_party_at(&p01, nbits as u32, mul_id_a, me, mbx);
    let cs: Vec<PartyShare> = b0
        .iter()
        .zip(&b1)
        .zip(&prod01)
        .map(|((x, y), xy)| xor(x, y, xy))
        .collect();

    let p2: Vec<(PartyShare, PartyShare)> = cs.iter().cloned().zip(b2.iter().cloned()).collect();
    let prod2 = mul_batch_party_at(&p2, nbits as u32, mul_id_b, me, mbx);
    cs.iter()
        .zip(&b2)
        .zip(&prod2)
        .map(|((c, y), cy)| xor(c, y, cy))
        .collect()
}

/// A batch of random bits shared over **all three** structures at once — `Z₂`,
/// `Z_p` and the ring `Z_{2^nbits}` — every one of them the same bit.
///
/// The `Z₂` half is drawn once and carried into both other structures, which is
/// the whole point: calling [`random_dabits_fp`] and [`random_dabits_ring`]
/// separately gives two *unrelated* bits, and code that assumed otherwise would
/// be wrong in a way no single-structure test can see.
///
/// [`random_wrap_masks`] needs exactly this. Its `Z_p` image is the bit of the
/// mask that `Π_PC` compares against, its ring image is what composes the mask,
/// and the two must agree or the comparison is against a different number than
/// the one being masked.
pub fn random_dabits_triple<T: Transport>(
    n: usize,
    nbits: usize,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> (Vec<[u64; 2]>, Vec<FpShare>, Vec<PartyShare>) {
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let z2 = random_bits_z2(n, base_op_id, me, mbx);
    let m = base_op_id.wrapping_add(n as u64);
    let fp = fp_image_of(&z2, m, m.wrapping_add(1), me, mbx);
    let ring = ring_image_of(&z2, nbits, m.wrapping_add(2), m.wrapping_add(3), me, mbx);
    (z2, fp, ring)
}

/// Operation ids one [`random_dabits_triple`] call consumes.
pub fn dabit_triple_id_span(n: usize) -> u64 {
    n as u64 + 4
}

/// Wire width for the bit-shares inside `random_wrap_masks`' δ computation.
///
/// A bit-share is an *arithmetic* sharing of a 0/1 value, so its components are
/// full-width ring elements even though the value is one bit — and this crate
/// serialises them as `u128`, 16 bytes each, unless a narrower width is asked for.
/// The δ chain never multiplies a bit-share by a wide operand (its only products
/// are bit×bit), so the shares only have to be valid mod `2^DELTA_WIRE_BITS`.
///
/// 4 rather than 2 for headroom: the carry-save step evaluates
/// `x+y+z − 2(xy+xz+yz) + 4xyz`, whose intermediates reach 7 before reduction.
/// The identity holds over the integers and therefore modulo anything, but a
/// width that cannot represent the intermediates makes debugging needlessly
/// confusing for no saving worth having.
///
/// Matches `truncate_batch`, which narrows its borrow the same way and for the
/// same reason.
const DELTA_WIRE_BITS: u32 = 4;

/// `Π_WA`'s masks, generated by **bit composition** — the offline phase of
/// Falcon's Fig. 6, and the piece that used to be dealt from a public seed.
///
/// Returns one `(x, bits, δ)` per element: a ring sharing of a uniform mask
/// `x ∈ [0, 2^l_bits)`, the `Z_p` shares of its bits (least significant first),
/// and `δ = wrap3(x₀,x₁,x₂,L)` as a replicated `Z₂` bit. **No party learns `x`.**
///
/// # Why this is the hard part of `Π_WA`
///
/// `δ` is a function of all three additive components of `x`, so no party can
/// compute it from its own view — each is missing a summand. Computing it *from
/// shares* is what `Π_WA` itself does, which is the circularity Falcon has to
/// break. Sampling `x` and dealing `δ` alongside it, as this code used to,
/// escapes the circularity by giving up the secret: a seed every party can
/// evaluate is not a secret, so all three could reconstruct the mask.
///
/// # The construction
///
/// Build `x` **out of** shared bits rather than decomposing a sampled `x`:
///
/// 1. Draw `l_bits` daBits per element ([`random_dabits_triple`]). Their `Z_p`
///    images *are* `bits` — free, no conversion. Their ring images compose `x`.
/// 2. `x = Σ 2^k b_k`, a local weighted sum of the ring images. `x` is uniform
///    because the bits are uniform and independent.
/// 3. `δ` is **bit `l_bits` of the integer sum `x₀+x₁+x₂`**. That is the one
///    identity that makes this cheap: writing `s = x₀+x₁+x₂` we have
///    `s = wrap3_exact·L + x` with `wrap3_exact ∈ {0,1,2}`, so
///    `δ = wrap3_exact mod 2 = ⌊s/L⌋ mod 2`, which is exactly bit `l_bits` of
///    `s`. One `a2b` at width `l_bits + 1` therefore yields it — the wrap
///    genuinely falls out of the carry chain, and the low bits it also computes
///    are simply discarded because step 1 already produced them for free.
/// 4. Move `δ` from the ring into `Z₂`, where `Π_WA` consumes it, by opening
///    `δ ⊕ r` against a fresh ring/`Z₂` daBit. `r` is a uniform secret bit, so
///    the opened value is uniform and independent of `δ`.
///
/// # Cost, measured
///
/// One `a2b` per element — the very protocol Falcon's online phase exists to avoid.
/// Measured at ℓ = 20 by `falcons_cost_is_mostly_offline`, per comparison and per
/// party: **289 B offline against 34 B online**, so preprocessing is **89% of the
/// bytes** and 71% of the messages, and 8.9× of the wall time.
///
/// Two things follow, and an earlier version of this comment got the first wrong.
///
/// It claimed Falcon is "not cheaper than [`NonlinMode::A2b`] once offline and online
/// are added together". **That is false at the current encoding**: 289 + 34 = 323 B
/// per comparison against `a2b`'s measured 954 B (1.29 MB over 1352 comparisons,
/// `bench-gc-compare`), so Falcon is ~3× cheaper in *total* bytes, not merely online.
/// The claim was written when ring shares still went on the wire as `u128`, which
/// inflated this protocol far more than `a2b` because it is the more
/// preprocessing-heavy of the two.
///
/// Second, and this one holds: because the split is 89/11, **any comparison that
/// charges totals charges Falcon roughly nine times its online cost.** That is the
/// right way to read the crossover in `docs/falcon-reproduction.md` — it is a
/// total-cost boundary, and an online-only boundary would sit far above it.
/// Preprocessing is input-independent and can be generated before any query arrives,
/// which is what makes the distinction a real deployment property rather than an
/// accounting trick.
///
/// [`NonlinMode::A2b`]: crate::interpreter::transport_backend::NonlinMode::A2b
pub fn random_wrap_masks<T: Transport>(
    count: usize,
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<(PartyShare, Vec<FpShare>, [u64; 2])> {
    use crate::crypto::aby3_protocol::{
        a2b_batch_party_at, mul_batch_party_at, reveal_batch_party, A2B_ID_BLOCK,
    };
    if count == 0 {
        return Vec::new();
    }
    // `1u128 << l_bits` must not overflow, and the XOR encoding `a+b−2ab` needs
    // at least two bits of headroom.
    assert!(
        (2..128).contains(&l_bits),
        "random_wrap_masks: l_bits = {l_bits} outside 2..128"
    );
    let l = l_bits as usize;
    let mask: Fr = (1u128 << l_bits) - 1;
    let total = count * l;

    // ── 1. `l_bits` daBits per element ──
    // The Z_p images are the mask's bits; the ring images compose the mask.
    let (_, fp, ring) = random_dabits_triple(total, l, base_op_id, me, mbx);
    let mut next = base_op_id.wrapping_add(dabit_triple_id_span(total));

    // ── 2. x = Σ 2^k b_k, local ──
    // Reducing each component mod L is safe: it changes the component by a
    // multiple of L, so the reconstructed value is unchanged mod L, and it is
    // what puts the summands in the range `wrap3` is defined over.
    let xs: Vec<PartyShare> = (0..count)
        .map(|e| {
            let mut acc = PartyShare { left: 0, right: 0 };
            for k in 0..l {
                let b = &ring[e * l + k];
                acc.left = acc.left.wrapping_add(b.left << k) & mask;
                acc.right = acc.right.wrapping_add(b.right << k) & mask;
            }
            acc
        })
        .collect();

    // ── 3. δ = bit `l_bits` of x₀+x₁+x₂ ──
    // At width `l_bits + 1` the sum is exact in the bits that matter: three
    // summands below `2^l_bits` total below `3·2^l_bits < 2^(l_bits+2)`, and
    // reducing mod `2^(l_bits+1)` leaves bits `0..=l_bits` untouched.
    //
    // **Narrowed to `DELTA_WIRE_BITS` on the wire**, which is the difference
    // between this offline phase costing 31.6 MB per diagnosis and 1.3 MB. The
    // precondition is the one `a2b_batch_party_at` states: every consumer must
    // work at that width or below. Here the only consumer is the XOR below,
    // which multiplies δ by another *bit* and then opens the result — so the
    // bit-shares only ever have to be valid mod `2^DELTA_WIRE_BITS`. This is the
    // same argument, and the same 32x, that `truncate_batch` relies on for its
    // borrow; getting it wrong is not subtle, because the opened value stops
    // being a bit and the assertion below fires.
    let decomp = a2b_batch_party_at(&xs, l + 1, DELTA_WIRE_BITS, next, me, mbx);
    next = next.wrapping_add(A2B_ID_BLOCK);
    let delta_ring: Vec<PartyShare> = decomp.iter().map(|bits| bits[l].clone()).collect();

    // ── 4. δ from the ring into Z₂ ──
    // Same narrow width throughout, and the components are reduced to it: the
    // reveal sums them mod `2^128`, so an unreduced component would open to a
    // large number rather than to a bit.
    let dmask: Fr = (1u128 << DELTA_WIRE_BITS) - 1;
    let (r_z2, r_ring) =
        random_dabits_ring(count, DELTA_WIRE_BITS as usize, next, me, mbx);
    next = next.wrapping_add(dabit_id_span(count));
    let pairs: Vec<(PartyShare, PartyShare)> = delta_ring
        .iter()
        .cloned()
        .zip(r_ring.iter().cloned())
        .collect();
    let prod = mul_batch_party_at(&pairs, DELTA_WIRE_BITS, next, me, mbx);
    next = next.wrapping_add(1);
    let xored: Vec<PartyShare> = delta_ring
        .iter()
        .zip(&r_ring)
        .zip(&prod)
        .map(|((d, r), dr)| PartyShare {
            left: d.left.wrapping_add(r.left).wrapping_sub(dr.left.wrapping_mul(2)) & dmask,
            right: d
                .right
                .wrapping_add(r.right)
                .wrapping_sub(dr.right.wrapping_mul(2))
                & dmask,
        })
        .collect();
    let opened = reveal_batch_party(&xored, next, me, mbx);

    // Public XOR lands in component 0 only — party 0's `left` and party 2's
    // `right` — so it is counted exactly once in the reconstruction.
    (0..count)
        .map(|e| {
            let c = opened[e] & dmask;
            assert!(
                c <= 1,
                "random_wrap_masks: δ ⊕ r opened to {c}, which is not a bit — \
                 the mask sharing and its bit decomposition disagree"
            );
            let c = c as u64;
            let delta = [
                r_z2[e][0] ^ if me == 0 { c } else { 0 },
                r_z2[e][1] ^ if me == 2 { c } else { 0 },
            ];
            (xs[e].clone(), fp[e * l..(e + 1) * l].to_vec(), delta)
        })
        .collect()
}

/// Operation ids one [`random_nonzero_masks`] call consumes.
///
/// Loose on purpose. Each resampling round spends one id on its opening plus one
/// per slot it redraws, and how many slots are redrawn is data-dependent — a slot
/// is resampled when its draw came up zero. This reserves the worst case (every
/// slot zero in every one of the eight rounds), which cannot actually happen but
/// costs nothing to reserve: operation ids are positions in a `u64` counter, not
/// a resource that runs out.
pub fn nonzero_mask_id_span(n: usize) -> u64 {
    n as u64 + 8 * (n as u64 + 2)
}

/// Operation ids one [`random_wrap_masks`] call consumes.
pub fn wrap_mask_id_span(count: usize, l_bits: u32) -> u64 {
    use crate::crypto::aby3_protocol::A2B_ID_BLOCK;
    dabit_triple_id_span(count * l_bits as usize)
        + A2B_ID_BLOCK
        + dabit_id_span(count)
        + 2 // the δ-conversion multiplication and its opening
}

/// A batch of random bits, shared over `Z₂` **and** over the ring `Z_{2^k}`.
///
/// The ring counterpart of [`random_dabits_fp`] and the same construction —
/// except that `−2·c·b₂` is a wrapping ring operation, and the ring
/// multiplication protocol is the ABY3 one rather than the field one.
pub fn random_dabits_ring<T: Transport>(
    n: usize,
    nbits: usize,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> (Vec<[u64; 2]>, Vec<PartyShare>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let z2 = random_bits_z2(n, base_op_id, me, mbx);
    let m = base_op_id.wrapping_add(n as u64);
    let ring = ring_image_of(&z2, nbits, m, m.wrapping_add(1), me, mbx);
    (z2, ring)
}

/// Truncation pairs for [`truncate_batch`], generated by bit composition.
///
/// Returns `count` `TruncPreproc`, each a uniform `r < 2^r_bits` with `r >> d` and
/// `r mod 2^d` shared consistently, plus the `Π_SS` selector the borrow's lift needs.
/// **No party learns `r`.**
///
/// # Why this is much cheaper than [`random_wrap_masks`]
///
/// Both build their mask from shared random bits, but wrap3 then needs `δ` — a *carry*
/// out of a three-summand sum, which costs an `a2b`. Truncation needs no carry at all,
/// because **shifting is re-weighting**: from the same `r_bits` daBits,
///
/// ```text
/// r         = Σ_k      2^k     · b_k
/// r >> d    = Σ_{k≥d}  2^{k−d} · b_k
/// r mod 2^d = Σ_{k<d}  2^k     · b_k
/// ```
///
/// are three *local* weighted sums of one set of shares. The invariant
/// `truncate_batch` relies on — `r = (r>>d)·2^d + r_lo`, exactly, over the integers —
/// holds by construction rather than being computed, so it cannot be violated by a
/// wrong carry. The whole protocol is the daBits: two batched multiplications.
///
/// Only the *ring* images are needed, so this draws [`random_dabits_ring`] rather than
/// the triple — the `Z_p` images `Π_PC` wants are not part of a truncation pair.
///
/// # `r_bits` is a *security* parameter, not only a correctness one
///
/// [`truncate_batch`] requires `a + r` not to wrap, so `r_bits` must leave room for the
/// largest `|a|`. That much is a correctness bound, and it is tempting to read it as the
/// whole story — generating fewer bits satisfies it and costs proportionally less.
///
/// **It is not the whole story, because [`truncate_batch`] opens `c = a + r`.** `r` is
/// also the mask hiding `a`, so shrinking it to the correctness minimum maximises
/// leakage: see [`TRUNC_STAT_BITS`] for the analysis and the ≈0.72 bits per truncation
/// that a zero-slack choice costs. Callers should take both widths from
/// [`trunc_widths`] rather than picking `r_bits` to fit the bound.
pub fn random_trunc_pairs<T: Transport>(
    count: usize,
    d: u32,
    r_bits: u32,
    l_bits: u32,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<TruncPreproc> {
    use crate::crypto::aby3_protocol::scalar_mul_party;
    if count == 0 {
        return Vec::new();
    }
    assert_full_ring_mask(r_bits, l_bits);
    assert!(d < l_bits, "random_trunc_pairs: need d < l_bits, got d={d}, l_bits={l_bits}");
    let rb = r_bits as usize;
    // One bit wider than the ring: `truncate_batch` forms `c − r` to get the wrap bit,
    // and that difference needs `l_bits + 1` bits.
    let wide = l_bits + 1;
    let mask: Fr = if wide >= 128 { Fr::MAX } else { (1u128 << wide) - 1 };

    // `r_bits` bits per element, plus two selectors each: one for the borrow's `Π_SS`
    // lift and one for the wrap's.
    let (_, bits) = random_dabits_ring(count * rb, wide as usize, base_op_id, me, mbx);
    let sel_base = base_op_id.wrapping_add(dabit_id_span(count * rb));
    let (sel_z2, sel_ring) = random_dabits_ring(count, l_bits as usize, sel_base, me, mbx);
    let selw_base = sel_base.wrapping_add(dabit_id_span(count));
    let (selw_z2, selw_ring) = random_dabits_ring(count, l_bits as usize, selw_base, me, mbx);

    let weighted = |b: &PartyShare, w: u32| -> PartyShare {
        let s = scalar_mul_party(b, 1u128 << w);
        PartyShare {
            left: s.left & mask,
            right: s.right & mask,
        }
    };
    let add = |a: PartyShare, b: PartyShare| PartyShare {
        left: a.left.wrapping_add(b.left) & mask,
        right: a.right.wrapping_add(b.right) & mask,
    };

    (0..count)
        .map(|e| {
            let mut r = PartyShare { left: 0, right: 0 };
            let mut r_shifted = PartyShare { left: 0, right: 0 };
            let mut r_lo = PartyShare { left: 0, right: 0 };
            for k in 0..rb {
                let b = &bits[e * rb + k];
                r = add(r, weighted(b, k as u32));
                if (k as u32) < d {
                    r_lo = add(r_lo, weighted(b, k as u32));
                } else {
                    r_shifted = add(r_shifted, weighted(b, k as u32 - d));
                }
            }
            TruncPreproc {
                r,
                r_shifted,
                r_lo,
                sel: SelectPreproc {
                    c_bit: sel_z2[e],
                    c_ring: sel_ring[e].clone(),
                },
                sel_wrap: SelectPreproc {
                    c_bit: selw_z2[e],
                    c_ring: selw_ring[e].clone(),
                },
            }
        })
        .collect()
}

/// Operation ids one [`random_trunc_pairs`] call consumes.
///
/// Two selector daBits per element, not one: the borrow and the wrap each need their own
/// `Π_SS` lift.
pub fn trunc_pair_id_span(count: usize, r_bits: u32) -> u64 {
    dabit_id_span(count * r_bits as usize) + 2 * dabit_id_span(count)
}

/// Random **non-zero** `Z_p` masks for `Π_PC`'s final opening.
///
/// `Π_PC` reveals `m · Π c[i]` and reads only its zero-ness. `m` must therefore be a
/// uniformly random *unit*: with `m = 1` the product itself is revealed, which says more
/// than the predicate, and with `m = 0` the opening reads zero whatever the product was —
/// at `p = 37` that is a **1-in-37 wrong answer**, not a negligible event.
///
/// # The mask has to be tested for zero without being opened
///
/// An earlier version of this function opened each candidate in order to test it, and
/// returned the same shares. The justification on it read: *"opening whether a mask is zero
/// leaks nothing: `m` is drawn independently of any input, and this runs offline."* The first
/// clause is true and the conclusion does not follow, because the code opened `m` itself
/// rather than its zero-ness, and `m`'s job is to blind a **later** opening. A public blinder
/// blinds nothing: with `m` known, `Π c[i] = d/m` is public, and the `c[i]` carry no
/// independent randomness, so that product is a deterministic function of the compared bits.
/// `the_private_compare_product_leaks_more_than_the_predicate` measures the consequence — at
/// `ℓ = 6` a single predicate class contains **31 distinct products**, so the opening carried
/// about 5 bits where the predicate is entitled to 1. In `wrap3_batch` the compared secret is
/// the mask `x` and `r = a + x` is public, so those bits are bits of the activation.
///
/// The repair is the standard one: draw a second random `s` and open `d = m·s`. Then
/// `d ≠ 0` implies `m ≠ 0`, and conditioned on any `d = δ ≠ 0` the mask is still uniform over
/// `Z_p^*`, because for each candidate `μ ≠ 0` there is exactly one `s = δ/μ` that produces
/// `δ`. So the transcript fixes nothing about `m`, which
/// `the_nonzero_test_does_not_narrow_the_mask` verifies exhaustively over `Z_p × Z_p`.
///
/// The cost is one extra random share and one extra batched multiplication per attempt, both
/// offline. The per-attempt failure probability rises from `1/p` to `2/p − 1/p²` because
/// either factor can vanish, so the retry budget is raised to keep the overall failure below
/// `2^{-40}`.
pub fn random_nonzero_masks<T: Transport>(
    n: usize,
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<FpShare> {
    use crate::crypto::fp_protocol::random_share;
    if n == 0 {
        return Vec::new();
    }
    use crate::crypto::fp_protocol::mul_batch;
    const ATTEMPTS: usize = 12;

    let mut out: Vec<FpShare> = (0..n)
        .map(|i| random_share(base_op_id.wrapping_add(i as u64), me, mbx))
        .collect();
    let mut op = base_op_id.wrapping_add(n as u64);
    // Only the slots not yet accepted are probed, and an accepted mask is never probed
    // again. Re-probing an accepted slot with a fresh witness would require all `n` slots to
    // pass in the *same* round, whose probability is `(1 - 2/p)^n` and collapses for a large
    // batch; accumulating per slot makes the failure probability per slot independent across
    // attempts instead. Every party derives `pending` from opened values, so the three stay
    // in lockstep and the operation identifiers advance identically.
    let mut pending: Vec<usize> = (0..n).collect();
    for _ in 0..ATTEMPTS {
        if pending.is_empty() {
            return out;
        }
        let witness: Vec<FpShare> = (0..pending.len())
            .map(|k| random_share(op.wrapping_add(k as u64), me, mbx))
            .collect();
        op = op.wrapping_add(pending.len() as u64);
        let pairs: Vec<(FpShare, FpShare)> = pending
            .iter()
            .zip(&witness)
            .map(|(&i, sw)| (out[i], *sw))
            .collect();
        let probe = mul_batch(&pairs, op, me, mbx);
        op = op.wrapping_add(1);
        let opened = reveal_batch(&probe, op, me, mbx);
        op = op.wrapping_add(1);

        // A zero product means `m = 0` or `s = 0` and does not say which, so the mask is
        // redrawn as well; keeping it could loop forever on a zero mask.
        let mut still: Vec<usize> = Vec::new();
        for (k, (&i, v)) in pending.iter().zip(&opened).enumerate() {
            if v.is_zero() {
                out[i] = random_share(op.wrapping_add(k as u64), me, mbx);
                still.push(i);
            }
        }
        op = op.wrapping_add(pending.len() as u64 + 1);
        pending = still;
    }
    if pending.is_empty() {
        return out;
    }
    // A zero product has probability `2/p − 1/p²` ≈ 0.053 at p = 37, so ATTEMPTS
    // independent failures is ≈ 2^-42. Continuing with a possibly-zero mask would silently
    // invert a comparison, so this stops instead.
    panic!("random_nonzero_masks: {ATTEMPTS} consecutive zero probes — PRF or modulus is wrong");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;
    use std::thread;

    /// **`Π_PC`'s final mask must stay secret, and `random_nonzero_masks` opens it.**
    ///
    /// `private_compare` computes `c[i] = u[i] + 1 + Σ_{k>i} w[k]` and opens `m · Π c[i]`,
    /// reading only whether the result is zero. Its own comment states the security
    /// argument exactly: *"the product is otherwise uniform, so only 'was some term zero'
    /// escapes"*. The `c[i]` carry **no independent blinding** — they are deterministic in
    /// the secret bits, the public `r` and the secret `β` — so `m` is the only thing
    /// standing between the transcript and `Π c[i]`.
    ///
    /// `random_nonzero_masks` opens `m` itself in order to test it for zero, and returns the
    /// same shares. Once `m` is public, `Π c[i] = d/m` is public too. This test establishes
    /// the half of the break that is a statement about the protocol rather than about the
    /// code: that `Π c[i]` is **not** a function of the comparison's output, so revealing it
    /// leaks strictly more than the predicate.
    ///
    /// In `wrap3_batch` the secret compared is the preprocessed mask `x`, against the public
    /// `r = a + x`, so information about `x` is information about the activation `a`.
    #[test]
    fn the_private_compare_product_leaks_more_than_the_predicate() {
        use crate::crypto::field::{Fp, P};
        const L: usize = 6; // bits per compared value

        // The product Π c[i], computed in the clear exactly as `private_compare` does.
        let product = |x: u64, r: u64, beta: u64| -> Fp {
            let bit = |v: u64, i: usize| (v >> i) & 1;
            let sign = Fp::new(1) - Fp::new(2) * Fp::new(beta);
            let w: Vec<Fp> = (0..L)
                .map(|i| {
                    if bit(r, i) == 0 {
                        Fp::new(bit(x, i))
                    } else {
                        Fp::new(1) - Fp::new(bit(x, i))
                    }
                })
                .collect();
            let u: Vec<Fp> = (0..L)
                .map(|i| sign * (Fp::new(bit(x, i)) - Fp::new(bit(r, i))))
                .collect();
            let mut suffix = [Fp::ZERO; L + 1];
            for i in (0..L).rev() {
                suffix[i] = suffix[i + 1] + w[i];
            }
            (0..L).fold(Fp::new(1), |acc, i| acc * (u[i] + suffix[i + 1] + Fp::new(1)))
        };

        // Group the secrets by the predicate the protocol is entitled to reveal, and count
        // how many distinct product values each group contains. One would mean the product
        // is a function of the output and revealing it would be harmless.
        let mut worst = 0usize;
        let mut witness = None;
        for r in 0..(1u64 << L) {
            for beta in 0..2u64 {
                let mut by_pred: [std::collections::BTreeSet<u64>; 2] =
                    [Default::default(), Default::default()];
                let mut example: [Option<u64>; 2] = [None, None];
                for x in 0..(1u64 << L) {
                    let d = product(x, r, beta);
                    let pred = usize::from(!d.is_zero());
                    if by_pred[pred].insert(d.value()) && example[pred].is_none() {
                        example[pred] = Some(x);
                    }
                }
                for (pred, seen) in by_pred.iter().enumerate() {
                    if seen.len() > worst {
                        worst = seen.len();
                        witness = Some((r, beta, pred, seen.len()));
                    }
                }
            }
        }
        let (r, beta, pred, distinct) = witness.expect("some group is non-empty");
        println!(
            "  worst case: r={r}, beta={beta}, predicate={pred} -> {distinct} distinct \
             products among secrets sharing that predicate (p={P})"
        );
        println!(
            "  so the opening carries up to log2({distinct}) = {:.1} bits, against the \
             1 bit the predicate is entitled to",
            (distinct as f64).log2()
        );

        // The break: some predicate class contains more than one product value, so the
        // opened value distinguishes secrets that the output does not.
        assert!(
            worst > 1,
            "if every predicate class had a single product value, revealing the product \
             would leak nothing and `m` would not need to be secret"
        );
    }

    /// **The zero test on `Π_PC`'s mask must not narrow the mask.**
    ///
    /// This is the other half of the repair described on `random_nonzero_masks`. We now open
    /// `d = m·s` rather than `m`, and the claim is that the transcript leaves the mask
    /// uniform over `Z_p^*`. The test is exhaustive over `Z_p × Z_p` rather than statistical,
    /// because the property is a counting statement and `p = 37` makes counting cheap.
    ///
    /// Two things are checked. Accepting on `d ≠ 0` really does exclude `m = 0`, so the
    /// protocol still gets the unit it needs; and every accepting `d` admits exactly one
    /// witness `s` per candidate mask, so conditioned on the opened value the mask remains
    /// uniform and the opening is simulatable without it.
    #[test]
    fn the_nonzero_test_does_not_narrow_the_mask() {
        use crate::crypto::field::{Fp, P};

        let mut zero_products = 0usize;
        for d in 1..P {
            // For an accepting opening `d`, count the witnesses for each candidate mask.
            let mut witnesses = vec![0usize; P as usize];
            for m in 0..P {
                for sw in 0..P {
                    if (Fp::new(m) * Fp::new(sw)).value() == d {
                        witnesses[m as usize] += 1;
                    }
                }
            }
            assert_eq!(
                witnesses[0], 0,
                "d={d} is non-zero, so m=0 must be excluded: accepting it would let a zero \
                 mask through and invert a comparison once in p"
            );
            for m in 1..P {
                assert_eq!(
                    witnesses[m as usize], 1,
                    "d={d}: mask {m} should have exactly one witness s, so that no accepting \
                     opening prefers one mask over another"
                );
            }
        }

        // The retry rate the doc quotes: a probe vanishes when either factor does.
        for m in 0..P {
            for sw in 0..P {
                if (Fp::new(m) * Fp::new(sw)).is_zero() {
                    zero_products += 1;
                }
            }
        }
        let rate = zero_products as f64 / (P * P) as f64;
        println!(
            "  every non-zero opening leaves all {} candidate masks equally likely; \
             retry rate {:.4} (2/p - 1/p^2 = {:.4})",
            P - 1,
            rate,
            2.0 / P as f64 - 1.0 / (P * P) as f64
        );
        assert!((rate - (2.0 / P as f64 - 1.0 / (P * P) as f64)).abs() < 1e-12);
        // 12 attempts at that rate is below 2^-40, which is what the retry budget claims.
        assert!(rate.powi(12) < 2f64.powi(-40), "the retry budget must reach 2^-40");
    }

    /// **The daBit's defining property**: the bit shared over `Z₂` and the bit
    /// shared over `Z_p` must be the *same* bit. If they drift, `Π_PC`'s blinding
    /// stops cancelling and every comparison built on it silently inverts for
    /// half its inputs.
    #[test]
    #[ignore] // real 3-party transport
    fn dabits_agree_across_z2_and_zp() {
        const N: usize = 24;
        let out = run3(|me, mbx| random_dabits_fp(N, 500, me, mbx));
        for i in 0..N {
            // Z₂: XOR of the three additive components.
            let z2 = (out[0].0[i][0] ^ out[1].0[i][0] ^ out[2].0[i][0]) & 1;
            // Z_p: sum of the three additive components.
            let zp = out[0].1[i].left + out[1].1[i].left + out[2].1[i].left;
            assert!(
                zp.value() == 0 || zp.value() == 1,
                "bit {i} reconstructed to {} over Z_p — not a bit",
                zp.value()
            );
            assert_eq!(z2, zp.value(), "bit {i} disagrees between Z₂ and Z_p");
        }
        // Both values must actually vary; a constant would satisfy the above.
        let bits: Vec<u64> = (0..N)
            .map(|i| (out[0].0[i][0] ^ out[1].0[i][0] ^ out[2].0[i][0]) & 1)
            .collect();
        assert!(
            bits.contains(&0) && bits.contains(&1),
            "24 draws were all the same bit — this is not randomness"
        );
    }

    /// The replicated sharing must be *consistent*: each party's `right`
    /// component is its successor's `left`. A daBit that broke this would be
    /// malformed in a way the online protocol cannot detect.
    #[test]
    #[ignore] // real 3-party transport
    fn dabit_sharings_are_well_formed() {
        const N: usize = 8;
        let out = run3(|me, mbx| random_dabits_fp(N, 600, me, mbx));
        for i in 0..N {
            for p in 0..3 {
                assert_eq!(
                    out[p].0[i][1],
                    out[(p + 1) % 3].0[i][0],
                    "Z₂ bit {i}: party {p}'s right is not party {}'s left",
                    (p + 1) % 3
                );
                assert_eq!(
                    out[p].1[i].right,
                    out[(p + 1) % 3].1[i].left,
                    "Z_p bit {i}: replication broken between {p} and {}",
                    (p + 1) % 3
                );
            }
        }
    }

    /// **No party may learn the bit.** The construction gives party 0 the value
    /// `c = b₀ ⊕ b₁` and party 1 the value `b₂`, and its security rests on
    /// neither holding both. This checks the consequence directly: over many
    /// draws, no single party's view predicts the bit better than chance.
    ///
    /// A weak test statistically, but it is the property that the *previous*
    /// implementation violated outright — it derived every component from a seed
    /// all three parties could compute — so a regression would be gross, not
    /// subtle.
    #[test]
    #[ignore] // real 3-party transport
    fn no_party_view_determines_the_bit() {
        const N: usize = 64;
        let out = run3(|me, mbx| random_dabits_fp(N, 700, me, mbx));
        for p in 0..3 {
            // The party's whole view of the Z₂ sharing is its own two components.
            let mut agree_with_left = 0;
            for i in 0..N {
                let b = (out[0].0[i][0] ^ out[1].0[i][0] ^ out[2].0[i][0]) & 1;
                if out[p].0[i][0] == b {
                    agree_with_left += 1;
                }
            }
            assert!(
                (8..=56).contains(&agree_with_left),
                "party {p}'s own component matched the secret bit {agree_with_left}/64 times — \
                 it is not independent of the secret"
            );
        }
    }

    /// The ring daBit, same properties. `Π_SS` consumes this one.
    #[test]
    #[ignore] // real 3-party transport
    fn ring_dabits_agree_across_z2_and_the_ring() {
        const N: usize = 24;
        const NBITS: usize = 20;
        let out = run3(|me, mbx| random_dabits_ring(N, NBITS, 800, me, mbx));
        let mask: Fr = (1u128 << NBITS) - 1;
        for i in 0..N {
            let z2 = (out[0].0[i][0] ^ out[1].0[i][0] ^ out[2].0[i][0]) & 1;
            let ring = out[0].1[i]
                .left
                .wrapping_add(out[1].1[i].left)
                .wrapping_add(out[2].1[i].left)
                & mask;
            assert!(ring <= 1, "bit {i} reconstructed to {ring} in the ring");
            assert_eq!(z2 as u128, ring, "bit {i} disagrees between Z₂ and the ring");
        }
    }

    /// **The property `Π_WA` cannot check for itself.** The mask `x`, its bits
    /// over `Z_p`, and its wrap `δ` are three separate objects, and the protocol
    /// silently returns the wrong sign if any pair of them disagrees. Nothing
    /// downstream detects it — a DReLU that is wrong for half its inputs still
    /// looks like a bit.
    ///
    /// So this reconstructs all three from the parties' shares and checks them
    /// against each other: the bits must be the binary expansion of the very `x`
    /// the ring shares reconstruct to, and `δ` must be the `wrap3` of that `x`'s
    /// own three summands.
    #[test]
    #[ignore] // real 3-party transport
    fn composed_wrap_masks_are_mutually_consistent() {
        const COUNT: usize = 16;
        const LB: u32 = 8;
        let out = run3(|me, mbx| random_wrap_masks(COUNT, LB, 1_000, me, mbx));
        let mask: Fr = (1u128 << LB) - 1;

        // Indexed on purpose: every check reads the *same* position across all
        // three parties' vectors, which an iterator over any one cannot express.
        #[allow(clippy::needless_range_loop)]
        for e in 0..COUNT {
            // Replication: each party's `right` is its successor's `left`.
            for p in 0..3 {
                assert_eq!(
                    out[p][e].0.right,
                    out[(p + 1) % 3][e].0.left,
                    "mask {e}: ring replication broken between {p} and {}",
                    (p + 1) % 3
                );
            }
            // The three additive summands, and the mask they reconstruct to.
            let c = [out[0][e].0.left, out[1][e].0.left, out[2][e].0.left];
            for (j, cj) in c.iter().enumerate() {
                assert!(*cj <= mask, "mask {e}: summand {j} is not reduced mod L");
            }
            let x = (c[0].wrapping_add(c[1]).wrapping_add(c[2])) & mask;

            // The Z_p bits must be the binary expansion of that same x.
            for k in 0..LB as usize {
                let b = out[0][e].1[k].left + out[1][e].1[k].left + out[2][e].1[k].left;
                assert_eq!(
                    u128::from(b.value()),
                    (x >> k) & 1,
                    "mask {e}: Z_p bit {k} disagrees with the ring value {x}"
                );
            }

            // And δ must be that x's own wrap3 — the piece no party can compute.
            let delta = (out[0][e].2[0] ^ out[1][e].2[0] ^ out[2][e].2[0]) & 1;
            let want = crate::crypto::falcon::wrap3(c[0] as u64, c[1] as u64, c[2] as u64, LB);
            assert_eq!(
                delta, want,
                "mask {e}: δ = {delta} but wrap3({}, {}, {}) = {want}",
                c[0], c[1], c[2]
            );
        }

        // The masks must actually be masks. A construction that produced a
        // constant, or only even values, would satisfy every check above.
        let values: Vec<u128> = (0..COUNT)
            .map(|e| {
                (out[0][e].0.left.wrapping_add(out[1][e].0.left).wrapping_add(out[2][e].0.left))
                    & mask
            })
            .collect();
        assert!(
            values.iter().collect::<std::collections::HashSet<_>>().len() > COUNT / 2,
            "16 masks took {} distinct values — this is not randomness",
            values.iter().collect::<std::collections::HashSet<_>>().len()
        );
        assert!(
            values.iter().any(|v| v & 1 == 1),
            "every mask was even — the low daBit is not being composed in"
        );
    }

    /// **No party may learn the mask** — the property whose absence made this
    /// protocol undeployable before bit composition landed.
    ///
    /// The regression being guarded is gross rather than subtle: the previous
    /// implementation derived all three summands from `op_id` by a public
    /// formula, so *every* party could reconstruct `x` exactly. This checks the
    /// consequence directly — a party's own two summands must not determine the
    /// mask — and would have failed outright against that version.
    #[test]
    #[ignore] // real 3-party transport
    fn no_party_view_determines_the_composed_mask() {
        const COUNT: usize = 32;
        const LB: u32 = 8;
        let out = run3(|me, mbx| random_wrap_masks(COUNT, LB, 2_000, me, mbx));
        let mask: Fr = (1u128 << LB) - 1;
        for p in 0..3 {
            let mut matched = 0;
            // Indexed for the same reason: the mask is reconstructed from all
            // three parties' entries at position `e`.
            #[allow(clippy::needless_range_loop)]
            for e in 0..COUNT {
                let x = (out[0][e].0.left.wrapping_add(out[1][e].0.left).wrapping_add(out[2][e].0.left))
                    & mask;
                // The party's whole view is its two summands; the best it can do
                // is guess from their sum.
                let guess = out[p][e].0.left.wrapping_add(out[p][e].0.right) & mask;
                if guess == x {
                    matched += 1;
                }
            }
            assert!(
                matched <= COUNT / 4,
                "party {p}'s own summands predicted the mask {matched}/{COUNT} times — \
                 it is not independent of the secret"
            );
        }
    }

    /// End to end: `Π_DReLU` on masks that came from the real offline phase.
    ///
    /// [`composed_wrap_masks_are_mutually_consistent`] checks the masks against
    /// each other; this checks them against the protocol that consumes them, over
    /// every value in the ring including both boundaries. The two together are
    /// what let the offline phase be believed — internal consistency alone would
    /// pass for a mask whose `δ` convention was off by one from `Π_WA`'s.
    #[test]
    #[ignore] // real 3-party transport
    fn drelu_over_composed_masks_matches_the_signed_value() {
        const LB: u32 = 8;
        let l = 1u64 << LB;
        let vals: Vec<u64> = (0..l).collect();
        let splits: Vec<[u64; 3]> = vals
            .iter()
            .map(|&v| {
                let s0 = v.wrapping_mul(37) % l;
                let s1 = v.wrapping_mul(91).wrapping_add(11) % l;
                [s0, s1, (v + 2 * l - s0 - s1) % l]
            })
            .collect();
        let expected: Vec<u64> = vals
            .iter()
            .map(|&v| u64::from(crate::crypto::falcon::signed(v, LB) >= 0))
            .collect();
        let n = vals.len();

        let out = run3(move |me, mbx| {
            let a: Vec<PartyShare> = splits
                .iter()
                .map(|s| PartyShare {
                    left: s[me % 3] as u128,
                    right: s[(me + 1) % 3] as u128,
                })
                .collect();
            // The whole offline phase, batched across all 256 comparisons.
            let off = 100_000;
            let wm = random_wrap_masks(n, LB, off, me, mbx);
            let off = off + wrap_mask_id_span(n, LB);
            let (beta_z2, beta_fp) = random_dabits_fp(n, off, me, mbx);
            let off = off + dabit_id_span(n);
            let m = random_nonzero_masks(n, off, me, mbx);
            let pre: Vec<WrapPreproc> = wm
                .into_iter()
                .enumerate()
                .map(|(i, (x, bits, delta))| WrapPreproc {
                    x,
                    bits,
                    delta,
                    beta_fp: beta_fp[i],
                    beta_bit: beta_z2[i],
                    m: m[i],
                })
                .collect();
            drelu_batch(&a, &pre, LB, PC_ID_BLOCK, me, mbx)
        });

        for (i, want) in expected.iter().enumerate() {
            let got = (out[0][i][0] ^ out[1][i][0] ^ out[2][i][0]) & 1;
            assert_eq!(
                got,
                *want,
                "DReLU at value {i} (signed {}) over a composed mask",
                crate::crypto::falcon::signed(i as u64, LB)
            );
        }
    }

    /// **The invariant `truncate_batch` rests on, checked from the shares.**
    ///
    /// A truncation pair is three objects that must agree exactly:
    /// `r = (r>>d)·2^d + (r mod 2^d)`, over the integers, with `r_lo < 2^d`. Bit
    /// composition makes that hold by construction — the three parts are re-weightings
    /// of one set of bit-shares — but "by construction" is a claim about code, so this
    /// reconstructs all three from the parties' shares and checks them against each
    /// other. A pair that violated it would produce silently wrong truncations, the
    /// same failure mode as an inconsistent `Π_WA` mask.
    #[test]
    #[ignore] // real 3-party transport
    fn composed_truncation_pairs_are_mutually_consistent() {
        const COUNT: usize = 16;
        const D: u32 = 4;
        const LB: u32 = 20;
        // The mask spans the whole ring — anything narrower leaks when `a + r` is
        // opened, and `random_trunc_pairs` refuses it.
        const RB: u32 = LB;
        let out = run3(|me, mbx| random_trunc_pairs(COUNT, D, RB, LB, 3_000, me, mbx));
        let mask: Fr = (1u128 << LB) - 1;
        let rec = |p: &[Vec<TruncPreproc>], e: usize, f: fn(&TruncPreproc) -> Fr| -> Fr {
            (f(&p[0][e]).wrapping_add(f(&p[1][e])).wrapping_add(f(&p[2][e]))) & mask
        };

        let mut seen = std::collections::HashSet::new();
        for e in 0..COUNT {
            let r = rec(&out, e, |t| t.r.left);
            let hi = rec(&out, e, |t| t.r_shifted.left);
            let lo = rec(&out, e, |t| t.r_lo.left);

            assert!(r < (1u128 << RB), "pair {e}: r = {r} exceeds 2^{RB}");
            assert!(lo < (1u128 << D), "pair {e}: r_lo = {lo} is not below 2^{D}");
            assert_eq!(
                r,
                (hi << D) + lo,
                "pair {e}: r = {r} but (r>>{D})·2^{D} + r_lo = {}",
                (hi << D) + lo
            );
            assert_eq!(hi, r >> D, "pair {e}: r_shifted {hi} != r>>{D} = {}", r >> D);

            // Replication must be well formed, or the pair is malformed in a way the
            // online protocol cannot detect.
            for p in 0..3 {
                assert_eq!(
                    out[p][e].r.right,
                    out[(p + 1) % 3][e].r.left,
                    "pair {e}: r replication broken between {p} and {}",
                    (p + 1) % 3
                );
            }
            seen.insert(r);
        }
        assert!(
            seen.len() > COUNT / 2,
            "{COUNT} pairs took only {} distinct values — not randomness",
            seen.len()
        );
    }

    /// **No party may learn `r`** — the property that makes the mask a mask.
    #[test]
    #[ignore] // real 3-party transport
    fn no_party_view_determines_a_truncation_mask() {
        const COUNT: usize = 32;
        const LB: u32 = 20;
        let out = run3(|me, mbx| random_trunc_pairs(COUNT, 4, LB, LB, 4_000, me, mbx));
        let mask: Fr = (1u128 << LB) - 1;
        for p in 0..3 {
            let mut matched = 0;
            #[allow(clippy::needless_range_loop)]
            for e in 0..COUNT {
                let r = (out[0][e].r.left)
                    .wrapping_add(out[1][e].r.left)
                    .wrapping_add(out[2][e].r.left)
                    & mask;
                let guess = out[p][e].r.left.wrapping_add(out[p][e].r.right) & mask;
                if guess == r {
                    matched += 1;
                }
            }
            assert!(
                matched <= COUNT / 4,
                "party {p}'s own summands predicted the mask {matched}/{COUNT} times"
            );
        }
    }

    /// A `Π_PC` mask must never be zero: the opening reads only zero-ness, so a
    /// zero mask reports "product is zero" whatever the product was. At `p = 37`
    /// an unchecked draw is wrong 1 time in 37, which is not a tail event.
    #[test]
    #[ignore] // real 3-party transport
    fn pc_masks_are_never_zero() {
        const N: usize = 200;
        let out = run3(|me, mbx| random_nonzero_masks(N, 900, me, mbx));
        // Indexed on purpose: each step reads the *same* position across all
        // three parties' vectors, which an iterator over any one cannot express.
        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            let m = out[0][i].left + out[1][i].left + out[2][i].left;
            assert!(!m.is_zero(), "mask {i} is zero — a comparison using it inverts");
        }
        // And they must be *masks*, i.e. varied — a constant 1 would pass the
        // above while revealing the raw product, which is the leak this fixes.
        let vals: std::collections::HashSet<u64> = (0..N)
            .map(|i| (out[0][i].left + out[1][i].left + out[2][i].left).value())
            .collect();
        assert!(
            vals.len() > 10,
            "only {} distinct masks in {N} draws — not uniform over Z_p",
            vals.len()
        );
    }

    fn run3<F, R>(f: F) -> Vec<R>
    where
        F: Fn(PartyId, &Mailbox<InprocHandle>) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let handles = InprocHandle::create(3);
        let f = std::sync::Arc::new(f);
        let joins: Vec<_> = handles
            .into_iter()
            .map(|h| {
                let f = f.clone();
                thread::spawn(move || {
                    let mbx = Mailbox::new(h);
                    let me = mbx.id();
                    f(me, &mbx)
                })
            })
            .collect();
        joins
            .into_iter()
            .map(|j| j.join().expect("party panicked"))
            .collect()
    }

    /// The transport `Π_PC` must compute `x ≥ r`, matching the local simulation
    /// that was verified exhaustively — over every `x ≠ r` at ℓ = 5.
    #[test]
    #[ignore] // real 3-party transport
    fn transport_private_compare_computes_ge() {
        const L: usize = 5;
        let cases: Vec<(u64, u64, u64)> = (0..(1u64 << L))
            .flat_map(|x| (0..(1u64 << L)).map(move |r| (x, r)))
            .filter(|(x, r)| x != r)
            .flat_map(|(x, r)| [(x, r, 0u64), (x, r, 1u64)])
            .collect();
        let expected: Vec<u64> = cases.iter().map(|&(x, r, _)| u64::from(x >= r)).collect();

        let out = run3(move |me, mbx| {
            let mut got = Vec::with_capacity(cases.len());
            for (k, &(x, r, beta)) in cases.iter().enumerate() {
                let bits: Vec<FpShare> = (0..L)
                    .map(|i| share_component(Fp::new((x >> i) & 1), 0, me))
                    .collect();
                let b = share_component(Fp::new(beta), 0, me);
                // Any unit works as the mask; 1 keeps the test deterministic.
                let m = share_component(Fp::ONE, 0, me);
                let bprime =
                    private_compare(&bits, r, &b, &m, (k as u64 + 1) * PC_ID_BLOCK, me, mbx);
                got.push(u64::from(bprime) ^ beta);
            }
            got
        });
        for party in &out {
            assert_eq!(party, &expected, "every party must agree with (x ≥ r)");
        }
    }

    /// One party's traffic for `n` back-to-back `Π_PC` calls at width `l`.
    fn measure_pc(n: usize, l: usize) -> (u64, u64) {
        use crate::bench::comm::CountingTransport;
        let handles = InprocHandle::create(3);
        let mut comm = None;
        let mut joins = Vec::new();
        for h in handles {
            let counted = CountingTransport::new(h, 3);
            if comm.is_none() {
                comm = Some(counted.handle());
            }
            joins.push(thread::spawn(move || {
                let mbx = Mailbox::new(counted);
                let me = mbx.id();
                let (x, r) = (0x5A5A_5A5Au64, 0x3C3C_3C3Cu64);
                for k in 0..n {
                    let bits: Vec<FpShare> = (0..l)
                        .map(|i| share_component(Fp::new((x >> i) & 1), 0, me))
                        .collect();
                    let b = share_component(Fp::ZERO, 0, me);
                    let m = share_component(Fp::ONE, 0, me);
                    let out =
                        private_compare(&bits, r, &b, &m, (k as u64 + 1) * PC_ID_BLOCK, me, &mbx);
                    assert_eq!(u64::from(out), u64::from(x >= r));
                }
            }));
        }
        for j in joins {
            j.join().expect("party panicked");
        }
        let s = comm.unwrap().stats();
        (s.messages, s.bytes)
    }

    /// Measure what one `Π_PC` costs on the wire at Falcon's own ℓ = 32.
    ///
    /// Reported as the **marginal** cost of an additional comparison — the
    /// difference between two calls and one — because a single run also carries
    /// the one-time PRF key handshake, which is setup rather than per-comparison
    /// traffic. Quoting the raw first-call figure would overstate the protocol
    /// by that handshake, and at these sizes it is not a rounding error.
    ///
    /// This is the first figure in the reproduction that is comparable to
    /// anything: Falcon reports rounds and bytes, and so does this.
    #[test]
    #[ignore] // real 3-party transport
    fn measure_one_private_compare_at_falcon_width() {
        const L: usize = 32;
        let (m1, b1) = measure_pc(1, L);
        let (m2, b2) = measure_pc(2, L);
        let (dm, db) = (m2 - m1, b2 - b1);
        let rounds = private_compare_rounds(L) as u64;
        let elements = private_compare_elements(L) as u64;
        let p = crate::crypto::field::P;

        println!(
            "\n  Π_PC at ℓ={L}, p={p} — per party, marginal\n    \
             rounds (formula)   : {rounds}\n    \
             messages (measured): {dm}\n    \
             bytes    (measured): {db}\n    \
             field elements     : {elements} (formula)\n    \
             bytes/element      : {:.2}\n    \
             [first call incl. one-time PRF handshake: {m1} msgs, {b1} bytes]\n",
            db as f64 / elements as f64
        );

        assert_eq!(
            dm, rounds,
            "one directed message per party per batched round"
        );
        // ~1 byte per field element plus framing: the check that the narrow
        // encoding is really in use rather than u128 slots, which would be 16×.
        assert!(
            db < elements * 8,
            "field traffic should be ~1 byte/element, got {db} bytes for {elements} elements"
        );
    }

    /// One party's traffic for a *batched* `Π_PC` of `n` comparisons at width
    /// `l`, together with the round count.
    fn measure_pc_batch(n: usize, l: usize) -> (u64, u64) {
        use crate::bench::comm::CountingTransport;
        let handles = InprocHandle::create(3);
        let mut comm = None;
        let mut joins = Vec::new();
        for h in handles {
            let counted = CountingTransport::new(h, 3);
            if comm.is_none() {
                comm = Some(counted.handle());
            }
            joins.push(thread::spawn(move || {
                let mbx = Mailbox::new(counted);
                let me = mbx.id();
                let items: Vec<(Vec<FpShare>, u64)> = (0..n)
                    .map(|k| {
                        let x = (k as u64).wrapping_mul(0x9E37_79B9);
                        let bits = (0..l)
                            .map(|i| share_component(Fp::new((x >> i) & 1), 0, me))
                            .collect();
                        (bits, x ^ 0x5A5A_5A5A)
                    })
                    .collect();
                let betas: Vec<FpShare> =
                    (0..n).map(|_| share_component(Fp::ZERO, 0, me)).collect();
                let masks: Vec<FpShare> = (0..n).map(|_| share_component(Fp::ONE, 0, me)).collect();
                private_compare_batch(&items, &betas, &masks, PC_ID_BLOCK, me, &mbx)
            }));
        }
        let outs: Vec<_> = joins
            .into_iter()
            .map(|j| j.join().expect("panic"))
            .collect();
        for o in &outs[1..] {
            assert_eq!(o, &outs[0], "parties must agree");
        }
        let s = comm.unwrap().stats();
        (s.messages, s.bytes)
    }

    /// The number that can actually be set against Falcon's published
    /// communication: the **marginal cost of one comparison inside a batched
    /// layer**, which is how a deployment runs it.
    ///
    /// Extrapolated to Falcon's LeNet — ~4 700 ReLUs and ~1 120 pooling windows
    /// (3 comparisons each), so ~8 000 DReLUs per inference — this is the figure
    /// that decides whether the reproduction is in the right ballpark or whether
    /// something is wrong with it. Printing the extrapolation rather than only
    /// the unit cost is deliberate: a per-op number that is never carried
    /// through to the workload is how an implementation stays plausibly wrong.
    /// **Where the offline bytes actually go**, which is what decides whether any
    /// given optimisation of the offline phase is worth building.
    ///
    /// `the_online_drelu_figure_is_not_the_whole_cost` shows preprocessing is 91% of
    /// the cost. That alone does not say *which part* of preprocessing, and the answer
    /// matters: a construction that removes the `a2b` is worth building only if the
    /// `a2b` is where the bytes are.
    ///
    /// Measured at ℓ = 32 it splits **54/46**: the `ℓ` daBits that supply the mask's
    /// bits over `Z_p` cost 304 B/element, and computing δ — the `a2b` plus the
    /// ring→`Z₂` conversion — costs 255 B. Neither half is negligible, so this is a
    /// split rather than a single target, and an arithmetic guess put it at 79/21
    /// before it was measured.
    ///
    /// What it settles is narrower and was the point: **replacing the `a2b` with a
    /// garbled comparator is not worth building.** That construction (derived in
    /// `docs/falcon-reproduction.md`) costs `ℓ` AND gates at 32 B each — 1024 B at
    /// ℓ = 32 — against the 255 B it would replace. It looked cheaper when it was
    /// derived because the `a2b` then ran at a 128-bit wire; narrowing that to
    /// `DELTA_WIRE_BITS = 4` made the thing it was meant to improve four times cheaper
    /// than the improvement.
    #[test]
    #[ignore] // real 3-party transport
    fn the_offline_bytes_split_between_dabits_and_the_delta_computation() {
        use crate::bench::comm::CountingTransport;
        const LB: u32 = 32;
        const N: usize = 400;

        // Each phase measured on its own, so the split is measured rather than
        // apportioned. Same batch size, so the numbers are directly comparable.
        let dabits = {
            let handles = InprocHandle::create(3);
            let mut comm = None;
            let mut joins = Vec::new();
            for h in handles {
                let counted = CountingTransport::new(h, 3);
                if comm.is_none() {
                    comm = Some(counted.handle());
                }
                joins.push(thread::spawn(move || {
                    let mbx = Mailbox::new(counted);
                    let me = mbx.id();
                    // ℓ per element, exactly as `random_wrap_masks` draws them.
                    let _ = random_dabits_triple(N * LB as usize, LB as usize, 700_000, me, &mbx);
                }));
            }
            for j in joins {
                j.join().expect("party panicked");
            }
            comm.unwrap().stats().bytes
        };

        // The whole offline phase, for reference.
        let (off, _on) = measure_drelu_phases(N, LB);
        let a2b_and_rest = off.bytes.saturating_sub(dabits);

        println!(
            "\n  offline composition at ℓ={LB}, n={N} — party 0\n    \
             ℓ daBits per element : {:>9} B  ({:>4.0} B/element, {:.0}%)\n    \
             a2b + conversion     : {:>9} B  ({:>4.0} B/element, {:.0}%)\n    \
             offline total        : {:>9} B  ({:>4.0} B/element)\n",
            dabits,
            dabits as f64 / N as f64,
            100.0 * dabits as f64 / off.bytes as f64,
            a2b_and_rest,
            a2b_and_rest as f64 / N as f64,
            100.0 * a2b_and_rest as f64 / off.bytes as f64,
            off.bytes,
            off.bytes as f64 / N as f64,
        );

        assert!(
            dabits > a2b_and_rest,
            "the daBits are expected to dominate the offline phase: daBits {dabits} B \
             vs everything else {a2b_and_rest} B. If this inverts, the δ computation \
             has become the target and a cheaper wrap3 is worth revisiting."
        );
    }

    /// **Which side of Falcon's line this reproduction lands on, once preprocessing
    /// is counted.**
    ///
    /// `the_lenet_extrapolation_holds_at_lenet_scale` measures the *online* DReLU at
    /// 52.8 B, below Falcon's analytic `3kn`. Taken alone that reads as this
    /// reproduction beating the paper it reproduces, which is the shape of every claim
    /// this repo has had to withdraw — so the first explanation to eliminate is the
    /// obvious one: that measurement runs against dealt preprocessing and counts no
    /// offline traffic at all.
    ///
    /// This measures both phases with the real offline phase, at two batch sizes, and
    /// reports the marginal total. It does not attempt to decide what Falcon's `3kn`
    /// covers — that is a question about the paper's accounting, not this code — but it
    /// does settle whether the online figure is the whole cost. It is not.
    #[test]
    #[ignore] // real 3-party transport, generates preprocessing for the whole batch
    fn the_online_drelu_figure_is_not_the_whole_cost() {
        const LB: u32 = 32;
        let (o1, n1) = measure_drelu_phases(500, LB);
        let (o2, n2) = measure_drelu_phases(1000, LB);

        // Marginals, so the fixed setup cancels out of both phases.
        let d = 500.0;
        let off_per = (o2.bytes - o1.bytes) as f64 / d;
        let on_per = (n2.bytes - n1.bytes) as f64 / d;
        let total = off_per + on_per;

        let k = crate::crypto::field::ELEM_BITS as f64;
        let falcon_3kn = 3.0 * LB as f64 * k / 8.0;
        let drelus = 8044.0; // LeNet's count

        println!(
            "\n  Π_DReLU marginal cost at ℓ={LB}, BOTH phases, per party\n    \
             offline : {off_per:8.1} B/DReLU\n    \
             online  : {on_per:8.1} B/DReLU\n    \
             total   : {total:8.1} B/DReLU   (Falcon 3kn = {falcon_3kn:.1})\n    \
             LeNet   : {:.2} MB/party total, of which {:.2} MB online\n    \
             offline share: {:.0}%\n",
            total * drelus / 1e6,
            on_per * drelus / 1e6,
            100.0 * off_per / total,
        );

        assert!(
            off_per > 0.0 && on_per > 0.0,
            "both phases must communicate — a zero means the split measures nothing"
        );
        // The claim being pinned: the online figure understates the real cost, so
        // comparing it against a figure that may be a total is not a like-for-like
        // comparison. If preprocessing ever became the smaller half, the reasoning in
        // `docs/falcon-reproduction.md` about the 0.42-vs-0.74 gap would need redoing.
        assert!(
            off_per > on_per,
            "preprocessing is expected to dominate: offline {off_per:.1} vs online \
             {on_per:.1} B/DReLU"
        );
    }

    /// **The LeNet figure is an extrapolation; this measures the assumption it
    /// rests on.**
    ///
    /// `measure_batched_private_compare_and_extrapolate_to_lenet` computes LeNet's
    /// cost as Falcon's analytic `DReLU = 4kn` times a counted 8 044 DReLUs. The
    /// per-operation term is checked against measurement elsewhere, but the *scaling*
    /// never was: everything was measured at n = 200–1000 and multiplied up by 8–40×.
    /// If batching degraded at scale — an extra round, a growing per-element overhead,
    /// a message-size cliff — the extrapolation would overstate how well this
    /// reproduction holds up, and nothing would have caught it.
    ///
    /// So this measures the online DReLU at LeNet's actual batch size. Two claims
    /// follow if it passes, and both are load-bearing for the comparison against
    /// Falcon's reported 0.74 MB:
    ///
    /// - **Marginal bytes per DReLU are flat in batch size**, so multiplying up is
    ///   legitimate rather than optimistic.
    /// - **Round count is constant in batch size**, which is the property that makes
    ///   a whole layer one synchronisation instead of thousands.
    ///
    /// A true end-to-end LeNet also folds in the linear layers and the scheme switches.
    /// ✅ **Done 2026-08-04** (`bench::cnn::lenet_end_to_end_is_exact_and_costed`): kernel
    /// size became a per-layer field, and LeNet-5 measures **4 693** comparisons rather than
    /// the 8 044 below, because Baru max-pools *before* the nonlinearity — they commute, so
    /// ReLU runs on 1 120 pooled values instead of 4 480 unpooled ones.
    ///
    /// This test is kept as-is. Its subject is whether the per-DReLU cost is **flat in batch
    /// size**, which is a property of the primitive and is what makes any extrapolation
    /// legitimate; 8 044 is simply the batch size it uses to check that. The end-to-end run
    /// vindicated it — 329 B per comparison at full-network scale against 323 B here, 2%
    /// apart — so the method was sound and only the count it was multiplied by was wrong.
    #[test]
    #[ignore] // real 3-party transport, and the largest batch here is slow
    fn the_lenet_extrapolation_holds_at_lenet_scale() {
        const L: u32 = 32;
        // LeNet's own count, as the extrapolation uses it.
        let relus = 6 * 24 * 24 + 16 * 8 * 8 + 120 + 84;
        let windows = 6 * 12 * 12 + 16 * 4 * 4;
        let drelus = relus + 3 * windows;
        assert_eq!(drelus, 8044, "LeNet's DReLU count, as the extrapolation states");

        // Marginal cost between successive sizes: the fixed framing cancels.
        let sizes = [500usize, 1000, 2000, 4000, drelus];
        let mut prev: Option<(usize, u64)> = None;
        let mut marginals = Vec::new();
        let mut round_counts = Vec::new();
        for n in sizes {
            let (msgs, bytes) = measure_drelu(n, L);
            round_counts.push(msgs);
            if let Some((pn, pb)) = prev {
                marginals.push(((n, pn), (bytes - pb) as f64 / (n - pn) as f64));
            }
            prev = Some((n, bytes));
        }

        let per: Vec<f64> = marginals.iter().map(|&(_, m)| m).collect();
        let (lo, hi) = per.iter().fold((f64::MAX, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
        let k = crate::crypto::field::ELEM_BITS as f64;
        let falcon_4kn = 4.0 * L as f64 * k / 8.0;

        println!(
            "\n  online Π_DReLU marginal cost vs batch size, ℓ={L}\n    \
             {}\n    \
             spread {lo:.1}–{hi:.1} B/DReLU  (Falcon's 4kn = {falcon_4kn:.1})\n    \
             rounds {round_counts:?}  (must be constant)\n    \
             LeNet {drelus} DReLUs → {:.2} MB/party at the measured rate\n",
            marginals
                .iter()
                .map(|&((n, pn), m)| format!("{pn}→{n}: {m:.1} B"))
                .collect::<Vec<_>>()
                .join("  "),
            hi * drelus as f64 / 1e6,
        );

        assert!(
            round_counts.windows(2).all(|w| w[0] == w[1]),
            "round count must not grow with batch size: {round_counts:?}"
        );
        // Flat to within 5%: wider would mean the per-element cost depends on the
        // batch, and multiplying a small-batch measurement up to 8 044 would then be
        // an assumption rather than arithmetic.
        assert!(
            hi <= lo * 1.05,
            "marginal bytes/DReLU must be flat in batch size, got {lo:.1}–{hi:.1} \
             ({:.1}% spread) — the extrapolation to {drelus} DReLUs is only valid if \
             this is",
            100.0 * (hi / lo - 1.0)
        );
    }

    #[test]
    #[ignore] // real 3-party transport
    fn measure_batched_private_compare_and_extrapolate_to_lenet() {
        const L: usize = 32;
        let (m1, b1) = measure_pc_batch(500, L);
        let (m2, b2) = measure_pc_batch(1000, L);
        let per_cmp_bytes = (b2 - b1) as f64 / 500.0;
        let rounds = private_compare_rounds(L) as u64;

        // Falcon's LeNet: conv1 6·24·24 + conv2 16·8·8 + fc 120 + fc 84 ReLUs,
        // plus 3 comparisons per 2×2 pooling window over 6·12·12 and 16·4·4.
        let relus = 6 * 24 * 24 + 16 * 8 * 8 + 120 + 84;
        let windows = 6 * 12 * 12 + 16 * 4 * 4;
        let drelus = relus + 3 * windows;

        // A DReLU is *not* one Π_PC. Falcon's table gives Π_PC = 2kn and
        // ReLU/DReLU = 4kn, so a DReLU is twice a private compare — the extra
        // covers wrap3's own kn and the select's kn. Extrapolating from Π_PC
        // alone would understate the network by 2×, which an earlier revision of
        // this test did.
        let k = crate::crypto::field::ELEM_BITS as f64;
        let drelu_bytes = 4.0 * L as f64 * k / 8.0;
        let pc_bytes_formula = 2.0 * L as f64 * k / 8.0;
        let per_party = drelu_bytes * drelus as f64;

        println!(
            "\n  Π_PC batched at ℓ={L}, p={} ({} bits/element) — per party\n    \
             rounds per batch      : {rounds} (independent of batch size)\n    \
             messages @ n=1000     : {m2}\n    \
             marginal bytes/compare: {per_cmp_bytes:.1}  (measured)\n    \
             Falcon's 2kn formula  : {pc_bytes_formula:.1}  (+1 element for our reveal)\n\n  \
             LeNet extrapolation via Falcon's DReLU = 4kn = {drelu_bytes:.0} B/party\n    \
             {drelus} DReLUs ({relus} ReLU + 3×{windows} pool)\n    \
             per party  : {:.2} MB\n    \
             Falcon reports: 0.74 MB/party (LeNet, semi-honest LAN)\n",
            crate::crypto::field::P,
            crate::crypto::field::ELEM_BITS,
            per_party / 1.0e6,
        );

        // The measured cost must match Falcon's Π_PC formula plus our one
        // revealed element.
        let expected = pc_bytes_formula + k / 8.0;
        assert!(
            (per_cmp_bytes - expected).abs() < 1.0,
            "measured {per_cmp_bytes:.1} B/compare should match Falcon's 2kn + reveal = {expected:.1}"
        );

        assert_eq!(m1, m2, "batching must not change the round/message count");
        assert!(
            per_cmp_bytes < 100.0,
            "marginal cost should approach the {} field elements, got {per_cmp_bytes:.1} bytes",
            private_compare_elements(L)
        );
    }

    /// Build the preprocessing for one element, centrally — which is exactly
    /// what a preprocessing functionality does. `δ = wrap3(x₁,x₂,x₃)` is not
    /// derivable by any single party (each is missing a summand), so it has to
    /// arrive this way rather than be computed locally.
    fn preproc(x_sum: [u64; 3], beta: u64, l_bits: u32, me: PartyId) -> WrapPreproc {
        use crate::crypto::aby3_protocol::share_component as ring_component;
        let mask = (1u128 << l_bits) - 1;
        let x = ((x_sum[0] as u128 + x_sum[1] as u128 + x_sum[2] as u128) & mask) as u64;
        // Ring shares of the mask, with the summands placed by component.
        let xs = PartyShare {
            left: x_sum[me % 3] as u128,
            right: x_sum[(me + 1) % 3] as u128,
        };
        let bits = (0..l_bits)
            .map(|i| share_component(Fp::new((x >> i) & 1), 0, me))
            .collect();
        let delta = crate::crypto::falcon::wrap3(x_sum[0], x_sum[1], x_sum[2], l_bits);
        let _ = ring_component(0u128, 0, me); // keep the import honest
        WrapPreproc {
            x: xs,
            bits,
            // Replicated bit: the whole value in component 0.
            delta: [u64::from(me == 0) * delta, u64::from(me == 2) * delta],
            beta_fp: share_component(Fp::new(beta), 0, me),
            beta_bit: [u64::from(me == 0) * beta, u64::from(me == 2) * beta],
            m: share_component(Fp::ONE, 0, me),
        }
    }

    /// `Π_DReLU` over the real transport must agree with the signed reading of
    /// the shared value, on every input — the doubling means no input has to be
    /// excluded.
    ///
    /// This is the local simulation's `drelu_is_exact_on_every_input` carried
    /// onto real messages, and it is what makes the DReLU cost measurable rather
    /// than derived from Falcon's formula.
    #[test]
    #[ignore] // real 3-party transport
    fn transport_drelu_matches_the_signed_value() {
        const LB: u32 = 8;
        let l = 1u64 << LB;
        // Every value in the ring, split into three summands deterministically.
        let vals: Vec<u64> = (0..l).collect();
        let splits: Vec<[u64; 3]> = vals
            .iter()
            .map(|&v| {
                let s0 = v.wrapping_mul(37) % l;
                let s1 = v.wrapping_mul(91).wrapping_add(11) % l;
                [s0, s1, (v + 2 * l - s0 - s1) % l]
            })
            .collect();
        let masks: Vec<[u64; 3]> = (0..l)
            .map(|i| {
                let a = i.wrapping_mul(53).wrapping_add(7) % l;
                let b = i.wrapping_mul(29).wrapping_add(3) % l;
                [a, b, (i.wrapping_mul(17) + 5) % l]
            })
            .collect();
        let expected: Vec<u64> = vals
            .iter()
            .map(|&v| u64::from(crate::crypto::falcon::signed(v, LB) >= 0))
            .collect();

        let out = run3(move |me, mbx| {
            let a: Vec<PartyShare> = splits
                .iter()
                .map(|s| PartyShare {
                    left: s[me % 3] as u128,
                    right: s[(me + 1) % 3] as u128,
                })
                .collect();
            let pre: Vec<WrapPreproc> = masks
                .iter()
                .enumerate()
                .map(|(i, m)| preproc(*m, (i % 2) as u64, LB, me))
                .collect();
            drelu_batch(&a, &pre, LB, PC_ID_BLOCK, me, mbx)
        });

        // Reconstruct each output bit from the three parties' components.
        for (i, want) in expected.iter().enumerate() {
            let got = (out[0][i][0] ^ out[1][i][0] ^ out[2][i][0]) & 1;
            assert_eq!(
                got,
                *want,
                "DReLU at value {i} (signed {})",
                crate::crypto::falcon::signed(i as u64, LB)
            );
        }
    }

    /// One party's traffic for a batched `Π_DReLU` of `n` elements at ℓ = 32.
    /// One phase's cost: wall time, messages and bytes for party 0.
    #[derive(Debug, Clone, Copy)]
    struct Phase {
        ms: f64,
        messages: u64,
        bytes: u64,
    }

    /// **The width-matched byte comparison against Falcon's reported LeNet figure.**
    ///
    /// Every byte number on `falcon-reproduction.md` was at `ℓ = 20` (the hybrid's
    /// activations calibrate there) against Falcon's `ℓ = 32`. Comparing them is comparing
    /// two ring widths, and this page's whole history is wrong answers of exactly that
    /// shape — so this measures **at their width**, and reports the quantities separately
    /// rather than picking the flattering one.
    ///
    /// # Why this is MPC-only, and why that is right
    ///
    /// The hybrid cannot run at `ℓ = 32`: `CnnModel` asserts `log2_t ≤ 30` because that is
    /// the depth-1 BFV noise budget at `n = 4096, q = 2^100`. But Falcon has **no FHE side**
    /// — its 0.74 MB is nonlinearity traffic — so the comparable quantity is the DReLU
    /// traffic, not a pipeline total. Reporting our FHE bytes against their MPC bytes would
    /// be the same category error in a new place.
    ///
    /// # Two DReLU counts, because the schedule differs
    ///
    /// `4 693` is what Baru's LeNet-5 actually asks for (max-pool before the nonlinearity;
    /// see `bench::cnn::lenet_end_to_end_is_exact_and_costed`). `8 044` is what an
    /// ReLU-then-pool schedule needs and what Falcon's own count corresponds to. Both are
    /// reported: the first is our cost, the second is the like-for-like workload.
    #[test]
    #[ignore] // real 3-party transport at LeNet scale, ℓ = 32
    fn the_width_matched_lenet_byte_comparison() {
        const L: u32 = 32; // Falcon's ring width
        const FALCON_REPORTED_MB: f64 = 0.74;

        for (label, n) in [("baru schedule (pool→relu)", 4693usize), ("relu→pool", 8044)] {
            let (off, on) = measure_drelu_phases(n, L);
            let total_mb = (off.bytes + on.bytes) as f64 / 1e6;
            let online_mb = on.bytes as f64 / 1e6;
            println!(
                "\n  {label}, n={n}, ℓ={L}, per party:\n    \
                 offline {:.2} MB  online {:.2} MB  total {:.2} MB\n    \
                 vs Falcon reported {FALCON_REPORTED_MB} MB → \
                 total {:.2}x, online {:.2}x",
                off.bytes as f64 / 1e6,
                online_mb,
                total_mb,
                total_mb / FALCON_REPORTED_MB,
                online_mb / FALCON_REPORTED_MB,
            );
            assert!(off.bytes > 0 && on.bytes > 0, "{label}: both phases must be measured");
            // The offline phase dominates, which is the fact that decides which number is
            // quotable against theirs — and it is why this reports both rather than one.
            assert!(
                off.bytes > on.bytes,
                "{label}: preprocessing should dominate; offline {} online {}",
                off.bytes, on.bytes
            );
        }

        println!(
            "\n  ⚠ Falcon's 0.74 MB is semi-honest LAN and this measurement does not \
             establish\n    whether it includes preprocessing. The total and online figures \
             above bracket\n    the comparison; quoting one of them as *the* answer is the \
             error this page\n    has already made twice."
        );
    }

    /// **`Π_DReLU`'s offline and online costs, measured apart.**
    ///
    /// This is the measurement `docs/falcon-reproduction.md` needed and could not
    /// make: `TransportBackend` runs Falcon's preprocessing inline, so every latency
    /// it reports is a *total*, and Falcon's own claim is about the **online** phase.
    /// Charging one protocol for a phase its competitor does not have is what
    /// invalidated the crossover in the first place — and reporting only totals is
    /// what let that go unnoticed.
    ///
    /// Nothing here needs a backend refactor, because [`drelu_batch`] already takes
    /// its preprocessing as a parameter: generate it, stop the clock, then run the
    /// online protocol against it.
    /// **Where Falcon's cost actually sits, offline against online.**
    ///
    /// The crossover in `docs/falcon-reproduction.md` was withdrawn three times, and
    /// the first withdrawal was because Falcon's preprocessing had been free (dealt
    /// from a public seed) while garbled circuits paid for everything. Once it was
    /// generated for real, every reported figure became a *total* — which is the
    /// opposite accounting error, because Falcon is designed to put work offline and
    /// its published numbers are online-only.
    ///
    /// This test measures the two phases apart so both claims can be stated without
    /// either bias. It asserts only what it can: that the split is lopsided in the
    /// direction Falcon's design implies. The printed figures are what the docs
    /// should quote when comparing against a protocol with no offline phase.
    #[test]
    #[ignore] // real 3-party transport
    fn falcons_cost_is_mostly_offline() {
        const LB: u32 = 20;
        const N: usize = 256;
        let (off, on) = measure_drelu_phases(N, LB);
        let per = |p: Phase| (p.bytes as f64 / N as f64, p.messages);

        println!(
            "\n  Π_DReLU at ℓ={LB}, n={N} — party 0, phases apart\n    \
             offline : {:>8.1} ms  {:>4} messages  {:>9.2} MB  ({:.0} B/op)\n    \
             online  : {:>8.1} ms  {:>4} messages  {:>9.2} MB  ({:.0} B/op)\n    \
             offline share: {:.0}% of bytes, {:.0}% of messages\n",
            off.ms,
            off.messages,
            off.bytes as f64 / 1e6,
            per(off).0,
            on.ms,
            on.messages,
            on.bytes as f64 / 1e6,
            per(on).0,
            100.0 * off.bytes as f64 / (off.bytes + on.bytes) as f64,
            100.0 * off.messages as f64 / (off.messages + on.messages) as f64,
        );

        assert!(
            off.bytes > 0 && on.bytes > 0,
            "both phases must actually communicate — a zero means the split is \
             measuring nothing"
        );
        // The design claim: preprocessing is the expensive half. If this ever
        // inverts, Falcon's online-only figures stop being the flattering ones and
        // the comparison against garbled circuits needs revisiting in the other
        // direction.
        assert!(
            off.bytes > on.bytes,
            "expected preprocessing to dominate bytes, got offline {} vs online {}",
            off.bytes,
            on.bytes
        );
    }

    fn measure_drelu_phases(n: usize, l_bits: u32) -> (Phase, Phase) {
        use crate::bench::comm::CountingTransport;
        use std::time::Instant;
        let l = 1u64 << l_bits;
        let handles = InprocHandle::create(3);
        let mut joins = Vec::new();
        for h in handles {
            let counted = CountingTransport::new(h, 3);
            let handle = counted.handle();
            joins.push(thread::spawn(move || {
                let mbx = Mailbox::new(counted);
                let me = mbx.id();
                let a: Vec<PartyShare> = (0..n)
                    .map(|i| {
                        let v = (i as u64).wrapping_mul(0x9E37) % l;
                        let s = [v / 3, v / 3, v - 2 * (v / 3)];
                        PartyShare {
                            left: s[me % 3] as u128,
                            right: s[(me + 1) % 3] as u128,
                        }
                    })
                    .collect();

                // ── Offline: everything query-independent ──
                let t0 = Instant::now();
                let off = 1_000_000;
                let wm = random_wrap_masks(n, l_bits, off, me, &mbx);
                let off2 = off + wrap_mask_id_span(n, l_bits);
                let (beta_z2, beta_fp) = random_dabits_fp(n, off2, me, &mbx);
                let off3 = off2 + dabit_id_span(n);
                let masks = random_nonzero_masks(n, off3, me, &mbx);
                let pre: Vec<WrapPreproc> = wm
                    .into_iter()
                    .enumerate()
                    .map(|(i, (x, bits, delta))| WrapPreproc {
                        x,
                        bits,
                        delta,
                        beta_fp: beta_fp[i],
                        beta_bit: beta_z2[i],
                        m: masks[i],
                    })
                    .collect();
                let offline = Phase {
                    ms: t0.elapsed().as_secs_f64() * 1e3,
                    messages: handle.stats().messages,
                    bytes: handle.stats().bytes,
                };

                // ── Online: the comparison itself ──
                let t1 = Instant::now();
                let _bits = drelu_batch(&a, &pre, l_bits, PC_ID_BLOCK, me, &mbx);
                let after = handle.stats();
                let online = Phase {
                    ms: t1.elapsed().as_secs_f64() * 1e3,
                    messages: after.messages - offline.messages,
                    bytes: after.bytes - offline.bytes,
                };
                (offline, online)
            }));
        }
        let mut out: Vec<(Phase, Phase)> = joins
            .into_iter()
            .map(|j| j.join().expect("party panicked"))
            .collect();
        out.remove(0)
    }

    /// `Π_Trunc`'s offline and online costs, measured apart — the truncation analogue of
    /// [`measure_drelu_phases`].
    fn measure_trunc_phases(n: usize, l_bits: u32, d: u32) -> (Phase, Phase) {
        use crate::bench::comm::CountingTransport;
        use std::time::Instant;
        let handles = InprocHandle::create(3);
        let mut joins = Vec::new();
        for h in handles {
            let counted = CountingTransport::new(h, 3);
            let handle = counted.handle();
            joins.push(thread::spawn(move || {
                let mbx = Mailbox::new(counted);
                let me = mbx.id();
                // Non-negative inputs below 2^(l-1), as a ReLU output is.
                let a: Vec<PartyShare> = (0..n)
                    .map(|i| {
                        let v = ((i as u64).wrapping_mul(7)) % (1 << (l_bits - 1));
                        let s = [v / 3, v / 3, v - 2 * (v / 3)];
                        PartyShare {
                            left: s[me % 3] as u128,
                            right: s[(me + 1) % 3] as u128,
                        }
                    })
                    .collect();

                // ── Offline: the pairs ──
                let t0 = Instant::now();
                let pre = random_trunc_pairs(n, d, l_bits, l_bits, 2_000_000, me, &mbx);
                let offline = Phase {
                    ms: t0.elapsed().as_secs_f64() * 1e3,
                    messages: handle.stats().messages,
                    bytes: handle.stats().bytes,
                };

                // ── Online: the truncation itself ──
                let t1 = Instant::now();
                let _out = truncate_batch(&a, &pre, d, l_bits, PC_ID_BLOCK, me, &mbx);
                let after = handle.stats();
                let online = Phase {
                    ms: t1.elapsed().as_secs_f64() * 1e3,
                    messages: after.messages - offline.messages,
                    bytes: after.bytes - offline.bytes,
                };
                (offline, online)
            }));
        }
        let mut out: Vec<(Phase, Phase)> = joins
            .into_iter()
            .map(|j| j.join().expect("party panicked"))
            .collect();
        out.remove(0)
    }

    /// **How much of a warm Falcon ReLU's online cost is the truncation?**
    ///
    /// With the offline phase pooled, a warm server's cost is nearly all rounds — its online
    /// bytes are ~9% of the total at ℓ = 32 — so this decides whether the one obvious
    /// remaining optimisation is worth building. [`truncate_batch`] runs **four** sequential
    /// interactive stages (open `c`, an `a2b` for the borrow, an `a2b` for the wrap, then two
    /// `Π_SS` lifts) and the borrow and wrap are mutually independent: they could share one
    /// batched `a2b` and one batched lift, halving the stage count.
    ///
    /// The share is measured before anything is built on it, because the last two structural
    /// estimates on this page were both badly wrong — the cheaper-δ idea (withdrawn after
    /// measurement) and a 10× over-estimate of how much preprocessing was unpooled. A
    /// protocol having redundant round-trips is not by itself a reason to fix them.
    #[test]
    #[ignore] // real 3-party transport
    fn how_much_of_a_warm_relu_is_the_truncation() {
        const L: u32 = 20;
        const D: u32 = 8;
        let n = 256usize;

        let (d_off, d_on) = measure_drelu_phases(n, L);
        let (t_off, t_on) = measure_trunc_phases(n, L, D);

        // A warm ReLU's online phase is the DReLU's online + the truncation's online (plus
        // one batched multiply, which is a single round and is not isolated here).
        let online_msgs = d_on.messages + t_on.messages;
        let online_bytes = d_on.bytes + t_on.bytes;
        println!("\n  per party, n={n}, ℓ={L}, d={D}:");
        println!(
            "    Π_DReLU   offline {:>5} msgs {:>9} B | online {:>4} msgs {:>8} B",
            d_off.messages, d_off.bytes, d_on.messages, d_on.bytes
        );
        println!(
            "    Π_Trunc   offline {:>5} msgs {:>9} B | online {:>4} msgs {:>8} B",
            t_off.messages, t_off.bytes, t_on.messages, t_on.bytes
        );
        println!(
            "    → truncation is {:.0}% of the warm online rounds and {:.0}% of its bytes",
            100.0 * t_on.messages as f64 / online_msgs.max(1) as f64,
            100.0 * t_on.bytes as f64 / online_bytes.max(1) as f64,
        );
        println!(
            "    → and {:.0}% of the OFFLINE bytes, which is what a cold server pays",
            100.0 * t_off.bytes as f64
                / (t_off.bytes + d_off.bytes).max(1) as f64,
        );
        println!(
            "\n    The two Π_SS lifts are already batched (free: both run at ℓ). Batching the\n    \
             two `a2b`s as well was tried and withdrawn — it cut these rounds 22 → 12 but\n    \
             raised online bytes 45%, because one call needs one width and the borrow's\n    \
             d+1 columns must widen to the wrap's ℓ+1. See `truncate_batch`."
        );

        assert!(d_on.messages > 0 && t_on.messages > 0, "both phases must be measured");
        // NOT asserted: that the truncation's preprocessing dominates its online phase. It
        // does not, and the first version of this test asserted it did — reasoning by
        // analogy with `Π_DReLU`, where preprocessing is ~91% of bytes. `Π_Trunc` is the
        // opposite shape: its offline is `ℓ` daBits, while its online **opens a full ring
        // element per value** and then runs two `a2b`s. That asymmetry is the finding, and
        // it explains why pooling the pairs bought only ~5%: there was little there to hoist.
        assert!(
            t_on.bytes > t_off.bytes,
            "the truncation is online-heavy, unlike Π_DReLU: online {} vs offline {}",
            t_on.bytes, t_off.bytes
        );
    }

    fn measure_drelu(n: usize, l_bits: u32) -> (u64, u64) {
        use crate::bench::comm::CountingTransport;
        let l = 1u64 << l_bits;
        let handles = InprocHandle::create(3);
        let mut comm = None;
        let mut joins = Vec::new();
        for h in handles {
            let counted = CountingTransport::new(h, 3);
            if comm.is_none() {
                comm = Some(counted.handle());
            }
            joins.push(thread::spawn(move || {
                let mbx = Mailbox::new(counted);
                let me = mbx.id();
                let a: Vec<PartyShare> = (0..n)
                    .map(|i| {
                        let v = (i as u64).wrapping_mul(0x9E37) % l;
                        let s = [v / 3, v / 3, v - 2 * (v / 3)];
                        PartyShare {
                            left: s[me % 3] as u128,
                            right: s[(me + 1) % 3] as u128,
                        }
                    })
                    .collect();
                let pre: Vec<WrapPreproc> = (0..n)
                    .map(|i| {
                        let k = i as u64;
                        preproc(
                            [
                                k.wrapping_mul(53) % l,
                                k.wrapping_mul(29) % l,
                                k.wrapping_mul(17) % l,
                            ],
                            (i % 2) as u64,
                            l_bits,
                            me,
                        )
                    })
                    .collect();
                drelu_batch(&a, &pre, l_bits, PC_ID_BLOCK, me, &mbx)
            }));
        }
        for j in joins {
            j.join().expect("panic");
        }
        let s = comm.unwrap().stats();
        (s.messages, s.bytes)
    }

    /// **The measured DReLU cost, against Falcon's published `4kn`.**
    ///
    /// Up to now the LeNet figure came from Falcon's own formula; this measures
    /// our construction instead. The marginal cost is the difference between two
    /// batch sizes, so the one-time handshake and any fixed overhead drop out.
    #[test]
    #[ignore] // real 3-party transport
    fn measure_drelu_against_falcons_4kn() {
        const L: u32 = 32;
        let (m1, b1) = measure_drelu(200, L);
        let (m2, b2) = measure_drelu(400, L);
        let per = (b2 - b1) as f64 / 200.0;

        let k = crate::crypto::field::ELEM_BITS as f64;
        // Falcon's table lists `wrap3 = 3kn` and, on a single row, "ReLU **and**
        // Derivative of ReLU = 4kn". A DReLU is wrap3 plus a local XOR, so 3kn
        // is the line to compare against; the extra kn is `Π_SS`, the select
        // that turns the bit into ReLU — which is *not* implemented here yet.
        // Comparing against 4kn would credit us for work we have not done.
        let falcon_wrap3 = 3.0 * L as f64 * k / 8.0;
        let falcon_relu = 4.0 * L as f64 * k / 8.0;
        let p = crate::crypto::field::P;
        let drelus = 6 * 24 * 24 + 16 * 8 * 8 + 120 + 84 + 3 * (6 * 12 * 12 + 16 * 4 * 4);

        println!(
            "\n  Π_DReLU batched at ℓ={L}, p={p} — per party\n    \
             messages           : {m2} (batch-size independent: {m1} at n=200)\n    \
             marginal bytes/op  : {per:.1}  (measured)\n    \
             Falcon wrap3 = 3kn : {falcon_wrap3:.1}   ratio {:.2}x\n    \
             Falcon ReLU  = 4kn : {falcon_relu:.1}   (adds Π_SS, not yet built here)\n\n  \
             LeNet ({drelus} DReLUs), DReLU only\n    \
             measured : {:.2} MB/party\n    \
             Falcon's full ReLU line would be {:.2} MB/party against 0.74 reported\n",
            per / falcon_wrap3,
            per * drelus as f64 / 1.0e6,
            falcon_relu * drelus as f64 / 1.0e6,
        );

        assert_eq!(m1, m2, "round count must not depend on batch size");
        // Within 2× of the wrap3 line either way: closer would be luck, further
        // would mean the construction differs from the paper's.
        assert!(
            per > falcon_wrap3 / 2.0 && per < falcon_wrap3 * 2.0,
            "measured {per:.1} B/DReLU should bracket Falcon's wrap3 {falcon_wrap3:.1}"
        );
    }

    fn sel_preproc(seed: u64, l_bits: u32, me: PartyId) -> SelectPreproc {
        let l = 1u128 << l_bits;
        // A random bit shared consistently in both rings.
        let c = [seed % 2, (seed / 2) % 2, (seed / 4) % 2];
        let cbit = (c[0] ^ c[1] ^ c[2]) & 1;
        // Ring sharing of the *same* value: components must reconstruct to cbit.
        let cr = [cbit as u128, 0u128, 0u128];
        let _ = l;
        SelectPreproc {
            c_bit: [c[me % 3], c[(me + 1) % 3]],
            c_ring: PartyShare {
                left: cr[me % 3],
                right: cr[(me + 1) % 3],
            },
        }
    }

    /// `Π_ReLU` over the transport must equal `max(0, a)` — the whole of
    /// Falcon's inference nonlinearity, end to end on real messages.
    #[test]
    #[ignore] // real 3-party transport
    fn transport_relu_matches_max_zero_a() {
        const LB: u32 = 8;
        let l = 1u64 << LB;
        let vals: Vec<u64> = (0..l).collect();
        let splits: Vec<[u64; 3]> = vals
            .iter()
            .map(|&v| {
                let s0 = v.wrapping_mul(37) % l;
                let s1 = v.wrapping_mul(91).wrapping_add(11) % l;
                [s0, s1, (v + 2 * l - s0 - s1) % l]
            })
            .collect();
        let masks: Vec<[u64; 3]> = (0..l)
            .map(|i| {
                [
                    i.wrapping_mul(53).wrapping_add(7) % l,
                    i.wrapping_mul(29).wrapping_add(3) % l,
                    i.wrapping_mul(17).wrapping_add(5) % l,
                ]
            })
            .collect();
        let expected: Vec<u64> = vals
            .iter()
            .map(|&v| {
                if crate::crypto::falcon::signed(v, LB) >= 0 {
                    v
                } else {
                    0
                }
            })
            .collect();

        let out = run3(move |me, mbx| {
            let a: Vec<PartyShare> = splits
                .iter()
                .map(|s| PartyShare {
                    left: s[me % 3] as u128,
                    right: s[(me + 1) % 3] as u128,
                })
                .collect();
            let wp: Vec<WrapPreproc> = masks
                .iter()
                .enumerate()
                .map(|(i, m)| preproc(*m, (i % 2) as u64, LB, me))
                .collect();
            let sp: Vec<SelectPreproc> = (0..a.len())
                .map(|i| sel_preproc(i as u64, LB, me))
                .collect();
            relu_batch(&a, &wp, &sp, LB, PC_ID_BLOCK, me, mbx)
        });

        let mask = (1u128 << LB) - 1;
        for (i, want) in expected.iter().enumerate() {
            let got = ((out[0][i].left + out[1][i].left + out[2][i].left) & mask) as u64;
            assert_eq!(
                got,
                *want,
                "ReLU at {i} (signed {})",
                crate::crypto::falcon::signed(i as u64, LB)
            );
        }
    }

    /// `Π_Maxpool` over the transport must return the true maximum of each 2×2
    /// window under the signed reading — the pooling the CNN benchmarks use.
    ///
    /// Inputs are held to `|a| < L/4` so `max − aᵢ` stays in the signed window,
    /// the bound the function documents.
    #[test]
    #[ignore] // real 3-party transport
    fn transport_maxpool_returns_the_window_max() {
        const LB: u32 = 8;
        let l = 1i128 << LB;
        // Every 4-tuple from a coarse grid inside ±L/4.
        let grid: Vec<i128> = (-(l / 4) + 1..l / 4).step_by(11).collect();
        let mut plain: Vec<[i128; 4]> = Vec::new();
        for &a in &grid {
            for &b in &grid {
                for &c in &grid {
                    for &d in &grid {
                        plain.push([a, b, c, d]);
                    }
                }
            }
        }
        let expected: Vec<i128> = plain.iter().map(|w| *w.iter().max().unwrap()).collect();
        let nwin = plain.len();

        let out = run3(move |me, mbx| {
            let windows: Vec<Vec<PartyShare>> = plain
                .iter()
                .map(|w| {
                    w.iter()
                        .map(|&v| {
                            let l = 1i128 << LB;
                            let e = ((v + l) % l) as u64;
                            let s = [e / 3, e / 3, e - 2 * (e / 3)];
                            PartyShare {
                                left: s[me % 3] as u128,
                                right: s[(me + 1) % 3] as u128,
                            }
                        })
                        .collect()
                })
                .collect();
            let wrap_pre: Vec<Vec<WrapPreproc>> = (0..nwin)
                .map(|i| {
                    (0..3)
                        .map(|k| preproc_seeded((i * 7 + k) as u64, LB, me))
                        .collect()
                })
                .collect();
            let sel_pre: Vec<Vec<SelectPreproc>> = (0..nwin)
                .map(|i| {
                    (0..3)
                        .map(|k| sel_preproc((i * 5 + k) as u64, LB, me))
                        .collect()
                })
                .collect();
            maxpool_batch(&windows, &wrap_pre, &sel_pre, LB, PC_ID_BLOCK, me, mbx)
        });

        let mask = (1u128 << LB) - 1;
        for (i, want) in expected.iter().enumerate() {
            let v = ((out[0][i].left + out[1][i].left + out[2][i].left) & mask) as u64;
            assert_eq!(
                crate::crypto::falcon::signed(v, LB),
                *want,
                "window {i}: max should be {want}"
            );
        }
    }

    /// `preproc` with a caller-chosen seed, for tests that need several
    /// independent bundles per element.
    fn preproc_seeded(seed: u64, l_bits: u32, me: PartyId) -> WrapPreproc {
        let l = 1u64 << l_bits;
        preproc(
            [
                seed.wrapping_mul(53).wrapping_add(7) % l,
                seed.wrapping_mul(29).wrapping_add(3) % l,
                seed.wrapping_mul(17).wrapping_add(5) % l,
            ],
            seed & 1,
            l_bits,
            me,
        )
    }

    /// Exact truncation on the transport must equal a plain arithmetic shift,
    /// on every value — no probability, no off-by-one.
    ///
    /// That is the whole point of the borrow: without it the result is wrong on
    /// roughly half the inputs (measured in
    /// `probabilistic_truncation_is_off_by_one_about_half_the_time`), and the
    /// guard bits an approximate scheme needs are what Falcon spends its smaller
    /// ring on.
    #[test]
    #[ignore] // real 3-party transport
    fn transport_truncation_is_exact() {
        const LB: u32 = 12;
        const D: u32 = 4;
        let l = 1u64 << LB;
        // Keep a + r inside the ring, which the preprocessing controls.
        let vals: Vec<u64> = (0..l / 2).step_by(3).collect();
        let expected: Vec<u64> = vals.iter().map(|&v| v >> D).collect();
        let n = vals.len();

        let out = run3(move |me, mbx| {
            let a: Vec<PartyShare> = vals
                .iter()
                .map(|&v| {
                    let s = [v / 3, v / 3, v - 2 * (v / 3)];
                    PartyShare {
                        left: s[me % 3] as u128,
                        right: s[(me + 1) % 3] as u128,
                    }
                })
                .collect();
            let pre: Vec<TruncPreproc> = (0..n)
                .map(|i| {
                    let l = 1u64 << LB;
                    // Mask small enough that a + r cannot wrap.
                    let r = ((i as u64).wrapping_mul(37) % (l / 4)) + 1;
                    let rs = [r / 3, r / 3, r - 2 * (r / 3)];
                    let sh = r >> D;
                    let ss = [sh, 0, 0];
                    TruncPreproc {
                        r: PartyShare {
                            left: rs[me % 3] as u128,
                            right: rs[(me + 1) % 3] as u128,
                        },
                        r_shifted: PartyShare {
                            left: ss[me % 3] as u128,
                            right: ss[(me + 1) % 3] as u128,
                        },
                        r_lo: {
                            let lo = r & ((1u64 << D) - 1);
                            let ls = [lo, 0, 0];
                            PartyShare {
                                left: ls[me % 3] as u128,
                                right: ls[(me + 1) % 3] as u128,
                            }
                        },
                        sel: sel_preproc(i as u64, LB, me),
                        sel_wrap: sel_preproc(50000 + i as u64, LB, me),
                    }
                })
                .collect();
            truncate_batch(&a, &pre, D, LB, PC_ID_BLOCK, me, mbx)
        });

        let mask = (1u128 << LB) - 1;
        for (i, want) in expected.iter().enumerate() {
            let got = ((out[0][i].left + out[1][i].left + out[2][i].left) & mask) as u64;
            assert_eq!(got, *want, "truncating value {} by {D}", i * 3);
        }
    }

    /// **Exact truncation over securely generated pairs**, which is what unlocks a pure
    /// Falcon ReLU.
    ///
    /// `transport_truncation_is_exact` proves the protocol correct against *dealt*
    /// preprocessing — a locally computed `[r/3, r/3, r−2(r/3)]` split that every party
    /// can reconstruct, so not secret. That was the last dealt piece in the Falcon stack
    /// after `random_wrap_masks` closed `Π_WA`, and it is the reason
    /// `cnn::fused_relu_rescale` routes Falcon's ReLU through `a2b`: without secure
    /// truncation pairs there is no secure Falcon ReLU to route to.
    ///
    /// This runs the same protocol over [`random_trunc_pairs`] and checks exactness over
    /// every value in range. Both tests are kept: one isolates the protocol, this one
    /// says the preprocessing it needs can be generated without a dealer.
    #[test]
    #[ignore] // real 3-party transport
    fn truncation_is_exact_over_securely_generated_pairs() {
        const LB: u32 = 12;
        const D: u32 = 4;
        // `r` spans the **whole** ring, so `a + r` wraps for roughly half the masks and
        // `truncate_batch`'s wrap correction is exercised on most values here rather
        // than never — which is the point: the earlier `RB = LB − 1` kept the sum inside
        // the ring, so this test passed without ever running the wrap path, and it
        // passed just as happily on the version that leaked.
        const RB: u32 = LB;
        let l = 1u64 << LB;
        let vals: Vec<u64> = (0..l / 2).step_by(7).collect();
        let expected: Vec<u64> = vals.iter().map(|&v| v >> D).collect();
        let n = vals.len();

        let out = run3(move |me, mbx| {
            let a: Vec<PartyShare> = vals
                .iter()
                .map(|&v| {
                    let s = [v / 3, v / 3, v - 2 * (v / 3)];
                    PartyShare {
                        left: s[me % 3] as u128,
                        right: s[(me + 1) % 3] as u128,
                    }
                })
                .collect();
            // Offline: the pairs, generated rather than dealt.
            let pre = random_trunc_pairs(n, D, RB, LB, 500_000, me, mbx);
            truncate_batch(&a, &pre, D, LB, PC_ID_BLOCK, me, mbx)
        });

        let mask = (1u128 << LB) - 1;
        let mut wrong = Vec::new();
        for (i, want) in expected.iter().enumerate() {
            let got = ((out[0][i].left + out[1][i].left + out[2][i].left) & mask) as u64;
            if got != *want {
                wrong.push((i * 7, got, *want));
            }
        }
        assert!(
            wrong.is_empty(),
            "truncation must be exact over generated pairs; {} of {n} wrong, first few \
             (value, got, want): {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(5)]
        );
    }

    /// A whole fixed-point layer end to end — `ReLU(truncate(W·x, d))` — against
    /// a plaintext reference computing the same thing.
    ///
    /// This is the reproduction's closing check: matmul, truncation and ReLU
    /// composed on the real transport, with secret weights, and the answer
    /// matching integer arithmetic exactly. Anything that only works in
    /// isolation fails here.
    #[test]
    #[ignore] // real 3-party transport
    fn a_whole_fixed_point_layer_matches_plaintext() {
        const LB: u32 = 16;
        const D: u32 = 3;
        const IN: usize = 6;
        const OUT: usize = 5;
        let l = 1u64 << LB;

        // Small non-negative values so W·x stays well inside the ring and the
        // truncation's `a + r` cannot wrap — the stated preconditions.
        let xv: Vec<u64> = (0..IN as u64).map(|k| k * 3 + 1).collect();
        let wv: Vec<Vec<u64>> = (0..OUT)
            .map(|j| (0..IN).map(|k| ((j * 2 + k) % 5) as u64).collect())
            .collect();
        let expected: Vec<u64> = wv
            .iter()
            .map(|row| {
                let z: u64 = row.iter().zip(&xv).map(|(a, b)| a * b).sum();
                z >> D // all non-negative, so ReLU is the identity here
            })
            .collect();

        let out = run3(move |me, mbx| {
            let share = |v: u64| {
                let s = [v / 3, v / 3, v - 2 * (v / 3)];
                PartyShare {
                    left: s[me % 3] as u128,
                    right: s[(me + 1) % 3] as u128,
                }
            };
            let x: Vec<PartyShare> = xv.iter().map(|&v| share(v)).collect();
            let w: Vec<Vec<PartyShare>> = wv
                .iter()
                .map(|r| r.iter().map(|&v| share(v)).collect())
                .collect();
            let pre = LayerPreproc {
                trunc: (0..OUT)
                    .map(|i| {
                        let r = ((i as u64).wrapping_mul(41) % (l / 8)) + 1;
                        let rs = [r / 3, r / 3, r - 2 * (r / 3)];
                        let sh = r >> D;
                        let lo = r & ((1u64 << D) - 1);
                        TruncPreproc {
                            r: PartyShare {
                                left: rs[me % 3] as u128,
                                right: rs[(me + 1) % 3] as u128,
                            },
                            r_shifted: PartyShare {
                                left: [sh, 0, 0][me % 3] as u128,
                                right: [sh, 0, 0][(me + 1) % 3] as u128,
                            },
                            r_lo: PartyShare {
                                left: [lo, 0, 0][me % 3] as u128,
                                right: [lo, 0, 0][(me + 1) % 3] as u128,
                            },
                            sel: sel_preproc(1000 + i as u64, LB, me),
                            sel_wrap: sel_preproc(51000 + i as u64, LB, me),
                        }
                    })
                    .collect(),
                wrap: (0..OUT)
                    .map(|i| preproc_seeded(2000 + i as u64, LB, me))
                    .collect(),
                sel: (0..OUT)
                    .map(|i| sel_preproc(3000 + i as u64, LB, me))
                    .collect(),
            };
            layer_batch(&x, &w, &pre, D, LB, PC_ID_BLOCK, me, mbx)
        });

        let mask = (1u128 << LB) - 1;
        for (j, want) in expected.iter().enumerate() {
            let got = ((out[0][j].left + out[1][j].left + out[2][j].left) & mask) as u64;
            assert_eq!(got, *want, "layer output {j}");
        }
    }

    /// **The external validation.** Falcon publishes per-protocol complexities;
    /// this reproduction must reproduce them, or it is not a reproduction.
    ///
    /// From the paper's complexity table (semi-honest), with `n` the input
    /// width, `k` the bits of a `Z_p` element:
    ///
    /// | protocol | rounds | comm |
    /// |---|---|---|
    /// | Private Compare | `2 + log₂ ℓ` | `2kn` |
    /// | wrap3 | `3 + log₂ ℓ` | `3kn` |
    /// | ReLU / DReLU | `5 + log₂ ℓ` | `4kn` |
    ///
    /// At ℓ = 32: **7 rounds and `2ℓ = 64` field elements** for `Π_PC`. This
    /// implementation performs 8 rounds sending 65 elements — **one more of
    /// each, and the same one**: the final opening. Falcon's table counts the
    /// interactive multiplication rounds of a protocol whose output is *shares*;
    /// ours additionally reveals `d`, which is one round carrying one element.
    ///
    /// Net of that accounting difference the two agree **exactly**:
    /// `32 (u) + 31 (tree) + 1 (mask) = 64` multiplication elements over
    /// `1 + log₂ ℓ + 1 = 7` rounds. That is a much sharper check than the
    /// whole-network 0.74 MB, because it isolates one protocol from the
    /// online/offline split and from the network architecture.
    #[test]
    fn the_cost_matches_falcons_published_complexity() {
        for l in [8usize, 16, 32, 64] {
            let log_l = (usize::BITS - (l - 1).leading_zeros()) as u64;

            // Falcon: 2 + log₂ ℓ interactive rounds, output left as shares.
            let falcon_rounds = 2 + log_l;
            let ours_multiply_rounds = private_compare_rounds(l) as u64 - 1; // minus the reveal
            assert_eq!(
                ours_multiply_rounds, falcon_rounds,
                "ℓ={l}: multiplication rounds must match Falcon's 2 + log₂ ℓ"
            );

            // Falcon: 2kn bits, i.e. 2ℓ field elements.
            let falcon_elements = 2 * l;
            let ours_multiply_elements = private_compare_elements(l) - 1; // minus the opened value
            assert_eq!(
                ours_multiply_elements, falcon_elements,
                "ℓ={l}: multiplication elements must match Falcon's 2ℓ"
            );
        }
    }

    /// The documented round and element counts must match the batching the code
    /// actually performs.
    #[test]
    fn the_cost_formulas_match_the_batching() {
        // ℓ = 32: 1 (u) + 5 (tree) + 1 (mask) + 1 (open).
        assert_eq!(private_compare_rounds(32), 8);
        assert_eq!(private_compare_elements(32), 32 + 31 + 2);
        // The naive shape would be 2ℓ = 64 rounds; batching is worth 8×.
        assert!(private_compare_rounds(32) * 8 == 64);
        assert_eq!(private_compare_rounds(5), 1 + 3 + 2);
    }
}
