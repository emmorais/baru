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

use crate::crypto::aby3_protocol::{mul_batch_party, reveal_components_packed_party, PartyShare};
use crate::crypto::falcon;
use crate::crypto::field::{Fp, FpShare};
use crate::crypto::types::Fr;
use crate::crypto::fp_protocol::{mul_batch, reveal_batch, share_component};
use crate::crypto::net::{Mailbox, PartyId, Transport};

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
    let mut layer: Vec<FpShare> =
        (0..l).map(|i| add_public(u[i] + suffix[i + 1], Fp::ONE, me)).collect();

    // Rounds 2..: multiply the c[i] together as a balanced tree.
    while layer.len() > 1 {
        let pairs: Vec<(FpShare, FpShare)> =
            layer.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0], c[1])).collect();
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
    assert!(items.iter().all(|(b, _)| b.len() == l), "batch must share one bit width");
    assert_eq!(betas.len(), n, "one blinding bit per item");
    assert_eq!(masks.len(), n, "one mask per item");
    let mut op = base_op_id;

    let signs: Vec<FpShare> =
        betas.iter().map(|b| add_public(-(*b + *b), Fp::ONE, me)).collect();

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
    reveal_batch(&d, op, me, mbx).into_iter().map(|v| !v.is_zero()).collect()
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
    let bprimes =
        private_compare_batch(&pc_items, &betas, &masks, base_op_id + 1, me, mbx);

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
    assert!(y.len() == n && b.len() == n && pre.len() == n, "select_batch length mismatch");
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
    let e: Vec<u64> = comps.iter().map(|c| ((c[0] ^ c[1] ^ c[2]) & 1) as u64).collect();

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
    let prod = mul_batch_party(&pairs, base_op_id + 1, me, mbx);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;
    use std::thread;

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
        joins.into_iter().map(|j| j.join().expect("party panicked")).collect()
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
                let bprime = private_compare(
                    &bits,
                    r,
                    &b,
                    &m,
                    (k as u64 + 1) * PC_ID_BLOCK,
                    me,
                    mbx,
                );
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
                    let bits: Vec<FpShare> =
                        (0..l).map(|i| share_component(Fp::new((x >> i) & 1), 0, me)).collect();
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

        assert_eq!(dm, rounds, "one directed message per party per batched round");
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
                let masks: Vec<FpShare> =
                    (0..n).map(|_| share_component(Fp::ONE, 0, me)).collect();
                private_compare_batch(&items, &betas, &masks, PC_ID_BLOCK, me, &mbx)
            }));
        }
        let outs: Vec<_> = joins.into_iter().map(|j| j.join().expect("panic")).collect();
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
                got, *want,
                "DReLU at value {i} (signed {})",
                crate::crypto::falcon::signed(i as u64, LB)
            );
        }
    }

    /// One party's traffic for a batched `Π_DReLU` of `n` elements at ℓ = 32.
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
                        PartyShare { left: s[me % 3] as u128, right: s[(me + 1) % 3] as u128 }
                    })
                    .collect();
                let pre: Vec<WrapPreproc> = (0..n)
                    .map(|i| {
                        let k = i as u64;
                        preproc(
                            [k.wrapping_mul(53) % l, k.wrapping_mul(29) % l, k.wrapping_mul(17) % l],
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
            c_ring: PartyShare { left: cr[me % 3], right: cr[(me + 1) % 3] },
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
            .map(|&v| if crate::crypto::falcon::signed(v, LB) >= 0 { v } else { 0 })
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
            let sp: Vec<SelectPreproc> =
                (0..a.len()).map(|i| sel_preproc(i as u64, LB, me)).collect();
            relu_batch(&a, &wp, &sp, LB, PC_ID_BLOCK, me, mbx)
        });

        let mask = (1u128 << LB) - 1;
        for (i, want) in expected.iter().enumerate() {
            let got = ((out[0][i].left + out[1][i].left + out[2][i].left) & mask) as u64;
            assert_eq!(
                got, *want,
                "ReLU at {i} (signed {})",
                crate::crypto::falcon::signed(i as u64, LB)
            );
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
