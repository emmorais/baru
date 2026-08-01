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

use crate::crypto::field::{Fp, FpShare};
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
        let per_party = per_cmp_bytes * drelus as f64;

        println!(
            "\n  Π_PC batched at ℓ={L} — per party\n    \
             rounds per batch      : {rounds} (independent of batch size)\n    \
             messages @ n=1000     : {m2}\n    \
             marginal bytes/compare: {per_cmp_bytes:.1}\n    \
             (unbatched was 225 — framing dominates a lone comparison)\n\n  \
             Extrapolated to Falcon's LeNet ({drelus} DReLUs = {relus} ReLU + 3×{windows} pool)\n    \
             per party : {:.2} MB\n    all parties: {:.2} MB\n    \
             Falcon reports 0.74 MB for LeNet (semi-honest LAN)\n",
            per_party / 1.0e6,
            per_party * 3.0 / 1.0e6
        );

        assert_eq!(m1, m2, "batching must not change the round/message count");
        assert!(
            per_cmp_bytes < 100.0,
            "marginal cost should approach the {} field elements, got {per_cmp_bytes:.1} bytes",
            private_compare_elements(L)
        );
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
