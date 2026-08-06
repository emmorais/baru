//! A small library of **composable MPC gadgets** — the building blocks a
//! Baru program chains to express private neural-network inference of any
//! depth and architecture.
//!
//! # The composition story (see `docs/paper-outline.md`)
//!
//! Each gadget here is a pure function of the `Backend` trait's oblivious
//! primitives (`add`/`sub`/`mul`/`lt`/`share_public`) — nothing more. That
//! matters for the paper's central framing: if each primitive UC-realizes
//! an ideal arithmetic-black-box operation, then every gadget built from
//! them does too (UC composition), and any *network* a user composes from
//! these gadgets in Baru is secure and correct by construction. The gadgets
//! are deliberately backend-agnostic (`&mut dyn Backend`), so the exact
//! same code runs on the in-process ABY3 simulator (fast tests below) and
//! on the real 3-party wire protocol (`TransportBackend`, used by the
//! benchmarks).
//!
//! Every gadget ships with a plaintext reference (`*_ref`) with identical
//! tie-breaking, so encrypted results are checked for *bit-exact* equality,
//! not approximate agreement.
//!
//! # What's here, and why it's enough for CNNs
//!
//! - [`relu`] — the ReLU nonlinearity (elementwise `max(x, 0)`).
//! - [`max2`] / [`max_reduce`] — pairwise and windowed maximum; a
//!   max-pooling layer is [`max_reduce`] applied per window.
//! - [`argmax`] — oblivious index of the maximum, the final decision of a
//!   classifier. This is the escape from the transformer softmax wall: a
//!   CNN *classifier* needs only argmax over logits, never `exp`/division.
//!
//! Together with the `matvec` gadget (`crate::precompile::mpc_matvec`,
//! which also covers convolutions via im2col) these reduce every layer of a
//! standard CNN — conv, ReLU, max-pool, fully-connected, argmax, and
//! (folded-in) batch-norm / residual adds — to primitives already tested to
//! exactness.

use crate::interpreter::backend::{Backend, ShareRef};

/// `mux(cond, a, b) = cond ? a : b`, for a 0/1 bit share `cond`.
/// Implemented as `b + cond·(a − b)` — one multiplication.
fn mux(be: &mut dyn Backend, cond: &ShareRef, a: &ShareRef, b: &ShareRef) -> ShareRef {
    let diff = be.sub(a, b);
    let sel = be.mul(cond, &diff);
    be.add(b, &sel)
}

/// `max(a, b)` on signed shares. `lt(a, b) = 1` iff `a < b`, so the mux
/// keeps `a` on ties (deterministic, matches [`max2_ref`]).
pub fn max2(be: &mut dyn Backend, a: &ShareRef, b: &ShareRef) -> ShareRef {
    let a_lt_b = be.lt(a, b);
    mux(be, &a_lt_b, b, a)
}

pub fn max2_ref(a: i64, b: i64) -> i64 {
    if a < b {
        b
    } else {
        a
    }
}

/// Maximum over a non-empty slice. A max-pooling window is exactly this over
/// the window's elements.
///
/// Reduces as a **tournament** over [`max2_batch`] — `⌈log₂ n⌉` batched
/// levels — rather than the left fold of [`max2`] it used to be, whose
/// comparisons were *dependent* (each waiting on the previous accumulator)
/// and so could not batch. Unlike [`argmax`] the tie-breaking rule does not
/// matter here: only the maximum *value* is returned, and that is the same
/// whichever equal element wins.
pub fn max_reduce(be: &mut dyn Backend, xs: &[ShareRef]) -> ShareRef {
    assert!(!xs.is_empty(), "max_reduce: empty input");
    let mut cur: Vec<ShareRef> = xs.to_vec();
    while cur.len() > 1 {
        let m = cur.len() / 2;
        let a: Vec<ShareRef> = (0..m).map(|k| cur[2 * k].clone()).collect();
        let b: Vec<ShareRef> = (0..m).map(|k| cur[2 * k + 1].clone()).collect();
        let mut next = max2_batch(be, &a, &b);
        if cur.len() % 2 == 1 {
            // Odd element carries to the next level unpaired.
            next.push(cur[cur.len() - 1].clone());
        }
        cur = next;
    }
    cur.pop().expect("tournament leaves one element")
}

pub fn max_reduce_ref(xs: &[i64]) -> i64 {
    *xs.iter().max().expect("max_reduce_ref: empty input")
}

/// Elementwise ReLU: `max(x, 0)` via `x · (0 < x)` (the mux-with-zero
/// special case — no `share_public` needed since the zero is `x − x`).
///
/// **One batched comparison and one batched multiply for the whole slice**,
/// not one of each per element. The elements are independent, so issuing
/// them together costs a constant number of communication rounds instead of
/// `O(len)`. Measured on a 64-wide layer through the real transport, the
/// per-element form sent 5x the messages and took 9x the wall-clock at
/// RTT 20 ms for bit-identical output (`baru bench-abstraction`) — the
/// values are the same, only the round structure differs.
pub fn relu(be: &mut dyn Backend, xs: &[ShareRef]) -> Vec<ShareRef> {
    if xs.is_empty() {
        return Vec::new();
    }
    let zeros: Vec<ShareRef> = xs.iter().map(|x| be.sub(x, x)).collect();
    let pairs: Vec<(ShareRef, ShareRef)> = zeros
        .iter()
        .zip(xs)
        .map(|(z, x)| (z.clone(), x.clone()))
        .collect();
    let pos = be.lt_batch(&pairs); // 1 iff x > 0
    let mul_pairs: Vec<(ShareRef, ShareRef)> = xs
        .iter()
        .zip(&pos)
        .map(|(x, p)| (x.clone(), p.clone()))
        .collect();
    be.mul_batch(&mul_pairs)
}

pub fn relu_ref(xs: &[i64]) -> Vec<i64> {
    xs.iter().map(|&x| x.max(0)).collect()
}

/// Oblivious argmax: returns a **share** of the index of the (first)
/// maximum element, at bit width `nbits`. The caller reveals it to learn
/// the predicted class — nothing else about the logits leaks.
///
/// Reduces as a **tournament**: pair adjacent `(value, index)` candidates,
/// decide every pair of a level with one [`Backend::lt_batch`], and select
/// both the surviving value and its index with one [`Backend::mul_batch`].
/// That is `⌈log₂ n⌉` levels of two batched operations each, against the
/// `n − 1` sequential comparisons a left fold costs.
///
/// The round count is why. A left fold's comparisons are *dependent* — each
/// needs the previous winner — so they cannot batch, and at `n = 10` classes
/// this gadget was the single largest phase of the MPC segment under WAN
/// latency (1.31 s of a 2.03 s segment at RTT 20 ms), despite doing less work
/// than any other. Every other layer of the pipeline had been batched; this
/// one could not be, until the reduction was reassociated into a tree.
/// Measured after: `1309 ms → 546 ms` at RTT 20 ms (2.4×), `127 ms → 60 ms`
/// at zero latency, and the `bench-cnn --img 8 --rtt-ms 20` run as a whole
/// `3309 ms → 2453 ms`.
///
/// **Ties keep the earlier index**, matching [`argmax_ref`]: a pair adopts
/// the right element only on `left < right` (strict), and the bracket pairs
/// adjacent elements in order, so every element of a left subtree has a
/// smaller index than every element of its right sibling. An odd element at
/// the end of a level carries forward unpaired, which preserves that order.
pub fn argmax(be: &mut dyn Backend, xs: &[ShareRef], nbits: usize) -> ShareRef {
    assert!(!xs.is_empty(), "argmax: empty input");
    if xs.len() == 1 {
        return be.share_public(0, nbits);
    }
    let mut vals: Vec<ShareRef> = xs.to_vec();
    let mut idxs: Vec<ShareRef> = (0..xs.len())
        .map(|i| be.share_public(i as u128, nbits))
        .collect();

    while vals.len() > 1 {
        let m = vals.len() / 2; // pairs this level; a trailing odd element carries
        let pairs: Vec<(ShareRef, ShareRef)> = (0..m)
            .map(|k| (vals[2 * k].clone(), vals[2 * k + 1].clone()))
            .collect();
        // adoptₖ = 1 iff left < right, i.e. the right element wins the pair.
        let adopt = be.lt_batch(&pairs);

        // Select value and index off the *same* condition bit. Both muxes go
        // into one `mul_batch`, so a level costs one comparison round plus one
        // multiplication round no matter how wide it is.
        let mut mul_pairs: Vec<(ShareRef, ShareRef)> = Vec::with_capacity(2 * m);
        for (k, cond) in adopt.iter().enumerate() {
            let dv = be.sub(&vals[2 * k + 1], &vals[2 * k]);
            let di = be.sub(&idxs[2 * k + 1], &idxs[2 * k]);
            mul_pairs.push((cond.clone(), dv));
            mul_pairs.push((cond.clone(), di));
        }
        let sels = be.mul_batch(&mul_pairs);

        let mut next_vals = Vec::with_capacity(m + 1);
        let mut next_idxs = Vec::with_capacity(m + 1);
        for k in 0..m {
            next_vals.push(be.add(&vals[2 * k], &sels[2 * k]));
            next_idxs.push(be.add(&idxs[2 * k], &sels[2 * k + 1]));
        }
        if vals.len() % 2 == 1 {
            next_vals.push(vals[vals.len() - 1].clone());
            next_idxs.push(idxs[idxs.len() - 1].clone());
        }
        vals = next_vals;
        idxs = next_idxs;
    }
    idxs.pop()
        .expect("argmax: tournament must leave one candidate")
}

pub fn argmax_ref(xs: &[i64]) -> usize {
    let mut best = 0usize;
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > xs[best] {
            best = i;
        }
    }
    best
}

/// Arithmetic shift-right of a **signed** share by a public `k` bits
/// (division by `2^k`, rounding toward −∞). Needed by SGD weight updates
/// (`W -= grad ≫ lr`), where the gradient is signed — unlike the fused
/// ReLU-rescale, whose logical shift is only valid on the non-negative
/// branch. Decomposes once (`a2b`), keeps bits `k..`, and sign-extends the
/// top `k` positions with the sign bit.
pub fn arith_shr(be: &mut dyn Backend, x: &ShareRef, k: usize, nbits: usize) -> ShareRef {
    if k == 0 {
        return x.clone();
    }
    let bits = be.a2b(x, nbits);
    let sign = bits[nbits - 1].clone();
    let out_bits: Vec<ShareRef> = (0..nbits)
        .map(|j| {
            if j + k < nbits {
                bits[j + k].clone()
            } else {
                sign.clone()
            }
        })
        .collect();
    let mut r = be.b2a(&out_bits);
    // b2a inherits the bit-shares' width (1); restore the value's width so
    // downstream ops (comparisons) decompose the full number.
    r.nbits = nbits;
    r
}

/// [`arith_shr`] over a whole slice in **one** decomposition round.
///
/// The shifts are independent, so `a2b_batch` decomposes every value in the
/// same ~9 rounds one value costs; `b2a` is local. The per-element form issues
/// one `a2b` per value, which is the round-structure defect this repository has
/// now found six times — see `docs/experiments.md` Part I.
pub fn arith_shr_batch(
    be: &mut dyn Backend,
    xs: &[ShareRef],
    k: usize,
    nbits: usize,
) -> Vec<ShareRef> {
    if k == 0 || xs.is_empty() {
        return xs.to_vec();
    }
    let all_bits = be.a2b_batch(xs, nbits);
    all_bits
        .iter()
        .map(|bits| {
            let sign = bits[nbits - 1].clone();
            let out_bits: Vec<ShareRef> = (0..nbits)
                .map(|j| {
                    if j + k < nbits {
                        bits[j + k].clone()
                    } else {
                        sign.clone()
                    }
                })
                .collect();
            let mut r = be.b2a(&out_bits);
            r.nbits = nbits;
            r
        })
        .collect()
}

pub fn arith_shr_ref(x: i64, k: usize) -> i64 {
    x >> k
}

// ─── Linear building blocks ────────────────────────────────────────────────────
//
// These express the *linear* layers (fully-connected, convolution) directly
// from `mul`/`add` for composition/correctness. In the efficient hybrid
// pipeline the linear map is instead evaluated under FHE (one packed
// ciphertext multiply, see `crate::bench::deep_nn`) and only the
// nonlinearities above run in MPC; the share-domain versions here are what a
// pure-MPC composition (or a correctness oracle) uses.

/// Fully-connected layer: `out[o] = Σ_i W[o][i]·x[i] + b[o]` over shares.
pub fn dense(
    be: &mut dyn Backend,
    x: &[ShareRef],
    weights: &[Vec<ShareRef>],
    bias: &[ShareRef],
) -> Vec<ShareRef> {
    if weights.is_empty() {
        return Vec::new();
    }
    // Every product in the layer is independent of every other, so the whole
    // weight matrix is **one** batched round; only the summation is sequential,
    // and addition is local (free). The per-element form this replaced issued
    // `out_dim x in_dim` dependent rounds — on an 8x64 layer, 14x the messages
    // and 45x the wall-clock at RTT 20 ms for identical output
    // (`baru bench-abstraction`).
    //
    // `dot_batch`, not `mul_batch`: the round count is the same either way, but
    // resharing once per **output** instead of once per **product** removes the
    // `in_dim` factor from the bytes. That is what Falcon's `MatMul = k·x·z`
    // means, and on the MedMNIST CNN it is 32x — 239 520 products against 7 474
    // outputs.
    let nbits = x.first().map(|s| s.nbits).unwrap_or(1);
    let dots: Vec<Vec<(ShareRef, ShareRef)>> = weights
        .iter()
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(w, xi)| (w.clone(), xi.clone()))
                .collect()
        })
        .collect();
    let prods = be.dot_batch(&dots, nbits);
    // `prods[r]` is already the whole dot product for row `r`, so only the
    // bias is left to add — and that is local.
    prods.iter().zip(bias).map(|(p, b)| be.add(p, b)).collect()
}

pub fn dense_ref(x: &[i64], weights: &[Vec<i64>], bias: &[i64]) -> Vec<i64> {
    weights
        .iter()
        .zip(bias)
        .map(|(row, &b)| row.iter().zip(x).map(|(&w, &xi)| w * xi).sum::<i64>() + b)
        .collect()
}

pub mod winograd;


/// A CHW tensor described by its dimensions; data is a flat row-major
/// `[channel][row][col]` vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chw {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Chw {
    fn idx(&self, c: usize, y: usize, x: usize) -> usize {
        (c * self.h + y) * self.w + x
    }
    fn len(&self) -> usize {
        self.c * self.h * self.w
    }
}

/// Valid (no-padding) 2-D convolution over shares. `weights` is
/// `[out_c][in_c·kh·kw]` (row-major kernel per output channel), `bias` is
/// `[out_c]`. Returns the flattened output tensor and its shape.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    be: &mut dyn Backend,
    input: &[ShareRef],
    in_shape: Chw,
    weights: &[Vec<ShareRef>],
    bias: &[ShareRef],
    kh: usize,
    kw: usize,
    stride: usize,
) -> (Vec<ShareRef>, Chw) {
    assert_eq!(input.len(), in_shape.len());
    let out_c = weights.len();
    let out_h = (in_shape.h - kh) / stride + 1;
    let out_w = (in_shape.w - kw) / stride + 1;
    let out_shape = Chw {
        c: out_c,
        h: out_h,
        w: out_w,
    };
    // Gather every (weight, input) product the layer needs, multiply them all
    // in **one** batched round, then accumulate locally. The per-element form
    // this replaced cost one communication round per multiply-accumulate,
    // i.e. `out_c x out_h x out_w x in_c x kh x kw` dependent rounds — the
    // same defect `dense` had, and the same fix. Windows overlap, so the same
    // input share appears in many pairs; that is fine, `mul_batch` does not
    // require distinct operands.
    let per_out = in_shape.c * kh * kw;
    let mut pairs = Vec::with_capacity(out_shape.len() * per_out);
    for kernel in weights.iter() {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut k = 0usize;
                for ic in 0..in_shape.c {
                    for dy in 0..kh {
                        for dx in 0..kw {
                            let iy = oy * stride + dy;
                            let ix = ox * stride + dx;
                            pairs
                                .push((kernel[k].clone(), input[in_shape.idx(ic, iy, ix)].clone()));
                            k += 1;
                        }
                    }
                }
            }
        }
    }
    let prods = be.mul_batch(&pairs);
    let mut out = Vec::with_capacity(out_shape.len());
    // `zip` would silently shorten the output if these ever disagreed, so the
    // invariant is asserted rather than absorbed.
    debug_assert_eq!(bias.len(), out_c, "one bias per output channel");
    for (oc, b) in bias.iter().enumerate().take(out_c) {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let base = ((oc * out_h + oy) * out_w + ox) * per_out;
                let mut acc = b.clone();
                for j in 0..per_out {
                    acc = be.add(&acc, &prods[base + j]);
                }
                out.push(acc);
            }
        }
    }
    (out, out_shape)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_ref(
    input: &[i64],
    in_shape: Chw,
    weights: &[Vec<i64>],
    bias: &[i64],
    kh: usize,
    kw: usize,
    stride: usize,
) -> (Vec<i64>, Chw) {
    let out_c = weights.len();
    let out_h = (in_shape.h - kh) / stride + 1;
    let out_w = (in_shape.w - kw) / stride + 1;
    let out_shape = Chw {
        c: out_c,
        h: out_h,
        w: out_w,
    };
    let mut out = Vec::with_capacity(out_shape.len());
    for (oc, kernel) in weights.iter().enumerate() {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = bias[oc];
                let mut k = 0usize;
                for ic in 0..in_shape.c {
                    for dy in 0..kh {
                        for dx in 0..kw {
                            acc += kernel[k]
                                * input[in_shape.idx(ic, oy * stride + dy, ox * stride + dx)];
                            k += 1;
                        }
                    }
                }
                out.push(acc);
            }
        }
    }
    (out, out_shape)
}

/// Max-pooling: the maximum over each non-overlapping (or strided)
/// `pool×pool` window, per channel — [`max_reduce`] over the window.
pub fn maxpool2d(
    be: &mut dyn Backend,
    input: &[ShareRef],
    in_shape: Chw,
    pool: usize,
    stride: usize,
) -> (Vec<ShareRef>, Chw) {
    let out_h = (in_shape.h - pool) / stride + 1;
    let out_w = (in_shape.w - pool) / stride + 1;
    let out_shape = Chw {
        c: in_shape.c,
        h: out_h,
        w: out_w,
    };

    // Gather every window (all the same size), then reduce them in lockstep:
    // one fold step over *all* windows at once. Each step is a single
    // `max2_batch` (one batched comparison + one batched select), so the
    // whole layer costs `pool²−1` batched steps instead of one `max_reduce`
    // per window. On backends without batching this is identical work in the
    // same order; on `TransportBackend` it collapses the round count. The
    // left-fold order matches [`max_reduce`], so ties break the same way.
    let mut windows: Vec<Vec<ShareRef>> = Vec::with_capacity(out_shape.len());
    for c in 0..in_shape.c {
        for oy in 0..out_h {
            for ox in 0..out_w {
                windows.push(
                    (0..pool)
                        .flat_map(|dy| (0..pool).map(move |dx| (dy, dx)))
                        .map(|(dy, dx)| {
                            input[in_shape.idx(c, oy * stride + dy, ox * stride + dx)].clone()
                        })
                        .collect(),
                );
            }
        }
    }
    let wsize = pool * pool;
    let mut acc: Vec<ShareRef> = windows.iter().map(|w| w[0].clone()).collect();
    for k in 1..wsize {
        let next: Vec<ShareRef> = windows.iter().map(|w| w[k].clone()).collect();
        acc = max2_batch(be, &acc, &next);
    }
    (acc, out_shape)
}

/// Elementwise `max(aᵢ, bᵢ)` over two equal-length slices, batching the
/// comparison and the select. Semantics match [`max2`] pairwise (keeps `aᵢ`
/// on ties).
pub fn max2_batch(be: &mut dyn Backend, a: &[ShareRef], b: &[ShareRef]) -> Vec<ShareRef> {
    assert_eq!(a.len(), b.len(), "max2_batch: length mismatch");
    if a.is_empty() {
        return Vec::new();
    }
    let pairs: Vec<(ShareRef, ShareRef)> = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x.clone(), y.clone()))
        .collect();
    let conds = be.lt_batch(&pairs); // condᵢ = 1 iff aᵢ < bᵢ
                                     // muxᵢ(cond, b, a) = aᵢ + condᵢ·(bᵢ − aᵢ): one batched multiply.
    let mul_pairs: Vec<(ShareRef, ShareRef)> = conds
        .iter()
        .zip(a.iter().zip(b))
        .map(|(cond, (x, y))| (cond.clone(), be.sub(y, x)))
        .collect();
    let sels = be.mul_batch(&mul_pairs);
    a.iter().zip(sels).map(|(x, s)| be.add(x, &s)).collect()
}

pub fn maxpool2d_ref(input: &[i64], in_shape: Chw, pool: usize, stride: usize) -> (Vec<i64>, Chw) {
    let out_h = (in_shape.h - pool) / stride + 1;
    let out_w = (in_shape.w - pool) / stride + 1;
    let out_shape = Chw {
        c: in_shape.c,
        h: out_h,
        w: out_w,
    };
    let mut out = Vec::with_capacity(out_shape.len());
    for c in 0..in_shape.c {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let window: Vec<i64> = (0..pool)
                    .flat_map(|dy| (0..pool).map(move |dx| (dy, dx)))
                    .map(|(dy, dx)| input[in_shape.idx(c, oy * stride + dy, ox * stride + dx)])
                    .collect();
                out.push(max_reduce_ref(&window));
            }
        }
    }
    (out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::aby3_simulator::Aby3Simulator;

    const NBITS: usize = 32;

    fn residue(v: i64) -> u128 {
        (v as i128).rem_euclid(1i128 << NBITS) as u128
    }
    fn to_signed(v: u128) -> i64 {
        (v as u32) as i32 as i64
    }
    fn share_vec(be: &mut Aby3Simulator, xs: &[i64]) -> Vec<ShareRef> {
        xs.iter()
            .map(|&v| be.share_secret(residue(v), NBITS, 0))
            .collect()
    }

    #[test]
    fn relu_matches_reference() {
        let cases: &[&[i64]] = &[&[3, -4, 0, 7, -1], &[-5, -6], &[10, 20, 30]];
        for xs in cases {
            let mut be = Aby3Simulator::new();
            let shares = share_vec(&mut be, xs);
            let out = relu(&mut be, &shares);
            let got: Vec<i64> = out.iter().map(|s| to_signed(be.reveal(s, None))).collect();
            assert_eq!(got, relu_ref(xs), "relu {xs:?}");
        }
    }

    #[test]
    fn max2_matches_reference_including_ties_and_signs() {
        for &(a, b) in &[(3i64, 5i64), (5, 3), (4, 4), (-2, -9), (-9, -2), (0, -1)] {
            let mut be = Aby3Simulator::new();
            let sa = be.share_secret(residue(a), NBITS, 0);
            let sb = be.share_secret(residue(b), NBITS, 0);
            let m = max2(&mut be, &sa, &sb);
            assert_eq!(
                to_signed(be.reveal(&m, None)),
                max2_ref(a, b),
                "max2({a},{b})"
            );
        }
    }

    #[test]
    fn max_reduce_matches_reference() {
        let cases: &[&[i64]] = &[&[3, 9, 2, 9, 1], &[-5, -1, -9], &[42], &[7, 7, 7]];
        for xs in cases {
            let mut be = Aby3Simulator::new();
            let shares = share_vec(&mut be, xs);
            let m = max_reduce(&mut be, &shares);
            assert_eq!(
                to_signed(be.reveal(&m, None)),
                max_reduce_ref(xs),
                "max_reduce {xs:?}"
            );
        }
    }

    /// The tournament reduction must break ties to the *first* maximum, like
    /// [`argmax_ref`] and like the left fold it replaced. Exhaustive over
    /// small alphabets, because tie-breaking is exactly the property a tree
    /// reassociation can silently get wrong, and it only shows up when the
    /// same maximum appears in two different subtrees. Odd lengths are
    /// included on purpose: they exercise the unpaired element that carries
    /// forward a level.
    #[test]
    fn argmax_breaks_ties_to_first_over_every_small_input() {
        fn all_vectors(len: usize, alphabet: &[i64], out: &mut Vec<Vec<i64>>) {
            let mut v = vec![alphabet[0]; len];
            loop {
                out.push(v.clone());
                let mut i = 0;
                loop {
                    if i == len {
                        return;
                    }
                    let pos = alphabet.iter().position(|&a| a == v[i]).unwrap();
                    if pos + 1 < alphabet.len() {
                        v[i] = alphabet[pos + 1];
                        break;
                    }
                    v[i] = alphabet[0];
                    i += 1;
                }
            }
        }

        let mut cases: Vec<Vec<i64>> = Vec::new();
        for len in 1..=8 {
            all_vectors(len, &[0, 1], &mut cases);
        }
        for len in 1..=5 {
            all_vectors(len, &[-1, 0, 2], &mut cases);
        }
        for xs in &cases {
            let mut be = Aby3Simulator::new();
            let shares = share_vec(&mut be, xs);
            let idx = argmax(&mut be, &shares, NBITS);
            let got = to_signed(be.reveal(&idx, None)) as usize;
            assert_eq!(got, argmax_ref(xs), "argmax {xs:?}");
        }
    }

    #[test]
    fn argmax_matches_reference_and_breaks_ties_to_first() {
        let cases: &[&[i64]] = &[
            &[3, 9, 2, 9, 1], // first max at index 1 (tie with 3 broken to 1)
            &[-5, -1, -9],
            &[10, 20, 30, 25],
            &[7, 7, 7], // all equal -> index 0
            &[1],
            &[9, 1, 2, 3, 4, 5, 6, 7, 8], // odd length, max first
            &[1, 2, 3, 4, 5, 6, 7, 8, 9], // odd length, max last
            &[5, 1, 5, 1, 5],             // ties across sibling subtrees
        ];
        for xs in cases {
            let mut be = Aby3Simulator::new();
            let shares = share_vec(&mut be, xs);
            let idx = argmax(&mut be, &shares, NBITS);
            let got = to_signed(be.reveal(&idx, None)) as usize;
            assert_eq!(got, argmax_ref(xs), "argmax {xs:?}");
        }
    }

    #[test]
    fn arith_shr_matches_reference_signed() {
        for &(x, k) in &[
            (8i64, 1usize),
            (-8, 1),
            (7, 2),
            (-7, 2),
            (-1, 3),
            (100, 4),
            (-100, 4),
            (0, 5),
        ] {
            let mut be = Aby3Simulator::new();
            let s = be.share_secret(residue(x), NBITS, 0);
            let r = arith_shr(&mut be, &s, k, NBITS);
            assert_eq!(
                to_signed(be.reveal(&r, None)),
                arith_shr_ref(x, k),
                "arith_shr({x},{k})"
            );
        }
    }

    #[test]
    fn arith_shr_result_is_usable_in_a_comparison() {
        // The b2a-nbits fix: the shifted value must be comparable afterwards
        // (regression for the CNN maxpool bug).
        let mut be = Aby3Simulator::new();
        let a = be.share_secret(residue(40), NBITS, 0);
        let b = be.share_secret(residue(9), NBITS, 0);
        let sa = arith_shr(&mut be, &a, 2, NBITS); // 10
        let m = max2(&mut be, &sa, &b); // max(10, 9) = 10
        assert_eq!(to_signed(be.reveal(&m, None)), 10);
    }

    #[test]
    fn dense_matches_reference() {
        let x = [3i64, -4, 5];
        let w = vec![vec![1i64, 2, 3], vec![-1, 0, 2]];
        let b = [1i64, -2];
        let mut be = Aby3Simulator::new();
        let sx = share_vec(&mut be, &x);
        let sw: Vec<Vec<ShareRef>> = w.iter().map(|r| share_vec(&mut be, r)).collect();
        let sb = share_vec(&mut be, &b);
        let out = dense(&mut be, &sx, &sw, &sb);
        let got: Vec<i64> = out.iter().map(|s| to_signed(be.reveal(s, None))).collect();
        assert_eq!(got, dense_ref(&x, &w, &b));
    }

    #[test]
    fn conv2d_and_maxpool2d_match_reference() {
        // 1 channel, 5×5 input; 2 output channels, 3×3 kernel, stride 1
        // → 2×3×3; then 2×2 maxpool stride 1 → 2×2×2.
        let in_shape = Chw { c: 1, h: 5, w: 5 };
        let input: Vec<i64> = (0..25).map(|i| (i as i64 % 7) - 3).collect();
        let weights = vec![
            (0..9).map(|i| (i as i64 % 3) - 1).collect::<Vec<_>>(),
            (0..9).map(|i| ((i * 2) as i64 % 5) - 2).collect::<Vec<_>>(),
        ];
        let bias = [1i64, -1];

        let (ref_conv, ref_shape) = conv2d_ref(&input, in_shape, &weights, &bias, 3, 3, 1);
        let (ref_pool, _) = maxpool2d_ref(&ref_conv, ref_shape, 2, 1);

        let mut be = Aby3Simulator::new();
        let sin = share_vec(&mut be, &input);
        let sw: Vec<Vec<ShareRef>> = weights.iter().map(|r| share_vec(&mut be, r)).collect();
        let sb = share_vec(&mut be, &bias);
        let (conv, conv_shape) = conv2d(&mut be, &sin, in_shape, &sw, &sb, 3, 3, 1);
        let conv_got: Vec<i64> = conv.iter().map(|s| to_signed(be.reveal(s, None))).collect();
        assert_eq!(conv_got, ref_conv, "conv2d");
        let (pool, _) = maxpool2d(&mut be, &conv, conv_shape, 2, 1);
        let pool_got: Vec<i64> = pool.iter().map(|s| to_signed(be.reveal(s, None))).collect();
        assert_eq!(pool_got, ref_pool, "maxpool2d");
    }

    #[test]
    fn full_cnn_classifier_composes_end_to_end() {
        // A complete LeNet-shaped forward pass, entirely from the gadget
        // library: conv → relu → maxpool → flatten → dense → relu → dense →
        // argmax. The encrypted classification must equal the plaintext one,
        // and only the class index is ever revealed.
        let in_shape = Chw { c: 1, h: 8, w: 8 };
        let input: Vec<i64> = (0..64).map(|i| (i as i64 * 3 % 11) - 5).collect();
        // conv: 2 out channels, 3×3, stride 1 → 2×6×6
        let cw = vec![
            (0..9).map(|i| (i as i64 % 3) - 1).collect::<Vec<_>>(),
            (0..9).map(|i| ((i as i64 + 1) % 3) - 1).collect::<Vec<_>>(),
        ];
        let cb = [1i64, 0];
        // after relu + 2×2 maxpool stride 2 → 2×3×3 = 18 features
        // dense 18→6, relu, dense 6→3, argmax
        let d1w: Vec<Vec<i64>> = (0..6)
            .map(|o| (0..18).map(|i| ((o * 18 + i) as i64 % 5) - 2).collect())
            .collect();
        let d1b: Vec<i64> = (0..6).map(|o| (o as i64 % 3) - 1).collect();
        let d2w: Vec<Vec<i64>> = (0..3)
            .map(|o| (0..6).map(|i| ((o * 6 + i) as i64 % 5) - 2).collect())
            .collect();
        let d2b: Vec<i64> = (0..3).map(|o| (o as i64 % 3) - 1).collect();

        // Plaintext reference.
        let expected = {
            let (c, cs) = conv2d_ref(&input, in_shape, &cw, &cb, 3, 3, 1);
            let r = relu_ref(&c);
            let (p, _) = maxpool2d_ref(&r, cs, 2, 2);
            let h = relu_ref(&dense_ref(&p, &d1w, &d1b));
            let logits = dense_ref(&h, &d2w, &d2b);
            argmax_ref(&logits)
        };

        // Encrypted composition.
        let mut be = Aby3Simulator::new();
        let sin = share_vec(&mut be, &input);
        let scw: Vec<Vec<ShareRef>> = cw.iter().map(|r| share_vec(&mut be, r)).collect();
        let scb = share_vec(&mut be, &cb);
        let (conv, conv_shape) = conv2d(&mut be, &sin, in_shape, &scw, &scb, 3, 3, 1);
        let relu1 = relu(&mut be, &conv);
        let (pool, _) = maxpool2d(&mut be, &relu1, conv_shape, 2, 2);
        let sd1w: Vec<Vec<ShareRef>> = d1w.iter().map(|r| share_vec(&mut be, r)).collect();
        let sd1b = share_vec(&mut be, &d1b);
        let d1 = dense(&mut be, &pool, &sd1w, &sd1b);
        let h = relu(&mut be, &d1);
        let sd2w: Vec<Vec<ShareRef>> = d2w.iter().map(|r| share_vec(&mut be, r)).collect();
        let sd2b = share_vec(&mut be, &d2b);
        let logits = dense(&mut be, &h, &sd2w, &sd2b);
        let idx = argmax(&mut be, &logits, NBITS);
        let got = to_signed(be.reveal(&idx, None)) as usize;

        assert_eq!(got, expected, "full CNN classifier composition");
    }
}
