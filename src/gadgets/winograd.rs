//! **Winograd fast convolution over secret shares** — trading interactive
//! multiplications for free local additions.
//!
//! Implements the fast convolution algorithm of CHIP (Wang and Wu, IEEE CSF
//! 2026, ePrint 2026/1399), which is Winograd's minimal filtering algorithm
//! [Winograd 1980; Lavin and Gray, CVPR 2016] applied to homomorphic
//! convolution. We apply it to *secret sharing* instead, and the reason it
//! transfers is that both settings share the asymmetry it exploits:
//! multiplication is expensive and addition is free.
//!
//! # The algorithm
//!
//! The one-dimensional case `F(2,3)` produces two outputs from a four-element
//! window and a three-tap filter using **four** multiplications where the
//! direct method uses six:
//!
//! ```text
//!   m1 = (d0 - d2) * g0
//!   m2 = (d1 + d2) * (g0 + g1 + g2)/2
//!   m3 = (d2 - d1) * (g0 - g1 + g2)/2
//!   m4 = (d1 - d3) * g2
//!
//!   y0 = m1 + m2 + m3
//!   y1 = m2 - m3 - m4
//! ```
//!
//! Nesting it in both dimensions gives `F(2x2, 3x3)`: a `2x2` output tile from
//! a `4x4` input window using **16** multiplications against the direct
//! method's `2*2*9 = 36`, a factor of `2.25`.
//!
//! # Why it survives exact integer arithmetic
//!
//! The halves are the obstacle, and they are only apparent. Look at where each
//! transform acts:
//!
//! | transform | operates on | entries |
//! |---|---|---|
//! | `B^T` (data) | the *input* | `0, ±1` |
//! | `A^T` (output) | the *result* | `0, ±1` |
//! | `G` (filter) | the *weights* | `1, ±1/2` |
//!
//! Only the filter transform is fractional. So we scale it: use `2*g0`,
//! `g0+g1+g2`, `g0-g1+g2` and `2*g2` in place of the four filter terms above.
//! Every `m_i` is then exactly twice the paper's, the one-dimensional output is
//! twice the true convolution, and the two-dimensional output — the transform
//! applied on both axes — is **four times** it.
//!
//! That factor is not a rounding error to be tolerated but an exact power of
//! two, and Baru's fixed-point lowering already emits a rescale after every
//! `fixed x fixed` multiplication. Rescaling by `shift + 2` instead of `shift`
//! absorbs it at no cost. [`WINOGRAD_OUTPUT_SCALE_LOG2`] is that exponent, and
//! [`conv2d_winograd`] returns values carrying it rather than silently
//! dividing, because a division by four over `Z_{2^k}` is a truncation
//! protocol and would cost far more than the multiplications this saves.
//!
//! # What it does and does not buy here
//!
//! It reduces **multiplications**, and in a secret-sharing protocol that means
//! bytes on the wire and local cryptographic work, both of which Baru measures
//! and both of which we found to be real terms. It does **not** reduce
//! *rounds*: [`super::conv2d`] already collects an entire layer into one
//! `mul_batch`, so both routes cost one blocking round and the saving is in the
//! width of that batch, not the number of them.
//!
//! It is also the wrong tool for Baru's *homomorphic* convolution, and we say
//! so rather than leaving the reader to wonder why the FHE path is untouched.
//! [`crate::fhe::conv_pack`] packs a convolution into polynomial coefficients,
//! where the `k^2` taps are computed inside a single polynomial product: a
//! whole layer costs `out_c` multiplications regardless of kernel size. Against
//! that, `36 -> 16` is a step backwards. The trick belongs where multiplication
//! is charged per tap, which for us is secret sharing rather than FHE.
//!
//! # Measured
//!
//! On a `2x10x10 -> 4` layer over the three-party transport, with `3x3` kernels
//! and secret weights: `4608 -> 2048` multiplications and `91844 -> 50884` bytes,
//! at `2` blocking waits either way. The byte figures decompose exactly ---
//! \SI{16}{\byte} per multiplication, which is one `u128` ring element, over
//! \SI{18116}{\byte} of input sharing common to both routes --- so the
//! *marginal* byte ratio is `2.25`, matching the multiplication ratio, while the
//! *total* reads `1.80`. That gap is not measurement noise; it is a common
//! additive term diluting a ratio, the same effect that made Falcon's byte
//! advantage look larger than it was until the evaluator link was counted.
//!
//! # Provenance
//!
//! The trick is CHIP's; the setting is not. CHIP applies it to CKKS with
//! channel-interleaved slot packing and reports `29` rotation keys. Baru has no
//! rotations and no CKKS, and needs neither for this: Winograd is plaintext
//! algebra over the ring, so it transfers to secret sharing unchanged. Nothing in
//! this module depends on approximate arithmetic, which is the reason it could be
//! adopted without reversing the decision to remove CKKS.

use crate::interpreter::backend::{Backend, ShareRef};

use super::Chw;

/// The kernel size this transform is specialised for.
pub const WINOGRAD_KERNEL: usize = 3;
/// Output tile side: `F(2x2, 3x3)`.
pub const WINOGRAD_TILE: usize = 2;
/// Input window side, `tile + kernel - 1`.
pub const WINOGRAD_WINDOW: usize = WINOGRAD_TILE + WINOGRAD_KERNEL - 1;

/// Outputs come back multiplied by `2^this`. See the module note: the filter
/// transform is scaled to clear its halves, once per axis.
pub const WINOGRAD_OUTPUT_SCALE_LOG2: u32 = 2;

/// Multiplications the direct method needs per `2x2` output tile, per input
/// channel: `tile^2 * kernel^2`.
pub const DIRECT_MULS_PER_TILE: usize =
    WINOGRAD_TILE * WINOGRAD_TILE * WINOGRAD_KERNEL * WINOGRAD_KERNEL;
/// Multiplications Winograd needs for the same tile: `window^2`.
pub const WINOGRAD_MULS_PER_TILE: usize = WINOGRAD_WINDOW * WINOGRAD_WINDOW;

/// `B^T d B` — the data transform, on one `4x4` window. Additions only.
///
/// `B^T` is
/// ```text
///   1  0 -1  0
///   0  1  1  0
///   0 -1  1  0
///   0 -1  0  1
/// ```
/// applied to the columns and then (as `B`) to the rows.
fn transform_data(be: &mut dyn Backend, d: &[ShareRef; 16]) -> Vec<ShareRef> {
    let at = |r: usize, c: usize| d[r * 4 + c].clone();
    // Columns first.
    let mut t = Vec::with_capacity(16);
    for c in 0..4 {
        let (r0, r1, r2, r3) = (at(0, c), at(1, c), at(2, c), at(3, c));
        t.push(be.sub(&r0, &r2));
        t.push(be.add(&r1, &r2));
        t.push(be.sub(&r2, &r1));
        t.push(be.sub(&r1, &r3));
    }
    // `t` is column-major at this point: t[c*4 + i]. Now the rows.
    let tt = |i: usize, c: usize| t[c * 4 + i].clone();
    let mut out = Vec::with_capacity(16);
    for i in 0..4 {
        let (c0, c1, c2, c3) = (tt(i, 0), tt(i, 1), tt(i, 2), tt(i, 3));
        out.push(be.sub(&c0, &c2));
        out.push(be.add(&c1, &c2));
        out.push(be.sub(&c2, &c1));
        out.push(be.sub(&c1, &c3));
    }
    out
}

/// `G g G^T` **scaled by two per axis**, so the result is integral.
///
/// The unscaled `G` is `[1,0,0; 1/2,1/2,1/2; 1/2,-1/2,1/2; 0,0,1]`; we use
/// `[2,0,0; 1,1,1; 1,-1,1; 0,0,2]`, which is `2G`. Applied on both axes the
/// filter is therefore `4` times the true one, and so is the output.
fn transform_filter(be: &mut dyn Backend, g: &[ShareRef; 9]) -> Vec<ShareRef> {
    let at = |r: usize, c: usize| g[r * 3 + c].clone();
    let dbl = |be: &mut dyn Backend, x: &ShareRef| be.add(x, x);
    // Columns: 3 -> 4.
    let mut t = Vec::with_capacity(12);
    for c in 0..3 {
        let (r0, r1, r2) = (at(0, c), at(1, c), at(2, c));
        let s = be.add(&r0, &r1);
        let sum = be.add(&s, &r2);
        let d = be.sub(&r0, &r1);
        let alt = be.add(&d, &r2);
        t.push(dbl(be, &r0));
        t.push(sum);
        t.push(alt);
        t.push(dbl(be, &r2));
    }
    // t[c*4 + i]; now rows 3 -> 4.
    let tt = |i: usize, c: usize| t[c * 4 + i].clone();
    let mut out = Vec::with_capacity(16);
    for i in 0..4 {
        let (c0, c1, c2) = (tt(i, 0), tt(i, 1), tt(i, 2));
        let s = be.add(&c0, &c1);
        let sum = be.add(&s, &c2);
        let d = be.sub(&c0, &c1);
        let alt = be.add(&d, &c2);
        out.push(dbl(be, &c0));
        out.push(sum);
        out.push(alt);
        out.push(dbl(be, &c2));
    }
    out
}

/// `A^T m A` — the output transform, `4x4` down to `2x2`. Additions only.
///
/// `A^T` is `[1 1 1 0; 0 1 -1 -1]`.
fn transform_output(be: &mut dyn Backend, m: &[ShareRef]) -> Vec<ShareRef> {
    let at = |r: usize, c: usize| m[r * 4 + c].clone();
    // Columns: 4 -> 2.
    let mut t = Vec::with_capacity(8);
    for c in 0..4 {
        let (r0, r1, r2, r3) = (at(0, c), at(1, c), at(2, c), at(3, c));
        let a = be.add(&r0, &r1);
        let y0 = be.add(&a, &r2);
        let b = be.sub(&r1, &r2);
        let y1 = be.sub(&b, &r3);
        t.push(y0);
        t.push(y1);
    }
    // t[c*2 + i]; rows 4 -> 2.
    let tt = |i: usize, c: usize| t[c * 2 + i].clone();
    let mut out = Vec::with_capacity(4);
    for i in 0..2 {
        let (c0, c1, c2, c3) = (tt(i, 0), tt(i, 1), tt(i, 2), tt(i, 3));
        let a = be.add(&c0, &c1);
        let y0 = be.add(&a, &c2);
        let b = be.sub(&c1, &c2);
        let y1 = be.sub(&b, &c3);
        out.push(y0);
        out.push(y1);
    }
    out
}

/// How many interactive multiplications each route needs for a given layer.
///
/// Reported by [`winograd_cost`] so a caller can decide *before* running, and
/// so a benchmark can check the saving it claims against the saving it gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvMulCount {
    pub direct: usize,
    pub winograd: usize,
}

impl ConvMulCount {
    pub fn ratio(&self) -> f64 {
        self.direct as f64 / self.winograd.max(1) as f64
    }
}

/// Multiplication counts for a `3x3` stride-1 layer of this shape.
///
/// Winograd covers the output in `2x2` tiles, so a layer whose output side is
/// odd pays for a padded tile; the count reflects that rather than assuming the
/// happy case.
pub fn winograd_cost(in_shape: Chw, out_c: usize, out_h: usize, out_w: usize) -> ConvMulCount {
    let tiles = out_h.div_ceil(WINOGRAD_TILE) * out_w.div_ceil(WINOGRAD_TILE);
    ConvMulCount {
        direct: out_c * out_h * out_w * in_shape.c * WINOGRAD_KERNEL * WINOGRAD_KERNEL,
        winograd: out_c * tiles * in_shape.c * WINOGRAD_MULS_PER_TILE,
    }
}

/// Valid `3x3` stride-1 convolution over shares, by the fast algorithm.
///
/// Same signature and same layout as [`super::conv2d`], with two differences a
/// caller must know about:
///
/// - **The output is scaled by `2^`[`WINOGRAD_OUTPUT_SCALE_LOG2`]**, including
///   the bias, which is scaled to match so the sum stays consistent.
/// - It is specialised to `3x3`, stride 1. Anything else is a caller error
///   rather than a silent fallback to the direct method, because a silent
///   fallback would make a benchmark comparing the two report the same numbers
///   twice.
pub fn conv2d_winograd(
    be: &mut dyn Backend,
    input: &[ShareRef],
    in_shape: Chw,
    weights: &[Vec<ShareRef>],
    bias: &[ShareRef],
) -> (Vec<ShareRef>, Chw) {
    assert_eq!(input.len(), in_shape.len());
    let out_c = weights.len();
    let out_h = in_shape.h - WINOGRAD_KERNEL + 1;
    let out_w = in_shape.w - WINOGRAD_KERNEL + 1;
    let out_shape = Chw {
        c: out_c,
        h: out_h,
        w: out_w,
    };
    let tiles_y = out_h.div_ceil(WINOGRAD_TILE);
    let tiles_x = out_w.div_ceil(WINOGRAD_TILE);

    // ── Filter transforms: once per (output channel, input channel) ──
    //
    // Hoisted out of the tile loop deliberately. The transform is linear and
    // depends only on the weights, so recomputing it per tile would add local
    // work proportional to the image while changing nothing — the same
    // hoisting the threshold decryption needed for its squared key.
    let mut gt: Vec<Vec<ShareRef>> = Vec::with_capacity(out_c * in_shape.c);
    for kernel in weights.iter() {
        for ic in 0..in_shape.c {
            let base = ic * WINOGRAD_KERNEL * WINOGRAD_KERNEL;
            let g: [ShareRef; 9] = std::array::from_fn(|j| kernel[base + j].clone());
            gt.push(transform_filter(be, &g));
        }
    }

    // ── Data transforms: once per (input channel, tile) ──
    let zero = be.share_public(0, input[0].nbits);
    let mut dt: Vec<Vec<ShareRef>> = Vec::with_capacity(in_shape.c * tiles_y * tiles_x);
    for ic in 0..in_shape.c {
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let d: [ShareRef; 16] = std::array::from_fn(|j| {
                    let (dy, dx) = (j / WINOGRAD_WINDOW, j % WINOGRAD_WINDOW);
                    let (iy, ix) = (ty * WINOGRAD_TILE + dy, tx * WINOGRAD_TILE + dx);
                    // Tiles at the edge may reach past the input when the output
                    // side is odd; those lanes are zero and their contributions
                    // are discarded when the tile is written back.
                    if iy < in_shape.h && ix < in_shape.w {
                        input[in_shape.idx(ic, iy, ix)].clone()
                    } else {
                        zero.clone()
                    }
                });
                dt.push(transform_data(be, &d));
            }
        }
    }

    // ── The one batched round: every Hadamard product in the layer ──
    let tiles = tiles_y * tiles_x;
    let mut pairs = Vec::with_capacity(out_c * tiles * in_shape.c * WINOGRAD_MULS_PER_TILE);
    for oc in 0..out_c {
        for t in 0..tiles {
            for ic in 0..in_shape.c {
                let g = &gt[oc * in_shape.c + ic];
                let d = &dt[ic * tiles + t];
                for j in 0..WINOGRAD_MULS_PER_TILE {
                    pairs.push((g[j].clone(), d[j].clone()));
                }
            }
        }
    }
    let prods = be.mul_batch(&pairs);

    // ── Accumulate over input channels, inverse-transform, scatter ──
    let scale = 1u128 << WINOGRAD_OUTPUT_SCALE_LOG2;
    let mut out = vec![zero.clone(); out_shape.len()];
    let mut cursor = 0usize;
    for (oc, b) in bias.iter().enumerate().take(out_c) {
        // The bias is added after the inverse transform, so it must carry the
        // same scale the transform imposes on the convolution itself.
        let mut b_scaled = b.clone();
        for _ in 0..WINOGRAD_OUTPUT_SCALE_LOG2 {
            b_scaled = be.add(&b_scaled, &b_scaled);
        }
        let _ = scale;
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut acc: Vec<ShareRef> = Vec::with_capacity(WINOGRAD_MULS_PER_TILE);
                for ic in 0..in_shape.c {
                    for j in 0..WINOGRAD_MULS_PER_TILE {
                        let p = prods[cursor + ic * WINOGRAD_MULS_PER_TILE + j].clone();
                        if ic == 0 {
                            acc.push(p);
                        } else {
                            acc[j] = be.add(&acc[j], &p);
                        }
                    }
                }
                cursor += in_shape.c * WINOGRAD_MULS_PER_TILE;
                let tile = transform_output(be, &acc);
                for (j, v) in tile.iter().enumerate() {
                    let (dy, dx) = (j / WINOGRAD_TILE, j % WINOGRAD_TILE);
                    let (oy, ox) = (ty * WINOGRAD_TILE + dy, tx * WINOGRAD_TILE + dx);
                    if oy < out_h && ox < out_w {
                        out[out_shape.idx(oc, oy, ox)] = be.add(v, &b_scaled);
                    }
                }
            }
        }
    }
    (out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gadgets::{conv2d, conv2d_ref};
    use crate::interpreter::aby3_simulator::Aby3Simulator;

    const NBITS: usize = 32;

    fn residue(v: i64) -> crate::crypto::types::Fr {
        (v as i128).rem_euclid(1i128 << NBITS) as crate::crypto::types::Fr
    }
    fn signed(v: u128) -> i64 {
        let m = 1i128 << NBITS;
        let x = (v as i128) & (m - 1);
        if x >= m / 2 { (x - m) as i64 } else { x as i64 }
    }

    /// Deterministic small tensors, values kept small so the 4x scale cannot overflow.
    fn fixture(c: usize, h: usize, w: usize, oc: usize) -> (Vec<i64>, Vec<Vec<i64>>, Vec<i64>) {
        let img: Vec<i64> = (0..c * h * w).map(|i| ((i * 7) % 11) as i64 - 5).collect();
        let ker: Vec<Vec<i64>> = (0..oc)
            .map(|o| (0..c * 9).map(|j| (((o * 3 + j) % 7) as i64) - 3).collect())
            .collect();
        let bias: Vec<i64> = (0..oc).map(|o| o as i64 - 1).collect();
        (img, ker, bias)
    }

    /// **The fast algorithm computes the convolution exactly, up to its stated scale.**
    ///
    /// Winograd is an algebraic identity, so "approximately" would mean a bug rather than a
    /// tolerance. The only licensed difference from the reference is the factor of four that
    /// clearing the filter transform's halves introduces, and the test divides by it explicitly
    /// rather than comparing ratios — a ratio comparison would pass on an output that was
    /// uniformly wrong.
    ///
    /// Shapes cover both parities of the output side, because an odd side forces a padded tile
    /// whose out-of-range lanes must be discarded and not written back.
    #[test]
    fn the_fast_convolution_is_exact_up_to_its_scale() {
        for &(c, h, w, oc) in &[(1usize, 5usize, 5usize, 1usize), (2, 6, 6, 3), (3, 7, 5, 2), (1, 4, 4, 1)] {
            let (img, ker, bias) = fixture(c, h, w, oc);
            let in_shape = Chw { c, h, w };
            let (want, want_shape) = conv2d_ref(&img, in_shape, &ker, &bias, 3, 3, 1);

            let mut be = Aby3Simulator::new();
            let sin: Vec<_> = img.iter().map(|&v| be.share_public(residue(v), NBITS)).collect();
            let sw: Vec<Vec<_>> = ker
                .iter()
                .map(|k| k.iter().map(|&v| be.share_public(residue(v), NBITS)).collect())
                .collect();
            let sb: Vec<_> = bias.iter().map(|&v| be.share_public(residue(v), NBITS)).collect();

            let (got, got_shape) = conv2d_winograd(&mut be, &sin, in_shape, &sw, &sb);
            assert_eq!(got_shape, want_shape, "shape must match the direct method");

            let scale = 1i64 << WINOGRAD_OUTPUT_SCALE_LOG2;
            for (i, (g, w_)) in got.iter().zip(&want).enumerate() {
                let opened = signed(be.reveal(g, None));
                assert_eq!(
                    opened % scale,
                    0,
                    "output {i} at shape {c}x{h}x{w}->{oc} must be divisible by the stated scale, got {opened}"
                );
                assert_eq!(
                    opened / scale,
                    *w_,
                    "output {i} at shape {c}x{h}x{w}->{oc}: winograd {} vs direct {w_}",
                    opened / scale
                );
            }
        }
    }

    /// **The saving is real and is where the paper says it is: multiplications, not rounds.**
    ///
    /// Both routes issue exactly one `mul_batch`, so a round count cannot distinguish them; the
    /// difference is the *width* of that batch, which is what a secret-sharing protocol charges
    /// bytes and local work for. The assertion is on the ratio rather than on either count, so
    /// the test survives a change of shape.
    #[test]
    fn winograd_cuts_multiplications_and_leaves_rounds_alone() {
        // A layer where the tiling is exact, so the count matches the 2.25x ideal.
        let in_shape = Chw { c: 4, h: 10, w: 10 };
        let (out_h, out_w) = (8, 8);
        let cost = winograd_cost(in_shape, 6, out_h, out_w);
        println!(
            "  4x10x10 -> 6 channels: direct {} muls, winograd {} ({:.2}x)",
            cost.direct,
            cost.winograd,
            cost.ratio()
        );
        assert_eq!(cost.direct, 6 * 8 * 8 * 4 * 9);
        assert_eq!(cost.winograd, 6 * 16 * 4 * 16);
        assert!(
            (cost.ratio() - 2.25).abs() < 1e-9,
            "an exactly tiled layer should hit the 2.25x ideal, got {:.4}",
            cost.ratio()
        );

        // And an odd output side, where a padded tile costs real multiplications. The ratio must
        // degrade rather than be reported as 2.25 anyway.
        let odd = winograd_cost(Chw { c: 1, h: 7, w: 7 }, 1, 5, 5);
        println!("  odd side 5x5: direct {} muls, winograd {} ({:.2}x)", odd.direct, odd.winograd, odd.ratio());
        assert_eq!(odd.winograd, 9 * 16, "3x3 tiles cover a 5x5 output");
        assert!(
            odd.ratio() < 2.25,
            "a padded tile must show up as a worse ratio, got {:.2}",
            odd.ratio()
        );
    }

    /// **What the saving is worth on the wire, measured on the real transport.**
    ///
    /// The multiplication count is an algebraic claim; this is the one that matters to a
    /// deployment. Both routes run the same layer over the same three-party transport, and we
    /// compare bytes and blocking. Bytes should fall by about the multiplication ratio, because
    /// `mul_batch` puts one ring element per multiplication on the wire; blocking should not move
    /// at all, because both routes issue exactly one batched call.
    ///
    /// We assert the *shape* of the result rather than absolute figures: bytes strictly down,
    /// blocking equal. A version of this that asserted a byte ratio would be pinning a
    /// serialization detail rather than the algorithm.
    #[test]
    #[ignore] // three-party transport
    fn the_saving_shows_up_as_bytes_and_not_as_rounds() {
        use crate::bench::comm::NetworkProfile;
        use crate::bench::hybrid_nn::{run_parties, MpcTransport};
        use crate::interpreter::transport_backend::TransportBackend;

        let (c, h, w, oc) = (2usize, 10usize, 10usize, 4usize);
        let (img, ker, bias) = fixture(c, h, w, oc);
        let in_shape = Chw { c, h, w };
        let predicted = winograd_cost(in_shape, oc, h - 2, w - 2);

        let run = |wino: bool| -> (u64, u64, i64) {
            let (img, ker, bias) = (img.clone(), ker.clone(), bias.clone());
            let (outs, comm) = run_parties(NetworkProfile::INSTANT, move |me| {
                let (img, ker, bias) = (img.clone(), ker.clone(), bias.clone());
                move |be: &mut TransportBackend<MpcTransport>| {
                    let sh = |be: &mut TransportBackend<MpcTransport>, v: &[i64]| -> Vec<ShareRef> {
                        let vals: Vec<crate::crypto::types::Fr> = v
                            .iter()
                            .map(|&t| if me == 0 { residue(t) } else { 0 })
                            .collect();
                        be.share_secret_batch(&vals, NBITS, 0)
                    };
                    let sin = sh(be, &img);
                    let sw: Vec<Vec<ShareRef>> = ker.iter().map(|k| sh(be, k)).collect();
                    let sb = sh(be, &bias);
                    let w0 = be.sync_waits();
                    let (out, _) = if wino {
                        conv2d_winograd(be, &sin, in_shape, &sw, &sb)
                    } else {
                        conv2d(be, &sin, in_shape, &sw, &sb, 3, 3, 1)
                    };
                    let waits = be.sync_waits() - w0;
                    (be.reveal(&out[0], None), waits)
                }
            });
            let (v, waits) = outs.into_iter().next().expect("three parties");
            (comm.bytes, waits, signed(v))
        };

        let (db, dw, dv) = run(false);
        let (gb, gw, gv) = run(true);
        let scale = 1i64 << WINOGRAD_OUTPUT_SCALE_LOG2;
        println!("  direct   : {db:8} bytes, {dw} waits, out[0] = {dv}");
        println!("  winograd : {gb:8} bytes, {gw} waits, out[0] = {gv} (= {} x {scale})", gv / scale);
        println!(
            "  predicted {} -> {} muls ({:.2}x); measured bytes {:.2}x",
            predicted.direct,
            predicted.winograd,
            predicted.ratio(),
            db as f64 / gb as f64
        );

        // The total byte ratio understates the saving, and by an amount we can account for
        // exactly rather than wave at. Both routes share their inputs identically before any
        // multiplication happens, so total bytes are `common + 16 * muls` for a `u128` ring
        // element. Solving the two measurements for that common term recovers the per-multiply
        // cost, and it must come out the same for both routes — if it does not, the two are not
        // differing only in multiplication count and the comparison means something else.
        let dm = predicted.direct as f64;
        let gm = predicted.winograd as f64;
        let per_mul = (db as f64 - gb as f64) / (dm - gm);
        let common = db as f64 - per_mul * dm;
        println!(
            "  decomposed: {:.1} bytes/multiply, {:.0} bytes common to both (input sharing)",
            per_mul, common
        );
        println!(
            "  so the MARGINAL byte ratio is {:.2}x, matching the multiplication ratio; the {:.2}x \
             total is diluted by the common term",
            (per_mul * dm) / (per_mul * gm),
            db as f64 / gb as f64
        );
        assert_eq!(gv, dv * scale, "both routes must compute the same convolution");
        assert!(gb < db, "winograd must send fewer bytes: {gb} vs {db}");
        assert!(
            (per_mul - 16.0).abs() < 0.5,
            "a multiplication should put one 16-byte ring element on the wire, got {per_mul:.2} \
             — if this drifts the byte model has changed and the decomposition is unsound"
        );
        assert!(
            common > 0.0 && common < db as f64,
            "the common term must be a real positive share of the traffic, got {common:.0}"
        );
        assert_eq!(
            gw, dw,
            "blocking must be unchanged — both routes issue one batched multiply, and a \
             difference here means one of them stopped batching"
        );
    }
}
