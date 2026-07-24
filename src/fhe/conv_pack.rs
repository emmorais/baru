//! Cheetah-style **convolution packing**: a whole multi-channel
//! convolution output channel from *one* ciphertext multiplication.
//!
//! # Why convolution beats the dense-layer limit
//!
//! For a dense layer, coefficient packing is capped: a degree-`n` product
//! yields at most `n/in_dim` *separable* outputs, because distinct rows need
//! disjoint weight windows (see `bench::hybrid_nn`). Convolution is the
//! opposite case — its windows are *supposed* to overlap, and polynomial
//! multiplication computes exactly that sliding-window sum for free. So one
//! multiplication produces every output pixel at once.
//!
//! # The layout
//!
//! Flatten the input `C×H×W` row-major into coefficients
//! `x[c·HW + r·W + col] = image[c][r][col]`, and pack the kernel **reversed
//! in all three dimensions** at matching strides:
//!
//! ```text
//! w[(C−1−c)·HW + a·W + b] = kernel[c][kh−1−a][kw−1−b]
//! ```
//!
//! Then the product coefficient
//! `P(oy,ox) = (C−1)·HW + (oy+kh−1)·W + (ox+kw−1)` equals
//! `Σ_c Σ_{a,b} kernel[c][a][b] · image[c][oy+a][ox+b]` — the convolution.
//!
//! Two things make this exact rather than approximate:
//! - **Channels pair off automatically.** The reversed channel offset gives
//!   `c + (C−1−c′) = C−1` only when `c = c′`, so every channel's
//!   contribution lands on the *same* coefficient and they sum — which is
//!   precisely what a multi-channel convolution wants.
//! - **No row-wrap contamination.** A term could only leak into `P` from a
//!   column carry (`col + b ≥ W`), which would require `ox ≤ −1`. Impossible
//!   for a valid output, so the read-out coefficients are clean. (Other
//!   coefficients *are* garbage; we simply never read them.)
//!
//! The caller must ensure the product degree stays below the ring dimension
//! ([`ConvShape::required_degree`]) so negacyclic wraparound cannot fold
//! into the output positions.
//!
//! # The win
//!
//! Expressing the same convolution as a Toeplitz matvec costs
//! `⌈out_c·out_h·out_w·(C·H·W) / n⌉` multiplications; this costs `out_c`.
//! For a 28×28 input, 3×3 kernel, 8 output channels at `n = 4096`: **1352
//! multiplications → 8**.

/// Geometry of a valid (no-padding, stride-1) convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    pub in_c: usize,
    pub h: usize,
    pub w: usize,
    pub kh: usize,
    pub kw: usize,
}

impl ConvShape {
    pub fn out_h(&self) -> usize {
        self.h - self.kh + 1
    }
    pub fn out_w(&self) -> usize {
        self.w - self.kw + 1
    }
    fn hw(&self) -> usize {
        self.h * self.w
    }
    /// Number of input coefficients consumed.
    pub fn input_len(&self) -> usize {
        self.in_c * self.hw()
    }
    /// Highest coefficient the product can touch. Must be `< n` so the
    /// negacyclic wraparound cannot contaminate output positions.
    pub fn required_degree(&self) -> usize {
        let max_x = self.in_c * self.hw() - 1;
        let max_w = (self.in_c - 1) * self.hw() + (self.kh - 1) * self.w + (self.kw - 1);
        max_x + max_w
    }
    /// Coefficient holding the convolution output at `(oy, ox)`.
    pub fn output_position(&self, oy: usize, ox: usize) -> usize {
        (self.in_c - 1) * self.hw() + (oy + self.kh - 1) * self.w + (ox + self.kw - 1)
    }
}

/// Pack the input tensor (`[in_c][h][w]`, row-major) into message
/// coefficients mod `t`.
pub fn pack_input(img: &[i64], shape: ConvShape, n: usize, t_mask: u64) -> Vec<u64> {
    assert_eq!(img.len(), shape.input_len(), "input length mismatch");
    assert!(
        shape.required_degree() < n,
        "conv packing needs degree {} < ring dimension {n}; split channels or use a larger ring",
        shape.required_degree()
    );
    let mut m = vec![0u64; n];
    for (i, &v) in img.iter().enumerate() {
        m[i] = (v as u64) & t_mask;
    }
    m
}

/// Pack one output channel's kernel (`[in_c][kh][kw]`, row-major),
/// **reversed in channel, row and column**, at the input's strides.
pub fn pack_kernel(kernel: &[i64], shape: ConvShape, n: usize, t_mask: u64) -> Vec<u64> {
    assert_eq!(kernel.len(), shape.in_c * shape.kh * shape.kw, "kernel length mismatch");
    let mut m = vec![0u64; n];
    for c in 0..shape.in_c {
        for a in 0..shape.kh {
            for b in 0..shape.kw {
                // reversed on every axis
                let kv = kernel[(c * shape.kh + (shape.kh - 1 - a)) * shape.kw + (shape.kw - 1 - b)];
                let idx = (shape.in_c - 1 - c) * shape.hw() + a * shape.w + b;
                m[idx] = (kv as u64) & t_mask;
            }
        }
    }
    m
}

/// Plaintext reference: valid convolution of `img` with one output
/// channel's `kernel`, returned row-major as `out_h × out_w`.
pub fn conv2d_ref(img: &[i64], kernel: &[i64], shape: ConvShape) -> Vec<i64> {
    let (oh, ow) = (shape.out_h(), shape.out_w());
    let mut out = vec![0i64; oh * ow];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = 0i64;
            for c in 0..shape.in_c {
                for a in 0..shape.kh {
                    for b in 0..shape.kw {
                        let k = kernel[(c * shape.kh + a) * shape.kw + b];
                        let x = img[(c * shape.h + (oy + a)) * shape.w + (ox + b)];
                        acc += k * x;
                    }
                }
            }
            out[oy * ow + ox] = acc;
        }
    }
    out
}

/// How many ciphertext multiplications a Toeplitz-matvec formulation of the
/// same convolution would need, for comparison in the benchmarks.
pub fn toeplitz_multiplications(shape: ConvShape, out_c: usize, n: usize) -> usize {
    let in_dim = shape.input_len();
    let rows = out_c * shape.out_h() * shape.out_w();
    let rows_per_chunk = (n / in_dim).saturating_sub(1).max(1);
    rows.div_ceil(rows_per_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fhe::bfv::{Bfv, BfvParams};

    fn pseudo(len: usize, seed: i64) -> Vec<i64> {
        (0..len)
            .map(|i| ((i as i64 * 7 + seed * 13) % 11) - 5)
            .collect()
    }

    /// Pure index-math check: the packed product's read-out coefficients
    /// equal the convolution, computed here with a plain integer negacyclic
    /// product (no encryption involved).
    #[test]
    fn packed_layout_computes_the_convolution() {
        let n = 512;
        for shape in [
            ConvShape { in_c: 1, h: 8, w: 8, kh: 3, kw: 3 },
            ConvShape { in_c: 2, h: 6, w: 7, kh: 3, kw: 2 },
            ConvShape { in_c: 3, h: 5, w: 5, kh: 2, kw: 2 },
        ] {
            let img = pseudo(shape.input_len(), 1);
            let kernel = pseudo(shape.in_c * shape.kh * shape.kw, 2);
            let t_mask = u64::MAX; // exact integers for this layout test
            let x = pack_input(&img, shape, n, t_mask);
            let k = pack_kernel(&kernel, shape, n, t_mask);

            // plain negacyclic product over i128
            let mut prod = vec![0i128; n];
            for (i, &xi) in x.iter().enumerate() {
                if xi == 0 { continue; }
                for (j, &kj) in k.iter().enumerate() {
                    if kj == 0 { continue; }
                    let p = (xi as i64 as i128) * (kj as i64 as i128);
                    let idx = i + j;
                    if idx < n { prod[idx] += p } else { prod[idx - n] -= p }
                }
            }
            let want = conv2d_ref(&img, &kernel, shape);
            for oy in 0..shape.out_h() {
                for ox in 0..shape.out_w() {
                    assert_eq!(
                        prod[shape.output_position(oy, ox)],
                        want[oy * shape.out_w() + ox] as i128,
                        "shape {shape:?} at ({oy},{ox})"
                    );
                }
            }
        }
    }

    /// End to end under BFV: **one** ciphertext×ciphertext multiplication
    /// (no relin) recovers a whole multi-channel output channel.
    #[test]
    fn one_bfv_multiplication_yields_a_whole_output_channel() {
        let s = Bfv::new(BfvParams::TEST_SMALL); // n = 64
        let n = s.params.n;
        let t_mask = s.params.t() - 1;
        let shape = ConvShape { in_c: 1, h: 5, w: 5, kh: 3, kw: 3 };
        assert!(shape.required_degree() < n);
        let img = pseudo(shape.input_len(), 3);
        let kernel = pseudo(shape.kh * shape.kw, 4);

        let (ck, pk) = s.keygen();
        let cx = s.encrypt_poly(&pk, &pack_input(&img, shape, n, t_mask));
        let ck_ct = s.encrypt_poly(&pk, &pack_kernel(&kernel, shape, n, t_mask));
        let prod = s.mul(&cx, &ck_ct); // ONE multiply, degree-2, no relin
        let out = s.decrypt_poly(&ck, &prod);

        let want = conv2d_ref(&img, &kernel, shape);
        for oy in 0..shape.out_h() {
            for ox in 0..shape.out_w() {
                let got = s.to_signed(out[shape.output_position(oy, ox)]);
                assert_eq!(got, want[oy * shape.out_w() + ox], "at ({oy},{ox})");
            }
        }
    }

    #[test]
    fn beats_the_toeplitz_formulation_at_realistic_sizes() {
        // MNIST-shaped: 28×28, 3×3 kernel, 8 output channels, n = 4096.
        let shape = ConvShape { in_c: 1, h: 28, w: 28, kh: 3, kw: 3 };
        let n = 4096;
        assert!(shape.required_degree() < n);
        let toeplitz = toeplitz_multiplications(shape, 8, n);
        let packed = 8; // one multiply per output channel
        assert!(
            toeplitz > 100 * packed,
            "expected a large win: toeplitz={toeplitz}, packed={packed}"
        );
    }
}
