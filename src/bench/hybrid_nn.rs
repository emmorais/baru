//! `baru bench-hybrid-nn` — one private dense layer (matvec + bias + ReLU),
//! computed end-to-end under five strategies, timed per phase and verified
//! against the plaintext reference:
//!
//!   mpc          everything under ABY3 shares (weights shared too — the
//!                fully-private setting, matching `bench-nn`'s MPC scope)
//!   tfhe         everything under TFHE: mul_plain linear + real PBS
//!                comparison/mux ReLU (exact, but every op is seconds)
//!   ckks         everything under CKKS — which forces the *deep*
//!                `CkksCompareBackend` context for the whole layer, because
//!                the polynomial-approximation ReLU needs real rescaling
//!                levels (the shallow SIMD context can't do ReLU at all)
//!   hybrid-tfhe  TFHE linear → fhe->share → ABY3 ReLU → share->fhe
//!   hybrid-ckks  shallow-CKKS SIMD linear (dot_plain, one ciphertext for
//!                the whole input) → fhe->share → ABY3 ReLU → share->fhe
//!   hybrid-ntru  SI-NTRU (LTV) packed linear: weights AND input encrypted,
//!                the whole matvec is ONE ct×ct multiplication decrypted
//!                under f² (no relinearization/bootstrapping), the whole
//!                output vector switched to MPC in ONE masked decryption
//!                → ABY3 ReLU → share->fhe (see `fhe::si_ntru`)
//!
//! The hybrids are the Gazelle pattern made concrete with `crate::hybrid`'s
//! conversions: run the linear phase where it's cheap (FHE, batched), and
//! the comparison where *it* is cheap (MPC, ~ms), never exposing the value.
//!
//! Scope notes, stated so the numbers are read fairly:
//! - FHE paths use public weights (`mul_plain`/`dot_plain` — the standard
//!   "public model, private input" deployment); the MPC path shares its
//!   weights (fully private). This *disadvantages* MPC in the comparison
//!   and it still wins the ReLU phase — the asymmetry is real, not an
//!   artifact of scope.
//! - `ckks` is *approximate*: its `ok` column checks agreement with the
//!   plain-f64 Newton-iteration reference at the same iteration count
//!   (`relu_approx_plain`), and the deviation from the *exact* ReLU is
//!   reported separately — at practical iteration counts and realistic
//!   pre-activation bounds that deviation is large (the documented
//!   slow-near-zero regime), which is itself one of the findings.
//! - Conversions here are in-process simulations of the mask-and-open /
//!   encrypt-shares protocols (see `crate::hybrid` for the trust model).

use std::time::Instant;

use serde::Serialize;

use crate::bench::Phase;
use crate::fhe::ckks_compare::{relu_approx_plain, CkksCompareBackend};
use crate::fhe::si_ntru::{ntru_poly_to_mpc_shares, SiNtru, SiNtruAdapter, SiNtruParams};
use crate::fhe::{CkksBackend, FheBackend, MockFheBackend, RealFheBackend, TfheBackend};
use crate::hybrid::{fhe_to_mpc_share, mpc_share_to_fhe, FheAdapter, IntFhe, RealFhe};
use crate::interpreter::aby3_backend::Aby3Backend;
use crate::interpreter::backend::{Backend, ShareRef};

const NBITS: usize = 32;

// ─── Data & plaintext reference ───────────────────────────────────────────────

/// Deterministic pseudo-values in roughly [-50, 50] (both signs, so ReLU is
/// genuinely exercised) — same convention as `bench::nn::gen_val`.
fn gen_val(seed: i64, i: usize) -> i64 {
    ((i as i64)
        .wrapping_mul(7)
        .wrapping_add(seed.wrapping_mul(13))
        % 101)
        - 50
}

fn gen_vec(n: usize, seed: i64) -> Vec<i64> {
    (0..n).map(|i| gen_val(seed, i)).collect()
}

pub struct LayerData {
    pub x: Vec<i64>,
    pub w: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
    /// Pre-activation reference (matvec + bias, before ReLU).
    pub pre: Vec<i64>,
    /// Exact reference output (after ReLU).
    pub expected: Vec<i64>,
}

pub fn make_layer(in_dim: usize, out_dim: usize) -> LayerData {
    let x = gen_vec(in_dim, 1);
    let w: Vec<Vec<i64>> = (0..out_dim)
        .map(|r| gen_vec(in_dim, 100 + r as i64))
        .collect();
    let bias = gen_vec(out_dim, 7);
    let pre: Vec<i64> = w
        .iter()
        .zip(&bias)
        .map(|(row, b)| row.iter().zip(&x).map(|(wi, xi)| wi * xi).sum::<i64>() + b)
        .collect();
    let expected = pre.iter().map(|&v| v.max(0)).collect();
    LayerData {
        x,
        w,
        bias,
        pre,
        expected,
    }
}

fn residue(v: i64) -> u128 {
    (v as i128).rem_euclid(1i128 << NBITS) as u128
}

/// Low-32-bit two's-complement decode (shares/converted ciphertexts carry
/// residues mod 2^32, possibly with share-sum carries above).
fn to_signed(v: u64) -> i64 {
    (v as u32) as i32 as i64
}

// ─── Reporting ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct StrategyReport {
    pub strategy: String,
    pub phases: Vec<Phase>,
    pub total_ms: f64,
    pub output: Vec<i64>,
    /// Exact match against the reference (for `ckks`: against the plain
    /// Newton-approximation reference at the same iteration count).
    pub ok: bool,
    /// Worst |output − exact ReLU| — nonzero only for the approximate
    /// `ckks` strategy (its polynomial ReLU error, quantified).
    pub max_abs_error_vs_exact: i64,
    pub notes: Vec<String>,
}

struct Timer {
    phases: Vec<Phase>,
    start: Instant,
}

impl Timer {
    fn new() -> Self {
        Timer {
            phases: Vec::new(),
            start: Instant::now(),
        }
    }
    fn phase<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let t0 = Instant::now();
        let out = f();
        self.phases.push(Phase {
            name,
            micros: t0.elapsed().as_micros(),
        });
        out
    }
    fn total_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

// ─── Shared MPC pieces ────────────────────────────────────────────────────────

/// Mux-style ReLU on shares (`bench-nn`'s Layer pattern).
fn mpc_relu(be: &mut Aby3Backend, z: &ShareRef) -> ShareRef {
    let zero = be.sub(z, z);
    let mask = be.lt(&zero, z);
    be.mul(z, &mask)
}

// ─── Strategies ───────────────────────────────────────────────────────────────

fn run_mpc_only(data: &LayerData) -> StrategyReport {
    let mut t = Timer::new();
    let mut be = Aby3Backend::new();

    let (xs, ws, bs) = t.phase("share", || {
        let xs: Vec<ShareRef> = data
            .x
            .iter()
            .map(|&v| be.share_secret(residue(v), NBITS, 0))
            .collect();
        let ws: Vec<Vec<ShareRef>> = data
            .w
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| be.share_secret(residue(v), NBITS, 0))
                    .collect()
            })
            .collect();
        let bs: Vec<ShareRef> = data
            .bias
            .iter()
            .map(|&v| be.share_secret(residue(v), NBITS, 0))
            .collect();
        (xs, ws, bs)
    });

    let pre = t.phase("linear", || {
        ws.iter()
            .zip(&bs)
            .map(|(row, b)| {
                let mut acc = be.mul(&row[0], &xs[0]);
                for (wi, xi) in row[1..].iter().zip(&xs[1..]) {
                    let p = be.mul(wi, xi);
                    acc = be.add(&acc, &p);
                }
                be.add(&acc, b)
            })
            .collect::<Vec<_>>()
    });

    let relu = t.phase("relu", || {
        pre.iter().map(|z| mpc_relu(&mut be, z)).collect::<Vec<_>>()
    });

    let output = t.phase("reveal", || {
        relu.iter()
            .map(|s| to_signed(be.reveal(s, None) as u64))
            .collect::<Vec<i64>>()
    });

    StrategyReport {
        strategy: "mpc".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        notes: vec!["ABY3, weights AND input private (shared)".into()],
    }
}

/// Shared TFHE linear phase: encrypt input, mul_plain/add per row, add bias.
fn tfhe_linear(
    t: &mut Timer,
    f: &TfheBackend,
    ck: &[u8],
    data: &LayerData,
) -> Vec<<TfheBackend as FheBackend>::Ct> {
    let cts = t.phase("encrypt", || {
        data.x
            .iter()
            .map(|&v| f.encrypt(ck, v as u64))
            .collect::<Vec<_>>()
    });
    t.phase("linear", || {
        data.w
            .iter()
            .zip(&data.bias)
            .map(|(row, &b)| {
                let mut acc = f.mul_plain(&cts[0], row[0]);
                for (ct, &wi) in cts[1..].iter().zip(&row[1..]) {
                    acc = f.add(&acc, &f.mul_plain(ct, wi));
                }
                f.add_plain(&acc, b)
            })
            .collect::<Vec<_>>()
    })
}

fn run_tfhe_only(data: &LayerData) -> StrategyReport {
    let mut t = Timer::new();
    let f = TfheBackend::new();
    let ck = t.phase("keygen", || {
        let (ck, sk) = f.keygen();
        TfheBackend::install_server_key(&sk);
        ck
    });

    let pre = tfhe_linear(&mut t, &f, &ck, data);

    let relu = t.phase("relu", || {
        pre.iter()
            .map(|ct| {
                let zero = f.encrypt_trivial(0);
                let mask = f.cmpslt(&zero, ct); // PBS comparison: 0 < pre
                f.mux(&mask, ct, &zero) // PBS mux
            })
            .collect::<Vec<_>>()
    });

    let output = t.phase("decrypt", || {
        relu.iter()
            .map(|ct| f.decrypt(&ck, ct) as i64)
            .collect::<Vec<i64>>()
    });

    StrategyReport {
        strategy: "tfhe".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        notes: vec!["public weights; exact PBS ReLU".into()],
    }
}

fn run_hybrid_tfhe(data: &LayerData) -> StrategyReport {
    let mut t = Timer::new();
    let f = TfheBackend::new();
    let ck = t.phase("keygen", || {
        let (ck, sk) = f.keygen();
        TfheBackend::install_server_key(&sk);
        ck
    });

    let pre = tfhe_linear(&mut t, &f, &ck, data);

    let mut be = Aby3Backend::new();
    let adapter = IntFhe(&f);
    let shares = t.phase("to-mpc", || {
        pre.iter()
            .map(|ct| fhe_to_mpc_share(&adapter, &ck, ct, &mut be, NBITS))
            .collect::<Vec<_>>()
    });
    let relu = t.phase("relu", || {
        shares
            .iter()
            .map(|s| mpc_relu(&mut be, s))
            .collect::<Vec<_>>()
    });
    let back = t.phase("to-fhe", || {
        relu.iter()
            .map(|s| mpc_share_to_fhe(&mut be, s, &adapter, &ck))
            .collect::<Vec<_>>()
    });

    let output = t.phase("decrypt", || {
        back.iter()
            .map(|ct| to_signed(adapter.decrypt(&ck, ct)))
            .collect::<Vec<i64>>()
    });

    StrategyReport {
        strategy: "hybrid-tfhe".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        notes: vec!["TFHE linear + ABY3 ReLU via scheme switch".into()],
    }
}

fn run_ckks_only(data: &LayerData, iterations: u32) -> StrategyReport {
    let mut t = Timer::new();
    // The whole layer must live in the deep context: linear costs 1 level
    // (one mul_plain), then the polynomial ReLU needs its full budget.
    let levels = 1 + CkksCompareBackend::levels_for_relu(iterations);

    let (ck, be, degree) = t.phase("keygen", || {
        let client = CkksCompareBackend::new(levels);
        let (ck, sk) = client.keygen();
        let degree = client.degree();
        let be = CkksCompareBackend::with_degree_and_server_keys(
            degree.try_into().expect("degree fits"),
            levels,
            &sk,
        );
        (ck, be, degree)
    });

    let cts = t.phase("encrypt", || {
        data.x
            .iter()
            .map(|&v| be.encrypt(&ck, v as f64))
            .collect::<Vec<_>>()
    });

    let pre = t.phase("linear", || {
        data.w
            .iter()
            .zip(&data.bias)
            .map(|(row, &b)| {
                let mut acc = be.mul_plain(&cts[0], row[0] as f64);
                for (ct, &wi) in cts[1..].iter().zip(&row[1..]) {
                    acc = be.add(&acc, &be.mul_plain(ct, wi as f64));
                }
                be.add_plain(&acc, b as f64)
            })
            .collect::<Vec<_>>()
    });

    // Activation-range calibration: a real deployment measures this on
    // sample data (what `ml::QuantizationReport` exists for); here the
    // plaintext reference plays that role.
    let bound = data
        .pre
        .iter()
        .map(|v| v.abs())
        .max()
        .unwrap_or(1)
        .max(1) as f64
        * 1.25;

    let relu = t.phase("relu", || {
        pre.iter()
            .map(|ct| be.relu_approx(ct, bound, iterations))
            .collect::<Vec<_>>()
    });

    let output = t.phase("decrypt", || {
        relu.iter()
            .map(|ct| be.decrypt(&ck, ct).round() as i64)
            .collect::<Vec<i64>>()
    });

    // "Working as expected" for an approximate scheme = matches the plain
    // f64 Newton reference at the same iteration count (±1 for CKKS noise
    // and rounding); the gap to the exact ReLU is the polynomial's own,
    // separately-reported approximation error.
    let approx_reference: Vec<i64> = data
        .pre
        .iter()
        .map(|&v| relu_approx_plain(v as f64, bound, iterations).round() as i64)
        .collect();
    let ok = output
        .iter()
        .zip(&approx_reference)
        .all(|(got, want)| (got - want).abs() <= 1);

    StrategyReport {
        strategy: "ckks".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        notes: vec![
            format!("deep context: degree={degree}, {levels} levels"),
            format!("Newton-sign ReLU: {iterations} iterations, bound={bound:.0}"),
            "ok = matches plain approx reference; err col = vs exact ReLU".into(),
        ],
    }
}

fn run_hybrid_ckks(data: &LayerData) -> StrategyReport {
    let mut t = Timer::new();
    let (ck, be) = t.phase("keygen", || {
        let client = CkksBackend::new();
        let (ck, sk) = client.keygen();
        (ck, CkksBackend::with_server_keys(&sk))
    });

    // The whole input vector is ONE SIMD ciphertext.
    let x_ct = t.phase("encrypt", || {
        let x_f: Vec<f64> = data.x.iter().map(|&v| v as f64).collect();
        be.encrypt_vec(&ck, &x_f)
    });

    let pre = t.phase("linear", || {
        data.w
            .iter()
            .zip(&data.bias)
            .map(|(row, &b)| {
                let row_f: Vec<f64> = row.iter().map(|&v| v as f64).collect();
                let dot = be.dot_plain(&x_ct, &row_f);
                be.add_plain(&dot, b as f64)
            })
            .collect::<Vec<_>>()
    });

    let mut mpc = Aby3Backend::new();
    let adapter = RealFhe(&be);
    let shares = t.phase("to-mpc", || {
        pre.iter()
            .map(|ct| fhe_to_mpc_share(&adapter, &ck, ct, &mut mpc, NBITS))
            .collect::<Vec<_>>()
    });
    let relu = t.phase("relu", || {
        shares
            .iter()
            .map(|s| mpc_relu(&mut mpc, s))
            .collect::<Vec<_>>()
    });
    let back = t.phase("to-fhe", || {
        relu.iter()
            .map(|s| mpc_share_to_fhe(&mut mpc, s, &adapter, &ck))
            .collect::<Vec<_>>()
    });

    let output = t.phase("decrypt", || {
        back.iter()
            .map(|ct| to_signed(adapter.decrypt(&ck, ct)))
            .collect::<Vec<i64>>()
    });

    StrategyReport {
        strategy: "hybrid-ckks".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        notes: vec![
            "shallow SIMD context; whole input = 1 ciphertext (dot_plain)".into(),
            "exact ABY3 ReLU via scheme switch — no deep context needed".into(),
        ],
    }
}

fn run_hybrid_ntru(data: &LayerData) -> StrategyReport {
    let mut t = Timer::new();
    let s = SiNtru::new(SiNtruParams::DEFAULT);
    let nbits = s.params.log2_t as usize;

    let ck = t.phase("keygen", || {
        let (ck, _sk) = s.keygen();
        ck
    });

    // Both the input AND the weight matrix are encrypted (fully private
    // linear phase, like the MPC strategy — stronger than the public-weight
    // CKKS/TFHE rows).
    let (cx, cw) = t.phase("encrypt", || {
        let cx = s.encrypt_poly(&ck, &s.pack_input(&data.x));
        let cw = s.encrypt_poly(&ck, &s.pack_weights(&data.w));
        (cx, cw)
    });

    // The ENTIRE matvec: one ciphertext×ciphertext scale-invariant
    // multiplication (row r's inner product lands at coefficient
    // r·in_dim + in_dim − 1; decrypts under f² — no relinearization).
    // Bias folds in as a plaintext Δ·b addition at each output position.
    let in_dim = data.x.len();
    let pre = t.phase("linear", || {
        let mut prod = s.mul(&cx, &cw);
        for (r, &b) in data.bias.iter().enumerate() {
            let pos = s.matvec_output_position(in_dim, r);
            let residue = (b as i128).rem_euclid(1i128 << nbits) as u64;
            prod = s.add_plain_at(&prod, pos, residue);
        }
        prod
    });

    // ONE polynomial-level mask-and-open converts every output coefficient
    // to MPC shares simultaneously.
    let positions: Vec<usize> = (0..data.w.len())
        .map(|r| s.matvec_output_position(in_dim, r))
        .collect();
    let mut be = Aby3Backend::new();
    let shares = t.phase("to-mpc", || {
        ntru_poly_to_mpc_shares(&s, &ck, &pre, &positions, 3, &mut be)
    });

    let relu = t.phase("relu", || {
        shares
            .iter()
            .map(|z| mpc_relu(&mut be, z))
            .collect::<Vec<_>>()
    });

    let adapter = SiNtruAdapter(&s);
    let back = t.phase("to-fhe", || {
        relu.iter()
            .map(|z| mpc_share_to_fhe(&mut be, z, &adapter, &ck))
            .collect::<Vec<_>>()
    });

    let output = t.phase("decrypt", || {
        back.iter()
            .map(|ct| s.to_signed(adapter.decrypt(&ck, ct) & (s.params.t() - 1)))
            .collect::<Vec<i64>>()
    });

    StrategyReport {
        strategy: "hybrid-ntru".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        notes: vec![
            format!(
                "SI-NTRU (LTV): n={}, q=2^{}, t=2^{} — see module docs for the \
                 overstretched-regime security caveat",
                s.params.n, s.params.log2_q, s.params.log2_t
            ),
            "weights AND input encrypted; whole matvec = ONE ct×ct mult, decrypted under f²"
                .into(),
            "whole output vector switched to MPC in ONE masked decryption".into(),
        ],
    }
}

/// Harness-validation strategy: same pipeline as the hybrids, but on the
/// plaintext mock FHE backend — instant, exercises every conversion.
fn run_hybrid_mock(data: &LayerData) -> StrategyReport {
    let mut t = Timer::new();
    let f = MockFheBackend;
    let (ck, _) = f.keygen();
    let cts = t.phase("encrypt", || {
        data.x
            .iter()
            .map(|&v| f.encrypt(&ck, v as u64))
            .collect::<Vec<_>>()
    });
    let pre = t.phase("linear", || {
        data.w
            .iter()
            .zip(&data.bias)
            .map(|(row, &b)| {
                let mut acc = f.mul_plain(&cts[0], row[0]);
                for (ct, &wi) in cts[1..].iter().zip(&row[1..]) {
                    acc = f.add(&acc, &f.mul_plain(ct, wi));
                }
                f.add_plain(&acc, b)
            })
            .collect::<Vec<_>>()
    });
    let mut be = Aby3Backend::new();
    let adapter = IntFhe(&f);
    let shares = t.phase("to-mpc", || {
        pre.iter()
            .map(|ct| fhe_to_mpc_share(&adapter, &ck, ct, &mut be, NBITS))
            .collect::<Vec<_>>()
    });
    let relu = t.phase("relu", || {
        shares
            .iter()
            .map(|s| mpc_relu(&mut be, s))
            .collect::<Vec<_>>()
    });
    let back = t.phase("to-fhe", || {
        relu.iter()
            .map(|s| mpc_share_to_fhe(&mut be, s, &adapter, &ck))
            .collect::<Vec<_>>()
    });
    let output = t.phase("decrypt", || {
        back.iter()
            .map(|ct| to_signed(adapter.decrypt(&ck, ct)))
            .collect::<Vec<i64>>()
    });
    StrategyReport {
        strategy: "hybrid-mock".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        notes: vec!["plaintext mock FHE — harness validation only".into()],
    }
}

fn max_err(got: &[i64], expected: &[i64]) -> i64 {
    got.iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs())
        .max()
        .unwrap_or(0)
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Mpc,
    Tfhe,
    Ckks,
    HybridTfhe,
    HybridCkks,
    HybridNtru,
    HybridMock,
}

impl Strategy {
    fn parse(s: &str) -> Result<Strategy, String> {
        match s {
            "mpc" => Ok(Strategy::Mpc),
            "tfhe" => Ok(Strategy::Tfhe),
            "ckks" => Ok(Strategy::Ckks),
            "hybrid-tfhe" => Ok(Strategy::HybridTfhe),
            "hybrid-ckks" => Ok(Strategy::HybridCkks),
            "hybrid-ntru" => Ok(Strategy::HybridNtru),
            "hybrid-mock" => Ok(Strategy::HybridMock),
            other => Err(format!(
                "unknown strategy '{other}' (expected mpc, tfhe, ckks, hybrid-tfhe, \
                 hybrid-ckks, hybrid-ntru, hybrid-mock)"
            )),
        }
    }
}

pub fn run_strategy(s: Strategy, data: &LayerData, relu_iterations: u32) -> StrategyReport {
    match s {
        Strategy::Mpc => run_mpc_only(data),
        Strategy::Tfhe => run_tfhe_only(data),
        Strategy::Ckks => run_ckks_only(data, relu_iterations),
        Strategy::HybridTfhe => run_hybrid_tfhe(data),
        Strategy::HybridCkks => run_hybrid_ckks(data),
        Strategy::HybridNtru => run_hybrid_ntru(data),
        Strategy::HybridMock => run_hybrid_mock(data),
    }
}

const USAGE: &str = "usage: baru bench-hybrid-nn [options]

One private dense layer (matvec + bias + ReLU) computed end-to-end under
five strategies — pure ABY3 MPC, pure TFHE, pure CKKS (deep context,
polynomial ReLU), and the two Gazelle-style hybrids that switch to MPC
shares for the ReLU — each timed per phase and verified against the
plaintext reference.

options:
  --in N               input dimension (default 8)
  --out M              output dimension (default 2)
  --strategies LIST    comma-separated subset (default: all five real ones)
  --relu-iterations K  Newton iterations for the ckks strategy (default 2)
  --json               emit reports as JSON
";

fn fmt_ms(micros: u128) -> String {
    let ms = micros as f64 / 1000.0;
    if ms >= 10_000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{ms:.1}ms")
    } else {
        format!("{micros}µs")
    }
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut in_dim = 8usize;
    let mut out_dim = 2usize;
    let mut relu_iterations = 2u32;
    let mut json = false;
    let mut strategies = vec![
        Strategy::Mpc,
        Strategy::HybridNtru,
        Strategy::HybridCkks,
        Strategy::HybridTfhe,
        Strategy::Ckks,
        Strategy::Tfhe,
    ];

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--in" => {
                i += 1;
                in_dim = args
                    .get(i)
                    .ok_or("--in needs a value")?
                    .parse()
                    .map_err(|_| "--in needs a positive integer".to_string())?;
            }
            "--out" => {
                i += 1;
                out_dim = args
                    .get(i)
                    .ok_or("--out needs a value")?
                    .parse()
                    .map_err(|_| "--out needs a positive integer".to_string())?;
            }
            "--relu-iterations" => {
                i += 1;
                relu_iterations = args
                    .get(i)
                    .ok_or("--relu-iterations needs a value")?
                    .parse()
                    .map_err(|_| "--relu-iterations needs a non-negative integer".to_string())?;
            }
            "--strategies" => {
                i += 1;
                let list = args.get(i).ok_or("--strategies needs a value")?;
                strategies = list
                    .split(',')
                    .map(Strategy::parse)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    if strategies.contains(&Strategy::Tfhe) || strategies.contains(&Strategy::HybridTfhe) {
        eprintln!(
            "warning: TFHE strategies cost seconds per multiply-accumulate — use --release \
             and small --in/--out."
        );
    }
    if strategies.contains(&Strategy::Ckks) {
        eprintln!(
            "warning: the pure-ckks strategy keygens a deep (degree-32768) context: \
             expect tens of seconds before the layer even starts."
        );
    }

    let data = make_layer(in_dim, out_dim);
    if !json {
        println!("layer({in_dim}x{out_dim})  reference output: {:?}", data.expected);
        println!("                 (pre-activation:   {:?})\n", data.pre);
    }

    let mut reports = Vec::new();
    for &s in &strategies {
        let r = run_strategy(s, &data, relu_iterations);
        if !json {
            let phases: Vec<String> = r
                .phases
                .iter()
                .map(|p| format!("{}={}", p.name, fmt_ms(p.micros)))
                .collect();
            println!(
                "{:<12} total={:<10} {}  {}",
                r.strategy,
                fmt_ms((r.total_ms * 1000.0) as u128),
                if r.ok { "✓" } else { "✗" },
                phases.join(" ")
            );
            println!("             output: {:?}", r.output);
            if r.max_abs_error_vs_exact != 0 {
                println!(
                    "             max |err| vs exact ReLU: {}",
                    r.max_abs_error_vs_exact
                );
            }
            for n in &r.notes {
                println!("             note: {n}");
            }
            println!();
        }
        reports.push(r);
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpc_only_matches_plaintext() {
        let data = make_layer(6, 3);
        let r = run_mpc_only(&data);
        assert!(r.ok, "mpc output {:?} != expected {:?}", r.output, data.expected);
        assert_eq!(r.max_abs_error_vs_exact, 0);
    }

    #[test]
    fn hybrid_mock_pipeline_matches_plaintext() {
        // Validates the full FHE-linear -> to-mpc -> relu -> to-fhe pipeline
        // without real-crypto cost.
        let data = make_layer(6, 3);
        let r = run_hybrid_mock(&data);
        assert!(
            r.ok,
            "hybrid-mock output {:?} != expected {:?}",
            r.output, data.expected
        );
    }

    #[test]
    #[ignore] // slow in debug (n=2048 schoolbook ring); fast in release
    fn hybrid_ntru_matches_plaintext() {
        let data = make_layer(8, 2);
        let r = run_hybrid_ntru(&data);
        assert!(
            r.ok,
            "hybrid-ntru output {:?} != expected {:?}",
            r.output, data.expected
        );
        assert_eq!(r.max_abs_error_vs_exact, 0, "hybrid must be exact");
    }

    #[test]
    #[ignore] // slow: real SEAL keygen (~seconds)
    fn hybrid_ckks_matches_plaintext() {
        let data = make_layer(8, 2);
        let r = run_hybrid_ckks(&data);
        assert!(
            r.ok,
            "hybrid-ckks output {:?} != expected {:?}",
            r.output, data.expected
        );
        assert_eq!(r.max_abs_error_vs_exact, 0, "hybrid must be exact");
    }
}
