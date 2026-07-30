//! `baru bench-hybrid-nn` — one private dense layer (matvec + bias + ReLU),
//! computed end-to-end under six strategies, timed per phase and verified
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
//!   hybrid-bfv   BFV packed linear: weights AND input encrypted, the whole
//!                matvec is ONE ct×ct multiplication per weight chunk
//!                decrypted under s² (no relinearization/bootstrapping), the
//!                whole output vector switched to MPC in ONE masked
//!                decryption per chunk → ABY3 ReLU → share->fhe
//!                (see `fhe::bfv`)
//!
//! # The MPC is the real protocol, not a simulator
//!
//! Every MPC phase here — the pure-MPC strategy, the hybrids' ReLU, and
//! both directions of the scheme switch — runs the genuine replicated
//! 3-party ABY3 **wire protocol** (`TransportBackend` over `InprocHandle`,
//! three party threads exchanging real serialized messages), NOT the
//! in-process `Aby3Simulator` simulator (whose algorithms differ from the
//! distributed protocol — e.g. its `Share3` multiplication reconstructs
//! and reshares locally, which costs nothing and reveals everything).
//! A [`NetworkProfile`] (`--rtt-ms`, `--bandwidth-mbps`) injects one-way
//! latency + a bytes/bandwidth term into every protocol message, so the
//! *same computation as the real protocol* is measured under whatever
//! network you posit — parties in one rack (0 ms) or on different
//! continents (100 ms+) — without deploying anything. Round-trip counts,
//! not per-op CPU, then dominate MPC cost, which is exactly the effect an
//! in-process simulator hides.
//!
//! The FHE side runs single-threaded (it IS single-party — that's FHE's
//! deployment model), but its WAN legs are charged too: a `link-io` phase
//! models uploading input ciphertexts / downloading results (1 RTT +
//! bytes/bandwidth), and each scheme switch pays its real round trips
//! (mask ciphertexts to the evaluator + opened value back = 1 RTT for
//! FHE→MPC; component ciphertexts in = ½ RTT for MPC→FHE) with the real
//! serialized ciphertext sizes.
//!
//! Scope notes, stated so the numbers are read fairly:
//! - FHE paths use public weights (`mul_plain`/`dot_plain`) EXCEPT
//!   hybrid-bfv, which encrypts them; the MPC path shares its weights
//!   (fully private).
//! - `ckks` is *approximate*: its `ok` column checks agreement with the
//!   plain-f64 Newton reference at the same iteration count; deviation
//!   from the exact ReLU is reported separately.
//! - Conversions simulate the mask-and-open protocol's data flow with a
//!   single decryption key standing in for threshold decryption (see
//!   `crate::hybrid` for the trust model).

use std::time::Instant;

use rand::Rng;
use serde::Serialize;

use crate::bench::comm::{CommStats, CountingTransport, LatencyTransport, NetworkProfile};
use crate::bench::Phase;
use crate::crypto::net::InprocHandle;
use crate::crypto::types::Fr;
use crate::fhe::ckks_compare::{relu_approx_plain, CkksCompareBackend};
use crate::fhe::bfv::{Bfv, BfvAdapter, BfvCt, BfvParams};
// `Backend` is already taken by the MPC interpreter's trait, below.
use crate::fhe::poly::{self, Backend as PolyBackend};
use crate::fhe::{CkksBackend, FheBackend, MockFheBackend, RealFheBackend, TfheBackend};
use crate::hybrid::{shares_to_fhe, FheAdapter, IntFhe, RealFhe};
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

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

fn residue(v: i64, nbits: usize) -> u128 {
    (v as i128).rem_euclid(1i128 << nbits) as u128
}

/// Two's-complement decode of the low `nbits` bits.
fn to_signed_n(v: u64, nbits: usize) -> i64 {
    let m = 1u64 << nbits;
    let v = v & (m - 1);
    if v >= m / 2 {
        (v as i64) - (m as i64)
    } else {
        v as i64
    }
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
    /// Real wire traffic of the MPC segment (party-0 perspective), when
    /// the strategy has one.
    pub comm: Option<CommStats>,
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

// ─── The real-protocol MPC session harness ────────────────────────────────────

pub type MpcTransport = CountingTransport<LatencyTransport<InprocHandle>>;

/// Run one closure per party, SPMD, on 3 threads over the real wire
/// protocol with `profile`'s latency/bandwidth on every message. Returns
/// per-party results (index = party id) and party-0's wire statistics.
pub fn run_parties<R, F>(profile: NetworkProfile, mk: impl Fn(usize) -> F) -> (Vec<R>, CommStats)
where
    R: Send + 'static,
    F: FnOnce(&mut TransportBackend<MpcTransport>) -> R + Send + 'static,
{
    let handles = InprocHandle::create(3);
    let mut joins = Vec::with_capacity(3);
    let mut comm = None;
    for (me, h) in handles.into_iter().enumerate() {
        let lat = LatencyTransport::new(h, profile);
        let counted = CountingTransport::new(lat, 3);
        if me == 0 {
            comm = Some(counted.handle());
        }
        let f = mk(me);
        joins.push(std::thread::spawn(move || {
            let mut be = TransportBackend::new(me, counted);
            f(&mut be)
        }));
    }
    let results: Vec<R> = joins
        .into_iter()
        .map(|j| j.join().expect("MPC party thread panicked"))
        .collect();
    (results, comm.unwrap().stats())
}

/// Model one leg of evaluator↔parties WAN traffic that does NOT go through
/// the party-to-party transport (FHE ciphertext upload/download and the
/// scheme-switch round trips): `half_rtts` one-way latencies plus the
/// bytes/bandwidth term, slept for real so it lands in phase timings.
pub fn sleep_link(profile: &NetworkProfile, bytes: usize, half_rtts: u32) {
    let bw = profile
        .bandwidth_bytes_per_sec
        .map(|bw| std::time::Duration::from_secs_f64(bytes as f64 / bw))
        .unwrap_or(std::time::Duration::ZERO);
    let d = profile.one_way_latency * half_rtts + bw;
    if !d.is_zero() {
        std::thread::sleep(d);
    }
}

/// Mux-style ReLU on shares (`bench-nn`'s Layer pattern), over whatever
/// backend — for this benchmark, always the real transport.
/// Import an FHE→MPC masked opening into shares, **batched per owner**.
///
/// Each of the three parties contributes one component of the masked value, so
/// the import is three sharings — of the whole vector, not of each element.
/// The per-element form costs `3 * len` interactive calls for identical output;
/// `share_secret_batch` has existed on `TransportBackend` all along (see
/// `bench-cnn`, which uses exactly this shape).
fn import_batched(
    be: &mut TransportBackend<MpcTransport>,
    my_comps: &[Fr],
    me: usize,
    nbits: usize,
) -> Vec<ShareRef> {
    let count = my_comps.len();
    let mut acc: Option<Vec<ShareRef>> = None;
    for owner in 0..3 {
        let vals: Vec<Fr> = if me == owner { my_comps.to_vec() } else { vec![0; count] };
        let shs = be.share_secret_batch(&vals, nbits, owner);
        acc = Some(match acc {
            // Folding the three owners is local addition — free.
            None => shs,
            Some(a) => a.iter().zip(&shs).map(|(x, y)| be.add(x, y)).collect(),
        });
    }
    acc.unwrap_or_default()
}

pub fn mpc_relu(be: &mut dyn Backend, z: &ShareRef) -> ShareRef {
    let zero = be.sub(z, z);
    let mask = be.lt(&zero, z);
    be.mul(z, &mask)
}

// ─── Strategies ───────────────────────────────────────────────────────────────

fn run_mpc_only(data: &LayerData, profile: NetworkProfile) -> StrategyReport {
    let start = Instant::now();
    let x = data.x.clone();
    let w = data.w.clone();
    let bias = data.bias.clone();

    let (mut per_party, comm) = run_parties(profile, |me| {
        let x = x.clone();
        let w = w.clone();
        let bias = bias.clone();
        move |be: &mut TransportBackend<MpcTransport>| {
            let mut t = Timer::new();
            let share_one = |be: &mut TransportBackend<MpcTransport>, v: i64| -> ShareRef {
                let fr = if me == 0 { residue(v, NBITS) } else { 0 };
                be.share_secret(fr, NBITS, 0)
            };
            let (xs, ws, bs) = t.phase("share", || {
                let xs: Vec<ShareRef> = x.iter().map(|&v| share_one(be, v)).collect();
                let ws: Vec<Vec<ShareRef>> = w
                    .iter()
                    .map(|row| row.iter().map(|&v| share_one(be, v)).collect())
                    .collect();
                let bs: Vec<ShareRef> = bias.iter().map(|&v| share_one(be, v)).collect();
                (xs, ws, bs)
            });
            let pre = t.phase("linear", || {
                // Champion precompile (batched Beaver triples — the best
                // known MPC matvec in this repo), on the real wire.
                let registry = crate::precompile::mpc_matvec::registry();
                let mv = registry.champion().matvec(be, &ws, &xs);
                mv.iter()
                    .zip(&bs)
                    .map(|(o, b)| be.add(o, b))
                    .collect::<Vec<_>>()
            });
            // Batched: one `lt_batch` + one `mul_batch` for the whole layer.
            // The elements are independent, so the per-element form below
            // ([`mpc_relu`]) costs O(out_dim) rounds for identical output.
            let relu = t.phase("relu", || crate::gadgets::relu(be, &pre));
            let out = t.phase("reveal", || {
                relu.iter()
                    .map(|s| to_signed_n(be.reveal(s, None) as u64, NBITS))
                    .collect::<Vec<i64>>()
            });
            (out, t.phases)
        }
    });

    let (output, phases) = per_party.swap_remove(0);
    StrategyReport {
        strategy: "mpc".into(),
        phases,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        comm: Some(comm),
        notes: vec![
            "real ABY3 wire protocol (3 threads), weights AND input private".into(),
            "matvec via champion precompile (batched)".into(),
        ],
    }
}

/// The hybrids' MPC segment, generic over the FHE side: mask-and-open the
/// pre-activation ciphertexts (1 RTT + mask-ciphertext bytes), run
/// import + ReLU + per-party component export on the real 3-party wire
/// protocol, re-encrypt the components (½ RTT + bytes). Returns the output
/// ciphertexts and the MPC segment's wire stats.
fn hybrid_mpc_relu<A: FheAdapter>(
    t: &mut Timer,
    fhe: &A,
    ck: &[u8],
    pre: &[A::Ct],
    nbits: usize,
    profile: NetworkProfile,
) -> (Vec<A::Ct>, CommStats) {
    let k = pre.len();
    let vmask: u64 = if nbits >= 64 {
        u64::MAX
    } else {
        (1u64 << nbits) - 1
    };

    let mut rng = rand::thread_rng();
    let masks: Vec<Vec<u64>> = (0..3)
        .map(|_| (0..k).map(|_| rng.gen::<u64>() & vmask).collect())
        .collect();

    let opened: Vec<u64> = t.phase("to-mpc", || {
        let mut conv_bytes = 0usize;
        let y: Vec<u64> = pre
            .iter()
            .enumerate()
            .map(|(j, ct)| {
                let mut masked = ct.clone();
                for m in &masks {
                    let enc = fhe.encrypt(ck, m[j]);
                    conv_bytes += fhe.ct_bytes(&enc);
                    masked = fhe.add(&masked, &enc);
                }
                fhe.decrypt(ck, &masked) & vmask
            })
            .collect();
        // Parties send Enc(rᵢ) to the evaluator; the opened y comes back:
        // one full round trip.
        sleep_link(&profile, conv_bytes + k * 8, 2);
        y
    });

    let (per_party, comm) = t.phase("mpc-relu", || {
        run_parties(profile, |me| {
            let my_masks = masks[me].clone();
            let opened = opened.clone();
            move |be: &mut TransportBackend<MpcTransport>| {
                // Import: each party shares ITS component (y−r₀ for party
                // 0, −rᵢ otherwise) via the real input-sharing protocol,
                // then the three shares are added locally.
                // Import, batched per owner: 3 rounds for the whole vector
                // instead of 3 per element. Folding the three owners is local.
                let my_comps: Vec<Fr> = (0..opened.len())
                    .map(|j| {
                        if me == 0 {
                            (opened[j] as u128).wrapping_sub(my_masks[j] as u128)
                        } else {
                            0u128.wrapping_sub(my_masks[j] as u128)
                        }
                    })
                    .collect();
                let shares: Vec<ShareRef> = import_batched(be, &my_comps, me, nbits);
                // Batched over the whole output vector: one comparison round
                // and one multiply round, not one of each per element.
                let relu: Vec<ShareRef> = crate::gadgets::relu(be, &shares);
                // Export: this party's canonical additive component —
                // local, zero communication.
                relu.iter()
                    .map(|s| be.my_additive_component(s))
                    .collect::<Vec<Fr>>()
            }
        })
    });

    let out = t.phase("to-fhe", || {
        let mut bytes = 0usize;
        let cts: Vec<A::Ct> = (0..k)
            .map(|j| {
                let comps: Vec<Fr> = (0..3).map(|p| per_party[p][j]).collect();
                let ct = shares_to_fhe(fhe, ck, &comps, nbits);
                bytes += 3 * fhe.ct_bytes(&ct);
                ct
            })
            .collect();
        // Parties send Enc(component) to the evaluator: half a round trip.
        sleep_link(&profile, bytes, 1);
        cts
    });
    (out, comm)
}

/// Shared TFHE linear phase: encrypt input, mul_plain/add per row, add bias.
fn tfhe_linear(
    t: &mut Timer,
    f: &TfheBackend,
    ck: &[u8],
    data: &LayerData,
    profile: &NetworkProfile,
) -> Vec<<TfheBackend as FheBackend>::Ct> {
    let cts = t.phase("encrypt", || {
        data.x
            .iter()
            .map(|&v| f.encrypt(ck, v as u64))
            .collect::<Vec<_>>()
    });
    t.phase("link-io", || {
        let ct_size = f.serialize_ct(&cts[0]).len();
        sleep_link(profile, ct_size * cts.len(), 2);
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

fn run_tfhe_only(data: &LayerData, profile: NetworkProfile) -> StrategyReport {
    let mut t = Timer::new();
    let f = TfheBackend::new();
    let ck = t.phase("keygen", || {
        let (ck, sk) = f.keygen();
        TfheBackend::install_server_key(&sk);
        ck
    });

    let pre = tfhe_linear(&mut t, &f, &ck, data, &profile);

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
        comm: None,
        notes: vec!["public weights; exact PBS ReLU".into()],
    }
}

fn run_hybrid_tfhe(data: &LayerData, profile: NetworkProfile) -> StrategyReport {
    let mut t = Timer::new();
    let f = TfheBackend::new();
    let ck = t.phase("keygen", || {
        let (ck, sk) = f.keygen();
        TfheBackend::install_server_key(&sk);
        ck
    });

    let pre = tfhe_linear(&mut t, &f, &ck, data, &profile);
    let adapter = IntFhe(&f);
    let (back, comm) = hybrid_mpc_relu(&mut t, &adapter, &ck, &pre, NBITS, profile);

    let output = t.phase("decrypt", || {
        back.iter()
            .map(|ct| to_signed_n(adapter.decrypt(&ck, ct), NBITS))
            .collect::<Vec<i64>>()
    });

    StrategyReport {
        strategy: "hybrid-tfhe".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        comm: Some(comm),
        notes: vec!["TFHE linear + real-protocol ABY3 ReLU via scheme switch".into()],
    }
}

fn run_ckks_only(data: &LayerData, iterations: u32, profile: NetworkProfile) -> StrategyReport {
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

    t.phase("link-io", || {
        let ct_size = be.serialize_ct(&cts[0]).len();
        sleep_link(&profile, ct_size * cts.len(), 2);
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
        comm: None,
        notes: vec![
            format!("deep context: degree={degree}, {levels} levels"),
            format!("Newton-sign ReLU: {iterations} iterations, bound={bound:.0}"),
            "ok = matches plain approx reference; err col = vs exact ReLU".into(),
        ],
    }
}

fn run_hybrid_ckks(data: &LayerData, profile: NetworkProfile) -> StrategyReport {
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

    t.phase("link-io", || {
        let ct_size = be.serialize_ct(&x_ct).len();
        sleep_link(&profile, ct_size, 2);
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

    let adapter = RealFhe(&be);
    let (back, comm) = hybrid_mpc_relu(&mut t, &adapter, &ck, &pre, NBITS, profile);

    let output = t.phase("decrypt", || {
        back.iter()
            .map(|ct| to_signed_n(adapter.decrypt(&ck, ct), NBITS))
            .collect::<Vec<i64>>()
    });

    StrategyReport {
        strategy: "hybrid-ckks".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        comm: Some(comm),
        notes: vec![
            "shallow SIMD context; whole input = 1 ciphertext (dot_plain)".into(),
            "exact real-protocol ABY3 ReLU via scheme switch — no deep context needed".into(),
        ],
    }
}

/// The fully-private hybrid: weights AND input encrypted, on BFV — an RLWE
/// lattice, so the standard `log q ≤ 109 at n = 4096` bound is the only
/// constraint. One ct×ct matvec per weight chunk
/// (decrypt under s², no relinearization), poly-level mask-and-open switch,
/// real-protocol ABY3 ReLU, packed re-encryption. This is the no-bootstrap
/// construction running end to end on a *provably-secure* FHE scheme.
fn run_hybrid_bfv(data: &LayerData, profile: NetworkProfile) -> StrategyReport {
    let mut t = Timer::new();
    let in_dim = data.x.len();
    let out_dim = data.w.len();
    // Plaintext modulus sized to the *actual* activation range: a d-term dot
    // product of ~50-magnitude values reaches ~2^21 at d=784, which would
    // silently wrap a fixed t=2^20. (A real deployment calibrates this from
    // sample data — that is `ml::QuantizationReport`'s job.) q stays 2^100,
    // so security is unchanged; only the noise budget Δ/2 = q/2t shrinks,
    // which the cap below keeps comfortable for depth 1.
    let max_abs_pre = data.pre.iter().map(|v| v.abs()).max().unwrap_or(1).max(1) as u64;
    let log2_t = (66 - max_abs_pre.leading_zeros()).max(20);
    assert!(
        log2_t <= 30,
        "activation range needs t=2^{log2_t}, beyond BFV's depth-1 noise budget at q=2^100"
    );
    let s = Bfv::new(BfvParams { n: 4096, log2_q: 100, log2_t });
    let nbits = s.params.log2_t as usize;
    let t_mask = s.params.t() - 1;
    // Chunking lifts the single-ciphertext ceiling: the packing bound
    // (rows+1)·in ≤ n caps rows per ciphertext, so wider layers split into
    // ⌈out/rows_per_chunk⌉ weight ciphertexts, each multiplied against the
    // SAME encrypted input. That count, ⌈out·in/n⌉, is the fundamental limit
    // of coefficient packing for a dense layer: a degree-n product yields at
    // most n/in *separable* outputs, since distinct rows need disjoint
    // weight windows. (Gazelle's SIMD packing achieves the same MAC
    // throughput but needs rotations + Galois keys — which would forfeit
    // this construction's no-evaluation-keys property. Our layout is the
    // rotation-free, Cheetah-style one.)
    assert!(2 * in_dim <= s.params.n, "in_dim exceeds the ring dimension");
    let rows_per_chunk = (s.params.n / in_dim).saturating_sub(1).min(out_dim.max(1));
    let chunks: Vec<&[Vec<i64>]> = data.w.chunks(rows_per_chunk).collect();
    let adapter = BfvAdapter(&s);

    let (ck, _server) = t.phase("keygen", || s.keygen());

    // Both input and weights encrypted (fully private linear phase). The
    // *weights* are query-independent, so their encryption is offline; only
    // the client's input encryption is on the critical path.
    let cws = t.phase("enc-weights", || {
        chunks
            .iter()
            .map(|chunk| s.encrypt_poly(&ck, &s.pack_weights(chunk)))
            .collect::<Vec<BfvCt>>()
    });
    let cx = t.phase("enc-input", || s.encrypt_poly(&ck, &s.pack_input(&data.x)));
    t.phase("link-io", || {
        let bytes =
            adapter.ct_bytes(&cx) + cws.iter().map(|c| adapter.ct_bytes(c)).sum::<usize>();
        sleep_link(&profile, bytes, 2);
    });

    // The matvec: one ct×ct multiply per weight chunk (degree-2, no relin),
    // all against the same input — whose forward NTT is computed once and
    // reused, saving 2 of 4 forward transforms per prime per chunk. Bias is
    // added on the MPC side.
    let pre_chunks: Vec<BfvCt> = t.phase("linear", || {
        let cxt = s.transform(&cx);
        cws.iter().map(|cw| s.mul_pre(&cxt, cw)).collect()
    });

    // Poly-level mask-and-open switch (masks are full uniform-mod-t polys).
    //
    // OFFLINE/ONLINE SPLIT (Delphi-style): both the to-MPC masks r_i and the
    // return-path randomness rho_i are *data-independent*, so their
    // encryptions are precomputed before the query arrives. Online, to-MPC
    // is 3 ciphertext adds + 1 decryption, and the return is 3 *plaintext*
    // adds — no fresh encryption on the critical path at all.
    let mut rng = rand::thread_rng();
    // FRESH masks per chunk. Reusing one mask set across chunks would be a
    // leak: every chunk opens the same coefficient positions, so two chunks
    // masked identically would reveal their difference (pre_i − pre_j) to
    // whoever sees both openings.
    let masks: Vec<Vec<Vec<u64>>> = (0..chunks.len())
        .map(|_| {
            (0..3)
                .map(|_| (0..s.params.n).map(|_| rng.gen::<u64>() & t_mask).collect())
                .collect()
        })
        .collect();
    let rho: Vec<Vec<u64>> = (0..3)
        .map(|_| {
            // Random only at the output positions; zero elsewhere so the
            // resulting ciphertext carries no junk for a following layer.
            let mut v = vec![0u64; s.params.n];
            for slot in v.iter_mut().take(out_dim) {
                *slot = rng.gen::<u64>() & t_mask;
            }
            v
        })
        .collect();
    let (enc_masks, enc_rho) = t.phase("offline-prep", || {
        let em: Vec<Vec<BfvCt>> = masks
            .iter()
            .map(|chunk_masks| {
                chunk_masks.iter().map(|m| s.encrypt_poly(&ck, m)).collect()
            })
            .collect();
        let er: Vec<BfvCt> = rho.iter().map(|r| s.encrypt_poly(&ck, r)).collect();
        (em, er)
    });

    let opened: Vec<Vec<u64>> = t.phase("to-mpc", || {
        let mut conv_bytes = 0usize;
        let y: Vec<Vec<u64>> = pre_chunks
            .iter()
            .enumerate()
            .map(|(ci, pre)| {
                let mut masked = pre.clone();
                for enc in &enc_masks[ci] {
                    conv_bytes += adapter.ct_bytes(enc);
                    masked = s.add(&masked, enc);
                }
                s.decrypt_poly(&ck, &masked)
            })
            .collect();
        sleep_link(&profile, conv_bytes + y.iter().map(|v| v.len() * 8).sum::<usize>(), 2);
        y
    });
    // Flatten (chunk, local row) -> per-output opened value and its chunk's
    // own masks.
    let mut flat_masks: Vec<[u64; 3]> = Vec::with_capacity(out_dim);
    let mut flat_opened: Vec<u64> = Vec::with_capacity(out_dim);
    for (ci, chunk) in chunks.iter().enumerate() {
        for r in 0..chunk.len() {
            let pos = s.matvec_output_position(in_dim, r);
            flat_masks.push([masks[ci][0][pos], masks[ci][1][pos], masks[ci][2][pos]]);
            flat_opened.push(opened[ci][pos]);
        }
    }
    let bias = data.bias.clone();

    let (per_party, comm) = t.phase("mpc-relu", || {
        run_parties(profile, |me| {
            let my_masks: Vec<u64> = flat_masks.iter().map(|m| m[me]).collect();
            let y = flat_opened.clone();
            let bias = bias.clone();
            move |be: &mut TransportBackend<MpcTransport>| {
                // Import + bias for every output first, so the nonlinearity
                // can be issued as ONE batch. Interleaving the ReLU into this
                // loop would serialise the comparisons and cost O(len) rounds.
                let my_comps: Vec<Fr> = (0..y.len())
                    .map(|j| {
                        if me == 0 {
                            (y[j] as u128).wrapping_sub(my_masks[j] as u128)
                        } else {
                            0u128.wrapping_sub(my_masks[j] as u128)
                        }
                    })
                    .collect();
                let raws = import_batched(be, &my_comps, me, nbits);
                // The bias is public-to-party-0, so it is ONE batched sharing.
                let bvals: Vec<Fr> = (0..y.len())
                    .map(|j| if me == 0 { residue(bias[j], nbits) } else { 0 })
                    .collect();
                let bshares = be.share_secret_batch(&bvals, nbits, 0);
                let zs: Vec<ShareRef> = raws
                    .iter()
                    .zip(&bshares)
                    .map(|(raw, b)| be.add(raw, b))
                    .collect();
                crate::gadgets::relu(be, &zs)
                    .iter()
                    .map(|relu| be.my_additive_component(relu))
                    .collect::<Vec<Fr>>()
            }
        })
    });

    // Return switch via **mask-and-correct**: party i published only
    // δ_i = c_i − ρ_i (uniform ⇒ leaks nothing); the evaluator folds it into
    // the precomputed Enc(ρ_i) with a *plaintext* addition. Σ(ρ_i + δ_i) =
    // Σ c_i, so this is the same interactive bootstrap — fresh, key_pow-1 —
    // with no online encryption.
    let back = t.phase("to-fhe", || {
        let mut sum: Option<BfvCt> = None;
        let mut bytes = 0usize;
        for (i, comps) in per_party.iter().enumerate() {
            let mut delta = vec![0u64; s.params.n];
            for (j, &c) in comps.iter().enumerate() {
                delta[j] = ((c as u64) & t_mask).wrapping_sub(rho[i][j]) & t_mask;
            }
            bytes += comps.len() * 8; // only the masked corrections go on the wire
            let e = s.add_plain_poly(&enc_rho[i], &delta);
            sum = Some(match sum {
                None => e,
                Some(acc) => s.add(&acc, &e),
            });
        }
        sleep_link(&profile, bytes, 1);
        sum.expect("three parties")
    });

    let output = t.phase("decrypt", || {
        let coeffs = s.decrypt_poly(&ck, &back);
        (0..out_dim).map(|j| s.to_signed(coeffs[j] & t_mask)).collect::<Vec<i64>>()
    });

    let total_ms = t.total_ms();
    let t_phases = t.phases.clone();
    StrategyReport {
        strategy: "hybrid-bfv".into(),
        total_ms,
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        comm: Some(comm),
        notes: {
            // Offline = query-independent work (keygen, weight encryption,
            // mask/rho precomputation). Online = the latency a query
            // actually experiences.
            let offline: f64 = t_phases
                .iter()
                .filter(|p| matches!(p.name, "keygen" | "enc-weights" | "offline-prep"))
                .map(|p| p.micros as f64 / 1000.0)
                .sum();
            let online = total_ms - offline;
            vec![
                format!("BFV (RLWE, SECURE): n={}, log q={}, one ct×ct (no relin, decrypt under s²)", s.params.n, s.params.log2_q),
                "no-bootstrap pipeline on a securely-parameterized RLWE scheme".into(),
                format!(
                    "offline/online split: {offline:.0}ms precomputed (query-independent), {online:.0}ms ONLINE — same total work, moved off the critical path"
                ),
            ]
        },
    }
}

/// Harness-validation strategy: the full hybrid pipeline (including the
/// real-protocol MPC segment) on the plaintext mock FHE backend.
fn run_hybrid_mock(data: &LayerData, profile: NetworkProfile) -> StrategyReport {
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
    let adapter = IntFhe(&f);
    let (back, comm) = hybrid_mpc_relu(&mut t, &adapter, &ck, &pre, NBITS, profile);
    let output = t.phase("decrypt", || {
        back.iter()
            .map(|ct| to_signed_n(adapter.decrypt(&ck, ct), NBITS))
            .collect::<Vec<i64>>()
    });
    StrategyReport {
        strategy: "hybrid-mock".into(),
        total_ms: t.total_ms(),
        phases: t.phases,
        ok: output == data.expected,
        max_abs_error_vs_exact: max_err(&output, &data.expected),
        output,
        comm: Some(comm),
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
    HybridBfv,
    HybridMock,
}

impl Strategy {
    /// The inverse of [`Strategy::parse`], and the same string each strategy
    /// reports as `StrategyReport::strategy` — which is what lets the
    /// cross-backend summary match rows up.
    fn name(self) -> &'static str {
        match self {
            Strategy::Mpc => "mpc",
            Strategy::Tfhe => "tfhe",
            Strategy::Ckks => "ckks",
            Strategy::HybridTfhe => "hybrid-tfhe",
            Strategy::HybridCkks => "hybrid-ckks",
            Strategy::HybridBfv => "hybrid-bfv",
            Strategy::HybridMock => "hybrid-mock",
        }
    }

    fn parse(s: &str) -> Result<Strategy, String> {
        match s {
            "mpc" => Ok(Strategy::Mpc),
            "tfhe" => Ok(Strategy::Tfhe),
            "ckks" => Ok(Strategy::Ckks),
            "hybrid-tfhe" => Ok(Strategy::HybridTfhe),
            "hybrid-ckks" => Ok(Strategy::HybridCkks),
            "hybrid-bfv" => Ok(Strategy::HybridBfv),
            "hybrid-mock" => Ok(Strategy::HybridMock),
            other => Err(format!(
                "unknown strategy '{other}' (expected mpc, tfhe, ckks, hybrid-tfhe, \
                 hybrid-ckks, hybrid-bfv, hybrid-mock)"
            )),
        }
    }
}

pub fn run_strategy(
    s: Strategy,
    data: &LayerData,
    relu_iterations: u32,
    profile: NetworkProfile,
) -> StrategyReport {
    match s {
        Strategy::Mpc => run_mpc_only(data, profile),
        Strategy::Tfhe => run_tfhe_only(data, profile),
        Strategy::Ckks => run_ckks_only(data, relu_iterations, profile),
        Strategy::HybridTfhe => run_hybrid_tfhe(data, profile),
        Strategy::HybridCkks => run_hybrid_ckks(data, profile),
        Strategy::HybridBfv => run_hybrid_bfv(data, profile),
        Strategy::HybridMock => run_hybrid_mock(data, profile),
    }
}

const USAGE: &str = "usage: baru bench-hybrid-nn [options]

One private dense layer (matvec + bias + ReLU) computed end-to-end under
six strategies — pure ABY3 MPC (real wire protocol, 3 party threads), pure
TFHE, pure CKKS (deep context, polynomial ReLU), and the Gazelle-style
hybrids (TFHE/CKKS/BFV linear, then a scheme switch to real-protocol
ABY3 for the ReLU) — each timed per phase and verified against the
plaintext reference. A network model injects latency/bandwidth into every
MPC protocol message AND the FHE upload/download + conversion round trips.

options:
  --in N               input dimension (default 8)
  --out M              output dimension (default 2)
  --strategies LIST    comma-separated subset (default: all except mock)
  --relu-iterations K  Newton iterations for the ckks strategy (default 2)
  --rtt-ms N           round-trip latency between any two machines (default 0)
  --bandwidth-mbps X   link bandwidth in Mbit/s (default: unlimited)
  --poly B             polynomial backend for the Rust FHE schemes: ntt
                       (default), ntl, or `all` to sweep both. Moves only the
                       hybrid-bfv row — ckks/tfhe use SEAL
                       and tfhe-rs, which have their own arithmetic, and mpc
                       has no polynomials at all. Flat rows there are the
                       expected result, not a bug. (ntl needs --features ntl)
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
    let mut rtt_ms: u64 = 0;
    let mut bandwidth_mbps: Option<f64> = None;
    let mut json = false;
    let mut poly_sweep: Vec<PolyBackend> = vec![PolyBackend::Ntt];
    let mut strategies = vec![
        Strategy::Mpc,
        Strategy::HybridBfv,
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
            "--rtt-ms" => {
                i += 1;
                rtt_ms = args
                    .get(i)
                    .ok_or("--rtt-ms needs a value")?
                    .parse()
                    .map_err(|_| "--rtt-ms needs a non-negative integer".to_string())?;
            }
            "--bandwidth-mbps" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("--bandwidth-mbps needs a value")?
                    .parse()
                    .map_err(|_| "--bandwidth-mbps needs a number".to_string())?;
                bandwidth_mbps = Some(v);
            }
            "--strategies" => {
                i += 1;
                let list = args.get(i).ok_or("--strategies needs a value")?;
                strategies = list
                    .split(',')
                    .map(Strategy::parse)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--poly" => {
                i += 1;
                let v = args.get(i).ok_or("--poly needs a value")?;
                poly_sweep = if v == "all" {
                    poly::available()
                } else {
                    let b = PolyBackend::parse(v)?;
                    if !poly::is_available(b) {
                        return Err(format!(
                            "polynomial backend {v:?} is not in this build (rebuild with --features ntl)"
                        ));
                    }
                    vec![b]
                };
            }
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    let mut profile =
        NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt_ms * 500));
    if let Some(mbps) = bandwidth_mbps {
        profile = profile.with_bandwidth(mbps * 1e6 / 8.0);
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
        println!(
            "layer({in_dim}x{out_dim})  rtt={rtt_ms}ms  bandwidth={}  reference: {:?}",
            bandwidth_mbps
                .map(|b| format!("{b}Mbps"))
                .unwrap_or_else(|| "unlimited".into()),
            data.expected
        );
        println!("                 (pre-activation: {:?})\n", data.pre);
    }

    let mut reports = Vec::new();
    let sweeping = poly_sweep.len() > 1;
    for &pb in &poly_sweep {
    if !json && sweeping {
        println!("── polynomial backend: {} ──\n", pb.name());
    }
    for &s in &strategies {
        let r = poly::with_backend(pb, || run_strategy(s, &data, relu_iterations, profile))?;
        if !json {
            let phases: Vec<String> = r
                .phases
                .iter()
                .map(|p| format!("{}={}", p.name, fmt_ms(p.micros)))
                .collect();
            let comm = r
                .comm
                .map(|c| format!("  comm[msgs={} bytes={}]", c.messages, c.bytes))
                .unwrap_or_default();
            println!(
                "{:<12} total={:<10} {}{}  {}",
                r.strategy,
                fmt_ms((r.total_ms * 1000.0) as u128),
                if r.ok { "✓" } else { "✗" },
                comm,
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
        reports.push((pb, r));
    }
    }

    if json {
        let payload: Vec<_> = reports
            .iter()
            .map(|(b, r)| serde_json::json!({ "poly_backend": b.name(), "report": r }))
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    // Same strategy, same layer, only the ring arithmetic swapped.
    if sweeping {
        let base = poly_sweep[0];
        println!("total-time speedup vs {}:", base.name());
        for &pb in poly_sweep.iter().skip(1) {
            let line: Vec<String> = strategies
                .iter()
                .filter_map(|s| {
                    let f = |b: PolyBackend| {
                        reports
                            .iter()
                            .find(|(rb, r)| *rb == b && r.strategy == s.name())
                            .map(|(_, r)| r.total_ms)
                    };
                    match (f(base), f(pb)) {
                        (Some(a), Some(c)) if c > 0.0 => Some(format!("{}={:.2}x", s.name(), a / c)),
                        _ => None,
                    }
                })
                .collect();
            println!("  {:<4} {}", pb.name(), line.join("  "));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpc_only_matches_plaintext_on_real_protocol() {
        let data = make_layer(6, 3);
        let r = run_mpc_only(&data, NetworkProfile::INSTANT);
        assert!(
            r.ok,
            "mpc output {:?} != expected {:?}",
            r.output, data.expected
        );
        assert_eq!(r.max_abs_error_vs_exact, 0);
        let comm = r.comm.expect("real protocol must report wire traffic");
        assert!(comm.messages > 0, "real protocol must send messages");
    }

    #[test]
    fn hybrid_mock_pipeline_matches_plaintext_on_real_protocol() {
        // Validates the full FHE-linear -> mask/open -> real-protocol
        // import+relu+export -> re-encrypt pipeline without crypto cost.
        let data = make_layer(6, 3);
        let r = run_hybrid_mock(&data, NetworkProfile::INSTANT);
        assert!(
            r.ok,
            "hybrid-mock output {:?} != expected {:?}",
            r.output, data.expected
        );
        assert!(r.comm.expect("has an MPC segment").messages > 0);
    }

    #[test]
    fn latency_profile_slows_the_mpc_segment_down() {
        // The whole point of the network model: the same computation must
        // get measurably slower when rounds cost something.
        let data = make_layer(4, 1);
        let fast = run_mpc_only(&data, NetworkProfile::INSTANT);
        let slow = run_mpc_only(
            &data,
            NetworkProfile::fixed_latency(std::time::Duration::from_millis(2)),
        );
        assert!(
            fast.ok && slow.ok,
            "fast: {:?} slow: {:?} expected: {:?}",
            fast.output,
            slow.output,
            data.expected
        );
        assert!(
            slow.total_ms > fast.total_ms + 5.0,
            "2ms one-way latency must visibly slow a multi-round protocol: \
             fast={:.1}ms slow={:.1}ms",
            fast.total_ms,
            slow.total_ms
        );
    }

    #[test]
    #[ignore] // slow in debug; fast in release
    fn hybrid_bfv_chunked_lifts_the_size_ceiling() {
        // (out+1)·in = 33·256 = 8448 > n = 4096, so this REQUIRES chunking
        // (it would have hit the old single-chunk assertion).
        let data = make_layer(256, 32);
        let r = run_hybrid_bfv(&data, NetworkProfile::INSTANT);
        assert!(r.ok, "chunked hybrid-bfv output {:?} != {:?}", r.output, data.expected);
        assert_eq!(r.max_abs_error_vs_exact, 0);
    }

    #[test]
    #[ignore] // slow in debug (n=4096 schoolbook ring); fast in release
    fn hybrid_bfv_secure_matches_plaintext() {
        // The no-bootstrap construction end to end on the SECURE (RLWE) FHE
        // scheme, real ABY3 protocol, exact vs plaintext.
        let data = make_layer(8, 2);
        let r = run_hybrid_bfv(&data, NetworkProfile::INSTANT);
        assert!(
            r.ok,
            "hybrid-bfv output {:?} != expected {:?}",
            r.output, data.expected
        );
        assert_eq!(r.max_abs_error_vs_exact, 0, "secure hybrid must be exact");
    }

    #[test]
    #[ignore] // slow: real SEAL keygen (~seconds)
    fn hybrid_ckks_matches_plaintext() {
        let data = make_layer(8, 2);
        let r = run_hybrid_ckks(&data, NetworkProfile::INSTANT);
        assert!(
            r.ok,
            "hybrid-ckks output {:?} != expected {:?}",
            r.output, data.expected
        );
        assert_eq!(r.max_abs_error_vs_exact, 0, "hybrid must be exact");
    }
}
