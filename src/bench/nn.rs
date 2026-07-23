//! `baru bench-nn` — size-swept private neural-network inference benchmarks.
//!
//! Unlike [`super`]'s expression-level harness, these workloads drive
//! `Backend` (MPC) / `FheBackend` (FHE) calls **directly**, bypassing the Lisp
//! interpreter/circuit-compiler entirely.  Two reasons:
//!   - Precise cost attribution: every millisecond and every byte measured
//!     comes from the crypto backend, not interpreter/AST overhead.
//!   - Scalability: sweeping a 784-element (MNIST-scale) vector as a Lisp
//!     list would add interpreter cost unrelated to what we're profiling.
//!
//! Workloads model the building blocks of a private NN forward pass:
//!   dot(n)          inner product of two length-n private vectors
//!   matvec(in,out)  a dense/linear layer's weight matrix times an input
//!   relu(n)         elementwise ReLU (the expensive nonlinearity: needs a
//!                   comparison in MPC, and is *impossible* to express
//!                   directly on CKKS's approximate ciphertexts)
//!   layer(in,out)   matvec + bias-add + relu — one full dense layer
//!
//! Backends:
//!   mpc         3-party ABY3 over `TransportBackend`, wrapped in
//!               [`super::comm`]'s counting/latency `Transport` decorators —
//!               the *real* replicated-secret-sharing wire protocol (not the
//!               legacy `Aby3Backend` simulator), so message counts/bytes are
//!               genuine.  Both the input vector and the weight matrix are
//!               secret-shared (the fully-private setting).
//!   fhe-mock    integer FHE pipeline on plaintext u64 "ciphertexts" (instant;
//!               use to validate the harness and get a computation-cost
//!               floor with zero crypto overhead).
//!   fhe-tfhe    real TFHE (FheUint64) — slow; opt in explicitly.
//!   fhe-ckks    real CKKS (SEAL), two forms: "circuit" (one ciphertext per
//!               scalar, matching the mock/tfhe shape) and "batched" (the
//!               whole vector in one SIMD ciphertext via `encrypt_vec` +
//!               `dot_plain`) — the batched/circuit contrast is itself one of
//!               the most important communication-size findings this harness
//!               surfaces.  ReLU is not attempted (impossible on approximate
//!               ciphertexts; `RealFheBackend` has no comparison/mux).
//!
//! FHE weights are plaintext (`mul_plain`/`add_plain`): the common
//! "public model, private input" deployment (matching `examples/mnist.rs`),
//! and the cheaper, more realistic default. This is a real scope difference
//! from the MPC path (private weights) — noted in each report's `notes`.

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::bench::comm::{CommStats, CountingTransport, LatencyTransport, NetworkProfile};
use crate::bench::Phase;
use crate::crypto::net::InprocHandle;
use crate::crypto::types::Fr;
use crate::fhe::backend::FheBackend;
use crate::fhe::mock_backend::MockFheBackend;
use crate::fhe::tfhe_backend::TfheBackend;
use crate::fhe::{CkksBackend, MockCkksBackend, RealFheBackend};
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

const NBITS: usize = 32;

// ─── Signed <-> field-element conversion (mirrors eval.rs's REPL convention) ──

fn signed_to_fr(n: i64, nbits: usize) -> Fr {
    let modulus = 1i128.checked_shl(nbits as u32).unwrap_or(0);
    if modulus == 0 {
        return n as i128 as u128;
    }
    (n as i128).rem_euclid(modulus) as u128
}

fn fr_to_signed(raw: Fr, nbits: usize) -> i64 {
    if nbits >= 128 {
        return raw as i128 as i64;
    }
    // The revealed `raw` is a full 128-bit wrapping field element (the sum of
    // Beaver-triple-protocol shares) — it is NOT already confined to `nbits`
    // bits, even though the true mathematical value is.  Reduce to the low
    // `nbits` bits first (mirrors the hardcoded `(raw as u32) as i32 as i64`
    // convention used elsewhere in this codebase for the nbits=32 case),
    // *then* apply the two's-complement sign decode.
    let modulus = 1u128 << nbits;
    let v = raw % modulus;
    let half = modulus / 2;
    if v >= half {
        (v as i128 - modulus as i128) as i64
    } else {
        v as i64
    }
}

/// Deterministic, reproducible pseudo-input in roughly `[-50, 50]`, spanning
/// both signs (important: ReLU/comparison correctness needs negative inputs
/// exercised, not just positive ones).
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

// ─── Workloads ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    Dot(usize),
    MatVec { in_dim: usize, out_dim: usize },
    Relu(usize),
    Layer { in_dim: usize, out_dim: usize },
}

impl Workload {
    pub fn label(&self) -> String {
        match self {
            Workload::Dot(n) => format!("dot({n})"),
            Workload::MatVec { in_dim, out_dim } => format!("matvec({in_dim}x{out_dim})"),
            Workload::Relu(n) => format!("relu({n})"),
            Workload::Layer { in_dim, out_dim } => format!("layer({in_dim}x{out_dim})"),
        }
    }

    /// Multiply-accumulate-equivalent element count, for throughput.
    pub fn op_count(&self) -> usize {
        match self {
            Workload::Dot(n) => *n,
            Workload::MatVec { in_dim, out_dim } => in_dim * out_dim,
            Workload::Relu(n) => *n,
            Workload::Layer { in_dim, out_dim } => in_dim * out_dim,
        }
    }

    /// Does this workload need a comparison (ReLU)? CKKS can't do this.
    pub fn needs_comparison(&self) -> bool {
        matches!(self, Workload::Relu(_) | Workload::Layer { .. })
    }
}

// ─── The standard suite ────────────────────────────────────────────────────────
//
// A fixed, named set of sizes/shapes — deliberately NOT ad hoc per-run
// choices — so results stay comparable across commits, backends, and
// precompiles over time (this is what makes `--json` output meaningful to
// diff months apart, and what a precompile competition ranks entries
// against). Changing these numbers changes what "the standard benchmark"
// means; treat that as a deliberate, reviewed decision, not a drive-by edit.
//
//   "quick"        — `default_vector_sizes`/`default_matvec_shapes`: small
//                    enough that a full run (even including TFHE) finishes
//                    in a reasonable time. The default.
//   "mnist-linear" — `MNIST_SCALE_VECTOR`/`MNIST_SCALE_MATVEC`: a real
//                    application shape (784 = examples/mnist.rs's flattened
//                    28x28 input, 10 = digit classes), opted into via
//                    `--mnist-scale` since it's slow, especially for TFHE.

/// The "quick" standard suite's vector sizes (`dot`, `relu`).
pub fn default_vector_sizes() -> Vec<usize> {
    vec![4, 16, 64, 256]
}

/// The "quick" standard suite's matrix shapes (`matvec`, `layer`).
pub fn default_matvec_shapes() -> Vec<(usize, usize)> {
    vec![(8, 8), (64, 32), (128, 64)]
}

/// The "mnist-linear" standard application's vector size — `examples/
/// mnist.rs`'s flattened 28x28 image, the realistic large-scale data point.
pub const MNIST_SCALE_VECTOR: usize = 784;
/// The "mnist-linear" standard application's matrix shape — 784 pixels in,
/// 10 digit classes out.
pub const MNIST_SCALE_MATVEC: (usize, usize) = (784, 10);

// ─── Plaintext reference (ground truth for correctness checks) ───────────────

fn plain_dot(a: &[i64], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn plain_matvec(w: &[Vec<i64>], x: &[i64]) -> Vec<i64> {
    w.iter().map(|row| plain_dot(row, x)).collect()
}

fn plain_relu(v: &[i64]) -> Vec<i64> {
    v.iter().map(|&x| if x > 0 { x } else { 0 }).collect()
}

fn plain_layer(w: &[Vec<i64>], x: &[i64], bias: &[i64]) -> Vec<i64> {
    let pre: Vec<i64> = plain_matvec(w, x)
        .iter()
        .zip(bias)
        .map(|(a, b)| a + b)
        .collect();
    plain_relu(&pre)
}

/// Deterministic weight matrix / bias, same generator family as `gen_vec`.
fn gen_matrix(in_dim: usize, out_dim: usize, seed: i64) -> Vec<Vec<i64>> {
    (0..out_dim)
        .map(|i| gen_vec(in_dim, seed + i as i64 * 1000))
        .collect()
}

// ─── Reports ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct NnReport {
    pub workload: String,
    pub backend: String,
    pub phases: Vec<Phase>,
    pub total_micros: u128,
    /// `op_count() / total_seconds` — comparable across backends.
    pub throughput_ops_per_sec: f64,
    /// Message-count/byte totals (MPC only; `None` for FHE, whose
    /// communication story is instead in `notes` as ciphertext count/bytes).
    pub comm: Option<CommStats>,
    pub ok: Option<bool>,
    pub notes: Vec<String>,
    pub error: Option<String>,
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
    fn total_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
    fn total_micros(&self) -> u128 {
        self.start.elapsed().as_micros()
    }
}

/// `ops / secs`, flooring the divisor so this never produces `inf`/`NaN` —
/// `serde_json` errors on non-finite floats, which would break `--json`
/// output for any workload fast enough to round to `0.0` elapsed seconds.
fn throughput(ops: usize, secs: f64) -> f64 {
    ops as f64 / secs.max(1e-9)
}

// ─── MPC runner (real replicated 3-party protocol, instrumented transport) ───

type MpcTransport = CountingTransport<LatencyTransport<InprocHandle>>;

/// One party's full evaluation of `workload`, run inside its own thread.
/// Party 0 owns all private data (who owns it doesn't affect cost — the
/// protocol is symmetric); other parties pass placeholder values.
#[allow(clippy::too_many_arguments)]
fn mpc_party_run(
    me: usize,
    transport: MpcTransport,
    workload: Workload,
    x: Vec<i64>,
    w: Vec<Vec<i64>>,
    bias: Vec<i64>,
    precompile_id: Option<&'static str>,
) -> (Vec<i64>, Vec<Phase>) {
    const OWNER: usize = 0;
    let mut be = TransportBackend::new(me, transport);
    let mut t = Timer::new();

    let share_vec = |be: &mut TransportBackend<MpcTransport>, vals: &[i64]| -> Vec<ShareRef> {
        vals.iter()
            .map(|&v| {
                let fr = if me == OWNER {
                    signed_to_fr(v, NBITS)
                } else {
                    0
                };
                be.share_secret(fr, NBITS, OWNER)
            })
            .collect()
    };

    let outputs: Vec<ShareRef> = match workload {
        Workload::Dot(_) => {
            let (xs, ys) = t.phase("share", || {
                (share_vec(&mut be, &x), share_vec(&mut be, &w[0]))
            });
            let out = t.phase("compute", || {
                let mut acc = be.mul(&xs[0], &ys[0]);
                for i in 1..xs.len() {
                    let term = be.mul(&xs[i], &ys[i]);
                    acc = be.add(&acc, &term);
                }
                acc
            });
            vec![out]
        }
        Workload::MatVec { .. } => {
            let (xs, ws) = t.phase("share", || {
                let xs = share_vec(&mut be, &x);
                let ws: Vec<Vec<ShareRef>> = w.iter().map(|row| share_vec(&mut be, row)).collect();
                (xs, ws)
            });
            let registry = crate::precompile::mpc_matvec::registry();
            let precompile = precompile_id
                .and_then(|id| registry.get(id))
                .unwrap_or_else(|| registry.champion());
            t.phase("compute", || precompile.matvec(&mut be, &ws, &xs))
        }
        Workload::Relu(_) => {
            let xs = t.phase("share", || share_vec(&mut be, &x));
            t.phase("compute", || {
                xs.iter()
                    .map(|xr| {
                        let zero = be.sub(xr, xr); // local, no round
                        let mask = be.lt(&zero, xr); // 1 round: 0 < x
                        be.mul(xr, &mask) // 1 round: x * mask
                    })
                    .collect()
            })
        }
        Workload::Layer { .. } => {
            let (xs, ws, bs) = t.phase("share", || {
                let xs = share_vec(&mut be, &x);
                let ws: Vec<Vec<ShareRef>> = w.iter().map(|row| share_vec(&mut be, row)).collect();
                let bs = share_vec(&mut be, &bias);
                (xs, ws, bs)
            });
            let registry = crate::precompile::mpc_matvec::registry();
            let precompile = precompile_id
                .and_then(|id| registry.get(id))
                .unwrap_or_else(|| registry.champion());
            t.phase("compute", || {
                let pre_activation = precompile.matvec(&mut be, &ws, &xs);
                pre_activation
                    .iter()
                    .zip(bs.iter())
                    .map(|(acc, b)| {
                        let acc = be.add(acc, b);
                        let zero = be.sub(&acc, &acc);
                        let mask = be.lt(&zero, &acc);
                        be.mul(&acc, &mask)
                    })
                    .collect()
            })
        }
    };

    let revealed = t.phase("reveal", || {
        outputs
            .iter()
            .map(|sr| fr_to_signed(be.reveal(sr, None), NBITS))
            .collect::<Vec<i64>>()
    });

    (revealed, t.phases)
}

/// Run the MPC path with the registry's champion precompile (the normal,
/// default-recommended path).
pub fn run_mpc(workload: Workload, profile: NetworkProfile) -> NnReport {
    run_mpc_as(workload, profile, None)
}

/// Run the MPC path forcing a *specific* precompile id (or the champion, if
/// `None`) — the basis of the precompile competition/leaderboard mode: run
/// every registered entry through the identical measurement and compare.
pub fn run_mpc_as(
    workload: Workload,
    profile: NetworkProfile,
    precompile_id: Option<&'static str>,
) -> NnReport {
    let (in_dim, out_dim) = match workload {
        Workload::Dot(n) => (n, 1),
        Workload::MatVec { in_dim, out_dim } => (in_dim, out_dim),
        Workload::Relu(n) => (n, 1),
        Workload::Layer { in_dim, out_dim } => (in_dim, out_dim),
    };
    let x = gen_vec(in_dim, 1);
    let w = gen_matrix(in_dim, out_dim, 2);
    let bias = gen_vec(out_dim, 3);

    let handles = InprocHandle::create(3);
    let mut comm_handles = Vec::with_capacity(3);
    let mut joins = Vec::with_capacity(3);
    let overall_start = Instant::now();

    for (me, h) in handles.into_iter().enumerate() {
        let lat = LatencyTransport::new(h, profile);
        let counted = CountingTransport::new(lat, 3);
        comm_handles.push(counted.handle());
        let x = x.clone();
        let w = w.clone();
        let bias = bias.clone();
        joins.push(std::thread::spawn(move || {
            mpc_party_run(me, counted, workload, x, w, bias, precompile_id)
        }));
    }

    let mut per_party: Vec<(Vec<i64>, Vec<Phase>)> = Vec::with_capacity(3);
    for j in joins {
        match j.join() {
            Ok(r) => per_party.push(r),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic in MPC party thread".to_string());
                return NnReport {
                    workload: workload.label(),
                    backend: "mpc".into(),
                    phases: vec![],
                    total_micros: overall_start.elapsed().as_micros(),
                    throughput_ops_per_sec: 0.0,
                    comm: None,
                    ok: None,
                    notes: vec![],
                    error: Some(msg),
                };
            }
        }
    }

    let total_secs = overall_start.elapsed().as_secs_f64();
    let result = per_party[0].0.clone();
    let expected = match workload {
        Workload::Dot(_) => vec![plain_dot(&x, &w[0])],
        Workload::MatVec { .. } => plain_matvec(&w, &x),
        Workload::Relu(_) => plain_relu(&x),
        Workload::Layer { .. } => plain_layer(&w, &x, &bias),
    };
    let ok = result == expected;

    let precompile_note = if matches!(workload, Workload::MatVec { .. } | Workload::Layer { .. }) {
        let registry = crate::precompile::mpc_matvec::registry();
        let used = precompile_id
            .and_then(|id| registry.get(id).map(|_| id))
            .unwrap_or(registry.champion_meta().id);
        Some(format!("matvec precompile={used}"))
    } else {
        None
    };

    // Party 0's phases and comm stats are representative (the protocol is
    // symmetric across parties; wall time is bounded by the slowest thread,
    // captured separately as total_micros over the whole join).
    let mut notes = vec![
        "3 parties, private input AND private weights".into(),
        format!("nbits={NBITS}"),
    ];
    notes.extend(precompile_note);
    NnReport {
        workload: workload.label(),
        backend: match precompile_id {
            Some(id) => format!("mpc[{id}]"),
            None => "mpc".into(),
        },
        phases: per_party[0].1.clone(),
        total_micros: (total_secs * 1e6) as u128,
        throughput_ops_per_sec: throughput(workload.op_count(), total_secs),
        comm: Some(comm_handles[0].stats()),
        ok: Some(ok),
        notes,
        error: None,
    }
}

// ─── FHE runner (integer backends: mock, tfhe) ───────────────────────────────

/// Public-weight pipeline: encrypt the input vector once, evaluate with
/// `mul_plain`/`add_plain` against plaintext weights — the common "public
/// model, private input" deployment (matches `examples/mnist.rs`).
///
/// `post_keygen` runs inside the `keygen` phase, right after `be.keygen()`
/// (TFHE needs this to install the server key derived from *this exact*
/// keygen call before any encrypt/eval happens against it).
pub fn run_fhe<B: FheBackend>(
    label: &str,
    be: &B,
    workload: Workload,
    post_keygen: impl Fn(&[u8]),
) -> NnReport {
    let (in_dim, out_dim) = match workload {
        Workload::Dot(n) => (n, 1),
        Workload::MatVec { in_dim, out_dim } => (in_dim, out_dim),
        Workload::Relu(n) => (n, 1),
        Workload::Layer { in_dim, out_dim } => (in_dim, out_dim),
    };
    let x = gen_vec(in_dim, 1);
    let w = gen_matrix(in_dim, out_dim, 2);
    let bias = gen_vec(out_dim, 3);

    let mut t = Timer::new();
    let ck = t.phase("keygen", || {
        let (ck, sk) = be.keygen();
        post_keygen(&sk);
        ck
    });

    let mut ct_bytes = 0usize;
    let x_ct: Vec<B::Ct> = t.phase("encrypt", || {
        x.iter()
            .map(|&v| {
                let ct = be.encrypt(&ck, signed_to_fr(v, 64) as u64);
                if ct_bytes == 0 {
                    ct_bytes = be.serialize_ct(&ct).len();
                }
                ct
            })
            .collect()
    });

    let compute = || -> Vec<B::Ct> {
        match workload {
            Workload::Dot(_) => {
                let mut acc = be.mul_plain(&x_ct[0], w[0][0]);
                for i in 1..x_ct.len() {
                    let term = be.mul_plain(&x_ct[i], w[0][i]);
                    acc = be.add(&acc, &term);
                }
                vec![acc]
            }
            Workload::MatVec { .. } => w
                .iter()
                .map(|row| {
                    let mut acc = be.mul_plain(&x_ct[0], row[0]);
                    for j in 1..row.len() {
                        let term = be.mul_plain(&x_ct[j], row[j]);
                        acc = be.add(&acc, &term);
                    }
                    acc
                })
                .collect(),
            Workload::Relu(_) => x_ct
                .iter()
                .map(|xc| {
                    let zero = be.encrypt_trivial(0);
                    let mask = be.cmpslt(&zero, xc);
                    be.mux(&mask, xc, &zero)
                })
                .collect(),
            Workload::Layer { .. } => w
                .iter()
                .zip(bias.iter())
                .map(|(row, b)| {
                    let mut acc = be.mul_plain(&x_ct[0], row[0]);
                    for j in 1..row.len() {
                        let term = be.mul_plain(&x_ct[j], row[j]);
                        acc = be.add(&acc, &term);
                    }
                    acc = be.add_plain(&acc, *b);
                    let zero = be.encrypt_trivial(0);
                    let mask = be.cmpslt(&zero, &acc);
                    be.mux(&mask, &acc, &zero)
                })
                .collect(),
        }
    };
    let out_ct = t.phase("eval", compute);

    let result: Vec<i64> = t.phase("decrypt", || {
        out_ct
            .iter()
            .map(|c| fr_to_signed(be.decrypt(&ck, c) as u128, 64))
            .collect()
    });

    let expected = match workload {
        Workload::Dot(_) => vec![plain_dot(&x, &w[0])],
        Workload::MatVec { .. } => plain_matvec(&w, &x),
        Workload::Relu(_) => plain_relu(&x),
        Workload::Layer { .. } => plain_layer(&w, &x, &bias),
    };
    let ok = result == expected;

    NnReport {
        workload: workload.label(),
        backend: label.to_string(),
        total_micros: t.total_micros(),
        throughput_ops_per_sec: throughput(workload.op_count(), t.total_secs()),
        phases: t.phases,
        comm: None,
        ok: Some(ok),
        notes: vec![
            "public (plaintext) weights, private input".into(),
            format!("ciphertexts_in={} ct_bytes_each={ct_bytes}", x_ct.len()),
        ],
        error: None,
    }
}

// ─── CKKS runner: "circuit" (per-scalar) and "batched" (SIMD) forms ──────────

/// Linear-only (no ReLU: `RealFheBackend` has no comparison/mux — approximate
/// ciphertexts fundamentally can't support one).  Two independent forms:
///   circuit  — one ciphertext per input scalar (mirrors `run_fhe`'s shape)
///   batched  — the whole vector in ONE SIMD ciphertext (`encrypt_vec` +
///              `dot_plain`) — dramatically fewer/larger messages.
pub fn run_ckks<B: RealFheBackend>(
    label_prefix: &str,
    be_client: &B,
    mk_eval: impl Fn(&[u8]) -> B,
    workload: Workload,
) -> Vec<NnReport> {
    let (in_dim, out_dim) = match workload {
        Workload::Dot(n) => (n, 1),
        Workload::MatVec { in_dim, out_dim } => (in_dim, out_dim),
        _ => unreachable!("CKKS runner only handles Dot/MatVec (no comparisons)"),
    };
    let x = gen_vec(in_dim, 1);
    let w = gen_matrix(in_dim, out_dim, 2);
    let x_f: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let w_f: Vec<Vec<f64>> = w
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    let expected: Vec<i64> = match workload {
        Workload::Dot(_) => vec![plain_dot(&x, &w[0])],
        Workload::MatVec { .. } => plain_matvec(&w, &x),
        _ => unreachable!(),
    };

    // ── circuit form: one ciphertext per scalar ──
    let circuit_report = {
        let mut t = Timer::new();
        let (ck, be) = t.phase("keygen", || {
            let (ck, sk) = be_client.keygen();
            (ck, mk_eval(&sk))
        });
        let mut ct_bytes = 0usize;
        let x_ct: Vec<B::Ct> = t.phase("encrypt", || {
            x_f.iter()
                .map(|&v| {
                    let ct = be_client.encrypt(&ck, v);
                    if ct_bytes == 0 {
                        ct_bytes = be_client.serialize_ct(&ct).len();
                    }
                    ct
                })
                .collect()
        });
        let out_ct: Vec<B::Ct> = t.phase("eval", || {
            w_f.iter()
                .map(|row| {
                    let mut acc = be.mul_plain(&x_ct[0], row[0]);
                    for j in 1..row.len() {
                        let term = be.mul_plain(&x_ct[j], row[j]);
                        acc = be.add(&acc, &term);
                    }
                    acc
                })
                .collect()
        });
        let result: Vec<i64> = t.phase("decrypt", || {
            out_ct
                .iter()
                .map(|c| be_client.decrypt(&ck, c).round() as i64)
                .collect()
        });
        NnReport {
            workload: workload.label(),
            backend: format!("{label_prefix}-circuit"),
            total_micros: t.total_micros(),
            throughput_ops_per_sec: throughput(workload.op_count(), t.total_secs()),
            phases: t.phases,
            comm: None,
            ok: Some(result == expected),
            notes: vec![
                "public weights, private input, one ciphertext per scalar".into(),
                format!("ciphertexts_in={} ct_bytes_each={ct_bytes}", x_ct.len()),
            ],
            error: None,
        }
    };

    // ── batched form: whole vector in one SIMD ciphertext ──
    let batched_report = {
        let mut t = Timer::new();
        let (ck, be) = t.phase("keygen", || {
            let (ck, sk) = be_client.keygen();
            (ck, mk_eval(&sk))
        });
        let mut ct_bytes = 0usize;
        let x_ct = t.phase("encrypt", || {
            let ct = be_client.encrypt_vec(&ck, &x_f);
            ct_bytes = be_client.serialize_ct(&ct).len();
            ct
        });
        let out_ct: Vec<B::Ct> = t.phase("eval", || {
            w_f.iter().map(|row| be.dot_plain(&x_ct, row)).collect()
        });
        let result: Vec<i64> = t.phase("decrypt", || {
            out_ct
                .iter()
                .map(|c| be_client.decrypt(&ck, c).round() as i64)
                .collect()
        });
        NnReport {
            workload: workload.label(),
            backend: format!("{label_prefix}-batched"),
            total_micros: t.total_micros(),
            throughput_ops_per_sec: throughput(workload.op_count(), t.total_secs()),
            phases: t.phases,
            comm: None,
            ok: Some(result == expected),
            notes: vec![
                "public weights, private input, whole vector in 1 SIMD ciphertext".into(),
                format!("ciphertexts_in=1 ct_bytes={ct_bytes}"),
            ],
            error: None,
        }
    };

    vec![circuit_report, batched_report]
}

// ─── Orchestration & reporting ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NnBackend {
    Mpc,
    FheMock,
    FheTfhe,
    FheCkks,
    FheCkksMock,
}

impl NnBackend {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "mpc" => Ok(NnBackend::Mpc),
            "fhe-mock" | "mock" => Ok(NnBackend::FheMock),
            "fhe-tfhe" | "tfhe" => Ok(NnBackend::FheTfhe),
            "fhe-ckks" | "ckks" => Ok(NnBackend::FheCkks),
            "fhe-ckks-mock" | "ckks-mock" => Ok(NnBackend::FheCkksMock),
            other => Err(format!(
                "unknown backend '{other}' (expected mpc, fhe-mock, fhe-tfhe, fhe-ckks, fhe-ckks-mock)"
            )),
        }
    }

    pub fn default_set() -> Vec<NnBackend> {
        vec![NnBackend::Mpc, NnBackend::FheMock, NnBackend::FheCkks]
    }
}

/// Run one workload on one backend, returning one or more report rows (CKKS
/// yields two: circuit + batched). Comparison-needing workloads are skipped
/// for CKKS (returns an empty vec) since it's a known, documented limitation
/// rather than a failure.
pub fn run_one(workload: Workload, backend: NnBackend, profile: NetworkProfile) -> Vec<NnReport> {
    match backend {
        NnBackend::Mpc => vec![run_mpc(workload, profile)],
        NnBackend::FheMock => vec![run_fhe("fhe-mock", &MockFheBackend, workload, |_| {})],
        NnBackend::FheTfhe => {
            let be = TfheBackend::new();
            vec![run_fhe("fhe-tfhe", &be, workload, |sk| {
                TfheBackend::install_server_key(sk)
            })]
        }
        NnBackend::FheCkks => {
            if workload.needs_comparison() {
                return vec![];
            }
            run_ckks(
                "fhe-ckks",
                &CkksBackend::new(),
                CkksBackend::with_server_keys,
                workload,
            )
        }
        NnBackend::FheCkksMock => {
            if workload.needs_comparison() {
                return vec![];
            }
            run_ckks(
                "fhe-ckks-mock",
                &MockCkksBackend,
                |_| MockCkksBackend,
                workload,
            )
        }
    }
}

pub fn all_workloads(mnist_scale: bool) -> Vec<Workload> {
    let mut w = Vec::new();
    for &n in &default_vector_sizes() {
        w.push(Workload::Dot(n));
        w.push(Workload::Relu(n));
    }
    for &(i, o) in &default_matvec_shapes() {
        w.push(Workload::MatVec {
            in_dim: i,
            out_dim: o,
        });
        w.push(Workload::Layer {
            in_dim: i,
            out_dim: o,
        });
    }
    if mnist_scale {
        w.push(Workload::Dot(MNIST_SCALE_VECTOR));
        w.push(Workload::MatVec {
            in_dim: MNIST_SCALE_MATVEC.0,
            out_dim: MNIST_SCALE_MATVEC.1,
        });
        w.push(Workload::Layer {
            in_dim: MNIST_SCALE_MATVEC.0,
            out_dim: MNIST_SCALE_MATVEC.1,
        });
    }
    w
}

fn fmt_duration(micros: u128) -> String {
    if micros >= 10_000_000 {
        format!("{:.1}s", micros as f64 / 1e6)
    } else if micros >= 10_000 {
        format!("{:.1}ms", micros as f64 / 1e3)
    } else {
        format!("{micros}µs")
    }
}

fn fmt_throughput(ops_per_sec: f64) -> String {
    if !ops_per_sec.is_finite() {
        "-".to_string()
    } else if ops_per_sec >= 1e6 {
        format!("{:.2}M/s", ops_per_sec / 1e6)
    } else if ops_per_sec >= 1e3 {
        format!("{:.2}K/s", ops_per_sec / 1e3)
    } else {
        format!("{ops_per_sec:.1}/s")
    }
}

fn print_report(r: &NnReport) {
    if let Some(err) = &r.error {
        println!("   {:<10} {:<16} error: {}", r.workload, r.backend, err);
        return;
    }
    let ok = match r.ok {
        Some(true) => "✓",
        Some(false) => "✗ MISMATCH",
        None => "-",
    };
    let comm = r
        .comm
        .map(|c| format!(" comm[msgs={} bytes={}]", c.messages, c.bytes))
        .unwrap_or_default();
    let phases: Vec<String> = r
        .phases
        .iter()
        .map(|p| format!("{}={}", p.name, fmt_duration(p.micros)))
        .collect();
    println!(
        "   {:<10} {:<16} total={:<9} throughput={:<10} {}{}  [{}]",
        r.workload,
        r.backend,
        fmt_duration(r.total_micros),
        fmt_throughput(r.throughput_ops_per_sec),
        ok,
        comm,
        phases.join(" "),
    );
    if !r.notes.is_empty() {
        println!("              {}", r.notes.join(" | "));
    }
}

const USAGE: &str = "usage: baru bench-nn [options]

Size-swept private-inference benchmark: dot / matvec / relu / layer workloads
across mpc, fhe-mock, fhe-tfhe, fhe-ckks. Reports timing, throughput, and (for
MPC) message counts/bytes.

options:
  --backends LIST    comma-separated: mpc,fhe-mock,fhe-tfhe,fhe-ckks,fhe-ckks-mock
                      (default: mpc,fhe-mock,fhe-ckks)
  --workload NAME     dot,matvec,relu,layer,all (default: all)
  --mnist-scale       also run the MNIST-scale (784-input) sizes (slow, esp. tfhe)
  --rtt-ms N          simulated one-way network latency injected into the MPC
                      transport, in milliseconds (default: 0 = in-process speed)
  --bandwidth-mbps N  simulated bandwidth cap for the same MPC transport
  --competition       for matvec/layer, run EVERY registered MPC precompile
                      (not just the champion) side by side — a leaderboard
                      of competing gadget implementations (see
                      `crate::precompile`), instead of one 'mpc' row.
  --json              emit JSON instead of tables
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut backends = NnBackend::default_set();
    let mut workload_filter = "all".to_string();
    let mut mnist_scale = false;
    let mut rtt_ms: u64 = 0;
    let mut bandwidth_mbps: Option<f64> = None;
    let mut json = false;
    let mut competition = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--mnist-scale" => mnist_scale = true,
            "--competition" => competition = true,
            "--backends" => {
                i += 1;
                let list = args.get(i).ok_or("--backends needs a value")?;
                backends = list
                    .split(',')
                    .map(NnBackend::parse)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--workload" => {
                i += 1;
                workload_filter = args.get(i).ok_or("--workload needs a value")?.clone();
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
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
        i += 1;
    }

    if backends.contains(&NnBackend::FheTfhe) {
        eprintln!(
            "warning: fhe-tfhe is slow (see examples/mnist.rs: minutes for a 784-input dot). \
             Use --release, and prefer small sizes / --workload dot,matvec (skip relu/layer, \
             which add per-element PBS comparisons) unless you have time to spare."
        );
    }

    let mut profile = NetworkProfile::fixed_latency(Duration::from_millis(rtt_ms));
    if let Some(mbps) = bandwidth_mbps {
        profile = profile.with_bandwidth(mbps * 1_000_000.0 / 8.0);
    }

    let workloads: Vec<Workload> = all_workloads(mnist_scale)
        .into_iter()
        .filter(|w| {
            workload_filter == "all"
                || match w {
                    Workload::Dot(_) => workload_filter.contains("dot"),
                    Workload::MatVec { .. } => workload_filter.contains("matvec"),
                    Workload::Relu(_) => workload_filter.contains("relu"),
                    Workload::Layer { .. } => workload_filter.contains("layer"),
                }
        })
        .collect();

    // Some backends (CKKS keygen with rotation keys, TFHE) can take seconds
    // to minutes per row; Rust fully block-buffers stdout once it's not a
    // terminal (e.g. piped to a log file), so without an explicit flush a
    // long run would show *zero* progress until it finishes. Flush after
    // every row so `baru bench-nn > log.txt & tail -f log.txt` actually works.
    use std::io::Write;
    let mut all_reports = Vec::new();
    for w in &workloads {
        if !json {
            println!("\n── {}", w.label());
            let _ = std::io::stdout().flush();
        }
        let is_matvec_shaped = matches!(w, Workload::MatVec { .. } | Workload::Layer { .. });
        for &b in &backends {
            let reports = if competition && b == NnBackend::Mpc && is_matvec_shaped {
                // Leaderboard: every registered precompile, same workload,
                // same measurement — not just the champion.
                crate::precompile::mpc_matvec::registry()
                    .entries()
                    .map(|(meta, _)| run_mpc_as(*w, profile, Some(meta.id)))
                    .collect()
            } else {
                run_one(*w, b, profile)
            };
            for r in reports {
                if !json {
                    print_report(&r);
                    let _ = std::io::stdout().flush();
                }
                all_reports.push(r);
            }
        }
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&all_reports).map_err(|e| e.to_string())?
        );
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_is_always_finite_for_json_output() {
        assert!(throughput(1000, 0.0).is_finite());
        assert!(throughput(0, 0.0).is_finite());
        assert!(throughput(1000, 1.5).is_finite());
    }

    #[test]
    fn fr_signed_roundtrip_and_reduces_high_bits() {
        // Basic roundtrip.
        for &n in &[-1234i64, 0, 42, -1, i32::MIN as i64, i32::MAX as i64] {
            let fr = signed_to_fr(n, 32);
            assert_eq!(fr_to_signed(fr, 32), n, "roundtrip failed for {n}");
        }
        // The critical case this bug fix addresses: a raw revealed value with
        // bits set *above* nbits (as Beaver-triple-protocol reconstruction
        // actually produces) must still decode correctly by reducing first.
        let raw_with_high_bits: Fr = (1u128 << 100) | signed_to_fr(-7, 32);
        assert_eq!(fr_to_signed(raw_with_high_bits, 32), -7);
    }

    #[test]
    fn mpc_dot_matches_plaintext() {
        let r = run_mpc(Workload::Dot(6), NetworkProfile::INSTANT);
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.ok, Some(true));
        assert!(r.comm.unwrap().messages > 0);
    }

    #[test]
    fn mpc_matvec_matches_plaintext() {
        let r = run_mpc(
            Workload::MatVec {
                in_dim: 5,
                out_dim: 3,
            },
            NetworkProfile::INSTANT,
        );
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.ok, Some(true));
    }

    #[test]
    fn mpc_relu_handles_negative_and_positive() {
        let r = run_mpc(Workload::Relu(10), NetworkProfile::INSTANT);
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.ok, Some(true));
    }

    #[test]
    fn mpc_layer_matches_plaintext() {
        let r = run_mpc(
            Workload::Layer {
                in_dim: 6,
                out_dim: 4,
            },
            NetworkProfile::INSTANT,
        );
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.ok, Some(true));
    }

    #[test]
    fn mpc_message_count_grows_with_size() {
        let small = run_mpc(Workload::Dot(4), NetworkProfile::INSTANT);
        let large = run_mpc(Workload::Dot(64), NetworkProfile::INSTANT);
        assert!(large.comm.unwrap().messages > small.comm.unwrap().messages);
        assert!(large.comm.unwrap().bytes > small.comm.unwrap().bytes);
    }

    #[test]
    fn injected_latency_increases_measured_wall_time() {
        let fast = run_mpc(Workload::Dot(4), NetworkProfile::INSTANT);
        let slow = run_mpc(
            Workload::Dot(4),
            NetworkProfile::fixed_latency(Duration::from_millis(5)),
        );
        assert!(slow.total_micros > fast.total_micros);
    }

    #[test]
    fn fhe_mock_dot_matvec_relu_layer_match_plaintext() {
        for w in [
            Workload::Dot(8),
            Workload::MatVec {
                in_dim: 5,
                out_dim: 3,
            },
            Workload::Relu(8),
            Workload::Layer {
                in_dim: 5,
                out_dim: 3,
            },
        ] {
            let r = run_fhe("fhe-mock", &MockFheBackend, w, |_| {});
            assert!(r.error.is_none(), "{}: {:?}", w.label(), r.error);
            assert_eq!(r.ok, Some(true), "{}", w.label());
        }
    }

    #[test]
    fn ckks_mock_circuit_and_batched_match_plaintext() {
        for w in [
            Workload::Dot(8),
            Workload::MatVec {
                in_dim: 6,
                out_dim: 2,
            },
        ] {
            let reports = run_ckks("ckks-mock", &MockCkksBackend, |_| MockCkksBackend, w);
            assert_eq!(reports.len(), 2);
            for r in &reports {
                assert_eq!(
                    r.ok,
                    Some(true),
                    "{} / {}: {:?}",
                    w.label(),
                    r.backend,
                    r.error
                );
            }
        }
    }

    #[test]
    fn ckks_skips_comparison_workloads() {
        assert!(run_one(
            Workload::Relu(4),
            NnBackend::FheCkksMock,
            NetworkProfile::INSTANT
        )
        .is_empty());
    }

    #[test]
    fn backend_parse_roundtrip() {
        assert_eq!(NnBackend::parse("mpc"), Ok(NnBackend::Mpc));
        assert_eq!(NnBackend::parse("ckks"), Ok(NnBackend::FheCkks));
        assert!(NnBackend::parse("nonsense").is_err());
    }
}
