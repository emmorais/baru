//! `baru bench` — benchmarking harness that runs the same program on every
//! privacy backend and reports per-phase wall-clock timings.
//!
//! Backends:
//!   plain        CEK interpreter (correctness baseline, no privacy)
//!   mpc-aby3     replicated 3-party MPC, in-process simulation
//!   fhe-mock     FHE pipeline (compile_fhe → evaluate_fhe) on plain u64 "ciphertexts"
//!   fhe-tfhe     same pipeline on real TFHE ciphertexts (FheUint64)
//!   zkp-binius64 circuit compile → prove → verify (inputs as private witnesses)
//!
//! Every case is a single expression over named inputs.  The plain result is
//! the ground truth; each backend's decrypted / reconstructed result is
//! checked against it.  Cases whose control flow depends on secret values
//! carry an `mpc_expr` variant written in mux style (`(* x (> x 0))`), since
//! the MPC interpreter cannot branch on a shared condition.
//!
//! `baru bench-nn` (see [`nn`]) is the size-swept companion harness focused on
//! private neural-network inference: it drives `Backend`/`FheBackend` calls
//! directly (bypassing the Lisp interpreter, for precise cost attribution and
//! scalability to larger sizes) over dot-product / matrix-vector / ReLU /
//! full-layer workloads, and reports communication (message counts, bytes —
//! see [`comm`]) alongside timing, to characterize MPC's and FHE's
//! complementary computation/communication trade-off.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::fhe::circuit_emitter::evaluate_fhe;
use crate::fhe::{FheBackend, MockFheBackend, TfheBackend};
use crate::interpreter::{parse, Participant, Types};
use crate::zkp::binius64_backend::Binius64Backend;
use crate::zkp::circuit_compiler::CircuitCompiler;
use crate::zkp::ZKPBackend;

// ─── Cases ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BenchCase {
    pub name: &'static str,
    /// Core expression in terms of the input variables.
    pub expr: &'static str,
    /// MPC-compatible rewrite when `expr` branches on a secret (mux style).
    pub mpc_expr: Option<&'static str>,
    pub inputs: &'static [(&'static str, i64)],
}

const DOT2_SRC: &str = "(letrec ((dot (lambda (a b) \
     (if (null? a) 0 \
         (+ (* (car a) (car b)) (dot (cdr a) (cdr b))))))) \
   (dot (list x y) '(2 3)))";

pub const SUITE: &[BenchCase] = &[
    BenchCase {
        name: "add-mul",
        expr: "(+ (* x 3) (* y 7))",
        mpc_expr: None,
        inputs: &[("x", 12), ("y", 5)],
    },
    BenchCase {
        name: "relu",
        expr: "(if (> x 0) x 0)",
        mpc_expr: Some("(* x (> x 0))"),
        inputs: &[("x", -7)],
    },
    BenchCase {
        name: "max",
        expr: "(if (> x y) x y)",
        mpc_expr: Some("(let ((m (> x y))) (+ (* x m) (* y (- 1 m))))"),
        inputs: &[("x", 9), ("y", 4)],
    },
    BenchCase {
        name: "dot2",
        expr: DOT2_SRC,
        mpc_expr: None,
        inputs: &[("x", 4), ("y", 5)],
    },
    BenchCase {
        name: "dot4-builtin",
        expr: "(dot (list a b c d) '(2 3 4 5))",
        mpc_expr: None,
        inputs: &[("a", 1), ("b", 2), ("c", 3), ("d", -4)],
    },
];

// ─── Reports ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Phase {
    pub name: &'static str,
    pub micros: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendReport {
    pub backend: String,
    pub phases: Vec<Phase>,
    pub total_micros: u128,
    /// Numeric result where the backend produces one (ZKP only verifies).
    pub result: Option<i64>,
    /// Result (or proof) checked against the plain baseline.
    pub ok: Option<bool>,
    /// Backend-specific metrics, e.g. gates=42, proof-bytes=1234.
    pub notes: Vec<String>,
    /// Set when the backend panicked or is unsupported for this case.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseReport {
    pub case: String,
    pub expr: String,
    pub inputs: Vec<(String, i64)>,
    pub baseline: Option<i64>,
    pub backends: Vec<BackendReport>,
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

    fn total(&self) -> Duration {
        self.start.elapsed()
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn let_wrap(expr: &str, inputs: &[(String, i64)], bind: impl Fn(&str, i64) -> String) -> String {
    let bindings: Vec<String> = inputs.iter().map(|(k, v)| bind(k, *v)).collect();
    format!("(let ({}) {})", bindings.join(" "), expr)
}

fn eval_quiet_num(src: &str) -> Result<i64, String> {
    let expr = parse(src).map_err(|e| format!("parse error: {}", e))?;
    let p = Participant::new(0);
    match p.eval_quiet(&expr) {
        Types::Num(n) => Ok(n),
        other => Err(format!("expected Num, got {:?}", other)),
    }
}

fn fmt_duration(micros: u128) -> String {
    if micros >= 10_000_000 {
        format!("{:.1}s", micros as f64 / 1e6)
    } else if micros >= 10_000 {
        format!("{:.1}ms", micros as f64 / 1e3)
    } else {
        format!("{}µs", micros)
    }
}

/// Run `f`, converting a panic into an error report row.
fn guarded(backend: &str, f: impl FnOnce() -> BackendReport) -> BackendReport {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".to_string());
            BackendReport {
                backend: backend.to_string(),
                phases: vec![],
                total_micros: 0,
                result: None,
                ok: None,
                notes: vec![],
                error: Some(msg),
            }
        }
    }
}

// ─── Backend runners ──────────────────────────────────────────────────────────

fn run_plain(expr: &str, inputs: &[(String, i64)]) -> BackendReport {
    let src = let_wrap(expr, inputs, |k, v| format!("({} {})", k, v));
    let mut t = Timer::new();
    let result = t.phase("eval", || eval_quiet_num(&src));
    match result {
        Ok(n) => BackendReport {
            backend: "plain".into(),
            total_micros: t.total().as_micros(),
            phases: t.phases,
            result: Some(n),
            ok: None,
            notes: vec![],
            error: None,
        },
        Err(e) => BackendReport {
            backend: "plain".into(),
            total_micros: t.total().as_micros(),
            phases: t.phases,
            result: None,
            ok: None,
            notes: vec![],
            error: Some(e),
        },
    }
}

/// MPC shares use the REPL's 32-bit convention; results are sign-extended
/// through 32 bits the same way the REPL displays them.
fn run_mpc(expr: &str, inputs: &[(String, i64)], baseline: Option<i64>) -> BackendReport {
    let src = let_wrap(&format!("(reconstruct {})", expr), inputs, |k, v| {
        format!("({} (share {} 32))", k, v)
    });
    let mut t = Timer::new();
    let result = t.phase("eval", || eval_quiet_num(&src));
    match result {
        Ok(raw) => {
            let val = (raw as u32) as i32 as i64;
            BackendReport {
                backend: "mpc-aby3".into(),
                total_micros: t.total().as_micros(),
                phases: t.phases,
                result: Some(val),
                ok: baseline.map(|b| b == val),
                notes: vec!["3 parties (in-process)".into(), "32-bit shares".into()],
                error: None,
            }
        }
        Err(e) => BackendReport {
            backend: "mpc-aby3".into(),
            total_micros: t.total().as_micros(),
            phases: t.phases,
            result: None,
            ok: None,
            notes: vec![],
            error: Some(e),
        },
    }
}

/// Shared FHE pipeline: compile → keygen → encrypt → eval → decrypt.
/// `post_keygen` runs inside the keygen phase (TFHE installs the server key).
fn run_fhe<B: FheBackend>(
    label: &str,
    be: &B,
    expr: &str,
    inputs: &[(String, i64)],
    baseline: Option<i64>,
    post_keygen: impl Fn(&[u8]),
) -> BackendReport {
    let mut t = Timer::new();

    let parsed = parse(expr).unwrap_or_else(|e| panic!("parse error: {}", e));
    let circuit = t.phase("compile", || {
        let mut cc = CircuitCompiler::new();
        let mut circuit = cc
            .compile_fhe(&parsed)
            .unwrap_or_else(|e| panic!("compile_fhe: {}", e));
        // All bench inputs are encrypted; compile_fhe can leave letrec binders
        // (e.g. `dot`) in public_inputs, so clear the list rather than filter it.
        circuit.public_inputs = Vec::new();
        circuit.encrypted_inputs = inputs.iter().map(|(k, _)| k.clone()).collect();
        circuit
    });

    let ck = t.phase("keygen", || {
        let (ck, sk) = be.keygen();
        post_keygen(&sk);
        ck
    });

    let mut ct_bytes = 0usize;
    let enc = t.phase("encrypt", || {
        let mut enc: HashMap<String, B::Ct> = HashMap::new();
        for (k, v) in inputs {
            let ct = be.encrypt(&ck, *v as u64);
            if ct_bytes == 0 {
                ct_bytes = be.serialize_ct(&ct).len();
            }
            enc.insert(k.clone(), ct);
        }
        enc
    });

    let out = t.phase("eval", || evaluate_fhe(&circuit, &HashMap::new(), &enc, be));
    let result = t.phase("decrypt", || be.decrypt(&ck, &out) as i64);

    BackendReport {
        backend: label.to_string(),
        total_micros: t.total().as_micros(),
        phases: t.phases,
        result: Some(result),
        ok: baseline.map(|b| b == result),
        notes: vec![
            format!("gates={}", circuit.num_gates),
            format!("ct-bytes={}", ct_bytes),
        ],
        error: None,
    }
}

/// CKKS (real-arithmetic) FHE pipeline: compile in real mode → keygen →
/// encrypt → eval → decrypt-and-round.  Approximate arithmetic: integer
/// results round exactly for the supported (+,-,*) circuits, so the
/// baseline comparison stays exact; unsupported gates fail into an error
/// row via `guarded`, which is the honest outcome for comparison/mux cases.
fn run_fhe_real<B: crate::fhe::RealFheBackend>(
    label: &str,
    be_client: &B,
    mk_eval: impl FnOnce(&[u8]) -> B,
    expr: &str,
    inputs: &[(String, i64)],
    baseline: Option<i64>,
) -> BackendReport {
    use crate::fhe::circuit_emitter::evaluate_fhe_real;

    let mut t = Timer::new();
    let parsed = parse(expr).unwrap_or_else(|e| panic!("parse error: {}", e));
    let circuit = t.phase("compile", || {
        let mut cc = CircuitCompiler::new_real();
        let mut circuit = cc
            .compile_fhe(&parsed)
            .unwrap_or_else(|e| panic!("compile_fhe (real): {}", e));
        circuit.public_inputs = Vec::new();
        circuit.encrypted_inputs = inputs.iter().map(|(k, _)| k.clone()).collect();
        circuit
    });

    let (ck, be_eval) = t.phase("keygen", || {
        let (ck, sk) = be_client.keygen();
        (ck, mk_eval(&sk))
    });

    let mut ct_bytes = 0usize;
    let enc = t.phase("encrypt", || {
        let mut enc: HashMap<String, B::Ct> = HashMap::new();
        for (k, v) in inputs {
            let ct = be_client.encrypt(&ck, *v as f64);
            if ct_bytes == 0 {
                ct_bytes = be_client.serialize_ct(&ct).len();
            }
            enc.insert(k.clone(), ct);
        }
        enc
    });

    let out = t.phase("eval", || {
        evaluate_fhe_real(&circuit, &HashMap::new(), &enc, &be_eval)
    });
    let real = t.phase("decrypt", || be_client.decrypt(&ck, &out));
    let result = real.round() as i64;

    BackendReport {
        backend: label.to_string(),
        total_micros: t.total().as_micros(),
        phases: t.phases,
        result: Some(result),
        ok: baseline.map(|b| b == result),
        notes: vec![
            format!("gates={}", circuit.num_gates),
            format!("ct-bytes={}", ct_bytes),
            format!("approx result={:.6}", real),
        ],
        error: None,
    }
}

fn run_zkp(expr: &str, inputs: &[(String, i64)]) -> BackendReport {
    let mut t = Timer::new();
    let be = Binius64Backend::new();
    let parsed = parse(expr).unwrap_or_else(|e| panic!("parse error: {}", e));
    let private_vars: Vec<String> = inputs.iter().map(|(k, _)| k.clone()).collect();

    let circuit = t.phase("compile", || {
        be.compile_circuit(&parsed, &private_vars)
            .unwrap_or_else(|e| panic!("compile: {}", e))
    });

    let private_values: HashMap<String, i64> =
        inputs.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let public_values: HashMap<String, i64> = HashMap::new();

    let proof = t.phase("prove", || {
        be.prove(&circuit, &private_values, &public_values)
            .unwrap_or_else(|e| panic!("prove: {}", e))
    });

    let valid = t.phase("verify", || {
        be.verify(&circuit, &proof, &public_values)
            .unwrap_or_else(|e| panic!("verify: {}", e))
    });

    BackendReport {
        backend: "zkp-binius64".into(),
        total_micros: t.total().as_micros(),
        phases: t.phases,
        result: None,
        ok: Some(valid),
        notes: vec![
            format!("gates={}", circuit.metadata.num_gates),
            format!("constraints={}", circuit.metadata.num_constraints),
            format!("proof-bytes={}", proof.proof_data.len()),
        ],
        error: None,
    }
}

// ─── Orchestration ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Backend {
    Plain,
    Mpc,
    FheMock,
    FheTfhe,
    FheCkks,
    FheCkksMock,
    Zkp,
}

impl Backend {
    pub fn parse(s: &str) -> Result<Backend, String> {
        match s {
            "plain" => Ok(Backend::Plain),
            "mpc" | "mpc-aby3" => Ok(Backend::Mpc),
            "fhe-mock" | "mock" => Ok(Backend::FheMock),
            "fhe-tfhe" | "tfhe" | "fhe" => Ok(Backend::FheTfhe),
            "fhe-ckks" | "ckks" => Ok(Backend::FheCkks),
            "fhe-ckks-mock" | "ckks-mock" => Ok(Backend::FheCkksMock),
            "zkp" | "zkp-binius64" | "binius64" => Ok(Backend::Zkp),
            other => Err(format!(
                "unknown backend '{}' (expected plain, mpc, fhe-mock, fhe-tfhe, fhe-ckks, \
                 fhe-ckks-mock, zkp)",
                other
            )),
        }
    }

    pub fn all() -> Vec<Backend> {
        vec![
            Backend::Plain,
            Backend::Mpc,
            Backend::FheMock,
            Backend::FheTfhe,
            Backend::FheCkks,
            Backend::Zkp,
        ]
    }
}

/// Average phase timings across `iters` runs of `run`, keeping the last
/// run's result / notes / error.
fn averaged(iters: u32, run: impl Fn() -> BackendReport) -> BackendReport {
    assert!(iters >= 1);
    let mut last = run();
    if iters == 1 || last.error.is_some() {
        return last;
    }
    let mut phase_sums: Vec<u128> = last.phases.iter().map(|p| p.micros).collect();
    let mut total_sum = last.total_micros;
    for _ in 1..iters {
        last = run();
        if last.error.is_some() {
            return last;
        }
        for (i, p) in last.phases.iter().enumerate() {
            phase_sums[i] += p.micros;
        }
        total_sum += last.total_micros;
    }
    for (i, p) in last.phases.iter_mut().enumerate() {
        p.micros = phase_sums[i] / iters as u128;
    }
    last.total_micros = total_sum / iters as u128;
    last
}

pub fn run_case(case: &BenchCase, backends: &[Backend], iters: u32) -> CaseReport {
    let inputs: Vec<(String, i64)> = case
        .inputs
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

    // Plain baseline always runs (it is the ground truth), but is only
    // reported when requested.
    let plain = guarded("plain", || {
        averaged(iters, || run_plain(case.expr, &inputs))
    });
    let baseline = plain.result;

    let mut reports = Vec::new();
    for b in backends {
        let report = match b {
            Backend::Plain => plain.clone(),
            Backend::Mpc => {
                let expr = case.mpc_expr.unwrap_or(case.expr);
                guarded("mpc-aby3", || {
                    averaged(iters, || run_mpc(expr, &inputs, baseline))
                })
            }
            Backend::FheMock => guarded("fhe-mock", || {
                averaged(iters, || {
                    run_fhe(
                        "fhe-mock",
                        &MockFheBackend,
                        case.expr,
                        &inputs,
                        baseline,
                        |_| {},
                    )
                })
            }),
            Backend::FheTfhe => guarded("fhe-tfhe", || {
                averaged(iters, || {
                    let be = TfheBackend::new();
                    run_fhe("fhe-tfhe", &be, case.expr, &inputs, baseline, |sk| {
                        TfheBackend::install_server_key(sk)
                    })
                })
            }),
            Backend::FheCkks => guarded("fhe-ckks", || {
                averaged(iters, || {
                    run_fhe_real(
                        "fhe-ckks",
                        &crate::fhe::CkksBackend::new(),
                        crate::fhe::CkksBackend::with_server_keys,
                        case.expr,
                        &inputs,
                        baseline,
                    )
                })
            }),
            Backend::FheCkksMock => guarded("fhe-ckks-mock", || {
                averaged(iters, || {
                    run_fhe_real(
                        "fhe-ckks-mock",
                        &crate::fhe::MockCkksBackend,
                        |_| crate::fhe::MockCkksBackend,
                        case.expr,
                        &inputs,
                        baseline,
                    )
                })
            }),
            Backend::Zkp => guarded("zkp-binius64", || {
                averaged(iters, || run_zkp(case.expr, &inputs))
            }),
        };
        reports.push(report);
    }

    CaseReport {
        case: case.name.to_string(),
        expr: case.expr.to_string(),
        inputs,
        baseline,
        backends: reports,
    }
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

const USAGE: &str = "usage: baru bench [options] [case ...]

Runs each case on every selected backend and reports per-phase timings.
With no case names, the built-in suite runs (use --list to see it).

options:
  --backends LIST   comma-separated: plain,mpc,fhe-mock,fhe-tfhe,fhe-ckks,fhe-ckks-mock,zkp
                    (default: all except fhe-ckks-mock)
  --iters N         average timings over N runs (default 1)
  --expr SRC        benchmark a custom expression instead of the suite
  --mpc-expr SRC    MPC-compatible rewrite of --expr (mux style, no secret ifs)
  --input NAME=VAL  input binding for --expr (repeatable)
  --json            emit reports as JSON instead of tables
  --list            list built-in cases and exit
  --store SPEC      persist every result: local:PATH or barudb:PATH
                    (see docs/barudb-api.md; `baru db query` reads it back)
  --tag NAME        tag stored records, e.g. --tag paper-table-3 (repeatable)
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut backends = Backend::all();
    let mut iters: u32 = 1;
    let mut json = false;
    let mut custom_expr: Option<String> = None;
    let mut custom_mpc_expr: Option<String> = None;
    let mut custom_inputs: Vec<(String, i64)> = Vec::new();
    let mut case_filter: Vec<String> = Vec::new();
    let mut store_spec: Option<String> = None;
    let mut tags: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{}", USAGE);
                return Ok(());
            }
            "--list" => {
                for c in SUITE {
                    let ins: Vec<String> = c
                        .inputs
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect();
                    println!("{:<10} {}   [{}]", c.name, c.expr, ins.join(" "));
                }
                return Ok(());
            }
            "--json" => json = true,
            "--backends" => {
                i += 1;
                let list = args.get(i).ok_or("--backends needs a value")?;
                backends = list
                    .split(',')
                    .map(Backend::parse)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--iters" => {
                i += 1;
                iters = args
                    .get(i)
                    .ok_or("--iters needs a value")?
                    .parse()
                    .map_err(|_| "--iters needs a positive integer".to_string())?;
                if iters == 0 {
                    return Err("--iters must be >= 1".into());
                }
            }
            "--expr" => {
                i += 1;
                custom_expr = Some(args.get(i).ok_or("--expr needs a value")?.clone());
            }
            "--mpc-expr" => {
                i += 1;
                custom_mpc_expr = Some(args.get(i).ok_or("--mpc-expr needs a value")?.clone());
            }
            "--input" => {
                i += 1;
                let kv = args.get(i).ok_or("--input needs NAME=VAL")?;
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("--input '{}' is not NAME=VAL", kv))?;
                let v: i64 = v
                    .parse()
                    .map_err(|_| format!("--input '{}': value is not an integer", kv))?;
                custom_inputs.push((k.to_string(), v));
            }
            "--store" => {
                i += 1;
                store_spec = Some(args.get(i).ok_or("--store needs a value")?.clone());
            }
            "--tag" => {
                i += 1;
                tags.push(args.get(i).ok_or("--tag needs a value")?.clone());
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option '{}'\n{}", other, USAGE));
            }
            case_name => case_filter.push(case_name.to_string()),
        }
        i += 1;
    }

    if cfg!(debug_assertions) && backends.contains(&Backend::FheTfhe) {
        eprintln!(
            "warning: debug build — TFHE is 10-50x slower than release. \
             Use `cargo run --release -- bench`."
        );
    }

    // Custom expression: leak the strings to fit the &'static BenchCase shape
    // (a one-shot CLI process, so the leak is harmless).
    let cases: Vec<BenchCase> = if let Some(expr) = custom_expr {
        if custom_inputs.is_empty() {
            return Err("--expr requires at least one --input NAME=VAL".into());
        }
        let inputs: &'static [(&'static str, i64)] = Box::leak(
            custom_inputs
                .iter()
                .map(|(k, v)| (&*Box::leak(k.clone().into_boxed_str()), *v))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        vec![BenchCase {
            name: "custom",
            expr: Box::leak(expr.into_boxed_str()),
            mpc_expr: custom_mpc_expr.map(|s| &*Box::leak(s.into_boxed_str())),
            inputs,
        }]
    } else {
        let mut cases: Vec<BenchCase> = SUITE.to_vec();
        if !case_filter.is_empty() {
            cases.retain(|c| case_filter.iter().any(|f| f == c.name));
            if cases.is_empty() {
                return Err(format!(
                    "no cases match {:?} (use --list to see available cases)",
                    case_filter
                ));
            }
        }
        cases
    };

    let mut reports = Vec::new();
    for case in &cases {
        if !json {
            let ins: Vec<String> = case
                .inputs
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            println!("\n── {}: {}", case.name, case.expr);
            println!("   inputs: {}", ins.join(" "));
        }
        let report = run_case(case, &backends, iters);
        if !json {
            print_table(&report);
        }
        reports.push(report);
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?
        );
    }
    if let Some(spec) = &store_spec {
        persist(&reports, spec, &tags, iters)?;
    }
    Ok(())
}

/// Write one [`crate::db::Run`] per (case, backend) into the experiment store.
///
/// Failed backends are recorded too, with the error in `notes`: "TFHE ran out
/// of memory on this shape" is a result worth keeping, and silently dropping it
/// would make the store's coverage look better than it is.
fn persist(
    reports: &[CaseReport],
    spec: &str,
    tags: &[String],
    iters: u32,
) -> Result<(), String> {
    use crate::db::{Metric, Run};

    let mut store = crate::db::open(spec)?;
    let mut written = 0usize;
    for report in reports {
        for b in &report.backends {
            let mut run = Run::new("bench", &report.case, &b.backend);
            run.ok = b.ok;
            run.tags = tags.to_vec();
            for phase in &b.phases {
                run.metrics.insert(
                    format!("{}_us", phase.name),
                    Metric::new(phase.micros as f64, "us"),
                );
            }
            run.metrics
                .insert("total_us".into(), Metric::new(b.total_micros as f64, "us"));
            run.config.insert("expr".into(), report.expr.clone());
            run.config.insert("iters".into(), iters.to_string());
            for (name, value) in &report.inputs {
                run.config.insert(format!("input.{}", name), value.to_string());
            }
            if let Some(r) = b.result {
                run.config.insert("result".into(), r.to_string());
            }
            if let Some(base) = report.baseline {
                run.config.insert("baseline".into(), base.to_string());
            }
            let mut notes = b.notes.join("; ");
            if let Some(err) = &b.error {
                if !notes.is_empty() {
                    notes.push_str("; ");
                }
                notes.push_str(&format!("error: {}", err));
            }
            run.notes = notes;
            store.put(&run)?;
            written += 1;
        }
    }
    eprintln!("stored {} records in {}", written, store.describe());
    Ok(())
}

fn print_table(report: &CaseReport) {
    println!(
        "   {:<13} {:>10}  {:>8}  {:<4} breakdown",
        "backend", "total", "result", "ok"
    );
    for b in &report.backends {
        if let Some(err) = &b.error {
            println!(
                "   {:<13} {:>10}  {:>8}  {:<4} error: {}",
                b.backend, "-", "-", "-", err
            );
            continue;
        }
        let result = b
            .result
            .map(|r| r.to_string())
            .unwrap_or_else(|| "-".into());
        let ok = match b.ok {
            Some(true) => "✓",
            Some(false) => "✗",
            None => "-",
        };
        let mut breakdown: Vec<String> = b
            .phases
            .iter()
            .map(|p| format!("{}={}", p.name, fmt_duration(p.micros)))
            .collect();
        if !b.notes.is_empty() {
            breakdown.push(format!("[{}]", b.notes.join(" ")));
        }
        println!(
            "   {:<13} {:>10}  {:>8}  {:<4} {}",
            b.backend,
            fmt_duration(b.total_micros),
            result,
            ok,
            breakdown.join(" ")
        );
    }
}

// ─── Tests (fast backends only — TFHE is exercised by `baru bench` itself) ────

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_backends() -> Vec<Backend> {
        vec![Backend::Plain, Backend::Mpc, Backend::FheMock, Backend::Zkp]
    }

    #[test]
    fn suite_runs_on_fast_backends() {
        for case in SUITE {
            let report = run_case(case, &fast_backends(), 1);
            assert!(report.baseline.is_some(), "case {}: no baseline", case.name);
            for b in &report.backends {
                assert!(
                    b.error.is_none(),
                    "case {} backend {}: {:?}",
                    case.name,
                    b.backend,
                    b.error
                );
                if b.backend != "plain" {
                    assert_eq!(
                        b.ok,
                        Some(true),
                        "case {} backend {}: result {:?} != baseline {:?}",
                        case.name,
                        b.backend,
                        b.result,
                        report.baseline
                    );
                }
            }
        }
    }

    #[test]
    fn averaging_keeps_phase_structure() {
        let inputs = vec![("x".to_string(), 3i64)];
        let r = averaged(3, || run_plain("(+ x 1)", &inputs));
        assert_eq!(r.result, Some(4));
        assert_eq!(r.phases.len(), 1);
    }

    #[test]
    fn unknown_backend_is_rejected() {
        assert!(Backend::parse("plonky3").is_err());
        assert_eq!(Backend::parse("tfhe"), Ok(Backend::FheTfhe));
    }
}

pub mod abstraction;
pub mod backend_aware;
pub mod circuit;
pub mod compiled_nn;
pub mod ckks_compare;
pub mod cohort;
pub mod cohort_eval;
pub mod cnn;
pub mod gc_compare;
pub mod deep_nn;
pub mod layer;
pub mod medmnist;
pub mod mixed_layer;
pub mod net;
pub mod ot;
pub mod pir;
pub mod psi;
pub mod mnist;
pub mod fhe_innerprod;
pub mod comm;
pub mod hybrid_nn;
pub mod nn;
pub mod train;
