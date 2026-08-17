//! `baru bench-heir` — HEIR as a **measured FHE baseline**, with a security gate.
//!
//! # Why HEIR is a baseline and not a target
//!
//! [HEIR](https://github.com/google/heir) is compiler infrastructure. Its paper
//! (arXiv:2508.11095) claims generality — "support all mainstream techniques in
//! homomorphic encryption, integrate with all major software libraries and
//! hardware accelerators" — and reports no private-ML performance evaluation.
//! Beating its numbers would not be a result.
//!
//! What it is good for is supplying the FHE arm of a comparison on **tuned C++**
//! (OpenFHE, Lattigo) instead of on Baru's own hand-written NTT. `docs/survey.md`
//! §1 concedes that "absolute figures here are not competitive with tuned
//! implementations"; this removes that confound in the one direction that
//! matters, because the objection to "MPC beat our FHE" is that our FHE was
//! ours. It also replaces `bench-layer`'s FHE column, which is not an FHE run at
//! all but a per-operation constant from `mpcc::fhe_cost` extrapolated through a
//! network profile.
//!
//! # The security gate, which is the point of this module
//!
//! `crypto::attestations` makes every Baru engine state what it implements, and
//! refuses a session that asks for more. This extends the same discipline to an
//! **external** baseline, because the trap is documented and large:
//! [google/heir#1232](https://github.com/google/heir/issues/1232) reports an
//! MNIST MLP at ~30 s and 3 GB with `HEStd_NotSet` — OpenFHE's "do not check"
//! mode, i.e. no security level at all — and **~6 min and 26 GB** at the secure
//! ring dimension. A 12× factor sits between the insecure and the secure
//! baseline, and it is the insecure one that a casual run produces.
//!
//! So [`HeirReport::gate`] refuses to let a figure be quoted unless the run
//! reports a level in [`SECURE_LEVELS`]. **An insecure baseline is worse than no
//! baseline**: it manufactures a speedup out of a parameter choice.
//!
//! # The second gate, and the harder problem it exposes
//!
//! **CKKS cannot compare, so an exact ReLU is not expressible on the FHE arm at
//! all.** Baru's MPC arm computes ReLU exactly. So the two arms cannot be made to
//! compute the same function by tuning a parameter — a choice has to be made, and
//! the two available choices measure different things:
//!
//! * **Give both arms `x²`.** Exact on each side, so the comparison is clean, and
//!   it is what Orion (ASPLOS'25) does for its MNIST networks. **This is the route
//!   taken**, and measurement settled a worry about it: `x²` costs the MPC arm
//!   *exactly* what ReLU costs — identical rounds, multiplications,
//!   decompositions and bytes (`the_fhe_friendly_activation_costs_the_mpc_arm_nothing`).
//!   Over plain integers it would be one multiplication against ReLU's bit
//!   decomposition, but in **fixed point** squaring doubles the scale, so the
//!   product needs a rescale — and a rescale is the same decomposition the
//!   comparison was. So switching does not flatter the MPC arm, and every finding
//!   measured with ReLU carries over unchanged.
//! * **Keep the MPC arm's exact ReLU and approximate it on the FHE arm** with a
//!   minimax polynomial. Then the degree has to be justified and the accuracy it
//!   costs reported — Orion measures 2.1% accuracy for 1.77× speed on that axis.
//!   Not implemented; `x²` made it unnecessary.
//!
//! [`Refusal::DifferentNetwork`] refuses to pick, and is checked **before** every
//! other condition: "the two arms computed different functions" is a stronger
//! objection than any disagreement in their outputs, and a class disagreement is
//! exactly how it would otherwise present — misread as approximation error.
//!
//! Beneath that, [`Refusal::Disagrees`] still requires the FHE arm to reproduce
//! the plaintext classification, because timing an approximation that gets the
//! answer wrong is not a comparison.
//!
//! # Process boundary, and why nothing here is a Cargo dependency
//!
//! HEIR is a Bazel/MLIR/C++ toolchain with a Python frontend. Baru drives it as a
//! subprocess over a JSON contract, the same shape as [`crate::db::barudb`]:
//! `BARU_HEIR_PY` overrides the script path, absence is reported with an
//! actionable message rather than a panic, and the tests probe-and-skip so CI
//! stays hermetic (CI installs no Python).
//!
//! # Status, and the real blocker
//!
//! **No FHE numbers have been measured yet**, and the reason is worth recording
//! because it is not the obvious one.
//!
//! `heir-py` 2026.8.1 installs and imports fine (Python >= 3.10 — it ships
//! `cp310-abi3` wheels only, so a 3.9 interpreter fails at `pip` time). The
//! signature was verified by introspection and matches what this module targets:
//!
//! ```text
//! compile(mlir_str=None, scheme="bgv", backend=None, config=None,
//!         debug=False, heir_opt_options=None)
//! ```
//!
//! **But the wheel cannot evaluate anything on its own.** `heir.compile()` calls
//! `openfhe_config.resolve_config()`, which looks for an OpenFHE **C++ install**
//! — headers under `/usr/local/include/openfhe` or `/usr/include/openfhe`, or the
//! `OPENFHE_INCLUDE_DIR` / `OPENFHE_LIB_DIR` / `OPENFHE_LINK_LIBS` environment
//! variables — and otherwise falls through to a *development* config that needs
//! HEIR itself built with Bazel, failing with `"Could not build development
//! config. Did you run bazel?"`. HEIR emits C++ and compiles it per program, so
//! the headers are a hard requirement, not a convenience. The PyPI `openfhe`
//! wheel does not satisfy it: it is Python bindings, not an installed toolchain.
//!
//! There is also a `heir.backends.cleartext` backend, and it is **not** a
//! shortcut: measured here, `eval` simply forwards to the original Python
//! function and prints a warning saying so. It exercises none of the compiler.
//!
//! So a measurement needs one of: OpenFHE built from source and installed (then
//! the `OPENFHE_*` variables), or a Bazel build of HEIR (~30 min clean, per its
//! README).
//!
//! What *is* tested: the request/report contract through a real subprocess (the
//! script's `--self-test` mode), the security gate firing on `HEStd_NotSet`, the
//! class-agreement gate on a mismatch, the different-activation gate, and the
//! actionable error when the script is absent. Until a secure run exists, **no
//! number from here may be quoted against `bench-2pc-nn`**.

use std::io::Write;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::ml::{Activation, QuantizedNetwork};

/// Security levels that count as secure. Anything else — notably OpenFHE's
/// `HEStd_NotSet` — is refused by [`HeirReport::gate`].
pub const SECURE_LEVELS: [&str; 3] = [
    "HEStd_128_classic",
    "HEStd_192_classic",
    "HEStd_256_classic",
];

/// Default location of the driver script, overridable with `BARU_HEIR_PY`.
const DEFAULT_SCRIPT: &str = "tools/heir_baseline.py";

// ─── The contract ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct HeirLayer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub activation: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HeirRequest {
    pub scheme: String,
    pub backend: String,
    pub security_level: String,
    /// How the FHE arm handles a nonlinearity. **Not a free parameter** — see
    /// [`Refusal::DifferentNetwork`]: CKKS cannot compute an exact ReLU, so this
    /// choice decides whether the two arms compute the same function at all.
    pub activation: String,
    pub repeats: u32,
    pub layers: Vec<HeirLayer>,
    pub inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HeirReport {
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
    /// True when the script evaluated in plaintext rather than under FHE. Such a
    /// report is never a measurement and [`HeirReport::gate`] rejects it.
    #[serde(default)]
    pub simulated: bool,
    #[serde(default)]
    pub heir_version: String,
    #[serde(default)]
    pub scheme: String,
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub ring_dim: u64,
    #[serde(default)]
    pub security_level: String,
    /// The activation the FHE arm actually used, echoed back so
    /// [`HeirReport::gate_with`] can check both arms computed the same function.
    #[serde(default)]
    pub activation: String,
    #[serde(default)]
    pub activation_degree: u32,
    #[serde(default)]
    pub keygen_ms: f64,
    #[serde(default)]
    pub encrypt_ms: f64,
    /// Homomorphic evaluation only — key generation is excluded, which E45
    /// records getting wrong the other way (keygen inside a timed region made
    /// FHE look 37× slower than it was).
    #[serde(default)]
    pub eval_ms: f64,
    #[serde(default)]
    pub decrypt_ms: f64,
    #[serde(default)]
    pub ciphertext_bytes: u64,
    #[serde(default)]
    pub classes: Vec<i64>,
    #[serde(default)]
    pub notes: String,
}

/// Why a report cannot be quoted. Separate from a transport error: the tool ran
/// and answered, and the answer is not usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The script reported a failure of its own.
    Failed(String),
    /// Plaintext evaluation, not FHE.
    Simulated,
    /// The parameters are not secure.
    Insecure(String),
    /// The FHE arm computed a different classification.
    Disagrees { got: Vec<i64>, want: Vec<i64> },
    /// The two arms are not computing the same function at all, so no
    /// classification agreement could make the comparison valid.
    DifferentNetwork { mpc: String, fhe: String },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Failed(e) => write!(f, "the HEIR driver failed: {e}"),
            Refusal::Simulated => write!(
                f,
                "the driver evaluated in plaintext (--self-test), which exercises the \
                 contract and measures nothing"
            ),
            Refusal::Insecure(level) => write!(
                f,
                "REFUSING to quote a latency at security level `{level}` — only {:?} \
                 count as secure. google/heir#1232 measured an MNIST MLP at ~30 s with \
                 HEStd_NotSet against ~6 min at the secure ring dimension, so an \
                 insecure baseline manufactures a 12x speedup out of a parameter \
                 choice. An insecure baseline is worse than no baseline.",
                SECURE_LEVELS
            ),
            Refusal::Disagrees { got, want } => write!(
                f,
                "the FHE arm classified {got:?} against the plaintext reference's \
                 {want:?}. CKKS approximates ReLU by a polynomial, so a disagreement \
                 means the degree is too low — raise `--degree`. Timing an \
                 approximation that gets the answer wrong against an exact protocol is \
                 not a comparison."
            ),
            Refusal::DifferentNetwork { mpc, fhe } => write!(
                f,
                "the two arms compute DIFFERENT FUNCTIONS: the MPC arm's activation is \
                 `{mpc}` and the FHE arm's is `{fhe}`. CKKS cannot compare, so an exact \
                 ReLU is not expressible on the FHE arm at all — which leaves two \
                 honest options, and this refuses to pick between them because they \
                 measure different things. (a) Give BOTH arms `x^2`: exact on each \
                 side, what Orion does for its MNIST networks, and it removes the \
                 activation's bit decomposition from the MPC arm entirely — so it also \
                 changes E48's finding rather than just enabling a comparison. \
                 (b) Keep the MPC arm's exact ReLU and let the FHE arm approximate it \
                 with a minimax polynomial, then justify the degree and report the \
                 accuracy it costs. Neither is a default."
            ),
        }
    }
}

impl HeirReport {
    /// Every condition that must hold before a timing here may be quoted.
    ///
    /// Deliberately one function rather than checks scattered across the caller:
    /// a reader should be able to see the whole list, and a future caller should
    /// not be able to forget one.
    pub fn gate(&self, expected: &[i64]) -> Result<(), Refusal> {
        self.gate_with(expected, None)
    }

    /// [`Self::gate`], additionally checking that the FHE arm's activation
    /// matches the MPC arm's.
    ///
    /// Checked *before* everything else, because "the two arms computed different
    /// functions" is a stronger objection than any disagreement in their output —
    /// and because a class disagreement is exactly how it would otherwise
    /// present, misread as approximation error.
    pub fn gate_with(&self, expected: &[i64], mpc_activation: Option<&str>) -> Result<(), Refusal> {
        if let (Some(mpc), fhe) = (mpc_activation, self.activation.as_str()) {
            if !fhe.is_empty() && mpc != fhe {
                return Err(Refusal::DifferentNetwork {
                    mpc: mpc.to_string(),
                    fhe: fhe.to_string(),
                });
            }
        }
        if !self.ok {
            return Err(Refusal::Failed(
                self.error.clone().unwrap_or_else(|| "unspecified".into()),
            ));
        }
        if self.simulated {
            return Err(Refusal::Simulated);
        }
        if !SECURE_LEVELS.contains(&self.security_level.as_str()) {
            return Err(Refusal::Insecure(self.security_level.clone()));
        }
        if self.classes != expected {
            return Err(Refusal::Disagrees {
                got: self.classes.clone(),
                want: expected.to_vec(),
            });
        }
        Ok(())
    }

    /// End-to-end latency per inference, excluding key generation — which is a
    /// per-session cost, not a per-query one, and which E45 records having
    /// wrongly charged to the query.
    pub fn latency_ms(&self) -> f64 {
        self.encrypt_ms + self.eval_ms + self.decrypt_ms
    }
}

// ─── Driving the tool ─────────────────────────────────────────────────────────

fn script_path() -> String {
    std::env::var("BARU_HEIR_PY").unwrap_or_else(|_| DEFAULT_SCRIPT.to_string())
}

/// Is the driver script present and runnable? Used by tests to skip rather than
/// fail, the `bench::cohort` convention — a suite that is always red stops being
/// read.
pub fn available() -> bool {
    std::path::Path::new(&script_path()).is_file()
}

/// Turn a quantized network into the float network the FHE arm evaluates.
///
/// The weights are divided by the fixed-point scale, so both arms compute the
/// same mathematical function; they differ in *how* — exact fixed point with an
/// exact ReLU against approximate reals with a polynomial ReLU. That difference
/// is the comparison, not a nuisance, and it is why the gate checks classes.
pub fn request_for(
    net: &QuantizedNetwork,
    inputs: &[Vec<i64>],
    scheme: &str,
    backend: &str,
    activation: &str,
    repeats: u32,
) -> HeirRequest {
    let scale = (1i64 << net.scale_bits) as f64;
    HeirRequest {
        scheme: scheme.to_string(),
        backend: backend.to_string(),
        security_level: SECURE_LEVELS[0].to_string(),
        activation: activation.to_string(),
        repeats,
        layers: net
            .layers
            .iter()
            .map(|l| HeirLayer {
                weights: l
                    .weights
                    .iter()
                    .map(|row| row.iter().map(|&w| w as f64 / scale).collect())
                    .collect(),
                // A bias lives at output scale in `QuantizedNetwork::forward`, so
                // it is divided once, not twice. Getting this wrong would make
                // the FHE arm compute a different network and the class gate
                // would catch it — but silently, as "approximation error".
                bias: l.bias.iter().map(|&b| b as f64 / scale).collect(),
                activation: match l.activation {
                    Activation::Relu => "relu".into(),
                    Activation::Identity => "identity".into(),
                    Activation::Square => "square".into(),
                },
            })
            .collect(),
        inputs: inputs
            .iter()
            .map(|x| x.iter().map(|&v| v as f64 / scale).collect())
            .collect(),
    }
}

/// Spawn the driver and parse its report. `extra` is passed through to the
/// script (`--self-test`, notably).
pub fn invoke(req: &HeirRequest, extra: &[&str]) -> Result<HeirReport, String> {
    invoke_at(&script_path(), req, extra)
}

/// [`invoke`] against an explicit script path.
///
/// Exists so a test can exercise the absent-driver path without mutating
/// `BARU_HEIR_PY` — a process-global that would race any concurrently running
/// test that reads it.
pub fn invoke_at(script: &str, req: &HeirRequest, extra: &[&str]) -> Result<HeirReport, String> {
    if !std::path::Path::new(&script).is_file() {
        return Err(format!(
            "cannot find the HEIR driver at `{script}`. Point BARU_HEIR_PY at \
             `tools/heir_baseline.py`, or run this command from the repository root. \
             The request/report contract is documented in that script's header."
        ));
    }
    let body = serde_json::to_vec(req).map_err(|e| e.to_string())?;

    let mut child = Command::new("python3")
        .arg(script)
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            format!(
                "cannot run `python3 {script}`: {e}\nHEIR is optional; `bench-2pc-nn` \
                 measures the MPC side without it. For the FHE arm you need Python \
                 >= 3.10 and `pip install \"heir-py[python,openfhe]\"` — heir-py \
                 publishes cp310-abi3 wheels only."
            )
        })?;
    child
        .stdin
        .as_mut()
        .ok_or("no stdin on the HEIR driver")?
        .write_all(&body)
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;

    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str::<HeirReport>(text.trim()).map_err(|e| {
        format!(
            "could not parse the HEIR driver's report: {e}\nstdout: {}\nstderr: {}",
            text.trim(),
            String::from_utf8_lossy(&out.stderr).trim()
        )
    })
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

const USAGE: &str = "usage: baru bench-heir [options]

  --shape A            network shape, `x`-separated dims (default 16x8x4)
  --scheme S           FHE scheme (default ckks)
  --backend B          HEIR exit backend (default openfhe)
  --activation A       how the FHE arm handles a nonlinearity: `square`
                       (both arms compute x^2, exact on each) or `relu_poly`
                       (FHE approximates the MPC arm's exact ReLU). There is
                       NO safe default — see the module docs
  --repeats N          timed repeats per input (default 3)
  --inputs N           inputs to classify (default 4)
  --scale N            fixed-point scale bits (default 8)
  --self-test          evaluate in plaintext: exercises the contract, measures
                       nothing, and is REFUSED as a quotable figure
  --json               emit the report as JSON
  -h, --help           this message

Drives HEIR (github.com/google/heir) as the FHE arm of the matched-threat-model
comparison, against the same network `bench-2pc-nn` runs under 2-party MPC.

Needs Python >= 3.10 and `pip install \"heir-py[python,openfhe]\"`.
A latency is REFUSED unless the run reports a real security level and reproduces
the plaintext classification — see this module's docs for why both matter.";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut shape = vec![16usize, 8, 4];
    let mut scheme = "ckks".to_string();
    let mut backend = "openfhe".to_string();
    let mut activation = "square".to_string();
    let mut repeats = 3u32;
    let mut n_inputs = 4usize;
    let mut scale_bits = 8u32;
    let mut self_test = false;
    let mut json = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--json" => json = true,
            "--self-test" => self_test = true,
            "--shape" => {
                let v = it
                    .next()
                    .ok_or_else(|| format!("--shape needs a value\n\n{USAGE}"))?;
                shape = v
                    .split('x')
                    .map(|d| {
                        d.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad dimension `{d}` in `{v}`"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--scheme" => scheme = it.next().ok_or("--scheme needs a value")?.clone(),
            "--backend" => backend = it.next().ok_or("--backend needs a value")?.clone(),
            "--activation" => activation = it.next().ok_or("--activation needs a value")?.clone(),
            "--repeats" => {
                repeats = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--repeats needs a number")?
            }
            "--inputs" => {
                n_inputs = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--inputs needs a number")?
            }
            "--scale" => {
                scale_bits = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--scale needs a number")?
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }
    if shape.len() < 2 {
        return Err(format!(
            "shape needs at least an input and an output\n\n{USAGE}"
        ));
    }

    // The same network and the same inputs `bench-2pc-nn` measures, so the two
    // arms are comparable by construction rather than by coincidence.
    let net = super::two_party_nn::fixture_for(&shape, scale_bits);
    let inputs: Vec<Vec<i64>> = (0..n_inputs)
        .map(|k| super::two_party_nn::input_for(shape[0], scale_bits, k))
        .collect();
    let expected: Vec<i64> = inputs
        .iter()
        .map(|x| super::two_party_nn::argmax_of(&net.forward(x)))
        .collect();

    let req = request_for(&net, &inputs, &scheme, &backend, &activation, repeats);
    let extra: Vec<&str> = if self_test {
        vec!["--self-test"]
    } else {
        vec![]
    };
    let report = invoke(&req, &extra)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    }

    let label = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    println!("HEIR as an FHE baseline — {label}, {scheme}/{backend}, activation {activation}");
    println!(
        "  heir {}   ring dim {}   security {}",
        if report.heir_version.is_empty() {
            "?"
        } else {
            &report.heir_version
        },
        report.ring_dim,
        if report.security_level.is_empty() {
            "(none reported)"
        } else {
            &report.security_level
        }
    );

    // The MPC arm (`bench-2pc-nn`) computes an exact ReLU. Passing that in is what
    // makes a mismatch surface as "different functions" rather than as a class
    // disagreement misread as approximation error.
    match report.gate_with(&expected, Some("relu")) {
        Ok(()) => {
            println!(
                "\n  keygen {:.1} ms (per session, excluded below)\n  \
                 encrypt {:.1} + eval {:.1} + decrypt {:.1} = {:.1} ms per inference\n  \
                 ciphertext {} B",
                report.keygen_ms,
                report.encrypt_ms,
                report.eval_ms,
                report.decrypt_ms,
                report.latency_ms(),
                report.ciphertext_bytes
            );
            println!(
                "  classification matches the plaintext reference on all {} inputs",
                expected.len()
            );
            if !report.notes.is_empty() {
                println!("  note: {}", report.notes);
            }
            println!(
                "\nCompare against `baru bench-2pc-nn --shapes {label}`. Both arms compute the\n\
                 same function on the same weights; the MPC arm is EXACT and 2-party\n\
                 dishonest-majority, this arm is single-server. Same corruption model,\n\
                 different interaction model — FHE lets the client go offline and 2PC\n\
                 does not."
            );
            Ok(())
        }
        Err(why) => {
            // Not an error when the user explicitly asked for the plaintext mode:
            // the refusal *is* the expected outcome, and demonstrating it is the
            // reason the mode exists.
            if self_test && why == Refusal::Simulated {
                println!("\n  REFUSED, as intended: {why}");
                println!(
                    "  the contract round-tripped and the classification was reproduced on \
                     all {} inputs,\n  which is what --self-test is for. Install HEIR for a \
                     real measurement.",
                    expected.len()
                );
                return Ok(());
            }
            // Print the refusal with real newlines — `main` renders a returned
            // error through `Debug`, which escapes them, and these messages are
            // the whole point of the gate.
            println!("\n  REFUSED: {why}");
            Err(match why {
                Refusal::Failed(_) => "the HEIR driver could not run — see above".into(),
                Refusal::Insecure(l) => format!("insecure security level `{l}` — see above"),
                Refusal::Disagrees { .. } => {
                    "the FHE arm computed a different classification — see above".to_string()
                }
                Refusal::Simulated => "plaintext run, not a measurement".into(),
                Refusal::DifferentNetwork { .. } => {
                    "the two arms compute different functions — see above".to_string()
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secure_report(classes: Vec<i64>) -> HeirReport {
        HeirReport {
            ok: true,
            error: None,
            simulated: false,
            heir_version: "test".into(),
            scheme: "ckks".into(),
            backend: "openfhe".into(),
            ring_dim: 32768,
            security_level: "HEStd_128_classic".into(),
            activation: "relu".into(),
            activation_degree: 0,
            keygen_ms: 100.0,
            encrypt_ms: 1.0,
            eval_ms: 10.0,
            decrypt_ms: 2.0,
            ciphertext_bytes: 1 << 18,
            classes,
            notes: String::new(),
        }
    }

    /// **The gate this module exists for.** google/heir#1232 measured ~30 s at
    /// `HEStd_NotSet` against ~6 min secure, so accepting an unset level would
    /// manufacture a 12× speedup out of a parameter choice.
    #[test]
    fn an_insecure_security_level_is_refused() {
        for level in ["HEStd_NotSet", "", "none", "128", "HEStd_128_quantum"] {
            let mut r = secure_report(vec![1, 2]);
            r.security_level = level.to_string();
            assert_eq!(
                r.gate(&[1, 2]),
                Err(Refusal::Insecure(level.to_string())),
                "level `{level}` must be refused"
            );
        }
        // And the three real levels are accepted, or the gate would be a wall.
        for level in SECURE_LEVELS {
            let mut r = secure_report(vec![1, 2]);
            r.security_level = level.to_string();
            assert!(r.gate(&[1, 2]).is_ok(), "level `{level}` must be accepted");
        }
    }

    /// A cheaper ReLU polynomial that gets the answer wrong must not be timeable.
    /// **The refusal that matters most**, and the one a reader is most likely to
    /// mistake for approximation error: CKKS cannot compute an exact ReLU, so if
    /// the FHE arm ran `square` while the MPC arm ran `relu` the two are not
    /// comparable at all, whatever their outputs happen to be.
    #[test]
    fn arms_computing_different_functions_are_refused_before_anything_else() {
        let mut r = secure_report(vec![1, 2]);
        r.activation = "square".into();
        assert_eq!(
            r.gate_with(&[1, 2], Some("relu")),
            Err(Refusal::DifferentNetwork {
                mpc: "relu".into(),
                fhe: "square".into()
            }),
            "matching classes must not rescue a mismatched activation"
        );

        // It outranks every other refusal, so a run that is *also* insecure and
        // *also* disagrees still reports the activation first — the strongest
        // objection, and the one that explains the others.
        let mut worse = secure_report(vec![9, 9]);
        worse.activation = "square".into();
        worse.security_level = "HEStd_NotSet".into();
        assert!(matches!(
            worse.gate_with(&[1, 2], Some("relu")),
            Err(Refusal::DifferentNetwork { .. })
        ));

        // And agreement passes.
        let mut ok = secure_report(vec![1, 2]);
        ok.activation = "relu".into();
        assert!(ok.gate_with(&[1, 2], Some("relu")).is_ok());
    }

    #[test]
    fn a_disagreeing_classification_is_refused() {
        let r = secure_report(vec![1, 0]);
        assert_eq!(
            r.gate(&[1, 2]),
            Err(Refusal::Disagrees {
                got: vec![1, 0],
                want: vec![1, 2]
            })
        );
    }

    #[test]
    fn a_plaintext_run_and_a_failed_run_are_both_refused() {
        let mut sim = secure_report(vec![1, 2]);
        sim.simulated = true;
        assert_eq!(sim.gate(&[1, 2]), Err(Refusal::Simulated));

        let mut bad = secure_report(vec![1, 2]);
        bad.ok = false;
        bad.error = Some("compile failed".into());
        assert_eq!(
            bad.gate(&[1, 2]),
            Err(Refusal::Failed("compile failed".into()))
        );
    }

    /// Key generation is a per-session cost, not a per-query one. E45 records
    /// charging it to the query and reporting a 37× gap that was 1.8×.
    #[test]
    fn latency_excludes_keygen() {
        let r = secure_report(vec![0]);
        assert_eq!(r.latency_ms(), 13.0);
        assert!(
            r.keygen_ms > r.latency_ms(),
            "the fixture should make the point"
        );
    }

    /// The scale is divided out exactly once, including on the bias — which lives
    /// at *output* scale in `QuantizedNetwork::forward`. Dividing it twice would
    /// make the FHE arm compute a different network, and the class gate would
    /// report that as approximation error rather than as the bug it is.
    #[test]
    fn the_request_dequantizes_weights_and_bias_consistently() {
        let net = super::super::two_party_nn::fixture_for(&[4, 3, 2], 8);
        let inputs = vec![super::super::two_party_nn::input_for(4, 8, 0)];
        let req = request_for(&net, &inputs, "ckks", "openfhe", "square", 1);
        let scale = 256.0;
        for (l, layer) in net.layers.iter().enumerate() {
            for (o, row) in layer.weights.iter().enumerate() {
                for (i, w) in row.iter().enumerate() {
                    assert_eq!(req.layers[l].weights[o][i], *w as f64 / scale);
                }
                assert_eq!(req.layers[l].bias[o], layer.bias[o] as f64 / scale);
            }
        }
        assert_eq!(req.security_level, "HEStd_128_classic");
    }

    /// The absent-tool path must be an actionable message naming the fallback,
    /// not a panic — the `db::barudb` convention.
    #[test]
    fn a_missing_driver_says_what_to_do() {
        let net = super::super::two_party_nn::fixture_for(&[4, 3, 2], 8);
        let inputs = vec![super::super::two_party_nn::input_for(4, 8, 0)];
        let req = request_for(&net, &inputs, "ckks", "openfhe", "square", 1);
        let err = invoke_at("/nonexistent/heir_baseline.py", &req, &[]).unwrap_err();
        assert!(err.contains("BARU_HEIR_PY"), "{err}");
        assert!(err.contains("heir_baseline.py"), "{err}");
    }

    /// The full contract through a real subprocess: Rust serializes a request,
    /// Python evaluates it, Rust parses the report, and the security gate refuses
    /// it — because `--self-test` honestly reports `HEStd_NotSet`.
    ///
    /// `#[ignore]`d because it spawns `python3`, which CI does not install.
    #[test]
    #[ignore]
    fn the_contract_round_trips_and_is_then_refused() {
        if !available() {
            eprintln!("skipping: no {DEFAULT_SCRIPT} — run from the repository root");
            return;
        }
        let shape = [6usize, 4, 3];
        let net = super::super::two_party_nn::fixture_for(&shape, 8);
        let inputs: Vec<Vec<i64>> = (0..3)
            .map(|k| super::super::two_party_nn::input_for(shape[0], 8, k))
            .collect();
        let expected: Vec<i64> = inputs
            .iter()
            .map(|x| super::super::two_party_nn::argmax_of(&net.forward(x)))
            .collect();

        let req = request_for(&net, &inputs, "ckks", "openfhe", "square", 1);
        let r = invoke(&req, &["--self-test"]).expect("the self-test path should run");
        assert!(r.ok, "self-test reported a failure: {:?}", r.error);
        assert!(r.simulated);

        // The float reference must reproduce the fixed-point classification, or
        // the two arms are not computing the same network and no FHE measurement
        // would mean anything.
        assert_eq!(
            r.classes, expected,
            "the float evaluation disagrees with the quantized one — the arms differ"
        );
        // And it is still refused, which is the gate working.
        assert_eq!(r.gate(&expected), Err(Refusal::Simulated));
    }
}
