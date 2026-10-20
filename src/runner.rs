//! `baru run <file.baru>` — evaluate a Baru **source file**, on the plain
//! interpreter or on the real 3-party MPC transport, and time it.
//!
//! # Why this exists
//!
//! The REPL reads one expression per line, so until now a Baru program longer
//! than a line had nowhere to live: every worked example in `LANGUAGE.md` is a
//! one-liner, and `examples/` held Rust files that *drive* the backends rather
//! than Baru programs. A CNN cannot be written on one line, so
//! `examples/medmnist.baru` needed a runner before it could exist.
//!
//! # The step budget
//!
//! [`crate::interpreter::eval`]'s trampoline has a per-expression step budget
//! whose REPL default (1 000 000) is a runaway-recursion guard sized for a
//! one-line expression. A real program is orders of magnitude larger — one
//! 28×28 convolution alone exceeds it — so the runner raises the budget via
//! [`crate::interpreter::eval::set_max_frames`]. `BARU_MAX_FRAMES` still wins,
//! so the guard can be pulled back down to debug a non-terminating loop.
//!
//! Expressions are evaluated one at a time (rather than wrapped in a `begin`)
//! so that a `define` at file scope lands in the participant's persistent store
//! and the next expression can see it — which is what makes a file a program
//! rather than one enormous `let`.
//!
//! # Two backends, one source text
//!
//! `--plain` builds one [`Participant::new_simulated`] and never shares
//! anything; `--mpc` builds three [`Participant::new_transport`] parties on an
//! [`InprocHandle`] mesh, each thread evaluating the *same* expressions
//! concurrently, which is the only configuration whose timings mean anything
//! (`new_simulated`'s own docs say so: a single simulated party has no peers and
//! the protocol cost is not exercised). The program is told which world it is in
//! through the injected `PRIVATE` binding, and the layer code branches on it
//! once, at the input — every builtin it then calls dispatches on the runtime
//! type of its argument, so `conv2d` and `dense` are the same source either way.

use crate::bench::comm::{CommStats, CountingTransport, LatencyTransport, NetworkProfile};
use crate::crypto::net::InprocHandle;
use crate::interpreter::ast::Expr;
use crate::interpreter::parser::{parse_tokens, tokenize};
use crate::interpreter::{Participant, Types};
use std::thread;
use std::time::{Duration, Instant};

/// Parse every top-level expression in a source string.
///
/// The REPL's `parse` returns after one expression and rejects the rest as
/// "trailing tokens"; a file is a sequence, so this drives
/// [`parse_tokens`] over one shared cursor until the tokens run out.
pub fn parse_program(src: &str) -> Result<Vec<Expr>, String> {
    let tokens = tokenize(src);
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos < tokens.len() {
        let before = pos;
        let expr = parse_tokens(&tokens, &mut pos)
            .map_err(|e| format!("parse error at token {before}: {e}"))?;
        out.push(expr);
        if pos == before {
            return Err(format!("parser made no progress at token {before}"));
        }
    }
    Ok(out)
}

/// The bindings the runner injects before the file's own expressions, so one
/// source text can be parameterised from the command line.
fn preamble(dataset: &str, images: usize, private: usize, nbits: usize) -> String {
    format!(
        "(define DATASET \"{dataset}\")\n\
         (define NIMAGES {images})\n\
         (define PRIVATE {private})\n\
         (define NBITS {nbits})\n"
    )
}

/// Render a result for printing. A `Share` is deliberately *not* revealed here:
/// the program is expected to `reconstruct` whatever it wants shown, so that the
/// reveal is visible in the Baru source rather than performed by the harness.
fn render(v: &Types) -> String {
    match v {
        Types::Num(n) => n.to_string(),
        Types::Fixed(_) => format!("{v:?}"),
        Types::Share(_) => "<share — the program did not reconstruct it>".to_string(),
        Types::List(xs) => {
            let inner: Vec<String> = xs.iter().map(render).collect();
            format!("({})", inner.join(" "))
        }
        other => format!("{other:?}"),
    }
}

/// Evaluate the whole program on one participant, returning the last value and
/// the elapsed evaluation time (parsing excluded).
fn eval_program(p: &Participant, prog: &[Expr]) -> (Types, Duration) {
    let t0 = Instant::now();
    let mut last = Types::Num(0);
    for e in prog {
        last = p.eval_quiet(e);
        if let Types::Error(..) = last {
            break;
        }
    }
    (last, t0.elapsed())
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut path: Option<String> = None;
    let mut dataset = "pneumonia".to_string();
    let mut images = 1usize;
    let mut mpc = false;
    // What the *program* hides, independent of which backend evaluates it:
    // 0 plaintext, 1 secret input, 2 secret input and secret model.
    let mut private: Option<usize> = None;
    // Which protocol backs comparisons and the fused nonlinearity. "gc" is the
    // constant-round garbled route the Rust hybrid pipeline uses; it is the
    // single biggest lever on a private run, and it has to be selected before
    // any party performs an operation, since the modes allocate ids differently.
    let mut nonlin = "a2b".to_string();
    let mut nbits = 64usize;
    let mut rtt_ms = 0u64;
    let mut bandwidth_mbps: Option<f64> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mpc" => mpc = true,
            "--plain" => mpc = false,
            "--nonlin" => nonlin = it.next().ok_or("--nonlin needs a value")?.clone(),
            "--gc" => nonlin = "gc".to_string(),
            "--private" => {
                private = Some(
                    it.next()
                        .ok_or("--private needs a value (0, 1 or 2)")?
                        .parse()
                        .map_err(|_| "bad --private")?,
                )
            }
            "--dataset" => dataset = it.next().ok_or("--dataset needs a value")?.clone(),
            "--images" => {
                images = it
                    .next()
                    .ok_or("--images needs a value")?
                    .parse()
                    .map_err(|_| "bad --images")?
            }
            "--nbits" => {
                nbits = it
                    .next()
                    .ok_or("--nbits needs a value")?
                    .parse()
                    .map_err(|_| "bad --nbits")?
            }
            "--rtt-ms" => {
                rtt_ms = it
                    .next()
                    .ok_or("--rtt-ms needs a value")?
                    .parse()
                    .map_err(|_| "bad --rtt-ms")?
            }
            "--bandwidth-mbps" => {
                bandwidth_mbps = Some(
                    it.next()
                        .ok_or("--bandwidth-mbps needs a value")?
                        .parse()
                        .map_err(|_| "bad --bandwidth-mbps")?,
                )
            }
            "--help" | "-h" => {
                println!(
                    "baru run <file.baru> [--plain|--mpc] [--dataset NAME] [--images N]\n\
                     \x20                  [--nbits N] [--rtt-ms N] [--bandwidth-mbps F]\n\n\
                     --plain (default) evaluates on the plain interpreter: no sharing, no\n\
                     encryption, so the figure is the cost of the computation itself.\n\
                     --mpc runs three parties over a real in-process transport, which is the\n\
                     only configuration whose timings are quotable."
                );
                return Ok(());
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => path = Some(other.to_string()),
        }
    }
    let path = path.ok_or("usage: baru run <file.baru> [--plain|--mpc]")?;
    // A file is a whole program, not a REPL line: one 28x28 convolution already
    // costs more trampoline steps than the REPL's runaway guard allows. See
    // `interpreter::eval::set_max_frames` — BARU_MAX_FRAMES still overrides.
    crate::interpreter::eval::set_max_frames(64_000_000_000);
    // Default: the plain backend runs the plaintext program, the MPC backend
    // runs the secret-input one. Either can be overridden — running PRIVATE=0 on
    // the MPC backend is the control that says how much of the cost is protocol.
    let private = private.unwrap_or(if mpc { 1 } else { 0 });
    if private > 2 {
        return Err("--private takes 0, 1 or 2".to_string());
    }
    let src = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let full = format!("{}{src}", preamble(&dataset, images, private, nbits));
    let prog = parse_program(&full)?;

    println!(
        "baru run {path} — {} backend, dataset {dataset}, {images} image(s)",
        if mpc { "3-party MPC" } else { "plain" }
    );
    println!(
        "  hiding   : {}",
        match private {
            0 => "nothing (plaintext)",
            1 => "the image (model public)",
            _ => "the image AND the model",
        }
    );
    if mpc {
        println!("  nonlin   : {nonlin}");
    }
    if mpc && (rtt_ms > 0 || bandwidth_mbps.is_some()) {
        println!(
            "  network: rtt={rtt_ms}ms, bandwidth={}",
            bandwidth_mbps
                .map(|b| format!("{b} Mbit/s"))
                .unwrap_or_else(|| "unlimited".to_string())
        );
    }
    println!("  {} top-level expressions", prog.len());

    if !mpc {
        let p = Participant::new_simulated(0);
        let (last, dt) = eval_program(&p, &prog);
        report(&last, dt, images, None);
        return Ok(());
    }

    let run = run_mpc(&prog, &nonlin, rtt_ms, bandwidth_mbps)?;
    report(
        &Types::Num(0),
        run.slowest,
        images,
        Some((&run.result, &run.stats)),
    );
    Ok(())
}

fn report(last: &Types, dt: Duration, images: usize, mpc: Option<(&String, &CommStats)>) {
    let ms = dt.as_secs_f64() * 1000.0;
    match mpc {
        Some((rendered, stats)) => {
            println!("  result   : {rendered}");
            println!(
                "  traffic  : {} messages, {:.2} MB party-to-party",
                stats.messages,
                stats.bytes as f64 / 1e6
            );
        }
        None => println!("  result   : {}", render(last)),
    }
    println!("  latency  : {ms:.0} ms total");
    if images > 0 {
        println!("  per image: {:.1} ms", ms / images as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_sequence_of_top_level_expressions() {
        let prog = parse_program("(define a 1)\n(define b 2)\n(+ a b)").expect("parses");
        assert_eq!(prog.len(), 3);
    }

    #[test]
    fn comments_and_blank_lines_do_not_become_expressions() {
        let prog = parse_program("; a comment\n\n(define a 1)\n; another\n(+ a 1)\n").expect("ok");
        assert_eq!(prog.len(), 2);
    }

    /// A truncated file must be a parse error, not a silently shorter program.
    #[test]
    fn unbalanced_parens_are_an_error() {
        assert!(parse_program("(define a (+ 1 2)").is_err());
    }

    /// The store must carry across top-level expressions, which is the whole
    /// reason the runner evaluates them one at a time instead of wrapping the
    /// file in a `begin` (which would share one `MAX_FRAMES` budget).
    #[test]
    fn definitions_persist_across_expressions() {
        let p = Participant::new_simulated(0);
        let prog = parse_program("(define (sq n) (* n n))\n(sq 7)").expect("parses");
        let (last, _) = eval_program(&p, &prog);
        assert!(matches!(last, Types::Num(49)), "got {last:?}");
    }

    #[test]
    fn preamble_injects_the_parameters_the_program_reads() {
        let src = preamble("organa", 5, 1, 64);
        let p = Participant::new_simulated(0);
        let prog = parse_program(&format!("{src}(list DATASET NIMAGES PRIVATE NBITS)")).unwrap();
        let (last, _) = eval_program(&p, &prog);
        match last {
            Types::List(xs) => {
                assert!(matches!(&xs[0], Types::Str(s) if s == "organa"));
                assert!(matches!(xs[1], Types::Num(5)));
                assert!(matches!(xs[2], Types::Num(1)));
                assert!(matches!(xs[3], Types::Num(64)));
            }
            other => panic!("got {other:?}"),
        }
    }

    /// Evaluation stops at the first error rather than running on with a
    /// poisoned store and reporting a later expression's value as the answer.
    #[test]
    fn evaluation_stops_at_the_first_error() {
        let p = Participant::new_simulated(0);
        let prog = parse_program("(medmnist-model \"no-such-set\")\n42").expect("parses");
        let (last, _) = eval_program(&p, &prog);
        assert!(matches!(last, Types::Error(..)), "got {last:?}");
    }
}

/// One 3-party MPC run of an already-parsed program.
#[derive(Debug)]
pub struct MpcRun {
    /// The rendered value of the program's last expression, as every party saw
    /// it — the run errors rather than returning if they disagreed.
    pub result: String,
    /// Wall clock of the **slowest** party: the computation is not finished
    /// until every party is, and quoting party 0 alone would undercount.
    pub slowest: Duration,
    pub stats: CommStats,
}

/// Evaluate `prog` on three parties over a real in-process transport.
///
/// Extracted from `run_cli` so tests can drive it: the property that matters
/// about `--nonlin` is that changing it does **not** change any answer, and that
/// is only checkable by running the same program on two protocols and comparing.
pub fn run_mpc(
    prog: &[Expr],
    nonlin: &str,
    rtt_ms: u64,
    bandwidth_mbps: Option<f64>,
) -> Result<MpcRun, String> {
    let handles = InprocHandle::create(3);
    let profile = {
        let base = NetworkProfile::fixed_latency(Duration::from_micros(rtt_ms * 500));
        match bandwidth_mbps {
            Some(mbps) => base.with_bandwidth(mbps * 125_000.0),
            None => base,
        }
    };
    let (tx, rx) = std::sync::mpsc::channel::<(usize, String, Duration, CommStats)>();
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        let prog = prog.to_vec();
        let tx = tx.clone();
        let nonlin = nonlin.to_string();
        joins.push(thread::spawn(move || {
            let counting = CountingTransport::new(h, 3);
            let stats = counting.handle();
            let transport = LatencyTransport::new(counting, profile, 3);
            let p = Participant::new_transport(i, transport);
            if let Err(e) = p.set_nonlinearity(&nonlin) {
                let _ = tx.send((i, format!("ERROR: {e}"), Duration::ZERO, stats.stats()));
                return;
            }
            let (last, dt) = eval_program(&p, &prog);
            // `Types` is !Send (it holds `Rc`s), so only the rendering crosses.
            let _ = tx.send((i, render(&last), dt, stats.stats()));
        }));
    }
    drop(tx);
    let mut results: Vec<(usize, String, Duration, CommStats)> = rx.iter().collect();
    for j in joins {
        j.join().map_err(|_| "a party thread panicked")?;
    }
    results.sort_by_key(|r| r.0);
    let p0 = results
        .iter()
        .find(|r| r.0 == 0)
        .ok_or("party 0 produced no result")?
        .clone();
    if let Some(bad) = results.iter().find(|r| r.1 != p0.1) {
        return Err(format!(
            "parties disagree — party 0 says {:?}, party {} says {:?}; DO NOT quote this run",
            p0.1, bad.0, bad.1
        ));
    }
    if p0.1.starts_with("ERROR:") {
        return Err(p0.1.clone());
    }
    Ok(MpcRun {
        result: p0.1,
        slowest: results.iter().map(|r| r.2).max().unwrap_or_default(),
        stats: p0.3,
    })
}
