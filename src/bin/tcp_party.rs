//! Multi-party REPL/evaluator binary.
//!
//! Usage (3 separate terminals, one per party):
//!
//!   party 0: tcp_party 0 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
//!   party 1: tcp_party 1 127.0.0.1:7001,127.0.0.1:7001,127.0.0.1:7002
//!   party 2: tcp_party 2 127.0.0.1:7002,127.0.0.1:7001,127.0.0.1:7002
//!
//! Party 0 reads expressions from a full-featured REPL (colours, tab-completion,
//! history, :help, :examples, :example <n>) and broadcasts them to peers.
//! All parties evaluate jointly via `TransportBackend`.
//!
//! Run with no arguments for an in-process 3-party demo.

use std::env;
use std::path::PathBuf;
use std::time::Duration;
use std::thread;
use std::panic;

use colored::*;
use rustyline::Editor;
use rustyline::error::ReadlineError;

use baru::crypto::net::{
    Transport, TcpTransport, InprocHandle,
    serialize_msg, deserialize_msg, MpcMsg, recv_blocking, recv_forever,
};
use baru::interpreter::{parse, Participant, SharedValue};
use baru::repl::{
    EXAMPLES, print_help, print_examples_list, print_examples_for, ReplHelper,
};

// ---------------------------------------------------------------------------
// Expression broadcast helpers (use share_id=0 for len + sequential ids for chunks)
// ---------------------------------------------------------------------------

fn broadcast_expr<T: Transport>(src: &str, transport: &T) {
    let bytes = src.as_bytes();
    transport.broadcast(serialize_msg(&MpcMsg::RevealShare { share_id: 0, value: bytes.len() as u128 }));
    for (i, chunk) in bytes.chunks(16).enumerate() {
        let mut v: u128 = 0;
        for (j, &b) in chunk.iter().enumerate() { v |= (b as u128) << (j * 8); }
        transport.broadcast(serialize_msg(&MpcMsg::RevealShare { share_id: (i + 1) as u64, value: v }));
    }
}

fn receive_expr<T: Transport>(transport: &T) -> String {
    // Wait indefinitely for the length header — no timeout between expressions.
    let len = loop {
        let (_from, msg) = recv_forever(transport);
        if let MpcMsg::RevealShare { share_id: 0, value: l } = deserialize_msg(&msg) {
            break l as usize;
        }
        // Any other message that arrived before the length header is ignored;
        // this can happen if a stray MPC message lingers in the queue.
    };
    let chunks = (len + 15) / 16;
    let mut bytes = vec![0u8; len];
    let mut received = 0usize;
    while received < chunks {
        if let Some((_from, msg)) = recv_blocking(transport, 5_000) {
            if let MpcMsg::RevealShare { share_id, value } = deserialize_msg(&msg) {
                if share_id >= 1 {
                    let start = (share_id as usize - 1) * 16;
                    for j in 0..16 {
                        if start + j < len { bytes[start + j] = ((value >> (j * 8)) & 0xff) as u8; }
                    }
                    received += 1;
                }
            }
        } else { panic!("receive_expr: timeout waiting for chunk"); }
    }
    String::from_utf8(bytes).expect("expression not valid UTF-8")
}

// ---------------------------------------------------------------------------
// In-process 3-party demo
//
// Each party runs in its own thread.  We print results inside the thread to
// avoid moving !Send SharedValue across thread boundaries.
// Two InprocHandle networks are created: one for expression distribution,
// one for MPC operations.
// ---------------------------------------------------------------------------

fn run_inproc_demo() {
    println!("Running in-process 3-party MPC demo...\n");

    let demos: &[&str] = &[
        "(reconstruct (share 7 8))",
        "(reconstruct (+ (share 10 8) (share 32 8)))",
        "(let ((w (share -1 8)) (x (share 3 8))) (reconstruct (* w x)))",
        "(let ((z (+ (share -2 8) (share 5 8)))) (reconstruct (if (> z 0) z 0)))",
    ];

    for &src in demos {
        println!("Expression: {}", src);
        // one network for MPC, one for expression distribution
        let mpc_handles = InprocHandle::create(3);
        let dist_handles = InprocHandle::create(3);

        let mut joins = vec![];
        for i in 0..3usize {
            let mpc_h = mpc_handles[i].clone();
            let dist_h = dist_handles[i].clone();
            let src_str = src.to_string();
            joins.push(thread::spawn(move || {
                let participant = Participant::new_transport(i, mpc_h);
                let resolved_src = if i == 0 {
                    broadcast_expr(&src_str, &dist_h);
                    src_str
                } else {
                    receive_expr(&dist_h)
                };
                let expr = parse(&resolved_src).expect("parse failed");
                let result = participant.eval(&expr);
                // print inside thread to avoid Send bound on SharedValue
                println!("  party {} => {:?}", i, result);
            }));
        }
        for j in joins { j.join().expect("thread panicked"); }
        println!();
    }
}

// ---------------------------------------------------------------------------
// TCP multi-party mode
// ---------------------------------------------------------------------------

fn history_path() -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".baru_tcp_history");
        Some(p)
    } else {
        None
    }
}

/// Display a SharedValue result with sign-extension for 32-bit MPC field values.
fn display_result(result: &SharedValue, participant: &Participant) -> String {
    match result {
        SharedValue::Share(sr) => {
            let raw = participant.reveal(sr);
            let signed = (raw as u32) as i32 as i64;
            format!("{}", signed.to_string().bright_green())
        }
        SharedValue::Num(n) => {
            // sign-extend through 32 bits for MPC field values
            let signed = (*n as u32) as i32 as i64;
            format!("{}", signed.to_string().bright_green())
        }
        other => format!("{}", format!("{:?}", other).bright_green()),
    }
}

fn run_tcp(me: usize, mpc_addrs: Vec<String>, dist_addrs: Vec<String>) {
    let mpc_transport = TcpTransport::new(me, mpc_addrs);
    let dist_transport = TcpTransport::new(me, dist_addrs);
    eprintln!("{}", format!("[party {}] TCP transports ready", me).bright_yellow());

    let participant = Participant::new_transport(me, mpc_transport);

    if me == 0 {
        // ── Party 0: full interactive REPL ──────────────────────────────────
        println!("{}", "ABY3 distributed MPC — party 0 (evaluator + broadcaster)".bright_green());
        println!("{}", "Type ':help' for language reference, ':examples' to browse examples, or ':quit' to exit.".bright_yellow());

        let helper = ReplHelper::new_tcp();
        let mut rl = Editor::new().expect("failed to create line editor");
        rl.set_helper(Some(helper));

        if let Some(p) = history_path() {
            if p.exists() { let _ = rl.load_history(&p); }
        }

        let mut verbose = false;

        loop {
            let readline = rl.readline(&format!("{}{}{}",
                "baru[party-0]".bright_cyan(), "> ".bright_white(), ""));
            let src = match readline {
                Ok(line) => line.trim().to_string(),
                Err(ReadlineError::Interrupted) => { println!("^C"); continue; }
                Err(ReadlineError::Eof) | Err(_) => {
                    broadcast_expr(":quit", &dist_transport);
                    break;
                }
            };

            if src.is_empty() || src.starts_with(';') { continue; }
            let _ = rl.add_history_entry(&src);

            // ── Special commands ────────────────────────────────────────────
            if src.starts_with(':') || src == "quit" || src == "exit" {
                let mut parts = src.splitn(3, ' ');
                let cmd = parts.next().unwrap_or("");
                match cmd {
                    ":quit" | "quit" | "exit" => {
                        broadcast_expr(":quit", &dist_transport);
                        println!("{}", "Signalled all parties to quit.".bright_yellow());
                        break;
                    }
                    ":help" => { print_help(true); continue; }
                    ":examples" => {
                        match parts.next() {
                            None => print_examples_list(),
                            Some(cat) => print_examples_for(cat),
                        }
                        continue;
                    }
                    ":example" => {
                        match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                            Some(n) if n >= 1 && n <= EXAMPLES.len() => {
                                let ex = &EXAMPLES[n - 1];
                                println!();
                                println!("{} [{}] {} ({})",
                                    "Running example".bright_yellow(),
                                    n.to_string().bright_white(),
                                    ex.name.bright_cyan(),
                                    ex.category);
                                println!("{} {}", "Code:    ".bright_yellow(), ex.code.bright_green());
                                println!("{} {}", "Expected:".bright_yellow(), ex.expected.bright_magenta());
                                println!();
                                // broadcast and fall through to normal eval below
                                participant.reset_per_expression();
                                broadcast_expr(ex.code, &dist_transport);
                                match parse(ex.code) {
                                    Ok(expr) => {
                                        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                                            if verbose {
                                                participant.eval_quiet_verbose(&expr)
                                            } else {
                                                participant.eval_quiet(&expr)
                                            }
                                        }));
                                        match result {
                                            Ok(val) => println!("{} {}", "=>".bright_yellow(), display_result(&val, &participant)),
                                            Err(e) => {
                                                let msg = if let Some(s) = e.downcast_ref::<String>() { s.clone() }
                                                          else if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
                                                          else { "unknown runtime error".to_string() };
                                                println!("{} {}", "runtime error:".bright_red(), msg);
                                                broadcast_expr(":abort", &dist_transport);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        println!("{} {}", "parse error:".bright_red(), e);
                                        broadcast_expr(":abort", &dist_transport);
                                    }
                                }
                            }
                            _ => println!("usage: :example <n>  (use ':examples' to list, 1–{})", EXAMPLES.len()),
                        }
                        continue;
                    }
                    ":verbose" => {
                        match parts.next() {
                            Some("on")  => { verbose = true;  println!("{}", "verbose: on".bright_green()); }
                            Some("off") => { verbose = false; println!("{}", "verbose: off".bright_yellow()); }
                            _ => {
                                verbose = !verbose;
                                println!("verbose: {}", if verbose { "on".bright_green() } else { "off".bright_yellow() });
                            }
                        }
                        continue;
                    }
                    _ => {
                        println!("{} {}  (type '{}' for reference)",
                            "unknown command:".bright_red(), cmd,
                            ":help".bright_cyan());
                        continue;
                    }
                }
            }

            // ── Normal expression ────────────────────────────────────────────
            // Reset per-expression backend state so all parties' ID counters start
            // at the same value for this expression (crucial after error recovery).
            participant.reset_per_expression();
            broadcast_expr(&src, &dist_transport);
            match parse(&src) {
                Err(e) => {
                    println!("{} {}", "parse error:".bright_red(), e);
                    // Workers already received the expression and got the same parse error.
                    // Broadcast :abort so they skip any pending reveal and return to wait.
                    broadcast_expr(":abort", &dist_transport);
                }
                Ok(expr) => {
                    // Wrap evaluation in catch_unwind so a panic (e.g. MPC protocol
                    // timeout, out-of-field value) doesn't crash the whole process.
                    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        if verbose {
                            participant.eval_quiet_verbose(&expr)
                        } else {
                            participant.eval_quiet(&expr)
                        }
                    }));
                    match result {
                        Ok(val) => {
                            println!("{} {}", "=>".bright_yellow(), display_result(&val, &participant));
                        }
                        Err(e) => {
                            let msg = if let Some(s) = e.downcast_ref::<String>() { s.clone() }
                                      else if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
                                      else { "unknown runtime error".to_string() };
                            println!("{} {}", "runtime error:".bright_red(), msg);
                            // Broadcast :abort so workers re-sync to the wait loop.
                            broadcast_expr(":abort", &dist_transport);
                        }
                    }
                }
            }
        }

        if let Some(p) = history_path() { let _ = rl.save_history(&p); }

    } else {
        // ── Parties 1/2: silent workers ──────────────────────────────────────
        eprintln!("{}", format!("[party {}] ready, waiting for expressions from party 0...", me).bright_yellow());
        loop {
            let src = receive_expr(&dist_transport);
            let src = src.trim().to_string();
            if src == ":quit" {
                eprintln!("{}", format!("[party {}] received :quit, exiting.", me).bright_yellow());
                break;
            }
            if src == ":abort" {
                // Party 0 had a parse or runtime error; skip this round.
                eprintln!("{}", format!("[party {}] received :abort, skipping expression.", me).bright_yellow());
                continue;
            }
            eprintln!("{}", format!("[party {}] evaluating: {}", me, src).bright_black());
            match parse(&src) {
                Ok(expr) => {
                    // Wrap in catch_unwind so a protocol panic doesn't crash the worker.
                    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        participant.reset_per_expression();
                        participant.eval_quiet(&expr)
                    }));
                    match result {
                        Ok(val) => {
                            // Mirror party 0's display_result to stay in sync on reveals.
                            let display = match &val {
                                SharedValue::Share(sr) => {
                                    let raw = participant.reveal(sr);
                                    ((raw as u32) as i32 as i64).to_string()
                                }
                                SharedValue::Num(n) => ((*n as u32) as i32 as i64).to_string(),
                                other => format!("{:?}", other),
                            };
                            eprintln!("{}", format!("[party {}] => {}  (waiting for next expression)", me, display).bright_black());
                        }
                        Err(e) => {
                            let msg = if let Some(s) = e.downcast_ref::<String>() { s.clone() }
                                      else if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
                                      else { "unknown runtime error".to_string() };
                            eprintln!("{}", format!("[party {}] runtime error: {}  (recovered, waiting for next)", me, msg).bright_red());
                            // Wait for the :abort that party 0 will broadcast.
                            let abort = receive_expr(&dist_transport);
                            if abort.trim() == ":quit" { break; }
                            // any other message (including :abort) — continue the loop
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{}", format!("[party {}] parse error: {}  (waiting for :abort)", me, e).bright_red());
                    // Wait for :abort from party 0 before re-entering the wait loop.
                    let abort = receive_expr(&dist_transport);
                    if abort.trim() == ":quit" { break; }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Legacy D-sum protocol (used by the integration test).
/// Each party broadcasts its d_value and prints `SUM: <total>`.
fn run_d_sum(me: usize, addrs: Vec<String>, d_val: u128) {
    use std::time::Duration as D;
    thread::sleep(D::from_millis(50 * me as u64));
    let transport = TcpTransport::new(me, addrs);
    transport.broadcast(serialize_msg(&MpcMsg::D(d_val)));

    let mut got = vec![false; 3];
    got[me] = true;
    let mut d_shares = vec![0u128; 3];
    d_shares[me] = d_val;
    let mut pending = 2;
    let start = std::time::Instant::now();
    while pending > 0 && start.elapsed() < Duration::from_secs(8) {
        if let Some((from, msg)) = transport.recv() {
            if let MpcMsg::D(v) = deserialize_msg(&msg) {
                if !got[from] { got[from] = true; d_shares[from] = v; pending -= 1; }
            }
        } else {
            thread::sleep(Duration::from_millis(10));
        }
    }
    if pending > 0 { eprintln!("[party {}] timed out", me); std::process::exit(1); }
    let sum: u128 = d_shares.iter().fold(0, |a, &v| a.wrapping_add(v));
    println!("SUM: {}", sum);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // no-args: in-process demo
    if args.len() < 3 {
        run_inproc_demo();
        return;
    }

    let me: usize = args[1].parse().expect("first argument must be party id (0/1/2)");
    let addrs: Vec<String> = args[2].split(',').map(|s| s.to_string()).collect();
    if addrs.len() != 3 { eprintln!("expected 3 addresses"); std::process::exit(1); }

    // legacy integration-test mode: <id> <addrs> <d_value>
    if args.len() >= 4 {
        if let Ok(d_val) = args[3].parse::<u128>() {
            run_d_sum(me, addrs, d_val);
            return;
        }
        // fourth arg is dist_addrs (MPC interactive mode)
        let dist_addrs: Vec<String> = args[3].split(',').map(|s| s.to_string()).collect();
        thread::sleep(Duration::from_millis(300 + me as u64 * 150));
        run_tcp(me, addrs, dist_addrs);
        return;
    }

    // two args: derive dist_addrs by bumping port by 100
    let dist_addrs: Vec<String> = addrs.iter().map(|addr| {
        if let Some(colon) = addr.rfind(':') {
            let port: u16 = addr[colon+1..].parse().unwrap_or(7000);
            format!("{}:{}", &addr[..colon], port + 100)
        } else { format!("{}:7100", addr) }
    }).collect();
    thread::sleep(Duration::from_millis(300 + me as u64 * 150));
    run_tcp(me, addrs, dist_addrs);
}
