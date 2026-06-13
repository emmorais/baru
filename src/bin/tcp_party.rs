//! Multi-party REPL/evaluator binary.
//!
//! Usage (3 separate terminals, one per party):
//!
//!   party 0: tcp_party 0 127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002
//!   party 1: tcp_party 1 127.0.0.1:7001,127.0.0.1:7001,127.0.0.1:7002
//!   party 2: tcp_party 2 127.0.0.1:7002,127.0.0.1:7001,127.0.0.1:7002
//!
//! Party 0 reads expressions from stdin and broadcasts them to peers.
//! All parties evaluate jointly via `TransportBackend`.
//!
//! Run with no arguments for an in-process 3-party demo.

use std::env;
use std::io::{self, BufRead, Write};
use std::time::Duration;
use std::thread;

use baru::crypto::net::{
    Transport, TcpTransport, InprocHandle,
    serialize_msg, deserialize_msg, MpcMsg, recv_blocking, recv_forever,
};
use baru::interpreter::{parse, Participant};

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

fn run_tcp(me: usize, mpc_addrs: Vec<String>, dist_addrs: Vec<String>) {
    // Two transports: one consumed by TransportBackend, one for expression distribution
    let mpc_transport = TcpTransport::new(me, mpc_addrs);
    let dist_transport = TcpTransport::new(me, dist_addrs);
    eprintln!("[party {}] TCP transports ready", me);

    let participant = Participant::new_transport(me, mpc_transport);

    if me == 0 {
        let stdin = io::stdin();
        print!("baru[party-0]> ");
        let _ = io::stdout().flush();
        for line in stdin.lock().lines() {
            let src = match line { Ok(l) => l.trim().to_string(), Err(_) => break };
            if src.is_empty() || src.starts_with(';') {
                print!("baru[party-0]> ");
                let _ = io::stdout().flush();
                continue;
            }
            if src == ":quit" {
                broadcast_expr(":quit", &dist_transport);
                break;
            }
            broadcast_expr(&src, &dist_transport);
            let expr = match parse(&src) {
                Ok(e) => e,
                Err(err) => { eprintln!("parse error: {}", err); continue; }
            };
            let result = participant.eval(&expr);
            println!("=> {:?}", result);
            print!("baru[party-0]> ");
            let _ = io::stdout().flush();
        }
    } else {
        eprintln!("[party {}] ready, waiting for expressions from party 0...", me);
        loop {
            let src = receive_expr(&dist_transport);
            if src.trim() == ":quit" { break; }
            eprintln!("[party {}] evaluating: {}", me, src.trim());
            let expr = match parse(src.trim()) { Ok(e) => e, Err(e) => { eprintln!("parse error: {}", e); continue; } };
            let result = participant.eval(&expr);
            eprintln!("[party {}] result: {:?}", me, result);
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
