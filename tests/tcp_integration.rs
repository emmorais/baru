use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::thread;
use std::path::Path;
use std::io::{Read, Write};

// ── helpers ──────────────────────────────────────────────────────────────────

fn find_bin() -> &'static Path {
    let release = Path::new("./target/release/tcp_party");
    let debug   = Path::new("./target/debug/tcp_party");
    if release.exists() { return release; }
    if debug.exists()   { return debug; }
    // Binary not found — build it automatically so tests work without a prior `cargo build`.
    let status = Command::new("cargo")
        .args(["build", "--bin", "tcp_party"])
        .status()
        .expect("failed to run cargo build");
    assert!(status.success(), "cargo build --bin tcp_party failed");
    if debug.exists() { debug } else { panic!("tcp_party binary still missing after build") }
}

/// Reserve `n` consecutive free ports and return their addresses.
fn free_ports(n: usize) -> Vec<String> {
    let listeners: Vec<_> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addrs: Vec<String> = listeners.iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect();
    drop(listeners);
    addrs
}

// ── legacy d-sum test (kept) ──────────────────────────────────────────────────

#[test]
#[ignore]
fn tcp_process_integration() {
    let addrs = free_ports(3);
    let bin_path = find_bin();
    let addrs_str = addrs.join(",");

    let mut children = Vec::new();
    for i in 0..3usize {
        let child = Command::new(bin_path)
            .arg(i.to_string())
            .arg(&addrs_str)
            .arg(match i { 0 => "10", 1 => "20", _ => "30" })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn tcp_party");
        children.push(child);
        thread::sleep(Duration::from_millis(250));
    }

    let watchers: Vec<_> = children.into_iter().enumerate().map(|(i, mut child)| {
        thread::spawn(move || {
            let mut stdout = child.stdout.take().unwrap();
            let mut stderr = child.stderr.take().unwrap();
            let out_t = thread::spawn(move || { let mut s = String::new(); let _ = stdout.read_to_string(&mut s); s });
            let err_t = thread::spawn(move || { let mut s = String::new(); let _ = stderr.read_to_string(&mut s); s });
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => { if Instant::now() > deadline { let _ = child.kill(); panic!("party {} timed out", i); } thread::sleep(Duration::from_millis(20)); }
                    Err(e) => panic!("wait error for party {}: {}", i, e),
                }
            }
            format!("{}{}", out_t.join().unwrap(), err_t.join().unwrap())
        })
    }).collect();

    for (i, w) in watchers.into_iter().enumerate() {
        let combined = w.join().unwrap_or_else(|e| std::panic::resume_unwind(e));
        assert!(combined.contains("SUM:"), "party {} output missing SUM:\n{}", i, combined);
        let after = combined.rsplitn(2, "SUM:").next().unwrap();
        let token = after.trim().split_whitespace().next()
            .unwrap_or_else(|| panic!("no number after SUM: in party {} output", i));
        let num: u128 = token.parse()
            .unwrap_or_else(|_| panic!("parse sum failed for party {}: {}", i, combined));
        assert_eq!(num, 60u128, "wrong sum for party {}:\n{}", i, combined);
    }
}

// ── interactive MPC expressions via piped stdin ───────────────────────────────

/// A single test case sent to the party-0 REPL.
struct Case {
    expr:     &'static str,
    expected: i64,
}

/// Spawn 3 tcp_party processes in TCP mode with fresh ports, send `cases` to
/// party 0's stdin (one per line, followed by `:quit`), then assert that the
/// `=> <value>` lines on party 0's stdout match the expected values in order.
///
/// Workers (parties 1/2) have their stdout/stderr drained to avoid blocking.
fn run_cases(cases: &[Case]) {
    let mpc_addrs  = free_ports(3);
    let dist_addrs = free_ports(3);
    let bin_path   = find_bin();

    let mpc_str  = mpc_addrs.join(",");
    let dist_str = dist_addrs.join(",");

    // Build the stdin payload for party 0: one expression per line then :quit
    let mut party0_input = String::new();
    for c in cases { party0_input.push_str(c.expr); party0_input.push('\n'); }
    party0_input.push_str(":quit\n");

    // Spawn workers 1 and 2 first; they need to be listening before party 0 connects.
    let mut worker_children = Vec::new();
    for i in 1..3usize {
        let child = Command::new(bin_path)
            .arg(i.to_string())
            .arg(&mpc_str)
            .arg(&dist_str)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn worker tcp_party");
        worker_children.push(child);
        thread::sleep(Duration::from_millis(200));
    }

    // Spawn party 0 with piped stdin so we can drive it programmatically.
    let mut party0 = Command::new(bin_path)
        .arg("0")
        .arg(&mpc_str)
        .arg(&dist_str)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn party-0 tcp_party");

    // Write all input up-front; party 0 will process it line by line.
    {
        let mut stdin = party0.stdin.take().unwrap();
        stdin.write_all(party0_input.as_bytes()).expect("write to party 0 stdin");
        // stdin closed on drop → party 0 gets EOF after :quit
    }

    // Drain workers in background threads so their pipes don't block.
    let worker_threads: Vec<_> = worker_children.into_iter().enumerate().map(|(wi, mut child)| {
        thread::spawn(move || {
            let mut stdout = child.stdout.take().unwrap();
            let mut stderr = child.stderr.take().unwrap();
            let ot = thread::spawn(move || { let mut s = String::new(); let _ = stdout.read_to_string(&mut s); s });
            let et = thread::spawn(move || { let mut s = String::new(); let _ = stderr.read_to_string(&mut s); s });
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => { if Instant::now() > deadline { let _ = child.kill(); break; } thread::sleep(Duration::from_millis(20)); }
                    Err(_) => break,
                }
            }
            (wi + 1, ot.join().unwrap(), et.join().unwrap())
        })
    }).collect();

    // Collect party 0's output.
    let mut party0_stdout_reader = party0.stdout.take().unwrap();
    let mut party0_stderr_reader = party0.stderr.take().unwrap();
    let out_t = thread::spawn(move || { let mut s = String::new(); let _ = party0_stdout_reader.read_to_string(&mut s); s });
    let err_t = thread::spawn(move || { let mut s = String::new(); let _ = party0_stderr_reader.read_to_string(&mut s); s });

    // Wait for party 0 with a hard deadline.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match party0.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() > deadline {
                    let _ = party0.kill();
                    panic!("party 0 timed out waiting for expressions to complete");
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => panic!("wait error for party 0: {}", e),
        }
    }

    let party0_out = out_t.join().unwrap();
    let party0_err = err_t.join().unwrap();

    // Wait for workers (best-effort).
    for wt in worker_threads { let _ = wt.join(); }

    // Strip ANSI escape codes so we can match plain numbers.
    let clean: String = strip_ansi(&party0_out);

    // Collect lines of the form "=> <number>" in order.
    let results: Vec<i64> = clean.lines()
        .filter_map(|line| {
            let t = line.trim();
            // Match "=> 42" or "=> -7" etc.
            let t = if let Some(rest) = t.strip_prefix("=>") { rest.trim() } else { return None; };
            t.split_whitespace().next().and_then(|tok| tok.parse::<i64>().ok())
        })
        .collect();

    assert_eq!(
        results.len(), cases.len(),
        "expected {} results from party 0, got {}\nstdout:\n{}\nstderr:\n{}",
        cases.len(), results.len(), party0_out, party0_err
    );

    for (i, (got, case)) in results.iter().zip(cases.iter()).enumerate() {
        assert_eq!(
            *got, case.expected,
            "case {} {:?}: expected {}, got {}\nstdout:\n{}",
            i, case.expr, case.expected, got, party0_out
        );
    }
}

/// Strip ANSI/VT100 escape sequences (e.g. color codes) from a string.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // consume ESC [ ... <letter>
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&d) = chars.peek() {
                    chars.next();
                    if d.is_ascii_alphabetic() { break; }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

// ── actual test cases ─────────────────────────────────────────────────────────

#[test]
#[ignore]
fn tcp_mpc_basic_arithmetic() {
    run_cases(&[
        Case { expr: "(reconstruct (+ (share 10 32) (share 32 32)))", expected: 42 },
        Case { expr: "(reconstruct (* (share 6 32) (share 7 32)))",   expected: 42 },
        Case { expr: "(reconstruct (- (share 100 32) (share 58 32)))", expected: 42 },
    ]);
}

#[test]
#[ignore]
fn tcp_mpc_signed_values() {
    run_cases(&[
        Case { expr: "(reconstruct (share -42 32))",                           expected: -42 },
        Case { expr: "(reconstruct (+ (share -10 32) (share 52 32)))",         expected: 42  },
        Case { expr: "(reconstruct (- (share 3 32) (share 10 32)))",           expected: -7  },
        Case { expr: "(reconstruct (* (share -6 32) (share -7 32)))",          expected: 42  },
    ]);
}

#[test]
#[ignore]
fn tcp_mpc_oblivious_select() {
    run_cases(&[
        // oblivious max(30, 20) = 30
        Case {
            expr: "(let ((a (share 30 32)) (b (share 20 32)) (one (share 1 32))) (let ((m (> a b))) (reconstruct (+ (* a m) (* b (- one m))))))",
            expected: 30,
        },
        // oblivious min(30, 20) = 20
        Case {
            expr: "(let ((a (share 30 32)) (b (share 20 32)) (one (share 1 32))) (let ((m (< a b))) (reconstruct (+ (* a m) (* b (- one m))))))",
            expected: 20,
        },
    ]);
}

#[test]
#[ignore]
fn tcp_mpc_relu() {
    run_cases(&[
        Case { expr: "(let ((z (share 5 32))) (if (> z 0) z 0))",  expected: 5 },
        Case { expr: "(let ((z (share -4 32))) (if (> z 0) z 0))", expected: 0 },
        Case { expr: "(let ((z (share 0 32))) (if (> z 0) z 0))",  expected: 0 },
    ]);
}

#[test]
#[ignore]
fn tcp_mpc_private_inputs() {
    run_cases(&[
        Case { expr: "(reconstruct (+ (share-from 0 10 32) (share-from 1 32 32)))", expected: 42 },
        Case { expr: "(reconstruct (* (share-from 0 6 32) (share-from 1 7 32)))",   expected: 42 },
        Case {
            expr: "(reconstruct (+ (share-from 0 10 32) (+ (share-from 1 20 32) (share-from 2 30 32))))",
            expected: 60,
        },
    ]);
}

#[test]
#[ignore]
fn tcp_mpc_error_recovery() {
    // After a parse or runtime error the session must recover and subsequent
    // expressions must still evaluate correctly.
    //
    // Strategy: sandwich a bad expression between two good ones in the stdin
    // stream.  We drive this manually (not via run_cases) so we can pass
    // non-evaluable lines and still assert exactly 2 good results.
    let mpc_addrs  = free_ports(3);
    let dist_addrs = free_ports(3);
    let bin_path   = find_bin();
    let mpc_str    = mpc_addrs.join(",");
    let dist_str   = dist_addrs.join(",");

    // stdin: good expression, bad expression, good expression, :quit
    let party0_input = concat!(
        "(reconstruct (+ (share 10 32) (share 32 32)))\n",  // => 42
        "(bad expression (\n",                               // parse error → :abort
        "(reconstruct (* (share 6 32) (share 7 32)))\n",    // => 42
        ":quit\n",
    );

    let mut workers: Vec<_> = (1..3).map(|i| {
        let c = Command::new(bin_path)
            .arg(i.to_string()).arg(&mpc_str).arg(&dist_str)
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().expect("spawn worker");
        thread::sleep(Duration::from_millis(200));
        c
    }).collect();

    let mut party0 = Command::new(bin_path)
        .arg("0").arg(&mpc_str).arg(&dist_str)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().expect("spawn party 0");

    party0.stdin.take().unwrap().write_all(party0_input.as_bytes()).unwrap();

    // drain workers
    let _wt: Vec<_> = workers.iter_mut().map(|c| {
        let mut o = c.stdout.take().unwrap();
        let mut e = c.stderr.take().unwrap();
        thread::spawn(move || { let mut s = String::new(); let _ = o.read_to_string(&mut s); let _ = e.read_to_string(&mut s); })
    }).collect();

    let mut p0_out = party0.stdout.take().unwrap();
    let out_t = thread::spawn(move || { let mut s = String::new(); let _ = p0_out.read_to_string(&mut s); s });

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match party0.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => { if Instant::now() > deadline { let _ = party0.kill(); panic!("party 0 timed out"); } thread::sleep(Duration::from_millis(20)); }
            Err(e) => panic!("wait: {}", e),
        }
    }

    let raw = out_t.join().unwrap();
    let clean = strip_ansi(&raw);

    // Collect "=> <number>" lines
    let results: Vec<i64> = clean.lines()
        .filter_map(|line| {
            let t = line.trim();
            let t = t.strip_prefix("=>")?;
            t.trim().split_whitespace().next()?.parse::<i64>().ok()
        })
        .collect();

    assert_eq!(results.len(), 2,
        "expected 2 results (first good, bad errored, second good)\nstdout:\n{}", raw);
    assert_eq!(results[0], 42, "first expression result\nstdout:\n{}", raw);
    assert_eq!(results[1], 42, "second expression result after error recovery\nstdout:\n{}", raw);

    // Also verify that an error line was reported (parse error or runtime error)
    assert!(clean.contains("parse error") || clean.contains("runtime error"),
        "expected an error line in output\nstdout:\n{}", raw);
}
