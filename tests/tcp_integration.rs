use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::thread;
use std::path::Path;
use std::io::Read;

#[test]
#[ignore]
fn tcp_process_integration() {
    // reserve three ephemeral ports
    let mut listeners = Vec::new();
    for _ in 0..3 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listeners.push(l);
    }
    let addrs: Vec<String> = listeners.iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect();
    drop(listeners); // free ports for children

    let release_bin = Path::new("./target/release/tcp_party");
    let debug_bin   = Path::new("./target/debug/tcp_party");
    let bin_path = if release_bin.exists() { release_bin }
                   else if debug_bin.exists() { debug_bin }
                   else { panic!("tcp_party binary not found; run `cargo build` first") };

    let addrs_str = addrs.join(",");

    // spawn three children with a small stagger so each listener is ready
    // before the next party starts connecting.
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

    // Watch all three children concurrently.  Draining output in a sub-thread
    // prevents the OS pipe buffer from filling up and stalling a child that
    // another child is waiting to hear from.
    let watchers: Vec<_> = children.into_iter().enumerate().map(|(i, mut child)| {
        thread::spawn(move || {
            let mut stdout = child.stdout.take().unwrap();
            let mut stderr = child.stderr.take().unwrap();

            // drain pipes so the child never blocks on writes
            let out_t = thread::spawn(move || {
                let mut s = String::new();
                let _ = stdout.read_to_string(&mut s);
                s
            });
            let err_t = thread::spawn(move || {
                let mut s = String::new();
                let _ = stderr.read_to_string(&mut s);
                s
            });

            // wait for the process with a hard timeout
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => {
                        if Instant::now() > deadline {
                            let _ = child.kill();
                            panic!("party {} timed out", i);
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(e) => panic!("wait error for party {}: {}", i, e),
                }
            }

            let out = out_t.join().unwrap();
            let err = err_t.join().unwrap();
            format!("{}{}", out, err)
        })
    }).collect();

    // collect results and assert SUM: 60 from each party
    for (i, w) in watchers.into_iter().enumerate() {
        let combined = w.join().unwrap_or_else(|e| {
            std::panic::resume_unwind(e)
        });
        assert!(combined.contains("SUM:"),
            "party {} output missing SUM:\n{}", i, combined);
        let after  = combined.rsplitn(2, "SUM:").next().unwrap();
        let token  = after.trim().split_whitespace().next()
            .unwrap_or_else(|| panic!("no number after SUM: in party {} output", i));
        let num: u128 = token.parse()
            .unwrap_or_else(|_| panic!("parse sum failed for party {}: {}", i, combined));
        assert_eq!(num, 60u128, "wrong sum for party {}:\n{}", i, combined);
    }
}




