use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::thread;
use std::path::Path;

// Integration test that spawns three tcp_party processes and verifies they
// exchange D messages over TCP and each prints the correct SUM. Uses
// ephemeral ports reserved briefly before spawning and waits with timeouts so
// it is CI-friendly. The binary `tcp_party` must be built by CI (either in
// release or debug); this test will look for release binary first, then debug.

#[test]
fn tcp_process_integration() {
    // reserve ephemeral ports by binding to :0 and reading assigned port
    let mut listeners = Vec::new();
    for _ in 0..3 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind ephemeral port");
        listeners.push(l);
    }
    let addrs: Vec<String> = listeners.iter().map(|l| l.local_addr().unwrap().to_string()).collect();
    // drop listeners so spawned processes can bind to the same ports
    drop(listeners);

    // locate the tcp_party binary (prefer release, fallback to debug)
    let release_bin = Path::new("./target/release/tcp_party");
    let debug_bin = Path::new("./target/debug/tcp_party");
    let bin_path = if release_bin.exists() {
        release_bin
    } else if debug_bin.exists() {
        debug_bin
    } else {
        panic!("tcp_party binary not found; please run `cargo build --release` before running this test");
    };

    let addrs_str = addrs.join(",");

    // spawn three processes with D values 10,20,30
    let mut children = Vec::new();
    for i in 0..3 {
        let arg_id = i.to_string();
        let arg_addrs = addrs_str.clone();
        let arg_d = match i {
            0 => "10".to_string(),
            1 => "20".to_string(),
            _ => "30".to_string(),
        };
        let child = Command::new(bin_path)
            .arg(arg_id)
            .arg(arg_addrs)
            .arg(arg_d)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn tcp_party");
        children.push(child);
        // small stagger to allow listener threads inside the binary to bind
        // increase from 50ms to 200ms to reduce flakiness on slower CI machines
        thread::sleep(Duration::from_millis(200));
    }

    // wait for children to finish with timeout and collect outputs
    let mut sums = Vec::new();
    for mut c in children {
        let start = Instant::now();
        let timeout = Duration::from_secs(8);
        loop {
            match c.try_wait() {
                Ok(Some(_status)) => {
                    let out = c.wait_with_output().expect("wait_with_output failed");
                    let mut combined = String::new();
                    combined.push_str(&String::from_utf8_lossy(&out.stdout));
                    combined.push_str(&String::from_utf8_lossy(&out.stderr));
                    sums.push(combined);
                    break;
                }
                Ok(None) => {
                    if start.elapsed() > timeout {
                        // kill and fail the test
                        let _ = c.kill();
                        panic!("tcp_party child timed out");
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Err(e) => panic!("error waiting for child: {}", e),
            }
        }
    }

    // each process should have printed a SUM: 60
    for s in sums {
        assert!(s.contains("SUM:"), "output missing SUM: {}", s);
        // parse the last SUM occurrence (some binaries may print intermediate SUMs)
        let after_last = s.rsplitn(2, "SUM:").next().unwrap();
        let first_token = after_last.trim().split_whitespace().next().expect("no number after SUM:");
        let num: u128 = first_token.parse().expect("parse sum");
        assert_eq!(num, 60u128, "unexpected sum from child output:\n{}", s);
    }
}
