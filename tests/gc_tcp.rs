//! Real-socket 3-party check of the garbled-circuit **Yao leg**
//! (`baru::crypto::gc_protocol`) over `TcpTransport` on localhost: two garblers
//! (parties 0/1) + one evaluator (party 2), each in its own thread with a real
//! TCP connection to the others. Proves the two-garbler seed sync, the OT-free
//! label delivery, and the Y2A resharing all survive real (ordered, framed)
//! transport — not just the in-process `InprocHandle` used by the unit tests.
//!
//! Marked `#[ignore]` like the other socket tests in this crate (it binds real
//! ports); run with `cargo test --test gc_tcp -- --ignored`.

use baru::crypto::aby3_protocol::{reveal, share_input, PartyShare};
use baru::crypto::gc_protocol::{gc_id_block, gc_lt_batch, gc_relu_batch};
use baru::crypto::net::{Mailbox, TcpTransport, Transport};
use std::thread;

/// Reserve `n` free localhost ports and return their addresses.
fn free_ports(n: usize) -> Vec<String> {
    let listeners: Vec<_> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addrs: Vec<String> = listeners
        .iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect();
    drop(listeners);
    addrs
}

/// Share `vals` (owner party 0) as RSS over the mailbox, ids `base..base+len`.
fn share_vec<T: Transport>(
    vals: &[i64],
    nbits: usize,
    base: u64,
    me: usize,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let mask = (1i128 << nbits) - 1;
    vals.iter()
        .enumerate()
        .map(|(i, &v)| {
            let val = if me == 0 { (v as i128 & mask) as u128 } else { 0 };
            share_input(val, base + i as u64, 0, me, mbx)
        })
        .collect()
}

#[test]
#[ignore]
fn gc_lt_over_tcp_3party() {
    let addrs = free_ports(3);
    let nbits = 20usize;
    let a_vals = vec![3i64, -4, 100, -100, 7, 500, -500, 0];
    let b_vals = vec![5i64, -4, 99, -50, 7, -500, 500, 1];
    let want: Vec<u128> = (0..a_vals.len()).map(|i| (a_vals[i] < b_vals[i]) as u128).collect();

    let mut handles = Vec::new();
    for me in 0..3usize {
        let addrs = addrs.clone();
        let av = a_vals.clone();
        let bv = b_vals.clone();
        let want = want.clone();
        handles.push(thread::spawn(move || {
            let t = TcpTransport::new(me, addrs);
            assert!(t.wait_ready(30_000), "party {me}: peers not ready");
            let mbx = Mailbox::new(t);
            let n = av.len();
            let a = share_vec(&av, nbits, 1, me, &mbx);
            let b = share_vec(&bv, nbits, 1 + n as u64, me, &mbx);
            let op_id = 100_000u64;
            let lt = gc_lt_batch(&a, &b, nbits, op_id, me, &mbx);
            let rid = op_id + 1 + gc_id_block(n, 1);
            let bits: Vec<u128> = lt
                .iter()
                .enumerate()
                .map(|(i, s)| reveal(s, rid + i as u64, None, me, &mbx) & 1)
                .collect();
            assert_eq!(bits, want, "party {me}: gc_lt over TCP");
        }));
    }
    for h in handles {
        h.join().expect("gc TCP party thread panicked");
    }
}

#[test]
#[ignore]
fn gc_relu_over_tcp_3party() {
    let addrs = free_ports(3);
    let nbits = 20usize;
    let shift = 8usize;
    let vals = vec![0i64, 100, -1, -4000, 1 << 17, 12345, -12345, 65536];
    let want: Vec<i64> = vals.iter().map(|&v| if v >= 0 { v >> shift } else { 0 }).collect();

    let mut handles = Vec::new();
    for me in 0..3usize {
        let addrs = addrs.clone();
        let vv = vals.clone();
        let want = want.clone();
        handles.push(thread::spawn(move || {
            let t = TcpTransport::new(me, addrs);
            assert!(t.wait_ready(30_000), "party {me}: peers not ready");
            let mbx = Mailbox::new(t);
            let n = vv.len();
            let v = share_vec(&vv, nbits, 1, me, &mbx);
            let op_id = 200_000u64;
            let acts = gc_relu_batch(&v, nbits, shift, op_id, me, &mbx);
            let width = nbits - shift;
            let rid = op_id + 1 + gc_id_block(n, width);
            let out: Vec<i64> = acts
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let raw = reveal(s, rid + i as u64, None, me, &mbx);
                    let m = (1i128 << nbits) - 1;
                    let x = raw as i128 & m;
                    if x >= (1i128 << (nbits - 1)) { (x - (1i128 << nbits)) as i64 } else { x as i64 }
                })
                .collect();
            assert_eq!(out, want, "party {me}: gc_relu over TCP");
        }));
    }
    for h in handles {
        h.join().expect("gc TCP party thread panicked");
    }
}
