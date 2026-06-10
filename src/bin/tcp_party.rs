use std::env;
use std::time::Duration;
use std::thread;
use baru::crypto::net::{Transport, serialize_msg, deserialize_msg, MpcMsg};

fn main() {
    // args: <id> <addrs_comma> <d_value>
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: tcp_party <id> <addrs_comma> <d_value>");
        std::process::exit(2);
    }
    let id: usize = args[1].parse().expect("invalid id");
    let addrs: Vec<String> = args[2].split(',').map(|s| s.to_string()).collect();
    let d_val: u128 = args[3].parse().expect("invalid d");

    // create transport and broadcast D
    let transport = baru::crypto::net::TcpTransport::new(id, addrs);

    // broadcast our D
    transport.broadcast(serialize_msg(&MpcMsg::D(d_val)));

    // collect two incoming Ds
    let mut got = vec![false; 3];
    got[id] = true;
    let mut d_shares = vec![0u128; 3];
    d_shares[id] = d_val;
    let mut pending = 2;
    let start = std::time::Instant::now();
    while pending > 0 && start.elapsed() < Duration::from_secs(5) {
        if let Some((from, msg)) = transport.recv() {
            let m: MpcMsg = deserialize_msg(&msg);
            if let MpcMsg::D(v) = m {
                if !got[from] { got[from] = true; d_shares[from] = v; pending -= 1; }
            }
        } else {
            thread::sleep(Duration::from_millis(10));
        }
    }

    let mut sum: u128 = 0;
    for v in d_shares { sum = sum.wrapping_add(v); }
    println!("SUM: {}", sum);
}
