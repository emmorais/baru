//! `baru bench-falcon` — Falcon's sign extraction against Baru's, on one wire.
//!
//! # What is compared, and why this operation
//!
//! Every private-inference system has to answer "is this shared value
//! negative?", and on the networks in this repo that question *is* the cost:
//! LeNet's traffic is essentially all DReLU (see
//! [falcon-reproduction.md](../../docs/falcon-reproduction.md)). So the sign is
//! the operation worth putting head-to-head, and it is the one where the three
//! routes genuinely differ:
//!
//! | route | how the sign is obtained |
//! |---|---|
//! | `falcon` | `Π_WA` wrap3 on doubled shares, over `Z_p` ([`crate::crypto::falcon_mpc`]) |
//! | `baru-a2b` | full bit decomposition, MSB of `a − 0` |
//! | `baru-gc`  | garbled comparator |
//!
//! All three produce the same bit from the same inputs over the **same**
//! transport, so message and byte counts are directly comparable. The harness
//! checks the three agree before reporting anything: a cost comparison between
//! routes computing different things is meaningless, which is the rule
//! [`crate::bench::abstraction`] already follows.
//!
//! # Why *not* whole-ReLU, yet
//!
//! Falcon's ReLU is `max(0,a)`; Baru's is a **fused** `relu∘rescale`. Those are
//! different functions, and comparing them would be comparing a truncation into
//! existence. Making that fair needs Falcon's truncation on the transport
//! (`crypto::falcon::truncate_exact` is still a local simulation), so it is
//! deliberately out of scope here rather than quietly folded in.
//!
//! # Reading the numbers
//!
//! Round counts are exact and comparable. **Byte counts are not directly
//! comparable across the ring/field boundary**: `Z_p` traffic is bit-packed at
//! 6 bits, ABY3 ring resharing is `Vec<u128>` — 16 B per element whatever
//! `nbits` is. The report prints both totals and the ring/garbled split so the
//! encoding is visible rather than baked into a ratio.

use std::time::Instant;

use serde::Serialize;

use crate::bench::comm::{CommStats, CountingTransport, NetworkProfile};
use crate::crypto::aby3_protocol::PartyShare;
use crate::crypto::falcon_mpc::{drelu_batch, WrapPreproc, PC_ID_BLOCK};
use crate::crypto::field::Fp;
use crate::crypto::fp_protocol::share_component as fp_component;
use crate::crypto::net::{InprocHandle, Mailbox, PartyId};
use crate::bench::hybrid_nn::{run_parties, MpcTransport};
use crate::interpreter::backend::Backend;
use crate::interpreter::transport_backend::TransportBackend;

/// One route's measured cost for `n` sign extractions.
#[derive(Debug, Clone, Serialize)]
pub struct RouteCost {
    pub route: String,
    pub messages: u64,
    pub bytes: u64,
    pub ring_bytes: u64,
    pub gc_bytes: u64,
    pub ms: f64,
}

#[derive(Debug, Serialize)]
pub struct FalconReport {
    pub n: usize,
    pub l_bits: u32,
    pub routes: Vec<RouteCost>,
    /// All routes agreed on every sign bit.
    pub agree: bool,
}

/// Deterministic three-way split of a ring value.
fn split(v: u64, l: u64) -> [u64; 3] {
    let s0 = v.wrapping_mul(37) % l;
    let s1 = v.wrapping_mul(91).wrapping_add(11) % l;
    [s0, s1, (v + 2 * l - s0 - s1) % l]
}

/// Preprocessing for one `Π_WA`, generated centrally as a preprocessing
/// functionality would. `δ` is not derivable by any single party.
fn wrap_preproc(seed: u64, l_bits: u32, me: PartyId) -> WrapPreproc {
    let l = 1u64 << l_bits;
    let xs = [
        seed.wrapping_mul(53).wrapping_add(7) % l,
        seed.wrapping_mul(29).wrapping_add(3) % l,
        seed.wrapping_mul(17).wrapping_add(5) % l,
    ];
    let mask = (1u128 << l_bits) - 1;
    let x = ((xs[0] as u128 + xs[1] as u128 + xs[2] as u128) & mask) as u64;
    let beta = seed & 1;
    WrapPreproc {
        x: PartyShare { left: xs[me % 3] as u128, right: xs[(me + 1) % 3] as u128 },
        bits: (0..l_bits).map(|i| fp_component(Fp::new((x >> i) & 1), 0, me)).collect(),
        delta: {
            let d = crate::crypto::falcon::wrap3(xs[0], xs[1], xs[2], l_bits);
            [u64::from(me == 0) * d, u64::from(me == 2) * d]
        },
        beta_fp: fp_component(Fp::new(beta), 0, me),
        beta_bit: [u64::from(me == 0) * beta, u64::from(me == 2) * beta],
        m: fp_component(Fp::ONE, 0, me),
    }
}

/// Run a closure on three parties over a counted transport, returning each
/// party's result plus **one party's** traffic.
fn measure<R, F>(f: F) -> (Vec<R>, CommStats, f64)
where
    R: Send + 'static,
    F: Fn(PartyId, &Mailbox<CountingTransport<InprocHandle>>) -> R + Send + Sync + 'static,
{
    let handles = InprocHandle::create(3);
    let f = std::sync::Arc::new(f);
    let mut comm = None;
    let mut joins = Vec::new();
    for h in handles {
        let counted = CountingTransport::new(h, 3);
        if comm.is_none() {
            comm = Some(counted.handle());
        }
        let f = f.clone();
        joins.push(std::thread::spawn(move || {
            let mbx = Mailbox::new(counted);
            let me = mbx.id();
            f(me, &mbx)
        }));
    }
    let t0 = Instant::now();
    let out: Vec<R> = joins.into_iter().map(|j| j.join().expect("party panicked")).collect();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (out, comm.unwrap().stats(), ms)
}

/// Compare the three routes on `n` sign extractions at ring width `l_bits`.
pub fn run(n: usize, l_bits: u32, profile: NetworkProfile) -> FalconReport {
    let l = 1u64 << l_bits;
    let vals: Vec<u64> = (0..n).map(|i| (i as u64).wrapping_mul(0x9E37) % l).collect();
    let want: Vec<u64> = vals
        .iter()
        .map(|&v| u64::from(crate::crypto::falcon::signed(v, l_bits) >= 0))
        .collect();

    // ── Falcon: Π_WA on doubled shares ────────────────────────────────────
    let vs = vals.clone();
    let (falcon_out, falcon_comm, falcon_ms) = measure(move |me, mbx| {
        let l = 1u64 << l_bits;
        let a: Vec<PartyShare> = vs
            .iter()
            .map(|&v| {
                let s = split(v, l);
                PartyShare { left: s[me % 3] as u128, right: s[(me + 1) % 3] as u128 }
            })
            .collect();
        let pre: Vec<WrapPreproc> =
            (0..vs.len()).map(|i| wrap_preproc(i as u64, l_bits, me)).collect();
        drelu_batch(&a, &pre, l_bits, PC_ID_BLOCK, me, mbx)
    });
    let falcon_bits: Vec<u64> = (0..n)
        .map(|i| (falcon_out[0][i][0] ^ falcon_out[1][i][0] ^ falcon_out[2][i][0]) & 1)
        .collect();

    // ── Baru: MSB via the backend's comparator, a2b and garbled ───────────
    let mut routes = vec![RouteCost {
        route: "falcon (wrap3)".into(),
        messages: falcon_comm.messages,
        bytes: falcon_comm.bytes,
        ring_bytes: falcon_comm.ring_bytes,
        gc_bytes: falcon_comm.gc_bytes,
        ms: falcon_ms,
    }];
    let mut agree = falcon_bits == want;

    for (label, gc) in [("baru-a2b", false), ("baru-gc", true)] {
        let vs = vals.clone();
        let t0 = Instant::now();
        let (out, comm) = run_parties(profile, move |_me| {
            let vs = vs.clone();
            move |be: &mut TransportBackend<MpcTransport>| {
                be.set_gc_nonlinearities(gc);
                // Inputs enter as *public* shares: free, exactly as Falcon's
                // `share_component` is, so neither side is charged for getting
                // its operands into shared form. What is measured is the
                // comparison alone.
                let zero = be.share_public(0, l_bits as usize);
                let pairs: Vec<_> = vs
                    .iter()
                    .map(|&v| (be.share_public(v as u128, l_bits as usize), zero.clone()))
                    .collect();
                // `lt(a, 0)` is the MSB of `a` — the same bit Falcon's DReLU
                // produces, complemented.
                let bits = be.lt_batch(&pairs);
                // Return *shares*, exactly as Falcon's `drelu_batch` does.
                // Revealing here would charge this route one opening per
                // element — 2n messages — for an output the other side leaves
                // shared. The reconstruction happens outside the measurement.
                bits.iter().map(|b| be.my_additive_component(b)).collect::<Vec<_>>()
            }
        });
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        // Reconstruct from the three parties' additive components, then
        // complement: `lt(a,0) = 1` means negative, so DReLU is its inverse.
        let got: Vec<u64> = (0..n)
            .map(|i| 1 - ((out[0][i].wrapping_add(out[1][i]).wrapping_add(out[2][i])) as u64 & 1))
            .collect();
        agree &= got == want;
        routes.push(RouteCost {
            route: label.into(),
            messages: comm.messages,
            bytes: comm.bytes,
            ring_bytes: comm.ring_bytes,
            gc_bytes: comm.gc_bytes,
            ms,
        });
    }

    FalconReport { n, l_bits, routes, agree }
}

const USAGE: &str = "usage: baru bench-falcon [--n N] [--bits L] [--json]

Falcon's sign extraction against Baru's own routes, on one transport.
  --n N      how many signs to extract in one batch (default 1024)
  --bits L   ring width (default 32, Falcon's own)
  --json     machine-readable output
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let (mut n, mut bits, mut json) = (1024usize, 32u32, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" => n = it.next().ok_or("--n needs a value")?.parse().map_err(|_| "bad --n")?,
            "--bits" => {
                bits = it.next().ok_or("--bits needs a value")?.parse().map_err(|_| "bad --bits")?
            }
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }
    let r = run(n, bits, NetworkProfile::INSTANT);
    if json {
        println!("{}", serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?);
        return Ok(());
    }
    println!("Sign extraction: Falcon's wrap3 vs Baru's routes");
    println!("n = {}, ring width = {} bits, same transport\n", r.n, r.l_bits);
    println!(
        "  {:<16}{:>10}{:>12}{:>12}{:>12}{:>10}",
        "route", "messages", "bytes", "ring B", "garbled B", "ms"
    );
    for c in &r.routes {
        println!(
            "  {:<16}{:>10}{:>12}{:>12}{:>12}{:>10.1}",
            c.route, c.messages, c.bytes, c.ring_bytes, c.gc_bytes, c.ms
        );
    }
    println!(
        "\n  all routes agree on every sign bit: {}",
        if r.agree { "yes" } else { "NO — costs are not comparable" }
    );
    println!(
        "\n  Bytes are not directly comparable across the ring/field boundary:\n  \
         Z_p traffic is bit-packed at 6 bits, ABY3 resharing is 16 B/element\n  \
         whatever `nbits` is. Compare messages, and read the byte split."
    );
    Ok(())
}
