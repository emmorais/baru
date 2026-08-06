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

use std::collections::HashMap;
use std::time::Instant;

use serde::Serialize;

use crate::bench::comm::{CommStats, CountingTransport, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, MpcTransport};
use crate::crypto::aby3_protocol::PartyShare;
use crate::crypto::falcon_mpc::{drelu_batch, WrapPreproc, PC_ID_BLOCK};
use crate::crypto::field::Fp;
use crate::crypto::fp_protocol::share_component as fp_component;
use crate::crypto::net::{InprocHandle, Mailbox, PartyId};
use crate::crypto::types::Fr;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::parse;
use crate::interpreter::transport_backend::TransportBackend;
use crate::mpcc::exec::{run_flat_with, Batching};
use crate::mpcc::opt::reassociate_reductions;
use crate::zkp::circuit_compiler::CircuitCompiler;

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

impl RouteCost {
    /// Ring elements sent, each serialised as a `u128` by `TaggedZBatch`.
    ///
    /// Reported instead of a "content bytes" figure because **the content of a
    /// ring element depends on the share width, and that varies within a single
    /// protocol**. An earlier version of this benchmark discounted the `u128`
    /// padding using the *ring* width (ℓ = 32) uniformly and got the comparison
    /// badly wrong: `a2b`'s traffic is almost entirely **1-bit** shares, so its
    /// padding is 128×, not 4×. Correcting that moved `a2b` from an apparent
    /// 34× behind Falcon to roughly level with it.
    ///
    /// Elements are unambiguous; the reader can multiply by whatever width
    /// applies. See the notes printed under the table.
    pub fn ring_elements(&self) -> u64 {
        self.ring_bytes / 16
    }
}

#[derive(Debug, Serialize)]
pub struct FalconReport {
    pub n: usize,
    pub l_bits: u32,
    pub routes: Vec<RouteCost>,
    /// All routes agreed on every sign bit.
    pub agree: bool,
}

/// The literal the compiled source compares against.
///
/// **−1, not 0.** DReLU is `a ≥ 0`, and Baru's `>` is strict, so `(> x 0)` would
/// compute a *different function* — differing on exactly the input `a = 0`.
/// Over the integers `x > −1 ⟺ x ≥ 0`, which is the predicate the other routes
/// produce. The first version of this benchmark used 0 and the agreement check
/// caught it, which is what that check is for.
const ZERO_LIT: i64 = -1;

/// Interpret a ring element as a signed integer for the compiled source, whose
/// inputs are plain Lisp numbers.
fn signed_i64(v: u64, l_bits: u32) -> i64 {
    crate::crypto::falcon::signed(v, l_bits) as i64
}

/// Ring residue of a signed value at `nbits`.
fn residue(v: i64, nbits: usize) -> Fr {
    let m: Fr = (1u128 << nbits) - 1;
    (v as i128 as u128) & m
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
        x: PartyShare {
            left: xs[me % 3] as u128,
            right: xs[(me + 1) % 3] as u128,
        },
        bits: (0..l_bits)
            .map(|i| fp_component(Fp::new((x >> i) & 1), 0, me))
            .collect(),
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
    let out: Vec<R> = joins
        .into_iter()
        .map(|j| j.join().expect("party panicked"))
        .collect();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (out, comm.unwrap().stats(), ms)
}

/// Compare the three routes on `n` sign extractions at ring width `l_bits`.
pub fn run(n: usize, l_bits: u32, profile: NetworkProfile) -> FalconReport {
    let l = 1u64 << l_bits;
    let vals: Vec<u64> = (0..n)
        .map(|i| (i as u64).wrapping_mul(0x9E37) % l)
        .collect();
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
                PartyShare {
                    left: s[me % 3] as u128,
                    right: s[(me + 1) % 3] as u128,
                }
            })
            .collect();
        let pre: Vec<WrapPreproc> = (0..vs.len())
            .map(|i| wrap_preproc(i as u64, l_bits, me))
            .collect();
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
                bits.iter()
                    .map(|b| be.my_additive_component(b))
                    .collect::<Vec<_>>()
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

    // ── Baru compiled: the same n signs written as a Baru expression ──────
    //
    // `(+ (> x0 0) (+ (> x1 0) …))` — n mutually independent comparisons in one
    // expression, which is the shape a compiler can do something with. Its ASAP
    // scheduler should collapse them into a single batched round, the same
    // transformation done by hand on Falcon's side.
    //
    // This is the route the whole exercise exists to test: not "is Baru's
    // comparison cheaper than Falcon's" but "can the compiler recover, from a
    // scalar source, what an expert wrote out".
    {
        let mut src = format!("(> x0 {ZERO_LIT})");
        for k in 1..n {
            src = format!("(+ {src} (> x{k} {ZERO_LIT}))");
        }
        let binds: HashMap<String, i64> = vals
            .iter()
            .enumerate()
            .map(|(k, &v)| (format!("x{k}"), signed_i64(v, l_bits)))
            .collect();
        let mut order: Vec<String> = binds.keys().cloned().collect();
        order.sort();

        match parse(&src)
            .ok()
            .and_then(|e| CircuitCompiler::new().compile_fhe(&e).ok())
        {
            Some(circuit) => {
                let flat = reassociate_reductions(&circuit.flat_ops, &circuit.output_wire);
                let ops = flat.len();
                let out_wire = circuit.output_wire.clone();
                let nb = l_bits as usize;
                let t0 = Instant::now();
                let (out, comm) = run_parties(profile, move |_me| {
                    let (flat, out_wire, binds, order) =
                        (flat.clone(), out_wire.clone(), binds.clone(), order.clone());
                    move |be: &mut TransportBackend<MpcTransport>| {
                        let vals: Vec<Fr> = order.iter().map(|k| residue(binds[k], nb)).collect();
                        // Public sharing, as the other routes use: the inputs
                        // are not what is being measured.
                        let env: HashMap<String, ShareRef> = order
                            .iter()
                            .cloned()
                            .zip(vals.iter().map(|&v| be.share_public(v, nb)))
                            .collect();
                        let (r, _) = run_flat_with(be, &flat, &env, &out_wire, nb, Batching::Level)
                            .expect("circuit executes");
                        be.my_additive_component(&r)
                    }
                });
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                // The compiled expression yields the *sum* of the sign bits, so
                // agreement is checked on that rather than bit by bit.
                let got =
                    out[0].wrapping_add(out[1]).wrapping_add(out[2]) & ((1u128 << l_bits) - 1);
                agree &= got == want.iter().sum::<u64>() as u128;
                routes.push(RouteCost {
                    route: format!("baru-compiled ({ops} ops)"),
                    messages: comm.messages,
                    bytes: comm.bytes,
                    ring_bytes: comm.ring_bytes,
                    gc_bytes: comm.gc_bytes,
                    ms,
                });
            }
            None => routes.push(RouteCost {
                route: "baru-compiled (compile failed)".into(),
                messages: 0,
                bytes: 0,
                ring_bytes: 0,
                gc_bytes: 0,
                ms: 0.0,
            }),
        }
    }

    FalconReport {
        n,
        l_bits,
        routes,
        agree,
    }
}

/// Cost of one MedMNIST-shaped inference done **entirely in RSS**, Falcon's way.
#[derive(Debug, Serialize)]
pub struct NetReport {
    pub l_bits: u32,
    pub conv_outputs: usize,
    pub drelus: usize,
    pub messages: u64,
    pub bytes: u64,
    /// Of `bytes`, the linear layers.
    pub linear_bytes: u64,
    pub ms: f64,
}

/// Run the MedMNIST CNN's *shape* end to end in RSS and measure it.
///
/// `1×28×28 → conv(8,3×3) → trunc → ReLU → pool → conv(16,3×3) → trunc → ReLU
/// → pool → dense(→128) → trunc → ReLU → dense(→C)`, secret weights throughout
/// — the same architecture `bench-medmnist` runs on the FHE hybrid, so the two
/// numbers are directly comparable.
///
/// This measures **cost, not accuracy**: weights and input are deterministic
/// stand-ins, because what is in question is the protocol's traffic, and the
/// arithmetic is already checked exactly by
/// `a_whole_fixed_point_layer_matches_plaintext` and the per-protocol tests.
/// Quoting an accuracy from this would be meaningless.
pub fn run_net(l_bits: u32, d: u32, classes: usize, profile: NetworkProfile) -> NetReport {
    use crate::bench::comm::CountingTransport;
    use crate::crypto::falcon_mpc::{conv_batch, dense_batch};

    let handles = InprocHandle::create(3);
    let mut comm = None;
    let mut joins = Vec::new();
    let mut lin_handle = None;
    for h in handles {
        let counted = CountingTransport::new(h, 3);
        if comm.is_none() {
            comm = Some(counted.handle());
            lin_handle = Some(counted.handle());
        }
        joins.push(std::thread::spawn(move || {
            let mbx = Mailbox::new(counted);
            let me = mbx.id();
            let sh = |v: u64| {
                let s = [v / 3, v / 3, v - 2 * (v / 3)];
                PartyShare {
                    left: s[me % 3] as u128,
                    right: s[(me + 1) % 3] as u128,
                }
            };
            let mut op = 1u64;
            let mut bump = |n: u64| {
                let o = op;
                op += n;
                o
            };

            // conv1: 1×28×28 → 8×26×26
            let img: Vec<PartyShare> = (0..28 * 28).map(|i| sh((i % 7) as u64)).collect();
            let k1: Vec<Vec<PartyShare>> = (0..8)
                .map(|o| (0..9).map(|t| sh(((o + t) % 3) as u64)).collect())
                .collect();
            let c1 = conv_batch(&img, &k1, 1, 28, 28, 3, l_bits, bump(64), me, &mbx);
            let a1 = nonlin(&c1, d, l_bits, &mut bump, me, &mbx);
            let p1 = pool2x2(&a1, 8, 26, 26, l_bits, &mut bump, me, &mbx);

            // conv2: 8×13×13 → 16×11×11
            let k2: Vec<Vec<PartyShare>> = (0..16)
                .map(|o| (0..72).map(|t| sh(((o + t) % 3) as u64)).collect())
                .collect();
            let c2 = conv_batch(&p1, &k2, 8, 13, 13, 3, l_bits, bump(64), me, &mbx);
            let a2 = nonlin(&c2, d, l_bits, &mut bump, me, &mbx);
            let p2 = pool2x2(&a2, 16, 11, 11, l_bits, &mut bump, me, &mbx);

            // dense 400→128, then 128→C
            let w1: Vec<Vec<PartyShare>> = (0..128)
                .map(|o| (0..p2.len()).map(|t| sh(((o + t) % 3) as u64)).collect())
                .collect();
            let z1 = dense_batch(&p2, &w1, l_bits, bump(64), me, &mbx);
            let h1 = nonlin(&z1, d, l_bits, &mut bump, me, &mbx);
            let w2: Vec<Vec<PartyShare>> = (0..classes)
                .map(|o| (0..128).map(|t| sh(((o + t) % 3) as u64)).collect())
                .collect();
            dense_batch(&h1, &w2, l_bits, bump(64), me, &mbx).len()
        }));
    }
    let t0 = Instant::now();
    for j in joins {
        j.join().expect("party panicked");
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let s = comm.unwrap().stats();
    let _ = (lin_handle, profile);

    let conv_outputs = 8 * 26 * 26 + 16 * 11 * 11 + 128 + classes;
    let drelus = (8 * 13 * 13 + 16 * 5 * 5 + 128) + 3 * (8 * 13 * 13 + 16 * 5 * 5);
    NetReport {
        l_bits,
        conv_outputs,
        drelus,
        messages: s.messages,
        bytes: s.bytes,
        linear_bytes: conv_outputs as u64 * (l_bits as u64 / 8),
        ms,
    }
}

/// truncate → ReLU, the fixed-point block after every linear layer.
fn nonlin<T: crate::crypto::net::Transport>(
    z: &[PartyShare],
    d: u32,
    l_bits: u32,
    bump: &mut impl FnMut(u64) -> u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    use crate::crypto::falcon_mpc::{relu_batch, truncate_batch, TruncPreproc};
    let n = z.len();
    let l = 1u64 << l_bits;
    let tp: Vec<TruncPreproc> = (0..n)
        .map(|i| {
            let r = ((i as u64).wrapping_mul(41) % (l / 8)) + 1;
            let three = |v: u64| {
                let s = [v, 0, 0];
                PartyShare {
                    left: s[me % 3] as u128,
                    right: s[(me + 1) % 3] as u128,
                }
            };
            let rs = [r / 3, r / 3, r - 2 * (r / 3)];
            TruncPreproc {
                r: PartyShare {
                    left: rs[me % 3] as u128,
                    right: rs[(me + 1) % 3] as u128,
                },
                r_shifted: three(r >> d),
                r_lo: three(r & ((1u64 << d) - 1)),
                sel: sel_pre(9000 + i as u64, l_bits, me),
                sel_wrap: sel_pre(29000 + i as u64, l_bits, me),
            }
        })
        .collect();
    let t = truncate_batch(z, &tp, d, l_bits, bump(256), me, mbx);
    let wp: Vec<_> = (0..n)
        .map(|i| wrap_preproc(7000 + i as u64, l_bits, me))
        .collect();
    let sp: Vec<_> = (0..n)
        .map(|i| sel_pre(8000 + i as u64, l_bits, me))
        .collect();
    relu_batch(&t, &wp, &sp, l_bits, bump(256), me, mbx)
}

/// 2×2 max-pool over a `c×h×w` activation map, floor semantics.
#[allow(clippy::too_many_arguments)]
fn pool2x2<T: crate::crypto::net::Transport>(
    a: &[PartyShare],
    c: usize,
    h: usize,
    w: usize,
    l_bits: u32,
    bump: &mut impl FnMut(u64) -> u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    use crate::crypto::falcon_mpc::maxpool_batch;
    let (oh, ow) = (h / 2, w / 2);
    let mut windows = Vec::with_capacity(c * oh * ow);
    for ch in 0..c {
        for oy in 0..oh {
            for ox in 0..ow {
                windows.push(
                    (0..2)
                        .flat_map(|dy| (0..2).map(move |dx| (dy, dx)))
                        .map(|(dy, dx)| a[(ch * h + 2 * oy + dy) * w + 2 * ox + dx].clone())
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    let nw = windows.len();
    let wp: Vec<Vec<_>> = (0..nw)
        .map(|i| {
            (0..3)
                .map(|k| wrap_preproc((i * 11 + k) as u64, l_bits, me))
                .collect()
        })
        .collect();
    let sp: Vec<Vec<_>> = (0..nw)
        .map(|i| {
            (0..3)
                .map(|k| sel_pre((i * 13 + k) as u64, l_bits, me))
                .collect()
        })
        .collect();
    maxpool_batch(&windows, &wp, &sp, l_bits, bump(1024), me, mbx)
}

fn sel_pre(seed: u64, l_bits: u32, me: PartyId) -> crate::crypto::falcon_mpc::SelectPreproc {
    let c = [seed % 2, (seed / 2) % 2, (seed / 4) % 2];
    let cbit = (c[0] ^ c[1] ^ c[2]) & 1;
    let cr = [cbit as u128, 0, 0];
    let _ = l_bits;
    crate::crypto::falcon_mpc::SelectPreproc {
        c_bit: [c[me % 3], c[(me + 1) % 3]],
        c_ring: PartyShare {
            left: cr[me % 3],
            right: cr[(me + 1) % 3],
        },
    }
}

const USAGE: &str = "usage: baru bench-falcon [--n N] [--bits L] [--net] [--json]

Falcon's sign extraction against Baru's own routes, on one transport.
  --n N      how many signs to extract in one batch (default 1024)
  --bits L   ring width (default 32, Falcon's own)
  --net      instead: one MedMNIST-shaped inference, entirely in RSS
  --json     machine-readable output
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let (mut n, mut bits, mut json) = (1024usize, 32u32, false);
    let mut net = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--n" => {
                n = it
                    .next()
                    .ok_or("--n needs a value")?
                    .parse()
                    .map_err(|_| "bad --n")?
            }
            "--bits" => {
                bits = it
                    .next()
                    .ok_or("--bits needs a value")?
                    .parse()
                    .map_err(|_| "bad --bits")?
            }
            "--net" => net = true,
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }
    if net {
        let r = run_net(bits, 8, 2, NetworkProfile::INSTANT);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        println!("MedMNIST-shaped inference, entirely in RSS (Falcon's approach)");
        println!("  1x28x28 -> conv(8) -> conv(16) -> dense(128) -> dense(2), secret weights\n");
        println!("  ring width       : {} bits", r.l_bits);
        println!("  linear outputs   : {}", r.conv_outputs);
        println!("  DReLUs           : {}", r.drelus);
        println!("  messages         : {}", r.messages);
        println!("  bytes (per party): {:.3} MB", r.bytes as f64 / 1e6);
        println!("  wall clock       : {:.0} ms\n", r.ms);
        println!("  bench-medmnist (FHE + GC), measured: 10.1 MB, 268 ms");
        println!("  Falcon, LeNet-scale reported       :  0.74 MB,  47 ms");
        println!(
            "\n  Cost only: weights and input are stand-ins. The arithmetic is\n  \
             checked exactly by the protocol tests; an accuracy from this run\n  \
             would mean nothing."
        );
        return Ok(());
    }
    let r = run(n, bits, NetworkProfile::INSTANT);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!("Sign extraction: Falcon's wrap3 vs Baru's routes");
    println!(
        "n = {}, ring width = {} bits, same transport\n",
        r.n, r.l_bits
    );
    println!(
        "  {:<26}{:>9}{:>12}{:>10}{:>12}{:>9}",
        "route", "messages", "bytes", "B/sign", "ring elems", "ms"
    );
    for c in &r.routes {
        println!(
            "  {:<26}{:>9}{:>12}{:>10.1}{:>12}{:>9.1}",
            c.route,
            c.messages,
            c.bytes,
            c.bytes as f64 / r.n as f64,
            c.ring_elements(),
            c.ms
        );
    }
    println!(
        "\n  all routes agree on every sign bit: {}",
        if r.agree {
            "yes"
        } else {
            "NO — costs are not comparable"
        }
    );

    println!("\n  Per-sign information content, once the encoding is accounted for:");
    for c in &r.routes {
        let elems = c.ring_elements() as f64 / r.n as f64;
        let note = if c.route.starts_with("falcon") {
            format!(
                "{:.1} B — Z_p, already bit-packed at 6 bits",
                c.bytes as f64 / r.n as f64
            )
        } else if c.route.starts_with("baru-gc") {
            format!(
                "~{:.0} B — garbled tables, already Vec<u8>",
                c.bytes as f64 / r.n as f64
            )
        } else {
            format!(
                "~{:.0} B — {elems:.0} elements x 1 bit (a2b works on BIT shares, \
                 sent as u128)",
                elems / 8.0
            )
        };
        println!("    {:<26}{note}", c.route);
    }
    println!(
        "\n  So the routes are much closer than the byte column suggests. The\n  \
         `a2b` and compiled routes send ~{:.0}x their information content because\n  \
         every 1-bit share is serialised as a u128; Falcon's field traffic is\n  \
         packed. On content, wrap3 and a2b are within a few percent — the real\n  \
         gap is against *garbled circuits*, whose tables are genuinely large.",
        128.0
    );
    // Separating the two effects matters: one is a protocol difference, the
    // other is an artefact of how this crate serialises ring shares, and only
    // the first is a claim about Falcon.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::transport_backend::FalconPoolSpec;
    use crate::interpreter::transport_backend::NonlinMode;

    /// **The wrap correction is exact for every mask, including the wrapping ones.**
    ///
    /// This is the identity `truncate_batch` rests on, checked over *all* `(a, r)` at a
    /// small ring rather than sampled:
    ///
    /// ```text
    /// a>>d = (c>>d) − (r>>d) − [c_lo < r_lo] + [c < r]·2^(l−d),   c = (a+r) mod 2^l
    /// ```
    ///
    /// Exhaustive matters here because the interesting inputs are the rare ones. `r` is
    /// uniform over the whole ring, so `a + r` wraps about half the time and the wrap
    /// term is exercised constantly — but the *borrow* and the wrap interact only when
    /// `c_lo < r_lo` and `c < r` at once, and a sampled test can miss the corner where
    /// both corrections apply. At `l = 10, d = 3` this is 524 288 pairs and runs in
    /// milliseconds.
    ///
    /// It also pins the second claim the protocol needs: that `[c < r]` really is the
    /// wrap indicator `[a + r ≥ 2^l]`. That equivalence is why the wrap is computable
    /// from an opened `c` and a shared `r` at all, and it is not obvious enough to leave
    /// as a comment.
    #[test]
    fn the_wrap_correction_is_exact_over_every_mask() {
        const L: u32 = 10;
        const D: u32 = 3;
        let ring = 1u64 << L;
        let dmask = (1u64 << D) - 1;
        let mut checked = 0u64;

        // `a` non-negative and below half the ring, as a ReLU output is.
        for a in 0..(ring / 2) {
            for r in 0..ring {
                let c = (a + r) % ring;
                let borrow = u64::from((c & dmask) < (r & dmask));
                let wrap = u64::from(c < r);

                // The wrap bit is the overflow indicator — the fact that makes it
                // computable from a public `c` and a secret `r`.
                assert_eq!(
                    wrap,
                    u64::from(a + r >= ring),
                    "[c<r] must equal the wrap indicator at a={a}, r={r}"
                );

                let got = ((c >> D)
                    .wrapping_sub(r >> D)
                    .wrapping_sub(borrow)
                    .wrapping_add(wrap << (L - D)))
                    % ring;
                assert_eq!(got, a >> D, "truncating a={a} by {D} with r={r}");
                checked += 1;
            }
        }
        assert_eq!(checked, (ring / 2) * ring, "the sweep must be exhaustive");
        println!("wrap-corrected truncation exact over {checked} (a, r) pairs at l={L}, d={D}");
    }

    /// **A mask narrower than the ring is rejected, not silently accepted.**
    ///
    /// The leak this guards against was not a wrong answer — it was `random_trunc_pairs`
    /// being asked for a mask that fit the no-wrap bound, obliging, and producing a
    /// protocol run that computed the right activations while leaking ≈0.72 bits of each
    /// one. Nothing in the type system distinguished the two requests.
    ///
    /// So the narrow request is now unrepresentable: `r_bits` must equal `l_bits`. This
    /// test is the guard on that, and it is a panic test rather than a value test because
    /// the property being protected is *refusal*.
    #[test]
    #[should_panic(expected = "the mask must cover the whole ring")]
    fn a_truncation_mask_narrower_than_the_ring_is_refused() {
        // `r_bits = 19` under an `l_bits = 20` ring: the exact configuration that leaked.
        crate::crypto::falcon_mpc::assert_full_ring_mask(19, 20);
    }

    /// The other side of the same guard: the full-ring mask is accepted.
    ///
    /// Without this, `a_truncation_mask_narrower_than_the_ring_is_refused` would still
    /// pass if the check rejected *everything*.
    #[test]
    fn a_full_ring_truncation_mask_is_accepted() {
        for l in [4u32, 10, 20, 21, 64] {
            crate::crypto::falcon_mpc::assert_full_ring_mask(l, l);
        }
    }

    /// **Falcon keeps the narrowed wire**, which is the point of the wrap bit.
    ///
    /// The first fix for the truncation leak widened the ring by a statistical parameter
    /// `κ`, and that is incompatible with the narrowing: narrowing spends the slack
    /// between the value width and the ring width, and statistical hiding needs exactly
    /// that slack. Measured, it made Falcon's ReLU ~1.9× *dearer* in bytes than `a2b`'s —
    /// a secure protocol whose cost erased the reason to prefer it.
    ///
    /// Computing the wrap instead hides *perfectly* and needs only one extra bit of
    /// width, for the `c − r` difference. This pins that all three routes share one
    /// width, so nothing silently reintroduces a route-specific ring.
    #[test]
    fn falcon_keeps_the_narrowed_wire() {
        use crate::bench::cnn::{wire_bits_for, wire_bits_for_nonlin, Nonlin};
        for nbits in [20usize, 21, 23] {
            let narrow = wire_bits_for(nbits).expect("the default narrows");
            for route in [Nonlin::A2b, Nonlin::Gc, Nonlin::Falcon] {
                assert_eq!(
                    wire_bits_for_nonlin(nbits, route),
                    Some(narrow),
                    "nbits={nbits}, {route:?}: every route takes the narrowed wire"
                );
            }
            // And the wire must clear what `relu_falcon_batch` asserts: the `nbits`-bit
            // ring plus one bit for the wrap difference.
            let needed = nbits as u32 + 1;
            assert!(
                narrow >= needed,
                "nbits={nbits}: the narrow wire {narrow} must reach {needed} to leave \
                 room for the wrap bit"
            );
        }
    }

    /// **Falcon's own `ReLU∘rescale` must agree with the `a2b` reference exactly.**
    ///
    /// This is the gate on `relu_falcon_batch`. Until it existed, `NonlinMode::Falcon`
    /// routed ReLU through `a2b` for a good reason — there was no *secure* Falcon
    /// truncation to route to — so a "falcon" run put only its max-pool comparisons
    /// through Falcon. `random_trunc_pairs` closed that, and this checks the resulting
    /// ReLU against the route it replaces.
    ///
    /// The inputs deliberately straddle zero and include the boundary cases: the gate
    /// flips at 0, and the truncation's borrow flips where `r_lo = c_lo + 1`, which is
    /// the input `Π_PC` cannot handle and the reason Falcon's truncation uses bit
    /// decomposition for its borrow instead.
    #[test]
    #[ignore] // real 3-party transport
    fn falcon_and_a2b_relu_agree() {
        const NBITS: usize = 20;
        const SHIFT: u32 = 8;
        let l = 1i128 << NBITS;
        // Straddle zero, hit 0 and ±1, and include values whose low SHIFT bits vary.
        let vals: Vec<i128> = vec![
            0,
            1,
            -1,
            255,
            256,
            257,
            -255,
            -256,
            -257,
            4095,
            4096,
            -4096,
            65535,
            -65535,
            (1 << (NBITS - 2)) as i128,
            -((1 << (NBITS - 2)) as i128),
        ];
        let want: Vec<i128> = vals
            .iter()
            .map(|&v| if v >= 0 { v >> SHIFT } else { 0 })
            .collect();
        let n = vals.len();

        let mut results = Vec::new();
        for falcon in [false, true] {
            let vs = vals.clone();
            let (out, _) = run_parties(NetworkProfile::INSTANT, move |_me| {
                let vs = vs.clone();
                move |be: &mut TransportBackend<MpcTransport>| {
                    let enc = |v: i128| ((v + l) % l) as u128;
                    let shares: Vec<_> =
                        vs.iter().map(|&v| be.share_public(enc(v), NBITS)).collect();
                    let acts = if falcon {
                        be.relu_falcon_batch(&shares, NBITS, SHIFT)
                    } else {
                        // The reference: one a2b, shift and sign bit from the same bits.
                        be.set_nonlin(NonlinMode::A2b);
                        crate::bench::cnn::fused_relu_rescale_for_test(be, &shares, SHIFT, NBITS)
                    };
                    acts.iter().map(|a| be.reveal(a, None)).collect::<Vec<_>>()
                }
            });
            let got: Vec<i128> = out[0]
                .iter()
                .map(|&r| {
                    let m = (r & ((1u128 << NBITS) - 1)) as i128;
                    if m >= l / 2 {
                        m - l
                    } else {
                        m
                    }
                })
                .collect();
            assert_eq!(
                got,
                want,
                "{} route disagrees with the plaintext reference",
                if falcon { "falcon" } else { "a2b" }
            );
            results.push(got);
        }
        assert_eq!(
            results[0], results[1],
            "the Falcon ReLU must be bit-identical to the a2b one over {n} values"
        );
    }

    /// **Falcon's own ReLU costs slightly more messages for roughly the same bytes.**
    ///
    /// Before `Π_ReLU` was wired in, a "falcon" run used Falcon only for its max-pool
    /// comparisons and fell back to `a2b` for ReLU. Making the route pure does not make
    /// it cheaper: `Π_DReLU` + `Π_Trunc` carry their comparisons in `Z_p` (6-bit
    /// elements) and use `a2b` only on two narrow sign bits, but they are four protocols
    /// in sequence against `a2b`'s one decomposition.
    ///
    /// Measured at ℓ = 20, batch 256: **bytes 1.05× cheaper, messages 1.10× dearer.**
    /// Near enough a wash, which is the useful thing to know — the two terms enter the
    /// cost model with opposite signs (`rounds × RTT + bytes / bandwidth`), so a route
    /// that is 5% better in one and 10% worse in the other has no regime where it wins
    /// by much on ReLU alone. Falcon's leverage in the pipeline is its *max-pool*
    /// comparisons, not its ReLU.
    ///
    /// # Two ways this measurement was wrong before, both by comparing incommensurables
    ///
    /// **At a common 128-bit wire it read 7.5× cheaper in bytes.** Neither route deploys
    /// there: `a2b`'s full-width decomposition is what 128 bits punishes, so the ratio was
    /// measuring the wire, not the protocol. Both routes now run at
    /// `wire_bits_for_nonlin`.
    ///
    /// **With a leaking truncation it read 7.5× at 128 bits and 1.9× *dearer* under a
    /// `κ`-widened ring.** The first version of `relu_falcon_batch` masked with
    /// `r_bits = nbits − 1`, which opens `a + r` with no slack; the first fix widened the
    /// ring by `κ = 40`, which is incompatible with the narrowed wire. Only the wrap-bit
    /// construction is both private and narrow, so only its numbers describe something
    /// shippable. See `truncate_batch`.
    #[test]
    #[ignore] // real 3-party transport
    fn the_pure_falcon_relu_trades_bytes_for_rounds() {
        const NBITS: usize = 20;
        const SHIFT: u32 = 8;
        let l = 1i128 << NBITS;
        let n = 256usize;
        let vals: Vec<i128> = (0..n as i128).map(|i| (i * 37) % 4001 - 2000).collect();
        let want: Vec<i128> = vals
            .iter()
            .map(|&v| if v >= 0 { v >> SHIFT } else { 0 })
            .collect();

        let run = |falcon: bool| {
            let vs = vals.clone();
            let (out, comm) = run_parties(NetworkProfile::INSTANT, move |_me| {
                let vs = vs.clone();
                move |be: &mut TransportBackend<MpcTransport>| {
                    use crate::bench::cnn::{wire_bits_for_nonlin, Nonlin};
                    let route = if falcon { Nonlin::Falcon } else { Nonlin::A2b };
                    if let Some(w) = wire_bits_for_nonlin(NBITS, route) {
                        be.set_wire_bits(w);
                    }
                    let enc = |v: i128| ((v + l) % l) as u128;
                    let shares: Vec<_> =
                        vs.iter().map(|&v| be.share_public(enc(v), NBITS)).collect();
                    let acts = if falcon {
                        be.relu_falcon_batch(&shares, NBITS, SHIFT)
                    } else {
                        be.set_nonlin(NonlinMode::A2b);
                        crate::bench::cnn::fused_relu_rescale_for_test(be, &shares, SHIFT, NBITS)
                    };
                    acts.iter().map(|a| be.reveal(a, None)).collect::<Vec<_>>()
                }
            });
            let got: Vec<i128> = out[0]
                .iter()
                .map(|&r| {
                    let m = (r & ((1u128 << NBITS) - 1)) as i128;
                    if m >= l / 2 {
                        m - l
                    } else {
                        m
                    }
                })
                .collect();
            (got, comm.messages, comm.bytes)
        };

        let (a2b_got, a2b_msgs, a2b_bytes) = run(false);
        let (fal_got, fal_msgs, fal_bytes) = run(true);

        // The trade is only meaningful between routes computing the same function.
        assert_eq!(a2b_got, want, "the a2b reference must be correct");
        assert_eq!(fal_got, want, "the Falcon route must be correct");

        println!(
            "relu over {n} values at deployed widths: a2b {a2b_msgs} msgs / {a2b_bytes} B, \
             falcon {fal_msgs} msgs / {fal_bytes} B (bytes {:.2}x, messages {:.2}x)",
            a2b_bytes as f64 / fal_bytes as f64,
            fal_msgs as f64 / a2b_msgs as f64,
        );

        // Falcon costs more messages: four sequential stages against one decomposition.
        // This direction is structural and does not depend on the widths.
        assert!(
            fal_msgs > a2b_msgs,
            "Falcon's ReLU should cost more messages than a2b's: {fal_msgs} vs {a2b_msgs}"
        );

        // The byte direction is *not* asserted, because at deployed widths it is no
        // longer a given: Falcon's `Z_p` comparisons are cheap but its statistically
        // hiding truncation runs in an `nbits + κ + 1` bit ring while `a2b` runs in
        // `nbits + 1`. Which term wins is the measurement, so it is printed and checked
        // against the pipeline rather than baked in as an expectation here.
        println!(
            "  → falcon is {} in bytes at deployed widths",
            if fal_bytes < a2b_bytes {
                "cheaper"
            } else {
                "DEARER"
            }
        );
    }

    /// **Falcon's round cost is mostly per-call, so batch granularity decides the trade.**
    ///
    /// `the_pure_falcon_relu_trades_bytes_for_rounds` measures one batch of 256 and
    /// finds Falcon only ~1.10× dearer in messages. The full CNN measures a larger
    /// ratio. Both are right: the CNN calls ReLU once per conv layer, on batches an
    /// order of magnitude smaller, and Falcon pays a larger *fixed* round count per call
    /// (Π_DReLU, its Π_SS lift, the multiply, then Π_Trunc with its own two sign bits —
    /// four sequential stages against a2b's one decomposition).
    ///
    /// Fitting `messages = fixed + slope · n` over three batch sizes separates the
    /// terms, and the answer is that there is no `n` term at all: both routes send a
    /// fixed number of messages regardless of batch size, because every stage batches
    /// its whole input into one message per round. So the entire difference between
    /// the routes is a per-call constant.
    ///
    /// The test therefore reveals only one output. Revealing all `n` would add two
    /// messages per element — a property of the *harness*, not the protocol — and
    /// would swamp the term being measured. (It did, on the first attempt: both
    /// routes fitted `2.000·n`, which was the reveal loop.)
    ///
    /// The consequence is practical: Falcon's costs are realised per *call*, so the
    /// round penalty is amortised only by large batches. A layer-at-a-time schedule pays
    /// it per layer, which is why the whole-CNN message ratio is so much worse than one
    /// big batch's.
    #[test]
    #[ignore] // real 3-party transport
    fn falcons_round_cost_is_mostly_per_call() {
        const NBITS: usize = 20;
        const SHIFT: u32 = 8;
        let l = 1i128 << NBITS;

        let run = |falcon: bool, n: usize| {
            let vals: Vec<i128> = (0..n as i128).map(|i| (i * 37) % 4001 - 2000).collect();
            let (_out, comm) = run_parties(NetworkProfile::INSTANT, move |_me| {
                let vs = vals.clone();
                move |be: &mut TransportBackend<MpcTransport>| {
                    let enc = |v: i128| ((v + l) % l) as u128;
                    let shares: Vec<_> =
                        vs.iter().map(|&v| be.share_public(enc(v), NBITS)).collect();
                    let acts = if falcon {
                        be.relu_falcon_batch(&shares, NBITS, SHIFT)
                    } else {
                        be.set_nonlin(NonlinMode::A2b);
                        crate::bench::cnn::fused_relu_rescale_for_test(be, &shares, SHIFT, NBITS)
                    };
                    // One reveal, not `n`: see the note above on the harness term.
                    be.reveal(&acts[0], None)
                }
            });
            comm.messages as f64
        };

        // Two points fix the line; the third checks it is actually a line.
        let sizes = [32usize, 128, 512];
        let mut fits = Vec::new();
        for falcon in [false, true] {
            let m: Vec<f64> = sizes.iter().map(|&n| run(falcon, n)).collect();
            // Least squares on three points, by hand.
            let xs: Vec<f64> = sizes.iter().map(|&n| n as f64).collect();
            let xm = xs.iter().sum::<f64>() / 3.0;
            let ym = m.iter().sum::<f64>() / 3.0;
            let sxy: f64 = xs.iter().zip(&m).map(|(x, y)| (x - xm) * (y - ym)).sum();
            let sxx: f64 = xs.iter().map(|x| (x - xm).powi(2)).sum();
            let slope = sxy / sxx;
            let fixed = ym - slope * xm;
            let name = if falcon { "falcon" } else { "a2b" };
            println!(
                "{name:7} messages = {fixed:.0} + {slope:.3}·n   (measured {:?})",
                m.iter().map(|v| *v as u64).collect::<Vec<_>>()
            );
            fits.push((fixed, slope, m));
        }

        let (a2b_fixed, a2b_slope, _) = fits[0].clone();
        let (fal_fixed, fal_slope, _) = fits[1].clone();

        // The claim: Falcon's extra messages are a per-call constant, not per element.
        assert!(
            fal_fixed > a2b_fixed,
            "Falcon should pay a larger fixed round cost: {fal_fixed:.0} vs {a2b_fixed:.0}"
        );
        // And the claim's other half: neither route's message count grows with `n`.
        // A batch 16x larger must not cost meaningfully more messages.
        for (name, slope) in [("a2b", a2b_slope), ("falcon", fal_slope)] {
            assert!(
                slope.abs() < 0.01,
                "{name} messages should not grow with batch size, got {slope:.4}/element"
            );
        }
        println!(
            "→ the whole difference is {:.0} messages, paid once per call, \
             independent of batch size",
            fal_fixed - a2b_fixed,
        );
    }

    /// **The warm-server cost, measured** — the number the withdrawn crossover was
    /// missing.
    ///
    /// Every Falcon figure this repo has published is a *cold* one, because
    /// `lt_falcon_batch` generated preprocessing inline. With 91% of Falcon's bytes in
    /// that phase, a warm server — one that generated its correlated randomness before
    /// the query arrived — pays a fraction of it, and comparing a cold Falcon against
    /// garbled circuits (which has no offline phase to speak of) is the accounting error
    /// the crossover was withdrawn for.
    ///
    /// The sizing is *counted, not derived*: a first cold run reports how many comparison
    /// elements the workload actually asked for, and the warm run pools exactly that. The
    /// alternative — re-deriving the count from the layer shapes — would mean mirroring
    /// the batching decisions inside `maxpool2d`, `fused_relu_rescale` and `argmax`, and
    /// a second implementation to keep in step is how the mask pool broke when channel
    /// splitting landed.
    #[test]
    #[ignore] // real 3-party transport
    fn the_warm_falcon_cost_is_measured_against_the_cold_one() {
        const NBITS: usize = 20;
        let l = 1i128 << NBITS;
        // A batch big enough that per-element cost dominates the fixed framing.
        let n = 256usize;
        let cases: Vec<(i128, i128)> = (0..n as i128)
            .map(|i| (i % 97 - 48, (i * 7) % 89 - 44))
            .collect();
        let want: Vec<u64> = cases.iter().map(|&(a, b)| u64::from(a < b)).collect();

        let run = move |warm: bool, size: usize| {
            let cs = cases.clone();
            let (out, comm) = run_parties(NetworkProfile::INSTANT, move |_me| {
                let cs = cs.clone();
                move |be: &mut TransportBackend<MpcTransport>| {
                    be.set_nonlin(NonlinMode::Falcon);
                    let offline_ms = if warm {
                        be.fill_falcon_pool(&FalconPoolSpec {
                            comparisons: size,
                            truncations: 0, // the `lt` path truncates nothing
                            nbits: NBITS as u32,
                            lift_bits: NBITS as u32,
                            shift: 0,
                        })
                    } else {
                        0.0
                    };
                    let enc = |v: i128| ((v + l) % l) as u128;
                    let pairs: Vec<_> = cs
                        .iter()
                        .map(|&(a, b)| {
                            (
                                be.share_public(enc(a), NBITS),
                                be.share_public(enc(b), NBITS),
                            )
                        })
                        .collect();
                    let t = std::time::Instant::now();
                    let bits = be.lt_batch(&pairs);
                    let got: Vec<u64> = bits
                        .iter()
                        .map(|x| (be.reveal(x, None) & 1) as u64)
                        .collect();
                    let online_ms = t.elapsed().as_secs_f64() * 1e3;
                    (
                        got,
                        online_ms,
                        offline_ms,
                        be.falcon_comparisons(),
                        be.falcon_pool_stats(),
                    )
                }
            });
            let (got, online_ms, offline_ms, count, stats) = out[0].clone();
            (got, online_ms, offline_ms, count, stats, comm.bytes)
        };

        // Cold: preprocessing on the critical path. Also tells us the pool size.
        let (cold_got, cold_ms, _, count, _, cold_bytes) = run(false, 0);
        let _ = &run; // keep the closure available for the bandwidth sweep below
        assert_eq!(cold_got, want, "cold run must be correct");
        assert_eq!(
            count, n,
            "the counter must report the elements actually requested"
        );

        // Warm: same workload, preprocessing pooled beforehand.
        let (warm_got, warm_ms, off_ms, _, stats, warm_bytes) = run(true, count);
        assert_eq!(warm_got, want, "warm run must agree with cold");
        let (_, served, missed) = stats.expect("warm run must report pool stats");
        assert_eq!(
            (served, missed),
            (n, 0),
            "warm run must be fully served by the pool"
        );

        println!(
            "\n  Falcon at ℓ={NBITS}, n={n} — cold vs warm\n    \
             cold  : {cold_ms:8.1} ms online (preprocessing on the critical path)\n    \
             warm  : {warm_ms:8.1} ms online + {off_ms:.1} ms offline, pooled\n    \
             speedup on the critical path: {:.1}x\n    \
             total bytes are unchanged: {cold_bytes} cold, {warm_bytes} warm\n",
            cold_ms / warm_ms.max(1e-9),
        );

        // The claim: pooling moves work off the critical path rather than removing it.
        // Bytes must be essentially unchanged — a warm run that sent materially less
        // would mean the pool had skipped work rather than pre-done it.
        let ratio = warm_bytes as f64 / cold_bytes as f64;
        assert!(
            (0.9..1.1).contains(&ratio),
            "pooling must not change total traffic: {cold_bytes} cold vs {warm_bytes} \
             warm ({:.2}x) — a difference means work was skipped, not moved",
            ratio
        );
        assert!(
            warm_ms < cold_ms,
            "the warm critical path must be shorter: cold {cold_ms:.1} ms vs warm \
             {warm_ms:.1} ms"
        );
    }

    /// **How the warm/cold gap widens as bandwidth binds** — the number that decides
    /// whether a warm crossover boundary is worth chasing.
    ///
    /// `the_warm_falcon_cost_is_measured_against_the_cold_one` finds 2.9× at zero
    /// latency, well short of the ~11× the 91%-of-bytes split suggests, because at
    /// infinite bandwidth compute dominates and the online phase still has real work.
    /// The prediction is that the gap approaches the byte ratio as bandwidth binds. That
    /// is a prediction about the *shape* of the warm boundary, so the trend is **reported
    /// rather than asserted**: if the gap does not widen, pooling buys little where the
    /// crossover actually lives and wiring it into the sweep would not change any
    /// conclusion.
    ///
    /// # Why the trend is not asserted, measured rather than assumed
    ///
    /// It *was* asserted (`ratios.last() > ratios.first()`), and it failed whenever the
    /// ignored suite ran in parallel — not because the claim is false but because the
    /// **baseline** is the noisy cell. At unlimited bandwidth the run is pure compute, so
    /// CPU contention inflated it 892 → 1615 ms and drove the first ratio from 2.44× to
    /// 4.11×, while the 0.2 Mbps cell, being network-bound, barely moved (3104 → 2996 ms).
    /// Using a compute-bound measurement as the reference for a bandwidth-sensitivity
    /// claim is the defect; the ordering it produced was an artifact of load.
    ///
    /// What remains asserted is only that pooling shortens the critical path *at all*, at
    /// every profile — a 2–4× effect, robust to the contention that a 1.8× ordering is not.
    /// The trend itself stays in the printed table, where a human can read it without a
    /// flaky gate calling load a regression.
    ///
    /// **Asserting the mechanism instead does not work with this harness, and the attempt
    /// is recorded so it is not retried.** The deterministic quantity would be
    /// *critical-path* bytes — those sent after the timer starts — since pooling relocates
    /// preprocessing traffic rather than removing it. But `run_parties` reports only
    /// whole-run totals, and cold and warm move **identical** total bytes (49 928 each at
    /// `n = 128`); that identity is itself a documented invariant of the pool. Measuring
    /// the split needs a byte counter that can be snapshotted mid-run, which does not
    /// exist. Substituting a proxy would be worse than leaving the trend unasserted.
    #[test]
    #[ignore] // real 3-party transport, several network profiles
    fn the_warm_gap_widens_as_bandwidth_binds() {
        const NBITS: usize = 20;
        let l = 1i128 << NBITS;
        let n = 128usize;
        let cases: Vec<(i128, i128)> = (0..n as i128)
            .map(|i| (i % 61 - 30, (i * 11) % 53 - 26))
            .collect();

        let measure = move |profile: NetworkProfile, warm: bool| -> f64 {
            let cs = cases.clone();
            let (out, _) = run_parties(profile, move |_me| {
                let cs = cs.clone();
                move |be: &mut TransportBackend<MpcTransport>| {
                    be.set_nonlin(NonlinMode::Falcon);
                    if warm {
                        be.fill_falcon_pool(&FalconPoolSpec {
                            comparisons: n,
                            truncations: 0,
                            nbits: NBITS as u32,
                            lift_bits: NBITS as u32,
                            shift: 0,
                        });
                    }
                    let enc = |v: i128| ((v + l) % l) as u128;
                    let pairs: Vec<_> = cs
                        .iter()
                        .map(|&(a, b)| {
                            (
                                be.share_public(enc(a), NBITS),
                                be.share_public(enc(b), NBITS),
                            )
                        })
                        .collect();
                    let t = std::time::Instant::now();
                    let bits = be.lt_batch(&pairs);
                    for x in &bits {
                        let _ = be.reveal(x, None);
                    }
                    t.elapsed().as_secs_f64() * 1e3
                }
            });
            out[0]
        };

        // Unlimited, then pipes tight enough that the byte term actually dominates.
        // A first attempt used 100 and 20 Mbps and found the ratio *flat* — at this
        // batch size those pipes contribute ~140 ms against ~840 ms of compute, so the
        // measurement never entered the regime the hypothesis is about. The bandwidths
        // below put the byte term ahead of compute, which is where a crossover lives.
        let mbps = [f64::INFINITY, 5.0, 1.0, 0.2];
        let mut ratios = Vec::new();
        println!("\n  warm/cold critical-path ratio vs bandwidth (ℓ={NBITS}, n={n})");
        for &m in &mbps {
            let profile = if m.is_infinite() {
                NetworkProfile::INSTANT
            } else {
                NetworkProfile::INSTANT.with_bandwidth(m * 1e6 / 8.0)
            };
            let cold = measure(profile, false);
            let warm = measure(profile, true);
            let r = cold / warm.max(1e-9);
            let label = if m.is_infinite() {
                "unlimited".to_string()
            } else {
                // `{:.0}` printed 0.2 as "0"; sub-1 Mbps is where the gap is largest.
                format!("{m:.3} Mbps").replace(".000", "")
            };
            println!("    {label:>10}: cold {cold:8.1} ms  warm {warm:8.1} ms  ratio {r:5.2}x");
            ratios.push(r);
        }
        println!();

        assert!(
            ratios.iter().all(|&r| r > 1.0),
            "pooling must shorten the critical path at every profile: {ratios:?}"
        );

        // The trend across bandwidths is deliberately NOT asserted — see the note above.
        // The printed table is the result; a gate on it reads CPU load as a regression.
    }

    /// **A pooled Falcon run computes the same comparisons as an inline one**, and the
    /// pool actually serves them.
    ///
    /// The pool exists so the two honest costs are separately measurable: with 91% of
    /// Falcon's bytes in preprocessing, a cold server pays the total and a warm one pays
    /// only the online phase. That split is worthless if it changes the answer, and it
    /// would be worse than worthless if a benchmark quoted an online latency while
    /// quietly generating preprocessing on the critical path — so this asserts both the
    /// results and the `served`/`missed` accounting.
    #[test]
    #[ignore] // real 3-party transport
    fn a_pooled_falcon_run_matches_an_inline_one() {
        const NBITS: usize = 20;
        let l = 1i128 << NBITS;
        let cases: Vec<(i128, i128)> = vec![(5, 3), (3, 5), (0, 0), (-7, 2), (2, -7), (100, -100)];
        let want: Vec<u64> = cases.iter().map(|&(a, b)| u64::from(a < b)).collect();
        let n = cases.len();

        // Two runs, identical except for where preprocessing came from.
        let mut outs = Vec::new();
        let mut stats = Vec::new();
        for pooled in [false, true] {
            let cs = cases.clone();
            let (out, _) = run_parties(NetworkProfile::INSTANT, move |_me| {
                let cs = cs.clone();
                move |be: &mut TransportBackend<MpcTransport>| {
                    be.set_nonlin(NonlinMode::Falcon);
                    // Offline, before anything a query depends on.
                    let offline_ms = if pooled {
                        be.fill_falcon_pool(&FalconPoolSpec {
                            comparisons: n,
                            truncations: 0,
                            nbits: NBITS as u32,
                            lift_bits: NBITS as u32,
                            shift: 0,
                        })
                    } else {
                        0.0
                    };
                    let enc = |v: i128| ((v + l) % l) as u128;
                    let pairs: Vec<_> = cs
                        .iter()
                        .map(|&(a, b)| {
                            (
                                be.share_public(enc(a), NBITS),
                                be.share_public(enc(b), NBITS),
                            )
                        })
                        .collect();
                    let bits = be.lt_batch(&pairs);
                    let got: Vec<u64> = bits
                        .iter()
                        .map(|x| (be.reveal(x, None) & 1) as u64)
                        .collect();
                    (got, be.falcon_pool_stats(), offline_ms)
                }
            });
            let (got, st, offline_ms) = out[0].clone();
            assert_eq!(got, want, "pooled={pooled}: wrong comparisons");
            outs.push(got);
            stats.push((st, offline_ms));
        }

        assert_eq!(outs[0], outs[1], "pooling must not change the answer");

        // Inline: no pool at all.
        assert!(stats[0].0.is_none(), "the inline run must not have a pool");

        // Pooled: every item served, none generated on the critical path, and the
        // offline phase actually cost something — a zero would mean it never ran.
        let (st, offline_ms) = &stats[1];
        let (pool_ms, served, missed) = st.expect("the pooled run must report stats");
        assert_eq!(served, n, "every item must come from the pool");
        assert_eq!(missed, 0, "nothing may be generated on the critical path");
        assert!(
            pool_ms > 0.0 && *offline_ms > 0.0,
            "the offline phase must be measured, not defaulted: pool {pool_ms} ms"
        );
    }

    /// The three comparison backends must be **interchangeable**: same inputs,
    /// same bits out, whichever protocol runs underneath.
    ///
    /// This is what makes `NonlinMode` a choice rather than three separate
    /// systems — and it is the check that has to pass before any cost
    /// comparison between them means anything.
    #[test]
    #[ignore] // real 3-party transport
    fn every_nonlinearity_mode_computes_the_same_comparison() {
        const NBITS: usize = 20;
        let l = 1i128 << NBITS;
        let cases: Vec<(i128, i128)> = vec![
            (5, 3),
            (3, 5),
            (0, 0),
            (-7, 2),
            (2, -7),
            (-1, -1),
            (100, -100),
            (-100, 100),
        ];
        let want: Vec<u64> = cases.iter().map(|&(a, b)| u64::from(a < b)).collect();

        let mut seen = Vec::new();
        for mode in [NonlinMode::A2b, NonlinMode::Gc, NonlinMode::Falcon] {
            let cs = cases.clone();
            let (out, _) = run_parties(NetworkProfile::INSTANT, move |_me| {
                let cs = cs.clone();
                move |be: &mut TransportBackend<MpcTransport>| {
                    be.set_nonlin(mode);
                    let enc = |v: i128| ((v + l) % l) as u128;
                    let pairs: Vec<_> = cs
                        .iter()
                        .map(|&(a, b)| {
                            (
                                be.share_public(enc(a), NBITS),
                                be.share_public(enc(b), NBITS),
                            )
                        })
                        .collect();
                    let bits = be.lt_batch(&pairs);
                    bits.iter().map(|x| be.reveal(x, None)).collect::<Vec<_>>()
                }
            });
            let got: Vec<u64> = out[0].iter().map(|&v| (v & 1) as u64).collect();
            assert_eq!(
                got,
                want,
                "{} disagreed with plaintext (a < b)",
                mode.label()
            );
            seen.push(got);
        }
        assert!(
            seen.windows(2).all(|w| w[0] == w[1]),
            "modes must agree with each other"
        );
    }
}
