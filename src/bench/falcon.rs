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
    use crate::interpreter::transport_backend::NonlinMode;

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
