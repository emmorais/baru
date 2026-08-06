//! `baru bench-conv` — **which backend should evaluate a convolution?**
//!
//! Baru's neural-network pipeline convolves homomorphically and does its
//! nonlinearities under secret sharing. That split was inherited rather than
//! measured for convolution specifically: the FHE side was chosen because
//! coefficient packing turns a whole layer into `out_c` multiplications, and the
//! secret-sharing side was never given a fair hearing, because the obvious
//! secret-shared convolution spends `k^2` interactive multiplications per output
//! pixel and loses without a contest.
//!
//! [`crate::gadgets::winograd`] changes the terms — `36 -> 16` multiplications
//! per `2x2` tile — so the contest is now worth holding. This benchmark holds it.
//!
//! # What makes the three routes comparable
//!
//! All of them **start from an encrypted or shared input and end with secret
//! shares of the convolution output**, which is what the next layer needs: the
//! ReLU that follows runs under secret sharing whichever way the convolution was
//! computed. Anything less than that boundary would flatter the homomorphic
//! route, which produces a *ciphertext* and still owes a scheme switch before
//! the network can continue.
//!
//! All of them keep the model secret. The homomorphic route multiplies
//! ciphertext by ciphertext against an encrypted model, exactly as
//! [`crate::bench::cnn`] does; the shared routes hold the weights as shares. A
//! comparison against plaintext weights would be measuring a different privacy
//! guarantee.
//!
//! All of them are checked against the same plaintext integer reference, and the
//! command fails rather than reports if any disagrees.
//!
//! # What is reported and why
//!
//! Bytes, blocking waits and wall clock, the three terms this artifact has found
//! to matter, plus the multiplication count where it is defined. The routes are
//! *not* summarised into a single winner, because which one wins is a property of
//! the deployment: the homomorphic route moves large ciphertexts and blocks
//! rarely, the shared routes move small elements and block once, and those trade
//! against each other exactly as the nonlinearity backends do.

use std::time::Instant;

use serde::Serialize;

use crate::bench::comm::{CommStats, NetworkProfile};
use crate::bench::hybrid_nn::{run_parties, MpcTransport};
use crate::crypto::types::Fr;
use crate::fhe::bfv::{Bfv, BfvParams};
use crate::fhe::conv_pack::{self, ConvShape};
use crate::gadgets::winograd::{conv2d_winograd, winograd_cost, WINOGRAD_OUTPUT_SCALE_LOG2};
use crate::gadgets::{conv2d, conv2d_ref, Chw};
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::TransportBackend;

fn residue(v: i64, nbits: usize) -> Fr {
    (v as i128).rem_euclid(1i128 << nbits) as Fr
}

fn to_signed(v: u128, nbits: usize) -> i64 {
    let m = 1i128 << nbits;
    let x = (v as i128) & (m - 1);
    if x >= m / 2 {
        (x - m) as i64
    } else {
        x as i64
    }
}

/// One measured strategy.
#[derive(Debug, Clone, Serialize)]
pub struct RoutePoint {
    pub route: String,
    /// Interactive multiplications, where the notion applies.
    pub muls: Option<usize>,
    pub messages: u64,
    pub bytes: u64,
    /// Times a party blocked for a peer.
    pub waits: u64,
    pub ms_by_rtt: Vec<(u64, f64)>,
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub in_c: usize,
    pub h: usize,
    pub w: usize,
    pub out_c: usize,
    pub k: usize,
    pub nbits: usize,
    pub routes: Vec<RoutePoint>,
}

/// The layer, its weights and the answer every route must produce.
struct Layer {
    img: Vec<i64>,
    in_shape: Chw,
    weights: Vec<Vec<i64>>,
    bias: Vec<i64>,
    want: Vec<i64>,
}

impl Layer {
    fn new(in_c: usize, h: usize, w: usize, out_c: usize, k: usize) -> Self {
        // Small values: the shared routes carry a 4x scale and the homomorphic
        // one a plaintext modulus, and an overflow in either would be measured as
        // a wrong answer rather than as the arithmetic bug it is.
        let img: Vec<i64> = (0..in_c * h * w).map(|i| ((i * 7) % 9) as i64 - 4).collect();
        let weights: Vec<Vec<i64>> = (0..out_c)
            .map(|o| {
                (0..in_c * k * k)
                    .map(|j| (((o * 5 + j * 3) % 7) as i64) - 3)
                    .collect()
            })
            .collect();
        let bias: Vec<i64> = (0..out_c).map(|o| (o as i64 % 3) - 1).collect();
        let in_shape = Chw { c: in_c, h, w };
        let (want, _) = conv2d_ref(&img, in_shape, &weights, &bias, k, k, 1);
        Layer {
            img,
            in_shape,
            weights,
            bias,
            want,
        }
    }
}

/// Run one shared-secret route over the transport and time it at each RTT.
fn measure_shared(
    name: &str,
    layer: &Layer,
    nbits: usize,
    rtts: &[u64],
    winograd: bool,
) -> RoutePoint {
    let out_h = layer.in_shape.h - 2;
    let out_w = layer.in_shape.w - 2;
    let cost = winograd_cost(layer.in_shape, layer.weights.len(), out_h, out_w);
    let muls = if winograd { cost.winograd } else { cost.direct };

    let mut ms_by_rtt = Vec::new();
    let mut comm = CommStats::default();
    let mut waits = 0u64;
    let mut correct = false;

    for (i, &rtt) in rtts.iter().enumerate() {
        let profile = if rtt == 0 {
            NetworkProfile::INSTANT
        } else {
            NetworkProfile::fixed_latency(std::time::Duration::from_micros(rtt * 500))
        };
        let (img, weights, bias) = (
            layer.img.clone(),
            layer.weights.clone(),
            layer.bias.clone(),
        );
        let in_shape = layer.in_shape;
        let t = Instant::now();
        let (outs, c) = run_parties(profile, move |me| {
            let (img, weights, bias) = (img.clone(), weights.clone(), bias.clone());
            move |be: &mut TransportBackend<MpcTransport>| {
                let sh = |be: &mut TransportBackend<MpcTransport>, v: &[i64]| -> Vec<ShareRef> {
                    let vals: Vec<Fr> = v
                        .iter()
                        .map(|&x| if me == 0 { residue(x, nbits) } else { 0 })
                        .collect();
                    be.share_secret_batch(&vals, nbits, 0)
                };
                let sin = sh(be, &img);
                let sw: Vec<Vec<ShareRef>> = weights.iter().map(|k| sh(be, k)).collect();
                let sb = sh(be, &bias);
                let w0 = be.sync_waits();
                let t0 = Instant::now();
                let (out, _) = if winograd {
                    conv2d_winograd(be, &sin, in_shape, &sw, &sb)
                } else {
                    conv2d(be, &sin, in_shape, &sw, &sb, 3, 3, 1)
                };
                // The convolution ends here. Opening the result is *verification
                // scaffolding*, not protocol: in a deployment these shares feed the
                // next layer's ReLU and are never opened. Timing the reveal loop
                // would charge the shared routes 2592 individual openings that the
                // homomorphic route does not pay, since its decryption returns a
                // whole channel at once.
                let conv_ns = t0.elapsed().as_nanos() as u64;
                let waits = be.sync_waits() - w0;
                let opened: Vec<u128> = out.iter().map(|s| be.reveal(s, None)).collect();
                (opened, waits, conv_ns)
            }
        });
        let _ = t;
        let (opened, w, conv_ns) = outs.into_iter().next().expect("three parties");
        let ms = conv_ns as f64 / 1e6;
        if i == 0 {
            comm = c;
            waits = w;
            // Winograd returns the convolution scaled; divide it out before
            // comparing, and require exact divisibility so a wrong answer cannot
            // be rounded into a right one.
            let scale = 1i64 << WINOGRAD_OUTPUT_SCALE_LOG2;
            correct = opened.iter().zip(&layer.want).all(|(o, &want)| {
                let v = to_signed(*o, nbits);
                if winograd {
                    v % scale == 0 && v / scale == want
                } else {
                    v == want
                }
            });
        }
        ms_by_rtt.push((rtt, ms));
    }

    RoutePoint {
        route: name.to_string(),
        muls: Some(muls),
        messages: comm.messages,
        bytes: comm.bytes,
        waits,
        ms_by_rtt,
        correct,
    }
}

/// The homomorphic route: coefficient-packed `ct x ct` convolution, then a
/// scheme switch back to shares.
///
/// The switch is the part that makes this comparable, and it is not free: three
/// parties each contribute a mask, the masked product is decrypted, and each
/// party's share is its own mask subtracted from the opened value. That is what
/// [`crate::bench::cnn`] does between every convolution and its ReLU, and
/// omitting it here would compare a ciphertext against a sharing.
fn measure_fhe(layer: &Layer, _nbits: usize, rtts: &[u64]) -> RoutePoint {
    let k = 3usize;
    let shape = ConvShape {
        in_c: layer.in_shape.c,
        h: layer.in_shape.h,
        w: layer.in_shape.w,
        kh: k,
        kw: k,
    };
    let out_c = layer.weights.len();
    // Parameters matching the CNN pipeline's depth-1 budget.
    let params = BfvParams {
        n: 4096,
        log2_q: 100,
        log2_t: 20,
    };
    let bfv = Bfv::new(params);
    let t_mask = params.t() - 1;

    let mut ms_by_rtt = Vec::new();
    let mut bytes = 0u64;
    let mut correct = false;
    let mut muls_used = 0usize;

    for (i, &rtt) in rtts.iter().enumerate() {
        // Key generation and key preparation are per-session and amortise over every
        // layer of every query, so they sit outside the timed region — charging one
        // layer for them would be the mirror of charging the shared routes for their
        // verification reveals.
        let (ck, pk) = bfv.keygen();
        let ek = bfv.encrypt_key(&pk);
        let dk = bfv.decrypt_key(&ck);
        let t = Instant::now();

        // Encrypt the image once and the model once, then one ct x ct product per
        // output channel — the whole point of coefficient packing.
        // Channel splitting, as `bench::cnn` does it: a layer too wide for one
        // ciphertext becomes several groups whose products are *added*, which is
        // free and consumes no depth. Omitting this would cap the homomorphic
        // route at small layers and make the crossover an artifact of our
        // implementation rather than of the packing.
        let (group_size, n_groups) = conv_pack::channel_groups(shape, params.n)
            .unwrap_or_else(|e| panic!("conv packing: {e}"));
        let gshape = conv_pack::group_shape(shape, group_size);
        let cxt: Vec<_> = (0..n_groups)
            .map(|gi| {
                let x = conv_pack::pack_input_group(
                    &layer.img, shape, gi, group_size, params.n, t_mask,
                );
                let cx = bfv.encrypt_poly_pre(&ek, &x);
                bfv.transform(&cx)
            })
            .collect();
        let mut link_bytes = 0usize;
        let mut opened_all: Vec<i64> = Vec::with_capacity(layer.want.len());

        for oc in 0..out_c {
            let mut prod: Option<crate::fhe::bfv::BfvCt> = None;
            for (gi, cxg) in cxt.iter().enumerate() {
                let wpoly = conv_pack::pack_kernel_group(
                    &layer.weights[oc], shape, gi, group_size, params.n, t_mask,
                );
                let cw = bfv.encrypt_poly_pre(&ek, &wpoly);
                let p = bfv.mul_pre(cxg, &cw);
                prod = Some(match prod {
                    None => p,
                    Some(acc) => bfv.add(&acc, &p),
                });
            }
            let prod = prod.expect("at least one channel group");
            // Scheme switch: three party masks, summed onto the product.
            let mut masked = prod;
            let mut masks = [0i64; 3];
            for (p, m) in masks.iter_mut().enumerate() {
                *m = ((p as i64 * 7919 + oc as i64 * 104729) % 1000) - 500;
                let mut mv = vec![0u64; params.n];
                for slot in mv.iter_mut() {
                    *slot = residue(*m, 20) as u64;
                }
                let cm = bfv.encrypt_poly_pre(&ek, &mv);
                link_bytes += bfv.ct_bytes(&cm);
                masked = bfv.add(&masked, &cm);
            }
            link_bytes += bfv.ct_bytes(&masked);
            let y = bfv.decrypt_poly_pre(&dk, &masked);
            let msum: i64 = masks.iter().sum();
            for oy in 0..gshape.out_h() {
                for ox in 0..gshape.out_w() {
                    let pos = gshape.output_position(oy, ox);
                    let centred = {
                        let tt = params.t() as i64;
                        let v = y[pos] as i64;
                        let v = if v >= tt / 2 { v - tt } else { v };
                        v - msum
                    };
                    opened_all.push(centred + layer.bias[oc]);
                }
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i == 0 {
            bytes = link_bytes as u64;
            correct = opened_all == layer.want;
            muls_used = out_c * n_groups;
        }
        // Latency enters this route only through the ciphertext transfers, which
        // `sleep_link` would charge; we report the byte volume and let the reader
        // apply their own link rather than simulating one here.
        ms_by_rtt.push((rtt, ms));
    }

    RoutePoint {
        route: "fhe: packed ct x ct + switch".to_string(),
        muls: Some(muls_used),
        messages: 0,
        bytes,
        waits: 0,
        ms_by_rtt,
        correct,
    }
}

pub fn run(
    in_c: usize,
    h: usize,
    w: usize,
    out_c: usize,
    nbits: usize,
    rtts: &[u64],
) -> Result<Report, String> {
    let layer = Layer::new(in_c, h, w, out_c, 3);
    let routes = vec![
        measure_shared("mpc: direct", &layer, nbits, rtts, false),
        measure_shared("mpc: winograd", &layer, nbits, rtts, true),
        measure_fhe(&layer, nbits, rtts),
    ];
    Ok(Report {
        in_c,
        h,
        w,
        out_c,
        k: 3,
        nbits,
        routes,
    })
}

const USAGE: &str = "usage: baru bench-conv [options]

Which backend should evaluate a convolution? Three strategies for one 3x3
stride-1 layer, all keeping the model secret, all ending with secret shares of
the output, all checked against the same plaintext reference.

  mpc: direct     k^2 interactive multiplies per output pixel
  mpc: winograd   Winograd F(2x2,3x3), 16 multiplies per 2x2 tile instead of 36
  fhe: packed     Cheetah-style coefficient packing, ct x ct, then a scheme
                  switch back to shares

options:
  --in-c N        input channels        (default 8)
  --size N        input height = width  (default 16)
  --out-c N       output channels       (default 8)
  --bits N        share width           (default 32)
  --json          machine-readable output
  -h, --help      this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let (mut in_c, mut size, mut out_c, mut nbits, mut json) = (8usize, 16usize, 8usize, 32usize, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut num = |what: &str| -> Result<usize, String> {
            it.next()
                .ok_or_else(|| format!("{what} needs a value"))?
                .parse()
                .map_err(|_| format!("bad {what}"))
        };
        match a.as_str() {
            "--in-c" => in_c = num("--in-c")?,
            "--size" => size = num("--size")?,
            "--out-c" => out_c = num("--out-c")?,
            "--bits" => nbits = num("--bits")?,
            "--json" => json = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg `{other}`\n\n{USAGE}")),
        }
    }

    let r = run(in_c, size, size, out_c, nbits, &[0])?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?);
        return Ok(());
    }
    println!(
        "One 3x3 convolution, {}x{}x{} -> {} channels, {}-bit shares",
        r.in_c, r.h, r.w, r.out_c, r.nbits
    );
    println!("all routes keep the model secret and end with shares of the output\n");
    println!("  {:<30}{:>10}{:>12}{:>8}{:>10}", "route", "muls", "bytes", "waits", "ms");
    for p in &r.routes {
        println!(
            "  {:<30}{:>10}{:>12}{:>8}{:>10.0}",
            p.route,
            p.muls.map(|m| m.to_string()).unwrap_or_else(|| "-".into()),
            p.bytes,
            p.waits,
            p.ms_by_rtt.first().map(|x| x.1).unwrap_or(0.0)
        );
    }
    let bad: Vec<&str> = r.routes.iter().filter(|p| !p.correct).map(|p| p.route.as_str()).collect();
    if !bad.is_empty() {
        return Err(format!("routes disagree with the plaintext reference: {bad:?}"));
    }
    println!("\n  all routes reproduce the plaintext reference");
    println!(
        "  NOTE the homomorphic route blocks zero times and moves ciphertexts; the shared\n  \
         routes block once and move ring elements. Which is cheaper is a property of the\n  \
         LAYER SHAPE, not of the cryptosystem: fhe's traffic scales with output channels,\n  \
         the shared routes' with the layer's whole volume. Sweep --size to see them cross."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Where the crossover is, which is the question the benchmark exists to answer.**
    ///
    /// The homomorphic route's traffic is dominated by ciphertexts and so scales with the
    /// *number of output channels*; the shared routes' traffic scales with the *volume* of the
    /// layer, `out_c x out_h x out_w x in_c x k^2`. Those grow differently, so neither wins
    /// everywhere, and a benchmark reporting one shape would be reporting a preference rather
    /// than a result.
    ///
    /// This sweeps the spatial size at fixed channels and prints where the order changes. The
    /// assertion is that a crossover exists inside the swept range — if it ever stops existing,
    /// one route dominates and the assignment question for convolution has a constant answer,
    /// which would itself be worth knowing.
    #[test]
    #[ignore] // several transport runs and FHE keygens
    fn the_conv_backends_cross_over_within_realistic_layer_sizes() {
        println!("    shape            direct kB   winograd kB      fhe kB   cheap(bytes)  wino ms   fhe ms   byte win pays below");
        struct Row {
            wino_kb: f64,
            fhe_kb: f64,
            wino_ms: f64,
            fhe_ms: f64,
        }
        let mut rows: Vec<Row> = Vec::new();
        let mut winners = std::collections::BTreeSet::new();
        let mut breakevens: Vec<f64> = Vec::new();
        for &(ic, side, oc) in &[
            (2usize, 8usize, 3usize),
            (4, 12, 4),
            (8, 16, 8),
            (16, 20, 8),
            (16, 28, 16),
        ] {
            let r = run(ic, side, side, oc, 32, &[0]).expect("runs");
            let by = |name: &str| -> f64 {
                r.routes.iter().find(|p| p.route.starts_with(name)).expect("route").bytes as f64
                    / 1000.0
            };
            let (d, w, f) = (by("mpc: direct"), by("mpc: winograd"), by("fhe"));
            let cheapest = if w <= d && w <= f {
                "winograd"
            } else if f <= d && f <= w {
                "fhe"
            } else {
                "direct"
            };
            let ms = |name: &str| -> f64 {
                r.routes
                    .iter()
                    .find(|p| p.route.starts_with(name))
                    .expect("route")
                    .ms_by_rtt
                    .first()
                    .map(|x| x.1)
                    .unwrap_or(0.0)
            };
            let (wms, fms) = (ms("mpc: winograd"), ms("fhe"));
            // The byte advantage only pays if the link is slow enough that the extra
            // bytes cost more time than the compute gap. Below this bandwidth the
            // shared route wins on total time; above it the homomorphic one does.
            let byte_gain_mb = (f - w) / 1000.0;
            let compute_penalty_s = (wms - fms) / 1000.0;
            let breakeven_mbps = if byte_gain_mb > 0.0 && compute_penalty_s > 0.0 {
                Some(byte_gain_mb * 8.0 / compute_penalty_s)
            } else {
                None
            };
            println!(
                "  {ic:2}x{side:2}x{side:2} -> {oc:2}   {d:10.0} {w:13.0} {f:11.0}   {cheapest:<9} \
{wms:8.0} {fms:8.0}   {}",
                breakeven_mbps
                    .map(|b| format!("{b:.2} Mbps"))
                    .unwrap_or_else(|| "-".into())
            );
            winners.insert(cheapest);
            rows.push(Row {
                wino_kb: w,
                fhe_kb: f,
                wino_ms: wms,
                fhe_ms: fms,
            });
            if let Some(b) = breakeven_mbps {
                breakevens.push((b * 100.0).round() / 100.0);
            }
            for p in &r.routes {
                assert!(p.correct, "route `{}` wrong at {ic}x{side}x{side}->{oc}", p.route);
            }
        }
        // A summary computed from the rows rather than asserted alongside them. Three
        // successive versions of this paragraph were wrong — "bytes decide", then
        // "compute decides, the byte win is irrelevant", then "fhe is faster at every
        // shape" — each true of the data in front of me at the time and falsified by
        // the next measurement. A sentence derived from the table cannot go stale
        // while the table changes.
        let n = rows.len();
        let mpc_time_wins = rows.iter().filter(|r| r.wino_ms < r.fhe_ms).count();
        let mpc_byte_wins = rows.iter().filter(|r| r.wino_kb < r.fhe_kb).count();
        let mpc_both = rows
            .iter()
            .filter(|r| r.wino_ms < r.fhe_ms && r.wino_kb < r.fhe_kb)
            .count();
        let fhe_both = rows
            .iter()
            .filter(|r| r.fhe_ms < r.wino_ms && r.fhe_kb < r.wino_kb)
            .count();
        println!(
            "\n  Winograd never loses to direct — same computation, fewer multiplications —\n  \
             so the boundary that matters is winograd against fhe."
        );
        println!(
            "  Of {n} shapes: the shared route is cheaper in bytes on {mpc_byte_wins} and faster \
             on {mpc_time_wins};\n  \
             it dominates outright on {mpc_both}, and the homomorphic route dominates outright \
             on {fhe_both}."
        );
        println!(
            "  Where neither dominates, the last column is the bandwidth below which the shared\n  \
             route's byte advantage outweighs its compute penalty. Those figures are high — a\n  \
             layer has to be large, or a link very fast, before the homomorphic route is the\n  \
             right answer on total time."
        );

        assert!(
            winners.len() >= 2,
            "a crossover must exist in this range, or convolution has a constant best backend \
             and this benchmark should say so instead: {winners:?}"
        );
        // The break-even bandwidth must *fall* as the layer grows: the shared route's traffic
        // scales with the layer volume while the homomorphic route's scales with output
        // channels, so the byte advantage erodes. If this ever stopped holding, the two cost
        // models would not be what this benchmark claims they are.
        let falling = breakevens.windows(2).all(|p| p[1] <= p[0] + 1e-9);
        println!("  break-even bandwidths, contested shapes only: {breakevens:?}");
        assert!(
            falling,
            "the break-even bandwidth must fall as the layer grows — the shared route's traffic \
             scales with volume and the homomorphic route's with output channels — got {breakevens:?}"
        );
        assert!(
            mpc_both >= 1 && fhe_both >= 1,
            "each route must dominate outright somewhere in this range, or the sweep is not \
             bracketing the crossover: shared {mpc_both}, homomorphic {fhe_both} of {n}"
        );
    }

    /// **Every route must compute the same convolution, or the comparison is meaningless.**
    ///
    /// This is the gate the whole benchmark rests on. The three strategies are wildly
    /// different — `k^2` interactive multiplies, a Winograd transform with a scale to divide
    /// out, and a coefficient-packed homomorphic product opened through a masked decryption —
    /// and each has its own way of being subtly wrong. Checking them all against one plaintext
    /// reference is what makes a cost comparison between them mean anything at all.
    #[test]
    #[ignore] // three-party transport plus an FHE keygen
    fn every_route_computes_the_same_convolution() {
        let r = run(2, 8, 8, 3, 32, &[0]).expect("benchmark runs");
        for p in &r.routes {
            println!(
                "  {:<30} muls {:>6}  bytes {:>9}  waits {:>3}  correct {}",
                p.route,
                p.muls.map(|m| m.to_string()).unwrap_or_else(|| "-".into()),
                p.bytes,
                p.waits,
                p.correct
            );
        }
        for p in &r.routes {
            assert!(p.correct, "route `{}` disagrees with the plaintext reference", p.route);
        }
    }
}
