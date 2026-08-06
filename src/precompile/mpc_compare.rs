//! **The comparison gadget** — the first precompile whose champion depends on
//! the link rather than on the shape.
//!
//! # Why this gadget and not another
//!
//! Attributing a compiled network's blocking to its schedule levels
//! (`bench-compiled-nn`) says that three of eight interactive levels hold a
//! comparison and that those three carry about \\SI{72}{\\percent} of the total
//! blocking, at nine blocks each against two for a multiplication level. So the
//! comparison is the primitive whose implementation is worth choosing, and Baru
//! has three of them:
//!
//! | backend | waits | bytes | the case for it |
//! |---|---|---|---|
//! | `a2b` | 10 | \\SI{127}{\\kilo\\byte} | no preprocessing to schedule |
//! | `falcon` | 11 | \\SI{49}{\\kilo\\byte} | cheapest on the wire, by \\(2.2\\times\\) |
//! | `gc` | 3 | \\SI{110}{\\kilo\\byte} | constant round, cheapest in blocking |
//!
//! Measured at 64 comparisons, `ℓ = 32`, with Falcon's preprocessing filled in
//! advance so the figure is its online phase.
//!
//! # What the numbers force
//!
//! `gc` **strictly dominates** `a2b` here, in blocking and in bytes both, which
//! is worth stating plainly because `a2b` was the default the compiled executor
//! used: an improvement was available that no correctness test and no byte count
//! would ever have surfaced, and it took a blocking counter to see it.
//!
//! `falcon` against `gc` looked like the interesting pair, because neither
//! dominates on those two columns — `falcon` sends \(2.2\times\) fewer bytes and
//! blocks \(3.7\times\) more often. We predicted from that a champion which
//! flipped with the link, byte-cheap on a thin pipe and block-cheap on a slow
//! one, and built [`Registry::champion_for`] to express it. Then we measured, and
//! the prediction was wrong: `gc` is fastest at every point of the grid, and the
//! reason is a third term neither column contains.
//!
//! | backend | local (network free) | at RTT 0, \SI{10}{\mega\bit\per\second} | at RTT 40, \SI{100}{\mega\bit\per\second} |
//! |---|---|---|---|
//! | `a2b` | \SI{33}{\milli\second} | \SI{206}{\milli\second} | \SI{334}{\milli\second} |
//! | `falcon` | \SI{79}{\milli\second} | \SI{130}{\milli\second} | \SI{1055}{\milli\second} |
//! | `gc` | **\SI{13}{\milli\second}** | **\SI{105}{\milli\second}** | **\SI{201}{\milli\second}** |
//!
//! Those local figures are the \num{64}-wide microbenchmark, and they understate
//! the spread: most of each is fixed per-call cost. Measured on a whole diagnosis
//! instead --- the workload the assignment is actually for --- the nonlinearity
//! compute is **gc \SI{137}{\milli\second}, a2b \SI{1063}{\milli\second},
//! falcon \SI{2248}{\milli\second}** for the same \num{7137} comparisons, so the
//! true marginal spread is \(1 : 7.8 : 16.4\) rather than \(1 : 2.5 : 6.1\). The
//! registry carries session totals rather than these; see the `SHAPE` constant.
//!
//! With the network free, `falcon` costs \(6\times\) what `gc` does. That is
//! **local cryptographic work** — its \(\Z_p\) arithmetic and its preprocessing —
//! and a model of the form `waits × RTT + bytes / bandwidth` charges every
//! candidate zero for it. So the model could not rank the three on a fast link at
//! all, and on a thin pipe it preferred `falcon` at \SI{130}{\milli\second} over
//! `gc` at \SI{105}{\milli\second}: the \SI{78}{\kilo\byte} `falcon` saves is
//! worth \SI{62}{\milli\second} at \SI{10}{\mega\bit\per\second}, and it pays
//! \SI{66}{\milli\second} more in local work to save it. `MeasuredCost` therefore
//! carries a `local_secs` term, measured on `NetworkProfile::INSTANT`, and with it
//! the selection agrees with the wall clock at every point swept.
//!
//! Two things survive the correction and one does not. What survives is the
//! machinery — a champion keyed on the link is the right shape, it is exercised on
//! synthetic entries that do trade places, and the byte-versus-blocking trade is
//! real at the level of whole routes, where Baru's own nonlinearity crossover
//! moves between 30 and \SI{40}{\mega\bit\per\second}. What survives more
//! importantly is that the assignment was worth making at all: `gc` over `a2b` is
//! a \(2.75\times\) reduction in blocking on the compiled network. What does not
//! survive is the claim that this gadget is where the champion flips. It is not,
//! and the test that asserted a flip now asserts the measured fact instead.
//!
//! # What this module deliberately does not do
//!
//! It does not search. The costs here are transcribed from measurements taken by
//! `the_wait_cost_of_each_comparison_backend`, and selection is an evaluation of
//! the latency model on three candidates — not an on-line exploration. That is
//! the design the project settled on for neural networks specifically: the space
//! of building blocks is small and stable, the measurements accumulate, and a
//! lookup beats a search that would have to pay for its own exploration at
//! deployment time.
//!
//! It also does not weaken the human gate. Registration is a code change, the
//! costs are transcribed by hand from a named test, and `champion_for` only ranks
//! entries a maintainer has already admitted.

use crate::bench::comm::NetworkProfile;
use crate::interpreter::backend::{Backend, ShareRef};
use crate::interpreter::transport_backend::NonlinMode;

use super::{MeasuredCost, PrecompileMeta, Registry};

/// One way of comparing secret-shared values.
///
/// The signature is the batched one deliberately: a per-element comparison
/// interface would make the round-efficient implementations unrepresentable, and
/// this repo has already measured what per-element interfaces cost
/// (`matvec(128×64)` at \\SI{26}{\\second} and 115k messages).
pub trait MpcCompare: Send + Sync {
    /// `[a_i < b_i]` for each pair, as a shared bit.
    fn lt_batch(&self, be: &mut dyn Backend, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef>;

    /// Which nonlinearity mode this entry needs the backend to be in, if any.
    ///
    /// The comparison protocol lives *below* `Backend::lt_batch`, which
    /// dispatches on the backend's mode. So an entry does not reimplement the
    /// protocol — it selects it. That is why the compiled executor inherits a
    /// protocol choice without the compiler knowing anything about it.
    fn mode(&self) -> Option<NonlinMode>;
}

macro_rules! cmp_entry {
    ($name:ident, $mode:expr) => {
        pub struct $name;
        impl MpcCompare for $name {
            fn lt_batch(
                &self,
                be: &mut dyn Backend,
                pairs: &[(ShareRef, ShareRef)],
            ) -> Vec<ShareRef> {
                be.lt_batch(pairs)
            }
            fn mode(&self) -> Option<NonlinMode> {
                Some($mode)
            }
        }
    };
}

cmp_entry!(A2bCompare, NonlinMode::A2b);
cmp_entry!(FalconCompare, NonlinMode::Falcon);
cmp_entry!(GcCompare, NonlinMode::Gc);

/// The shape every cost below was measured at. Quoted on each entry so a figure
/// cannot travel without it.
/// The shape every cost below is stated at. Quoted on each entry so a figure
/// cannot travel without it.
///
/// # Why these are session totals and not marginal costs
///
/// Every figure here is the whole cost of one \num{64}-wide call, fixed part
/// included, and the three terms are therefore on the same footing. That is worth
/// recording because we tried to improve it and had to revert.
///
/// The microbenchmark's `local_secs` really is mostly fixed cost. Measuring 64
/// comparisons in a fresh session gives 13, 33 and \SI{79}{\milli\second} for
/// gc, a2b and falcon; measuring the *same* comparison work inside a whole CNN
/// diagnosis --- \num{7137} comparisons, the workload this assignment exists for
/// --- gives 137, 1063 and \SI{2248}{\milli\second}, i.e. marginal costs of 1.2,
/// 9.5 and \SI{20.2}{\milli\second} at the same width. So a \num{64}-wide call is
/// roughly \(10\times\) its own marginal cost, and the *ratios* differ too:
/// \(1 : 2.5 : 6.1\) from the microbenchmark against \(1 : 7.8 : 16.4\) marginal.
/// The microbenchmark understates how much cheaper garbled circuits are.
///
/// Substituting the marginal figures **broke the model**, and
/// `the_selected_backend_is_the_one_that_measures_fastest` caught it: with local
/// cost divided by ten and `bytes` left alone, the byte term dominates and the
/// selection flips to Falcon on a thin pipe, where measurement says garbled
/// circuits win. The mistake was to make one term marginal and not the other ---
/// and `bytes` cannot be decomposed the same way, because Falcon's
/// \SI{49.1}{\kilo\byte} includes *its own* preprocessing traffic, which is not a
/// fixed cost common to the candidates but a protocol-specific one. There is no
/// common term to strip.
///
/// So the entries stay as validated totals. The marginal measurement is not
/// discarded --- it says the assignment is *more* right than these numbers imply,
/// never less, since every correction to the ratio favours the champion already
/// selected. Making both terms marginal would need per-diagnosis byte counts for
/// all three backends measured on one footing, which is a larger job than the
/// ranking currently requires.
const SHAPE: &str = "64 comparisons, l=32, online (preprocessing pre-filled), \
                     session totals incl. fixed cost; local_secs on INSTANT";

fn build_default_registry() -> Registry<dyn MpcCompare> {
    let mut r = Registry::new(
        PrecompileMeta {
            id: "cmp-a2b",
            author: "baru-core",
            description: "batched bit decomposition; MSB of the difference",
            cost: Some(MeasuredCost {
                waits: 10,
                bytes: 127_100,
                local_secs: 0.033,
                at: SHAPE,
            }),
        },
        Box::new(A2bCompare) as Box<dyn MpcCompare>,
    );
    r.register(
        PrecompileMeta {
            id: "cmp-falcon",
            author: "baru-core",
            description: "Falcon Pi_DReLU over Z_p; cheapest on the wire, dearest in blocking",
            cost: Some(MeasuredCost {
                waits: 11,
                bytes: 49_100,
                local_secs: 0.079,
                at: SHAPE,
            }),
        },
        Box::new(FalconCompare),
    );
    r.register(
        PrecompileMeta {
            id: "cmp-gc",
            author: "baru-core",
            description: "three-party garbled circuits; constant round",
            cost: Some(MeasuredCost {
                waits: 3,
                bytes: 109_900,
                local_secs: 0.013,
                at: SHAPE,
            }),
        },
        Box::new(GcCompare),
    );
    // The unconditional default, for a caller with no link to describe. `gc`
    // rather than `a2b` because it dominates it outright — see the module note.
    assert!(r.set_champion("cmp-gc"));
    r
}

/// The process-wide registry.
pub fn registry() -> &'static Registry<dyn MpcCompare> {
    use std::sync::OnceLock;
    static R: OnceLock<Registry<dyn MpcCompare>> = OnceLock::new();
    R.get_or_init(build_default_registry)
}

/// The comparison backend to use on this link, as a mode the backend can be set
/// to.
///
/// This is the whole per-subtask assignment, for one subtask, in the form a
/// caller can actually use: given a deployment, which protocol should the
/// comparison levels run.
pub fn mode_for(profile: NetworkProfile) -> Option<NonlinMode> {
    registry().champion_for(profile).mode()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn profile(rtt_ms: u64, mbps: Option<f64>) -> NetworkProfile {
        let p = NetworkProfile::fixed_latency(Duration::from_micros(rtt_ms * 500));
        match mbps {
            Some(m) => p.with_bandwidth(m * 1_000_000.0 / 8.0),
            None => p,
        }
    }

    /// **Which backend wins, across the grid — and the answer is one of them, everywhere.**
    ///
    /// This test was written to assert that the champion *flips*, on the strength of Falcon
    /// being \(2.2\times\) cheaper on the wire. Measurement says otherwise: garbled circuits
    /// win at every point of the grid, because Falcon's wire advantage is worth less than its
    /// \(6\times\) penalty in local arithmetic at any bandwidth in this range. The assertion is
    /// now the measured fact, so that a future change which *does* produce a flip — a cheaper
    /// Falcon local path, a garbling scheme with more traffic, a thinner link than we sweep —
    /// fails this test and gets looked at.
    ///
    /// The regime-keyed machinery is not thereby pointless, and it is worth being precise about
    /// why: it is exercised on synthetic entries in [`super::tests`], where a blocking-cheap and
    /// a byte-cheap candidate do trade places, and the trade it expresses is one Baru measures
    /// at the level of whole routes (garbled versus Falcon nonlinearity in `bench-medmnist`,
    /// which crosses over between 30 and \SI{40}{\mega\bit\per\second}). What collapsed is the
    /// claim that *this* gadget is the example of it.
    #[test]
    fn garbled_circuits_win_the_comparison_at_every_point_of_the_grid() {
        let grid = [
            (0u64, Some(10.0)),
            (0, Some(100.0)),
            (0, None),
            (20, Some(10.0)),
            (20, Some(100.0)),
            (20, None),
            (100, Some(10.0)),
        ];
        println!("    rtt   mbps   champion     predicted ms");
        for (rtt, mbps) in grid {
            let p = profile(rtt, mbps);
            let m = registry().champion_meta_for(p);
            let ms = m.cost.expect("champions are costed").predict_secs(p) * 1000.0;
            println!(
                "  {rtt:5}  {:5}   {:<12} {ms:8.1}",
                mbps.map(|m| format!("{m:.0}"))
                    .unwrap_or_else(|| "inf".into()),
                m.id
            );
            assert_eq!(
                m.id, "cmp-gc",
                "garbled circuits measured fastest at every point swept; a flip here means the \
                 cost table changed and the module note needs rewriting (rtt {rtt}, {mbps:?})"
            );
        }
    }

    /// **`a2b` is dominated, so it never wins anywhere on the grid.**
    ///
    /// A dominated entry is worth keeping registered — it is the baseline the other two are
    /// measured against, and it is what the executor used to default to — but if selection ever
    /// returned it, either the costs or the model would be wrong.
    #[test]
    fn the_dominated_backend_is_never_selected() {
        for rtt in [0u64, 1, 5, 20, 100, 500] {
            for mbps in [Some(1.0), Some(10.0), Some(100.0), Some(1000.0), None] {
                let p = profile(rtt, mbps);
                let id = registry().champion_meta_for(p).id;
                assert_ne!(
                    id, "cmp-a2b",
                    "a2b is dominated in both terms and must never be selected (rtt {rtt}, {mbps:?})"
                );
            }
        }
    }

    /// **Does the model pick the backend that actually wins? Measured, at two grid points.**
    ///
    /// Everything above is a prediction from transcribed costs. This runs the comparison under
    /// each candidate over a simulated link and checks that the one the registry selects is the
    /// one the wall clock prefers — the step that turns a lookup table into a validated one.
    ///
    /// Two points, chosen because the prediction disagrees between them: a zero-latency thin pipe
    /// where bytes decide and Falcon is predicted to win, and a high-latency link where blocking
    /// decides and garbled circuits are. A table that got the *direction* of the trade wrong
    /// would pass a single-point check and fail this one.
    ///
    /// The tolerance is deliberately loose. Wall clock on this harness carries run-to-run
    /// variance around \SI{13}{\percent}, so the test asserts the selected backend is not
    /// meaningfully *worse* than the alternative rather than that it is strictly better — a
    /// selection that ties is a selection that costs nothing, and asserting strict wins on a
    /// noisy measurement is how this repo previously published a compiler speedup that was noise.
    #[test]
    #[ignore] // three-party transport, several full runs
    fn the_selected_backend_is_the_one_that_measures_fastest() {
        use crate::bench::hybrid_nn::{run_parties, MpcTransport};
        use crate::crypto::types::Fr;
        use crate::interpreter::transport_backend::{FalconPoolSpec, TransportBackend};
        use std::time::Instant;

        const NBITS: usize = 32;
        const WIDTH: usize = 64;

        // One timed run of `WIDTH` comparisons under `mode`, over `profile`.
        let time_one = |profile: NetworkProfile, mode: NonlinMode| -> f64 {
            let t = Instant::now();
            let (_o, _c) = run_parties(profile, move |me| {
                move |be: &mut TransportBackend<MpcTransport>| {
                    if let Some(w) = crate::bench::cnn::wire_bits_for(NBITS) {
                        be.set_wire_bits(w);
                    }
                    let vals: Vec<Fr> = (0..WIDTH)
                        .map(|i| {
                            if me == 0 {
                                ((i as i64 - 8) as i128).rem_euclid(1i128 << NBITS) as Fr
                            } else {
                                0
                            }
                        })
                        .collect();
                    let xs = be.share_secret_batch(&vals, NBITS, 0);
                    let ys = be.share_secret_batch(&vals, NBITS, 0);
                    be.set_nonlin(mode);
                    if mode == NonlinMode::Falcon {
                        let _ = be.fill_falcon_pool(&FalconPoolSpec {
                            comparisons: WIDTH,
                            truncations: 0,
                            nbits: NBITS as u32,
                            lift_bits: NBITS as u32,
                            shift: 0,
                        });
                    }
                    let pairs: Vec<_> = xs.iter().cloned().zip(ys.iter().cloned()).collect();
                    let out = be.lt_batch(&pairs);
                    be.reveal(&out[0], None)
                }
            });
            t.elapsed().as_secs_f64() * 1000.0
        };

        let points = [
            ("no latency, unlimited", profile(0, None)),
            ("thin pipe, no latency", profile(0, Some(10.0))),
            ("high latency", profile(40, Some(100.0))),
        ];
        let mut failures: Vec<(&str, &str, f64, &str, f64, f64)> = Vec::new();
        for (label, p) in points {
            let picked = registry().champion_meta_for(p);
            let picked_mode = registry()
                .champion_for(p)
                .mode()
                .expect("entries carry a mode");
            let mut measured: Vec<(&str, f64)> = Vec::new();
            for (id, mode) in [
                ("cmp-a2b", NonlinMode::A2b),
                ("cmp-falcon", NonlinMode::Falcon),
                ("cmp-gc", NonlinMode::Gc),
            ] {
                measured.push((id, time_one(p, mode)));
            }
            let best = measured
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .expect("three modes");
            let picked_ms = measured
                .iter()
                .find(|(id, _)| *id == picked.id)
                .expect("the picked entry was measured")
                .1;
            println!("  {label} — registry picks {} ({picked_mode:?})", picked.id);
            for (id, ms) in &measured {
                let mark = if *id == best.0 { " *fastest" } else { "" };
                println!("      {id:<12} {ms:8.1} ms{mark}");
            }
            // The selection must be at least as good as the best, within the harness's noise.
            let overhead = picked_ms / best.1;
            println!("      selected/fastest = {overhead:.2}x");
            failures.push((label, picked.id, picked_ms, best.0, best.1, overhead));
        }
        for (label, pid, pms, bid, bms, over) in &failures {
            if *over > 1.30 {
                println!(
                    "  MISMATCH {label}: picked {pid} at {pms:.0} ms, best {bid} at {bms:.0} ms ({over:.2}x)"
                );
            }
        }
        assert!(
            failures.iter().all(|f| f.5 <= 1.30),
            "the cost table mispredicts on at least one link; see the MISMATCH lines above"
        );
    }

    /// An uncosted gadget must behave exactly as it did before `champion_for` existed.
    ///
    /// This is the compatibility property that lets the field be added without auditing every
    /// registry: `matvec` records no costs, and its champion is unconditional.
    #[test]
    fn an_uncosted_registry_falls_back_to_its_explicit_champion() {
        let r = crate::precompile::mpc_matvec::registry();
        let unconditional = r.champion_meta().id;
        for rtt in [0u64, 20, 500] {
            assert_eq!(
                r.champion_meta_for(profile(rtt, Some(10.0))).id,
                unconditional,
                "an uncosted registry must ignore the link"
            );
        }
    }
}
