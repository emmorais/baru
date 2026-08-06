//! Precompiled gadgets: hand-optimized, native Rust implementations of
//! high-level cryptographic operations that Baru programs invoke as
//! **builtins** instead of hand-deriving them from primitive `Backend`/
//! `FheBackend` calls.
//!
//! # Why
//!
//! `src/bench/nn.rs` found that Baru's naive, per-element MPC matvec costs
//! O(n) communication rounds with no batching (`matvec(128×64)` ≈ 26s,
//! 115k messages). Re-deriving that same naive loop every time a Baru
//! program needs a matvec bakes the bottleneck into the language. A
//! **precompile** is the fix: a native routine that implements the whole
//! operation efficiently (e.g. one batched communication round instead of
//! one per multiplication — see [`mpc_matvec::BatchedMatVec`]), registered
//! once, and reused by every program that calls the corresponding builtin.
//!
//! # Vocabulary
//!
//! - **Gadget** — a named, reusable operation with a fixed signature (e.g.
//!   "MatVec for MPC"). Defined as one Rust trait per (operation, backend
//!   family) pair, e.g. [`mpc_matvec::MpcMatVec`]. The trait *is* the
//!   interface competing implementations must satisfy.
//! - **Precompile** — one concrete, named implementation of a Gadget. There
//!   can be many for the same Gadget (that's the point — see below).
//! - **[`Registry`]** — holds every registered Precompile for one Gadget
//!   trait, plus which one is the current **champion** — the implementation
//!   Baru's builtins actually dispatch to. The champion is set explicitly (by
//!   a maintainer, informed by a `bench-nn` measurement), never
//!   auto-selected — only a real measurement can say which precompile is
//!   actually fastest for a given shape/backend, and only review can say
//!   whether it's trustworthy.
//!
//! # Comparing precompiles
//!
//! Because every Precompile for a Gadget shares one trait, they're
//! interchangeable and directly comparable: run each one through the same
//! correctness check (against a plaintext reference — mandatory, never
//! waived) and the same [`crate::bench::comm`] instrumentation
//! (`bench-nn --competition`), and see which one actually wins for a given
//! shape. This is a benchmarking/leaderboard tool, not an automated
//! reward mechanism — deliberately so. A Precompile is *compiled code*, and
//! there is no lightweight way to verify compiled code beyond running it and
//! reading it: unlike a pure data computation, whose result can be checked
//! trustlessly by re-execution, a faster implementation could be faster because
//! it is wrong. Promoting a challenger to champion is a
//! reviewed, human decision informed by the benchmark, not something the
//! benchmark result alone should trigger.
//!
//! # Adding a new gadget
//!
//! 1. Define a trait: `pub trait MpcRelu: Send + Sync { fn relu(&self, be:
//!    &mut dyn Backend, xs: &[ShareRef]) -> Vec<ShareRef>; }`
//! 2. Implement it at least once (a correct, possibly-naive baseline).
//! 3. `Registry::new(...)` it, `register()` competitors, `set_champion()`.
//! 4. Wire a Baru builtin to look up the champion and call it (see
//!    `Participant::register_builtins`'s `matvec` for the pattern).

pub mod mpc_compare;
pub mod mpc_matvec;

/// Metadata every Precompile carries, independent of which Gadget trait it
/// implements — what the [`Registry`]/competition needs to identify, credit,
/// and describe an entry.
#[derive(Debug, Clone, Copy)]
pub struct PrecompileMeta {
    /// Stable, unique-per-gadget identifier (e.g. "matvec-naive").
    pub id: &'static str,
    /// Free-form attribution. A submission system with real identities would
    /// replace this with a verified key rather than a string.
    pub author: &'static str,
    pub description: &'static str,
    /// What this entry was **measured** to cost, if it has been measured.
    ///
    /// `None` means "registered but never costed", and such an entry is never
    /// selected by [`Registry::champion_for`] — an unmeasured implementation
    /// cannot win a comparison it did not enter.
    pub cost: Option<MeasuredCost>,
}

/// A measured cost, in the two terms the latency model actually charges.
///
/// Both are **per unit of work** at a stated shape, so a caller can scale them.
/// They are measurements, not estimates: `waits` comes from
/// [`crate::crypto::net::Mailbox::sync_waits`] and `bytes` from
/// [`crate::bench::comm`]. The reason both are needed is that neither orders the
/// candidates on its own — Baru's comparison backends include one that is
/// cheapest in blocking and dearest in bytes and another that is the reverse, so
/// which wins is a property of the *link*, not of the protocol.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredCost {
    /// Times a party blocked for a peer. Multiplied by the round-trip time.
    pub waits: u64,
    /// Bytes on the wire, per party. Divided by the bandwidth.
    pub bytes: u64,
    /// Wall-clock seconds with **no network cost at all** — the local
    /// cryptographic work, measured on `NetworkProfile::INSTANT`.
    ///
    /// This term exists because leaving it out gave a wrong answer, not because
    /// the model looked incomplete. A network-only model charges every candidate
    /// zero at zero latency and unlimited bandwidth, so it cannot rank them
    /// there; and the candidates are not close on this axis. Baru's three
    /// comparison backends measure \SI{13}{\milli\second} (garbled),
    /// \SI{33}{\milli\second} (bit decomposition) and \SI{79}{\milli\second}
    /// (Falcon) for the same 64 comparisons with the network free — a
    /// \(6\times\) spread that decides the ranking on any fast link and that
    /// `waits` and `bytes` between them cannot express.
    pub local_secs: f64,
    /// The shape the figures were measured at, for the record — a cost quoted
    /// without the shape it was measured at is how this repo's earlier figures
    /// came to be compared against each other wrongly.
    pub at: &'static str,
}

impl MeasuredCost {
    /// Predicted seconds under `profile`: local work, plus blocking against the
    /// round trip, plus bytes against the bandwidth.
    ///
    /// The first term is the one this repo added late and learnt from. The
    /// network-only version of this function selected a backend that measured
    /// \(1.24\times\) slower than the alternative on a thin pipe, because the
    /// backend it preferred was cheaper on the wire and considerably dearer in
    /// local arithmetic.
    pub fn predict_secs(&self, profile: crate::bench::comm::NetworkProfile) -> f64 {
        let rtt = 2.0 * profile.one_way_latency.as_secs_f64();
        let bw = self.bytes as f64 / profile.bandwidth_bytes_per_sec.unwrap_or(f64::INFINITY);
        self.local_secs + self.waits as f64 * rtt + bw
    }
}

/// A generic registry of competing Precompiles for one Gadget trait `P`
/// (e.g. `Registry<dyn mpc_matvec::MpcMatVec>`). Written once, reusable for
/// every future gadget — adding a new competing implementation never
/// requires touching this type.
pub struct Registry<P: ?Sized> {
    entries: Vec<(PrecompileMeta, Box<P>)>,
    champion: usize,
}

impl<P: ?Sized> Registry<P> {
    /// Start a registry with one (typically the reference/baseline) entry,
    /// which becomes the initial champion.
    pub fn new(first_meta: PrecompileMeta, first: Box<P>) -> Self {
        Registry {
            entries: vec![(first_meta, first)],
            champion: 0,
        }
    }

    /// Register a competing implementation. Does not change the champion —
    /// call [`Self::set_champion`] once a benchmark says it should win.
    pub fn register(&mut self, meta: PrecompileMeta, p: Box<P>) {
        self.entries.push((meta, p));
    }

    /// Point the champion at the entry with this id. Returns `false` (no-op)
    /// if no entry has that id.
    pub fn set_champion(&mut self, id: &str) -> bool {
        match self.entries.iter().position(|(m, _)| m.id == id) {
            Some(idx) => {
                self.champion = idx;
                true
            }
            None => false,
        }
    }

    /// The implementation Baru builtins should dispatch to right now.
    pub fn champion(&self) -> &P {
        &self.entries[self.champion].1
    }

    pub fn champion_meta(&self) -> &PrecompileMeta {
        &self.entries[self.champion].0
    }

    /// The implementation that is cheapest **for this link**, among those that
    /// have been measured.
    ///
    /// This exists because a single global champion is the wrong shape for the
    /// question. Baru's three comparison backends are measured at 64
    /// comparisons, `ℓ = 32`: garbled circuits block 3 times and send
    /// \SI{110}{\kilo\byte}, Falcon blocks 11 times and sends 49 --- so
    /// garbled circuits win a high-latency link and Falcon wins a thin one, and
    /// there is no assignment that is right for both. `champion()` answers
    /// "which one by default"; this answers "which one here".
    ///
    /// Registration remains the human gate: this ranks entries a maintainer has
    /// already admitted and costed, and it never promotes an unmeasured one. If
    /// no entry carries a cost it falls back to [`Self::champion`], so a gadget
    /// that has never been costed behaves exactly as before.
    pub fn champion_for(&self, profile: crate::bench::comm::NetworkProfile) -> &P {
        &self.entries[self.champion_index_for(profile)].1
    }

    /// [`Self::champion_for`]'s choice, as metadata — for a benchmark that wants
    /// to report *which* entry a regime selected rather than only run it.
    pub fn champion_meta_for(
        &self,
        profile: crate::bench::comm::NetworkProfile,
    ) -> &PrecompileMeta {
        &self.entries[self.champion_index_for(profile)].0
    }

    fn champion_index_for(&self, profile: crate::bench::comm::NetworkProfile) -> usize {
        // Ranked on the predicted time first, then on the raw counters as
        // tie-breaks. The tie-breaks are not cosmetic: `NetworkProfile::INSTANT`
        // has zero latency *and* unlimited bandwidth, so the prediction is
        // \SI{0}{\second} for every candidate and the model genuinely cannot
        // discriminate. Falling back to registration order there returned a
        // strictly dominated entry, which is a wrong answer rather than an
        // arbitrary one — so when the link cannot separate two candidates, prefer
        // the one that is cheaper on the counters themselves, blocking first.
        // With that, an entry dominated in both terms can never be selected.
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, (m, _))| m.cost.map(|c| (i, c)))
            .min_by(|(_, a), (_, b)| {
                let (pa, pb) = (a.predict_secs(profile), b.predict_secs(profile));
                pa.partial_cmp(&pb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.waits.cmp(&b.waits))
                    .then(a.bytes.cmp(&b.bytes))
            })
            .map(|(i, _)| i)
            .unwrap_or(self.champion)
    }

    /// Look up a specific entry by id (e.g. to force a particular
    /// implementation in a benchmark/competition run).
    pub fn get(&self, id: &str) -> Option<&P> {
        self.entries
            .iter()
            .find(|(m, _)| m.id == id)
            .map(|(_, p)| &**p)
    }

    /// Every registered entry, for a competition/leaderboard run.
    pub fn entries(&self) -> impl Iterator<Item = (&PrecompileMeta, &P)> {
        self.entries.iter().map(|(m, p)| (m, &**p))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    trait Dummy: Send + Sync {
        fn value(&self) -> i32;
    }
    struct Fixed(i32);
    impl Dummy for Fixed {
        fn value(&self) -> i32 {
            self.0
        }
    }

    fn meta(id: &'static str) -> PrecompileMeta {
        PrecompileMeta {
            id,
            author: "test",
            description: "",
            cost: None,
        }
    }

    fn costed(id: &'static str, waits: u64, bytes: u64) -> PrecompileMeta {
        PrecompileMeta {
            id,
            author: "test",
            description: "",
            cost: Some(MeasuredCost {
                waits,
                bytes,
                local_secs: 0.0,
                at: "test",
            }),
        }
    }

    /// **Selection ranks only measured entries, and only ever among admitted ones.**
    ///
    /// Two properties in one test because they are the same property: an entry with no measured
    /// cost is not a candidate, so registering something uncosted cannot silently take over
    /// dispatch, and an entry that has never been admitted is not in the registry at all.
    #[test]
    fn regime_selection_ignores_uncosted_entries() {
        let p =
            crate::bench::comm::NetworkProfile::fixed_latency(std::time::Duration::from_millis(10));
        let mut r = Registry::new(
            costed("cheap", 1, 1_000_000),
            Box::new(Fixed(1)) as Box<dyn Dummy>,
        );
        // An uncosted challenger that would win on any real measurement it might have.
        r.register(meta("unmeasured"), Box::new(Fixed(2)));
        assert_eq!(
            r.champion_meta_for(p).id,
            "cheap",
            "an uncosted entry must not be selectable, whatever it might cost"
        );
        // Once costed, and only then, it can win.
        let mut r2 = Registry::new(
            costed("dear", 100, 1_000_000),
            Box::new(Fixed(1)) as Box<dyn Dummy>,
        );
        r2.register(costed("cheap", 1, 1_000), Box::new(Fixed(2)));
        assert_eq!(r2.champion_meta_for(p).id, "cheap");
        // And `champion()` is untouched by any of it: the explicit default still stands.
        assert_eq!(r2.champion_meta().id, "dear");
    }

    /// The two terms must actually trade, or the model collapses to one column.
    #[test]
    fn blocking_wins_a_slow_link_and_bytes_win_a_thin_one() {
        // A blocks rarely and sends a lot; B is the reverse.
        let mut r = Registry::new(
            costed("blocky", 100, 1_000),
            Box::new(Fixed(1)) as Box<dyn Dummy>,
        );
        r.register(costed("chatty", 1, 10_000_000), Box::new(Fixed(2)));
        let high_latency =
            crate::bench::comm::NetworkProfile::fixed_latency(std::time::Duration::from_millis(50))
                .with_bandwidth(1e9);
        let thin = crate::bench::comm::NetworkProfile::fixed_latency(std::time::Duration::ZERO)
            .with_bandwidth(1e6);
        assert_eq!(r.champion_meta_for(high_latency).id, "chatty");
        assert_eq!(r.champion_meta_for(thin).id, "blocky");
    }

    #[test]
    fn first_registered_entry_is_the_initial_champion() {
        let r = Registry::new(meta("a"), Box::new(Fixed(1)) as Box<dyn Dummy>);
        assert_eq!(r.champion().value(), 1);
        assert_eq!(r.champion_meta().id, "a");
    }

    #[test]
    fn set_champion_switches_dispatch_and_reports_unknown_ids() {
        let mut r = Registry::new(meta("a"), Box::new(Fixed(1)) as Box<dyn Dummy>);
        r.register(meta("b"), Box::new(Fixed(2)));
        assert_eq!(r.champion().value(), 1);
        assert!(r.set_champion("b"));
        assert_eq!(r.champion().value(), 2);
        assert!(!r.set_champion("nonexistent"));
        assert_eq!(
            r.champion().value(),
            2,
            "unknown id must not change champion"
        );
    }

    #[test]
    fn get_and_entries_see_every_registered_precompile() {
        let mut r = Registry::new(meta("a"), Box::new(Fixed(1)) as Box<dyn Dummy>);
        r.register(meta("b"), Box::new(Fixed(2)));
        r.register(meta("c"), Box::new(Fixed(3)));
        assert_eq!(r.len(), 3);
        assert_eq!(r.get("c").unwrap().value(), 3);
        assert!(r.get("z").is_none());
        let ids: Vec<&str> = r.entries().map(|(m, _)| m.id).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }
}
