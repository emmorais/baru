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
//! reward mechanism — deliberately so. A Precompile is *compiled code*;
//! unlike the FHE/MPC/storage markets in `crate::fhe::lifecycle` /
//! `crate::mpc_market` / `crate::storage` (which verify a *pure data
//! computation's result* — re-execution, co-signature, Merkle proof — fully
//! trustlessly), there is no lightweight way to verify compiled code beyond
//! running it and reading it. Promoting a challenger to champion is a
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

pub mod mpc_matvec;

/// Metadata every Precompile carries, independent of which Gadget trait it
/// implements — what the [`Registry`]/competition needs to identify, credit,
/// and describe an entry.
#[derive(Debug, Clone, Copy)]
pub struct PrecompileMeta {
    /// Stable, unique-per-gadget identifier (e.g. "matvec-naive").
    pub id: &'static str,
    /// Free-form attribution. A future reputation-tied submission system
    /// would replace this with a real `crypto::identity::IdentityId`.
    pub author: &'static str,
    pub description: &'static str,
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
        }
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
