//! `MatVec` gadget for MPC: `out_dim` independent dot products of a
//! secret-shared `in_dim`-vector against a secret-shared `out_dim × in_dim`
//! weight matrix — the core linear-layer operation of private NN inference.
//!
//! Two competing precompiles, both checked against a plaintext reference in
//! `src/bench/nn.rs`'s tests:
//!   - [`NaiveMatVec`] — the straightforward per-element loop (what Baru's
//!     `matvec` builtin and `bench-nn` both did before this module existed).
//!     `in_dim * out_dim` sequential `Backend::mul` calls: one communication
//!     round *per multiplication*.
//!   - [`BatchedMatVec`] — flattens every multiplication in the whole matvec
//!     into a single [`Backend::mul_batch`] call: one communication round
//!     for the *entire* matvec, independent of shape. This is the champion
//!     (see `default_registry`) — confirmed faster and lower-message-count
//!     by `bench-nn`'s precompile competition mode.

use std::sync::OnceLock;

use crate::interpreter::backend::{Backend, ShareRef};
use crate::precompile::{PrecompileMeta, Registry};

/// The MatVec-for-MPC gadget interface. Any correct implementation of
/// `w · x` against a live [`Backend`] satisfies it, regardless of internal
/// strategy — that's what lets [`NaiveMatVec`] and [`BatchedMatVec`] compete
/// on equal footing.
pub trait MpcMatVec: Send + Sync {
    /// `w` is `out_dim` rows of `in_dim` shares each; `x` is `in_dim` shares.
    /// Returns `out_dim` shares, `w · x`.
    fn matvec(&self, be: &mut dyn Backend, w: &[Vec<ShareRef>], x: &[ShareRef]) -> Vec<ShareRef>;
}

/// Reference baseline: one `Backend::mul` + `Backend::add` per element,
/// row by row. Always correct (uses only the universally-available `mul`),
/// but costs `in_dim * out_dim` communication rounds.
pub struct NaiveMatVec;

impl MpcMatVec for NaiveMatVec {
    fn matvec(&self, be: &mut dyn Backend, w: &[Vec<ShareRef>], x: &[ShareRef]) -> Vec<ShareRef> {
        w.iter()
            .map(|row| {
                let mut acc = be.mul(&row[0], &x[0]);
                for j in 1..row.len() {
                    let term = be.mul(&row[j], &x[j]);
                    acc = be.add(&acc, &term);
                }
                acc
            })
            .collect()
    }
}

/// Flattens every `(w[i][j], x[j])` pair across the *whole* matrix into one
/// [`Backend::mul_batch`] call, then sums each row's `in_dim` products
/// locally (additions are already free — no communication). One
/// communication round for the entire matvec, independent of `out_dim`.
pub struct BatchedMatVec;

impl MpcMatVec for BatchedMatVec {
    fn matvec(&self, be: &mut dyn Backend, w: &[Vec<ShareRef>], x: &[ShareRef]) -> Vec<ShareRef> {
        let in_dim = x.len();
        let pairs: Vec<(ShareRef, ShareRef)> = w
            .iter()
            .flat_map(|row| row.iter().cloned().zip(x.iter().cloned()))
            .collect();
        let products = be.mul_batch(&pairs);
        products
            .chunks(in_dim)
            .map(|chunk| {
                let mut acc = chunk[0].clone();
                for p in &chunk[1..] {
                    acc = be.add(&acc, p);
                }
                acc
            })
            .collect()
    }
}

fn build_default_registry() -> Registry<dyn MpcMatVec> {
    let mut r = Registry::new(
        PrecompileMeta {
            id: "matvec-naive",
            author: "baru-core",
            description: "per-element mul+add loop; O(in_dim*out_dim) MPC rounds",
        },
        Box::new(NaiveMatVec) as Box<dyn MpcMatVec>,
    );
    r.register(
        PrecompileMeta {
            id: "matvec-batched",
            author: "baru-core",
            description: "single mul_batch call for the whole matrix; O(1) MPC rounds",
        },
        Box::new(BatchedMatVec),
    );
    // Benchmarked winner (see `bench-nn --competition matvec`): fewer rounds,
    // fewer messages, strictly faster at every shape tested. Champions are
    // set here deliberately, never auto-selected from a live run.
    assert!(r.set_champion("matvec-batched"));
    r
}

static REGISTRY: OnceLock<Registry<dyn MpcMatVec>> = OnceLock::new();

/// The shared MatVec-for-MPC precompile registry Baru's `matvec` builtin
/// dispatches through.
pub fn registry() -> &'static Registry<dyn MpcMatVec> {
    REGISTRY.get_or_init(build_default_registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;
    use crate::interpreter::transport_backend::TransportBackend;

    fn signed_to_fr(n: i64, nbits: usize) -> u128 {
        (n as i128).rem_euclid(1i128 << nbits) as u128
    }
    fn fr_to_signed(raw: u128, nbits: usize) -> i64 {
        let modulus = 1u128 << nbits;
        let v = raw % modulus;
        let half = modulus / 2;
        if v >= half {
            (v as i128 - modulus as i128) as i64
        } else {
            v as i64
        }
    }

    /// Run `precompile` against a real 3-party `TransportBackend` for
    /// `w · x` (party 0 owns the data) and return party 0's revealed result.
    /// `thread::scope` lets us borrow `precompile` across the party threads
    /// without requiring a `'static` reference (the champion/registry tests
    /// do have one, but the naive/batched direct-comparison test doesn't).
    fn run_matvec(precompile: &(dyn MpcMatVec + Sync), w: Vec<Vec<i64>>, x: Vec<i64>) -> Vec<i64> {
        const NBITS: usize = 32;
        let handles = InprocHandle::create(3);
        std::thread::scope(|scope| {
            let joins: Vec<_> = handles
                .into_iter()
                .enumerate()
                .map(|(me, h)| {
                    let w = &w;
                    let x = &x;
                    scope.spawn(move || {
                        let mut be = TransportBackend::new(me, h);
                        let share_row = |be: &mut TransportBackend<InprocHandle>, row: &[i64]| {
                            row.iter()
                                .map(|&v| {
                                    let fr = if me == 0 { signed_to_fr(v, NBITS) } else { 0 };
                                    be.share_secret(fr, NBITS, 0)
                                })
                                .collect::<Vec<_>>()
                        };
                        let x_shares = share_row(&mut be, x);
                        let w_shares: Vec<Vec<_>> =
                            w.iter().map(|row| share_row(&mut be, row)).collect();
                        let out = precompile.matvec(&mut be, &w_shares, &x_shares);
                        out.into_iter()
                            .map(|sr| fr_to_signed(be.reveal(&sr, None), NBITS))
                            .collect::<Vec<i64>>()
                    })
                })
                .collect();
            joins.into_iter().map(|j| j.join().unwrap()).next().unwrap()
        })
    }

    fn plain_matvec(w: &[Vec<i64>], x: &[i64]) -> Vec<i64> {
        w.iter()
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn naive_and_batched_agree_with_plaintext() {
        let w = vec![vec![1, -2, 3], vec![4, 5, -6], vec![0, 7, 8]];
        let x = vec![9, -1, 2];
        let expected = plain_matvec(&w, &x);

        assert_eq!(run_matvec(&NaiveMatVec, w.clone(), x.clone()), expected);
        assert_eq!(run_matvec(&BatchedMatVec, w, x), expected);
    }

    #[test]
    fn registry_champion_is_batched_and_both_are_registered() {
        let reg = registry();
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.champion_meta().id, "matvec-batched");
        assert!(reg.get("matvec-naive").is_some());
        assert!(reg.get("matvec-batched").is_some());
    }

    #[test]
    fn champion_matches_plaintext_via_registry_lookup() {
        let w = vec![vec![2, 3], vec![-4, 5]];
        let x = vec![6, -7];
        let expected = plain_matvec(&w, &x);
        let champion = registry().champion();
        assert_eq!(run_matvec(champion, w, x), expected);
    }
}
