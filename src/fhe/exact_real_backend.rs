//! An **exact real-arithmetic** FHE backend, in plain `f64`.
//!
//! This is the only implementor of [`RealFheBackend`], and therefore the only
//! evaluator of circuits compiled in real-arithmetic mode (`fhe-compile-ckks`,
//! `CircuitCompiler`'s `real_arith` flag). It has no keys, no noise and no
//! ciphertext expansion: a "ciphertext" is the slot vector itself.
//!
//! # Why it outlived CKKS
//!
//! It began as `MockCkksBackend`, the exact twin of a SEAL-backed CKKS backend,
//! used to separate *approximation error* from *wiring error* — if the mock and
//! the real backend disagreed, the bug was in CKKS's noise or rescaling; if they
//! agreed and both were wrong, the bug was in the circuit.
//!
//! The CKKS backend was removed on 2026-08-03 (see `docs/experiments.md` §II.4:
//! approximate comparison via polynomial sign costs seconds per operation with
//! real error, and it took 3.4 MB of vendored C++ plus the entire Windows
//! cmake/MSVC build story with it). This half stayed because it is pure Rust,
//! carries no dependency, and removing it would have deleted the real-arithmetic
//! compiler path's only evaluator — collateral damage rather than simplification.
//!
//! The name is kept so the type is still findable from the history and from the
//! `ckks-mock` benchmark rows that reference it.

use crate::fhe::backend::RealFheBackend;

/// Exact real arithmetic in plain `f64`: no keys, no noise, no expansion.
///
/// Named for the CKKS backend it used to mirror; see the module header.
pub struct MockCkksBackend;

impl RealFheBackend for MockCkksBackend {
    type Ct = Vec<f64>;

    fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        (vec![], vec![])
    }

    fn encrypt(&self, _ck: &[u8], value: f64) -> Vec<f64> {
        vec![value]
    }

    fn encrypt_vec(&self, _ck: &[u8], values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }

    fn decrypt(&self, _ck: &[u8], ct: &Vec<f64>) -> f64 {
        ct.first().copied().unwrap_or(0.0)
    }

    fn decrypt_vec(&self, _ck: &[u8], ct: &Vec<f64>, n: usize) -> Vec<f64> {
        let mut out = ct.clone();
        out.resize(n, 0.0);
        out.truncate(n);
        out
    }

    fn encrypt_public(&self, value: f64) -> Vec<f64> {
        vec![value]
    }

    fn serialize_ct(&self, ct: &Vec<f64>) -> Vec<u8> {
        bincode::serialize(ct).unwrap()
    }

    fn deserialize_ct(&self, bytes: &[u8]) -> Vec<f64> {
        bincode::deserialize(bytes).expect("mock CKKS ct")
    }

    fn backend_id(&self) -> &'static str {
        "ckks-mock"
    }

    fn add(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|i| a.get(i).unwrap_or(&0.0) + b.get(i).unwrap_or(&0.0))
            .collect()
    }

    fn sub(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|i| a.get(i).unwrap_or(&0.0) - b.get(i).unwrap_or(&0.0))
            .collect()
    }

    fn mul(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|i| a.get(i).unwrap_or(&0.0) * b.get(i).unwrap_or(&0.0))
            .collect()
    }

    fn add_plain(&self, a: &Vec<f64>, b: f64) -> Vec<f64> {
        // SEAL's add_plain adds the constant to every slot; mirror that.
        a.iter().map(|x| x + b).collect()
    }

    fn mul_plain(&self, a: &Vec<f64>, b: f64) -> Vec<f64> {
        a.iter().map(|x| x * b).collect()
    }

    fn dot_plain(&self, a: &Vec<f64>, weights: &[f64]) -> Vec<f64> {
        vec![a
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()]
    }
}
