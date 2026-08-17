//! **SPDZ2k** — authenticated additive sharing mod `2^k`, dishonest majority,
//! **malicious** security with abort.
//!
//! The upgrade [`super::semi2k`] documents as missing: every shared value
//! carries an information-theoretic MAC, and every opening is checked against
//! it. A dishonest majority means no honest party can be relied on to recompute
//! anything, so replication — the technique the honest-majority side of Baru
//! uses throughout — is unavailable here. MACs are what replaces it.
//!
//! # The ring, and why `s` extra bits are not optional
//!
//! Over a field, a MAC `m = α·x` is unforgeable because guessing `α` is
//! hopeless. Over `Z_2^k` that argument collapses: the ring has zero divisors,
//! so an adversary adding an error `δ = 2^{k−1}` survives multiplication by half
//! of all `α`.
//!
//! SPDZ2k's answer is to carry values in `Z_2^k` but do all arithmetic — and all
//! MACs — in `Z_2^{k+s}`, with the MAC key `α` drawn from `Z_2^s`. An error that
//! actually changes the result is nonzero mod `2^k`, and such an error survives
//! the check only for a `2^{-s}` fraction of keys.
//!
//! This is the same structural point that lets the honest-majority layer verify
//! multiplications for free ([`crate::crypto::aby3_verify`]): there the 128-bit
//! ring already contained the slack, so no widening was needed. Here the slack
//! must be paid for, because the values genuinely occupy the full `k` bits. The
//! contrast is worth stating in the paper — the same soundness obstacle, priced
//! differently by the threat model.
//!
//! # Scope, stated
//!
//! * **Online phase.** Multiplication consumes *authenticated* Beaver triples.
//!   Producing those maliciously (MASCOT-style, via OT with a sacrifice) is a
//!   separate protocol and is **not** implemented; this engine takes them from a
//!   dealer, exactly as [`crate::mpcc::vm`]'s `DealerEngine` does, and says so.
//!   That makes this the *online* cost of SPDZ2k, which is precisely what the
//!   registry row's online parameters describe.
//! * **Abort is deferred.** [`Spdz2kEngine::mac_check`] takes one verdict over
//!   every value opened so far. Checking per-opening would leak: whether a check
//!   fails can depend on secret data, so the abort must not be observable per
//!   operation. Same discipline as the honest-majority layer.

use rand::Rng;

use super::vm::{Counters, Modulus, ShareEngine};

/// One party's view of an authenticated sharing.
///
/// The simulation holds every party's view at once, so each field is a vector
/// indexed by party. `value` sums to `x` and `mac` sums to `α·x`, both mod
/// `2^{k+s}`.
#[derive(Clone, Debug)]
pub struct AuthShare {
    pub value: Vec<u128>,
    pub mac: Vec<u128>,
}

/// An opened value, retained until the deferred MAC check.
#[derive(Clone, Debug)]
struct Opened {
    public: u128,
    mac: Vec<u128>,
}

pub struct Spdz2kEngine {
    /// Value bits. Semantically meaningful results live mod `2^k`.
    k: u32,
    /// Statistical security bits. Soundness of the MAC check is `~2^-s`.
    s: u32,
    parties: usize,
    counters: Counters,
    /// Additive shares of the MAC key `α ∈ Z_2^s`.
    alpha: Vec<u128>,
    /// Every value opened since the last check.
    opened: Vec<Opened>,
    /// Authenticated Beaver triples, dealer-supplied — see the module header.
    pool: Vec<(AuthShare, AuthShare, AuthShare)>,
}

/// The MAC check rejected: some opened value did not carry a valid MAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacCheckFailed;

impl std::fmt::Display for MacCheckFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SPDZ2k MAC check failed: an opened value was not authentic")
    }
}

impl std::error::Error for MacCheckFailed {}

impl Spdz2kEngine {
    /// `k` value bits, `s` statistical security bits. Arithmetic runs mod
    /// `2^{k+s}`, which must fit a `u128`.
    pub fn new(parties: usize, k: u32, s: u32) -> Result<Self, String> {
        if parties < 2 {
            return Err(format!("spdz2k needs at least 2 parties, got {parties}"));
        }
        if k + s > 127 {
            return Err(format!(
                "spdz2k: k + s = {} exceeds the 127-bit working ring",
                k + s
            ));
        }
        if s < 16 {
            return Err(format!(
                "spdz2k: s = {s} gives only 2^-{s} soundness; refusing to pretend that is security"
            ));
        }
        let mut rng = rand::thread_rng();
        // α ∈ Z_2^s, additively shared.
        let alpha_val = rng.gen::<u128>() & ((1u128 << s) - 1);
        let mut alpha = vec![0u128; parties];
        let mut acc = 0u128;
        for a in alpha.iter_mut().take(parties - 1) {
            *a = rng.gen::<u128>();
            acc = acc.wrapping_add(*a);
        }
        alpha[parties - 1] = alpha_val.wrapping_sub(acc);
        Ok(Spdz2kEngine {
            k,
            s,
            parties,
            counters: Counters::default(),
            alpha,
            opened: Vec::new(),
            pool: Vec::new(),
        })
    }

    /// Working-ring mask, `2^{k+s} − 1`.
    fn wmask(&self) -> u128 {
        if self.k + self.s >= 128 {
            u128::MAX
        } else {
            (1u128 << (self.k + self.s)) - 1
        }
    }

    /// Value mask, `2^k − 1` — what a result actually means.
    fn vmask(&self) -> u128 {
        if self.k >= 128 {
            u128::MAX
        } else {
            (1u128 << self.k) - 1
        }
    }

    fn alpha_value(&self) -> u128 {
        self.alpha.iter().fold(0u128, |a, x| a.wrapping_add(*x)) & self.wmask()
    }

    /// Share a value with its MAC. Used for inputs and for dealt triples.
    fn authenticate(&self, value: u128) -> AuthShare {
        let mut rng = rand::thread_rng();
        let w = self.wmask();
        let v = value & w;
        let mut value_shares = vec![0u128; self.parties];
        let mut acc = 0u128;
        for sh in value_shares.iter_mut().take(self.parties - 1) {
            *sh = rng.gen::<u128>() & w;
            acc = acc.wrapping_add(*sh) & w;
        }
        value_shares[self.parties - 1] = v.wrapping_sub(acc) & w;

        let mac_total = self.alpha_value().wrapping_mul(v) & w;
        let mut mac_shares = vec![0u128; self.parties];
        let mut macc = 0u128;
        for sh in mac_shares.iter_mut().take(self.parties - 1) {
            *sh = rng.gen::<u128>() & w;
            macc = macc.wrapping_add(*sh) & w;
        }
        mac_shares[self.parties - 1] = mac_total.wrapping_sub(macc) & w;

        AuthShare {
            value: value_shares,
            mac: mac_shares,
        }
    }

    fn open_value(&self, a: &AuthShare) -> u128 {
        a.value.iter().fold(0u128, |x, y| x.wrapping_add(*y)) & self.wmask()
    }

    /// Deal `n` authenticated Beaver triples.
    ///
    /// Dealer-supplied, and that is the honest description: generating these
    /// against a malicious adversary is MASCOT's job and is not implemented
    /// here. The online phase below is maliciously secure *given* correct
    /// triples.
    pub fn deal_triples(&mut self, n: usize) {
        let mut rng = rand::thread_rng();
        let w = self.wmask();
        for _ in 0..n {
            let a = rng.gen::<u128>() & w;
            let b = rng.gen::<u128>() & w;
            let c = a.wrapping_mul(b) & w;
            let (sa, sb, sc) = (
                self.authenticate(a),
                self.authenticate(b),
                self.authenticate(c),
            );
            self.pool.push((sa, sb, sc));
        }
        self.counters.triples_generated += n as u64;
    }

    fn triple(&mut self) -> (AuthShare, AuthShare, AuthShare) {
        if self.pool.is_empty() {
            self.deal_triples(64);
        }
        self.counters.triples += 1;
        self.pool.pop().expect("pool refilled")
    }

    /// Record an opening for the deferred check.
    fn note_opening(&mut self, public: u128, mac: Vec<u128>) {
        self.opened.push(Opened { public, mac });
    }

    /// How many openings the pending check covers.
    pub fn pending_openings(&self) -> usize {
        self.opened.len()
    }

    /// The deferred verdict over every value opened since the last call.
    ///
    /// For random public `χ_j`, each party computes
    /// `σᵢ = Σⱼ χⱼ·(macⱼ⁽ⁱ⁾ − αᵢ·xⱼ)` and the parties check `Σᵢ σᵢ = 0`. In a
    /// real deployment the `σᵢ` are committed before being opened, so no party
    /// can adapt its share after seeing the others'; the in-process simulation
    /// holds every view at once, so that step has nothing to simulate.
    ///
    /// An adversary that opened a wrong value must have introduced an error
    /// nonzero mod `2^k`; such an error passes only for a `2^-s` fraction of
    /// MAC keys, and `α` is never revealed.
    pub fn mac_check(&mut self) -> Result<(), MacCheckFailed> {
        if self.opened.is_empty() {
            return Ok(());
        }
        let w = self.wmask();
        let mut rng = rand::thread_rng();
        let chi: Vec<u128> = (0..self.opened.len())
            .map(|_| rng.gen::<u128>() & w)
            .collect();

        let mut total = 0u128;
        for i in 0..self.parties {
            let mut sigma = 0u128;
            for (j, op) in self.opened.iter().enumerate() {
                let term = op.mac[i].wrapping_sub(self.alpha[i].wrapping_mul(op.public)) & w;
                sigma = sigma.wrapping_add(chi[j].wrapping_mul(term)) & w;
            }
            total = total.wrapping_add(sigma) & w;
        }
        self.opened.clear();
        if total == 0 {
            Ok(())
        } else {
            Err(MacCheckFailed)
        }
    }

    /// Corrupt one party's share of an opened value, for testing that the check
    /// actually catches deviation.
    #[cfg(test)]
    fn tamper_last_opening(&mut self, delta: u128) {
        if let Some(op) = self.opened.last_mut() {
            op.public = op.public.wrapping_add(delta);
        }
    }
}

impl ShareEngine for Spdz2kEngine {
    type Share = AuthShare;

    fn modulus(&self) -> Modulus {
        Modulus::Ring { k: self.k }
    }
    fn parties(&self) -> usize {
        self.parties
    }
    fn counters(&self) -> &Counters {
        &self.counters
    }
    fn counters_mut(&mut self) -> &mut Counters {
        &mut self.counters
    }

    fn input(&mut self, value: u64, _party: usize) -> AuthShare {
        // Authenticated shares are twice the size of plain ones, which is the
        // headline online overhead of the SPDZ family.
        self.counters.online_bytes += 2 * (self.k as u64 + self.s as u64) / 8;
        self.authenticate(value as u128)
    }

    fn constant(&mut self, value: u64) -> AuthShare {
        let w = self.wmask();
        let v = (value as u128) & w;
        // A public constant is held by party 0; every party's MAC share is
        // `αᵢ·c`, which needs no communication.
        let mut value_shares = vec![0u128; self.parties];
        value_shares[0] = v;
        let mac = (0..self.parties)
            .map(|i| self.alpha[i].wrapping_mul(v) & w)
            .collect();
        AuthShare {
            value: value_shares,
            mac,
        }
    }

    fn add(&mut self, a: &AuthShare, b: &AuthShare) -> AuthShare {
        let w = self.wmask();
        AuthShare {
            value: a
                .value
                .iter()
                .zip(&b.value)
                .map(|(x, y)| x.wrapping_add(*y) & w)
                .collect(),
            mac: a
                .mac
                .iter()
                .zip(&b.mac)
                .map(|(x, y)| x.wrapping_add(*y) & w)
                .collect(),
        }
    }

    fn sub(&mut self, a: &AuthShare, b: &AuthShare) -> AuthShare {
        let w = self.wmask();
        AuthShare {
            value: a
                .value
                .iter()
                .zip(&b.value)
                .map(|(x, y)| x.wrapping_sub(*y) & w)
                .collect(),
            mac: a
                .mac
                .iter()
                .zip(&b.mac)
                .map(|(x, y)| x.wrapping_sub(*y) & w)
                .collect(),
        }
    }

    fn add_const(&mut self, a: &AuthShare, c: u64) -> AuthShare {
        let w = self.wmask();
        let cv = (c as u128) & w;
        let mut out = a.clone();
        out.value[0] = out.value[0].wrapping_add(cv) & w;
        for i in 0..self.parties {
            out.mac[i] = out.mac[i].wrapping_add(self.alpha[i].wrapping_mul(cv)) & w;
        }
        out
    }

    fn mul_const(&mut self, a: &AuthShare, c: u64) -> AuthShare {
        let w = self.wmask();
        let cv = (c as u128) & w;
        AuthShare {
            value: a.value.iter().map(|x| x.wrapping_mul(cv) & w).collect(),
            mac: a.mac.iter().map(|x| x.wrapping_mul(cv) & w).collect(),
        }
    }

    /// Beaver multiplication on authenticated shares.
    ///
    /// Opens `d = x − a` and `e = y − b`, both of which are masked by fresh
    /// triple randomness and therefore safe to publish — but both are *recorded*
    /// for the MAC check, because a party that lies about `d` or `e` shifts the
    /// product and nothing else would notice.
    fn mul_batch(&mut self, pairs: &[(AuthShare, AuthShare)]) -> Vec<AuthShare> {
        let w = self.wmask();
        let mut out = Vec::with_capacity(pairs.len());
        for (x, y) in pairs {
            let (a, b, c) = self.triple();
            let d_share = self.sub(x, &a);
            let e_share = self.sub(y, &b);
            let d = self.open_value(&d_share);
            let e = self.open_value(&e_share);
            self.note_opening(d, d_share.mac.clone());
            self.note_opening(e, e_share.mac.clone());
            self.counters.online_bytes += 2 * 2 * (self.k as u64 + self.s as u64) / 8;

            // z = c + d·b + e·a + d·e
            let db = self.mul_const(&b, (d & u64::MAX as u128) as u64);
            let ea = self.mul_const(&a, (e & u64::MAX as u128) as u64);
            let mut z = self.add(&c, &db);
            z = self.add(&z, &ea);
            let de = d.wrapping_mul(e) & w;
            // Adding the public `d·e` keeps the MAC consistent because
            // `add_const` folds `αᵢ·de` into every party's MAC share.
            let mut zz = z.clone();
            zz.value[0] = zz.value[0].wrapping_add(de) & w;
            for i in 0..self.parties {
                zz.mac[i] = zz.mac[i].wrapping_add(self.alpha[i].wrapping_mul(de)) & w;
            }
            out.push(zz);
        }
        out
    }

    /// Open values, recording each for the deferred MAC check.
    ///
    /// The returned values are **not trustworthy until [`Spdz2kEngine::mac_check`]
    /// returns `Ok`** — an unverified opening is exactly what an adversary would
    /// like acted upon.
    fn open_batch(&mut self, values: &[AuthShare]) -> Vec<u64> {
        let vm = self.vmask();
        let mut out = Vec::with_capacity(values.len());
        for sh in values {
            let public = self.open_value(sh);
            self.note_opening(public, sh.mac.clone());
            self.counters.online_bytes += (self.k as u64 + self.s as u64) / 8;
            out.push((public & vm) as u64);
        }
        out
    }

    /// Dealt locally, and **this is a disclosed gap, not a definition** — unlike
    /// `DealerEngine`, this engine claims a threat model that a dealer violates.
    ///
    /// It deliberately does *not* delegate to `vm::random_bits_batch`. That
    /// construction XORs one party-supplied bit per party, and a **malicious**
    /// party can supply a value that is not a bit: the result is then not a
    /// sharing of 0 or 1, `bit_decompose_batch` returns garbage, and no MAC
    /// detects it because the MACs are computed over the corrupted value and the
    /// run stays internally consistent. Authenticating a non-bit is not the same
    /// as proving it is a bit.
    ///
    /// Closing it costs one batched `b·(b−1)` multiplication plus an opening
    /// checked against zero, which this engine's deferred [`Self::mac_check`]
    /// verdict is already the right place to host. Until then the bits here are
    /// dealt, so **this engine's comparison costs are a lower bound** and are not
    /// commensurable with Semi2k's or LowGear's measured ones. `RUNNABLE` does
    /// not include `spdz2k`, so no sweep quotes them.
    fn random_bits(&mut self, n: usize) -> Vec<AuthShare> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| {
                let bit = rng.gen::<bool>() as u128;
                self.authenticate(bit)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Spdz2kEngine {
        Spdz2kEngine::new(2, 32, 32).expect("valid parameters")
    }

    #[test]
    fn refuses_parameters_that_would_only_look_secure() {
        assert!(
            Spdz2kEngine::new(2, 32, 8).is_err(),
            "s = 8 is not security"
        );
        assert!(
            Spdz2kEngine::new(2, 100, 64).is_err(),
            "k + s must fit the ring"
        );
        assert!(
            Spdz2kEngine::new(1, 32, 32).is_err(),
            "needs at least two parties"
        );
    }

    #[test]
    fn linear_operations_preserve_the_mac() {
        let mut e = engine();
        let x = e.input(7, 0);
        let y = e.input(5, 1);
        let sum = e.add(&x, &y);
        let scaled = e.mul_const(&sum, 3);
        let shifted = e.add_const(&scaled, 4);
        let got = e.open_batch(&[shifted]);
        assert_eq!(got[0], (7 + 5) * 3 + 4);
        assert!(e.mac_check().is_ok(), "linear ops must keep MACs valid");
    }

    #[test]
    fn multiplication_is_correct_and_authenticated() {
        let mut e = engine();
        e.deal_triples(8);
        let x = e.input(11, 0);
        let y = e.input(13, 1);
        let z = e.mul_batch(&[(x, y)]).pop().expect("one product");
        let got = e.open_batch(&[z]);
        assert_eq!(got[0], 11 * 13);
        assert!(
            e.mac_check().is_ok(),
            "an honest run must pass the MAC check"
        );
    }

    /// The point of the whole engine: a value opened as something other than
    /// what it is must be rejected.
    #[test]
    fn a_forged_opening_is_rejected() {
        let mut e = engine();
        let x = e.input(42, 0);
        let _ = e.open_batch(&[x]);
        e.tamper_last_opening(1);
        assert_eq!(
            e.mac_check(),
            Err(MacCheckFailed),
            "a forged opening must fail the MAC check"
        );
    }

    /// An error that is zero mod `2^k` changes no meaningful value. The check is
    /// not claimed to catch it, and the `s` extra bits are exactly what makes
    /// errors that *do* matter detectable.
    #[test]
    fn the_check_is_about_errors_that_change_the_result() {
        let mut e = engine();
        let x = e.input(42, 0);
        let opened = e.open_batch(&[x]);
        assert_eq!(opened[0], 42);
        // A deviation of one in the low bits is meaningful and caught.
        e.tamper_last_opening(1);
        assert!(e.mac_check().is_err());
    }

    #[test]
    fn the_verdict_is_deferred_across_many_openings() {
        let mut e = engine();
        e.deal_triples(16);
        let mut acc = e.input(1, 0);
        for i in 2..6u64 {
            let v = e.input(i, 1);
            acc = e.mul_batch(&[(acc, v)]).pop().expect("product");
        }
        let got = e.open_batch(&[acc]);
        assert_eq!(got[0], 120, "1*2*3*4*5");
        // Each multiplication opened two values, plus the final opening.
        assert!(
            e.pending_openings() >= 9,
            "openings must accumulate for a single deferred verdict, got {}",
            e.pending_openings()
        );
        assert!(e.mac_check().is_ok());
        assert_eq!(e.pending_openings(), 0, "a check consumes what it covered");
    }

    #[test]
    fn a_forgery_anywhere_in_a_long_run_is_still_caught() {
        let mut e = engine();
        e.deal_triples(16);
        let x = e.input(3, 0);
        let y = e.input(4, 1);
        let z = e.mul_batch(&[(x, y)]).pop().expect("product");
        e.tamper_last_opening(1 << 20);
        let _ = e.open_batch(&[z]);
        assert_eq!(e.mac_check(), Err(MacCheckFailed));
    }
}
