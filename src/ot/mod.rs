//! **Oblivious transfer** — the sender offers two strings, the receiver takes
//! one, and neither learns the other's secret.
//!
//! This is the primitive that pays twice, which is why the research agenda puts
//! it ahead of the other "criptolandia" items:
//!
//! * **Stage 3 of the healthcare workflow** — private set intersection, "which
//!   patients appear at both hospitals", is built on OT.
//! * **MPC preprocessing** — Semi2k's and MASCOT's triples come from OT
//!   (`docs/mpc-compiler-roadmap.md` P1 and P3). The cost parameters in
//!   [`crate::mpcc::protocols`] for those protocols are currently *literature
//!   estimates*; measured OT throughput is what turns them into measurements.
//!
//! # What is here
//!
//! * [`base`] — Chou–Orlandi "simplest OT" over Ristretto255. Two exponentiations
//!   for the receiver, two for the sender, one round each way. Public-key
//!   operations, so a few tens of microseconds per OT: fine for a handful,
//!   hopeless for millions.
//! * [`iknp`] — IKNP OT extension. Turns `κ = 128` base OTs into as many cheap
//!   OTs as you like, at one hash and `κ` bits of communication each. This is
//!   what makes OT usable as a preprocessing primitive.
//!
//! Both are **semi-honest**, matching the threat model the rest of Baru's MPC
//! assumes. That is a real restriction and not a footnote: IKNP as implemented
//! gives a malicious receiver the ability to send inconsistent columns and learn
//! bits of the sender's `s`, which is what the consistency checks in KOS15 exist
//! to prevent. Malicious security is roadmap P2 territory, and the same document
//! that promises it should not pretend this is it.
//!
//! # Shape of the code
//!
//! The protocols are written as explicit message structs with a `bytes()` on
//! each, driven by in-process `simulate_*` helpers. Two reasons: the byte counts
//! reported by [`crate::bench::ot`] are then the real serialized sizes rather
//! than an estimate, and the round structure stays visible instead of being
//! buried in a transport.

use sha2::{Digest, Sha256};

pub mod base;
pub mod iknp;

/// Security parameter: OT-extension seed count and string width, in bits.
pub const KAPPA: usize = 128;
/// `KAPPA` in bytes.
pub const KAPPA_BYTES: usize = KAPPA / 8;

/// A `KAPPA`-bit string — the unit both layers transfer.
pub type Block = [u8; KAPPA_BYTES];

/// Bytes moved in each direction, so a cost claim is a measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct OtTraffic {
    pub sender_to_receiver: usize,
    pub receiver_to_sender: usize,
    /// Round trips, which is what a WAN charges for.
    pub rounds: usize,
}

impl OtTraffic {
    pub fn total(&self) -> usize {
        self.sender_to_receiver + self.receiver_to_sender
    }
}

/// Test-only helper: a uniform block.
#[cfg(test)]
pub(crate) fn random_block(rng: &mut impl rand::Rng) -> Block {
    let mut b = [0u8; KAPPA_BYTES];
    rng.fill_bytes(&mut b);
    b
}

pub(crate) fn xor_block(a: &Block, b: &Block) -> Block {
    let mut out = [0u8; KAPPA_BYTES];
    for i in 0..KAPPA_BYTES {
        out[i] = a[i] ^ b[i];
    }
    out
}

/// Key-derivation / correlation-robust hash, domain-separated by index.
///
/// The index matters: without it the same point or row would derive the same
/// key at every position, and in the extension a repeated row would leak the
/// relation between two transfers.
pub(crate) fn hash_block(domain: &[u8], index: usize, data: &[u8]) -> Block {
    let mut h = Sha256::new();
    h.update(domain);
    h.update((index as u64).to_le_bytes());
    h.update(data);
    let digest = h.finalize();
    let mut out = [0u8; KAPPA_BYTES];
    out.copy_from_slice(&digest[..KAPPA_BYTES]);
    out
}

/// A bit-matrix stored **by column**, which is how both sides of IKNP build it
/// and the transpose the protocol actually needs.
///
/// `rows` columns of `m` bits each; the protocol wants row `i` across all
/// columns. Extraction is a bit-lookup per column rather than a blocked
/// transpose — `O(κ)` per row, so `O(mκ)` overall, the same asymptotics as the
/// cache-efficient Eklundh transpose an optimized implementation would use, with
/// a worse constant. Called out because it is the one place this module trades
/// speed for legibility, and the benchmark is where that shows up.
#[derive(Debug, Clone)]
pub(crate) struct BitColumns {
    cols: Vec<Vec<u8>>,
    m: usize,
}

impl BitColumns {
    pub(crate) fn zeros(cols: usize, m: usize) -> Self {
        BitColumns {
            cols: vec![vec![0u8; m.div_ceil(8)]; cols],
            m,
        }
    }

    pub(crate) fn from_columns(cols: Vec<Vec<u8>>, m: usize) -> Self {
        BitColumns { cols, m }
    }

    pub(crate) fn set(&mut self, j: usize, i: usize, bit: bool) {
        if bit {
            self.cols[j][i / 8] |= 1 << (i % 8);
        } else {
            self.cols[j][i / 8] &= !(1 << (i % 8));
        }
    }

    pub(crate) fn get(&self, j: usize, i: usize) -> bool {
        self.cols[j][i / 8] >> (i % 8) & 1 == 1
    }

    /// Row `i` packed into a `KAPPA`-bit block (one bit per column).
    pub(crate) fn row(&self, i: usize) -> Block {
        let mut out = [0u8; KAPPA_BYTES];
        for j in 0..self.cols.len().min(KAPPA) {
            if self.get(j, i) {
                out[j / 8] |= 1 << (j % 8);
            }
        }
        out
    }

    pub(crate) fn len(&self) -> usize {
        self.m
    }
}

/// Pack `bits` little-endian into bytes.
pub(crate) fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        if *b {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_extraction_is_the_transpose_of_column_writes() {
        let (cols, m) = (KAPPA, 37);
        let mut bc = BitColumns::zeros(cols, m);
        // Set a diagonal-ish pattern and read it back the other way round.
        for i in 0..m {
            for j in 0..cols {
                if (i + j) % 3 == 0 {
                    bc.set(j, i, true);
                }
            }
        }
        for i in 0..m {
            let row = bc.row(i);
            for j in 0..cols {
                let got = row[j / 8] >> (j % 8) & 1 == 1;
                assert_eq!(got, (i + j) % 3 == 0, "row {i} col {j}");
            }
        }
    }

    #[test]
    fn hash_is_separated_by_index_and_domain() {
        let d = [7u8; 32];
        assert_ne!(hash_block(b"a", 0, &d), hash_block(b"a", 1, &d));
        assert_ne!(hash_block(b"a", 0, &d), hash_block(b"b", 0, &d));
    }

    #[test]
    fn xor_is_an_involution() {
        let mut rng = rand::thread_rng();
        let (a, b) = (random_block(&mut rng), random_block(&mut rng));
        assert_eq!(xor_block(&xor_block(&a, &b), &b), a);
    }
}
