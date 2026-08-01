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
//! * `base` — Chou–Orlandi "simplest OT" over Ristretto255. Two exponentiations
//!   for the receiver, two for the sender, one round each way. Public-key
//!   operations, so a few tens of microseconds per OT: fine for a handful,
//!   hopeless for millions.
//! * `iknp` — IKNP OT extension. Turns `κ = 128` base OTs into as many cheap
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

/// One batch of 1-out-of-2 OTs.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// What the receiver got, one block per OT.
    pub received: Vec<Block>,
    pub traffic: OtTraffic,
}

/// A 1-out-of-2 oblivious-transfer construction on `κ`-bit messages.
///
/// The trait exists so [`crate::bench::ot`] iterates over constructions instead
/// of naming them, and so a future silent-OT / SoftSpoken implementation appears
/// in the comparison for free. Same pattern as the rest of the tree
/// ([`crate::fhe::FheBackend`], [`crate::db::ResultStore`]).
///
/// # The wrinkle this interface has to be honest about
///
/// [`base`] natively transfers *chosen* messages. [`iknp`] natively produces
/// *random* ones — the parties end up holding correlated strings nobody chose.
/// Presenting both as "chosen-message OT" therefore costs the extension one extra
/// flight: the sender pads its real messages with the random outputs
/// (`e_i = y_i ⊕ m_i`) and ships `2·κ` bits per OT. That derandomization is
/// standard, and its cost is **included** in the reported traffic rather than
/// quietly dropped — which is why the extension's bytes/OT under this trait are
/// higher than the raw `κ`-bits-per-OT figure `bench-ot` reports for the
/// random-OT form. Both numbers are real; they measure different interfaces.
pub trait OtProtocol {
    fn name(&self) -> &'static str;

    /// Whether the construction needs a public-key operation per transfer, which
    /// is what decides whether it can be used at scale.
    fn public_key_per_ot(&self) -> bool;

    /// Deliver `messages[i].{0,1}` according to `choices[i]`.
    fn transfer(&self, choices: &[bool], messages: &[(Block, Block)]) -> Result<Outcome, String>;
}

/// Every construction implemented today.
pub fn all() -> Vec<Box<dyn OtProtocol>> {
    vec![Box::new(base::ChouOrlandi), Box::new(iknp::Iknp)]
}

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

/// Multiply in `GF(2^128)` modulo `x^128 + x^7 + x^2 + x + 1`.
///
/// The KOS15 consistency check ([`iknp::ExtSender::kos_verify`]) is a random
/// linear combination over this field: the receiver proves it used one
/// consistent choice-bit vector across all `κ` columns, and soundness comes
/// from the combination being taken over a *field* rather than bit by bit.
///
/// Bits are little-endian within the block, so bit `i` of the polynomial is bit
/// `i` of the corresponding `u128`. The reduction constant `0x87` is the
/// low-order form of `x^7 + x^2 + x + 1`, the same polynomial AES-GCM uses.
pub(crate) fn gf128_mul(a: &Block, b: &Block) -> Block {
    let x = u128::from_le_bytes(*a);
    let y = u128::from_le_bytes(*b);
    let mut acc: u128 = 0;
    let mut cur = x;
    for i in 0..128 {
        if (y >> i) & 1 == 1 {
            acc ^= cur;
        }
        let overflow = cur >> 127;
        cur <<= 1;
        if overflow == 1 {
            cur ^= 0x87;
        }
    }
    acc.to_le_bytes()
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

    /// The payoff of the trait: one property, asserted against *every*
    /// construction. A new OT implementation is correct-by-this-test or it does
    /// not ship, and the benchmark picks it up without being edited.
    #[test]
    fn every_construction_delivers_the_chosen_message() {
        let mut rng = rand::thread_rng();
        let n = 64;
        let choices: Vec<bool> = (0..n).map(|_| rand::Rng::gen::<bool>(&mut rng)).collect();
        let msgs: Vec<(Block, Block)> = (0..n)
            .map(|_| (random_block(&mut rng), random_block(&mut rng)))
            .collect();

        for proto in all() {
            let out = proto
                .transfer(&choices, &msgs)
                .unwrap_or_else(|e| panic!("{}: {e}", proto.name()));
            assert_eq!(out.received.len(), n, "{}", proto.name());
            for (i, (&c, m)) in choices.iter().zip(&msgs).enumerate() {
                let want = if c { m.1 } else { m.0 };
                assert_eq!(
                    out.received[i],
                    want,
                    "{} returned the wrong message at {i}",
                    proto.name()
                );
            }
            assert!(out.traffic.total() > 0, "{}", proto.name());
        }
    }

    /// And the trait must carry the fact that decides usability at scale.
    #[test]
    fn only_the_base_protocol_needs_public_key_operations_per_ot() {
        let pk: Vec<(&str, bool)> = all()
            .iter()
            .map(|p| (p.name(), p.public_key_per_ot()))
            .collect();
        assert!(pk.contains(&("base", true)));
        assert!(pk.contains(&("extension", false)));
    }

    #[test]
    fn xor_is_an_involution() {
        let mut rng = rand::thread_rng();
        let (a, b) = (random_block(&mut rng), random_block(&mut rng));
        assert_eq!(xor_block(&xor_block(&a, &b), &b), a);
    }
}
