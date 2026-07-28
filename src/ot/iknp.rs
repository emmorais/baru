//! IKNP OT extension — `κ` base OTs in, arbitrarily many cheap OTs out.
//!
//! Base OT ([`super::base`]) costs public-key operations per transfer, so a
//! million of them is out of the question. IKNP pays for `κ = 128` base OTs once
//! and then produces each additional OT for one hash and `κ` bits.
//!
//! # The protocol, and the role reversal that makes it work
//!
//! The extension's *receiver* acts as the base OTs' **sender**, which is the
//! part worth stating plainly because it is where the construction is easy to
//! misread.
//!
//! To produce `m` OTs on `κ`-bit strings, with the extension receiver holding
//! choice bits `r ∈ {0,1}^m`:
//!
//! ```text
//! Receiver  holds both base keys, so for each j < κ it sets
//!                t_j = G(k0_j)  and sends  u_j = G(k0_j) ⊕ G(k1_j) ⊕ r
//! Sender    holds s ∈ {0,1}^κ (its base-OT choices) and only k_{s_j}, so
//!                q_j = G(k_{s_j}) ⊕ (s_j · u_j) = t_j ⊕ (s_j · r)
//! ```
//!
//! Now read the matrices **by row**. Row `i` satisfies `qⁱ = tⁱ ⊕ (r_i · s)`, so
//! the sender can derive both
//!
//! ```text
//! y0_i = H(i, qⁱ)        y1_i = H(i, qⁱ ⊕ s)
//! ```
//!
//! while the receiver computes only `H(i, tⁱ)` — which equals `y0_i` when
//! `r_i = 0` (then `qⁱ = tⁱ`) and `y1_i` when `r_i = 1` (then `qⁱ ⊕ s = tⁱ`).
//! The receiver cannot get the other one without `s`, and the sender cannot
//! learn `r_i` because `tⁱ` is uniform and independent of it.
//!
//! Note that only *one* round of communication carries the `m`-bit columns —
//! `κ` columns of `m` bits, i.e. `κ` bits per produced OT, which is the entire
//! online cost.
//!
//! # Security: semi-honest only, and here is the attack
//!
//! `H` must be correlation-robust: the sender evaluates it at `qⁱ` and
//! `qⁱ ⊕ s` for a single fixed `s` across every row, so a plain
//! collision-resistant hash is not enough in general. SHA-256 with per-row
//! domain separation is the standard practical choice and what [`super::hash_block`]
//! does.
//!
//! Against a **malicious receiver** this is broken, by design of the original
//! paper rather than by an oversight here: nothing forces the receiver to use
//! the same `r` in every column. Sending `(t_j, t_j ⊕ r_j)` with a different
//! `r_j` per column makes the sender's `q` rows depend on `s` in a way the
//! receiver can test, leaking `s` bit by bit. KOS15 adds a correlation check to
//! close this; that is roadmap P2 work, and until it exists this module must not
//! be used to instantiate a maliciously secure protocol.

use rand::Rng;

use super::{
    base, hash_block, pack_bits, xor_block, BitColumns, Block, OtTraffic, KAPPA, KAPPA_BYTES,
};

const DOMAIN: &[u8] = b"baru/ot/iknp/v1";

/// The extension's single online message: **one** `m`-bit column per base OT.
///
/// Sending one column rather than two is the standard IKNP form and is what
/// makes the online cost `κ` bits per produced OT instead of `2κ`. It works
/// because the receiver knows *both* base-OT keys: it can send the single
/// correlation `u_j = G(k0_j) ⊕ G(k1_j) ⊕ r`, and the sender — which knows only
/// `k_{s_j}` — recovers `q_j = t_j ⊕ (s_j · r)` from it either way.
#[derive(Debug, Clone)]
pub struct Columns {
    /// `u_j` per base OT.
    pub cols: Vec<Vec<u8>>,
    pub m: usize,
}

impl Columns {
    pub fn bytes(&self) -> usize {
        self.cols.iter().map(|c| c.len()).sum()
    }
}

/// Expand a base-OT key into an `m`-bit pseudorandom column.
///
/// The base OTs transfer `κ`-bit keys, not `m`-bit columns, so each key is
/// stretched here. That is the standard trick and it is what keeps the base
/// phase independent of `m`.
fn expand(key: &Block, j: usize, m: usize) -> Vec<u8> {
    let bytes = m.div_ceil(8);
    let mut out = Vec::with_capacity(bytes);
    let mut counter = 0usize;
    while out.len() < bytes {
        let block = hash_block(b"baru/ot/iknp/prg", j * 4096 + counter, key);
        out.extend_from_slice(&block);
        counter += 1;
    }
    out.truncate(bytes);
    // Zero the slack bits in the final byte so column XORs stay canonical.
    if !m.is_multiple_of(8) {
        let last = out.len() - 1;
        out[last] &= (1u8 << (m % 8)) - 1;
    }
    out
}

fn xor_into(a: &mut [u8], b: &[u8]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

/// The extension receiver: holds the choice bits, plays base-OT **sender**.
pub struct ExtReceiver {
    t: BitColumns,
    choices: Vec<bool>,
}

impl ExtReceiver {
    /// Build the `κ` column pairs and the message the sender needs.
    ///
    /// `base_keys[j]` is the `(k0, k1)` pair the receiver holds as base-OT
    /// *sender* for base OT `j`.
    /// Takes no randomness: `t` is derived from the base-OT keys, which is
    /// where the extension's entropy already lives.
    pub fn new(choices: &[bool], base_keys: &[(Block, Block)]) -> Result<(Self, Columns), String> {
        if base_keys.len() != KAPPA {
            return Err(format!(
                "IKNP needs {KAPPA} base OTs, got {}",
                base_keys.len()
            ));
        }
        let m = choices.len();
        if m == 0 {
            return Err("IKNP needs at least one output OT".to_string());
        }
        let r = pack_bits(choices);
        let mut t_cols = Vec::with_capacity(KAPPA);
        let mut sent = Vec::with_capacity(KAPPA);
        for (j, (k0, k1)) in base_keys.iter().enumerate() {
            // t_j = G(k0_j), and the one column actually sent is the
            // correlation u_j = G(k0_j) ⊕ G(k1_j) ⊕ r.  Deriving t_j from the
            // key is safe precisely because the sender learns only one of the
            // two keys, so for s_j = 1 it cannot compute t_j at all.
            let t_j = expand(k0, j, m);
            let mut u_j = t_j.clone();
            xor_into(&mut u_j, &expand(k1, j, m));
            xor_into(&mut u_j, &r);
            sent.push(u_j);
            t_cols.push(t_j);
        }
        Ok((
            ExtReceiver {
                t: BitColumns::from_columns(t_cols, m),
                choices: choices.to_vec(),
            },
            Columns { cols: sent, m },
        ))
    }

    /// The `m` output strings — one per OT, the one the choice bit selected.
    pub fn outputs(&self) -> Vec<Block> {
        (0..self.t.len())
            .map(|i| hash_block(DOMAIN, i, &self.t.row(i)))
            .collect()
    }

    pub fn choices(&self) -> &[bool] {
        &self.choices
    }
}

/// The extension sender: holds `s`, plays base-OT **receiver**.
pub struct ExtSender {
    q: BitColumns,
    s: Block,
}

impl ExtSender {
    /// Recover the `q` columns from the padded pairs using the base-OT keys the
    /// sender obtained (`base_out[j]` = the key for choice bit `s_j`).
    pub fn new(s_bits: &[bool], base_out: &[Block], cols: &Columns) -> Result<Self, String> {
        if s_bits.len() != KAPPA || base_out.len() != KAPPA {
            return Err(format!("IKNP needs {KAPPA} base OTs"));
        }
        if cols.cols.len() != KAPPA {
            return Err(format!("expected {KAPPA} columns, got {}", cols.cols.len()));
        }
        let m = cols.m;
        let mut q = BitColumns::zeros(KAPPA, m);
        let mut s = [0u8; KAPPA_BYTES];
        for (j, (&s_j, key)) in s_bits.iter().zip(base_out).enumerate() {
            if s_j {
                s[j / 8] |= 1 << (j % 8);
            }
            // q_j = G(k_{s_j}) ⊕ (s_j · u_j):
            //   s_j = 0 → G(k0_j)                              = t_j
            //   s_j = 1 → G(k1_j) ⊕ G(k0_j) ⊕ G(k1_j) ⊕ r      = t_j ⊕ r
            let mut col = expand(key, j, m);
            if s_j {
                xor_into(&mut col, &cols.cols[j]);
            }
            for i in 0..m {
                if col[i / 8] >> (i % 8) & 1 == 1 {
                    q.set(j, i, true);
                }
            }
        }
        Ok(ExtSender { q, s })
    }

    /// Both output strings per OT.
    pub fn outputs(&self) -> Vec<(Block, Block)> {
        (0..self.q.len())
            .map(|i| {
                let row = self.q.row(i);
                (
                    hash_block(DOMAIN, i, &row),
                    hash_block(DOMAIN, i, &xor_block(&row, &self.s)),
                )
            })
            .collect()
    }
}

/// Result of an end-to-end extension run.
pub struct Extended {
    /// `(y0_i, y1_i)` the sender holds.
    pub sender: Vec<(Block, Block)>,
    /// `y_{r_i}` the receiver holds.
    pub receiver: Vec<Block>,
    pub traffic: OtTraffic,
    /// Bytes the `κ` base OTs cost — amortized away as `m` grows.
    pub base_bytes: usize,
}

/// Run `κ` base OTs and then extend to `choices.len()` OTs, in process.
pub fn simulate(choices: &[bool]) -> Result<Extended, String> {
    let mut rng = rand::thread_rng();

    // Base phase, with the roles reversed: the extension receiver is the
    // base-OT sender, so it is the one that holds both keys.
    let s_bits: Vec<bool> = (0..KAPPA).map(|_| rng.gen::<bool>()).collect();
    let (base_sender, setup) = base::Sender::new(&mut rng);
    let (base_receiver, base_choices) = base::Receiver::new(&setup, &s_bits, &mut rng)?;
    let base_keys = base_sender.keys(&base_choices)?;
    let base_out: Vec<Block> = base_receiver.keys().to_vec();
    let base_bytes = setup.bytes() + base_choices.bytes();

    // Extension phase: one message, κ bits per produced OT.
    let (ext_receiver, cols) = ExtReceiver::new(choices, &base_keys)?;
    let ext_sender = ExtSender::new(&s_bits, &base_out, &cols)?;

    Ok(Extended {
        sender: ext_sender.outputs(),
        receiver: ext_receiver.outputs(),
        traffic: OtTraffic {
            sender_to_receiver: setup.bytes(),
            receiver_to_sender: base_choices.bytes() + cols.bytes(),
            // S, then R, then the columns.
            rounds: 3,
        },
        base_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ot::random_block;

    fn check(m: usize) {
        let mut rng = rand::thread_rng();
        let choices: Vec<bool> = (0..m).map(|_| rng.gen::<bool>()).collect();
        let ext = simulate(&choices).unwrap();
        assert_eq!(ext.sender.len(), m);
        assert_eq!(ext.receiver.len(), m);
        for (i, &c) in choices.iter().enumerate() {
            let (y0, y1) = ext.sender[i];
            let want = if c { y1 } else { y0 };
            assert_eq!(ext.receiver[i], want, "OT {i} (choice {c}) mismatched");
            // The receiver must not hold the other string.
            let other = if c { y0 } else { y1 };
            assert_ne!(ext.receiver[i], other, "OT {i} leaked the unchosen string");
        }
    }

    /// The correctness property, across sizes that exercise the partial final
    /// byte of every column (m not a multiple of 8) as well as aligned ones.
    #[test]
    fn extended_ots_are_correct_at_every_alignment() {
        for m in [1, 7, 8, 9, 63, 64, 65, 128, 200] {
            check(m);
        }
    }

    /// The point of the extension: cost per OT must fall as `m` grows, since
    /// the public-key phase is fixed at `κ` base OTs.
    #[test]
    fn cost_per_ot_falls_with_batch_size() {
        let small = simulate(&[true; 128]).unwrap();
        let large = simulate(&[true; 8192]).unwrap();
        let per = |e: &Extended, m: usize| e.traffic.total() as f64 / m as f64;
        // The floor is the protocol's own online cost, κ bits = 16 bytes per
        // OT; the base phase is what amortizes away on top of that.
        let floor = KAPPA as f64 / 8.0;
        assert!(
            per(&large, 8192) < per(&small, 128) / 2.0,
            "per-OT cost should amortize: {:.1} vs {:.1} bytes",
            per(&large, 8192),
            per(&small, 128)
        );
        assert!(
            per(&large, 8192) < floor * 1.2,
            "large batches should approach the {floor} B/OT online floor, got {:.1}",
            per(&large, 8192)
        );
        // And the base phase is a fixed cost, independent of m.
        assert_eq!(small.base_bytes, large.base_bytes);
    }

    /// The online cost must be the κ bits per OT the protocol promises.
    #[test]
    fn online_cost_is_kappa_bits_per_ot() {
        let m = 4096;
        let mut rng = rand::thread_rng();
        let choices: Vec<bool> = (0..m).map(|_| rng.gen::<bool>()).collect();
        let base_keys: Vec<(Block, Block)> = (0..KAPPA)
            .map(|_| (random_block(&mut rng), random_block(&mut rng)))
            .collect();
        let (_, cols) = ExtReceiver::new(&choices, &base_keys).unwrap();
        // One column per base OT, m bits each ⇒ exactly κ bits per output OT.
        assert_eq!(cols.bytes(), KAPPA * m / 8);
    }

    #[test]
    fn wrong_base_ot_counts_are_refused() {
        let mut rng = rand::thread_rng();
        let few: Vec<(Block, Block)> = (0..4)
            .map(|_| (random_block(&mut rng), random_block(&mut rng)))
            .collect();
        assert!(ExtReceiver::new(&[true], &few).is_err());
        let ok: Vec<(Block, Block)> = (0..KAPPA)
            .map(|_| (random_block(&mut rng), random_block(&mut rng)))
            .collect();
        assert!(ExtReceiver::new(&[], &ok).is_err(), "m = 0");
    }
}
