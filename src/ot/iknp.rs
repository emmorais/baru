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
    base, gf128_mul, hash_block, pack_bits, xor_block, BitColumns, Block, OtTraffic, KAPPA,
    KAPPA_BYTES,
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

// ─── KOS15 consistency check ─────────────────────────────────────────────────

/// Extra random OTs appended before the check.
///
/// The check opens a random linear combination of the receiver's choice bits.
/// Without padding, that combination would be a nontrivial function of the
/// *real* choices and would leak. Appending `KAPPA` rows with uniformly random
/// choice bits makes the opened value uniform and independent of the real ones,
/// at a cost that vanishes as `m` grows.
pub const KOS_PAD: usize = KAPPA;

/// The receiver's answer to the sender's challenge.
///
/// `x = Σ χᵢ·rᵢ` and `t = Σ χᵢ·tᵢ`, both in `GF(2^128)` — two blocks total,
/// whatever `m` is, so the check is O(1) in communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KosResponse {
    pub x: Block,
    pub t: Block,
}

impl KosResponse {
    pub fn bytes(&self) -> usize {
        2 * KAPPA_BYTES
    }
}

/// Expand the sender's challenge seed into per-row coefficients.
///
/// The seed is chosen by the **sender**, and only after the columns have
/// arrived. That ordering is the whole point: a receiver that could predict the
/// coefficients could choose inconsistent columns whose inconsistency the
/// combination cancels.
fn challenge(seed: &Block, i: usize) -> Block {
    hash_block(b"baru/ot/kos/chi", i, seed)
}

impl ExtReceiver {
    /// Answer the sender's challenge.
    pub fn kos_response(&self, seed: &Block) -> KosResponse {
        let mut x = [0u8; KAPPA_BYTES];
        let mut t = [0u8; KAPPA_BYTES];
        for i in 0..self.t.len() {
            let chi = challenge(seed, i);
            // rᵢ ∈ {0,1}, so the scalar product is an XOR when the bit is set.
            if self.choices[i] {
                x = xor_block(&x, &chi);
            }
            t = xor_block(&t, &gf128_mul(&chi, &self.t.row(i)));
        }
        KosResponse { x, t }
    }
}

impl ExtSender {
    /// Verify the receiver used one consistent choice vector across all columns.
    ///
    /// Because `qᵢ = tᵢ ⊕ (rᵢ·s)` for an honest receiver, summing against the
    /// challenge gives `Σχᵢqᵢ = Σχᵢtᵢ ⊕ (Σχᵢrᵢ)·s = t ⊕ x·s`. A receiver that
    /// used a different `r` in different columns cannot satisfy this except
    /// with probability `2^-128` over the sender's choice of challenge.
    ///
    /// This is what closes the attack the module header describes: without it a
    /// malicious receiver recovers the sender's `s` bit by bit, which breaks
    /// every OT the extension produces.
    pub fn kos_verify(&self, seed: &Block, resp: &KosResponse) -> bool {
        let mut q = [0u8; KAPPA_BYTES];
        for i in 0..self.q.len() {
            let chi = challenge(seed, i);
            q = xor_block(&q, &gf128_mul(&chi, &self.q.row(i)));
        }
        let expected = xor_block(&resp.t, &gf128_mul(&resp.x, &self.s));
        q == expected
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

/// Run the extension **with the KOS15 consistency check**, in process.
///
/// Differs from [`simulate`] in three ways, all of them required for the check
/// to mean anything:
///
/// 1. `KOS_PAD` extra OTs with random choice bits are appended, so the opened
///    linear combination is independent of the real choices;
/// 2. the sender draws its challenge **after** the columns arrive;
/// 3. the outputs are truncated back to `choices.len()`, so the padding is
///    invisible to callers.
///
/// Returns `Err` if the check fails — that is a detected cheating receiver, not
/// a protocol error.
pub fn simulate_checked(choices: &[bool]) -> Result<Extended, String> {
    let mut rng = rand::thread_rng();
    let m = choices.len();
    if m == 0 {
        return Err("IKNP needs at least one output OT".to_string());
    }

    let mut padded = choices.to_vec();
    padded.extend((0..KOS_PAD).map(|_| rng.gen::<bool>()));

    let s_bits: Vec<bool> = (0..KAPPA).map(|_| rng.gen::<bool>()).collect();
    let (base_sender, setup) = base::Sender::new(&mut rng);
    let (base_receiver, base_choices) = base::Receiver::new(&setup, &s_bits, &mut rng)?;
    let base_keys = base_sender.keys(&base_choices)?;
    let base_out: Vec<Block> = base_receiver.keys().to_vec();
    let base_bytes = setup.bytes() + base_choices.bytes();

    let (ext_receiver, cols) = ExtReceiver::new(&padded, &base_keys)?;
    let ext_sender = ExtSender::new(&s_bits, &base_out, &cols)?;

    // The challenge is sampled here — after the columns exist — on purpose.
    let mut seed = [0u8; KAPPA_BYTES];
    rng.fill(&mut seed);
    let resp = ext_receiver.kos_response(&seed);
    if !ext_sender.kos_verify(&seed, &resp) {
        return Err("KOS consistency check failed: receiver used inconsistent columns".to_string());
    }

    let mut sender_out = ext_sender.outputs();
    let mut receiver_out = ext_receiver.outputs();
    sender_out.truncate(m);
    receiver_out.truncate(m);

    Ok(Extended {
        sender: sender_out,
        receiver: receiver_out,
        traffic: OtTraffic {
            sender_to_receiver: setup.bytes() + KAPPA_BYTES, // + challenge seed
            receiver_to_sender: base_choices.bytes() + cols.bytes() + resp.bytes(),
            // S, R, columns, challenge, response.
            rounds: 5,
        },
        base_bytes,
    })
}

#[cfg(test)]
mod kos_tests {
    use super::*;

    fn gf(b: u128) -> Block {
        b.to_le_bytes()
    }

    #[test]
    fn gf128_is_a_field_on_the_cases_the_check_relies_on() {
        let one = gf(1);
        let a = gf(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
        assert_eq!(gf128_mul(&a, &one), a, "1 must be the identity");
        assert_eq!(gf128_mul(&one, &a), a, "multiplication must commute with 1");
        assert_eq!(gf128_mul(&a, &gf(0)), gf(0), "0 must annihilate");

        // Distributivity over XOR is exactly what the linear-combination
        // argument in `kos_verify` depends on.
        let b = gf(0xdead_beef_dead_beef_0000_0000_0000_0001);
        let c = gf(0x9999_8888_7777_6666_5555_4444_3333_2222);
        let lhs = gf128_mul(&a, &xor_block(&b, &c));
        let rhs = xor_block(&gf128_mul(&a, &b), &gf128_mul(&a, &c));
        assert_eq!(lhs, rhs, "multiplication must distribute over XOR");
    }

    #[test]
    fn checked_extension_agrees_with_the_unchecked_one() {
        let choices: Vec<bool> = (0..64).map(|i| i % 3 == 0).collect();
        let out = simulate_checked(&choices).expect("honest run must pass the check");
        assert_eq!(out.receiver.len(), choices.len(), "padding must be hidden");
        assert_eq!(out.sender.len(), choices.len());
        for (i, &c) in choices.iter().enumerate() {
            let expected = if c { out.sender[i].1 } else { out.sender[i].0 };
            assert_eq!(out.receiver[i], expected, "OT {i} delivered the wrong string");
        }
    }

    /// Run one extension with `corrupt_cols` columns tampered with, and report
    /// whether the sender's check rejected.
    fn cheat_is_caught(corrupt_cols: &[usize]) -> bool {
        let mut rng = rand::thread_rng();
        let choices: Vec<bool> = (0..32).map(|_| rng.gen()).collect();

        let s_bits: Vec<bool> = (0..KAPPA).map(|_| rng.gen::<bool>()).collect();
        let (base_sender, setup) = base::Sender::new(&mut rng);
        let (base_receiver, base_choices) =
            base::Receiver::new(&setup, &s_bits, &mut rng).expect("base ot");
        let base_keys = base_sender.keys(&base_choices).expect("base keys");
        let base_out: Vec<Block> = base_receiver.keys().to_vec();

        let (ext_receiver, mut cols) = ExtReceiver::new(&choices, &base_keys).expect("extend");
        for &j in corrupt_cols {
            cols.cols[j][0] ^= 0xff;
        }
        let ext_sender = ExtSender::new(&s_bits, &base_out, &cols).expect("sender");

        let mut seed = [0u8; KAPPA_BYTES];
        rng.fill(&mut seed);
        let resp = ext_receiver.kos_response(&seed);
        !ext_sender.kos_verify(&seed, &resp)
    }

    /// The attack the module header describes: a receiver that uses a different
    /// choice-bit vector in different columns. Without the check this leaks the
    /// sender's `s` bit by bit.
    ///
    /// Corrupting many columns is caught with overwhelming probability — each
    /// one independently has a `1/2` chance of mattering (see
    /// [`a_single_corrupted_column_is_a_one_bit_guess`]), so 40 of them leave a
    /// `2^-40` escape probability.
    #[test]
    fn inconsistent_columns_are_rejected() {
        let cols: Vec<usize> = (0..40).collect();
        assert!(
            cheat_is_caught(&cols),
            "a receiver deviating across 40 columns must be caught"
        );
    }

    /// **A property of KOS worth stating rather than hiding.**
    ///
    /// Tampering with a *single* column is invisible exactly when the sender's
    /// `s_j = 0`, because then `q_j = G(k0_j)` and the transmitted column is
    /// never read. So a one-column deviation is a **one-bit guess at `s`** that
    /// succeeds with probability `1/2` — and when it succeeds, the sender's view
    /// is genuinely unaffected, so there is nothing to detect.
    ///
    /// This is not a weakness of the implementation; it is why KOS is stated as
    /// "the receiver may guess `k` bits of `s`, succeeding with probability
    /// `2^-k`", and why an aborting sender must not be restarted with the same
    /// `s`. A test asserting single-column detection would be asserting
    /// something false — it was, and it failed intermittently until this was
    /// understood.
    #[test]
    fn a_single_corrupted_column_is_a_one_bit_guess() {
        let trials = 60;
        let caught = (0..trials).filter(|_| cheat_is_caught(&[5])).count();
        // Binomial(60, 1/2): the chance of falling outside 12..48 is far below
        // any tolerable flake rate, while the interval is wide enough that the
        // test is not itself flaky.
        assert!(
            (12..=48).contains(&caught),
            "expected roughly half of single-column cheats to be caught, got {caught}/{trials}"
        );
    }

    #[test]
    fn an_honest_transcript_verifies() {
        let mut rng = rand::thread_rng();
        let choices: Vec<bool> = (0..48).map(|_| rng.gen()).collect();

        let s_bits: Vec<bool> = (0..KAPPA).map(|_| rng.gen::<bool>()).collect();
        let (base_sender, setup) = base::Sender::new(&mut rng);
        let (base_receiver, base_choices) =
            base::Receiver::new(&setup, &s_bits, &mut rng).expect("base ot");
        let base_keys = base_sender.keys(&base_choices).expect("base keys");
        let base_out: Vec<Block> = base_receiver.keys().to_vec();

        let (ext_receiver, cols) = ExtReceiver::new(&choices, &base_keys).expect("extend");
        let ext_sender = ExtSender::new(&s_bits, &base_out, &cols).expect("sender");

        let mut seed = [0u8; KAPPA_BYTES];
        rng.fill(&mut seed);
        let resp = ext_receiver.kos_response(&seed);
        assert!(ext_sender.kos_verify(&seed, &resp), "honest run must verify");
    }
}

/// The IKNP extension as an [`super::OtProtocol`].
///
/// The extension produces *random* OTs, so serving chosen messages needs one
/// extra flight: the sender pads each message with the random output it holds and
/// sends both, `2·κ` bits per OT. That cost is added to the reported traffic —
/// see the note on [`super::OtProtocol`] for why the number is therefore higher
/// than the random-OT figure.
pub struct Iknp;

impl super::OtProtocol for Iknp {
    fn name(&self) -> &'static str {
        "extension"
    }
    fn public_key_per_ot(&self) -> bool {
        false // κ base OTs total, then symmetric-key only
    }
    fn transfer(
        &self,
        choices: &[bool],
        messages: &[(Block, Block)],
    ) -> Result<super::Outcome, String> {
        if choices.len() != messages.len() {
            return Err(format!(
                "{} choices for {} message pairs",
                choices.len(),
                messages.len()
            ));
        }
        let ext = simulate(choices)?;
        // Derandomize: e_b = y_b XOR m_b, and the receiver opens the one it can.
        let received = ext
            .sender
            .iter()
            .zip(messages)
            .zip(&ext.receiver)
            .zip(choices)
            .map(|((((y0, y1), (m0, m1)), got), &c)| {
                let (e0, e1) = (xor_block(y0, m0), xor_block(y1, m1));
                xor_block(got, if c { &e1 } else { &e0 })
            })
            .collect();
        let mut traffic = ext.traffic;
        // The pads: two κ-bit blocks per OT, sender to receiver.
        traffic.sender_to_receiver += 2 * messages.len() * KAPPA_BYTES;
        Ok(super::Outcome { received, traffic })
    }
}

/// **Correlated OT** over `Z_2^k`: the receiver gets `r + c·δ`, the sender keeps
/// `r`, and only one `k`-bit correction is sent per OT.
///
/// This is the interface Beaver-triple generation actually wants, and it is
/// cheaper than the chosen-message one for a specific reason: because the two
/// messages differ by a *known* correlation, the sender does not have to pad both.
/// From random OT outputs `(y0, y1)` it sets `r = y0` and sends
/// `t = δ − (y1 − y0)`; the receiver adds `t` exactly when its choice bit is 1:
///
/// ```text
/// c = 0 → y0            = r
/// c = 1 → y1 + t = y1 + δ − y1 + y0 = r + δ
/// ```
///
/// Cost is therefore `κ` bits for the extension column plus `k` bits of
/// correction — which is the figure [`crate::mpcc::ot_cost`] prices Beaver
/// triples from. Using the chosen-message interface here instead would send two
/// `κ`-bit pads and overcharge triple generation by roughly 2×.
pub struct Correlated {
    /// `r` per OT, the sender's half of the additive sharing.
    pub sender_r: Vec<u64>,
    /// `r + c·δ` per OT, the receiver's half.
    pub receiver: Vec<u64>,
    pub traffic: OtTraffic,
}

/// Run `choices.len()` correlated OTs with per-OT correlations `deltas`,
/// in process, mod `2^k`.
pub fn correlated(choices: &[bool], deltas: &[u64], k: u32) -> Result<Correlated, String> {
    if choices.len() != deltas.len() {
        return Err(format!(
            "{} choices for {} correlations",
            choices.len(),
            deltas.len()
        ));
    }
    let mask: u64 = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
    let ext = simulate(choices)?;

    let block_u64 = |b: &Block| -> u64 { u64::from_le_bytes(b[..8].try_into().unwrap()) & mask };

    let mut sender_r = Vec::with_capacity(choices.len());
    let mut receiver = Vec::with_capacity(choices.len());
    for (i, (&c, &delta)) in choices.iter().zip(deltas).enumerate() {
        let (y0, y1) = ext.sender[i];
        let (r, y1v) = (block_u64(&y0), block_u64(&y1));
        // t = delta - (y1 - y0), all mod 2^k.
        let t = delta.wrapping_sub(y1v.wrapping_sub(r)) & mask;
        let got = block_u64(&ext.receiver[i]);
        let out = if c { got.wrapping_add(t) & mask } else { got };
        sender_r.push(r);
        receiver.push(out);
    }

    let mut traffic = ext.traffic;
    // One k-bit correction per OT, sender to receiver.
    traffic.sender_to_receiver += choices.len() * (k as usize).div_ceil(8);
    Ok(Correlated {
        sender_r,
        receiver,
        traffic,
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

    /// The defining property of correlated OT.
    #[test]
    fn correlated_ot_delivers_r_plus_c_delta() {
        let mut rng = rand::thread_rng();
        let k = 64u32;
        let n = 200;
        let choices: Vec<bool> = (0..n).map(|_| rng.gen::<bool>()).collect();
        let deltas: Vec<u64> = (0..n).map(|_| rng.gen::<u64>()).collect();
        let out = correlated(&choices, &deltas, k).unwrap();
        for i in 0..n {
            let want = if choices[i] {
                out.sender_r[i].wrapping_add(deltas[i])
            } else {
                out.sender_r[i]
            };
            assert_eq!(out.receiver[i], want, "COT {i} (choice {})", choices[i]);
        }
    }

    /// COT must be cheaper than the chosen-message interface, since it sends one
    /// `k`-bit correction instead of two `κ`-bit pads. That gap is exactly why
    /// triple generation is priced from this interface.
    #[test]
    fn correlated_ot_is_cheaper_than_chosen_message() {
        use crate::ot::OtProtocol;
        // Large enough that the fixed kappa base OTs amortize: at n = 2048 the
        // base phase still adds 2.0 B/OT on top of the analytic 24.
        let n = 8192;
        let mut rng = rand::thread_rng();
        let choices: Vec<bool> = (0..n).map(|_| rng.gen::<bool>()).collect();
        let deltas: Vec<u64> = (0..n).map(|_| rng.gen::<u64>()).collect();
        let cot = correlated(&choices, &deltas, 64).unwrap();

        let msgs: Vec<(Block, Block)> = (0..n)
            .map(|_| (random_block(&mut rng), random_block(&mut rng)))
            .collect();
        let chosen = Iknp.transfer(&choices, &msgs).unwrap();

        let per = |b: usize| b as f64 / n as f64;
        assert!(
            per(cot.traffic.total()) < per(chosen.traffic.total()),
            "COT {:.1} B/OT should beat chosen-message {:.1} B/OT",
            per(cot.traffic.total()),
            per(chosen.traffic.total())
        );
        // kappa/8 column + k/8 correction = 16 + 8 = 24 B, plus the amortized base.
        let floor = 16.0 + 8.0;
        assert!(
            per(cot.traffic.total()) >= floor && per(cot.traffic.total()) < floor + 1.0,
            "COT should sit just above the {floor} B/OT analytic floor, got {:.2}",
            per(cot.traffic.total())
        );
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
