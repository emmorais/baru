//! A real (semi-honest) garbled-circuit engine, built to *measure* the cost
//! of comparisons — the operation that dominates the MPC side of the hybrid
//! pipeline (every ReLU, max-pool and argmax is a comparison, i.e. the sign
//! bit of a subtraction).
//!
//! It produces genuine garbled tables on the wire and does genuine hashing per
//! gate, so the reported bytes and compute time are real. (It replaced a
//! cleartext-logic placeholder, `interpreter::gc_backend`, which nothing ever
//! constructed and which was deleted on 2026-07-30.) The point is a head-to-head against the batched ABY3 bit
//! decomposition (`aby3_protocol::a2b_batch`): **garbled circuits evaluate a
//! comparison in a constant number of rounds, whereas bit decomposition
//! needs ~log(bitwidth) rounds** — so GC is expected to win on high-latency
//! (WAN) links and lose on a fast LAN, where its per-gate crypto and garbled
//! tables cost more than a few batched ring multiplications.
//!
//! # What is implemented
//!
//! - **Free-XOR** (Kolesnikov–Schneider): a global offset `R` with lsb 1;
//!   the two labels of a wire are `W0` and `W0 ⊕ R`. XOR and NOT gates are
//!   free (no table, no hashing, no communication).
//! - **Half-gates** (Zahur–Rosulek–Evans, EUROCRYPT 2015): an AND gate costs
//!   **two** ciphertexts on the wire and two hashes to evaluate.
//! - **Hash** `H(label, tweak)` = the first 16 bytes of
//!   `SHA-256(label ‖ tweak)`. (AES-based hashing would be faster on hardware
//!   with AES-NI; SHA-256 is already a dependency and is correct — the
//!   *ratio* to bit decomposition, not the absolute constant, is the point.)
//!
//! # Circuit
//!
//! The comparison `lt(a,b)` is the most-significant bit of the two's
//! complement difference `a − b`, matching `Backend::lt`. Equivalently it is
//! `MSB(x + y)` for the two arithmetic shares `x, y` of `a − b`, which is a
//! `b`-bit ripple adder: `b − 1` AND gates (one carry per bit) plus free
//! XORs. The output is a garbled bit whose semantics match a boolean sharing
//! (evaluator holds the masked value, garbler holds the mask).

use sha2::{Digest, Sha256};

/// A 128-bit wire label.
pub type Label = [u8; 16];

const ZERO: Label = [0u8; 16];

#[inline]
fn xor(a: &Label, b: &Label) -> Label {
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = a[i] ^ b[i];
    }
    o
}

#[inline]
fn lsb(a: &Label) -> bool {
    a[0] & 1 == 1
}

/// `H(label, tweak)` — the gate-hashing function. Truncated SHA-256 of the
/// label concatenated with a per-gate tweak (prevents cross-gate reuse).
#[inline]
fn hash(label: &Label, tweak: u64) -> Label {
    let mut h = Sha256::new();
    h.update(label);
    h.update(tweak.to_le_bytes());
    let d = h.finalize();
    let mut o = [0u8; 16];
    o.copy_from_slice(&d[..16]);
    o
}

/// One garbled AND gate: two ciphertexts (the half-gate rows).
#[derive(Clone, Copy)]
pub struct AndTable {
    pub tg: Label,
    pub te: Label,
}

/// Statistics from garbling/evaluating a batch of comparisons.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// AND gates garbled (XOR/NOT are free).
    pub and_gates: usize,
    /// Bytes placed on the wire: garbled tables + evaluator input labels.
    /// (In the 3-party honest-majority setting the input labels are sent by
    /// the co-garbler, so no oblivious transfer is charged; see the note in
    /// `bench::gc_compare`.)
    pub wire_bytes: usize,
    /// Communication rounds. A shared-output comparison is one round: the
    /// garbler(s) send tables + input labels, the evaluator evaluates
    /// locally and both hold a share of the result.
    pub rounds: usize,
}

/// The garbler: holds the global offset and generates labels/tables.
pub struct Garbler {
    r: Label,
    next_tweak: u64,
}

impl Garbler {
    pub fn new(seed: u64) -> Self {
        // Derive a deterministic R with lsb 1 (free-XOR requirement).
        let mut r = hash(&ZERO, seed ^ 0x9E37_79B9_7F4A_7C15);
        r[0] |= 1;
        Garbler { r, next_tweak: 1 }
    }

    /// Fresh input-wire zero-label.
    pub fn input_zero(&mut self, seed: u64) -> Label {
        let t = self.next_tweak;
        self.next_tweak += 1;
        hash(&[seed as u8; 16], t ^ 0xA5A5_A5A5)
    }

    /// The active label carrying `value` on a wire whose zero-label is `w0`.
    pub fn active(&self, w0: &Label, value: bool) -> Label {
        if value {
            xor(w0, &self.r)
        } else {
            *w0
        }
    }

    /// Garble a XOR gate (free): returns the output zero-label.
    #[inline]
    pub fn xor_gate(&self, a0: &Label, b0: &Label) -> Label {
        xor(a0, b0)
    }

    /// Garble an AND gate with an explicit gate tweak `g` (the caller assigns
    /// tweaks deterministically so the evaluator can reproduce them without
    /// receiving them on the wire). Uses hash tweaks `g` and `g+1`; assign
    /// successive gates even tweaks `2i` to avoid overlap. Returns
    /// `(output_zero_label, table)`.
    pub fn and_gate_t(&self, a0: &Label, b0: &Label, g: u64) -> (Label, AndTable) {
        let (j, j2) = (g, g + 1);
        let a1 = xor(a0, &self.r);
        let b1 = xor(b0, &self.r);
        let pa = lsb(a0);
        let pb = lsb(b0);

        // Generator half-gate.
        let ha0 = hash(a0, j);
        let ha1 = hash(&a1, j);
        let tg = xor(&xor(&ha0, &ha1), &if pb { self.r } else { ZERO });
        let wg0 = xor(&ha0, &if pa { tg } else { ZERO });

        // Evaluator half-gate.
        let hb0 = hash(b0, j2);
        let hb1 = hash(&b1, j2);
        let te = xor(&xor(&hb0, &hb1), a0);
        let we0 = xor(&hb0, &if pb { xor(&te, a0) } else { ZERO });

        let c0 = xor(&wg0, &we0);
        (c0, AndTable { tg, te })
    }
}

/// Evaluate a garbled AND gate given active input labels and the table.
pub fn eval_and(a: &Label, b: &Label, table: &AndTable, gate_tweak: u64) -> Label {
    let (j, j2) = (gate_tweak, gate_tweak + 1);
    let wg = xor(&hash(a, j), &if lsb(a) { table.tg } else { ZERO });
    let we = xor(&hash(b, j2), &if lsb(b) { xor(&table.te, a) } else { ZERO });
    xor(&wg, &we)
}

/// Evaluate a garbled XOR gate (free).
#[inline]
pub fn eval_xor(a: &Label, b: &Label) -> Label {
    xor(a, b)
}

/// Garble the circuit `MSB(x + y)` over `b`-bit inputs and return the
/// per-bit garbled tables plus the two input zero-label vectors. This is the
/// comparison circuit: with `x, y` the arithmetic shares of `a − b`, its
/// output bit is `lt(a, b)`.
pub struct ComparatorGarbling {
    pub x0: Vec<Label>,
    pub y0: Vec<Label>,
    pub tables: Vec<AndTable>,
    pub gate_tweaks: Vec<u64>,
    pub out0: Label,
    /// Output permute bit (the garbler's share of the result): the evaluator
    /// observes `msb ⊕ out_mask`, and this is `out_mask = lsb(out0)`.
    pub out_mask: bool,
}

impl Garbler {
    pub fn garble_comparator(&mut self, b: usize) -> ComparatorGarbling {
        let x0: Vec<Label> = (0..b).map(|i| self.input_zero(i as u64)).collect();
        let y0: Vec<Label> = (0..b).map(|i| self.input_zero((b + i) as u64)).collect();
        let mut tables = Vec::with_capacity(b - 1);
        let mut gate_tweaks = Vec::with_capacity(b - 1);

        // Ripple carry: c_0 = 0; c_{i+1} = MAJ(x_i, y_i, c_i)
        //   = ((x_i ⊕ c_i) ∧ (y_i ⊕ c_i)) ⊕ c_i   (one AND per bit)
        // Output MSB = x_{b-1} ⊕ y_{b-1} ⊕ c_{b-1}.
        let mut carry0 = ZERO; // constant-0 zero-label is the all-zero label
        let mut carry_is_const0 = true;
        for i in 0..b - 1 {
            let (u0, v0);
            if carry_is_const0 {
                u0 = x0[i];
                v0 = y0[i];
            } else {
                u0 = self.xor_gate(&x0[i], &carry0);
                v0 = self.xor_gate(&y0[i], &carry0);
            }
            // Deterministic gate tweak `2i` — the evaluator reproduces it from
            // the gate index, so tweaks are never sent on the wire.
            let g = 2 * i as u64;
            let (and0, table) = self.and_gate_t(&u0, &v0, g);
            tables.push(table);
            gate_tweaks.push(g);
            carry0 = if carry_is_const0 {
                and0
            } else {
                self.xor_gate(&and0, &carry0)
            };
            carry_is_const0 = false;
        }
        // MSB = x_{b-1} ⊕ y_{b-1} ⊕ c_{b-1}
        let msb0 = self.xor_gate(&self.xor_gate(&x0[b - 1], &y0[b - 1]), &carry0);
        let out_mask = lsb(&msb0);

        ComparatorGarbling {
            x0,
            y0,
            tables,
            gate_tweaks,
            out0: msb0,
            out_mask,
        }
    }
}

/// The garbled fused ReLU∘rescale: `out = (v >= 0) ? (v >> shift) : 0`, where
/// `v = x + y` is the pre-activation reconstructed from its two arithmetic
/// shares. This is the *whole* nonlinearity the hybrid pipeline runs
/// (`bench::cnn::fused_relu_rescale_batch`), garbled — proving a garbled
/// circuit computes the entire ReLU layer in one round, not just the
/// comparison. Output is the `b - shift` bits of the activation.
pub struct ReluGarbling {
    pub x0: Vec<Label>,
    pub y0: Vec<Label>,
    pub tables: Vec<AndTable>,
    pub gate_tweaks: Vec<u64>,
    pub out0: Vec<Label>,
    /// Per-output-bit garbler mask (`lsb(out0[i])`); the evaluator observes
    /// `out_i ⊕ mask_i`, so the activation is boolean-shared.
    pub out_masks: Vec<bool>,
}

impl Garbler {
    pub fn garble_relu_rescale(&mut self, b: usize, shift: usize) -> ReluGarbling {
        let x0: Vec<Label> = (0..b).map(|i| self.input_zero(i as u64)).collect();
        let y0: Vec<Label> = (0..b).map(|i| self.input_zero((b + i) as u64)).collect();
        let mut tables = Vec::new();
        let mut gate_tweaks = Vec::new();
        let mut g = 0u64;

        // Full-adder chain: sum bits s_i = x_i ⊕ y_i ⊕ c_i (free), carry
        // c_{i+1} = MAJ(x_i, y_i, c_i) = ((x_i⊕c_i)∧(y_i⊕c_i))⊕c_i.
        let mut s0 = vec![ZERO; b];
        let mut carry0 = ZERO;
        let mut carry_const0 = true;
        for i in 0..b {
            if carry_const0 {
                s0[i] = self.xor_gate(&x0[i], &y0[i]);
            } else {
                s0[i] = self.xor_gate(&self.xor_gate(&x0[i], &y0[i]), &carry0);
            }
            if i < b - 1 {
                let (u0, v0) = if carry_const0 {
                    (x0[i], y0[i])
                } else {
                    (self.xor_gate(&x0[i], &carry0), self.xor_gate(&y0[i], &carry0))
                };
                let (and0, table) = self.and_gate_t(&u0, &v0, g);
                tables.push(table);
                gate_tweaks.push(g);
                g += 2;
                carry0 = if carry_const0 { and0 } else { self.xor_gate(&and0, &carry0) };
                carry_const0 = false;
            }
        }

        // keep = NOT(sign) = NOT(s_{b-1}); free-XOR NOT flips the zero-label.
        let sign0 = s0[b - 1];
        let keep0 = xor(&sign0, &self.r);

        // out_i = keep ∧ s_{i+shift}, for i in 0..(b-shift).
        let width = b - shift;
        let mut out0 = Vec::with_capacity(width);
        for j in 0..width {
            let (o0, table) = self.and_gate_t(&keep0, &s0[shift + j], g);
            tables.push(table);
            gate_tweaks.push(g);
            g += 2;
            out0.push(o0);
        }
        let out_masks = out0.iter().map(lsb).collect();

        ReluGarbling { x0, y0, tables, gate_tweaks, out0, out_masks }
    }
}

/// Evaluate a garbled comparator on concrete inputs `x, y` (the evaluator's
/// view: it holds active labels, delivered by the garbler(s)). Returns the
/// observed output bit `msb ⊕ out_mask`. Used by the self-test to confirm
/// correctness against the plaintext `MSB(x+y)`.
pub fn eval_comparator(g: &Garbler, c: &ComparatorGarbling, x: u128, y: u128, b: usize) -> bool {
    // Active input labels (in a real run these arrive on the wire).
    let xa: Vec<Label> = (0..b).map(|i| g.active(&c.x0[i], (x >> i) & 1 == 1)).collect();
    let ya: Vec<Label> = (0..b).map(|i| g.active(&c.y0[i], (y >> i) & 1 == 1)).collect();

    let mut carry = ZERO;
    let mut carry_const0 = true;
    for i in 0..b - 1 {
        let (u, v);
        if carry_const0 {
            u = xa[i];
            v = ya[i];
        } else {
            u = eval_xor(&xa[i], &carry);
            v = eval_xor(&ya[i], &carry);
        }
        let and = eval_and(&u, &v, &c.tables[i], c.gate_tweaks[i]);
        carry = if carry_const0 { and } else { eval_xor(&and, &carry) };
        carry_const0 = false;
    }
    let msb = eval_xor(&eval_xor(&xa[b - 1], &ya[b - 1]), &carry);
    lsb(&msb)
}

/// Evaluate a garbled fused ReLU∘rescale (self-test helper): returns the
/// observed output bits `out_i ⊕ mask_i`.
pub fn eval_relu_rescale(
    g: &Garbler,
    c: &ReluGarbling,
    x: u128,
    y: u128,
    b: usize,
    shift: usize,
) -> Vec<bool> {
    let xa: Vec<Label> = (0..b).map(|i| g.active(&c.x0[i], (x >> i) & 1 == 1)).collect();
    let ya: Vec<Label> = (0..b).map(|i| g.active(&c.y0[i], (y >> i) & 1 == 1)).collect();
    let mut sa = vec![ZERO; b];
    let mut carry = ZERO;
    let mut carry_const0 = true;
    let mut ti = 0usize;
    for i in 0..b {
        sa[i] = if carry_const0 {
            eval_xor(&xa[i], &ya[i])
        } else {
            eval_xor(&eval_xor(&xa[i], &ya[i]), &carry)
        };
        if i < b - 1 {
            let (u, v) = if carry_const0 {
                (xa[i], ya[i])
            } else {
                (eval_xor(&xa[i], &carry), eval_xor(&ya[i], &carry))
            };
            let and = eval_and(&u, &v, &c.tables[ti], c.gate_tweaks[ti]);
            ti += 1;
            carry = if carry_const0 { and } else { eval_xor(&and, &carry) };
            carry_const0 = false;
        }
    }
    // keep = NOT(sign): at eval the active label is unchanged.
    let keep = sa[b - 1];
    let width = b - shift;
    (0..width)
        .map(|j| {
            let o = eval_and(&keep, &sa[shift + j], &c.tables[ti + j], c.gate_tweaks[ti + j]);
            lsb(&o)
        })
        .collect()
}

/// Cost of garbling + evaluating `n` independent fused ReLU∘rescale ops in
/// the 3-party honest-majority setting. `2b - shift - 1` AND gates each
/// (`b-1` adder + `b-shift` mux), plus `2b` input labels; output is
/// boolean-shared so a downstream B2A returns it to arithmetic.
pub fn relu_stats(n: usize, b: usize, shift: usize) -> GcStats {
    let and_gates = n * ((b - 1) + (b - shift));
    let table_bytes = and_gates * 2 * 16;
    let input_label_bytes = n * (2 * b) * 16;
    GcStats { and_gates, wire_bytes: table_bytes + input_label_bytes, rounds: 1 }
}

/// Cost of garbling + evaluating `n` independent `b`-bit comparisons in the
/// 3-party honest-majority setting (input labels delivered by the co-garbler,
/// no oblivious transfer). Bytes = garbled tables (`2·16` per AND) + one
/// active label per input wire (`2b` labels of 16 bytes).
pub fn comparator_stats(n: usize, b: usize) -> GcStats {
    let and_gates = n * (b - 1);
    let table_bytes = and_gates * 2 * 16;
    let input_label_bytes = n * (2 * b) * 16;
    GcStats {
        and_gates,
        wire_bytes: table_bytes + input_label_bytes,
        rounds: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The garbled comparator must output `MSB(x+y mod 2^b)` for every input
    /// — i.e. real garbling + evaluation is correct, not a cleartext stub.
    #[test]
    fn garbled_comparator_matches_plaintext() {
        let b = 20;
        let mask = (1u128 << b) - 1;
        let mut g = Garbler::new(12345);
        let circuit = g.garble_comparator(b);
        let cases: &[(u128, u128)] = &[
            (0, 0),
            (1, 0),
            (mask, 1),
            (12345, 67890),
            (mask, mask),
            (1 << (b - 1), 0),
            (0, 1 << (b - 1)),
            ((1 << (b - 1)) - 1, 1),
        ];
        for &(x, y) in cases {
            let want = (((x + y) & mask) >> (b - 1)) & 1 == 1;
            let observed = eval_comparator(&g, &circuit, x, y, b);
            // The true bit is observed ⊕ out_mask (garbler's share).
            let got = observed ^ circuit.out_mask;
            assert_eq!(got, want, "MSB(x+y) for x={x}, y={y}");
        }
    }

    /// The garbled fused ReLU∘rescale must match the plaintext
    /// `(v>=0) ? (v>>shift) : 0` for signed `v = x+y`, over `b` bits — the
    /// whole nonlinearity the pipeline runs, garbled and correct.
    #[test]
    fn garbled_relu_rescale_matches_plaintext() {
        let (b, shift) = (20usize, 8usize);
        let mask = (1u128 << b) - 1;
        let mut g = Garbler::new(777);
        let c = g.garble_relu_rescale(b, shift);
        let signed = |v: u128| -> i128 {
            if v >> (b - 1) & 1 == 1 { v as i128 - (1i128 << b) } else { v as i128 }
        };
        let cases: &[(u128, u128)] = &[
            (0, 0),
            (100, 50),
            (mask, 0),            // -1 -> ReLU 0
            (mask - 3, 1),        // small negative -> 0
            (1 << (b - 2), 0),    // positive
            (12345, 20000),
        ];
        for &(x, y) in cases {
            let s = (x + y) & mask;
            let v = signed(s);
            let want = if v >= 0 { (v >> shift) as u128 } else { 0 };
            let observed = eval_relu_rescale(&g, &c, x, y, b, shift);
            let mut got = 0u128;
            for (i, (&o, &m)) in observed.iter().zip(&c.out_masks).enumerate() {
                got |= ((o ^ m) as u128) << i;
            }
            assert_eq!(got, want, "ReLU∘rescale for x={x}, y={y} (v={v})");
        }
    }

    /// Free-XOR: XOR and NOT add no garbled tables; only AND gates cost bytes.
    #[test]
    fn only_and_gates_cost_bytes() {
        let b = 16;
        let s = comparator_stats(1, b);
        assert_eq!(s.and_gates, b - 1, "one AND per carry bit");
        assert_eq!(s.rounds, 1, "shared-output comparison is one round");
    }
}
