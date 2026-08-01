//! Real 3-party (two-garbler) garbled-circuit protocol — the **Yao leg** of
//! the ABY3 sharing triangle, wired over the transport.
//!
//! # Why this exists
//!
//! Comparisons dominate the MPC side of the hybrid pipeline (every ReLU,
//! max-pool and argmax is the sign bit of a subtraction). The batched ABY3 bit
//! decomposition (`aby3_protocol::a2b_party`) computes them in `O(log b)`
//! communication rounds; a garbled circuit does it in **one** round regardless
//! of bit width, which wins on high-latency links (see `bench::gc_compare`).
//! This module turns the single-process cost harness in `crypto::garbled` into
//! the actual distributed protocol, with `a2y`/`y2a` conversions so a garbled
//! comparison composes with the rest of the RSS pipeline.
//!
//! # No oblivious transfer (the ABY3 trick)
//!
//! Shares are replicated: `v = s₀+s₁+s₂` with party `i` holding `(sᵢ, sᵢ₊₁)`
//! (`aby3_protocol::PartyShare`). Fix garblers `{P0, P1}` and evaluator `P2`.
//! Then
//!
//! - `P0 = (s₀, s₁)` knows `x = s₀+s₁`, and
//! - `P1 = (s₁, s₂)` knows `y = s₂`,
//!
//! so the value is `v = x + y` where **each summand is known to one garbler**.
//! A garbled input wire only needs *one* party who both knows the wire's value
//! and can compute its active label — and both garblers know every zero-label
//! and the global offset `R`, because they seed an *identical* `Garbler` from
//! the pairwise PRF key `k₀₁` that `P2` does not hold. So each garbler simply
//! sends the active labels for its own summand; there is no wire whose value is
//! private to the evaluator, hence **no OT** (the expensive part of 2-party
//! garbling). This is exactly the honest-majority setting baru targets
//! (`docs/malicious-hybrid.md` §2); the circuit is the unchanged 2-input
//! `MSB(x+y)` / `ReLU(x+y)` from `crypto::garbled`.
//!
//! # Roles
//!
//! For now the roles are fixed: garblers are parties 0 and 1, the evaluator is
//! party 2. Rotating the evaluator (to balance load, or to garble the layers
//! whose inputs a given party is missing) is a straightforward generalization
//! left for later.
//!
//! # Security: still semi-honest, with one deviation point closed
//!
//! [`gc_lt_batch_table_checked`] adds the garbled-table cross-check, which
//! catches a garbler that substitutes a circuit computing a different function
//! — the strongest attack available here — for 32 bytes per batch.
//!
//! That is **not** sufficient for malicious security, and this module does not
//! claim it. The OT-free construction that makes this leg fast is exactly what
//! blocks the rest: because `x = s₀+s₁` is known to garbler 0 alone, no other
//! party can recompute its input labels, so replication does not reach them.
//! [`GC_MALICIOUS_GAPS`] states the three remaining deviations precisely and
//! gives the three-input-adder design that closes them.
//!
//! Callers that need maliciously secure comparisons should use the `a2b` route
//! in [`crate::crypto::aby3_verify::lt_batch_checked`], which is complete; the
//! garbled leg is a latency optimisation, off by default.
//!
//! # Output (Y2A)
//!
//! The garbled output is boolean-shared: the evaluator observes `out ⊕ mask`
//! and both garblers hold the identical `mask`. `y2a_*` turns that back into a
//! standard RSS *arithmetic* share by re-sharing the evaluator's observed bits
//! (owner `P2`) and the garblers' mask bits (owner `P0`), XORing per bit
//! (`o ⊕ m = o + m − 2·o·m`, one batched multiply), and — for the multi-bit
//! ReLU output — recomposing `Σ 2ⁱ·(oᵢ⊕mᵢ)` locally.

use crate::crypto::aby3_protocol::{
    add_party, mul_batch_party, scalar_mul_party, share_input_batch, sub_party, PartyShare,
};
use crate::crypto::garbled::{eval_and, eval_xor, AndTable, Garbler, Label};
use crate::crypto::net::{serialize_msg, Mailbox, MpcMsg, PartyId, Transport};
use crate::crypto::security::{Abort, AbortReason, Checked};
use crate::crypto::types::Fr;

const LABEL_BYTES: usize = 16;
const TABLE_BYTES: usize = 2 * LABEL_BYTES;
const TIMEOUT_MS: u64 = 60_000;

/// The evaluator party id (parties 0 and 1 are the garblers).
pub const EVALUATOR: PartyId = 2;

/// True if `me` is one of the two garblers.
#[inline]
pub fn is_garbler(me: PartyId) -> bool {
    me == 0 || me == 1
}

/// Deterministic per-item garbling seed shared by the two garblers. Both derive
/// it from the pairwise key `k₀₁` (garbler 0's `mine`, garbler 1's `prev`) so
/// they produce a bit-identical circuit; the evaluator lacks `k₀₁` and never
/// computes this. A distinct seed per `(op_id, item)` keeps every item's wire
/// labels independent — reusing labels across items would let the evaluator XOR
/// two active labels and learn the XOR of the two inputs.
fn item_seed(k01: &[u8; 32], op_id: u64, item: usize) -> u64 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(k01);
    h.update(op_id.to_le_bytes());
    h.update((item as u64).to_le_bytes());
    let d = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[..8]);
    u64::from_le_bytes(b)
}

/// The pairwise key `k₀₁` shared by the two garblers, as seen by `me`.
fn garbler_shared_key<T: Transport>(mbx: &Mailbox<T>, me: PartyId) -> [u8; 32] {
    let keys = mbx.prf_keys(me);
    match me {
        0 => keys.mine, // shared with party 1 (my successor)
        1 => keys.prev, // shared with party 0 (my predecessor)
        _ => panic!("garbler_shared_key: party {me} is not a garbler"),
    }
}

#[inline]
fn read_label(buf: &[u8], off: usize) -> Label {
    let mut l = [0u8; LABEL_BYTES];
    l.copy_from_slice(&buf[off..off + LABEL_BYTES]);
    l
}

#[inline]
fn push_label(buf: &mut Vec<u8>, l: &Label) {
    buf.extend_from_slice(l);
}

/// Bit `i` of the low `nbits` of a share component value.
#[inline]
fn bit(v: Fr, i: usize) -> bool {
    (v >> i) & 1 == 1
}

// ─── The two circuit shapes we garble ───────────────────────────────────────
//
// Both reduce to the 2-input circuits already in `crypto::garbled`:
//   * comparator: MSB(x + y) over `nbits` bits          → 1 output bit
//   * fused relu: (v>=0) ? (v>>shift) : 0, v = x + y     → (nbits-shift) bits
// with x = s₀+s₁ (garbler 0) and y = s₂ (garbler 1).

#[derive(Clone, Copy)]
enum Shape {
    /// One output bit: the sign of `x + y`.
    Comparator,
    /// `nbits - shift` output bits: fused ReLU∘rescale of `x + y`.
    Relu { shift: usize },
}

impl Shape {
    fn n_and(&self, nbits: usize) -> usize {
        match self {
            Shape::Comparator => nbits - 1,
            Shape::Relu { shift } => (nbits - 1) + (nbits - shift),
        }
    }
    /// Garble one item, returning `(input_zero_labels_x, input_zero_labels_y,
    /// tables, per-output-bit garbler mask)`.
    fn garble(
        &self,
        g: &mut Garbler,
        nbits: usize,
    ) -> (Vec<Label>, Vec<Label>, Vec<AndTable>, Vec<bool>) {
        match self {
            Shape::Comparator => {
                let c = g.garble_comparator(nbits);
                (c.x0, c.y0, c.tables, vec![c.out_mask])
            }
            Shape::Relu { shift } => {
                let c = g.garble_relu_rescale(nbits, *shift);
                (c.x0, c.y0, c.tables, c.out_masks)
            }
        }
    }
}

/// Evaluator side: given the garbled tables and the active input labels for the
/// two summands, evaluate one item and return the observed output bits
/// (`out ⊕ mask`). Mirrors the gate order and deterministic tweaks of the
/// garbler (`crypto::garbled`), so no tweaks travel on the wire.
fn eval_item(
    shape: Shape,
    nbits: usize,
    tables: &[AndTable],
    xa: &[Label],
    ya: &[Label],
) -> Vec<bool> {
    // Ripple-carry adder over the two summands, shared by both shapes.
    // Produces the `nbits` sum labels (mod 2^nbits); ANDs consume tables 0..nbits-1.
    let mut s = vec![[0u8; LABEL_BYTES]; nbits];
    let mut carry = [0u8; LABEL_BYTES];
    let mut carry_const0 = true;
    let mut ti = 0usize;
    for i in 0..nbits {
        s[i] = if carry_const0 {
            eval_xor(&xa[i], &ya[i])
        } else {
            eval_xor(&eval_xor(&xa[i], &ya[i]), &carry)
        };
        if i < nbits - 1 {
            let (u, v) = if carry_const0 {
                (xa[i], ya[i])
            } else {
                (eval_xor(&xa[i], &carry), eval_xor(&ya[i], &carry))
            };
            let and = eval_and(&u, &v, &tables[ti], 2 * i as u64);
            ti += 1;
            carry = if carry_const0 {
                and
            } else {
                eval_xor(&and, &carry)
            };
            carry_const0 = false;
        }
    }
    match shape {
        Shape::Comparator => {
            // MSB(x+y) = s[nbits-1]; observed bit is its lsb.
            vec![s[nbits - 1][0] & 1 == 1]
        }
        Shape::Relu { shift } => {
            // keep = NOT(sign) = s[nbits-1] active label unchanged at eval;
            // out_j = keep ∧ s[shift+j], tables continue after the adder.
            let keep = s[nbits - 1];
            let width = nbits - shift;
            (0..width)
                .map(|j| {
                    let g = 2 * (nbits - 1) as u64 + 2 * j as u64;
                    let o = eval_and(&keep, &s[shift + j], &tables[ti + j], g);
                    o[0] & 1 == 1
                })
                .collect()
        }
    }
}

// ─── The distributed protocol ────────────────────────────────────────────────

/// Result of the garble+evaluate phase, before Y2A: on the garblers, the
/// per-item per-output-bit mask; on the evaluator, the per-item observed bits.
struct GcOutput {
    /// Garblers only: `masks[item][bit]`.
    masks: Vec<Vec<bool>>,
    /// Evaluator only: `observed[item][bit]`.
    observed: Vec<Vec<bool>>,
    /// Evaluator only, checked path only: garbler 1's table digest disagreed
    /// with what garbler 0 shipped.
    table_mismatch: bool,
}

/// Run the garble/evaluate round for `n` items of the given `shape`. `xs`/`ys`
/// are the per-item summand values (`x = s₀+s₁` on garbler 0, `y = s₂` on
/// garbler 1); ignored on the other parties. `op_id` tags the two garbler
/// messages.
/// Digest of a garbler's tables for the whole batch, used by the checked path
/// to detect a garbler that ships a circuit computing the wrong function.
fn table_digest(all: &[Vec<AndTable>]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"baru/gc/tables/v1");
    for item in all {
        for t in item {
            h.update(t.tg);
            h.update(t.te);
        }
    }
    h.finalize().into()
}

fn garble_evaluate<T: Transport>(
    shape: Shape,
    nbits: usize,
    xs: &[Fr],
    ys: &[Fr],
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
    verify: bool,
) -> GcOutput {
    // Complete the pairwise-PRF ring handshake on *all three* parties before
    // the roles diverge. Otherwise, on a standalone GC call (the first protocol
    // op, before any multiplication has run the handshake), the garblers would
    // block in `prf_keys` waiting for the evaluator's key while the evaluator
    // sits in `recv` waiting for garbled payloads — a deadlock until timeout.
    // Idempotent (the keys are cached) once any earlier op has established them.
    let _ = mbx.prf_keys(me);

    let n = xs.len().max(ys.len()).max(1);
    let n = if is_garbler(me) {
        if me == 0 {
            xs.len()
        } else {
            ys.len()
        }
    } else {
        n
    };

    if is_garbler(me) {
        let k01 = garbler_shared_key(mbx, me);
        let mut payload = Vec::new();
        let mut masks = Vec::with_capacity(n);
        // Garbler 1 derives a bit-identical circuit from k₀₁ but normally
        // discards the tables, since garbler 0 is the one that ships them.
        // Retaining them is what makes the cross-check free.
        let mut all_tables: Vec<Vec<AndTable>> = Vec::new();
        for item in 0..n {
            let seed = item_seed(&k01, op_id, item);
            let mut g = Garbler::new(seed);
            let (x0, y0, tables, mask) = shape.garble(&mut g, nbits);
            if verify && me == 1 {
                all_tables.push(tables.clone());
            }
            if me == 0 {
                // Garbler 0 ships the tables + active labels for x = s₀+s₁.
                for t in &tables {
                    push_label(&mut payload, &t.tg);
                    push_label(&mut payload, &t.te);
                }
                let xv = xs[item];
                for i in 0..nbits {
                    push_label(&mut payload, &g.active(&x0[i], bit(xv, i)));
                }
            } else {
                // Garbler 1 ships active labels for y = s₂ only.
                let yv = ys[item];
                for i in 0..nbits {
                    push_label(&mut payload, &g.active(&y0[i], bit(yv, i)));
                }
            }
            masks.push(mask);
        }
        if verify && me == 1 {
            // One digest for the whole batch, appended after the labels.
            payload.extend_from_slice(&table_digest(&all_tables));
        }
        mbx.send(
            EVALUATOR,
            serialize_msg(&MpcMsg::GcFromGarbler {
                op_id,
                garbler: me,
                payload,
            }),
        );
        GcOutput {
            masks,
            observed: Vec::new(),
            table_mismatch: false,
        }
    } else {
        // Evaluator: collect both garblers' payloads (any order).
        let mut p0: Option<Vec<u8>> = None;
        let mut p1: Option<Vec<u8>> = None;
        while p0.is_none() || p1.is_none() {
            let (_from, m) = mbx
                .recv_matching(
                    TIMEOUT_MS,
                    |m| matches!(m, MpcMsg::GcFromGarbler { op_id: o, .. } if *o == op_id),
                )
                .unwrap_or_else(|| panic!("gc: evaluator timeout on op_id={op_id}"));
            if let MpcMsg::GcFromGarbler {
                garbler, payload, ..
            } = m
            {
                if garbler == 0 {
                    p0 = Some(payload);
                } else {
                    p1 = Some(payload);
                }
            }
        }
        let p0 = p0.unwrap();
        let p1 = p1.unwrap();

        let n_and = shape.n_and(nbits);
        let g0_per = n_and * TABLE_BYTES + nbits * LABEL_BYTES; // tables + x labels
        let g1_per = nbits * LABEL_BYTES; // y labels
        let digest_bytes = if verify { 32 } else { 0 };
        let n = (p1.len() - digest_bytes) / g1_per;
        debug_assert_eq!(p0.len(), n * g0_per, "garbler-0 payload size");
        debug_assert_eq!(
            p1.len(),
            n * g1_per + digest_bytes,
            "garbler-1 payload size"
        );

        let mut observed = Vec::with_capacity(n);
        let mut seen_tables: Vec<Vec<AndTable>> = Vec::new();
        for item in 0..n {
            let base0 = item * g0_per;
            let mut tables = Vec::with_capacity(n_and);
            for t in 0..n_and {
                let o = base0 + t * TABLE_BYTES;
                tables.push(AndTable {
                    tg: read_label(&p0, o),
                    te: read_label(&p0, o + LABEL_BYTES),
                });
            }
            let xbase = base0 + n_and * TABLE_BYTES;
            let xa: Vec<Label> = (0..nbits)
                .map(|i| read_label(&p0, xbase + i * LABEL_BYTES))
                .collect();
            let ybase = item * g1_per;
            let ya: Vec<Label> = (0..nbits)
                .map(|i| read_label(&p1, ybase + i * LABEL_BYTES))
                .collect();
            observed.push(eval_item(shape, nbits, &tables, &xa, &ya));
            if verify {
                seen_tables.push(tables);
            }
        }
        // Both garblers derive the same circuit from k₀₁, so garbler 1's digest
        // pins what garbler 0 shipped. With at most one corruption, one of them
        // is honest, so a substituted circuit cannot survive this.
        let table_mismatch = verify && {
            let claimed = &p1[n * g1_per..];
            table_digest(&seen_tables) != claimed
        };
        GcOutput {
            masks: Vec::new(),
            observed,
            table_mismatch,
        }
    }
}

/// Y2A: convert the boolean-shared garbled output into RSS arithmetic shares.
///
/// For each output wire, `v = observed ⊕ mask` with `observed` known to the
/// evaluator (`P2`) and `mask` to the garblers (`P0`/`P1`). We re-share both as
/// RSS bits (owner `P2` / owner `P0`), XOR per bit with one batched multiply,
/// then recompose `Σ 2ⁱ·(oᵢ⊕mᵢ)` locally per item. `id_base` must head a block
/// of at least `4·n·width + 8` reserved ids.
fn y2a<T: Transport>(
    out: &GcOutput,
    n: usize,
    width: usize,
    result_nbits: usize,
    id_base: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    // Flatten to per-(item,bit) so we can batch the XOR multiply.
    let total = n * width;
    let mut next_id = id_base;
    let take = |n: &mut u64| {
        let v = *n;
        *n += 1;
        v
    };

    // Re-share observed bits (owner = evaluator) and mask bits (owner = P0).
    //
    // **Two batched rounds, not `2·n·width` sequential ones.** These loops used
    // to call `share_input` per (item, bit): at a 5408-wide conv layer and
    // width 12 that is 129 792 blocking round trips for a single ReLU, which is
    // most of the phase's cost and none of its work. `share_input_batch` sends
    // the same bytes with the same randomness in one message per peer.
    //
    // This is the fifth instance of one defect — the scheme-switch import,
    // `gadgets::relu`, `bench-hybrid-nn` and `bench-deep-nn` all had it, and the
    // batched primitive existed the whole time in every case.
    let obs_vals: Vec<Fr> = (0..n)
        .flat_map(|item| {
            (0..width).map(move |b| {
                if me == EVALUATOR {
                    out.observed[item][b] as Fr
                } else {
                    0
                }
            })
        })
        .collect();
    let obs_sh = share_input_batch(&obs_vals, take(&mut next_id), EVALUATOR, me, mbx);

    // Both garblers hold the mask; owner 0 supplies it.
    let mask_vals: Vec<Fr> = (0..n)
        .flat_map(|item| {
            (0..width).map(move |b| if me == 0 { out.masks[item][b] as Fr } else { 0 })
        })
        .collect();
    let mask_sh = share_input_batch(&mask_vals, take(&mut next_id), 0, me, mbx);

    // XOR per bit: o ⊕ m = o + m − 2·o·m. One batched multiply for all bits.
    let prod_base = take(&mut next_id);
    let pairs: Vec<(PartyShare, PartyShare)> = (0..total)
        .map(|k| (obs_sh[k].clone(), mask_sh[k].clone()))
        .collect();
    let prods = mul_batch_party(&pairs, prod_base, me, mbx);
    let xor_bits: Vec<PartyShare> = (0..total)
        .map(|k| {
            sub_party(
                &add_party(&obs_sh[k], &mask_sh[k]),
                &scalar_mul_party(&prods[k], 2),
            )
        })
        .collect();

    // Recompose Σ 2ⁱ·bitᵢ per item (local).
    let _ = result_nbits;
    (0..n)
        .map(|item| {
            let mut acc = PartyShare { left: 0, right: 0 };
            for b in 0..width {
                let term = scalar_mul_party(&xor_bits[item * width + b], 1u128 << b);
                acc = add_party(&acc, &term);
            }
            acc
        })
        .collect()
}

// ─── Public entry points ─────────────────────────────────────────────────────

/// Reserved id block per `gc_lt_batch` / `gc_relu_batch` call, given `n` items
/// of output width `w`.
///
/// **Deliberately larger than what Y2A now consumes.** Since the two
/// `share_input` loops became two *batched* calls, Y2A spends 2 ids plus the
/// multiply block rather than `2·n·w` — but the reservation stays, because it is
/// a lockstep counter every party advances by the same amount and shrinking it
/// buys nothing while risking a collision with a caller that has not been
/// updated. Over-reserving ids is free; under-reserving is a silent
/// cross-operation collision.
pub fn gc_id_block(n: usize, width: usize) -> u64 {
    (4 * n * width + 16) as u64
}

/// Garbled `a < b` for a whole batch: returns RSS arithmetic shares (0/1) of
/// `MSB((aᵢ−bᵢ) mod 2^nbits)`, i.e. signed `aᵢ < bᵢ`. One garble round + one
/// resharing round regardless of `nbits`. `op_id` must head a reserved id block
/// (`op_id` for the garble messages; `op_id+1..` for Y2A — reserve
/// `1 + gc_id_block(n, 1)`).
pub fn gc_lt_batch<T: Transport>(
    a: &[PartyShare],
    b: &[PartyShare],
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    assert_eq!(a.len(), b.len(), "gc_lt_batch: mismatched batch lengths");
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    let mask = if nbits >= 128 {
        Fr::MAX
    } else {
        (1u128 << nbits) - 1
    };
    // pre = a − b (local); x = s₀+s₁ (garbler 0), y = s₂ (garbler 1).
    let diffs: Vec<PartyShare> = (0..n).map(|i| sub_party(&a[i], &b[i])).collect();
    let (xs, ys) = summands(&diffs, mask, me);
    let out = garble_evaluate(Shape::Comparator, nbits, &xs, &ys, op_id, me, mbx, false);
    y2a(&out, n, 1, nbits, op_id + 1, me, mbx)
}

/// [`gc_lt_batch`] with the garbled-table cross-check enabled.
///
/// **This is not a maliciously secure comparison.** It closes exactly one of
/// the protocol's deviation points — see [`GC_MALICIOUS_GAPS`] — and is exposed
/// so the cost of that check can be measured and so the remaining gaps are
/// stated in code rather than only in prose. Callers requiring malicious
/// security must use the `a2b` route in
/// [`crate::crypto::aby3_verify::lt_batch_checked`].
pub fn gc_lt_batch_table_checked<T: Transport>(
    a: &[PartyShare],
    b: &[PartyShare],
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Checked<Vec<PartyShare>> {
    assert_eq!(a.len(), b.len(), "gc_lt_batch: mismatched batch lengths");
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mask = if nbits >= 128 {
        Fr::MAX
    } else {
        (1u128 << nbits) - 1
    };
    let diffs: Vec<PartyShare> = (0..n).map(|i| sub_party(&a[i], &b[i])).collect();
    let (xs, ys) = summands(&diffs, mask, me);
    let out = garble_evaluate(Shape::Comparator, nbits, &xs, &ys, op_id, me, mbx, true);
    let mismatch = out.table_mismatch;
    // Finish the protocol before reporting, for the same reason every other
    // check in this crate is deferred: bailing out here would desynchronise the
    // parties — the garblers would block forever in the Y2A resharing that the
    // evaluator never joined — and would make the abort observable at a point
    // that depends on which party detected it. The shares below are meaningless
    // when `mismatch` is set; the caller must honour the `Err`.
    let shares = y2a(&out, n, 1, nbits, op_id + 1, me, mbx);
    if mismatch {
        return Err(Abort::new(
            AbortReason::GarbleMismatch,
            "gc table cross-check",
        ));
    }
    Ok(shares)
}

/// The deviations the garbled leg does **not** yet cover, and what would close
/// them. Kept as a documented constant so the gap cannot quietly go stale.
///
/// # What the table cross-check does cover
///
/// Both garblers derive a bit-identical circuit from the pairwise key `k₀₁`, so
/// garbler 1 can digest the tables garbler 0 must have shipped. With at most
/// one corruption, one of the two is honest, so a garbler that substitutes a
/// circuit computing a *different function* — the most powerful attack
/// available here — is caught. Cost: 32 bytes per batch.
///
/// # What it does not cover
///
/// **1. Input labels for `x`.** The protocol is OT-free precisely because
/// `x = s₀+s₁` is known to garbler 0 alone: garbler 1 holds `(s₁, s₂)` and the
/// evaluator holds `(s₂, s₀)`, so neither can reconstruct `x`. Nobody can
/// recompute garbler 0's active labels, so replication does not reach them, and
/// a corrupt garbler 0 can evaluate the circuit at a value of its choosing.
///
/// **2. Input labels for `y`.** `y = s₂` is known to garbler 1 and the
/// evaluator, but the evaluator lacks `k₀₁` and so cannot derive the zero
/// labels needed to check what it was sent.
///
/// **3. A lying evaluator.** The garblers hold the output masks but not the
/// output, so an evaluator that misreports `observed` is not contradicted by
/// anything.
///
/// # What would close them
///
/// Garble a **three-input** adder over `(s₀, s₁, s₂)` instead of the two-input
/// `x + y`. Each additive component is then known to exactly two parties, which
/// restores replication on every input wire:
///
/// * `s₁` is known to both garblers — cross-checkable by digest, as the tables
///   already are;
/// * `s₀` is known to garbler 0 and the evaluator, so garbler 1 (which knows
///   the zero labels but not the value) can send the evaluator the hash pair
///   `(H(S⁰), H(S¹))` and let it verify the label it received against the bit
///   it already holds;
/// * `s₂` symmetrically, with the garblers' roles swapped.
///
/// Gap 3 is closed by standard output authenticity: the garblers commit to both
/// output labels, and the evaluator must return the *label* rather than the
/// bit — it only holds one of the two and cannot forge the other.
///
/// The cost is a wider adder (three summands rather than two) plus two hash
/// pairs per input wire. This is a real design, not a sketch, but it means
/// rewriting the garbled circuit shapes in [`crate::crypto::garbled`], which is
/// why it is scoped separately rather than bolted on here.
pub const GC_MALICIOUS_GAPS: &str =
    "garbled leg: tables cross-checked; input labels and evaluator output are not";

/// Garbled fused ReLU∘rescale for a batch: `outᵢ = (vᵢ>=0) ? (vᵢ>>shift) : 0`
/// for signed `vᵢ` held in `vals`. Returns RSS arithmetic shares of the
/// activations. One garble round + one resharing round. Reserve
/// `1 + gc_id_block(n, nbits-shift)` ids from `op_id`.
pub fn gc_relu_batch<T: Transport>(
    vals: &[PartyShare],
    nbits: usize,
    shift: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let n = vals.len();
    if n == 0 {
        return Vec::new();
    }
    let mask = if nbits >= 128 {
        Fr::MAX
    } else {
        (1u128 << nbits) - 1
    };
    let (xs, ys) = summands(vals, mask, me);
    let width = nbits - shift;
    let out = garble_evaluate(
        Shape::Relu { shift },
        nbits,
        &xs,
        &ys,
        op_id,
        me,
        mbx,
        false,
    );
    y2a(&out, n, width, nbits, op_id + 1, me, mbx)
}

/// Split each replicated share into the two garbler summands: garbler 0 gets
/// `x = (left+right) mod 2^nbits` (its `s₀+s₁`), garbler 1 gets `y = right`
/// (its `s₂`). Non-garblers get empty vectors (they don't send labels).
fn summands(shares: &[PartyShare], mask: Fr, me: PartyId) -> (Vec<Fr>, Vec<Fr>) {
    match me {
        0 => (
            shares
                .iter()
                .map(|s| s.left.wrapping_add(s.right) & mask)
                .collect(),
            Vec::new(),
        ),
        1 => (Vec::new(), shares.iter().map(|s| s.right & mask).collect()),
        _ => (Vec::new(), Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::aby3_protocol::{reveal, share_input};
    use crate::crypto::net::InprocHandle;
    use std::thread;

    /// Drive a 3-party in-process run: `f(me, mbx)` on each party, joined.
    fn run3<F, R>(f: F) -> Vec<R>
    where
        F: Fn(PartyId, &Mailbox<InprocHandle>) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let handles = InprocHandle::create(3);
        let f = std::sync::Arc::new(f);
        let mut js = Vec::new();
        for h in handles {
            let f = f.clone();
            js.push(thread::spawn(move || {
                let mbx = Mailbox::new(h);
                let me = mbx.id();
                f(me, &mbx)
            }));
        }
        js.into_iter().map(|j| j.join().unwrap()).collect()
    }

    /// Share a public vector of signed values as RSS (owner 0), each in `nbits`.
    fn share_vec<T: Transport>(
        vals: &[i64],
        nbits: usize,
        base_id: u64,
        me: PartyId,
        mbx: &Mailbox<T>,
    ) -> Vec<PartyShare> {
        let mask = (1i128 << nbits) - 1;
        vals.iter()
            .enumerate()
            .map(|(i, &v)| {
                let val = if me == 0 { (v as i128 & mask) as Fr } else { 0 };
                share_input(val, base_id + i as u64, 0, me, mbx)
            })
            .collect()
    }

    #[test]
    fn gc_lt_matches_plaintext_3party() {
        let nbits = 20usize;
        let a_vals = vec![3i64, -4, 100, -100, 0, 500, -500, 1];
        let b_vals = vec![5i64, -4, 99, -50, 1, -500, 500, 1];
        let want: Vec<u128> = (0..a_vals.len())
            .map(|i| (a_vals[i] < b_vals[i]) as u128)
            .collect();

        let av = a_vals.clone();
        let bv = b_vals.clone();
        let results = run3(move |me, mbx| {
            let n = av.len();
            let a = share_vec(&av, nbits, 1, me, mbx);
            let b = share_vec(&bv, nbits, 1 + n as u64, me, mbx);
            let op_id = 1000u64;
            let lt = gc_lt_batch(&a, &b, nbits, op_id, me, mbx);
            // reveal each bit
            let rid = op_id + 1 + gc_id_block(n, 1);
            lt.iter()
                .enumerate()
                .map(|(i, s)| reveal(s, rid + i as u64, None, me, mbx) & 1)
                .collect::<Vec<u128>>()
        });
        for r in results {
            assert_eq!(r, want, "gc_lt bits");
        }
    }

    #[test]
    fn gc_relu_matches_plaintext_3party() {
        let nbits = 20usize;
        let shift = 8usize;
        let vals = vec![0i64, 100, -1, -4000, 1 << 17, 12345, -12345, 65536];
        let want: Vec<i64> = vals
            .iter()
            .map(|&v| if v >= 0 { v >> shift } else { 0 })
            .collect();

        let vv = vals.clone();
        let results = run3(move |me, mbx| {
            let n = vv.len();
            let v = share_vec(&vv, nbits, 1, me, mbx);
            let op_id = 2000u64;
            let acts = gc_relu_batch(&v, nbits, shift, op_id, me, mbx);
            let width = nbits - shift;
            let rid = op_id + 1 + gc_id_block(n, width);
            acts.iter()
                .enumerate()
                .map(|(i, s)| {
                    let raw = reveal(s, rid + i as u64, None, me, mbx);
                    let m = (1i128 << nbits) - 1;
                    let x = raw as i128 & m;
                    if x >= (1i128 << (nbits - 1)) {
                        (x - (1i128 << nbits)) as i64
                    } else {
                        x as i64
                    }
                })
                .collect::<Vec<i64>>()
        });
        for r in results {
            assert_eq!(r, want, "gc_relu activations");
        }
    }
}
