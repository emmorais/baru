//! Cheating-detection tests for the malicious ABY3 layer.
//!
//! These are the tests the malicious upgrade exists for. A protocol that never
//! aborts on a correct run proves nothing on its own — every semi-honest
//! protocol has that property. What has to be shown is that a party which
//! *deviates* is caught, at every point the protocol lets it deviate.
//!
//! The harness runs three real parties in three threads over the in-process
//! transport, with one party's outgoing messages rewritten by a [`Deviant`]
//! wrapper. That is a genuine active attack on the wire, not a simulated one:
//! the corrupt party runs the honest code and its messages are corrupted in
//! transit, so the honest parties see exactly what a real adversary would send.

use std::sync::Arc;

use baru::crypto::aby3_protocol::{reveal, share_input, PartyShare};
use baru::crypto::aby3_verify::{
    a2b_batch_checked, add, check, generate_mac_pairs, lt_batch_checked, mac_existing_batch,
    mul_batch_checked, reveal_checked, share_public, MalShare, Verifier,
};
use baru::crypto::gc_protocol::gc_lt_batch_table_checked;
use baru::crypto::net::{
    deserialize_msg, serialize_msg, InprocHandle, Mailbox, Message, MpcMsg, PartyId, Transport,
};
use baru::crypto::security::{require_security, AbortReason, AttestedSecurity, Checked, Security};
use baru::crypto::types::Fr;
use baru::interpreter::backend::Backend;
use baru::interpreter::malicious_backend::MaliciousBackend;

/// How one party deviates: given the recipient and the honest message, return
/// the message to actually send, or `None` to let it pass untouched.
type TamperRule = Arc<dyn Fn(PartyId, &MpcMsg) -> Option<MpcMsg> + Send + Sync>;

/// A transport that rewrites this party's outgoing messages.
///
/// `rule` returns `Some(replacement)` to corrupt a message and `None` to let it
/// pass. Applied to both `send` and `broadcast`, so it can model a party that
/// lies to one peer, to both, or inconsistently to each.
#[derive(Clone)]
struct Deviant {
    inner: InprocHandle,
    rule: TamperRule,
}

impl Deviant {
    fn wrap(
        inner: InprocHandle,
        rule: impl Fn(PartyId, &MpcMsg) -> Option<MpcMsg> + Send + Sync + 'static,
    ) -> Self {
        Deviant {
            inner,
            rule: Arc::new(rule),
        }
    }

    fn maybe_corrupt(&self, to: PartyId, msg: &Message) -> Message {
        let parsed: MpcMsg = deserialize_msg(msg);
        match (self.rule)(to, &parsed) {
            Some(replacement) => serialize_msg(&replacement),
            None => msg.clone(),
        }
    }
}

impl Transport for Deviant {
    fn id(&self) -> PartyId {
        self.inner.id()
    }

    fn send(&self, to: PartyId, msg: Message) {
        let m = self.maybe_corrupt(to, &msg);
        self.inner.send(to, m);
    }

    fn broadcast(&self, msg: Message) {
        // Corrupt per recipient, so a rule can lie to one peer and not the
        // other — the inconsistency that replication is supposed to catch.
        for to in 0..3 {
            if to == self.inner.id() {
                continue;
            }
            let m = self.maybe_corrupt(to, &msg);
            self.inner.send(to, m);
        }
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        self.inner.recv()
    }

    fn wait_ready(&self, timeout_ms: u64) -> bool {
        self.inner.wait_ready(timeout_ms)
    }
}

/// The computation every test runs: share two secrets, multiply them, add a
/// public constant, reveal, then take the deferred verdict.
///
/// Returns the revealed value and the verdict. Deliberately exercises input
/// sharing, multiplication, a local linear operation and a reveal — the four
/// places a party can act.
///
/// `local_mul_offset` models the part of an attack a wire-tap cannot: a corrupt
/// party that adds an offset to the product must also keep its **own** copy of
/// that share consistent with what it sent, or the sharing becomes malformed
/// and the far cheaper transcript check catches it instead. Applying the offset
/// both on the wire and locally produces a well-formed sharing of the *wrong*
/// value — which is precisely the attack the MAC check exists for, and the only
/// one it can be credited with catching.
fn session(
    me: PartyId,
    mbx: &Mailbox<impl Transport>,
    a: Fr,
    b: Fr,
    local_mul_offset: Option<Fr>,
) -> (Fr, Checked<()>) {
    let keys = mbx.prf_keys(me);
    let mut v = Verifier::new(me, &keys);

    // Inputs: party 0 owns `a`, party 1 owns `b`.
    let a_val = share_input(a, 1, 0, me, mbx);
    v.note_input(1, &a_val);
    let b_val = share_input(b, 2, 1, me, mbx);
    v.note_input(2, &b_val);

    // MAC each input: [r·x] costs one multiplication per input, once.
    let key = v.key().clone();
    let macs = baru::crypto::aby3_protocol::mul_batch_party(
        &[(key.clone(), a_val.clone()), (key.clone(), b_val.clone())],
        10,
        me,
        mbx,
    );
    let x = MalShare {
        val: a_val,
        mac: macs[0].clone(),
    };
    let y = MalShare {
        val: b_val,
        mac: macs[1].clone(),
    };

    // z = a*b + 7
    let mut z = mul_batch_checked(&[(x, y)], 20, me, mbx, &mut v)
        .pop()
        .expect("one product");
    if let Some(d) = local_mul_offset {
        z.val.left = z.val.left.wrapping_add(d);
    }
    let seven = share_public(7, me, &key);
    let out = add(&z, &seven);

    let revealed = reveal_checked(&out.val, 30, me, mbx, &mut v);
    let verdict = check(me, mbx, 40, &mut v);
    (revealed, verdict)
}

/// Run `session` on three parties, with `corrupt` (if any) rewriting its
/// outgoing traffic. Returns each party's `(revealed, verdict)`.
fn run(a: Fr, b: Fr, corrupt: Option<(PartyId, TamperRule)>) -> Vec<(Fr, Checked<()>)> {
    run_with(a, b, corrupt, None)
}

/// [`run`], plus the local-state half of an attack — see `session`.
fn run_with(
    a: Fr,
    b: Fr,
    corrupt: Option<(PartyId, TamperRule)>,
    local_mul_offset: Option<Fr>,
) -> Vec<(Fr, Checked<()>)> {
    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        let is_corrupt = corrupt.as_ref().is_some_and(|(who, _)| *who == i);
        let rule = corrupt
            .as_ref()
            .filter(|(who, _)| *who == i)
            .map(|(_, r)| r.clone());
        let offset = if is_corrupt { local_mul_offset } else { None };
        joins.push(std::thread::spawn(move || match rule {
            Some(r) => {
                let mbx = Mailbox::new(Deviant::wrap(h, move |to, m| r(to, m)));
                session(i, &mbx, a, b, offset)
            }
            None => {
                let mbx = Mailbox::new(h);
                session(i, &mbx, a, b, offset)
            }
        }));
    }
    joins
        .into_iter()
        .map(|j| j.join().expect("party panicked"))
        .collect()
}

/// Verdicts of the two honest parties when `corrupt` is the corrupt one.
fn honest_verdicts(results: &[(Fr, Checked<()>)], corrupt: PartyId) -> Vec<&Checked<()>> {
    results
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != corrupt)
        .map(|(_, (_, v))| v)
        .collect()
}

#[test]
fn honest_run_agrees_and_does_not_abort() {
    let results = run(6, 7, None);
    for (value, verdict) in &results {
        assert_eq!(*value, 49, "6*7+7 should reconstruct to 49");
        assert!(verdict.is_ok(), "honest run must not abort: {verdict:?}");
    }
}

/// Rebuild a resharing message with `f` applied to every value.
///
/// The payload is bit-packed, so a tamper has to unpack, act on the values, and
/// repack — expressing the attack on what the protocol carries rather than on
/// how it is encoded.
fn tamper(base_op_id: u64, bits: u32, data: &[u8], f: impl Fn(u128) -> u128) -> MpcMsg {
    let n = data.len() * 8 / bits as usize;
    let vals: Vec<u128> = baru::crypto::aby3_protocol::unpack_ring(data, n, bits)
        .into_iter()
        .map(f)
        .collect();
    MpcMsg::TaggedZBatch {
        base_op_id,
        bits,
        data: baru::crypto::aby3_protocol::pack_ring(&vals, bits),
    }
}

/// The central case: a corrupt party adds an offset to its resharing message.
/// Nothing can recompute `z_i`, so only the MAC check can catch this.
#[test]
fn corrupted_multiplication_is_caught() {
    for corrupt in 0..3 {
        let rule = Arc::new(|_to: PartyId, m: &MpcMsg| match m {
            MpcMsg::TaggedZBatch {
                base_op_id,
                bits,
                data,
            } if *base_op_id == 20 => {
                // Offset the *value* half of the batch only, leaving the MAC
                // half consistent — the subtle attack, not a blunt one.
                let n = data.len() * 8 / *bits as usize;
                let vals: Vec<u128> = baru::crypto::aby3_protocol::unpack_ring(data, n, *bits)
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| if i == 0 { v.wrapping_add(1) } else { v })
                    .collect();
                Some(MpcMsg::TaggedZBatch {
                    base_op_id: *base_op_id,
                    bits: *bits,
                    data: baru::crypto::aby3_protocol::pack_ring(&vals, *bits),
                })
            }
            _ => None,
        });
        let results = run_with(6, 7, Some((corrupt, rule)), Some(1));
        for verdict in honest_verdicts(&results, corrupt) {
            let abort = verdict
                .as_ref()
                .expect_err("a corrupted product must abort");
            assert_eq!(
                abort.reason,
                AbortReason::MultiplicationCheck,
                "party {corrupt} cheating in the multiply should fail the sacrifice check"
            );
        }
    }
}

/// An error that is zero modulo the value width does not change any meaningful
/// value, and the module header's soundness argument explicitly does not claim
/// to catch it. Asserting this keeps the claim honest: the check catches errors
/// that *matter*, not every possible perturbation.
#[test]
fn an_error_that_cannot_change_the_result_is_not_claimed_to_be_caught() {
    let rule = Arc::new(|_to: PartyId, m: &MpcMsg| match m {
        MpcMsg::TaggedZBatch {
            base_op_id,
            bits,
            data,
        } if *base_op_id == 20 => Some(tamper(*base_op_id, *bits, data, |v| {
            v.wrapping_add(1u128 << 100)
        })),
        _ => None,
    });
    let results = run(6, 7, Some((1, rule)));
    // The revealed low bits are still correct: 2^100 does not reach them.
    for (i, (value, _)) in results.iter().enumerate() {
        if i != 1 {
            assert_eq!(*value & ((1u128 << 64) - 1), 49);
        }
    }
}

#[test]
fn inconsistent_input_sharing_is_caught() {
    // Party 0 deals `a`. Send party 2 a different component than party 1 gets.
    let rule = Arc::new(|to: PartyId, m: &MpcMsg| match m {
        MpcMsg::InputShare { share_id, value } if *share_id == 1 && to == 2 => {
            Some(MpcMsg::InputShare {
                share_id: *share_id,
                value: value.wrapping_add(1),
            })
        }
        _ => None,
    });
    let results = run(6, 7, Some((0, rule)));
    for verdict in honest_verdicts(&results, 0) {
        let abort = verdict
            .as_ref()
            .expect_err("inconsistent dealing must abort");
        assert!(
            matches!(
                abort.reason,
                AbortReason::ViewMismatch | AbortReason::MultiplicationCheck
            ),
            "unexpected reason {:?}",
            abort.reason
        );
    }
}

#[test]
fn a_lie_during_reveal_is_caught() {
    let rule = Arc::new(|_to: PartyId, m: &MpcMsg| match m {
        MpcMsg::RevealShare { share_id, value } if *share_id == 30 => Some(MpcMsg::RevealShare {
            share_id: *share_id,
            value: value.wrapping_add(1),
        }),
        _ => None,
    });
    let results = run(6, 7, Some((2, rule)));
    for verdict in honest_verdicts(&results, 2) {
        let abort = verdict.as_ref().expect_err("a false reveal must abort");
        assert!(
            matches!(
                abort.reason,
                AbortReason::RevealMismatch | AbortReason::ViewMismatch
            ),
            "unexpected reason {:?}",
            abort.reason
        );
    }
}

/// A party that lies to only *one* peer during reveal cannot be caught by that
/// peer's free local check — it is caught by the transcript comparison. This
/// separates the two mechanisms rather than letting one mask the other.
#[test]
fn a_reveal_lie_told_to_only_the_non_holder_is_caught_by_the_transcript() {
    // Party 2 broadcasts component 2. Its predecessor (party 1) holds a copy
    // and checks it for free; party 0 does not, so lie only to party 0.
    let rule = Arc::new(|to: PartyId, m: &MpcMsg| match m {
        MpcMsg::RevealShare { share_id, value } if *share_id == 30 && to == 0 => {
            Some(MpcMsg::RevealShare {
                share_id: *share_id,
                value: value.wrapping_add(1),
            })
        }
        _ => None,
    });
    let results = run(6, 7, Some((2, rule)));
    for verdict in honest_verdicts(&results, 2) {
        let abort = verdict
            .as_ref()
            .expect_err("a reveal lie to the non-holder must still abort");
        assert_eq!(abort.reason, AbortReason::ViewMismatch);
    }
}

/// Abort must be a value, not a panic: callers need to tell cheating apart from
/// a crashed or hung peer, and the benchmarks need to count aborts.
#[test]
fn abort_is_a_result_not_a_panic() {
    let rule = Arc::new(|_to: PartyId, m: &MpcMsg| match m {
        MpcMsg::TaggedZBatch {
            base_op_id,
            bits,
            data,
        } if *base_op_id == 20 => Some(tamper(*base_op_id, *bits, data, |v| v.wrapping_add(1))),
        _ => None,
    });
    // `run` joins every thread and would propagate a panic; reaching this
    // assertion at all is the property under test.
    let results = run(3, 4, Some((1, rule)));
    assert!(honest_verdicts(&results, 1).iter().all(|v| v.is_err()));
}

/// The sound way to MAC a sharing that has none: consume a pre-verified random
/// pair and attach the MAC with local affine arithmetic. This is the path bit
/// decomposition needs, since its leaves are built locally without MACs.
///
/// Multiplying `[r]·[x]` instead would appear to work here and is *unsound* —
/// see `MacPair`'s documentation for the selective-failure attack it opens.
fn pair_mac_session(me: PartyId, mbx: &Mailbox<impl Transport>, a: Fr, b: Fr) -> (Fr, Checked<()>) {
    let keys = mbx.prf_keys(me);
    let mut v = Verifier::new(me, &keys);

    let a_val = share_input(a, 1, 0, me, mbx);
    v.note_input(1, &a_val);
    let b_val = share_input(b, 2, 1, me, mbx);
    v.note_input(2, &b_val);

    // One batched round produces the whole pool; one batched round attaches.
    let pairs = generate_mac_pairs(2, 10, me, mbx, &keys, &mut v);
    let macd = mac_existing_batch(&[a_val, b_val], &pairs, 200, me, mbx, &mut v);

    let z = mul_batch_checked(&[(macd[0].clone(), macd[1].clone())], 20, me, mbx, &mut v)
        .pop()
        .expect("one product");
    let revealed = reveal_checked(&z.val, 30, me, mbx, &mut v);
    let verdict = check(me, mbx, 40, &mut v);
    (revealed, verdict)
}

fn run_pair_mac(a: Fr, b: Fr, corrupt: Option<(PartyId, TamperRule)>) -> Vec<(Fr, Checked<()>)> {
    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        let rule = corrupt
            .as_ref()
            .filter(|(who, _)| *who == i)
            .map(|(_, r)| r.clone());
        joins.push(std::thread::spawn(move || match rule {
            Some(r) => {
                let mbx = Mailbox::new(Deviant::wrap(h, move |to, m| r(to, m)));
                pair_mac_session(i, &mbx, a, b)
            }
            None => {
                let mbx = Mailbox::new(h);
                pair_mac_session(i, &mbx, a, b)
            }
        }));
    }
    joins
        .into_iter()
        .map(|j| j.join().expect("party panicked"))
        .collect()
}

#[test]
fn pair_based_macs_give_a_correct_honest_run() {
    for (a, b) in [(6u128, 7u128), (0, 5), (123, 456)] {
        let results = run_pair_mac(a, b, None);
        for (value, verdict) in &results {
            assert_eq!(*value, a.wrapping_mul(b));
            assert!(
                verdict.is_ok(),
                "honest pair-MAC run must not abort: {verdict:?}"
            );
        }
    }
}

/// Cheating while the random pairs are produced must still be caught — the
/// pairs are only trustworthy because the same sacrifice covers them.
///
/// Crucially this abort leaks nothing: `ρ` is uniform and independent of the
/// inputs, which is the entire reason MACs are attached this way.
#[test]
fn cheating_in_pair_generation_is_caught() {
    let rule = Arc::new(|_to: PartyId, m: &MpcMsg| match m {
        MpcMsg::TaggedZBatch {
            base_op_id,
            bits,
            data,
        } if *base_op_id == 10 => Some(tamper(*base_op_id, *bits, data, |v| v.wrapping_add(1))),
        _ => None,
    });
    let results = run_pair_mac(6, 7, Some((1, rule)));
    for verdict in honest_verdicts(&results, 1) {
        // Which check fires is deliberately not asserted. Corrupting the wire
        // alone leaves the sharing malformed, so the cheap reveal comparison
        // catches it before the sacrifice does; an adversary that also keeps
        // its local state consistent is caught by the sacrifice instead
        // (`corrupted_multiplication_is_caught` covers that path). What matters
        // is that no corrupted pair can reach a MAC.
        verdict
            .as_ref()
            .expect_err("a corrupted random pair must abort");
    }
}

/// The masked opening that attaches a MAC must reveal nothing. `d = x − ρ` is a
/// one-time pad, so the same secret masked by two different pairs must produce
/// unrelated openings.
#[test]
fn each_mac_attachment_consumes_a_distinct_pad() {
    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        joins.push(std::thread::spawn(move || {
            let mbx = Mailbox::new(h);
            let keys = mbx.prf_keys(i);
            let mut v = Verifier::new(i, &keys);
            let pairs = generate_mac_pairs(2, 10, i, &mbx, &keys, &mut v);
            // Same value, two different pairs: the two openings must differ.
            let x = share_input(42, 1, 0, i, &mbx);
            let out = mac_existing_batch(&[x.clone(), x], &pairs, 200, i, &mbx, &mut v);
            (out[0].mac.left, out[1].mac.left)
        }));
    }
    for j in joins {
        let (m0, m1) = j.join().expect("party panicked");
        assert_ne!(
            m0, m1,
            "two attachments of the same value must use independent pads"
        );
    }
}

// ─── Bit decomposition and comparisons ───────────────────────────────────────

/// Decompose `a` and compare `a < b`, both under the malicious protocol.
fn a2b_session(
    me: PartyId,
    mbx: &Mailbox<impl Transport>,
    a: Fr,
    b: Fr,
    nbits: usize,
) -> (Vec<Fr>, Fr, Checked<()>) {
    let keys = mbx.prf_keys(me);
    let mut v = Verifier::new(me, &keys);

    let a_val = share_input(a, 1, 0, me, mbx);
    v.note_input(1, &a_val);
    let b_val = share_input(b, 2, 1, me, mbx);
    v.note_input(2, &b_val);

    let pairs = generate_mac_pairs(2, 10, me, mbx, &keys, &mut v);
    let macd = mac_existing_batch(&[a_val, b_val], &pairs, 200, me, mbx, &mut v);

    let bits = a2b_batch_checked(&macd[..1], nbits, 1_000, me, mbx, &keys, &mut v)
        .pop()
        .expect("one lane");
    let opened: Vec<Fr> = bits
        .iter()
        .enumerate()
        .map(|(k, bit)| reveal_checked(&bit.val, 5_000 + k as u64, me, mbx, &mut v))
        .collect();

    let lt = lt_batch_checked(
        &macd[..1],
        &macd[1..2],
        nbits,
        20_000,
        me,
        mbx,
        &keys,
        &mut v,
    )
    .pop()
    .expect("one comparison");
    let lt_open = reveal_checked(&lt.val, 30_000, me, mbx, &mut v);

    let verdict = check(me, mbx, 40_000, &mut v);
    (opened, lt_open, verdict)
}

fn run_a2b(
    a: Fr,
    b: Fr,
    nbits: usize,
    corrupt: Option<(PartyId, TamperRule)>,
) -> Vec<(Vec<Fr>, Fr, Checked<()>)> {
    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        let rule = corrupt
            .as_ref()
            .filter(|(who, _)| *who == i)
            .map(|(_, r)| r.clone());
        joins.push(std::thread::spawn(move || match rule {
            Some(r) => {
                let mbx = Mailbox::new(Deviant::wrap(h, move |to, m| r(to, m)));
                a2b_session(i, &mbx, a, b, nbits)
            }
            None => {
                let mbx = Mailbox::new(h);
                a2b_session(i, &mbx, a, b, nbits)
            }
        }));
    }
    joins
        .into_iter()
        .map(|j| j.join().expect("party panicked"))
        .collect()
}

/// The malicious decomposition must be bit-exact against the plaintext one, and
/// the comparison must agree with signed `<` on two's-complement values.
#[test]
fn malicious_a2b_and_comparison_are_bit_exact() {
    let nbits = 8;
    let mask = (1u128 << nbits) - 1;
    // Includes negatives in two's-complement form, where the sign bit matters.
    //
    // Operands are kept inside the range where `a − b` does not overflow the
    // signed width. `lt` is `MSB(a − b)`, so it is only correct when the
    // difference is representable — with nbits = 8 the pair (200, 100) is
    // (−56, 100) whose difference is −156, outside [−128, 127], and the sign
    // bit then reports the wrong answer. That is a property of the
    // comparison itself, shared with the semi-honest path, not of the
    // malicious upgrade.
    for (a, b) in [
        (5u128, 9u128),
        (9, 5),
        (0, 0),
        (255, 1),
        (100, 20),
        (20, 100),
        (7, 7),
    ] {
        let results = run_a2b(a, b, nbits, None);
        let a_signed = ((a & mask) as i128)
            - if a & (1 << (nbits - 1)) != 0 {
                1 << nbits
            } else {
                0
            };
        let b_signed = ((b & mask) as i128)
            - if b & (1 << (nbits - 1)) != 0 {
                1 << nbits
            } else {
                0
            };
        for (bits, lt, verdict) in &results {
            assert!(verdict.is_ok(), "honest a2b must not abort: {verdict:?}");
            let recomposed: u128 = bits.iter().enumerate().map(|(k, bit)| (bit & 1) << k).sum();
            assert_eq!(recomposed, a & mask, "decomposition of {a} is wrong");
            assert_eq!(
                *lt & 1,
                (a_signed < b_signed) as u128,
                "signed {a_signed} < {b_signed}"
            );
        }
    }
}

/// Cheating anywhere inside the decomposition's multiplication rounds must
/// abort. This is the check that matters most in practice: every ReLU,
/// max-pool and argmax in the repo reduces to this code path.
#[test]
fn cheating_inside_bit_decomposition_is_caught() {
    // Corrupt the carry-save round of the first decomposition (op ids are
    // allocated from 1_000 upward inside `a2b_batch_checked`).
    let rule = Arc::new(|_to: PartyId, m: &MpcMsg| match m {
        MpcMsg::TaggedZBatch {
            base_op_id,
            bits,
            data,
        } if *base_op_id > 1_000 && *base_op_id < 5_000 => {
            Some(tamper(*base_op_id, *bits, data, |v| v.wrapping_add(1)))
        }
        _ => None,
    });
    let results = run_a2b(5, 9, 8, Some((1, rule)));
    for (i, (_, _, verdict)) in results.iter().enumerate() {
        if i != 1 {
            verdict
                .as_ref()
                .expect_err("cheating inside a2b must abort");
        }
    }
}

/// A corrupt party must not be able to binary-share bits that disagree with the
/// arithmetic component its co-holder actually holds. Nothing goes on the wire
/// when the leaves are built, so only the transcript can catch this.
#[test]
fn inconsistent_decomposition_leaves_are_caught() {
    // Corrupting the *input* sharing makes one party's arithmetic component
    // disagree with its co-holder's copy, so the leaves derived from it differ
    // too.
    let rule = Arc::new(|to: PartyId, m: &MpcMsg| match m {
        MpcMsg::InputShare { share_id, value } if *share_id == 1 && to == 2 => {
            Some(MpcMsg::InputShare {
                share_id: *share_id,
                value: value.wrapping_add(1),
            })
        }
        _ => None,
    });
    let results = run_a2b(5, 9, 8, Some((0, rule)));
    for (i, (_, _, verdict)) in results.iter().enumerate() {
        if i != 0 {
            verdict
                .as_ref()
                .expect_err("inconsistent leaves must abort");
        }
    }
}

// ─── The Backend, end to end ─────────────────────────────────────────────────

/// Run `body` on three `MaliciousBackend` parties and collect each party's
/// (result, verdict).
fn run_backend<R: Send + 'static>(
    body: impl Fn(&mut MaliciousBackend<InprocHandle>) -> R + Send + Sync + Clone + 'static,
) -> Vec<(R, Checked<()>)> {
    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        let body = body.clone();
        joins.push(std::thread::spawn(move || {
            let mut be = MaliciousBackend::new(i, h);
            let out = body(&mut be);
            let verdict = be.finish();
            (out, verdict)
        }));
    }
    joins
        .into_iter()
        .map(|j| j.join().expect("party panicked"))
        .collect()
}

/// The backend must attest malicious security — and, unlike everything else in
/// the tree, actually be entitled to.
#[test]
fn the_malicious_backend_attests_malicious() {
    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        joins.push(std::thread::spawn(move || {
            let be = MaliciousBackend::new(i, h);
            let ok = require_security("malicious-aby3", &be, Security::Malicious).is_ok();
            (be.attested_security(), ok)
        }));
    }
    for j in joins {
        let (level, ok) = j.join().expect("party panicked");
        assert_eq!(level, Security::Malicious);
        assert!(ok, "the gate must admit the backend it was built for");
    }
}

/// Arithmetic through the trait, checked end to end.
#[test]
fn backend_arithmetic_is_correct_and_verifies() {
    let results = run_backend(|be| {
        let a = be.share_secret(12, 32, 0);
        let b = be.share_secret(5, 32, 1);
        let prod = be.mul(&a, &b);
        let sum = be.add(&prod, &a);
        let diff = be.sub(&sum, &b);
        be.reveal(&diff, None)
    });
    for (value, verdict) in &results {
        assert!(
            verdict.is_ok(),
            "honest backend run must verify: {verdict:?}"
        );
        assert_eq!(*value, 12 * 5 + 12 - 5);
    }
}

/// Comparisons and equality through the trait — the operations every gadget in
/// the repo is built from.
#[test]
fn backend_comparisons_match_plaintext() {
    for (a, b) in [(3u128, 9u128), (9, 3), (7, 7), (0, 1)] {
        let results = run_backend(move |be| {
            let x = be.share_secret(a, 16, 0);
            let y = be.share_secret(b, 16, 1);
            let lt = be.lt(&x, &y);
            let le = be.le(&x, &y);
            let eq = be.eq(&x, &y);
            (
                be.reveal(&lt, None) & 1,
                be.reveal(&le, None) & 1,
                be.reveal(&eq, None) & 1,
            )
        });
        for ((lt, le, eq), verdict) in &results {
            assert!(verdict.is_ok(), "{a} vs {b}: {verdict:?}");
            assert_eq!(*lt, (a < b) as u128, "lt {a} {b}");
            assert_eq!(*le, (a <= b) as u128, "le {a} {b}");
            assert_eq!(*eq, (a == b) as u128, "eq {a} {b}");
        }
    }
}

/// A ReLU — the operation the whole private-inference pipeline is made of —
/// computed through the malicious backend and checked against plaintext.
#[test]
fn backend_relu_matches_plaintext() {
    let nbits = 16;
    for v in [0i128, 5, 100, -1, -50] {
        let enc = if v < 0 { (1i128 << nbits) + v } else { v } as u128;
        let results = run_backend(move |be| {
            let x = be.share_secret(enc, nbits, 0);
            let zero = be.share_public(0, nbits);
            // relu(x) = x * (x > 0) = x * (1 - (x < 0))  — the repo's own form.
            let neg = be.lt(&x, &zero);
            let one = be.share_public(1, nbits);
            let pos = be.sub(&one, &neg);
            let out = be.mul(&x, &pos);
            be.reveal(&out, None)
        });
        let expected = v.max(0) as u128;
        for (got, verdict) in &results {
            assert!(verdict.is_ok(), "relu({v}): {verdict:?}");
            assert_eq!(*got & ((1 << nbits) - 1), expected, "relu({v})");
        }
    }
}

/// Every multiplication the backend performs must be registered. A path that
/// skipped registration would leave products unverified while everything still
/// appeared to work — the most dangerous regression possible here.
#[test]
fn backend_registers_every_product() {
    let results = run_backend(|be| {
        let a = be.share_secret(3, 16, 0);
        let b = be.share_secret(4, 16, 1);
        let before = be.verified_products();
        let _ = be.mul_batch(&[(a.clone(), b.clone()), (b.clone(), a.clone())]);
        be.verified_products() - before
    });
    for (delta, verdict) in &results {
        assert!(verdict.is_ok());
        assert_eq!(*delta, 2, "two multiplications must register two products");
    }
}

/// The verifier must not be rewound between expressions: operation ids seed the
/// correlated randomness and tag the transcript, so reuse would be unsound.
#[test]
fn reset_per_expression_keeps_the_verification_state() {
    let results = run_backend(|be| {
        let a = be.share_secret(3, 16, 0);
        let b = be.share_secret(4, 16, 1);
        let _ = be.mul(&a, &b);
        let before = be.verified_products();
        be.reset_per_expression();
        assert_eq!(
            be.verified_products(),
            before,
            "resetting an expression must not discard accumulated verification"
        );
        // Ids must keep advancing so nothing collides after the reset.
        let c = be.share_secret(7, 16, 0);
        c.id
    });
    for (id, verdict) in &results {
        assert!(verdict.is_ok());
        assert!(*id > 1, "ids must not rewind on reset");
    }
}

// ─── Garbled circuits: the table cross-check ─────────────────────────────────

/// The checked garbled comparison must agree with plaintext on an honest run.
#[test]
fn gc_table_checked_comparison_is_correct() {
    let nbits = 16;
    for (a, b) in [(3u128, 9u128), (9, 3), (5, 5)] {
        let handles = InprocHandle::create(3);
        let mut joins = Vec::new();
        for (i, h) in handles.into_iter().enumerate() {
            joins.push(std::thread::spawn(move || {
                let mbx = Mailbox::new(h);
                let x = share_input(a, 1, 0, i, &mbx);
                let y = share_input(b, 2, 1, i, &mbx);
                let out = gc_lt_batch_table_checked(&[x], &[y], nbits, 100, i, &mbx)
                    .expect("honest garbling must not abort");
                reveal(&out[0], 900, None, i, &mbx)
            }));
        }
        let vals: Vec<Fr> = joins
            .into_iter()
            .map(|j| j.join().expect("party panicked"))
            .collect();
        for v in vals {
            assert_eq!(v & 1, (a < b) as u128, "gc lt {a} {b}");
        }
    }
}

/// A garbler that ships a substituted circuit must be caught by the other
/// garbler's digest. This is the strongest attack the garbled leg admits — a
/// corrupt garbler could otherwise evaluate an entirely different function.
#[test]
fn a_substituted_garbled_circuit_is_caught() {
    let nbits = 16;
    // Garbler 0 corrupts the tables it ships; garbler 1's digest disagrees.
    let rule = Arc::new(|_to: PartyId, m: &MpcMsg| match m {
        MpcMsg::GcFromGarbler {
            op_id,
            garbler: 0,
            payload,
        } => {
            let mut p = payload.clone();
            p[0] ^= 0xff; // flip a bit inside the first garbled table
            Some(MpcMsg::GcFromGarbler {
                op_id: *op_id,
                garbler: 0,
                payload: p,
            })
        }
        _ => None,
    });

    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        // Every party gets a Deviant wrapper; only party 0's rule fires, so
        // both branches have the same transport type.
        let rule = rule.clone();
        joins.push(std::thread::spawn(move || {
            let corrupt = i == 0;
            let mbx = Mailbox::new(Deviant::wrap(
                h,
                move |to, m| {
                    if corrupt {
                        rule(to, m)
                    } else {
                        None
                    }
                },
            ));
            let x = share_input(3, 1, 0, i, &mbx);
            let y = share_input(9, 2, 1, i, &mbx);
            gc_lt_batch_table_checked(&[x], &[y], nbits, 100, i, &mbx).map(|_| ())
        }));
    }
    let results: Vec<Checked<()>> = joins
        .into_iter()
        .map(|j| j.join().expect("party panicked"))
        .collect();
    // The evaluator is the party that holds both payloads and does the check.
    let abort = results[2]
        .as_ref()
        .expect_err("a substituted circuit must be caught by the evaluator");
    assert_eq!(abort.reason, AbortReason::GarbleMismatch);
}

/// The verifier must see every multiplication. A path that bypasses
/// `mul_batch_checked` would silently leave products unverified, which is the
/// most dangerous possible regression here because nothing else would fail.
#[test]
fn every_multiplication_reaches_the_verifier() {
    let handles = InprocHandle::create(3);
    let mut joins = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        joins.push(std::thread::spawn(move || {
            let mbx = Mailbox::new(h);
            let keys = mbx.prf_keys(i);
            let mut v = Verifier::new(i, &keys);
            let key = v.key().clone();
            let zero = PartyShare { left: 0, right: 0 };
            let x = MalShare {
                val: zero.clone(),
                mac: zero.clone(),
            };
            let pairs: Vec<(MalShare, MalShare)> = (0..5).map(|_| (x.clone(), x.clone())).collect();
            let _ = mul_batch_checked(&pairs, 100, i, &mbx, &mut v);
            let _ = share_public(3, i, &key); // local: must not register
            v.product_count()
        }));
    }
    for j in joins {
        assert_eq!(
            j.join().expect("party panicked"),
            5,
            "verifier must hold exactly one entry per multiplication"
        );
    }
}
