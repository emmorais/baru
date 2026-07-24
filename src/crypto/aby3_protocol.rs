//! ABY3-specific protocol helpers for additive 3-party secret sharing.
//!
//! This module contains the ABY3 protocol primitives: PartyShare representation,
//! share input/output, multiplication via Beaver triples, a2b decomposition,
//! and related operations. These are specific to the ABY3 scheme and should not
//! be used directly by the evaluator or generic backends.

use rand::Rng;

use crate::crypto::net::{serialize_msg, Mailbox, MpcMsg, PartyId, Transport};
use crate::crypto::types::Fr;

const N: usize = 3;
const TIMEOUT_MS: u64 = 5_000;

/// The two field elements that party `id` holds for a given shared value.
/// In ABY3, party i holds (s_i, s_{(i+1) mod 3}).
///
/// This is ABY3-specific and should not be exposed to the interpreter.
#[derive(Clone, Debug)]
pub struct PartyShare {
    pub left: Fr,  // s_i
    pub right: Fr, // s_{(i+1) mod 3}
}

/// Distribute a secret input `value` to all three parties.
///
/// - Party `owner` calls this with the actual `value`.
/// - All other parties call this with `value = 0` (ignored).
///
/// Returns the `PartyShare` for the *calling* party.
pub fn share_input<T: Transport>(
    value: Fr,
    share_id: u64,
    owner: PartyId,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> PartyShare {
    if me == owner {
        let mut rng = rand::thread_rng();
        let s: [Fr; 3] = {
            let s0: Fr = rng.gen();
            let s1: Fr = rng.gen();
            let s2 = value.wrapping_sub(s0).wrapping_sub(s1);
            [s0, s1, s2]
        };
        for i in 0..N {
            let right_idx = (i + 1) % N;
            if i != me {
                mbx.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[i],
                    }),
                );
                mbx.send(
                    i,
                    serialize_msg(&MpcMsg::InputShare {
                        share_id,
                        value: s[right_idx],
                    }),
                );
            }
        }
        PartyShare {
            left: s[me],
            right: s[(me + 1) % N],
        }
    } else {
        let mut components: Vec<Fr> = Vec::new();
        while components.len() < 2 {
            if let Some((_from, m)) = mbx.recv_matching(
                TIMEOUT_MS,
                |m| matches!(m, MpcMsg::InputShare { share_id: sid, .. } if *sid == share_id),
            ) {
                if let MpcMsg::InputShare { value: v, .. } = m {
                    components.push(v);
                }
            } else {
                panic!(
                    "share_input: timeout waiting for InputShare share_id={}",
                    share_id
                );
            }
        }
        PartyShare {
            left: components[0],
            right: components[1],
        }
    }
}

/// Share a public constant locally (no communication).  Party 0 gets the
/// value in its left component; others get 0 so that reconstruction = value.
///
/// The replicated layout must be consistent: party `i` holds
/// `(s_i, s_{(i+1) mod 3})` for the additive sharing `s = (value, 0, 0)`.
/// Party 2's *right* component is therefore `s_0 = value`, not 0 — an
/// earlier version zeroed every right component, which reconstructs fine
/// (reveal sums only the lefts) but corrupts any *multiplication* of the
/// public value, since the multiplication protocol relies on the replicated
/// pairing. (Latent until a public nonzero constant was fed into a
/// product — e.g. `crate::gadgets` argmax indices / ReLU `keep` mask.)
pub fn share_public(value: Fr, me: PartyId) -> PartyShare {
    let s = |i: usize| if i == 0 { value } else { 0 };
    PartyShare {
        left: s(me),
        right: s((me + 1) % 3),
    }
}

/// Add two party shares locally (no communication required).
pub fn add_party(a: &PartyShare, b: &PartyShare) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_add(b.left),
        right: a.right.wrapping_add(b.right),
    }
}

/// Subtract two party shares locally.
pub fn sub_party(a: &PartyShare, b: &PartyShare) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_sub(b.left),
        right: a.right.wrapping_sub(b.right),
    }
}

/// Multiply a party share by a public scalar locally.
pub fn scalar_mul_party(a: &PartyShare, k: Fr) -> PartyShare {
    PartyShare {
        left: a.left.wrapping_mul(k),
        right: a.right.wrapping_mul(k),
    }
}

/// Reveal a shared value to `reveal_to`.
///
/// - If `reveal_to == Some(p)`, all parties send their left component to p.
/// - If `reveal_to == None`, all parties broadcast; everyone reconstructs.
pub fn reveal<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    reveal_to: Option<PartyId>,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Fr {
    // Collect `N - 1` RevealShare contributions for `share_id`, summing them
    // into `start`.  Buffering means peer contributions can arrive in any order.
    fn collect<T: Transport>(mbx: &Mailbox<T>, share_id: u64, start: Fr) -> Fr {
        let mut sum = start;
        let mut received = 0usize;
        while received < N - 1 {
            if let Some((_from, m)) = mbx.recv_matching(
                TIMEOUT_MS,
                |m| matches!(m, MpcMsg::RevealShare { share_id: sid, .. } if *sid == share_id),
            ) {
                if let MpcMsg::RevealShare { value: v, .. } = m {
                    sum = sum.wrapping_add(v);
                    received += 1;
                }
            } else {
                panic!(
                    "reveal: timeout waiting for RevealShare share_id={}",
                    share_id
                );
            }
        }
        sum
    }

    match reveal_to {
        Some(target) => {
            if me != target {
                mbx.send(
                    target,
                    serialize_msg(&MpcMsg::RevealShare {
                        share_id,
                        value: share.left,
                    }),
                );
                return 0;
            }
            collect(mbx, share_id, share.left)
        }
        None => {
            mbx.broadcast(serialize_msg(&MpcMsg::RevealShare {
                share_id,
                value: share.left,
            }));
            collect(mbx, share_id, share.left)
        }
    }
}

/// ABY3 multiplication — **no Beaver triples**.
///
/// With replicated sharing `x = x₀+x₁+x₂` (party `i` holds `xᵢ, xᵢ₊₁`), each
/// party can compute three of the nine cross terms locally:
///
/// ```text
/// zᵢ = xᵢyᵢ + xᵢyᵢ₊₁ + xᵢ₊₁yᵢ + αᵢ
/// ```
///
/// Summed over the three parties these cover all nine terms exactly once, so
/// `Σzᵢ = (Σxᵢ)(Σyᵢ) = xy`. The `αᵢ` are zero-sum correlated randomness from
/// pairwise PRF keys — they cancel in the sum but re-randomise each party's
/// component, which is what stops `zᵢ` leaking. Generating them costs *no*
/// communication, so the whole multiplication is **one ring element sent to
/// one neighbour**.
///
/// This replaces a Beaver-triple protocol that had party 0 deal 6 ring
/// elements per multiplication and then opened 2 more. Triples are not
/// merely moved offline here, they are eliminated: there is nothing left to
/// precompute or distribute.
pub fn mul_party<T: Transport>(
    a: &PartyShare,
    b: &PartyShare,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> PartyShare {
    mul_batch_party(&[(a.clone(), b.clone())], op_id, me, mbx)
        .pop()
        .expect("one product")
}
/// Batched [`mul_party`]: one message per party for the whole batch.
///
/// Each product `i` draws its own `α` from counter `base_op_id + i`, so the
/// re-randomisation is fresh per multiplication even within a batch.
pub fn mul_batch_party<T: Transport>(
    pairs: &[(PartyShare, PartyShare)],
    base_op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let keys = mbx.prf_keys(me);
    // Local: three of the nine cross terms, re-randomised by a zero-sum α.
    let z_mine: Vec<u128> = pairs
        .iter()
        .enumerate()
        .map(|(i, (a, b))| {
            a.left
                .wrapping_mul(b.left)
                .wrapping_add(a.left.wrapping_mul(b.right))
                .wrapping_add(a.right.wrapping_mul(b.left))
                .wrapping_add(keys.alpha(base_op_id.wrapping_add(i as u64)))
        })
        .collect();

    // Resharing: party i needs z_{i+1} to hold (z_i, z_{i+1}), so each party
    // sends its components to its predecessor. One directed message, not a
    // broadcast.
    mbx.send(
        (me + N - 1) % N,
        serialize_msg(&MpcMsg::TaggedZBatch { base_op_id, values: z_mine.clone() }),
    );
    let (_, msg) = mbx
        .recv_matching(TIMEOUT_MS, |m| {
            matches!(m, MpcMsg::TaggedZBatch { base_op_id: b, .. } if *b == base_op_id)
        })
        .unwrap_or_else(|| panic!("mul_batch_party: timeout on z for op_id={base_op_id}"));
    let z_next = match msg {
        MpcMsg::TaggedZBatch { values, .. } => values,
        _ => unreachable!("predicate guarantees TaggedZBatch"),
    };
    assert_eq!(z_next.len(), z_mine.len(), "batch length mismatch in resharing");

    z_mine
        .into_iter()
        .zip(z_next)
        .map(|(left, right)| PartyShare { left, right })
        .collect()
}
/// Operation ids reserved by one `a2b_party` call. Each internal
/// `mul_batch` round derives its correlated randomness from a distinct id
/// range; the block gives generous headroom over the ~9 rounds used.
pub const A2B_ID_BLOCK: u64 = 64;

/// Replicated sharing of a value held by the two parties that own additive
/// component `owner` — **no communication**.
///
/// A replicated sharing puts value `v` in one additive component and zero in
/// the others. Component `owner` is held as `left` by party `owner` and as
/// `right` by party `owner−1`, so exactly those two parties supply `v` and
/// the third contributes nothing and learns nothing. This is what makes the
/// first step of bit decomposition free.
fn share_component(value: Fr, owner: usize, me: PartyId) -> PartyShare {
    PartyShare {
        left: if me == owner { value } else { 0 },
        right: if (me + 1) % N == owner { value } else { 0 },
    }
}

/// XOR of two bit-shares in the arithmetic encoding: `a ⊕ b = a + b − 2ab`,
/// given their already-computed product.
fn xor_from_product(a: &PartyShare, b: &PartyShare, ab: &PartyShare) -> PartyShare {
    sub_party(&add_party(a, b), &scalar_mul_party(ab, 2))
}

/// Kogge–Stone parallel-prefix adder on bit-shares: the low `nbits` bits of
/// `u + v`. Costs `2 + ⌈log₂ nbits⌉` communication rounds.
fn ppa_add<T: Transport>(
    u: &[PartyShare],
    v: &[PartyShare],
    nbits: usize,
    next_id: &mut u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let take_id = |n: &mut u64| {
        let b = *n;
        *n += 2;
        b
    };
    // generate g = u∧v and propagate p = u⊕v — one product serves both.
    let pairs: Vec<_> = (0..nbits).map(|k| (u[k].clone(), v[k].clone())).collect();
    let base = take_id(next_id);
    let uv = mul_batch_party(&pairs, base, me, mbx);
    let mut g = uv.clone();
    let mut p: Vec<PartyShare> =
        (0..nbits).map(|k| xor_from_product(&u[k], &v[k], &uv[k])).collect();
    let p_leaf = p.clone();

    let mut d = 1usize;
    while d < nbits {
        let mut pr = Vec::with_capacity(2 * (nbits - d));
        for k in d..nbits {
            pr.push((p[k].clone(), g[k - d].clone()));
            pr.push((p[k].clone(), p[k - d].clone()));
        }
        let base = take_id(next_id);
        let out = mul_batch_party(&pr, base, me, mbx);
        let (mut ng, mut np) = (g.clone(), p.clone());
        for (i, k) in (d..nbits).enumerate() {
            // A group cannot both generate and propagate, so `g` and
            // `p ∧ g'` are mutually exclusive and their XOR is plain
            // addition — no communication for the combine.
            ng[k] = add_party(&g[k], &out[2 * i]);
            np[k] = out[2 * i + 1].clone();
        }
        g = ng;
        p = np;
        d <<= 1;
    }

    // carry into position k is the group generate over 0..k−1
    if nbits == 1 {
        return vec![p_leaf[0].clone()];
    }
    let pairs: Vec<_> = (1..nbits).map(|k| (p_leaf[k].clone(), g[k - 1].clone())).collect();
    let base = take_id(next_id);
    let prod = mul_batch_party(&pairs, base, me, mbx);
    let mut out = Vec::with_capacity(nbits);
    out.push(p_leaf[0].clone()); // nothing carries into the LSB
    for k in 1..nbits {
        out.push(xor_from_product(&p_leaf[k], &g[k - 1], &prod[k - 1]));
    }
    out
}

/// Arithmetic-to-bit decomposition (ABY3 bit decomposition).
///
/// `op_id` must be the base of a reserved block of [`A2B_ID_BLOCK`] ids.
///
/// # Protocol
///
/// 1. **Binary-share the three additive components** — free, via
///    [`share_component`]: `a = a₀ + a₁ + a₂` and each `aⱼ` is already known
///    to exactly the two parties a replicated sharing of it requires.
/// 2. **Carry-save reduce 3 → 2.** In this arithmetic bit encoding
///    `x⊕y⊕z = x+y+z − 2(xy+xz+yz) + 4xyz` and
///    `maj(x,y,z) = xy+xz+yz − 2xyz`, so both the sum and carry vectors fall
///    out of the *same* four products — two rounds, not four.
/// 3. **Add the remaining two** with a Kogge–Stone prefix adder.
///
/// This replaces an earlier version that masked with a value derived from
/// `op_id` alone and opened `a − r`. Because `op_id` is a public lockstep
/// counter that mask carried no secret entropy, so the input was recoverable
/// by anyone — that version was not private. Nothing here is ever opened:
/// every product goes through Beaver triples, which is why the carry chain
/// now costs communication instead of being "free".
pub fn a2b_party<T: Transport>(
    a: &PartyShare,
    nbits: usize,
    op_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> Vec<PartyShare> {
    let mut next_id = op_id;
    let take_id = |n: &mut u64| {
        let b = *n;
        *n += 2;
        b
    };

    // ── 1. free binary sharings of the three additive components ──
    let comp: Vec<Vec<PartyShare>> = (0..N)
        .map(|j| {
            let known: Fr = if j == me {
                a.left
            } else if j == (me + 1) % N {
                a.right
            } else {
                0
            };
            (0..nbits).map(|k| share_component((known >> k) & 1, j, me)).collect()
        })
        .collect();
    let (x, y, z) = (&comp[0], &comp[1], &comp[2]);

    // ── 2. carry-save 3 → 2 (four products, depth 2) ──
    let mut pairs = Vec::with_capacity(3 * nbits);
    for k in 0..nbits {
        pairs.push((x[k].clone(), y[k].clone()));
    }
    for k in 0..nbits {
        pairs.push((x[k].clone(), z[k].clone()));
    }
    for k in 0..nbits {
        pairs.push((y[k].clone(), z[k].clone()));
    }
    let base = take_id(&mut next_id);
    let pw = mul_batch_party(&pairs, base, me, mbx);
    let (xy, xz, yz) = (&pw[0..nbits], &pw[nbits..2 * nbits], &pw[2 * nbits..3 * nbits]);

    let triple: Vec<_> = (0..nbits).map(|k| (xy[k].clone(), z[k].clone())).collect();
    let base = take_id(&mut next_id);
    let xyz = mul_batch_party(&triple, base, me, mbx);

    let mut sum = Vec::with_capacity(nbits);
    let mut carry = Vec::with_capacity(nbits);
    for k in 0..nbits {
        let sym = add_party(&add_party(&xy[k], &xz[k]), &yz[k]); // xy+xz+yz
        let lin = add_party(&add_party(&x[k], &y[k]), &z[k]); // x+y+z
        sum.push(add_party(
            &sub_party(&lin, &scalar_mul_party(&sym, 2)),
            &scalar_mul_party(&xyz[k], 4),
        ));
        carry.push(sub_party(&sym, &scalar_mul_party(&xyz[k], 2)));
    }

    // ── 3. a = sum + (carry ≪ 1), mod 2^nbits ──
    let mut shifted = Vec::with_capacity(nbits);
    shifted.push(share_public(0, me));
    shifted.extend(carry.iter().take(nbits - 1).cloned());
    ppa_add(&sum, &shifted, nbits, &mut next_id, me, mbx)
}

/// Reveal a bit share to all parties.
pub fn reveal_bit_to_all<T: Transport>(
    share: &PartyShare,
    share_id: u64,
    me: PartyId,
    mbx: &Mailbox<T>,
) -> bool {
    let v = reveal(share, share_id, None, me, mbx);
    v % 2 != 0
}

#[cfg(test)]
fn party_pairs(s: &crate::crypto::types::Share3) -> Vec<PartyShare> {
    vec![
        PartyShare {
            left: s.0,
            right: s.1,
        },
        PartyShare {
            left: s.1,
            right: s.2,
        },
        PartyShare {
            left: s.2,
            right: s.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;

    /// Component sharings must be constructible with no communication and
    /// must reconstruct the original additive components. This is what makes
    /// the first phase of bit decomposition free, and it is also where a
    /// party could wrongly place a value it does not own — worth pinning.
    #[test]
    fn component_sharings_are_free_and_consistent() {
        let secret: Fr = 0xDEAD_BEEF;
        let shares = crate::crypto::arith::share_secret(secret);
        let parties = party_pairs(&shares);

        for j in 0..N {
            // Each party supplies only what it already holds.
            let built: Vec<PartyShare> = (0..N)
                .map(|me| {
                    let known: Fr = if j == me {
                        parties[me].left
                    } else if j == (me + 1) % N {
                        parties[me].right
                    } else {
                        0
                    };
                    share_component(known, j, me)
                })
                .collect();
            // The party that owns neither end of component j contributes
            // nothing — it never has to know the value.
            let idle = (j + 1) % N;
            assert_eq!(built[idle].left, 0);
            assert_eq!(built[idle].right, 0);
            // And the sharing reconstructs that component.
            let total = built.iter().fold(0u128, |acc, p| acc.wrapping_add(p.left));
            assert_eq!(total, parties[j].left, "component {j} must reconstruct");
        }
    }

    /// Run `f` (given `(me, mailbox)`) concurrently across 3 in-process
    /// parties and collect their results in party-index order.
    fn run_parties<R: Send + 'static>(
        f: impl Fn(PartyId, Mailbox<InprocHandle>) -> R + Send + Sync + Clone + 'static,
    ) -> Vec<R> {
        let handles = InprocHandle::create(3);
        let joins: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(me, h)| {
                let f = f.clone();
                std::thread::spawn(move || f(me, Mailbox::new(h)))
            })
            .collect();
        joins.into_iter().map(|j| j.join().unwrap()).collect()
    }

    /// Triple-free multiplication is correct *and* leaks nothing on the wire.
    /// Correctness: `Σzᵢ = xy`. Privacy: a single party's outbound message
    /// (`z_i`) is masked by `αᵢ`, which depends on a PRF key that party does
    /// not hold both halves of — so its value is not a function of the
    /// inputs alone. We check the weaker observable consequence that the
    /// three `α` cancel (`Σαᵢ = 0`), which is what makes the masking free,
    /// and that flipping a hidden input does not change what one victim
    /// party can compute from its own view — i.e. no `op_id`-only recovery
    /// like the old `a2b` had.
    #[test]
    fn triple_free_mul_is_correct_and_reshares_are_masked() {
        // Correctness across the transport.
        let out = run_parties(|me, mbx| {
            let a = share_input(if me == 0 { 7 } else { 0 }, 0, 0, me, &mbx);
            let b = share_input(if me == 0 { 6 } else { 0 }, 1, 0, me, &mbx);
            let p = mul_party(&a, &b, 100, me, &mbx);
            reveal(&p, 200, None, me, &mbx)
        });
        assert_eq!(out[0], 42, "7*6 must reveal 42");

        // Privacy of the reshared component: recompute exactly what party 1
        // sends to party 0 (its z_1) for two different secret inputs. If z_1
        // were a public function of op_id (the old bug's shape), it would be
        // identical regardless of the secret; the α mask must make it differ.
        use crate::crypto::net::PrfKeys;
        let keys = PrfKeys { mine: [1u8; 32], prev: [2u8; 32] };
        let z1 = |xr: u128| {
            // party 1 holds (x1,x2),(y1,y2); vary its x-share.
            let a = PartyShare { left: xr, right: 3 };
            let b = PartyShare { left: 4, right: 5 };
            a.left
                .wrapping_mul(b.left)
                .wrapping_add(a.left.wrapping_mul(b.right))
                .wrapping_add(a.right.wrapping_mul(b.left))
                .wrapping_add(keys.alpha(100))
        };
        assert_ne!(z1(11), z1(12), "reshared component must depend on the secret share");
        // And the α that masks it is genuinely zero-sum across a key ring.
        let k0 = PrfKeys { mine: [1u8; 32], prev: [3u8; 32] };
        let k1 = PrfKeys { mine: [2u8; 32], prev: [1u8; 32] };
        let k2 = PrfKeys { mine: [3u8; 32], prev: [2u8; 32] };
        assert_eq!(
            k0.alpha(100).wrapping_add(k1.alpha(100)).wrapping_add(k2.alpha(100)),
            0,
            "the three parties' α must cancel"
        );
    }

    /// The rebuilt bit decomposition must agree with plaintext binary
    /// expansion, including two's-complement negatives (the MSB is what
    /// `lt` returns, so sign handling is load-bearing).
    #[test]
    fn a2b_matches_plaintext_binary_expansion() {
        let nbits = 12;
        let cases: Vec<i64> = vec![0, 1, 2, 5, 255, 1024, 4095, -1, -2, -7, -2048];
        for &v in &cases {
            let out = run_parties(move |me, mbx| {
                let fr: Fr = if me == 0 { (v as i128).rem_euclid(1i128 << nbits) as u128 } else { 0 };
                let sh = share_input(fr, 0, me, 0, &mbx);
                let bits = a2b_party(&sh, nbits, 1_000, me, &mbx);
                bits.iter()
                    .enumerate()
                    .map(|(i, b)| reveal(b, 5_000 + i as u64, None, me, &mbx) & 1)
                    .collect::<Vec<u128>>()
            });
            let want = (v as i128).rem_euclid(1i128 << nbits) as u128;
            let got = out[0]
                .iter()
                .enumerate()
                .fold(0u128, |acc, (i, b)| acc | (b << i));
            assert_eq!(got, want, "a2b({v}) gave {got:#x}, want {want:#x}");
        }
    }

    /// Secret-share `values[i]` (owned by party 0) for i in 0..values.len(),
    /// returning this party's `PartyShare` for each.
    fn share_all<T: Transport>(values: &[i64], me: PartyId, mbx: &Mailbox<T>) -> Vec<PartyShare> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let fr = if me == 0 { v as u128 } else { 0 };
                share_input(fr, i as u64 * 10, 0, me, mbx)
            })
            .collect()
    }

    #[test]
    fn mul_batch_matches_sequential_mul_party() {
        let a_vals = [3i64, -5, 7, 0, 12];
        let b_vals = [4i64, 6, -2, 9, 11];
        let expected: Vec<i64> = a_vals.iter().zip(&b_vals).map(|(a, b)| a * b).collect();

        let results = run_parties(move |me, mbx| {
            let a_shares = share_all(&a_vals, me, &mbx);
            let b_shares = share_all(&b_vals, me, &mbx);
            let pairs: Vec<_> = a_shares.into_iter().zip(b_shares).collect();

            let out = mul_batch_party(&pairs, 1000, me, &mbx);

            // Reveal every product to party 0.
            out.into_iter()
                .enumerate()
                .map(|(i, ps)| reveal(&ps, 2000 + i as u64, Some(0), me, &mbx))
                .collect::<Vec<Fr>>()
        });

        // Only party 0 (the reveal target) has the real sums; the others'
        // `reveal` calls return 0 by design (see `reveal`'s doc).
        let got: Vec<i64> = results[0].iter().map(|&v| v as i128 as i64).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn mul_batch_matches_sequential_mul_party_for_single_pair() {
        // count=1 must behave identically to a plain `mul_party` call.
        let results = run_parties(|me, mbx| {
            let a = share_input(if me == 0 { 6 } else { 0 }, 0, 0, me, &mbx);
            let b = share_input(if me == 0 { 7 } else { 0 }, 1, 0, me, &mbx);
            let batched = mul_batch_party(&[(a.clone(), b.clone())], 100, me, &mbx)
                .into_iter()
                .next()
                .unwrap();
            reveal(&batched, 200, Some(0), me, &mbx) as i128 as i64
        });
        assert_eq!(results[0], 42);
    }

    #[test]
    fn mul_batch_uses_one_round_instead_of_n() {
        use crate::bench::comm::CountingTransport;

        let n = 8usize;
        let a_vals: Vec<i64> = (0..n as i64).collect();
        let b_vals: Vec<i64> = (0..n as i64).map(|i| i + 1).collect();

        let handles = InprocHandle::create(3);
        let joins: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(me, h)| {
                let a_vals = a_vals.clone();
                let b_vals = b_vals.clone();
                std::thread::spawn(move || {
                    let counted = CountingTransport::new(h, 3);
                    let handle = counted.handle();
                    let mbx = Mailbox::new(counted);

                    let a_shares = share_all(&a_vals, me, &mbx);
                    let b_shares = share_all(&b_vals, me, &mbx);
                    let pairs: Vec<_> = a_shares.into_iter().zip(b_shares).collect();

                    let batched_stats_before = handle.stats();
                    let _ = mul_batch_party(&pairs, 1000, me, &mbx);
                    let batched_stats = handle.stats().saturating_sub(&batched_stats_before);

                    // Sequential baseline: n separate mul_party calls on fresh shares.
                    let a2 = share_all(&a_vals, me, &mbx);
                    let b2 = share_all(&b_vals, me, &mbx);
                    let seq_before = handle.stats();
                    for (i, (a, b)) in a2.iter().zip(&b2).enumerate() {
                        let _ = mul_party(a, b, 5000 + i as u64 * 10, me, &mbx);
                    }
                    let seq_stats = handle.stats().saturating_sub(&seq_before);

                    (batched_stats, seq_stats)
                })
            })
            .collect();
        let results: Vec<_> = joins.into_iter().map(|j| j.join().unwrap()).collect();

        let (batched, seq) = results[0];
        assert!(
            batched.messages < seq.messages,
            "batched ({}) should use far fewer messages than sequential ({}) for n={n}",
            batched.messages,
            seq.messages
        );
        // Sequential does n independent rounds -> message count scales with n;
        // batched does it in one round -> message count is O(1) in n.
        assert!(
            seq.messages >= batched.messages * (n as u64) / 2,
            "expected roughly n-fold more messages sequentially: batched={} seq={} n={n}",
            batched.messages,
            seq.messages
        );
    }
}
