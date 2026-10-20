//! Element-wise **vector** builtins — the batched primitives, reachable from
//! the language.
//!
//! # Why these exist
//!
//! [`crate::interpreter::backend::Backend`] exposes `mul_batch`, `lt_batch` and
//! `a2b_batch`, which run *any* number of independent lanes in the round count
//! of one. `LANGUAGE.md` §6.2 states the consequence — "cost is rounds, not
//! operations" — and warns that calling the per-element form in a loop pays
//! `O(n)` rounds for nothing. Until now that warning was addressed only to Rust
//! code, because the language had no way to reach the batched forms: `dot`,
//! `vsum` and `matvec` were the entire vector surface, and none of them covers a
//! comparison or a shift.
//!
//! Measured cost of the omission: `examples/medmnist.baru` — a CNN whose layers
//! were written with scalar `>`/`ash-right` — spent **864 361 messages** on one
//! diagnosis where the Rust pipeline computing the same network spends **61**,
//! because ~12 600 comparisons each paid their own ~9 rounds instead of one
//! round per layer (`docs/experiments.md` IV.2b).
//!
//! # What they are, and what they are not
//!
//! These are *primitives*, not gadgets: element-wise arithmetic, comparison,
//! shift, max and a strided view. Deliberately no `conv2d`, `relu` or `maxpool`
//! builtin — those are the network, and the network stays written in Baru. The
//! relationship is NumPy's to Python: the loops move into the primitive, the
//! program does not move into Rust.
//!
//! | Builtin | Rounds on shares |
//! |---|---|
//! | `vadd` `vsub` | 0 — local |
//! | `vscale` | 0 for a public scalar; 1 (`mul_batch`) for a secret one |
//! | `vmul` | 1 (`mul_batch`), or 0 if one side is public |
//! | `vlt` `vge` | 1 (`lt_batch`) |
//! | `vmax` | 2 (`lt_batch` + `mul_batch`) |
//! | `vashr` | 1 batched `a2b`, then local recomposition |
//! | `vrelu` | the fused ReLU∘rescale — **1 garble round** on `--nonlin gc` |
//! | `vslice` `vrep` | 0 — data movement |
//!
//! Every one also has a plain-integer path, so the same source text runs on the
//! plain interpreter; that is what makes a plaintext-vs-private comparison of
//! one program possible at all.

use super::ast::{ErrObj, Types};
use super::backend::{Backend, ShareRef};
use super::eval::{int_to_fr_signed, vec_add_pub, vec_mul_pub};
use std::sync::{Arc, Mutex};

type Be = Arc<Mutex<Box<dyn Backend>>>;

fn as_list(v: &Types, who: &str) -> Result<Vec<Types>, Types> {
    match v {
        Types::List(xs) => Ok(xs.clone()),
        Types::Vector(v) => Ok(v.borrow().clone()),
        other => Err(ErrObj::raise_with(
            format!("{who} expects a list or vector"),
            vec![other.clone()],
        )),
    }
}

fn same_len(a: &[Types], b: &[Types], who: &str) -> Result<(), Types> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(ErrObj::raise(format!(
            "{who}: length mismatch ({} vs {})",
            a.len(),
            b.len()
        )))
    }
}

fn all_shares(xs: &[Types]) -> bool {
    !xs.is_empty() && xs.iter().all(|t| matches!(t, Types::Share(_)))
}

fn all_ints(xs: &[Types]) -> bool {
    xs.iter().all(|t| matches!(t, Types::Num(_)))
}

fn refs(xs: &[Types]) -> Vec<ShareRef> {
    xs.iter()
        .map(|t| match t {
            Types::Share(s) => s.clone(),
            _ => unreachable!("checked by all_shares"),
        })
        .collect()
}

fn nums(xs: &[Types]) -> Vec<i64> {
    xs.iter()
        .map(|t| match t {
            Types::Num(n) => *n,
            _ => unreachable!("checked by all_ints"),
        })
        .collect()
}

fn wrap(v: Vec<ShareRef>) -> Types {
    Types::List(v.into_iter().map(Types::Share).collect())
}

/// The share width to compare or inject at: the widest operand present.
fn width(a: &[Types], b: &[Types]) -> usize {
    a.iter()
        .chain(b)
        .filter_map(|t| match t {
            Types::Share(s) => Some(s.nbits),
            _ => None,
        })
        .max()
        .unwrap_or(64)
}

/// Lift a mixed list to share refs, injecting public integers locally
/// (`share_public` costs nothing) so a batched primitive can take the whole lane.
fn lift(
    be: &mut dyn Backend,
    xs: &[Types],
    nbits: usize,
    who: &str,
) -> Result<Vec<ShareRef>, Types> {
    xs.iter()
        .map(|t| match t {
            Types::Share(s) => Ok(s.clone()),
            Types::Num(n) => Ok(be.share_public(int_to_fr_signed(*n, nbits), nbits)),
            other => Err(ErrObj::raise_with(
                format!("{who}: expects integers or shares"),
                vec![other.clone()],
            )),
        })
        .collect()
}

/// `xs >> n` element-wise, one batched `a2b` for the whole vector.
///
/// Mirrors [`Backend::shift_right`]'s semantics exactly — a *logical* shift on
/// the two's-complement residue, recomposed at the original width so downstream
/// signed comparisons size correctly (see `docs/experiments.md` I.2c, which is
/// what happens when they do not). Valid on the non-negative branch, which is
/// why the fused ReLU∘rescale multiplies by the sign mask rather than trusting
/// this on negatives.
fn vashr_shares(be: &mut dyn Backend, xs: &[ShareRef], n: usize) -> Vec<ShareRef> {
    if xs.is_empty() {
        return Vec::new();
    }
    let nbits = xs.iter().map(|s| s.nbits).max().unwrap_or(64);
    let bits = be.a2b_batch(xs, nbits);
    bits.iter()
        .zip(xs)
        .map(|(b, x)| {
            // One zero bit-share, reused for every vacated high position: `b2a`
            // only reads each component, so sharing the handle is safe.
            let zero = be.sub(x, x);
            let mut out: Vec<ShareRef> = Vec::with_capacity(nbits);
            out.extend(b.iter().skip(n).cloned());
            while out.len() < nbits {
                out.push(zero.clone());
            }
            be.b2a(&out)
        })
        .collect()
}

pub fn vector_builtins(backend: Be) -> Vec<(String, Types)> {
    let mut out: Vec<(String, Types)> = Vec::new();

    // ── vadd / vsub — local on shares ───────────────────────────────────────
    for (name, sub) in [("vadd", false), ("vsub", true)] {
        let be = backend.clone();
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                return ErrObj::raise(format!("{name} expects 2 args: ({name} a b)"));
            }
            let (a, b) = match (as_list(&args[0], name), as_list(&args[1], name)) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => return e,
            };
            if let Err(e) = same_len(&a, &b, name) {
                return e;
            }
            if all_ints(&a) && all_ints(&b) {
                let (x, y) = (nums(&a), nums(&b));
                return Types::List(
                    x.iter()
                        .zip(&y)
                        .map(|(p, q)| {
                            Types::Num(if sub {
                                p.wrapping_sub(*q)
                            } else {
                                p.wrapping_add(*q)
                            })
                        })
                        .collect(),
                );
            }
            let mut acc = Vec::with_capacity(a.len());
            for (x, y) in a.iter().zip(&b) {
                let y2 = if sub {
                    match vec_mul_pub(&be, y, &Types::Num(-1)) {
                        Ok(v) => v,
                        Err(e) => return e,
                    }
                } else {
                    y.clone()
                };
                match vec_add_pub(&be, x, &y2) {
                    Ok(v) => acc.push(v),
                    Err(e) => return e,
                }
            }
            Types::List(acc)
        });
        out.push((name.to_string(), Types::NativeFunc(f)));
    }

    // ── vscale: scale a vector by ONE scalar ────────────────────────────────
    //
    // Both cases matter and they differ by a factor of `len`. A **public**
    // scalar is `mul_public` per element: local, no rounds. A **secret** scalar
    // — which is what a convolution tap is once the model itself is shared — is
    // one `mul_batch` for the whole vector, broadcasting the scalar down the
    // lanes. Doing that per element instead is one round per activation, and on
    // a shared-weight convolution that was 48 672 rounds where 72 suffice.
    {
        let be = backend.clone();
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                return ErrObj::raise("vscale expects 2 args: (vscale vec k)");
            }
            let xs = match as_list(&args[0], "vscale") {
                Ok(v) => v,
                Err(e) => return e,
            };
            let k = &args[1];
            if let (Types::Share(ks), true) = (k, all_shares(&xs)) {
                if xs.is_empty() {
                    return Types::List(Vec::new());
                }
                let mut g = be.lock().unwrap_or_else(|e| e.into_inner());
                let pairs: Vec<(ShareRef, ShareRef)> =
                    refs(&xs).into_iter().map(|x| (x, ks.clone())).collect();
                return wrap(g.mul_batch(&pairs));
            }
            let mut acc = Vec::with_capacity(xs.len());
            for x in &xs {
                match vec_mul_pub(&be, x, k) {
                    Ok(v) => acc.push(v),
                    Err(e) => return e,
                }
            }
            Types::List(acc)
        });
        out.push(("vscale".to_string(), Types::NativeFunc(f)));
    }

    // ── vmul: one batched multiplication ────────────────────────────────────
    {
        let be = backend.clone();
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                return ErrObj::raise("vmul expects 2 args: (vmul a b)");
            }
            let (a, b) = match (as_list(&args[0], "vmul"), as_list(&args[1], "vmul")) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => return e,
            };
            if let Err(e) = same_len(&a, &b, "vmul") {
                return e;
            }
            if all_ints(&a) && all_ints(&b) {
                let (x, y) = (nums(&a), nums(&b));
                return Types::List(
                    x.iter()
                        .zip(&y)
                        .map(|(p, q)| Types::Num(p.wrapping_mul(*q)))
                        .collect(),
                );
            }
            // Both secret: one round for the whole vector.
            if all_shares(&a) && all_shares(&b) {
                let mut g = be.lock().unwrap_or_else(|e| e.into_inner());
                let pairs: Vec<(ShareRef, ShareRef)> = refs(&a).into_iter().zip(refs(&b)).collect();
                return wrap(g.mul_batch(&pairs));
            }
            // Mixed: share×public is local per element, so no batching is owed.
            let mut acc = Vec::with_capacity(a.len());
            for (x, y) in a.iter().zip(&b) {
                match vec_mul_pub(&be, x, y) {
                    Ok(v) => acc.push(v),
                    Err(e) => return e,
                }
            }
            Types::List(acc)
        });
        out.push(("vmul".to_string(), Types::NativeFunc(f)));
    }

    // ── vlt / vge: one batched comparison ───────────────────────────────────
    for (name, negate) in [("vlt", false), ("vge", true)] {
        let be = backend.clone();
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                return ErrObj::raise(format!("{name} expects 2 args: ({name} a b)"));
            }
            let (a, b) = match (as_list(&args[0], name), as_list(&args[1], name)) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => return e,
            };
            if let Err(e) = same_len(&a, &b, name) {
                return e;
            }
            if all_ints(&a) && all_ints(&b) {
                let (x, y) = (nums(&a), nums(&b));
                return Types::List(
                    x.iter()
                        .zip(&y)
                        .map(|(p, q)| {
                            Types::Num(if negate {
                                (p >= q) as i64
                            } else {
                                (p < q) as i64
                            })
                        })
                        .collect(),
                );
            }
            let nbits = width(&a, &b);
            let mut g = be.lock().unwrap_or_else(|e| e.into_inner());
            let (ra, rb) = match (
                lift(&mut **g, &a, nbits, name),
                lift(&mut **g, &b, nbits, name),
            ) {
                (Ok(x), Ok(y)) => (x, y),
                (Err(e), _) | (_, Err(e)) => return e,
            };
            let pairs: Vec<(ShareRef, ShareRef)> = ra.into_iter().zip(rb).collect();
            let lt = g.lt_batch(&pairs);
            if !negate {
                return wrap(lt);
            }
            // `a >= b` is `1 - (a < b)`; the subtraction is local.
            let one = g.share_public(1, nbits);
            wrap(lt.iter().map(|c| g.sub(&one, c)).collect())
        });
        out.push((name.to_string(), Types::NativeFunc(f)));
    }

    // ── vmax: element-wise maximum, two batched rounds ──────────────────────
    {
        let be = backend.clone();
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                return ErrObj::raise("vmax expects 2 args: (vmax a b)");
            }
            let (a, b) = match (as_list(&args[0], "vmax"), as_list(&args[1], "vmax")) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => return e,
            };
            if let Err(e) = same_len(&a, &b, "vmax") {
                return e;
            }
            if all_ints(&a) && all_ints(&b) {
                let (x, y) = (nums(&a), nums(&b));
                return Types::List(
                    x.iter()
                        .zip(&y)
                        .map(|(p, q)| Types::Num(*p.max(q)))
                        .collect(),
                );
            }
            let nbits = width(&a, &b);
            let mut g = be.lock().unwrap_or_else(|e| e.into_inner());
            let (ra, rb) = match (
                lift(&mut **g, &a, nbits, "vmax"),
                lift(&mut **g, &b, nbits, "vmax"),
            ) {
                (Ok(x), Ok(y)) => (x, y),
                (Err(e), _) | (_, Err(e)) => return e,
            };
            // Ties keep `a`, matching gadgets::max2_batch and maxpool2d_ref.
            wrap(crate::gadgets::max2_batch(&mut **g, &ra, &rb))
        });
        out.push(("vmax".to_string(), Types::NativeFunc(f)));
    }

    // ── vashr: element-wise shift, one batched a2b ──────────────────────────
    {
        let be = backend.clone();
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                return ErrObj::raise("vashr expects 2 args: (vashr vec n)");
            }
            let xs = match as_list(&args[0], "vashr") {
                Ok(v) => v,
                Err(e) => return e,
            };
            let n = match &args[1] {
                Types::Num(n) if *n >= 0 => *n as usize,
                other => {
                    return ErrObj::raise_with(
                        "vashr expects a non-negative shift",
                        vec![other.clone()],
                    )
                }
            };
            if all_ints(&xs) {
                return Types::List(
                    nums(&xs)
                        .iter()
                        .map(|v| Types::Num(v.wrapping_shr(n as u32)))
                        .collect(),
                );
            }
            if !all_shares(&xs) {
                return ErrObj::raise("vashr expects all integers or all shares");
            }
            let mut g = be.lock().unwrap_or_else(|e| e.into_inner());
            wrap(vashr_shares(&mut **g, &refs(&xs), n))
        });
        out.push(("vashr".to_string(), Types::NativeFunc(f)));
    }

    // ── vrelu: the fused ReLU∘rescale, at whatever protocol is configured ───
    //
    // `(vrelu xs shift)` == `(vmul (vashr xs shift) (vge xs zeros))` in value,
    // and not in cost: spelled out, it is a bit decomposition plus a batched
    // multiply; as one call it can be **one garble round for the whole layer**
    // (`--nonlin gc`) or Falcon's DReLU+truncation. This is the primitive the
    // hybrid pipeline's 268 ms rests on, and the reason it is a builtin rather
    // than a Lisp definition is that the protocol choice is not expressible in
    // the spelled-out form.
    {
        let be = backend.clone();
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            let (xs, shift) = match args.as_slice() {
                [v, Types::Num(n)] if *n >= 0 => match as_list(v, "vrelu") {
                    Ok(x) => (x, *n as usize),
                    Err(e) => return e,
                },
                _ => return ErrObj::raise("vrelu expects (vrelu vec shift) with shift >= 0"),
            };
            if xs.is_empty() {
                return Types::List(Vec::new());
            }
            if all_ints(&xs) {
                return Types::List(
                    nums(&xs)
                        .iter()
                        .map(|v| Types::Num(if *v >= 0 { v >> shift } else { 0 }))
                        .collect(),
                );
            }
            if !all_shares(&xs) {
                return ErrObj::raise("vrelu expects all integers or all shares");
            }
            let mut g = be.lock().unwrap_or_else(|e| e.into_inner());
            wrap(g.relu_rescale_batch(&refs(&xs), shift))
        });
        out.push(("vrelu".to_string(), Types::NativeFunc(f)));
    }

    // ── vslice: a 2-D strided view, flattened. Pure data movement ───────────
    //
    // `(vslice v base rows cols row-stride col-stride)` gathers
    // `v[base + r*row_stride + c*col_stride]` for r<rows, c<cols. Two strides
    // rather than one because both users need both: a convolution tap is a
    // contiguous-column window (`col-stride` 1) at row stride `iw`, and a
    // 2×2/stride-2 pooling phase is `col-stride` 2 at row stride `2*iw`.
    {
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            let (v, base, rows, cols, rs, cs) = match args.as_slice() {
                [v, Types::Num(b), Types::Num(r), Types::Num(c), Types::Num(rs), Types::Num(cs)] => {
                    (v, *b, *r, *c, *rs, *cs)
                }
                _ => {
                    return ErrObj::raise(
                        "vslice expects (vslice vec base rows cols row-stride col-stride)",
                    )
                }
            };
            let xs = match as_list(v, "vslice") {
                Ok(x) => x,
                Err(e) => return e,
            };
            if base < 0 || rows < 0 || cols < 0 || rs < 0 || cs < 0 {
                return ErrObj::raise("vslice: every argument must be non-negative");
            }
            let (base, rows, cols, rs, cs) = (
                base as usize,
                rows as usize,
                cols as usize,
                rs as usize,
                cs as usize,
            );
            let mut acc = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    let i = base + r * rs + c * cs;
                    match xs.get(i) {
                        Some(x) => acc.push(x.clone()),
                        None => {
                            return ErrObj::raise(format!(
                                "vslice: index {i} is past the end of a {}-element vector",
                                xs.len()
                            ))
                        }
                    }
                }
            }
            Types::List(acc)
        });
        out.push(("vslice".to_string(), Types::NativeFunc(f)));
    }

    // ── vrep: a constant vector. Needed for biases and for the zero vector a
    //    batched comparison takes as its other operand.
    {
        let f = Arc::new(move |args: Vec<Types>| -> Types {
            let (x, n) = match args.as_slice() {
                [x, Types::Num(n)] if *n >= 0 => (x.clone(), *n as usize),
                _ => return ErrObj::raise("vrep expects (vrep x n) with n >= 0"),
            };
            Types::List(vec![x; n])
        });
        out.push(("vrep".to_string(), Types::NativeFunc(f)));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::aby3_simulator::Aby3Simulator;

    fn backend() -> Be {
        Arc::new(Mutex::new(
            Box::new(Aby3Simulator::new()) as Box<dyn Backend>
        ))
    }

    fn call(bs: &[(String, Types)], name: &str, args: Vec<Types>) -> Types {
        let f = bs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
            .unwrap_or_else(|| panic!("{name} must be registered"));
        match f {
            Types::NativeFunc(g) => g(args),
            other => panic!("{name} is {other:?}"),
        }
    }

    fn pub_list(xs: &[i64]) -> Types {
        Types::List(xs.iter().map(|&v| Types::Num(v)).collect())
    }

    fn shared_list(be: &Be, xs: &[i64], nbits: usize) -> Types {
        let mut g = be.lock().unwrap();
        Types::List(
            xs.iter()
                .map(|&v| Types::Share(g.share_secret(int_to_fr_signed(v, nbits), nbits, 0)))
                .collect(),
        )
    }

    fn return_unsigned(raw: u128) -> i64 {
        raw as i64
    }

    fn reveal(be: &Be, t: &Types, nbits: usize) -> Vec<i64> {
        let mut g = be.lock().unwrap();
        match t {
            Types::List(xs) => xs
                .iter()
                .map(|x| match x {
                    Types::Num(n) => *n,
                    Types::Share(s) => {
                        let raw = g.reveal(s, None);
                        let w = s.nbits.min(nbits);
                        // A comparison result is a 1-bit share holding 0 or 1;
                        // sign-extending it would read 1 as -1. Only wider
                        // shares carry a two's-complement sign.
                        if w <= 1 {
                            return_unsigned(raw)
                        } else {
                            let m = 1i128 << w;
                            let v = (raw as i128).rem_euclid(m);
                            if v >= m / 2 {
                                (v - m) as i64
                            } else {
                                v as i64
                            }
                        }
                    }
                    other => panic!("unexpected element {other:?}"),
                })
                .collect(),
            other => panic!("expected a list, got {other:?}"),
        }
    }

    /// The whole point of these builtins is that they are the *same function* as
    /// the scalar path, only batched. Anything else is a silently wrong answer,
    /// so every one is checked against plain integer arithmetic on both the
    /// public and the secret path, including negative operands and ties.
    #[test]
    fn every_vector_builtin_matches_plain_integer_arithmetic() {
        const N: usize = 24;
        let a: Vec<i64> = vec![
            0, 1, -1, 5, -5, 255, -255, 1000, -1000, 3, 3, 7, 8, -8, 2, -2, 99, -99, 17, -17, 64,
            -64, 1, 0,
        ];
        let b: Vec<i64> = vec![
            0, -1, 1, 5, 5, -255, 255, 999, -999, 4, 2, 7, -8, 8, -2, 2, 100, -100, 16, -16, 63,
            -63, 0, 1,
        ];
        assert_eq!(a.len(), N);
        assert_eq!(b.len(), N);

        let be = backend();
        let bs = vector_builtins(be.clone());
        let nbits = 32usize;
        let sa = shared_list(&be, &a, nbits);
        let sb = shared_list(&be, &b, nbits);

        let cases: Vec<(&str, Vec<i64>)> = vec![
            ("vadd", a.iter().zip(&b).map(|(x, y)| x + y).collect()),
            ("vsub", a.iter().zip(&b).map(|(x, y)| x - y).collect()),
            ("vmul", a.iter().zip(&b).map(|(x, y)| x * y).collect()),
            (
                "vlt",
                a.iter().zip(&b).map(|(x, y)| (x < y) as i64).collect(),
            ),
            (
                "vge",
                a.iter().zip(&b).map(|(x, y)| (x >= y) as i64).collect(),
            ),
            ("vmax", a.iter().zip(&b).map(|(x, y)| *x.max(y)).collect()),
        ];

        for (name, want) in cases {
            let got_pub = reveal(
                &be,
                &call(&bs, name, vec![pub_list(&a), pub_list(&b)]),
                nbits,
            );
            assert_eq!(got_pub, want, "{name}: public path");
            let got_sec = reveal(&be, &call(&bs, name, vec![sa.clone(), sb.clone()]), nbits);
            assert_eq!(got_sec, want, "{name}: secret path");
            // Mixed operands must agree too — a public weight against a secret
            // activation is the common case in a convolution.
            let got_mix = reveal(&be, &call(&bs, name, vec![sa.clone(), pub_list(&b)]), nbits);
            assert_eq!(got_mix, want, "{name}: secret x public");
        }
    }

    /// `vashr` is the one that broke before (I.2c): the result must be usable as
    /// the operand of a later comparison, which means its width must survive.
    #[test]
    fn vashr_matches_the_shift_and_keeps_its_width() {
        let be = backend();
        let bs = vector_builtins(be.clone());
        let nbits = 32usize;
        // Non-negative only: the shift is defined on that branch, which is why
        // the fused ReLU multiplies by the sign mask instead of trusting it.
        let xs: Vec<i64> = vec![0, 1, 255, 256, 300, 1000, 5000, 65535, 100_000];
        let want: Vec<i64> = xs.iter().map(|v| v >> 8).collect();

        let got_pub = reveal(
            &be,
            &call(&bs, "vashr", vec![pub_list(&xs), Types::Num(8)]),
            nbits,
        );
        assert_eq!(got_pub, want, "public shift");

        let shifted = call(
            &bs,
            "vashr",
            vec![shared_list(&be, &xs, nbits), Types::Num(8)],
        );
        assert_eq!(reveal(&be, &shifted, nbits), want, "secret shift");

        match &shifted {
            Types::List(ys) => {
                for y in ys {
                    match y {
                        Types::Share(s) => assert_eq!(
                            s.nbits, nbits,
                            "a shifted share must keep its width or later comparisons \
                             run at the wrong one (experiments.md I.2c)"
                        ),
                        other => panic!("expected shares, got {other:?}"),
                    }
                }
            }
            other => panic!("expected a list, got {other:?}"),
        }

        // And the failing composition from I.2c: compare two shifted values.
        let cmp = call(
            &bs,
            "vge",
            vec![shifted.clone(), shared_list(&be, &want, nbits)],
        );
        assert_eq!(
            reveal(&be, &cmp, nbits),
            vec![1; xs.len()],
            "shifted >= itself"
        );
    }

    /// A secret scalar must batch, not loop — and must still be the same product.
    #[test]
    fn vscale_agrees_for_public_and_secret_scalars() {
        let be = backend();
        let bs = vector_builtins(be.clone());
        let nbits = 32usize;
        let xs: Vec<i64> = vec![1, -2, 3, -4, 5, 0, 255, -255];
        for k in [0i64, 1, -1, 7, -7] {
            let want: Vec<i64> = xs.iter().map(|v| v * k).collect();
            let got_pub = reveal(
                &be,
                &call(
                    &bs,
                    "vscale",
                    vec![shared_list(&be, &xs, nbits), Types::Num(k)],
                ),
                nbits,
            );
            assert_eq!(got_pub, want, "public scalar {k}");
            let ks = match shared_list(&be, &[k], nbits) {
                Types::List(mut v) => v.remove(0),
                other => panic!("{other:?}"),
            };
            let got_sec = reveal(
                &be,
                &call(&bs, "vscale", vec![shared_list(&be, &xs, nbits), ks]),
                nbits,
            );
            assert_eq!(got_sec, want, "secret scalar {k}");
        }
    }

    /// `vslice` is the strided view both the convolution and the pool are built
    /// on; an off-by-one in either stride is a wrong network that still runs.
    #[test]
    fn vslice_gathers_the_documented_strided_view() {
        let be = backend();
        let bs = vector_builtins(be.clone());
        let v: Vec<i64> = (0..24).collect();
        // A 3x3 conv tap over a 4x6 input at offset (1,1): rows stride 6, cols 1.
        let got = reveal(
            &be,
            &call(
                &bs,
                "vslice",
                vec![
                    pub_list(&v),
                    Types::Num(7),
                    Types::Num(2),
                    Types::Num(3),
                    Types::Num(6),
                    Types::Num(1),
                ],
            ),
            64,
        );
        assert_eq!(got, vec![7, 8, 9, 13, 14, 15]);
        // A stride-2 pooling phase: row stride 12, col stride 2.
        let got = reveal(
            &be,
            &call(
                &bs,
                "vslice",
                vec![
                    pub_list(&v),
                    Types::Num(0),
                    Types::Num(2),
                    Types::Num(3),
                    Types::Num(12),
                    Types::Num(2),
                ],
            ),
            64,
        );
        assert_eq!(got, vec![0, 2, 4, 12, 14, 16]);
    }

    #[test]
    fn length_mismatch_and_overrun_are_errors() {
        let be = backend();
        let bs = vector_builtins(be.clone());
        assert!(matches!(
            call(&bs, "vadd", vec![pub_list(&[1, 2]), pub_list(&[1])]),
            Types::Error(..)
        ));
        assert!(matches!(
            call(
                &bs,
                "vslice",
                vec![
                    pub_list(&[1, 2, 3]),
                    Types::Num(0),
                    Types::Num(2),
                    Types::Num(3),
                    Types::Num(3),
                    Types::Num(1)
                ]
            ),
            Types::Error(..)
        ));
    }
}
