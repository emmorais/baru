//! Depth-reducing emission for secure-computation programs.
//!
//! The cost of running a program under a secure-computation protocol is not
//! given by its operation count. Operations that do not communicate — addition,
//! subtraction, multiplication by a public constant — are free in the only
//! currency that matters on a wide-area link, and operations that do communicate
//! are paid at the round-trip time once per *level* of the dependence graph, not
//! once per operation, because independent operations at the same level are
//! batched into a single message.
//!
//! Two programs computing the same function can therefore differ in wall-clock
//! cost by a large factor while performing an identical number of operations, and
//! the difference is entirely the shape of the dependence graph. This module
//! collects the rewrites that change that shape.
//!
//! The observation is not specific to secret sharing. A fully homomorphic
//! compiler faces the same question when it decides where to place bootstrap
//! operations: the expensive resource is multiplicative depth, the program's
//! operation count is nearly irrelevant to it, and the compiler is free to
//! re-associate as long as the computed function is preserved. The rewrites here
//! are the secret-sharing counterpart, with communication rounds in place of
//! bootstraps.
//!
//! Both functions below emit into a `let*` binding list in the source language of
//! [`crate::zkp::circuit_compiler`], since that is the frontend both network
//! benchmarks generate.

/// Emit a sum as a balanced tree and return the expression computing it.
///
/// Addition does not communicate, so unlike [`argmax_tree`] this buys no rounds.
/// It is here because a left fold over `n` terms produces an expression nested
/// `n` deep, and the compiler front end walks that nesting recursively: a single
/// convolution tap over a 784-element input is a 784-deep expression, and a
/// realistic network overflows the default thread stack while still being
/// compiled. Balancing takes the nesting to `⌈log₂ n⌉`, which is what allows a
/// wide layer to be emitted at all. The arithmetic is identical — in a ring, and
/// in a prime field, addition is associative exactly, so this changes no value.
pub fn sum_tree(mut terms: Vec<String>) -> String {
    assert!(!terms.is_empty(), "a sum has at least one term");
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        for pair in terms.chunks(2) {
            match pair {
                [only] => next.push(only.clone()),
                [a, b] => next.push(format!("(+ {a} {b})")),
                _ => unreachable!("chunks(2) yields one or two elements"),
            }
        }
        terms = next;
    }
    terms.pop().expect("a sum has at least one term")
}

/// Emit an argmax as a single-elimination tournament, returning the name bound to
/// the winning index. New bindings are appended to `binds`.
///
/// The obvious way to write an argmax is a left fold carrying the running best:
/// `best ← max(best, xᵢ)`. It performs `n−1` comparisons, which is optimal, and
/// it is what a plaintext implementation would do. Under a secure-computation
/// protocol it is nonetheless the wrong program. Each comparison reads the output
/// of the one before it, and a comparison is not a local operation: it is a bit
/// decomposition followed by a logarithmic-depth borrow-propagation chain, so it
/// spans several communication rounds. A left fold has sequential depth `n−1` in
/// units of that subprotocol.
///
/// A tournament performs exactly the same `n−1` comparisons — no arithmetic is
/// saved — but arranges them so that the comparisons within one bracket round are
/// mutually independent, and the scheduler collapses each bracket into a single
/// communication round. Depth falls from `n−1` to `⌈log₂ n⌉`: for ten classes,
/// four instead of nine.
///
/// A competitor with no opponent in a bracket advances uncompared, which is what
/// keeps the comparison count at `n−1` for every `n` rather than rounding up to a
/// power of two.
pub fn argmax_tree(binds: &mut Vec<String>, values: &[String], tag: &str) -> String {
    assert!(!values.is_empty(), "an argmax is over at least one value");
    // Each competitor carries its value and the index it stands for. The index
    // begins as a public constant and becomes secret at the first comparison.
    let mut round: Vec<(String, String)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i.to_string()))
        .collect();
    let mut n = 0usize;
    while round.len() > 1 {
        let mut next: Vec<(String, String)> = Vec::with_capacity(round.len().div_ceil(2));
        for pair in round.chunks(2) {
            match pair {
                [only] => next.push(only.clone()),
                [(lv, li), (rv, ri)] => {
                    let c = format!("{tag}c{n}");
                    let v = format!("{tag}v{n}");
                    let k = format!("{tag}k{n}");
                    // One comparison, then two selections sharing its bit. The
                    // two selections are independent of each other, so they cost
                    // one round between them, not two.
                    binds.push(format!("({c} (> {rv} {lv}))"));
                    binds.push(format!("({v} (+ {lv} (* {c} (- {rv} {lv}))))"));
                    binds.push(format!("({k} (+ {li} (* {c} (- {ri} {li}))))"));
                    next.push((v, k));
                    n += 1;
                }
                _ => unreachable!("chunks(2) yields one or two elements"),
            }
        }
        round = next;
    }
    round.pop().expect("an argmax is over at least one value").1
}

/// Emit an argmax whose depth is **one** comparison, whatever the number of
/// classes, returning the name bound to the winning index.
///
/// [`argmax_tree`] takes the depth of an argmax from `n−1` comparisons to
/// `⌈log₂ n⌉`, which is the best a tournament can do because each bracket must
/// know who won the last. Dropping to constant depth means giving up the
/// tournament: compute *every* pairwise comparison instead, all `n(n−1)/2` of
/// which are mutually independent and therefore batch into a single round, and
/// then read the winner off the comparison matrix.
///
/// Index `i` wins exactly when it beats every other candidate. Writing `bᵢⱼ` for
/// `[xⱼ > xᵢ]` with `i < j`,
///
/// ```text
/// winnerᵢ = ∏_{j>i} (1 − bᵢⱼ) · ∏_{j<i} bⱼᵢ
/// ```
///
/// which is one for the *lowest-indexed* maximum and zero for everything else —
/// the same tie-breaking the fold and the tournament have, since both replace the
/// running best only on a strict improvement. The index is then `Σ i · winnerᵢ`,
/// a public-constant combination and so free. The conjunctions are products of
/// shared bits, and every candidate's product tree is independent of every other's,
/// so all `n` of them together cost `⌈log₂(n−1)⌉` multiplication rounds — rounds
/// of *multiplication*, not of comparison, which is the whole point: a
/// multiplication is one exchange and a comparison is roughly nine.
///
/// The trade is `n(n−1)/2` comparisons against `n−1`: for ten classes, forty-five
/// against nine, five times the traffic. Whether that is worth constant depth is
/// not a matter of taste but of the link, and it is measured rather than assumed —
/// `docs/experiments.md` §VI.9 records the classifier spending 96% of its
/// wide-area time on latency with roughly forty times the byte headroom, which is
/// what licenses spending bytes here. On a link where bandwidth dominates,
/// [`argmax_tree`] is the right choice and this one is not.
pub fn argmax_pairwise(binds: &mut Vec<String>, values: &[String], tag: &str) -> String {
    let n = values.len();
    assert!(n > 0, "an argmax is over at least one value");
    if n == 1 {
        // Nothing to compare, and no comparison should be emitted.
        let name = format!("{tag}idx");
        binds.push(format!("({name} 0)"));
        return name;
    }

    // All pairwise comparisons. Mutually independent, so the scheduler puts every
    // one of them in the same round.
    let bit = |i: usize, j: usize| format!("{tag}b{i}_{j}");
    for i in 0..n {
        for j in (i + 1)..n {
            binds.push(format!("({} (> {} {}))", bit(i, j), values[j], values[i]));
        }
    }

    // Each candidate's conjunction, as a balanced product tree. The trees are
    // grown level by level across all candidates at once so that the whole level
    // is one batch.
    let mut trees: Vec<Vec<String>> = (0..n)
        .map(|i| {
            let mut terms = Vec::with_capacity(n - 1);
            for j in 0..n {
                if j > i {
                    // `i` must not be beaten by a later candidate.
                    terms.push(format!("(- 1 {})", bit(i, j)));
                } else if j < i {
                    // `i` must strictly beat every earlier candidate.
                    terms.push(bit(j, i));
                }
            }
            terms
        })
        .collect();
    let mut level = 0usize;
    while trees.iter().any(|t| t.len() > 1) {
        for (i, terms) in trees.iter_mut().enumerate() {
            if terms.len() <= 1 {
                continue;
            }
            let mut next = Vec::with_capacity(terms.len().div_ceil(2));
            for (p, pair) in terms.chunks(2).enumerate() {
                match pair {
                    [only] => next.push(only.clone()),
                    [a, b] => {
                        let name = format!("{tag}a{i}_{level}_{p}");
                        binds.push(format!("({name} (* {a} {b}))"));
                        next.push(name);
                    }
                    _ => unreachable!("chunks(2) yields one or two elements"),
                }
            }
            *terms = next;
        }
        level += 1;
    }

    // Weighted sum of the indicators. Every weight is a public constant, so this
    // costs no communication at all.
    let idx = sum_tree(
        trees
            .iter()
            .enumerate()
            .skip(1) // index 0 contributes 0 · winner₀
            .map(|(i, t)| format!("(* {i} {})", t[0]))
            .collect(),
    );
    let name = format!("{tag}idx");
    binds.push(format!("({name} {idx})"));
    name
}

#[cfg(test)]
mod tests {

    /// The constant-depth argmax must agree with the tournament on every input,
    /// including the tie cases where the tie-breaking rule is what decides.
    #[test]
    fn the_pairwise_argmax_agrees_with_the_tournament_including_on_ties() {
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let cases: Vec<Vec<i64>> = vec![
            vec![3, 9, 1, 4, 15, 92, 6, 500, 5, 35],
            // Maximum first, maximum last.
            vec![500, 9, 1, 4, 15, 92, 6, 5, 5, 35],
            vec![3, 9, 1, 4, 15, 92, 6, 5, 5, 500],
            // Ties for the maximum: both routes must pick the lowest index.
            vec![7, 7, 1, 4, 15, 15, 6, 5, 5, 3],
            vec![15, 9, 1, 4, 15, 15, 6, 5, 5, 3],
            // Everything equal.
            vec![4; 10],
            // Negatives, and a maximum that is negative.
            vec![-3, -9, -1, -4, -15, -92, -6, -500, -5, -35],
            // Two candidates, and one.
            vec![5, 5],
            vec![9, 2],
            vec![2, 9],
            vec![42],
        ];
        for vals in &cases {
            let names: Vec<String> = (0..vals.len()).map(|i| format!("x{i}")).collect();
            let ins: HashMap<String, i64> =
                names.iter().cloned().zip(vals.iter().copied()).collect();
            // The plaintext reference: the lowest index attaining the maximum.
            let want = vals
                .iter()
                .enumerate()
                .max_by_key(|(i, v)| (**v, std::cmp::Reverse(*i)))
                .map(|(i, _)| i as i64)
                .unwrap();

            let mut got = Vec::new();
            let mut rounds = Vec::new();
            for pairwise in [false, true] {
                let mut binds: Vec<String> = Vec::new();
                let idx = if pairwise {
                    argmax_pairwise(&mut binds, &names, "p")
                } else {
                    argmax_tree(&mut binds, &names, "t")
                };
                let src = if binds.is_empty() {
                    idx.clone()
                } else {
                    format!("(let* ({}) {idx})", binds.join(" "))
                };
                let prog = compile(&src, &cfg).expect("compiles");
                let sched = schedule(&prog);
                let out =
                    crate::mpcc::run_protocol("dealer", 2, &prog, &sched, &ins).expect("runs");
                got.push(out.outputs[0].1);
                rounds.push(sched.round_count());
            }
            assert_eq!(got[0], want, "tournament on {vals:?}");
            assert_eq!(got[1], want, "pairwise on {vals:?}");
            // Constant depth is the claim, so for ten candidates the pairwise
            // route must be strictly shallower. At two candidates they coincide.
            if vals.len() >= 4 {
                assert!(
                    rounds[1] < rounds[0],
                    "pairwise {} should be shallower than tournament {} on {} values",
                    rounds[1],
                    rounds[0],
                    vals.len()
                );
            }
        }
    }

    /// And it must pay for that depth in the currency the doc comment claims:
    /// quadratically many comparisons, not more rounds.
    #[test]
    fn constant_depth_costs_quadratically_many_comparisons() {
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };
        let names: Vec<String> = (0..10).map(|i| format!("x{i}")).collect();
        let count = |src: &str| {
            let prog = compile(src, &cfg).expect("compiles");
            let sched = schedule(&prog);
            (
                prog.code
                    .iter()
                    .filter(|i| matches!(i.op, crate::mpcc::ir::Op::LtS))
                    .count(),
                sched.round_count(),
            )
        };
        let mut tb = Vec::new();
        let ti = argmax_tree(&mut tb, &names, "t");
        let (tc, tr) = count(&format!("(let* ({}) {ti})", tb.join(" ")));
        let mut pb = Vec::new();
        let pi = argmax_pairwise(&mut pb, &names, "p");
        let (pc, pr) = count(&format!("(let* ({}) {pi})", pb.join(" ")));

        assert_eq!(tc, 9, "a tournament needs n-1 comparisons");
        assert_eq!(pc, 45, "the pairwise route needs n(n-1)/2");
        assert!(
            pr < tr,
            "and buys depth with them: {pr} rounds against {tr}"
        );
    }
    use super::*;
    use crate::mpcc::{compile, schedule, LowerConfig};
    use std::collections::HashMap;

    /// The depth of the emitted expression, in nested parentheses.
    fn nesting(s: &str) -> usize {
        let mut d = 0usize;
        let mut max = 0usize;
        for c in s.chars() {
            match c {
                '(' => {
                    d += 1;
                    max = max.max(d);
                }
                ')' => d -= 1,
                _ => {}
            }
        }
        max
    }

    #[test]
    fn a_balanced_sum_is_logarithmically_deep_not_linearly() {
        let terms: Vec<String> = (0..64).map(|i| format!("x{i}")).collect();
        let tree = sum_tree(terms.clone());
        let fold = terms
            .iter()
            .cloned()
            .reduce(|a, b| format!("(+ {a} {b})"))
            .unwrap();
        // Same number of additions, since a sum of `n` terms needs `n-1` of them
        // however it is bracketed.
        assert_eq!(tree.matches("(+").count(), 63);
        assert_eq!(fold.matches("(+").count(), 63);
        // The nesting is where they differ, and it is the nesting that the
        // compiler front end recurses through.
        assert_eq!(nesting(&tree), 6, "log2(64)");
        assert_eq!(nesting(&fold), 63);
    }

    #[test]
    fn a_sum_of_one_term_needs_no_addition() {
        assert_eq!(sum_tree(vec!["x".to_string()]), "x");
    }

    /// The claim this module exists for: the tournament and the fold agree on the
    /// answer, spend the same number of comparisons, and differ in rounds.
    #[test]
    fn the_tournament_agrees_with_the_fold_but_is_shallower() {
        // Ten distinct values, the maximum third from the end so that neither a
        // fold nor a tournament finds it by accident at a boundary.
        let vals: [i64; 10] = [3, 9, 1, 4, 15, 92, 6, 500, 5, 35];
        let names: Vec<String> = (0..vals.len()).map(|i| format!("x{i}")).collect();

        let mut tb: Vec<String> = Vec::new();
        let ti = argmax_tree(&mut tb, &names, "t");
        let tournament = format!("(let* ({}) {ti})", tb.join(" "));

        // The left fold, written out as the benchmarks used to emit it.
        let mut fb: Vec<String> = Vec::new();
        let mut best = names[0].clone();
        let mut idx = "0".to_string();
        for (i, l) in names.iter().enumerate().skip(1) {
            fb.push(format!("(fc{i} (> {l} {best}))"));
            fb.push(format!("(fv{i} (+ {best} (* fc{i} (- {l} {best}))))"));
            fb.push(format!("(fk{i} (+ {idx} (* fc{i} (- {i} {idx}))))"));
            best = format!("fv{i}");
            idx = format!("fk{i}");
        }
        let fold = format!("(let* ({}) {idx})", fb.join(" "));

        let ins: HashMap<String, i64> = names
            .iter()
            .cloned()
            .zip(vals.iter().copied())
            .collect::<HashMap<_, _>>();
        let cfg = LowerConfig {
            parties: 2,
            bit_width: 32,
            ..Default::default()
        };

        let mut got = Vec::new();
        for src in [&tournament, &fold] {
            let prog = compile(src, &cfg).expect("compiles");
            let sched = schedule(&prog);
            let out = crate::mpcc::run_protocol("dealer", 2, &prog, &sched, &ins).expect("runs");
            got.push((
                out.outputs[0].1,
                sched.round_count(),
                prog.code
                    .iter()
                    .filter(|i| matches!(i.op, crate::mpcc::ir::Op::LtS))
                    .count(),
            ));
        }
        let (tv, tr, tc) = got[0];
        let (fv, fr, fc) = got[1];

        // Both find index 7, where 500 sits.
        assert_eq!(tv, 7, "tournament");
        assert_eq!(fv, 7, "fold");
        // Identical arithmetic: nine comparisons for ten candidates, either way.
        assert_eq!(tc, 9, "tournament comparisons");
        assert_eq!(fc, 9, "fold comparisons");
        // And the whole point: strictly fewer rounds for the same work.
        assert!(
            tr < fr,
            "tournament should be shallower: {tr} rounds vs {fr}"
        );
    }
}
