//! Round scheduling — the pass that makes MPC costing meaningful.
//!
//! In secret-sharing MPC the dominant online cost is not the number of
//! operations but the number of *communication rounds*: on a WAN, one round is
//! tens of milliseconds, during which a party could have done millions of local
//! multiplications.  MP-SPDZ's central optimization is merging independent
//! openings into the same round; this module is Baru's equivalent.
//!
//! The schedule is ASAP (as-soon-as-possible) over the data-dependence DAG:
//!
//! * a **local** instruction inherits the level of its latest operand — it costs
//!   nothing and does not delay anything;
//! * an **interactive** instruction sits one level *after* its latest operand,
//!   and every interactive instruction at the same level is executed in the
//!   same round.
//!
//! ASAP is optimal for depth, and depth is exactly the round count, so this is
//! the minimum-round schedule for a fixed lowering.  It is also why
//! [`crate::mpcc::lower`] does not need to emit vectorized instructions to get
//! batching credit: 784 independent scalar `Mul`s all land at the same level and
//! therefore in one round, which is how a batching protocol would run them.
//!
//! Note what this *does* assume: that the protocol can batch an unbounded number
//! of operations into one round.  Real protocols have message-size limits, so
//! [`crate::mpcc::cost`] adds the bandwidth term separately rather than pretending
//! a round is free of volume.

use serde::Serialize;

use super::ir::{Instr, Program};

/// One communication round: the interactive instructions executed together.
#[derive(Debug, Clone, Serialize)]
pub struct Round {
    /// Indices into [`Program::code`].
    pub instrs: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Schedule {
    /// `level[i]` is the round index (0-based) at which instruction `i` runs.
    /// Local instructions carry the level of the round they are ready by.
    pub level: Vec<usize>,
    /// Interactive instructions grouped by round, in execution order.
    pub rounds: Vec<Round>,
}

impl Schedule {
    pub fn round_count(&self) -> usize {
        self.rounds.len()
    }

    /// The widest round, a proxy for peak message volume.
    pub fn max_round_ops(&self) -> usize {
        self.rounds
            .iter()
            .map(|r| r.instrs.len())
            .max()
            .unwrap_or(0)
    }

    /// Mean interactive operations per round — the batching factor.  A value
    /// near 1 means the program is latency-bound (a serial dependence chain);
    /// a large value means it is bandwidth-bound and will do well on a WAN.
    pub fn batching_factor(&self) -> f64 {
        if self.rounds.is_empty() {
            return 0.0;
        }
        let total: usize = self.rounds.iter().map(|r| r.instrs.len()).sum();
        total as f64 / self.rounds.len() as f64
    }
}

/// Compute the minimum-round ASAP schedule for `prog`.
pub fn schedule(prog: &Program) -> Schedule {
    // Which instruction last wrote each register.  SSA in practice, but the
    // lowering reuses cached constant registers, so take the latest writer.
    let mut writer: Vec<Option<usize>> = vec![None; prog.regs.len()];
    let mut level: Vec<usize> = vec![0; prog.code.len()];
    let mut rounds: Vec<Round> = Vec::new();

    for (idx, instr) in prog.code.iter().enumerate() {
        let ready = operand_level(instr, &writer, &level);
        let lvl = if prog.is_interactive(instr) {
            while rounds.len() <= ready {
                rounds.push(Round { instrs: Vec::new() });
            }
            rounds[ready].instrs.push(idx);
            // Results are available to consumers only after this round.
            ready + 1
        } else {
            ready
        };
        level[idx] = lvl;
        for d in &instr.dst {
            writer[*d as usize] = Some(idx);
        }
    }

    Schedule { level, rounds }
}

/// The earliest level at which every operand of `instr` is available.
fn operand_level(instr: &Instr, writer: &[Option<usize>], level: &[usize]) -> usize {
    instr
        .src
        .iter()
        .filter_map(|r| writer[*r as usize])
        .map(|w| level[w])
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpcc::ir::{Domain, Instr, Op, Secrecy};
    use crate::mpcc::lower::{lower, LowerConfig};

    fn secret(p: &mut Program) -> u32 {
        let r = p.alloc(Domain::Ring { k: 64 }, Secrecy::Secret, 1, None);
        p.push(Instr {
            op: Op::Input { party: 0 },
            dst: vec![r],
            src: vec![],
            width: 1,
            label: None,
        });
        r
    }

    fn mul(p: &mut Program, a: u32, b: u32) -> u32 {
        let d = p.alloc(Domain::Ring { k: 64 }, Secrecy::Secret, 1, None);
        p.push(Instr {
            op: Op::Mul,
            dst: vec![d],
            src: vec![a, b],
            width: 1,
            label: None,
        });
        d
    }

    #[test]
    fn independent_multiplications_share_one_round() {
        let mut p = Program::new(Domain::Ring { k: 64 });
        let (a, b, c, d) = (
            secret(&mut p),
            secret(&mut p),
            secret(&mut p),
            secret(&mut p),
        );
        mul(&mut p, a, b);
        mul(&mut p, c, d);
        let s = schedule(&p);
        assert_eq!(s.round_count(), 1);
        assert_eq!(s.rounds[0].instrs.len(), 2);
        assert_eq!(s.batching_factor(), 2.0);
    }

    #[test]
    fn a_dependence_chain_costs_one_round_per_link() {
        let mut p = Program::new(Domain::Ring { k: 64 });
        let a = secret(&mut p);
        let b = secret(&mut p);
        let m1 = mul(&mut p, a, b);
        let m2 = mul(&mut p, m1, b);
        mul(&mut p, m2, b);
        let s = schedule(&p);
        assert_eq!(s.round_count(), 3);
        assert_eq!(s.batching_factor(), 1.0);
    }

    #[test]
    fn a_dot_product_is_log_depth_in_rounds_not_linear() {
        // Products are independent (1 round); the additions are local, so the
        // whole dot product plus the final open is 2 rounds regardless of size.
        let src = "(dot (list a b c d) (list e f g h))";
        let expr = crate::interpreter::parse(src).expect("parse");
        let mut cc = crate::zkp::circuit_compiler::CircuitCompiler::new();
        let circuit = cc.compile_fhe(&expr).expect("compile");
        let p = lower(
            &circuit.flat_ops,
            &circuit.output_wire,
            &LowerConfig::default(),
        )
        .expect("lower");
        let s = schedule(&p);
        assert_eq!(
            s.round_count(),
            2,
            "4 parallel products + 1 open:\n{}",
            p.listing()
        );
        assert_eq!(s.rounds[0].instrs.len(), 4, "all products in one round");
    }
}
