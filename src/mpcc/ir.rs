//! The Baru MPC bytecode — a protocol-neutral, *vectorized* program
//! representation.
//!
//! This is deliberately **not** [`crate::zkp::circuit_compiler::FlatOp`].
//! `FlatOp` is a scalar SSA designed for ZKP constraint emission and FHE gate
//! evaluation; it is a fine *input* to MPC lowering but it cannot express the
//! four things MPC costing needs (see `docs/mpc-compiler-roadmap.md` §3):
//!
//! * **vectors** — one `Mul` over 784 lanes is *one* communication round, not
//!   784.  Every instruction carries a SIMD `width`.
//! * **domain** — the same multiplication costs one Beaver triple in `Z_{2^k}`
//!   and ~`k` AND gates in `GF(2)`.  Registers are domain-tagged.
//! * **secrecy** — `clear × secret` is local, `secret × secret` costs a round.
//!   Registers carry a [`Secrecy`] tag and the cost model reads it.
//! * **openings** — the unit of MPC communication is the *open*, and `FlatOp`
//!   has no such notion.
//!
//! Instructions are split into **local** (no communication) and **interactive**
//! (consumes at least one round and possibly preprocessing material).  Which
//! side of the line an instruction falls on can depend on its operands, so the
//! predicate lives on [`Program::is_interactive`], not on [`Op`] alone.
//!
//! Composite primitives (`LtS`, `BitDecomp`, `TruncPr`, `Div`, …) are kept as
//! *primitives* rather than expanded during lowering, because their cost is
//! wildly protocol-dependent: a garbled-circuit comparison is `k` AND gates in
//! one round, a ring comparison is an edaBit plus a carry chain, a field
//! comparison needs random bits and a statistical mask.  Expanding them here
//! would bake one protocol's strategy into every protocol's cost.

use std::collections::HashMap;
use std::fmt;

use serde::Serialize;

/// Index into [`Program::regs`].
pub type Reg = u32;

// ─── Domains ──────────────────────────────────────────────────────────────────

/// The algebraic structure a register's elements live in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Domain {
    /// `Z_{2^k}` — matches machine words, what ABY3 / Semi2k / SPDZ2k use.
    Ring { k: u32 },
    /// `Z_p` for a prime of `bits` bits — Shamir / MASCOT / LowGear.
    Field { bits: u32 },
    /// `GF(2)`, one bit per lane — garbled circuits and binary secret sharing.
    Binary,
}

impl Domain {
    /// Bits per element, i.e. the size of one share component on the wire.
    pub fn element_bits(&self) -> u32 {
        match self {
            Domain::Ring { k } => *k,
            Domain::Field { bits } => *bits,
            Domain::Binary => 1,
        }
    }

    pub fn element_bytes(&self) -> f64 {
        f64::from(self.element_bits()) / 8.0
    }

    /// The coarse family a protocol advertises support for.
    pub fn kind(&self) -> DomainKind {
        match self {
            Domain::Ring { .. } => DomainKind::Ring,
            Domain::Field { .. } => DomainKind::Field,
            Domain::Binary => DomainKind::Binary,
        }
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Domain::Ring { k } => write!(f, "ring2^{}", k),
            Domain::Field { bits } => write!(f, "field~2^{}", bits),
            Domain::Binary => write!(f, "gf2"),
        }
    }
}

/// Domain family, without the parameter — what [`crate::mpcc::protocols`]
/// matches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum DomainKind {
    Ring,
    Field,
    Binary,
}

impl fmt::Display for DomainKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DomainKind::Ring => write!(f, "mod 2^k"),
            DomainKind::Field => write!(f, "mod p"),
            DomainKind::Binary => write!(f, "binary"),
        }
    }
}

/// Whether every party knows a register's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Secrecy {
    /// Identical in every party's view — arithmetic on it is free.
    Clear,
    /// Secret-shared.
    Secret,
}

// ─── Registers ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct RegInfo {
    pub domain: Domain,
    pub secrecy: Secrecy,
    /// SIMD lanes.  `1` for a scalar.
    pub width: u32,
    /// Provenance for listings — the source wire name where there is one.
    pub name: Option<String>,
}

impl RegInfo {
    /// Total bits this register occupies in one party's share.
    pub fn bits(&self) -> u64 {
        u64::from(self.width) * u64::from(self.domain.element_bits())
    }
}

// ─── Instructions ─────────────────────────────────────────────────────────────

/// The opcode.  Operand and result registers live on [`Instr`], so analysis
/// passes can walk data flow without matching every variant.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Op {
    // ── always local ──────────────────────────────────────────────────────
    /// Load a public immediate into a clear register.
    LdI(i64),
    /// A party's private input becomes a secret register.
    Input {
        party: usize,
    },
    /// Register copy (SSA plumbing, folded away by most protocols).
    Copy,
    /// Lane-wise addition / subtraction.  Local in every protocol.
    Add,
    Sub,
    /// Lane-wise negation.
    Neg,
    /// Multiply by a public constant.
    MulC(i64),
    /// Add a public constant.
    AddC(i64),
    /// Shift left by a public amount — multiplication by `2^amount`, local.
    ShlC(u32),
    /// `GF(2)` xor / not.  Local in every binary protocol (free-XOR in GC).
    Xor,
    Not,
    /// Declare a clear register as a program result.
    Output,

    // ── interactive (see `Program::is_interactive` for the operand-dependent
    //    cases: `Mul` with a clear operand is local) ─────────────────────────
    /// Lane-wise multiplication.
    Mul,
    /// Reveal a secret register to all parties, producing a clear one.
    Open,
    /// `GF(2)` and.
    And,
    /// Arithmetic → binary: one arithmetic register of `width` lanes becomes a
    /// binary register of `width * nbits` lanes, bit 0 of lane 0 first.
    BitDecomp {
        nbits: u32,
    },
    /// Binary → arithmetic, the inverse of [`Op::BitDecomp`].
    BitComp {
        nbits: u32,
    },
    /// Signed / unsigned less-than, result is a 0/1 secret in the same domain.
    LtS,
    LtU,
    /// Equality with zero, result 0/1.
    EqZ,
    /// Truncation by a public shift.  *Not* local in an arithmetic domain —
    /// this is the classic MPC gotcha that Baru's Q16.16 rescale runs into.
    TruncPr {
        shift: u32,
    },
    /// `cond ? t : e`, lowered by protocols to `t + cond*(e-t)`.
    Mux,
    /// Truncating signed division and remainder (two results).
    Div,
    /// Population count of each lane.
    PopCount {
        nbits: u32,
    },
    /// Cross-domain share conversion (daBit / edaBit based).
    Convert,
}

impl Op {
    /// Interactive regardless of operands.  Operand-dependent cases
    /// (`Mul`/`Mux` with a clear operand) are resolved by
    /// [`Program::is_interactive`].
    pub fn always_interactive(&self) -> bool {
        matches!(
            self,
            Op::Open
                | Op::And
                | Op::BitDecomp { .. }
                | Op::BitComp { .. }
                | Op::LtS
                | Op::LtU
                | Op::EqZ
                | Op::TruncPr { .. }
                | Op::Div
                | Op::PopCount { .. }
                | Op::Convert
        )
    }

    pub fn mnemonic(&self) -> &'static str {
        match self {
            Op::LdI(_) => "ldi",
            Op::Input { .. } => "input",
            Op::Copy => "copy",
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Neg => "neg",
            Op::MulC(_) => "mulc",
            Op::AddC(_) => "addc",
            Op::ShlC(_) => "shlc",
            Op::Xor => "xor",
            Op::Not => "not",
            Op::Output => "output",
            Op::Mul => "mul",
            Op::Open => "open",
            Op::And => "and",
            Op::BitDecomp { .. } => "bitdecomp",
            Op::BitComp { .. } => "bitcomp",
            Op::LtS => "lts",
            Op::LtU => "ltu",
            Op::EqZ => "eqz",
            Op::TruncPr { .. } => "truncpr",
            Op::Mux => "mux",
            Op::Div => "div",
            Op::PopCount { .. } => "popcount",
            Op::Convert => "convert",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Instr {
    pub op: Op,
    /// Registers written.  More than one only for [`Op::Div`].
    pub dst: Vec<Reg>,
    /// Registers read, in operand order.
    pub src: Vec<Reg>,
    /// SIMD lanes this instruction processes.
    pub width: u32,
    /// Input/output variable name, where applicable.
    pub label: Option<String>,
}

// ─── Programs ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Program {
    /// Registers, in SSA order.
    pub regs: Vec<RegInfo>,
    pub code: Vec<Instr>,
    /// `(variable, register)` for each secret input, in declaration order.
    pub inputs: Vec<(String, Reg)>,
    /// `(variable, register)` for each opened result.
    pub outputs: Vec<(String, Reg)>,
    /// The arithmetic domain the program was lowered for.  Binary registers may
    /// still appear (via [`Op::BitDecomp`]) — mixed circuits are the norm.
    pub domain: Domain,
    /// Bit width used for comparisons, bit-decompositions and truncations.
    /// Distinct from `domain.element_bits()`: Baru's REPL convention is 32-bit
    /// values carried in 64-bit shares, and the cost of a comparison follows the
    /// *value* width, not the share width.
    pub bit_width: u32,
}

impl Program {
    pub fn new(domain: Domain) -> Self {
        Program::with_bit_width(domain, domain.element_bits())
    }

    /// Does this program need **masked bit-decomposition**?
    ///
    /// Every comparison-shaped operation reduces to one, and masked opening
    /// costs [`crate::mpcc::vm::KAPPA`] bits of statistical headroom *above*
    /// the value width. A protocol whose modulus is too small to supply that
    /// headroom can still run arithmetic, so this is the predicate that
    /// separates "cannot run this program" from "cannot run at all" — see
    /// [`crate::mpcc::lowgear::LowGearEngine`].
    pub fn needs_bit_decomposition(&self) -> bool {
        self.code.iter().any(|i| {
            matches!(
                i.op,
                Op::BitDecomp { .. }
                    | Op::BitComp { .. }
                    | Op::LtS
                    | Op::LtU
                    | Op::EqZ
                    | Op::TruncPr { .. }
                    | Op::Div
                    | Op::PopCount { .. }
            )
        })
    }

    pub fn with_bit_width(domain: Domain, bit_width: u32) -> Self {
        Program {
            regs: Vec::new(),
            code: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            domain,
            bit_width,
        }
    }

    pub fn alloc(
        &mut self,
        domain: Domain,
        secrecy: Secrecy,
        width: u32,
        name: Option<String>,
    ) -> Reg {
        self.regs.push(RegInfo {
            domain,
            secrecy,
            width,
            name,
        });
        (self.regs.len() - 1) as Reg
    }

    pub fn reg(&self, r: Reg) -> &RegInfo {
        &self.regs[r as usize]
    }

    pub fn push(&mut self, instr: Instr) {
        self.code.push(instr);
    }

    /// True when executing `instr` requires communication.
    ///
    /// `Mul` and `Mux` are the operand-dependent cases: multiplying by a clear
    /// value, or selecting on a clear condition, is a local operation in every
    /// protocol.  Everything else follows [`Op::always_interactive`], except
    /// that an operation whose operands are *all* clear is public computation
    /// and therefore local too.
    pub fn is_interactive(&self, instr: &Instr) -> bool {
        let any_secret = instr
            .src
            .iter()
            .any(|r| self.reg(*r).secrecy == Secrecy::Secret);
        if !any_secret {
            return false;
        }
        match instr.op {
            Op::Mul => instr
                .src
                .iter()
                .all(|r| self.reg(*r).secrecy == Secrecy::Secret),
            // `Mux(c,t,e)` costs one multiplication, by the condition.
            Op::Mux => self.reg(instr.src[0]).secrecy == Secrecy::Secret,
            ref op => op.always_interactive(),
        }
    }

    /// `Secret` when any operand of `instr` is secret-shared.  An instruction
    /// whose operands are all clear is public computation and costs nothing in
    /// any protocol — not even gates in a garbled circuit.
    pub fn srcs_secrecy(&self, instr: &Instr) -> Secrecy {
        if instr
            .src
            .iter()
            .any(|r| self.reg(*r).secrecy == Secrecy::Secret)
        {
            Secrecy::Secret
        } else {
            Secrecy::Clear
        }
    }

    /// Aggregate instruction/lane counts, split local vs interactive.
    pub fn counts(&self) -> Counts {
        let mut c = Counts::default();
        for instr in &self.code {
            let interactive = self.is_interactive(instr);
            c.instructions += 1;
            c.lanes += u64::from(instr.width);
            if interactive {
                c.interactive += 1;
                c.interactive_lanes += u64::from(instr.width);
            } else {
                c.local += 1;
            }
            *c.by_op.entry(instr.op.mnemonic()).or_insert(0) += 1;
        }
        c
    }

    /// A human-readable bytecode listing.
    pub fn listing(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "; domain {}, {}-bit values\n",
            self.domain, self.bit_width
        ));
        for (name, r) in &self.inputs {
            out.push_str(&format!("; input  {} -> r{}\n", name, r));
        }
        for (i, instr) in self.code.iter().enumerate() {
            let dsts = instr
                .dst
                .iter()
                .map(|r| format!("r{}", r))
                .collect::<Vec<_>>()
                .join(", ");
            let srcs = instr
                .src
                .iter()
                .map(|r| format!("r{}", r))
                .collect::<Vec<_>>()
                .join(", ");
            let arrow = if dsts.is_empty() { "" } else { " <- " };
            let tag = if self.is_interactive(instr) { "*" } else { " " };
            out.push_str(&format!(
                "{:>4}{} {}{}{:<9} {}{}\n",
                i,
                tag,
                dsts,
                arrow,
                instr.op.mnemonic(),
                srcs,
                match &instr.label {
                    Some(l) => format!("  ; {}", l),
                    None => {
                        if instr.width > 1 {
                            format!("  ; x{}", instr.width)
                        } else {
                            String::new()
                        }
                    }
                }
            ));
        }
        for (name, r) in &self.outputs {
            out.push_str(&format!("; output {} <- r{}\n", name, r));
        }
        out.push_str("; '*' marks interactive instructions\n");
        out
    }
}

/// Instruction census produced by [`Program::counts`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct Counts {
    pub instructions: u64,
    pub local: u64,
    pub interactive: u64,
    /// Total SIMD lanes across all instructions.
    pub lanes: u64,
    /// Total SIMD lanes across interactive instructions only.
    pub interactive_lanes: u64,
    pub by_op: HashMap<&'static str, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_multiplication_is_local() {
        let mut p = Program::new(Domain::Ring { k: 64 });
        let s = p.alloc(Domain::Ring { k: 64 }, Secrecy::Secret, 1, None);
        let c = p.alloc(Domain::Ring { k: 64 }, Secrecy::Clear, 1, None);
        let d = p.alloc(Domain::Ring { k: 64 }, Secrecy::Secret, 1, None);
        let instr = Instr {
            op: Op::Mul,
            dst: vec![d],
            src: vec![s, c],
            width: 1,
            label: None,
        };
        assert!(!p.is_interactive(&instr));

        let s2 = p.alloc(Domain::Ring { k: 64 }, Secrecy::Secret, 1, None);
        let instr = Instr {
            op: Op::Mul,
            dst: vec![d],
            src: vec![s, s2],
            width: 1,
            label: None,
        };
        assert!(p.is_interactive(&instr));
    }

    #[test]
    fn public_only_computation_never_communicates() {
        let mut p = Program::new(Domain::Ring { k: 64 });
        let a = p.alloc(Domain::Ring { k: 64 }, Secrecy::Clear, 1, None);
        let b = p.alloc(Domain::Ring { k: 64 }, Secrecy::Clear, 1, None);
        let instr = Instr {
            op: Op::LtS,
            dst: vec![a],
            src: vec![a, b],
            width: 1,
            label: None,
        };
        assert!(!p.is_interactive(&instr));
    }

    #[test]
    fn element_sizes() {
        assert_eq!(Domain::Ring { k: 64 }.element_bytes(), 8.0);
        assert_eq!(Domain::Field { bits: 128 }.element_bytes(), 16.0);
        assert_eq!(Domain::Binary.element_bits(), 1);
    }
}
