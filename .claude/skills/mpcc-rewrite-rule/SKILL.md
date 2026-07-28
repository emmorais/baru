---
name: mpcc-rewrite-rule
description: Add a backend-aware circuit rewrite rule to Baru's mpcc::opt optimizer, or a gadget lowering it can use. Use when asked to add/propose an optimization, rewrite rule, peephole, algebraic identity, strength reduction, reassociation, gadget lowering, or "make this circuit cheaper" for MPC/FHE/ZKP backends. Also use when investigating why a rule is accepted for one protocol and rejected for another.
---

# Adding a rewrite rule to `mpcc::opt`

Baru's optimizer takes the target protocol as a **parameter** and keeps a rewrite
only when that protocol's predicted cost improves. Your job when adding a rule is
therefore *not* to argue the rewrite is good — it is to make it available and let
the cost model judge. Two consequences shape everything below:

- **You never have to be right, only productive.** Nothing lands without passing
  the differential oracle, and nothing is applied unless the cost model says it
  pays. Propose freely; the harness disposes.
- **A rule that helps nobody is still a useful contribution** if the reason is
  interesting. `algebraic`/`cse` survive under every backend and act as controls.

## Before you start: is it actually a rewrite rule?

| You want to… | Do this instead |
|---|---|
| make an existing rule fire in more places | extend that rule's `*_sites` / candidate predicate |
| change how an op is *lowered* to a protocol | `mpcc::lower`, not `opt` |
| change what an op *costs* | `mpcc::cost` + `mpcc::protocols`; see `ot_cost.rs` for the measured-input pattern |
| make the circuit smaller for FHE depth | still `opt` — but price it under an FHE backend, and note `cost.rs` models MPC protocols only |

## The five-minute version

1. Add a variant to `Rule` in `src/mpcc/opt.rs`, with a doc comment saying what
   it rewrites.
2. Add it to `Rule::ALL` and `Rule::name()`.
3. Write `fn my_rule(flat: &[FlatOp], output_wire: &str, only: Option<&str>) -> Vec<FlatOp>`.
4. Dispatch it from `Rule::apply_at`.
5. Decide whether it is **per-site** (see below) and answer `Rule::sites`.
6. Write tests: correctness via `equivalent`, plus a per-backend divergence test.
7. Run `cargo test --release --lib mpcc::opt` and
   `cargo run --release -- mpcc optimize --expr '<program>' --wan`.

## What you are rewriting

`FlatOp` (in `src/zkp/circuit_compiler.rs`) is a scalar SSA list. Each op names a
`dst` wire and reads wires. It is the *shared* ZKP/FHE IR — **do not change
`FlatOp` itself** to serve a rewrite; both other backends depend on it.

The ops you will most often touch: `Const`, `IAdd32`, `ISub32`, `IMul`, `Shl`,
`Shr`, `BAnd`, `BOr`, `BXor`, `BNot`, `Mux`, `ICmpEq`, `ICmpUlt`, `ICmpSlt`,
`IDiv`. Semantics are wrapping two's-complement in `u64`; `Shr` is **logical**
everywhere (an arithmetic shift is a gadget, not an op).

Invariants your output must preserve:

- **SSA order.** Every wire is written before it is read. If you emit a
  replacement at the position of the op you replaced, its operands are already
  defined — that is why the existing rules do exactly that.
- **The `dst` of the rewritten op keeps its name.** Downstream ops read it, and
  per-site addressing relies on it (below).
- **Fresh wires** are `opt$N`, allocated by skipping names already in the
  circuit. Copy the `taken`/`name` closure from `strength_reduce`.
- **Single-use before you restructure.** If you fold an intermediate away, check
  `use_counts(flat, output_wire)` says it is read once and is not the output
  wire. `reassociate` refuses chains whose intermediates are shared, and there is
  a test for it.

## Per-site or whole-circuit?

`Rule::sites` returns `Some(wires)` for rules whose benefit **depends on the
operand**, and `None` for rules that are cheap wherever they apply.

```rust
fn sites(&self, flat: &[FlatOp], output_wire: &str) -> Option<Vec<String>> {
    match self {
        Rule::StrengthReduce => Some(strength_reduce_sites(flat)),
        Rule::Reassociate => Some(reassociate_sites(flat, output_wire)),
        Rule::Algebraic | Rule::Cse | Rule::FuseShifts => None,
    }
}
```

Choose per-site if you can imagine one program where the rule pays at one site
and not another — `x * 4` versus `x * 100` is the canonical case. A site is named
by **the wire it writes**, which survives the rewrite, so accepting one site does
not move the others.

Two ways to enumerate sites, both used in the tree:

- A shared predicate, so enumeration cannot drift from the rewrite
  (`strength_reduce_candidate` feeds both `strength_reduce_sites` and the rule).
  Prefer this.
- The rewrite as its own oracle: a site is a wire where applying the rule *only
  there* changes the circuit (`reassociate_sites`). Simpler, `O(n)` rewrites to
  enumerate — fine at these circuit sizes, and `sites_considered` reports the
  cost.

Per-site search prices every site separately, so it is not free. `sites_considered`
is reported so coverage stays visible; a 2351-instruction circuit across all 11
backends takes ~1 minute.

## Correctness is checked, not argued

```rust
equivalent(&original, &rewritten, output_wire, 64)?   // random differential test
```

`equivalent` evaluates both circuits through the **existing** `FlatOp`
interpreter (`evaluate_fhe` on `MockFheBackend`, whose `Ct = u64` makes it exact).
Reusing that evaluator is deliberate — a private copy of the semantics would
drift, and the oracle would then be checking the optimizer against itself.

`optimize(..., verify: true)` runs it for you on the final circuit. In a unit test
call it directly on your rule's output so a failure points at your rule.

**This is random testing, not proof.** It is the standard the FHE backends are
held to, and it has caught real bugs. It will not catch a rewrite that is wrong on
a measure-zero input set — so if your identity has edge cases (division by zero,
shift by ≥ width, `i64::MIN` negation), add explicit cases rather than trusting
64 random trials. State the assumption in the rule's doc comment.

## Then show the divergence

A rule is interesting when backends disagree about it. The tests to write:

```rust
// 1. Correctness, directly on the rule.
let rebuilt = my_rule(&flat, &out, None);
equivalent(&flat, &rebuilt, &out, 64).unwrap();

// 2. Someone takes it.
let taken = opt_wan("<program>", "shamir");
assert!(taken.applied.iter().any(|a| a.rule == "my-rule"));

// 3. Someone declines it, for a reason you can name.
let declined = opt_wan("<program>", "bmr");
assert!(declined.applied.iter().all(|a| a.rule != "my-rule"));
```

Helpers already in the test module: `compile`, `cfg_for`, `dep_for`, `opt`,
`opt_wan`, `chain`.

Protocol families and what they charge for — this is how you predict who takes
your rule:

| family | protocols | pays for | indifferent to |
|---|---|---|---|
| arithmetic secret sharing | `aby3`, `shamir`, `semi2k`, `spdz2k`, `mascot`, `lowgear`, `atlas`, `rep-field` | multiplications (1 round each), depth on a WAN | additions, public-constant multiplies (both local/free) |
| binary, not constant-round | `semi-bin`, `tinier` | AND gates **and** AND-depth — an adder is real depth here | XOR, shifts (free-XOR, wire relabelling) |
| constant-round | `bmr`, `yao` | AND gates only | depth, entirely |

The asymmetries worth exploiting: a public-constant multiply is free under
arithmetic sharing but a full multiplier under GC; right-shift/truncation is free
in GC and expensive in arithmetic; depth is rounds everywhere except
constant-round.

## Then measure it

```sh
cargo run --release -- mpcc optimize --expr '<program>' --wan
cargo run --release -- mpcc optimize --expr '<program>' --wan --only shamir,bmr,semi-bin
```

Read the `rules kept` column — `rule×n` means a per-site rule was taken at `n`
sites. Different rule sets or different site counts per backend **is** the result.

## Writing it up

Repo convention, and it is enforced by the reviewer more than the compiler:

- Put the *reason* in the rule's doc comment, including which backends should
  care and why. `reassociate`'s comment is the model.
- **Never write a results table you have not run.** Run the command, paste the
  real numbers. (This has gone wrong before.)
- If the rule turns out to help nobody, say so and keep the analysis — a negative
  result with a mechanism is worth more than a deleted branch.
- Update `docs/research-agenda.md` §4 if the rule changes the per-backend story,
  and `src/mpcc/opt.rs`'s module docstring if it adds a new *kind* of rule.

## Known-good gadget costs

`docs/gadget-costs.md` is the curated cost table (comparator, adder, multiplier,
argmax, truncation) — check a proposed lowering against it before claiming a win.
It is deliberately a small checked-in file rather than a retrieval pipeline: at
this scale a table you can diff beats a RAG you cannot.

## Current rules, for shape

| rule | per-site | what it does |
|---|---|---|
| `algebraic` | no | `x*1→x`, `x*0→0`, `x±0→x`, shift-by-0, mux on a known condition |
| `cse` | no | reuse an identical earlier computation |
| `fuse-shifts` | no | `(x<<a)<<b → x<<(a+b)` |
| `strength-reduce` | **yes** | `x*c` → shifts and adds, for `c` with few set bits |
| `reassociate` | **yes** | rebalance an associative chain: depth `n-1` → `⌈log₂ n⌉` |

Ideas not yet implemented, in rough order of how likely the cost model is to take
them: mux-chain flattening; comparison narrowing (compare `k` bits when the range
is known); `ISub32(0,x)` → negate; constant folding across `Shr`/`Shl` pairs;
distributing a shared factor out of a sum (`a*x + b*x → (a+b)*x`, a win only where
multiplies cost rounds); reassociation *for balance* on `Mux` trees.
