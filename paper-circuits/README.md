# Paper: backend-aware circuit optimization for private AI

Second paper, distinct from `../paper/` (switch-as-bootstrap). Plan and gap
analysis in [`../docs/paper-circuit-optimization.md`](../docs/paper-circuit-optimization.md).

## Build

```sh
latexmk -pdf main.tex
```

## Every number traces to a command

| claim | command |
|---|---|
| composing costs no more than hand-batching | `baru bench-abstraction --n 64 --out-dim 8` |
| layer rounds flat in width; CSE; reassociation | `baru bench-circuit` |
| whole network vs expert, ablation ladder | `baru bench-compiled-nn` |
| backend-aware divergence matrix | `baru bench-backend-aware` |
| CSE on a real AI circuit (2365 → 1764 gates) | `cargo run --release --example mnist` |
| protocol assignment | `baru mpcc assign --expr ...` |

Correctness gates that must be green for any of it to mean anything:

```sh
cargo test --release --lib
cargo test --release --lib -- --ignored --skip bench::cohort --skip bench::net
cargo test --release --test fhe_differential -- --ignored
```
