# Paper: *The Switch Is the Bootstrap*

LaTeX source for the paper describing the `baru` construction — deep private
inference at zero FHE bootstraps via scheme switching.

## Files

- `main.tex` — the paper (standard `article` class; only widely-available
  packages: `amsmath`, `amsthm`, `booktabs`, `hyperref`, `cleveref`, …).
- `references.bib` — bibliography.

## Build

Any TeX Live / MiKTeX installation with `bibtex`:

```sh
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

or, if `latexmk` is available:

```sh
latexmk -pdf main.tex
```

The document deliberately avoids venue-specific classes (`llncs`, `iacrcc`,
`iacrtrans`) so it compiles anywhere. To target a venue, swap the first line
for that venue's `\documentclass` and move the author block into its macros.

## Mapping to the artifact

The paper's claims map to the experiments `E1`–`E20` recorded in
[`../docs/paper-outline.md`](../docs/paper-outline.md), each reproducible from
a `baru bench-*` subcommand or a `cargo test`. All quantitative figures in the
paper are produced by those commands (see `../README.md`), not hand-entered.

Headline correspondences:

| paper section | artifact |
|---|---|
| §3 Switch-as-bootstrap | `bench-deep-nn` (exact at every depth, 0 bootstraps) |
| §4 Evaluation-key-free linear layers | `bench-fhe-innerprod`, `fhe::bfv`, `fhe::conv_pack` |
| §4.2 multi-modular NTT | `fhe::ntt` differential tests |
| §5 Triple-free / batched MPC | `aby3_protocol` (`triple_free_mul_*`, `a2b_batch_*`) |
| §5 batched scheme-switch import | `aby3_protocol::share_input_batch`, `batched_input_sharing_matches_the_scalar_protocol`, `scheme_switch_import_does_not_scale_with_layer_width` |
| §6 Composability / gadgets | `src/gadgets/`, `bench-abstraction` (composing costs no more than hand-batching) |
| §7.2 MPC-segment cost table | `bench-cnn --img 28` (prints the per-phase split) |
| §7.3 Garbled vs bit-decomposition | `bench-gc-compare`, `bench-cnn --img 28 --both`, `gc_and_a2b_nonlinearities_agree` |
| §7.3 Tournament argmax | `gadgets::argmax`, `argmax_breaks_ties_to_first_over_every_small_input` |
| §7.5 Preprocessing the switch | `bench-cnn --img 28 --gc --precompute`, `precomputed_masks_change_nothing_but_the_timing` |
| §7.6 Private diagnosis on MedMNIST | `bench-medmnist --dataset all --hybrid 3 --gc --precompute` |
| §7.7 Two-conv MNIST, 98.89% | `bench-mnist`, `two_conv_hybrid_matches_plaintext` |

Performance state, the method that produced these numbers, and the prioritized
next steps are in
[`../docs/private-inference-performance.md`](../docs/private-inference-performance.md).

Every number in §7.2, §7.3, §7.5 and §7.6 comes from one of those commands on an
Apple-silicon laptop, single-threaded, `--release`. Ratios transfer across
machines; absolute seconds do not.
