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
| §6 Composability / gadgets | `src/gadgets/` |
| §7.4 Two-conv MNIST, 98.05% | `bench-mnist`, `two_conv_hybrid_matches_plaintext` |
