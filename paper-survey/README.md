# Paper: *Measuring Privacy-Preserving Computation in One Place*

LaTeX source for the cross-paradigm survey — MPC, FHE, ZKP and their hybrids,
measured in a single harness. The prose draft this was written from is
[`../docs/survey.md`](../docs/survey.md).

## Files

- `main.tex` — the paper (standard `article` class; only widely-available
  packages: `amsmath`, `booktabs`, `siunitx`, `hyperref`, `cleveref`, …).
- `references.bib` — bibliography, 51 entries. Seeded from
  [`../paper/references.bib`](../paper/references.bib) and extended with the
  ~20 entries this paper needs (Falcon, SecureNN, Chida, Eerikson, Overdrive,
  SPDZ2k, MP-SPDZ, IKNP, Chou–Orlandi, KOS, Binius, HyCC, Silph, SING, CirC,
  SEAL, NTL, TFHE-rs, Wilson, ABY).

## Build

```sh
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

or `latexmk -pdf main.tex`. Builds clean: 16 pages, no warnings, no overfull
boxes, all 42 citations resolved.

The document deliberately avoids venue-specific classes so it compiles
anywhere. To target a venue, swap the first line for that venue's
`\documentclass` and move the author block into its macros.

## How this paper differs from the other two

This repository carries three papers, and they should not be merged:

| paper | claim |
|---|---|
| [`../paper/`](../paper/) | *The Switch Is the Bootstrap* — one sharp result: deep private inference at zero FHE bootstraps |
| [`../paper-circuits/`](../paper-circuits/) | circuit optimization: the compiler against hand-written baselines |
| **this one** | breadth and methodology: the cross-paradigm comparisons no single-claim paper has room to make |

This paper should **cite** the other two rather than absorb them.

## Framing

It is written as a **Systematization of Knowledge with an artifact**, not a
pure survey. The distinction matters for venue choice and is argued in
[`../docs/survey.md` §12](../docs/survey.md); briefly, the distinguishing asset
is *original cross-paradigm measurement under uniform accounting*, and several
findings contradict or qualify published claims — including our own. A pure
survey is judged on literature coverage, which is this work's weakest axis.

Suggested venues, in order: **PoPETs/PETS (SoK track)**, then IEEE TDSC or the
Journal of Cryptographic Engineering. ACM Computing Surveys only if reframed
toward coverage and away from original measurement, which we would advise
against.

## Provenance of the numbers

| table | command |
|---|---|
| `tab:malicious` — cost of active security | `bench-malicious --n 32 --bits 32` |
| `tab:triples` — SHE vs OT preprocessing | `bench-triples` |
| `tab:layer` — FHE↔MPC crossover | `bench-layer` |
| `tab:crossover` — nonlinearity crossover | `bench-medmnist --sweep` |
| `tab:medmnist` — private diagnosis | `bench-medmnist --dataset all` |
| `tab:deep` — zero-bootstrap depth scaling | `bench-deep-nn --depths 2,4,8,16,32` |
| `tab:zkp` — proving a quantized neuron | `bench-zkp-layer --widths 1,4,16,64,256,784` |
| §6.2 — cost model vs a real socket | `bench-net` |

`tab:malicious`, `tab:triples` and `tab:crossover` were re-run on 2026-08-01
for this draft. The rest are cited from
[`../docs/experiments.md`](../docs/experiments.md), which carries the full
provenance and the corrections behind each row.

All figures are from one Apple M1 Pro, `--release`. **Ratios transfer across
machines; absolute times do not** — the paper says so in §10 rather than
leaving it to the reader.

## Known gaps a reviewer will probe

Both are disclosed in §10 of the paper rather than buried:

1. **The Falcon backend's preprocessing is dealt from a common seed and is not
   secure** — all three parties can reconstruct the mask. The online protocol
   and its measured costs are real; the offline phase is not. This affects
   `tab:crossover`, which measures online cost.
2. **The network model is validated on loopback**, not a WAN, so the RTT and
   bandwidth constants are modelled rather than measured in the deployment the
   tables describe.
