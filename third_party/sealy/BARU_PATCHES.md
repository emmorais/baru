# Vendored sealy 0.2.0 — Baru patches

Vendored from crates.io `sealy = "0.2.0"` (Rust bindings for Microsoft SEAL 4.x,
Apache-2.0/MIT; SEAL itself is MIT).  Pruned to keep the tree small: removed
`SEAL/thirdparty` (vendored zlib/zstd sources), `SEAL/dotnet`, `SEAL/android`,
`SEAL/pipelines`, `SEAL/native/{tests,bench,examples}`, and crate `benches/`.

Local modifications:

1. `build.rs`: `SEAL_USE_ZLIB=OFF`, `SEAL_USE_ZSTD=OFF`.
   The vendored zlib 1.2.x fails to compile under clang 21 / macOS 26 SDK
   (`fdopen` macro clash), and compression only affects serialized sizes.

2. `src/evaluator/ckks.rs`: added `rotate_vector`, `rotate_vector_inplace`,
   `rescale_to_next`, `rescale_to_next_inplace`, `complex_conjugate` on
   `CKKSEvaluator`.  These SEAL C API entry points
   (`Evaluator_RotateVector`, `Evaluator_RescaleToNext`,
   `Evaluator_ComplexConjugate`) were never wrapped upstream; the upstream
   `rotate_rows`/`rotate_columns` wrappers call BFV-only entry points and
   throw under CKKS (upstream tests are `#[ignore]`d for this reason).
