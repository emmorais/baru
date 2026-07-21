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

3. `SEAL/native/src/seal/c/defines.h` (Windows/MSVC): the C API decoration
   falls back to `__declspec(dllimport)` when no `*_EXPORTS` macro is defined.
   sealy links SEAL_C **statically**, and cmake only auto-defines `sealc_EXPORTS`
   for a *shared* library, so the static MSVC build hit that fallback — and MSVC
   rejects *defining* a `dllimport` function (error C2491).  The non-exporting
   branch was changed to plain `extern "C"`, which is correct for a static lib.

4. `src/memory.rs`, `src/parameters/mod.rs` (Windows/MSVC): FFI type widths made
   platform-portable.  The bindings were written against Unix LP64, where
   `c_ulong`/`i64` happen to be 64-bit; on Windows LLP64 they are not.
   - `MemoryPoolHandle_UseCount` takes `long*` → use `c_long` (i32 on Windows,
     i64 on Unix) and widen the return to `i64`.
   - `EncParams_GetParmsId` takes `uint64_t*` → use `u64` (was `c_ulong`, which
     is only 64-bit on LP64).

   These are correct on both platforms, not Windows-only workarounds.

## Building on Windows

SEAL's C wrapper `#include`s `CorError.h`, which ships with the .NET Framework
SDK (NETFXSDK), not the modern Windows SDK.  Install it via an older SDK
(`winget install Microsoft.WindowsSDK.10.0.18362`) and build from the repo root
with `build-windows.cmd` — it enters the VS developer environment and prepends
the NETFXSDK `um` include dir.  See the repo README for details.
