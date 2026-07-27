# BaruDB — the encrypted experiment store

**Repository:** <https://github.com/emmorais/blindsearch>

**Purpose.** Long-running privacy-preserving-ML experiments produce results over
weeks (TFHE MNIST inference alone is ~47 min/run). Those results have to survive
across sessions, machines and months so that a paper can cite them, and they are
themselves sensitive (model weights, timing side-channel data, unpublished
numbers). BaruDB is the durable, encrypted, *searchable* home for them; Baru is
its first client.

This document is the **contract**: what Baru needs BaruDB to provide, and what
Baru guarantees about what it sends. It is written from Baru's side so that
blindsearch can implement against it. Baru ships a local-file store implementing
the same trait, so Baru is never blocked on BaruDB being ready.

---

## 1. Data model

One record per **run**. A run is one (program, backend, configuration) execution
with its measurements. Records are append-only and immutable; a correction is a
new record superseding an old one by id.

```
Run
  id             ULID-like, sortable, unique              (clear — it is the key)
  created_at     RFC3339 UTC                              (clear)
  suite          e.g. "bench", "bench-cnn", "mpcc-compare" (searchable keyword)
  case           e.g. "mnist-linear", "dot4-builtin"      (searchable keyword)
  backend        e.g. "mpc-aby3", "fhe-tfhe", "fhe-ckks"  (searchable keyword)
  tags           free-form keywords                        (searchable keywords)
  provenance     git commit, dirty flag, host, os, cpu,
                 rustc version, cargo profile              (encrypted payload)
  config         arbitrary JSON (sizes, party count,
                 network profile, security parameters)     (encrypted payload)
  metrics        name → f64 (+ unit)                       (encrypted payload)
  notes          free text                                 (encrypted payload)
  ok             whether the run met its correctness check  (searchable keyword)
```

The split matters: **keywords are what BaruDB can search over; the payload is
what BaruDB can only store and return.** Baru will never ask BaruDB to filter on
a metric value — range queries over encrypted numerics are exactly the hard part
blindsearch is still building (see its `DESIGN.md`). Baru filters keywords
server-side and metrics client-side after decryption. If/when BaruDB gains
order-preserving or range-capable indexes, §6 says how the contract extends.

### Keyword derivation

Baru emits keywords as `field:value` strings, lowercased, e.g.

```
suite:bench-cnn   case:mnist-linear   backend:fhe-tfhe   ok:true
tag:paper-table-3   host:eduardo-mbp   commit:981e639
```

That is the entire vocabulary a query needs, and it is deterministic, so
blindsearch's existing deterministic-token construction is sufficient — no new
crypto is required for v1.

---

## 2. Required API (v1)

Baru is happy with **either** transport; the trait in `src/db/mod.rs` abstracts it.

### 2a. Process transport (simplest — implement this first)

```
blindsearch put    --db <path> --keywords <k1,k2,...> [--stdin]      → {"id": "..."}
blindsearch get    --db <path> --id <id>                             → record JSON on stdout
blindsearch search --db <path> --keyword <k> [--keyword <k> ...]     → {"ids": [...]}
blindsearch list   --db <path> [--limit N] [--after <id>]            → {"ids": [...]}
blindsearch stats  --db <path>                                       → {"records": N, "bytes": N}
```

Rules Baru relies on:

* The password comes from `BLINDSEARCH_PASSWORD` when set, so Baru can run
  unattended in a benchmark loop; interactive prompting stays the default when it
  is unset.
* `put` reads the record body from stdin (`--stdin`) and encrypts it. The body is
  opaque JSON to BaruDB.
* Multiple `--keyword` flags to `search` mean **conjunction** (AND). Baru needs AND;
  OR can be done client-side by unioning result sets.
* Exit code 0 on success, non-zero on failure, with a human-readable message on
  stderr. Machine-readable output on stdout only.
* `search` on an unknown keyword returns `{"ids": []}`, not an error.

### 2b. HTTP transport (preferred once it exists)

```
POST /v1/records          {keywords: [..], body: <base64 ciphertext or JSON>} → {id}
GET  /v1/records/{id}                                                        → record
POST /v1/search           {keywords: [..], limit, after}                      → {ids: [..]}
GET  /v1/stats                                                               → {records, bytes}
```

Baru already runs an axum control plane (`src/rest.rs`), so mirroring that shape
keeps the two services operationally identical.

---

## 3. What Baru guarantees

* **Baru encrypts nothing itself.** Confidentiality is BaruDB's job; Baru hands
  over plaintext JSON plus a keyword list over a local channel. This keeps a
  single implementation of the cryptography, in the repo that was designed for it.
* **Idempotence by id.** Baru generates the record id; re-`put`ting the same id is
  a no-op or an error, never a silent duplicate.
* **Bounded record size.** A run record is < 64 KiB. Large artefacts (ciphertexts,
  key material, model weights) are *not* stored in BaruDB — they go to the
  content-addressed store in `src/storage.rs`, and the run record carries the
  Merkle root. That keeps BaruDB's index small and gives us proof-of-retrievability
  on the heavy data for free.
* **No secrets in keywords.** Keywords are metadata only. Anything whose leakage
  matters (weights, per-sample timings) lives in the encrypted payload. This is
  the honest statement of the leakage profile: BaruDB's deterministic tokens leak
  keyword equality and access patterns, and Baru's keyword vocabulary is chosen so
  that this leakage is acceptable.

## 4. What Baru does *not* need (please do not build for us)

* Range/order queries over metrics — client-side is fine at our scale
  (thousands of records, not millions).
* Multi-writer concurrency. One benchmark process writes at a time.
* Deletion. Append-only is a feature for reproducibility.
* Transactions.

## 5. Baru-side surface

```
src/db/
  mod.rs      ResultStore trait, `baru db` CLI
  record.rs   Run / Metric / Provenance schema (serde)
  local.rs    JSONL store — the always-available default
  blindsearch.rs  BaruDB client over §2a, then §2b
  query.rs    client-side filtering, aggregation, CSV/LaTeX export
```

```
baru db put    --suite bench --case relu --backend mpc-aby3 --metric eval_us=134 ...
baru db query  --backend fhe-tfhe --case mnist-linear --format table|csv|latex|json
baru db stats
baru db export --suite bench-cnn --format latex > paper/tables/cnn.tex
```

Benchmark harnesses gain `--store <local:path|blindsearch:path|blindsearch:url>`;
with it, every `baru bench*` run is persisted automatically. Paper tables are then
generated from the store rather than pasted from scrollback — which is the whole
point: **the numbers in the paper are queries, not transcriptions.**

## 6. Extension points

* **Range-capable index.** If BaruDB gains encrypted range queries, Baru starts
  emitting bucketed numeric keywords (`metric:eval_us:bucket:1e5..1e6`) and pushes
  the coarse filter server-side, keeping exact comparison client-side.
* **Multi-client / sharing.** Publishing means letting a reviewer verify a table.
  The natural shape is a read-only key that decrypts a *subset* (one suite), which
  argues for per-suite key derivation in BaruDB rather than one database key.
* **Proof of retrievability.** `src/storage.rs` already does Merkle-committed
  epoch challenges. Pointing it at the BaruDB file makes "the experiment archive
  is still intact" a checkable claim.
