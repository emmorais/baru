# Prompt for the barudb repo

Paste the block below into a Claude Code session opened in
`~/github/barudb` (or wherever the checkout lives). It is written to be
self-contained: it restates the contract rather than assuming the other repo can
read `baru/docs/barudb-api.md`.

---

I want to add a **store API** to barudb so that another project of mine
(Baru, an MPC/FHE/ZKP compiler) can use it as a durable encrypted archive for
long-running experiment results. Baru's client is already written and expects a
specific interface, so please implement exactly this and not a variation.

## Context

Baru runs privacy-preserving-ML benchmarks that take minutes to hours per case
(a single TFHE MNIST inference is ~47 min). Each run produces a small JSON
record of timings and metadata that has to survive for months so it can end up
in a paper. The records are sensitive: unpublished numbers, model configuration,
timing data. barudb already does the hard part — AES-256-GCM with Argon2id
key derivation and deterministic keyword tokens — so it should own **all** of the
cryptography. Baru hands over plaintext JSON plus a keyword list and never
encrypts anything itself.

## What to add: five subcommands

```
barudb put    --db <path> --keywords k1,k2,... --stdin      → {"id": "..."}
barudb get    --db <path> --id <id>                         → the record's JSON
barudb search --db <path> --keyword k [--keyword k ...]     → {"ids": [...]}
barudb list   --db <path> [--limit N] [--after <id>]        → {"ids": [...]}
barudb stats  --db <path>                                   → {"records": N, "bytes": N}
```

Requirements the client depends on — these are not negotiable, the client will
break without them:

1. **`put` reads the record body from stdin** (`--stdin`) and encrypts it. The
   body is opaque JSON to barudb: parse it as UTF-8 bytes, do not validate
   or reshape it.
2. **Ids are caller-supplied.** The body contains an `"id"` field, a string like
   `1753605600123-9f3a1c04`, sortable so that lexicographic order is
   chronological. Use that as the record's key. `put` must echo it back
   unchanged in `{"id": ...}`. If you mint your own id instead, the client
   errors out — it needs `get --id` to work with the id it generated. Extract it
   with a minimal JSON parse of the body; do not require any other field.
3. **Records are immutable.** Putting an id that already exists is an error
   (non-zero exit, message on stderr), never a silent overwrite or duplicate.
   Append-only is a reproducibility feature here, so also: **no delete
   subcommand.**
4. **Multiple `--keyword` flags to `search` mean AND**, not OR. Baru unions
   client-side when it wants OR.
5. **A search for an unknown keyword returns `{"ids": []}` and exit code 0** —
   an empty result is not an error.
6. **`get` on a missing id** should exit non-zero with the word `not found` in
   the stderr message (the client special-cases that string to return "no such
   record" rather than propagating a failure).
7. **Machine-readable output on stdout only**, one JSON object, nothing else.
   Human-facing messages, progress, and warnings go to stderr. Anything on
   stdout that is not the JSON object breaks parsing.
8. **Password from `BARUDB_PASSWORD` when that variable is set**, falling
   back to the existing interactive prompt when it is not. (The binary reads
   `BLINDSEARCH_PASSWORD` today — `PASSWORD_ENV` in `src/main.rs`. Please switch
   the primary name to `BARUDB_PASSWORD` to match the crate, and keep accepting
   the old one for a while so existing scripts keep working.) Baru runs these
   commands inside unattended benchmark loops that may last hours, so there is
   nobody to answer a prompt. Please keep the prompt as the default for
   interactive use — only the env var should bypass it.
9. **Exit code 0 on success, non-zero on any failure.**

## Keywords

Baru emits keywords as lowercase `field:value` strings, for example:

```
suite:bench   case:mnist-linear   backend:fhe-tfhe   ok:true
tag:paper-table-3   host:eduardo-mbp   commit:981e639
```

They are always deterministic and always metadata — never a metric value, never
anything from the record's configuration or notes. Baru chose that vocabulary
precisely so that the leakage from deterministic tokens (keyword equality and
access patterns) is acceptable. Please keep the existing deterministic-token
construction; no new cryptography is needed for this.

## Scale and what *not* to build

Thousands of records, not millions. One writer at a time. Records are under
64 KiB each — bulk artefacts (ciphertexts, keys, model weights) are stored
elsewhere and referenced by hash, so the index stays small.

So please do **not** build, for this API: range or ordering queries over
encrypted values, multi-writer concurrency control, transactions, or deletion.
If a design decision is a trade-off between simplicity and generality here,
choose simplicity.

## Deliverables

1. The five subcommands, wired into the existing CLI alongside `demo` and
   `legacy` without changing their behaviour.
2. Integration tests covering: put→get round-trip; conjunctive search; unknown
   keyword returns an empty list with exit 0; duplicate id is rejected; `get` on
   a missing id exits non-zero saying "not found"; `BARUDB_PASSWORD` works
   non-interactively; stdout contains only the JSON object.
3. A short section in README.md documenting the five commands, and a note in
   DESIGN.md stating the leakage profile of this API in one paragraph (what an
   adversary with the database file learns: number of records, record sizes,
   keyword equality patterns, and query access patterns).
4. Tell me if anything above conflicts with the existing design — I would rather
   change Baru's client than have barudb grow an awkward shape.

## Later, not now

Once this works I will want an HTTP transport mirroring the same five
operations (`POST /v1/records`, `GET /v1/records/{id}`, `POST /v1/search`,
`GET /v1/stats`), because Baru already runs an axum control plane and having the
two services look alike would simplify operations. Don't build it yet — just
avoid designing anything that would make it awkward.
