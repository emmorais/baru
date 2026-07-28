//! Patient-data client — stage 1 of the healthcare workflow in
//! `docs/research-agenda.md` §2.
//!
//! Where [`crate::db::barudb::BaruDbStore`] archives *our own* experiment
//! results, this drives BaruDB's relational side: declare a table, insert rows,
//! and ask cohort questions ("how many patients aged 40–60 with HbA1c > 7 and
//! diagnosis E11?") without the server seeing the records.
//!
//! ## Why a plaintext mode matters more than it sounds
//!
//! BaruDB makes the plaintext mode part of a database's *identity* —
//! `plain:<path>` rather than a `--plain` flag — so a real database can never be
//! reinterpreted by a stray flag.  Baru uses that for two things:
//!
//! * **A baseline.**  "Encrypted cohort queries are affordable" is not a claim
//!   until there is a plaintext number to divide by.
//! * **A differential oracle.**  The same rows and the same predicate must give
//!   the same count both ways.  That check is worth more than the timing: it is
//!   the strongest correctness test the encrypted query path has, and it costs
//!   nothing to run alongside the benchmark.
//!
//! ## Reading the timings honestly
//!
//! BaruDB reports `crypto_us` *and* `key_derivation_us` separately, which turns
//! out to matter a great deal.  Argon2id key derivation is ~99% of `crypto_us`
//! and is paid **once per process**, not once per query.  A harness that spawns
//! a process per query and reports `crypto_us` as "the cost of encryption"
//! would be reporting the cost of *starting up* — off by two orders of
//! magnitude.  [`Timing::marginal_crypto_us`] is the number that actually scales
//! with the query.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::barudb::Cli;

/// A column declaration: `name:type:mode`.
///
/// The `mode` is a leakage decision, not a performance one — `ore` admits range
/// queries and reveals the order of the column's values; `det` admits equality
/// and reveals which rows share a value; `opaque` admits neither and reveals
/// nothing beyond length.  Keeping them in one type keeps that choice visible
/// at the call site rather than buried in a schema string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub ty: String,
    pub mode: String,
}

impl Column {
    pub fn new(name: &str, ty: &str, mode: &str) -> Self {
        Column {
            name: name.to_string(),
            ty: ty.to_string(),
            mode: mode.to_string(),
        }
    }

    fn spec(&self) -> String {
        format!("{}:{}:{}", self.name, self.ty, self.mode)
    }
}

/// What one BaruDB invocation spent, in microseconds.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Timing {
    pub total_us: u64,
    pub crypto_us: u64,
    pub io_us: u64,
    pub key_derivation_us: u64,
}

impl Timing {
    fn parse(v: &Value) -> Timing {
        let t = v.get("timing").unwrap_or(&Value::Null);
        let get = |k: &str| t.get(k).and_then(Value::as_u64).unwrap_or(0);
        Timing {
            total_us: get("total_us"),
            crypto_us: get("crypto_us"),
            io_us: get("io_us"),
            key_derivation_us: get("key_derivation_us"),
        }
    }

    /// Cryptographic work that scales with the query, i.e. excluding the
    /// once-per-process key derivation.  This is the term to quote.
    pub fn marginal_crypto_us(&self) -> u64 {
        self.crypto_us.saturating_sub(self.key_derivation_us)
    }

    /// Total minus key derivation — comparable across processes.
    pub fn marginal_total_us(&self) -> u64 {
        self.total_us.saturating_sub(self.key_derivation_us)
    }
}

/// A BaruDB database holding tabular data.
pub struct CohortDb {
    cli: Cli,
}

impl CohortDb {
    /// `db` is passed to BaruDB verbatim, so `plain:<path>` selects the
    /// plaintext mode exactly as BaruDB defines it.
    pub fn new(db: impl Into<String>) -> Self {
        CohortDb { cli: Cli::new(db) }
    }

    pub fn encrypted(&self) -> bool {
        self.cli.encrypted()
    }

    pub fn describe(&self) -> &str {
        &self.cli.db
    }

    /// Declare a table.  Idempotent from Baru's point of view: re-declaring an
    /// existing table is reported by BaruDB and surfaced as an error here.
    pub fn create_table(&self, table: &str, columns: &[Column]) -> Result<(), String> {
        let schema = columns
            .iter()
            .map(Column::spec)
            .collect::<Vec<_>>()
            .join(",");
        let args = vec![
            "table".to_string(),
            "create".to_string(),
            "--name".to_string(),
            table.to_string(),
            "--schema".to_string(),
            schema,
        ];
        self.cli.run_nested(2, &args, None)?;
        Ok(())
    }

    /// Insert rows given as JSON lines.
    pub fn insert(&self, table: &str, rows_jsonl: &str) -> Result<(u64, Timing), String> {
        let args = vec![
            "insert".to_string(),
            "--table".to_string(),
            table.to_string(),
            "--stdin".to_string(),
            "--timing".to_string(),
        ];
        let out = self.cli.run(&args, Some(rows_jsonl))?;
        let v = self.cli.json(&out, "insert")?;
        Ok((
            v.get("rows").and_then(Value::as_u64).unwrap_or(0),
            Timing::parse(&v),
        ))
    }

    /// Which rows match `predicate`.
    ///
    /// A strictly larger disclosure than [`CohortDb::cohort_count`], and only
    /// appropriate where the caller is the data owner. The deployment this serves
    /// is `bench-cohort-eval`: the **hospital** selects a cohort out of its own
    /// records — so learning which of its own patients matched reveals nothing it
    /// did not already know — and the encrypted store learns nothing either way.
    /// The *model owner* never sees ids, only the cohort size, which is what makes
    /// the downstream private inference meaningful.
    ///
    /// Do not reach for this where the querier is not the data owner; that is what
    /// `cohort_count` is for.
    pub fn cohort_ids(&self, table: &str, predicate: &str) -> Result<(Vec<u64>, Timing), String> {
        let args = vec![
            "cohort".to_string(),
            "--table".to_string(),
            table.to_string(),
            "--where".to_string(),
            predicate.to_string(),
            "--ids".to_string(),
            "--timing".to_string(),
        ];
        let out = self.cli.run(&args, None)?;
        let v = self.cli.json(&out, "cohort")?;
        let ids = v
            .get("ids")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("barudb cohort --ids returned no ids: {}", out))?
            .iter()
            .map(|x| {
                // BaruDB returns row ordinals in insertion order, which is what
                // lets a caller line them up against its own records.
                x.as_u64()
                    .ok_or_else(|| format!("cohort id was not an integer: {x}"))
            })
            .collect::<Result<Vec<u64>, String>>()?;
        Ok((ids, Timing::parse(&v)))
    }

    /// How many rows match `predicate`.
    ///
    /// Only a count — returning row ids is a larger disclosure; use
    /// [`CohortDb::cohort_ids`] where the caller is the data owner and genuinely
    /// needs them.
    pub fn cohort_count(&self, table: &str, predicate: &str) -> Result<(u64, Timing), String> {
        let args = vec![
            "cohort".to_string(),
            "--table".to_string(),
            table.to_string(),
            "--where".to_string(),
            predicate.to_string(),
            "--count".to_string(),
            "--timing".to_string(),
        ];
        let out = self.cli.run(&args, None)?;
        let v = self.cli.json(&out, "cohort")?;
        let count = v
            .get("count")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("barudb cohort returned no count: {}", out))?;
        Ok((count, Timing::parse(&v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_database_is_recognised_as_unencrypted() {
        assert!(!CohortDb::new("plain:/tmp/x.redb").encrypted());
        assert!(CohortDb::new("/tmp/x.redb").encrypted());
    }

    #[test]
    fn column_specs_round_trip_into_the_schema_string() {
        let c = Column::new("hba1c", "fixed2", "ore");
        assert_eq!(c.spec(), "hba1c:fixed2:ore");
    }

    #[test]
    fn key_derivation_is_excluded_from_the_marginal_cost() {
        // The failure this guards against: quoting a 16 ms Argon2id startup as
        // if it were the per-query cost of encryption.
        let t = Timing {
            total_us: 30_299,
            crypto_us: 16_419,
            io_us: 4_381,
            key_derivation_us: 16_261,
        };
        assert_eq!(t.marginal_crypto_us(), 158);
        assert_eq!(t.marginal_total_us(), 14_038);
    }

    #[test]
    fn a_missing_binary_is_reported_with_the_fallback() {
        std::env::set_var("BARUDB_BIN", "barudb-definitely-not-installed");
        let db = CohortDb::new("/tmp/nope.redb");
        let err = db.cohort_count("patients", "age>1").unwrap_err();
        std::env::remove_var("BARUDB_BIN");
        assert!(err.contains("barudb"), "{}", err);
    }
}
