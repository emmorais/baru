//! Client for **BaruDB** (<https://github.com/emmorais/barudb>, crate and
//! binary `barudb`) — the encrypted, searchable experiment store.
//!
//! This implements the process transport of `docs/barudb-api.md` §2a:
//!
//! ```text
//! barudb put    --db <path> --keywords k1,k2 --stdin   → {"id": "..."}
//! barudb get    --db <path> --id <id>                  → record JSON
//! barudb search --db <path> --keyword k [--keyword k]  → {"ids": [...]}
//! barudb list   --db <path>                            → {"ids": [...]}
//! barudb stats  --db <path>                            → {"records":N,"bytes":N}
//! ```
//!
//! ## Why a process boundary rather than a Cargo dependency
//!
//! BaruDB *is* a library crate, so Baru could link it directly.  It talks to it
//! over a pipe instead, on purpose:
//!
//! * **The key never enters Baru's address space.**  `docs/barudb-api.md` says
//!   Baru hands over plaintext and BaruDB owns the cryptography.  Across a
//!   process boundary that is a fact; inside one binary it would be a
//!   convention, one `pub` away from being violated by accident.
//! * **No dependency-tree coupling.**  BaruDB brings redb, argon2, aes-gcm,
//!   arkworks and bulletproofs; Baru brings SEAL (a C++ build), tfhe-rs and
//!   binius64.  Neither project should pay for the other's build.
//! * **The cost is irrelevant here.**  A benchmark run writes ~30 records of
//!   <64 KiB against workloads measured in minutes to hours, so ~30 process
//!   spawns cost well under a second.  If per-sample logging ever pushes that to
//!   thousands of records, the fix is a batch `put`, not linking.
//!
//! Baru never supplies the password: `barudb` reads it from its own environment
//! or prompts, and Baru only inherits whatever is set.  That variable is still
//! named `BLINDSEARCH_PASSWORD` in the `barudb` binary; see the open item in
//! `docs/barudb-api.md` §2a.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::Value;

use super::record::Run;
use super::{ResultStore, StoreStats};

/// Driving the `barudb` executable: shared by the experiment store here and by
/// the patient-data client in [`crate::db::cohort`].
///
/// The database is identified by a *string*, not a path, because BaruDB makes
/// the plaintext mode part of a database's identity (`plain:<path>`) rather
/// than a command-line flag.  That is a better design than the `--plain` flag
/// this side originally asked for: there is no flag to forget, and none left
/// over on a previous command line to reinterpret a real database.  Keeping it
/// a string here preserves that property instead of quietly normalising it away.
#[derive(Clone)]
pub struct Cli {
    pub db: String,
    /// The `barudb` executable; overridable via `BARUDB_BIN`.
    pub binary: String,
}

impl Cli {
    pub fn new(db: impl Into<String>) -> Self {
        Cli {
            db: db.into(),
            binary: std::env::var("BARUDB_BIN").unwrap_or_else(|_| "barudb".to_string()),
        }
    }

    /// True unless this is a `plain:` database.
    pub fn encrypted(&self) -> bool {
        !self.db.starts_with("plain:")
    }

    /// Run a subcommand, optionally piping `stdin`, and return stdout.
    pub fn run(&self, args: &[String], stdin: Option<&str>) -> Result<String, String> {
        self.run_nested(1, args, stdin)
    }

    /// As [`Cli::run`], but `lead` leading words come before `--db`, so that a
    /// sub-subcommand like `table create` can be expressed.
    pub fn run_nested(
        &self,
        lead: usize,
        args: &[String],
        stdin: Option<&str>,
    ) -> Result<String, String> {
        let mut cmd = Command::new(&self.binary);
        cmd.args(&args[..lead])
            .arg("--db")
            .arg(&self.db)
            .args(&args[lead..])
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "cannot run `{} {}`: {}\n\
                 BaruDB is not available.  Put `barudb` on PATH (or point BARUDB_BIN at \
                 the built binary), or use `--store local:<path>`.  The expected interface \
                 is documented in docs/barudb-api.md §2a.",
                self.binary,
                args.join(" "),
                e
            )
        })?;

        if let Some(body) = stdin {
            child
                .stdin
                .as_mut()
                .ok_or("cannot write to barudb stdin")?
                .write_all(body.as_bytes())
                .map_err(|e| format!("cannot write record to barudb: {}", e))?;
        }

        let out = child
            .wait_with_output()
            .map_err(|e| format!("barudb failed: {}", e))?;
        if !out.status.success() {
            return Err(format!(
                "`{} {}` exited with {}: {}",
                self.binary,
                args.join(" "),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    }

    /// Parse stdout as a JSON object, naming the command in any error.
    pub fn json(&self, stdout: &str, what: &str) -> Result<Value, String> {
        serde_json::from_str(stdout.trim()).map_err(|e| {
            format!(
                "barudb {} returned non-JSON output: {} ({})",
                what, e, stdout
            )
        })
    }
}

pub struct BaruDbStore {
    cli: Cli,
}

impl BaruDbStore {
    pub fn new(db: impl Into<PathBuf>) -> Self {
        BaruDbStore {
            cli: Cli::new(db.into().display().to_string()),
        }
    }

    fn run(&self, args: &[String], stdin: Option<&str>) -> Result<String, String> {
        self.cli.run(args, stdin)
    }

    fn ids_from(&self, stdout: &str) -> Result<Vec<String>, String> {
        let v: Value = serde_json::from_str(stdout.trim())
            .map_err(|e| format!("barudb returned non-JSON output: {} ({})", e, stdout))?;
        let ids = v
            .get("ids")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("expected {{\"ids\": [...]}}, got: {}", stdout))?;
        Ok(ids
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect())
    }

    fn fetch_all(&self, ids: &[String]) -> Result<Vec<Run>, String> {
        let mut runs = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(r) = self.get(id)? {
                runs.push(r);
            }
        }
        Ok(runs)
    }
}

impl ResultStore for BaruDbStore {
    fn put(&mut self, run: &Run) -> Result<String, String> {
        let body = serde_json::to_string(run).map_err(|e| e.to_string())?;
        let args = vec![
            "put".to_string(),
            "--keywords".to_string(),
            run.keywords().join(","),
            "--stdin".to_string(),
        ];
        let out = self.run(&args, Some(&body))?;
        // The id is Baru's, per docs/barudb-api.md §3; echoing a different one
        // back means the two sides disagree about identity, which would silently
        // break every later `get`.
        let v: Value = serde_json::from_str(out.trim())
            .map_err(|e| format!("barudb put returned non-JSON output: {} ({})", e, out))?;
        match v.get("id").and_then(Value::as_str) {
            Some(id) if id == run.id => Ok(run.id.clone()),
            Some(id) => Err(format!(
                "barudb assigned id `{}` but Baru minted `{}`; ids must be caller-supplied \
                 (docs/barudb-api.md §3)",
                id, run.id
            )),
            None => Ok(run.id.clone()),
        }
    }

    fn get(&self, id: &str) -> Result<Option<Run>, String> {
        let args = vec!["get".to_string(), "--id".to_string(), id.to_string()];
        match self.run(&args, None) {
            Ok(out) if out.trim().is_empty() => Ok(None),
            Ok(out) => serde_json::from_str(out.trim())
                .map(Some)
                .map_err(|e| format!("record {} is not a Baru run record: {}", id, e)),
            // A missing id is not an error condition for the caller.
            Err(e) if e.contains("not found") => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn search(&self, keywords: &[String]) -> Result<Vec<Run>, String> {
        let mut args = vec!["search".to_string()];
        for k in keywords {
            args.push("--keyword".to_string());
            args.push(k.to_lowercase());
        }
        let ids = self.ids_from(&self.run(&args, None)?)?;
        self.fetch_all(&ids)
    }

    fn all(&self) -> Result<Vec<Run>, String> {
        let ids = self.ids_from(&self.run(&["list".to_string()], None)?)?;
        self.fetch_all(&ids)
    }

    fn stats(&self) -> Result<StoreStats, String> {
        let out = self.run(&["stats".to_string()], None)?;
        let v: Value = serde_json::from_str(out.trim())
            .map_err(|e| format!("barudb stats returned non-JSON output: {}", e))?;
        Ok(StoreStats {
            records: v.get("records").and_then(Value::as_u64).unwrap_or(0),
            bytes: v.get("bytes").and_then(Value::as_u64).unwrap_or(0),
            // BaruDB reports this itself now, and a `plain:` database is not
            // encrypted — taking its word for it beats assuming.
            encrypted: v
                .get("encrypted")
                .and_then(Value::as_bool)
                .unwrap_or_else(|| self.cli.encrypted()),
            backend: format!("barudb ({})", self.cli.db),
        })
    }

    fn describe(&self) -> String {
        format!("barudb:{}", self.cli.db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_barudb_binary_says_what_to_do_instead() {
        std::env::set_var("BARUDB_BIN", "barudb-definitely-not-installed");
        let store = BaruDbStore::new("/tmp/nope.db");
        let err = store.stats().unwrap_err();
        std::env::remove_var("BARUDB_BIN");
        assert!(err.contains("--store local:"), "{}", err);
        assert!(err.contains("docs/barudb-api.md"), "{}", err);
    }

    #[test]
    fn keywords_sent_to_the_store_are_the_records_own() {
        // The client must not invent its own vocabulary: whatever leaks is
        // exactly what `Run::keywords` decided to expose.
        let run = Run::new("bench", "relu", "mpc-aby3").with_tag("wan");
        let joined = run.keywords().join(",");
        assert!(joined.contains("suite:bench"));
        assert!(joined.contains("tag:wan"));
        assert!(!joined.contains("notes"));
    }
}
