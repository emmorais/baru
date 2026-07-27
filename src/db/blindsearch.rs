//! Client for **BaruDB** (<https://github.com/emmorais/blindsearch>) — the
//! encrypted, searchable experiment store.
//!
//! This implements the process transport of `docs/barudb-api.md` §2a:
//!
//! ```text
//! blindsearch put    --db <path> --keywords k1,k2 --stdin   → {"id": "..."}
//! blindsearch get    --db <path> --id <id>                  → record JSON
//! blindsearch search --db <path> --keyword k [--keyword k]  → {"ids": [...]}
//! blindsearch list   --db <path>                            → {"ids": [...]}
//! blindsearch stats  --db <path>                            → {"records":N,"bytes":N}
//! ```
//!
//! Baru hands BaruDB **plaintext JSON plus a keyword list** and lets BaruDB do
//! the cryptography.  That is deliberate: there should be exactly one
//! implementation of searchable encryption, in the repository designed for it,
//! rather than a second half-implementation here.  What Baru owns is the
//! *schema* and the *keyword vocabulary* — see [`Run::keywords`] for the
//! leakage-relevant part.
//!
//! Until blindsearch ships those subcommands this client will fail at the first
//! call with the command it tried to run, which is the intended behaviour: the
//! contract is executable documentation, and [`crate::db::local::JsonlStore`]
//! is the fallback that keeps benchmarking unblocked in the meantime.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::Value;

use super::record::Run;
use super::{ResultStore, StoreStats};

pub struct BlindSearchStore {
    /// Path to the encrypted database that blindsearch manages.
    db: PathBuf,
    /// The `blindsearch` executable; overridable via `BLINDSEARCH_BIN`.
    binary: String,
}

impl BlindSearchStore {
    pub fn new(db: impl Into<PathBuf>) -> Self {
        BlindSearchStore {
            db: db.into(),
            binary: std::env::var("BLINDSEARCH_BIN").unwrap_or_else(|_| "blindsearch".to_string()),
        }
    }

    /// Run a subcommand, optionally piping `stdin`, and return stdout.
    fn run(&self, args: &[String], stdin: Option<&str>) -> Result<String, String> {
        let mut cmd = Command::new(&self.binary);
        cmd.arg(args[0].as_str())
            .arg("--db")
            .arg(&self.db)
            .args(&args[1..])
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
                 BaruDB is not available.  Install blindsearch (or set BLINDSEARCH_BIN), \
                 or use `--store local:<path>`.  The expected interface is documented in \
                 docs/barudb-api.md §2a.",
                self.binary,
                args.join(" "),
                e
            )
        })?;

        if let Some(body) = stdin {
            child
                .stdin
                .as_mut()
                .ok_or("cannot write to blindsearch stdin")?
                .write_all(body.as_bytes())
                .map_err(|e| format!("cannot write record to blindsearch: {}", e))?;
        }

        let out = child
            .wait_with_output()
            .map_err(|e| format!("blindsearch failed: {}", e))?;
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

    fn ids_from(&self, stdout: &str) -> Result<Vec<String>, String> {
        let v: Value = serde_json::from_str(stdout.trim())
            .map_err(|e| format!("blindsearch returned non-JSON output: {} ({})", e, stdout))?;
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

impl ResultStore for BlindSearchStore {
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
            .map_err(|e| format!("blindsearch put returned non-JSON output: {} ({})", e, out))?;
        match v.get("id").and_then(Value::as_str) {
            Some(id) if id == run.id => Ok(run.id.clone()),
            Some(id) => Err(format!(
                "blindsearch assigned id `{}` but Baru minted `{}`; ids must be caller-supplied \
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
            .map_err(|e| format!("blindsearch stats returned non-JSON output: {}", e))?;
        Ok(StoreStats {
            records: v.get("records").and_then(Value::as_u64).unwrap_or(0),
            bytes: v.get("bytes").and_then(Value::as_u64).unwrap_or(0),
            encrypted: true,
            backend: format!("blindsearch ({})", self.db.display()),
        })
    }

    fn describe(&self) -> String {
        format!("blindsearch:{}", self.db.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_blindsearch_binary_says_what_to_do_instead() {
        std::env::set_var("BLINDSEARCH_BIN", "blindsearch-definitely-not-installed");
        let store = BlindSearchStore::new("/tmp/nope.db");
        let err = store.stats().unwrap_err();
        std::env::remove_var("BLINDSEARCH_BIN");
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
