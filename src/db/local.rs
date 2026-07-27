//! The always-available store: one JSON object per line, appended.
//!
//! Baru must never be blocked on BaruDB being ready, and a benchmark that runs
//! for three hours must not lose its result because a database was
//! unreachable.  So the default store is a file: no daemon, no password, no
//! network, and trivially greppable when something goes wrong.
//!
//! It is *not* encrypted — that is the whole reason [`crate::db::blindsearch`]
//! exists.  Use this for scratch work and CI; use BaruDB for anything whose
//! numbers will end up in a paper.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use super::record::Run;
use super::{ResultStore, StoreStats};

pub struct JsonlStore {
    path: PathBuf,
}

impl JsonlStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        JsonlStore { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read every record.  Malformed lines are reported rather than skipped:
    /// silently dropping an experiment result is worse than failing loudly.
    fn read_all(&self) -> Result<Vec<Run>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&self.path)
            .map_err(|e| format!("cannot open {}: {}", self.path.display(), e))?;
        let mut runs = Vec::new();
        for (n, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("{}:{}: {}", self.path.display(), n + 1, e))?;
            if line.trim().is_empty() {
                continue;
            }
            let run: Run = serde_json::from_str(&line).map_err(|e| {
                format!("{}:{}: malformed record: {}", self.path.display(), n + 1, e)
            })?;
            runs.push(run);
        }
        Ok(runs)
    }
}

impl ResultStore for JsonlStore {
    fn put(&mut self, run: &Run) -> Result<String, String> {
        if self.get(&run.id)?.is_some() {
            return Err(format!(
                "record {} already exists — records are immutable",
                run.id
            ));
        }
        if let Some(dir) = self.path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("cannot create {}: {}", dir.display(), e))?;
            }
        }
        let line = serde_json::to_string(run).map_err(|e| e.to_string())?;
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("cannot open {}: {}", self.path.display(), e))?;
        writeln!(f, "{}", line).map_err(|e| format!("cannot write record: {}", e))?;
        Ok(run.id.clone())
    }

    fn get(&self, id: &str) -> Result<Option<Run>, String> {
        Ok(self.read_all()?.into_iter().find(|r| r.id == id))
    }

    fn search(&self, keywords: &[String]) -> Result<Vec<Run>, String> {
        Ok(self
            .read_all()?
            .into_iter()
            .filter(|r| r.matches(keywords))
            .collect())
    }

    fn all(&self) -> Result<Vec<Run>, String> {
        self.read_all()
    }

    fn stats(&self) -> Result<StoreStats, String> {
        let runs = self.read_all()?;
        let bytes = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        Ok(StoreStats {
            records: runs.len() as u64,
            bytes,
            encrypted: false,
            backend: format!("local jsonl ({})", self.path.display()),
        })
    }

    fn describe(&self) -> String {
        format!("local:{}", self.path.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store() -> (tempfile::TempDir, JsonlStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("runs.jsonl");
        (dir, JsonlStore::new(path))
    }

    #[test]
    fn put_get_search_round_trip() {
        let (_d, mut s) = tmp_store();
        let a = Run::new("bench", "relu", "mpc-aby3")
            .with_metric("eval_us", 134.0, "us")
            .with_ok(true);
        let b = Run::new("bench", "relu", "fhe-tfhe")
            .with_metric("eval_us", 2_800_000.0, "us")
            .with_ok(true);
        s.put(&a).unwrap();
        s.put(&b).unwrap();

        assert_eq!(s.get(&a.id).unwrap().as_ref(), Some(&a));
        assert_eq!(s.get("nope").unwrap(), None);
        assert_eq!(s.all().unwrap().len(), 2);

        let tfhe = s.search(&["backend:fhe-tfhe".into()]).unwrap();
        assert_eq!(tfhe.len(), 1);
        assert_eq!(tfhe[0].id, b.id);

        // Conjunction, and an unknown keyword yields nothing rather than an error.
        assert_eq!(
            s.search(&["backend:fhe-tfhe".into(), "case:relu".into()])
                .unwrap()
                .len(),
            1
        );
        assert!(s.search(&["case:nosuch".into()]).unwrap().is_empty());
    }

    #[test]
    fn records_are_immutable() {
        let (_d, mut s) = tmp_store();
        let a = Run::new("bench", "relu", "mpc-aby3");
        s.put(&a).unwrap();
        let err = s.put(&a).unwrap_err();
        assert!(err.contains("immutable"), "{}", err);
    }

    #[test]
    fn a_missing_file_is_an_empty_store_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let s = JsonlStore::new(dir.path().join("does-not-exist.jsonl"));
        assert!(s.all().unwrap().is_empty());
        assert_eq!(s.stats().unwrap().records, 0);
    }

    #[test]
    fn a_corrupt_line_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runs.jsonl");
        std::fs::write(&path, "{not json}\n").unwrap();
        let s = JsonlStore::new(path);
        let err = s.all().unwrap_err();
        assert!(err.contains("malformed record"), "{}", err);
    }
}
