//! The experiment store — durable, queryable results for long-running
//! privacy-preserving-ML experiments.
//!
//! A single TFHE MNIST inference takes ~47 minutes; a CKKS-vs-TFHE sweep runs
//! for days.  Results that live only in terminal scrollback cannot be cited,
//! re-checked, or compared against a run from three months ago — and they are
//! themselves sensitive (unpublished numbers, model weights, timing data).  So
//! Baru writes every run to a store:
//!
//! * [`local::JsonlStore`] — a plain append-only file.  Always available, never
//!   the reason a benchmark loses its result.
//! * [`barudb::BaruDbStore`] — **BaruDB**
//!   (<https://github.com/emmorais/barudb>, crate and binary `barudb`),
//!   the encrypted searchable store, driven over a pipe.  Baru supplies the
//!   schema and the keyword vocabulary; BaruDB owns the cryptography, and the
//!   process boundary is what keeps that division real rather than merely
//!   conventional.  The contract is `docs/barudb-api.md`.
//!
//! Both sit behind [`ResultStore`], so harnesses take a `--store` string and
//! neither knows nor cares which one it got.
//!
//! ## CLI
//!
//! ```text
//! baru db put    --suite bench --case relu --backend mpc-aby3 --metric eval_us=134us
//! baru db query  --backend fhe-tfhe --format table
//! baru db export --suite bench-cnn --format latex > paper/tables/cnn.tex
//! baru db stats
//! ```

pub mod barudb;
pub mod cohort;
pub mod local;
pub mod query;
pub mod record;

use serde::Serialize;

pub use barudb::BaruDbStore;
pub use cohort::{CohortDb, Column as CohortColumn};
pub use local::JsonlStore;
pub use query::{aggregate, render, Filter, Format, Group};
pub use record::{Metric, Provenance, Run};

/// Size and nature of a store, for `baru db stats`.
#[derive(Debug, Clone, Serialize)]
pub struct StoreStats {
    pub records: u64,
    pub bytes: u64,
    /// Whether the payload is encrypted at rest.  Reported so a run that
    /// *should* have gone to BaruDB and silently went to a file is visible.
    pub encrypted: bool,
    pub backend: String,
}

/// What Baru needs from an experiment store.
///
/// Deliberately small: append, fetch by id, conjunctive keyword search, and a
/// full scan.  No deletion (append-only is a reproducibility feature), no
/// transactions, no numeric predicates — see `docs/barudb-api.md` §4 for why
/// each of those is out of scope.
pub trait ResultStore {
    /// Append a record.  Ids are minted by Baru, so re-putting one is an error
    /// rather than a silent duplicate.
    fn put(&mut self, run: &Run) -> Result<String, String>;
    fn get(&self, id: &str) -> Result<Option<Run>, String>;
    /// Records matching **every** keyword.  An unknown keyword yields an empty
    /// result, not an error.
    fn search(&self, keywords: &[String]) -> Result<Vec<Run>, String>;
    fn all(&self) -> Result<Vec<Run>, String>;
    fn stats(&self) -> Result<StoreStats, String>;
    /// The `--store` string that would reopen this store.
    fn describe(&self) -> String;
}

/// Open a store from a `--store` specifier.
///
/// * `local:PATH` (or a bare path) — [`JsonlStore`]
/// * `barudb:PATH` — [`BaruDbStore`]
///
/// Defaults to `local:baru-results.jsonl` when the specifier is empty.
pub fn open(spec: &str) -> Result<Box<dyn ResultStore>, String> {
    let spec = if spec.is_empty() {
        "local:baru-results.jsonl"
    } else {
        spec
    };
    match spec.split_once(':') {
        Some(("local", path)) => Ok(Box::new(JsonlStore::new(path))),
        Some(("barudb", path)) => Ok(Box::new(BaruDbStore::new(path))),
        Some((scheme, _)) if scheme.len() > 1 => Err(format!(
            "unknown store scheme `{}`; expected local:PATH or barudb:PATH",
            scheme
        )),
        // No scheme (or a Windows drive letter): treat it as a plain path.
        _ => Ok(Box::new(JsonlStore::new(spec))),
    }
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

const USAGE: &str = "\
baru db — the experiment store

  baru db put    --suite S --case C --backend B [record opts]
  baru db query  [filter opts] [--format table|csv|latex|json] [--raw]
  baru db export [filter opts] --format latex        (alias for query)
  baru db get    --id ID
  baru db stats

  --store local:PATH | barudb:PATH          (default local:baru-results.jsonl)

record opts
  --metric name=value[unit]   repeatable
  --config key=value          repeatable
  --tag name                  repeatable
  --ok true|false
  --note TEXT
  --artifact-root HEX         Merkle root of bulk data held in `baru::storage`

filter opts
  --suite S  --case C  --backend B  --tag T  --commit SHA  --ok true|false
  --metric-col name           restrict the columns shown (repeatable)
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("");
    let rest = if args.is_empty() { args } else { &args[1..] };
    match sub {
        "" | "help" | "--help" | "-h" => {
            print!("{}", USAGE);
            Ok(())
        }
        "put" => cmd_put(rest),
        "query" | "export" => cmd_query(rest),
        "get" => cmd_get(rest),
        "stats" => cmd_stats(rest),
        other => Err(format!("unknown subcommand `{}`\n\n{}", other, USAGE)),
    }
}

/// Minimal flag reader shared by the subcommands: every flag takes a value
/// except the ones listed in `bare`.
fn flags(args: &[String], bare: &[&str]) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let f = &args[i];
        if !f.starts_with("--") {
            return Err(format!("unexpected argument `{}`\n\n{}", f, USAGE));
        }
        if bare.contains(&f.as_str()) {
            out.push((f.clone(), String::new()));
            i += 1;
        } else {
            let v = args
                .get(i + 1)
                .ok_or_else(|| format!("{} needs a value", f))?;
            out.push((f.clone(), v.clone()));
            i += 2;
        }
    }
    Ok(out)
}

fn store_of(fs: &[(String, String)]) -> Result<Box<dyn ResultStore>, String> {
    let spec = fs
        .iter()
        .find(|(f, _)| f == "--store")
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    open(&spec)
}

fn parse_bool(v: &str) -> Result<bool, String> {
    match v {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => Err(format!("expected true/false, got `{}`", other)),
    }
}

fn cmd_put(args: &[String]) -> Result<(), String> {
    let fs = flags(args, &[])?;
    let get = |name: &str| fs.iter().find(|(f, _)| f == name).map(|(_, v)| v.as_str());
    let suite = get("--suite").ok_or("--suite is required")?;
    let case = get("--case").ok_or("--case is required")?;
    let backend = get("--backend").ok_or("--backend is required")?;

    let mut run = Run::new(suite, case, backend);
    for (f, v) in &fs {
        match f.as_str() {
            "--metric" => {
                let (n, m) = Metric::parse_assignment(v)?;
                run.metrics.insert(n, m);
            }
            "--config" => {
                let (k, val) = v
                    .split_once('=')
                    .ok_or_else(|| format!("config `{}` must look like key=value", v))?;
                run.config.insert(k.to_string(), val.to_string());
            }
            "--tag" => run.tags.push(v.clone()),
            "--ok" => run.ok = Some(parse_bool(v)?),
            "--note" => run.notes = v.clone(),
            "--artifact-root" => run.artifact_root = Some(v.clone()),
            "--suite" | "--case" | "--backend" | "--store" => {}
            other => return Err(format!("unknown flag `{}`\n\n{}", other, USAGE)),
        }
    }

    let mut store = store_of(&fs)?;
    let id = store.put(&run)?;
    println!("{}  ->  {}", id, store.describe());
    Ok(())
}

fn cmd_query(args: &[String]) -> Result<(), String> {
    let fs = flags(args, &["--raw"])?;
    let mut filter = Filter::default();
    let mut format = Format::Table;
    let mut cols: Vec<String> = Vec::new();
    let mut raw = false;
    for (f, v) in &fs {
        match f.as_str() {
            "--suite" => filter.suite = Some(v.clone()),
            "--case" => filter.case = Some(v.clone()),
            "--backend" => filter.backend = Some(v.clone()),
            "--commit" => filter.commit = Some(v.clone()),
            "--tag" => filter.tags.push(v.clone()),
            "--ok" => filter.ok = Some(parse_bool(v)?),
            "--metric-col" => cols.push(v.clone()),
            "--format" => {
                format = Format::parse(v).ok_or_else(|| format!("unknown format `{}`", v))?
            }
            "--raw" => raw = true,
            "--store" => {}
            other => return Err(format!("unknown flag `{}`\n\n{}", other, USAGE)),
        }
    }
    let store = store_of(&fs)?;
    let runs = if filter.is_empty() {
        store.all()?
    } else {
        store.search(&filter.keywords())?
    };
    if raw {
        println!(
            "{}",
            serde_json::to_string_pretty(&runs).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if runs.is_empty() {
        eprintln!("no records match {:?}", filter.keywords());
    }
    print!("{}", render(&aggregate(&runs), &cols, format));
    Ok(())
}

fn cmd_get(args: &[String]) -> Result<(), String> {
    let fs = flags(args, &[])?;
    let id = fs
        .iter()
        .find(|(f, _)| f == "--id")
        .map(|(_, v)| v.clone())
        .ok_or("--id is required")?;
    let store = store_of(&fs)?;
    match store.get(&id)? {
        Some(run) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&run).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        None => Err(format!("no record with id `{}`", id)),
    }
}

fn cmd_stats(args: &[String]) -> Result<(), String> {
    let fs = flags(args, &[])?;
    let store = store_of(&fs)?;
    let s = store.stats()?;
    println!("backend    {}", s.backend);
    println!("records    {}", s.records);
    println!("bytes      {}", s.bytes);
    println!(
        "encrypted  {}{}",
        s.encrypted,
        if s.encrypted {
            ""
        } else {
            "   (results are stored in the clear — use barudb: for anything publishable)"
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_specifiers_resolve_to_the_right_backend() {
        assert!(open("local:/tmp/x.jsonl")
            .unwrap()
            .describe()
            .starts_with("local:"));
        assert!(open("barudb:/tmp/x.db")
            .unwrap()
            .describe()
            .starts_with("barudb:"));
        // The old `blindsearch:` scheme is gone, not silently accepted.
        assert!(open("blindsearch:/tmp/x.db").is_err());
        // A bare path is a local file.
        assert!(open("results.jsonl")
            .unwrap()
            .describe()
            .starts_with("local:"));
        assert!(open("").unwrap().describe().contains("baru-results.jsonl"));
        assert!(open("s3:/bucket/x").is_err());
    }

    #[test]
    fn cli_put_then_query_round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runs.jsonl");
        let store = format!("local:{}", path.display());
        let argv = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };

        run_cli(&argv(&[
            "put",
            "--store",
            &store,
            "--suite",
            "bench",
            "--case",
            "relu",
            "--backend",
            "mpc-aby3",
            "--metric",
            "eval_us=134us",
            "--ok",
            "true",
        ]))
        .expect("put");

        let s = open(&store).unwrap();
        let runs = s.search(&["case:relu".into()]).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].metrics["eval_us"].value, 134.0);
        assert_eq!(runs[0].ok, Some(true));
        assert_eq!(s.stats().unwrap().records, 1);
        assert!(!s.stats().unwrap().encrypted);

        run_cli(&argv(&["query", "--store", &store, "--format", "csv"])).expect("query");
        run_cli(&argv(&["get", "--store", &store, "--id", &runs[0].id])).expect("get");
        assert!(run_cli(&argv(&["get", "--store", &store, "--id", "nope"])).is_err());
    }

    #[test]
    fn missing_required_flags_are_reported() {
        assert!(run_cli(&["put".to_string()]).is_err());
        assert!(run_cli(&["put".to_string(), "--suite".to_string()]).is_err());
        assert!(run_cli(&["nonsense".to_string()]).is_err());
        assert!(run_cli(&[]).is_ok());
    }
}
