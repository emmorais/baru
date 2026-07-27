//! The experiment record — one immutable row per (program, backend, config) run.
//!
//! The schema is split the way `docs/barudb-api.md` requires: a small set of
//! **keywords** that the encrypted store can search over, and an opaque
//! **payload** it can only hold and return.  Keeping that split explicit in the
//! type is what keeps the leakage profile honest — anything that ends up in
//! [`Run::keywords`] is metadata whose equality pattern the store learns.

use std::collections::BTreeMap;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// A measured quantity.  Units are carried alongside the value so a table can
/// be rendered without a lookup table of conventions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    pub value: f64,
    #[serde(default)]
    pub unit: String,
}

impl Metric {
    pub fn new(value: f64, unit: &str) -> Self {
        Metric {
            value,
            unit: unit.to_string(),
        }
    }

    /// Parse `name=value[unit]`, e.g. `eval_us=1340us` or `gates=2365`.
    pub fn parse_assignment(s: &str) -> Result<(String, Metric), String> {
        let (name, rest) = s
            .split_once('=')
            .ok_or_else(|| format!("metric `{}` must look like name=value[unit]", s))?;
        let split = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e'))
            .unwrap_or(rest.len());
        let (num, unit) = rest.split_at(split);
        let value: f64 = num
            .parse()
            .map_err(|_| format!("metric `{}` has a non-numeric value `{}`", s, num))?;
        Ok((name.to_string(), Metric::new(value, unit)))
    }
}

/// Where and from what code a run came.  Encrypted payload, never a keyword —
/// except the commit, which is useful enough to search on to be worth leaking.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub commit: String,
    pub dirty: bool,
    pub host: String,
    pub os: String,
    pub arch: String,
    pub profile: String,
    pub baru_version: String,
}

impl Provenance {
    /// Read the current checkout and machine.  Every field degrades to
    /// `"unknown"` rather than failing: a missing `git` must not lose a
    /// three-hour benchmark result.
    pub fn detect() -> Provenance {
        let git = |args: &[&str]| -> Option<String> {
            let out = Command::new("git").args(args).output().ok()?;
            if !out.status.success() {
                return None;
            }
            Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
        };
        Provenance {
            commit: git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into()),
            dirty: git(&["status", "--porcelain"])
                .map(|s| !s.is_empty())
                .unwrap_or(false),
            host: std::env::var("HOSTNAME")
                .ok()
                .or_else(|| {
                    Command::new("hostname")
                        .output()
                        .ok()
                        .filter(|o| o.status.success())
                        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                })
                .unwrap_or_else(|| "unknown".into()),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            profile: if cfg!(debug_assertions) {
                "debug".into()
            } else {
                "release".into()
            },
            baru_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// One run.  Append-only: a correction is a new record, never an edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// Sortable unique id — lexicographic order is chronological order.
    pub id: String,
    /// RFC3339 UTC.
    pub created_at: String,
    /// Harness that produced this, e.g. `bench`, `bench-cnn`, `mpcc-compare`.
    pub suite: String,
    /// Program or workload name.
    pub case: String,
    /// Backend / protocol, e.g. `mpc-aby3`, `fhe-tfhe`, `fhe-ckks`.
    pub backend: String,
    /// Whether the run passed its correctness check, when it had one.
    #[serde(default)]
    pub ok: Option<bool>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub provenance: Provenance,
    /// Free-form run configuration (sizes, party count, security parameters).
    #[serde(default)]
    pub config: BTreeMap<String, String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, Metric>,
    #[serde(default)]
    pub notes: String,
    /// Merkle root of any bulk artefact held in [`crate::storage`].  Heavy data
    /// (ciphertexts, keys, weights) never goes in the record itself — see
    /// `docs/barudb-api.md` §3.
    #[serde(default)]
    pub artifact_root: Option<String>,
}

impl Run {
    /// A new record with a freshly minted id and timestamp.
    pub fn new(suite: &str, case: &str, backend: &str) -> Run {
        let (id, created_at) = mint_id();
        Run {
            id,
            created_at,
            suite: suite.to_string(),
            case: case.to_string(),
            backend: backend.to_string(),
            ok: None,
            tags: Vec::new(),
            provenance: Provenance::detect(),
            config: BTreeMap::new(),
            metrics: BTreeMap::new(),
            notes: String::new(),
            artifact_root: None,
        }
    }

    pub fn with_metric(mut self, name: &str, value: f64, unit: &str) -> Self {
        self.metrics
            .insert(name.to_string(), Metric::new(value, unit));
        self
    }

    pub fn with_config(mut self, key: &str, value: impl ToString) -> Self {
        self.config.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn with_ok(mut self, ok: bool) -> Self {
        self.ok = Some(ok);
        self
    }

    /// The searchable metadata, as `field:value` tokens.
    ///
    /// This is *exactly* what the encrypted store learns the equality pattern
    /// of, so the vocabulary is deliberately small and deliberately
    /// non-sensitive: no metric values, no configuration, no notes.
    pub fn keywords(&self) -> Vec<String> {
        let mut k = vec![
            format!("suite:{}", self.suite.to_lowercase()),
            format!("case:{}", self.case.to_lowercase()),
            format!("backend:{}", self.backend.to_lowercase()),
        ];
        if let Some(ok) = self.ok {
            k.push(format!("ok:{}", ok));
        }
        if !self.provenance.commit.is_empty() {
            k.push(format!("commit:{}", self.provenance.commit));
        }
        if !self.provenance.host.is_empty() {
            k.push(format!("host:{}", self.provenance.host.to_lowercase()));
        }
        for t in &self.tags {
            k.push(format!("tag:{}", t.to_lowercase()));
        }
        k
    }

    /// Whether this record matches every keyword in `wanted` (conjunction),
    /// which is the filter semantics `docs/barudb-api.md` §2a specifies.
    pub fn matches(&self, wanted: &[String]) -> bool {
        let have = self.keywords();
        wanted
            .iter()
            .all(|w| have.iter().any(|h| h == &w.to_lowercase()))
    }
}

// ─── Ids and timestamps ───────────────────────────────────────────────────────

/// `{millis}-{random}`: unique, and lexicographically sortable by time, which
/// is what the store's `list --after` pagination relies on.
fn mint_id() -> (String, String) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = now.as_millis() as u64;
    let salt: u32 = rand::random();
    (
        format!("{:013}-{:08x}", millis, salt),
        rfc3339_utc(now.as_secs()),
    )
}

/// Format a Unix timestamp as RFC3339 UTC without pulling in a date crate.
///
/// Uses Howard Hinnant's `civil_from_days`, which is exact for all dates in
/// the proleptic Gregorian calendar.
pub fn rfc3339_utc(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let secs_of_day = unix_secs % 86_400;

    // Shift the epoch to 0000-03-01 so leap days land at the end of the cycle.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March-based
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_correct_for_known_instants() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(1_000_000_000), "2001-09-09T01:46:40Z");
        // A leap day, to exercise the era arithmetic.
        assert_eq!(rfc3339_utc(1_709_164_800), "2024-02-29T00:00:00Z");
    }

    #[test]
    fn ids_sort_chronologically_and_do_not_collide() {
        let a = Run::new("bench", "relu", "mpc-aby3");
        let b = Run::new("bench", "relu", "mpc-aby3");
        assert_ne!(a.id, b.id);
        assert!(a.id <= b.id, "{} should not sort after {}", a.id, b.id);
    }

    #[test]
    fn keywords_never_leak_metric_values() {
        let run = Run::new("bench", "MNIST-linear", "fhe-tfhe")
            .with_metric("eval_us", 2_829_000_000.0, "us")
            .with_config("secret_weights", "true")
            .with_tag("Paper-Table-3")
            .with_ok(true);
        let k = run.keywords();
        assert!(k.contains(&"suite:bench".to_string()));
        assert!(k.contains(&"case:mnist-linear".to_string()));
        assert!(k.contains(&"tag:paper-table-3".to_string()));
        assert!(k.contains(&"ok:true".to_string()));
        assert!(
            !k.iter()
                .any(|s| s.contains("2829") || s.contains("eval_us")),
            "metric values must stay in the encrypted payload: {:?}",
            k
        );
    }

    #[test]
    fn keyword_matching_is_conjunctive_and_case_insensitive() {
        let run = Run::new("bench", "relu", "mpc-aby3").with_tag("wan");
        assert!(run.matches(&["suite:bench".into(), "TAG:WAN".into()]));
        assert!(!run.matches(&["suite:bench".into(), "tag:lan".into()]));
    }

    #[test]
    fn metric_assignments_parse_with_and_without_units() {
        let (n, m) = Metric::parse_assignment("eval_us=1340us").unwrap();
        assert_eq!(n, "eval_us");
        assert_eq!(m, Metric::new(1340.0, "us"));
        let (n, m) = Metric::parse_assignment("gates=2365").unwrap();
        assert_eq!(n, "gates");
        assert_eq!(m, Metric::new(2365.0, ""));
        assert!(Metric::parse_assignment("bogus").is_err());
        assert!(Metric::parse_assignment("x=abc").is_err());
    }

    #[test]
    fn records_round_trip_through_json() {
        let run = Run::new("bench", "relu", "mpc-aby3")
            .with_metric("eval_us", 1.5, "us")
            .with_ok(false);
        let json = serde_json::to_string(&run).unwrap();
        let back: Run = serde_json::from_str(&json).unwrap();
        assert_eq!(run, back);
    }
}
