//! Client-side querying, aggregation and export.
//!
//! Keyword filtering happens in the store (encrypted or not); everything
//! numeric happens here, after decryption, because range queries over encrypted
//! metrics are exactly the part BaruDB has not built yet — see
//! `docs/barudb-api.md` §1.  At Baru's scale (thousands of records, not
//! millions) that is the right trade.
//!
//! The export formats exist for one reason: **the numbers in the paper should
//! be a query, not a transcription.**  `baru db export --format latex` emits a
//! table that `paper/main.tex` can `\input`, so a re-run updates the paper.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::record::Run;

/// Keyword-level filter — everything here can be pushed into the store.
#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub suite: Option<String>,
    pub case: Option<String>,
    pub backend: Option<String>,
    pub tags: Vec<String>,
    pub ok: Option<bool>,
    pub commit: Option<String>,
}

impl Filter {
    /// The `field:value` keywords this filter corresponds to.
    pub fn keywords(&self) -> Vec<String> {
        let mut k = Vec::new();
        if let Some(v) = &self.suite {
            k.push(format!("suite:{}", v.to_lowercase()));
        }
        if let Some(v) = &self.case {
            k.push(format!("case:{}", v.to_lowercase()));
        }
        if let Some(v) = &self.backend {
            k.push(format!("backend:{}", v.to_lowercase()));
        }
        if let Some(v) = &self.commit {
            k.push(format!("commit:{}", v));
        }
        if let Some(ok) = self.ok {
            k.push(format!("ok:{}", ok));
        }
        for t in &self.tags {
            k.push(format!("tag:{}", t.to_lowercase()));
        }
        k
    }

    pub fn is_empty(&self) -> bool {
        self.keywords().is_empty()
    }
}

/// One metric summarised over the runs in a group.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub unit: String,
}

/// Runs grouped by `(case, backend)` — the natural unit for a results table.
#[derive(Debug, Clone, Serialize)]
pub struct Group {
    pub case: String,
    pub backend: String,
    pub runs: usize,
    /// `None` when no run in the group recorded a correctness verdict; `false`
    /// when *any* run failed, because a single failure invalidates the row.
    pub ok: Option<bool>,
    pub metrics: BTreeMap<String, Summary>,
}

/// Group and summarise. Groups come back in `(case, backend)` order so a table
/// is stable across invocations.
pub fn aggregate(runs: &[Run]) -> Vec<Group> {
    let mut buckets: BTreeMap<(String, String), Vec<&Run>> = BTreeMap::new();
    for r in runs {
        buckets
            .entry((r.case.clone(), r.backend.clone()))
            .or_default()
            .push(r);
    }
    buckets
        .into_iter()
        .map(|((case, backend), rs)| {
            let mut metrics: BTreeMap<String, Summary> = BTreeMap::new();
            let names: BTreeSet<&String> = rs.iter().flat_map(|r| r.metrics.keys()).collect();
            for name in names {
                let vals: Vec<f64> = rs
                    .iter()
                    .filter_map(|r| r.metrics.get(name).map(|m| m.value))
                    .collect();
                let unit = rs
                    .iter()
                    .find_map(|r| r.metrics.get(name).map(|m| m.unit.clone()))
                    .unwrap_or_default();
                let n = vals.len();
                let mean = vals.iter().sum::<f64>() / n as f64;
                metrics.insert(
                    name.clone(),
                    Summary {
                        n,
                        mean,
                        min: vals.iter().cloned().fold(f64::INFINITY, f64::min),
                        max: vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        unit,
                    },
                );
            }
            let verdicts: Vec<bool> = rs.iter().filter_map(|r| r.ok).collect();
            let ok = if verdicts.is_empty() {
                None
            } else {
                Some(verdicts.iter().all(|v| *v))
            };
            Group {
                case,
                backend,
                runs: rs.len(),
                ok,
                metrics,
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Table,
    Csv,
    Latex,
    Json,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s.to_ascii_lowercase().as_str() {
            "table" | "text" => Some(Format::Table),
            "csv" => Some(Format::Csv),
            "latex" | "tex" => Some(Format::Latex),
            "json" => Some(Format::Json),
            _ => None,
        }
    }
}

/// The metric columns to show: the caller's selection, or every metric present.
fn columns(groups: &[Group], wanted: &[String]) -> Vec<String> {
    if !wanted.is_empty() {
        return wanted.to_vec();
    }
    let mut set: BTreeSet<String> = BTreeSet::new();
    for g in groups {
        set.extend(g.metrics.keys().cloned());
    }
    set.into_iter().collect()
}

fn cell(g: &Group, metric: &str) -> String {
    match g.metrics.get(metric) {
        // The mean of a single run is that run, so one formatting rule covers
        // both cases; spread lives in `Summary::min`/`max` for the JSON export.
        Some(s) => format!("{:.4}", s.mean),
        None => "—".to_string(),
    }
}

/// Render grouped results.
pub fn render(groups: &[Group], wanted: &[String], format: Format) -> String {
    let cols = columns(groups, wanted);
    match format {
        Format::Json => serde_json::to_string_pretty(groups).unwrap_or_else(|e| e.to_string()),
        Format::Csv => {
            let mut out = String::from("case,backend,runs,ok");
            for c in &cols {
                out.push(',');
                out.push_str(c);
            }
            out.push('\n');
            for g in groups {
                out.push_str(&format!(
                    "{},{},{},{}",
                    g.case,
                    g.backend,
                    g.runs,
                    match g.ok {
                        Some(true) => "true",
                        Some(false) => "false",
                        None => "",
                    }
                ));
                for c in &cols {
                    out.push(',');
                    out.push_str(&cell(g, c).replace('—', ""));
                }
                out.push('\n');
            }
            out
        }
        Format::Latex => {
            let unit_of = |m: &str| {
                groups
                    .iter()
                    .find_map(|g| g.metrics.get(m))
                    .map(|s| s.unit.clone())
                    .unwrap_or_default()
            };
            let mut out = String::new();
            // Keep the emitted file ASCII: it gets \input into a LaTeX document
            // whose input encoding we do not control.
            out.push_str("% generated by `baru db export` -- do not edit by hand\n");
            out.push_str(&format!(
                "\\begin{{tabular}}{{ll{}}}\n\\hline\n",
                "r".repeat(cols.len())
            ));
            out.push_str("case & backend");
            for c in &cols {
                let u = unit_of(c);
                let head = if u.is_empty() {
                    c.clone()
                } else {
                    format!("{} ({})", c, u)
                };
                out.push_str(&format!(" & {}", latex_escape(&head)));
            }
            out.push_str(" \\\\\n\\hline\n");
            for g in groups {
                out.push_str(&format!(
                    "{} & {}",
                    latex_escape(&g.case),
                    latex_escape(&g.backend)
                ));
                for c in &cols {
                    out.push_str(&format!(" & {}", cell(g, c).replace('—', "---")));
                }
                out.push_str(" \\\\\n");
            }
            out.push_str("\\hline\n\\end{tabular}\n");
            out
        }
        Format::Table => {
            let mut widths: Vec<usize> = vec!["case".len(), "backend".len(), "runs".len()];
            for c in &cols {
                widths.push(c.len().max(10));
            }
            for g in groups {
                widths[0] = widths[0].max(g.case.len());
                widths[1] = widths[1].max(g.backend.len());
            }
            let mut out = format!(
                "{:<w0$}  {:<w1$}  {:>4}  {:<3}",
                "case",
                "backend",
                "runs",
                "ok",
                w0 = widths[0],
                w1 = widths[1]
            );
            for (i, c) in cols.iter().enumerate() {
                out.push_str(&format!("  {:>w$}", c, w = widths[3 + i]));
            }
            out.push('\n');
            for g in groups {
                out.push_str(&format!(
                    "{:<w0$}  {:<w1$}  {:>4}  {:<3}",
                    g.case,
                    g.backend,
                    g.runs,
                    match g.ok {
                        Some(true) => "ok",
                        Some(false) => "FAIL",
                        None => "-",
                    },
                    w0 = widths[0],
                    w1 = widths[1]
                ));
                for (i, c) in cols.iter().enumerate() {
                    out.push_str(&format!("  {:>w$}", cell(g, c), w = widths[3 + i]));
                }
                out.push('\n');
            }
            out
        }
    }
}

fn latex_escape(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('_', "\\_")
        .replace('%', "\\%")
        .replace('&', "\\&")
        .replace('#', "\\#")
        .replace('$', "\\$")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs() -> Vec<Run> {
        vec![
            Run::new("bench", "relu", "mpc-aby3")
                .with_metric("eval_us", 100.0, "us")
                .with_ok(true),
            Run::new("bench", "relu", "mpc-aby3")
                .with_metric("eval_us", 200.0, "us")
                .with_ok(true),
            Run::new("bench", "relu", "fhe-tfhe")
                .with_metric("eval_us", 500_000.0, "us")
                .with_metric("ct_bytes", 515_000.0, "B")
                .with_ok(false),
        ]
    }

    #[test]
    fn aggregation_averages_repeats_and_keeps_units() {
        let g = aggregate(&runs());
        assert_eq!(g.len(), 2);
        let aby3 = g.iter().find(|x| x.backend == "mpc-aby3").unwrap();
        assert_eq!(aby3.runs, 2);
        assert_eq!(aby3.metrics["eval_us"].mean, 150.0);
        assert_eq!(aby3.metrics["eval_us"].min, 100.0);
        assert_eq!(aby3.metrics["eval_us"].max, 200.0);
        assert_eq!(aby3.metrics["eval_us"].unit, "us");
    }

    #[test]
    fn one_failing_run_invalidates_the_whole_row() {
        let g = aggregate(&runs());
        let tfhe = g.iter().find(|x| x.backend == "fhe-tfhe").unwrap();
        assert_eq!(tfhe.ok, Some(false));
    }

    #[test]
    fn missing_metrics_render_as_gaps_not_zeros() {
        let g = aggregate(&runs());
        let table = render(&g, &[], Format::Table);
        assert!(table.contains('—'), "expected a gap marker:\n{}", table);
        let csv = render(&g, &[], Format::Csv);
        assert!(
            csv.starts_with("case,backend,runs,ok,ct_bytes,eval_us\n"),
            "{}",
            csv
        );
        assert!(csv.contains("relu,mpc-aby3,2,true,,150.0000"));
    }

    #[test]
    fn latex_escapes_underscores_so_the_paper_compiles() {
        let g = aggregate(&runs());
        let tex = render(&g, &["eval_us".to_string()], Format::Latex);
        assert!(tex.contains("eval\\_us"), "{}", tex);
        assert!(tex.contains("\\begin{tabular}"));
        assert!(!tex.contains("—"));
    }

    #[test]
    fn filters_become_store_keywords() {
        let f = Filter {
            suite: Some("bench".into()),
            backend: Some("FHE-TFHE".into()),
            tags: vec!["Paper".into()],
            ok: Some(true),
            ..Filter::default()
        };
        let mut k = f.keywords();
        k.sort();
        assert_eq!(
            k,
            vec!["backend:fhe-tfhe", "ok:true", "suite:bench", "tag:paper"]
        );
    }
}
