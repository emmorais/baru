//! `baru bench-cohort-eval` — **cohort-conditioned private evaluation**, the first
//! result that *composes* two stages of the healthcare workflow.
//!
//! Everything measured so far measures one stage alone: `bench-cohort` times a
//! cohort query (stage 1), `bench-medmnist` classifies a scan (stage 4), and
//! `docs/research-agenda.md` §2 sets ≈1.0× against 10³–10⁶×. That is a good
//! argument and it is the easy one. This asks the question a clinician actually
//! has:
//!
//! > *"What is this model's accuracy on patients aged 40–60 with HbA1c > 7?"*
//!
//! answered without the storage provider learning the cohort predicate's results
//! and without the model owner learning which patients were selected.
//!
//! # The deployment, stated precisely
//!
//! Three parties, and who learns what matters more than the numbers:
//!
//! * **The hospital** holds the records and the images. It selects the cohort out
//!   of *its own* data, so learning which of its own patients matched reveals
//!   nothing it did not already know.
//! * **The storage provider** holds only ciphertext and ORE/DET tokens. It
//!   evaluates the range predicate without learning the attribute values.
//! * **The model owner** performs private inference and learns only the **cohort
//!   size** — never which patients, never their attributes, never the images.
//!
//! That is why this uses [`crate::db::cohort::CohortDb::cohort_ids`] rather than
//! `cohort_count`: the ids go to the data owner, which is the one party for whom
//! they are not a disclosure.
//!
//! # Why the attributes are synthetic, and what that costs the claim
//!
//! **MedMNIST carries no demographics.** The images are real biomedical scans; the
//! ages and HbA1c values attached to them here are generated. That is:
//!
//! * **fine for a cost-structure claim** — the encrypted range query does the same
//!   work whether an age is real or generated, so "what does composing these two
//!   stages cost" is answerable;
//! * **not fine for any clinical or fairness claim** — a subgroup accuracy gap
//!   measured on generated demographics is an artifact of the generator, not a
//!   property of the model. The benchmark prints this in its own output, not only
//!   in a doc.
//!
//! Real demographics mean a dataset that carries them (CheXpert, MIMIC-CXR; both
//! need credentialed access). That is the upgrade path, and it is the honest
//! alternative to inventing one.
//!
//! # What is checked
//!
//! The cohort is computed three ways — encrypted BaruDB, plaintext BaruDB, and a
//! local plaintext filter — and all three must agree, or the command fails. So this
//! is a differential test of the encrypted query path as well as a measurement.

use std::time::Instant;

use serde::Serialize;

use crate::bench::comm::NetworkProfile;
use crate::bench::medmnist;
use crate::db::cohort::{CohortDb, Column};

/// A synthetic patient attribute row, attached to a real image.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Attrs {
    pub age: i64,
    /// HbA1c in percent, two decimals — `fixed2` in BaruDB's schema.
    pub hba1c: f64,
}

/// Deterministic synthetic attributes, so a run is reproducible.
///
/// Deliberately *independent* of the image's label: correlating them would
/// manufacture a subgroup accuracy gap out of the generator, which is exactly the
/// artifact the module docs warn about.
fn attrs_for(i: usize) -> Attrs {
    Attrs {
        age: 18 + ((i * 37) % 62) as i64,
        hba1c: 4.5 + (((i * 53) % 70) as f64) / 10.0,
    }
}

fn matches(a: &Attrs, min_age: i64, max_age: i64, min_hba1c: f64) -> bool {
    a.age >= min_age && a.age <= max_age && a.hba1c > min_hba1c
}

#[derive(Debug, Clone, Serialize)]
pub struct CohortEvalReport {
    pub dataset: String,
    pub task: String,
    pub records: usize,
    pub predicate: String,
    /// Cohort size, from the encrypted database.
    pub cohort: usize,
    /// Encrypted, plaintext-BaruDB and local-filter cohorts all agreed.
    pub cohorts_agree: bool,
    /// Accuracy over the selected cohort, integer model.
    pub cohort_correct: usize,
    pub cohort_accuracy: f64,
    /// Accuracy over every record, for comparison.
    pub overall_accuracy: f64,
    /// Standard error of the cohort accuracy, in percentage points.
    ///
    /// Reported because the cohort is small: without it a reader sees a
    /// cohort-vs-overall gap and reads significance into sampling noise. With the
    /// synthetic attributes here there is nothing to read either way, but the
    /// number is what makes that checkable rather than asserted.
    pub cohort_accuracy_se: f64,
    /// Gap in standard errors — under ~2 is noise.
    pub gap_in_se: f64,
    /// Ingest and selection cost, microseconds (BaruDB's marginal figures).
    pub ingest_us: u64,
    pub select_us: u64,
    /// Encrypted inference on a subset of the cohort.
    pub encrypted_evaluated: usize,
    pub encrypted_matches_plaintext: usize,
    pub inference_ms_per_image: f64,
    pub bootstraps: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    dataset: &str,
    db_dir: &std::path::Path,
    min_age: i64,
    max_age: i64,
    min_hba1c: f64,
    hybrid_n: usize,
) -> Result<CohortEvalReport, String> {
    let data = medmnist::load(dataset)?;
    let model = data.model();
    let n = data.images.len();
    let predicate = format!("age>={min_age} and age<={max_age} and hba1c>{min_hba1c:.2}");

    // ── Stage 1: the cohort, three ways ─────────────────────────────────────
    let rows: String = (0..n)
        .map(|i| {
            let a = attrs_for(i);
            format!("{{\"age\":{},\"hba1c\":{:.2}}}\n", a.age, a.hba1c)
        })
        .collect();
    let columns = [
        Column::new("age", "int", "ore"),
        Column::new("hba1c", "fixed2", "ore"),
    ];

    let enc_path = db_dir.join("cohort-eval.redb");
    let plain_path = db_dir.join("cohort-eval.plain.redb");
    let _ = std::fs::remove_file(&enc_path);
    let _ = std::fs::remove_file(&plain_path);

    let enc = CohortDb::new(enc_path.to_string_lossy().to_string());
    let plain = CohortDb::new(format!("plain:{}", plain_path.to_string_lossy()));

    let mut ingest_us = 0u64;
    let mut select_us = 0u64;
    let mut cohorts: Vec<Vec<u64>> = Vec::new();
    for (db, count_it) in [(&enc, true), (&plain, false)] {
        db.create_table("patients", &columns)?;
        let (inserted, t) = db.insert("patients", &rows)?;
        if inserted as usize != n {
            return Err(format!("inserted {inserted} of {n} rows"));
        }
        if count_it {
            ingest_us = t.marginal_total_us();
        }
        let (ids, t) = db.cohort_ids("patients", &predicate)?;
        if count_it {
            select_us = t.marginal_total_us();
        }
        cohorts.push(ids);
    }
    // The third view: a plain local filter over the attributes we generated.
    let local: Vec<u64> = (0..n)
        .filter(|i| matches(&attrs_for(*i), min_age, max_age, min_hba1c))
        .map(|i| i as u64)
        .collect();
    let cohorts_agree = cohorts[0] == cohorts[1] && cohorts[1] == local;
    let cohort = cohorts[0].clone();

    // ── Stage 4: classify exactly that cohort ───────────────────────────────
    // The plaintext-integer forward is bit-exact against the encrypted hybrid, so
    // this is the deployed model's accuracy, not a float model's.
    let mut cohort_correct = 0usize;
    for &id in &cohort {
        let i = id as usize;
        if model.forward(&data.images[i]).0 == data.labels[i] as usize {
            cohort_correct += 1;
        }
    }
    let overall_correct = data
        .images
        .iter()
        .zip(&data.labels)
        .filter(|(img, &l)| model.forward(img).0 == l as usize)
        .count();

    // Demonstrate the composition end to end on a few cohort members: the
    // encrypted prediction must equal the plaintext one.
    let hybrid_n = hybrid_n.min(cohort.len());
    let t0 = Instant::now();
    let (mut matches_plain, mut bootstraps) = (0usize, 0u64);
    for &id in cohort.iter().take(hybrid_n) {
        let i = id as usize;
        let run =
            crate::bench::cnn::run_cnn_hybrid(&model, &data.images[i], NetworkProfile::INSTANT);
        if run.class == model.forward(&data.images[i]).0 {
            matches_plain += 1;
        }
        bootstraps += run.bootstraps;
    }
    let inference_ms_per_image = if hybrid_n > 0 {
        t0.elapsed().as_secs_f64() * 1000.0 / hybrid_n as f64
    } else {
        0.0
    };

    let cohort_acc = if cohort.is_empty() {
        0.0
    } else {
        cohort_correct as f64 / cohort.len() as f64
    };
    // Binomial standard error of the cohort accuracy.
    let se = if cohort.is_empty() {
        0.0
    } else {
        (cohort_acc * (1.0 - cohort_acc) / cohort.len() as f64).sqrt()
    };

    Ok(CohortEvalReport {
        dataset: data.dataset.to_string(),
        task: data.task.to_string(),
        records: n,
        predicate,
        cohort: cohort.len(),
        cohorts_agree,
        cohort_correct,
        cohort_accuracy: cohort_acc,
        overall_accuracy: overall_correct as f64 / n as f64,
        cohort_accuracy_se: se,
        gap_in_se: if se > 0.0 {
            (cohort_acc - overall_correct as f64 / n as f64).abs() / se
        } else {
            0.0
        },
        ingest_us,
        select_us,
        encrypted_evaluated: hybrid_n,
        encrypted_matches_plaintext: matches_plain,
        inference_ms_per_image,
        bootstraps,
    })
}

const USAGE: &str = "usage: baru bench-cohort-eval [options]

Cohort-conditioned private evaluation: select a patient cohort with an ENCRYPTED
range predicate, then run the encrypted classifier over exactly that cohort.
Composes stage 1 (BaruDB) with stage 4 (private inference).

Patient attributes are SYNTHETIC — MedMNIST carries no demographics.  The images
are real; the ages and HbA1c values are generated.  See the module docs for what
that does and does not license.

Requires `barudb` on PATH or at $BARUDB_BIN.

options:
  --dataset NAME   pneumonia | organa            (default pneumonia)
  --dir PATH       where to put the databases    (default a temp dir)
  --age MIN,MAX    age range                     (default 40,60)
  --hba1c X        HbA1c lower bound, exclusive   (default 7.0)
  --hybrid N       encrypted inferences to run    (default 2)
  --json           machine-readable output
  -h, --help       this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut dataset = "pneumonia".to_string();
    let mut dir: Option<String> = None;
    let (mut min_age, mut max_age) = (40i64, 60i64);
    let mut min_hba1c = 7.0f64;
    let mut hybrid_n = 2usize;
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dataset" => dataset = it.next().ok_or("--dataset needs a value")?.clone(),
            "--dir" => dir = Some(it.next().ok_or("--dir needs a value")?.clone()),
            "--age" => {
                let v = it.next().ok_or("--age needs MIN,MAX")?;
                let (a, b) = v.split_once(',').ok_or("--age wants MIN,MAX")?;
                min_age = a.trim().parse().map_err(|_| "bad --age min")?;
                max_age = b.trim().parse().map_err(|_| "bad --age max")?;
            }
            "--hba1c" => {
                min_hba1c = it
                    .next()
                    .ok_or("--hba1c needs a value")?
                    .parse()
                    .map_err(|_| "bad --hba1c")?;
            }
            "--hybrid" => {
                hybrid_n = it
                    .next()
                    .ok_or("--hybrid needs a value")?
                    .parse()
                    .map_err(|_| "bad --hybrid")?;
            }
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let tmp = std::env::temp_dir().join("baru-cohort-eval");
    let db_dir = dir.map(std::path::PathBuf::from).unwrap_or(tmp);
    std::fs::create_dir_all(&db_dir).map_err(|e| e.to_string())?;

    let r = run(&dataset, &db_dir, min_age, max_age, min_hba1c, hybrid_n)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Cohort-conditioned private evaluation — stage 1 composed with stage 4");
    println!("{} — {}\n", r.dataset, r.task);
    println!(
        "  predicate            : {}  (over ENCRYPTED attributes)",
        r.predicate
    );
    println!(
        "  records / cohort     : {} / {}  ({:.1}% selected)",
        r.records,
        r.cohort,
        100.0 * r.cohort as f64 / r.records as f64
    );
    println!(
        "  cohorts agree        : {}  (encrypted == plaintext BaruDB == local filter)",
        if r.cohorts_agree { "yes" } else { "NO" }
    );
    println!();
    println!(
        "  accuracy on cohort   : {}/{} = {:.2}%",
        r.cohort_correct,
        r.cohort,
        r.cohort_accuracy * 100.0
    );
    println!(
        "  accuracy overall     : {:.2}%",
        r.overall_accuracy * 100.0
    );
    println!(
        "  cohort std. error    : ±{:.1} points on {} patients — the gap above is {:.1} SE, {}",
        r.cohort_accuracy_se * 100.0,
        r.cohort,
        r.gap_in_se,
        if r.gap_in_se < 2.0 {
            "i.e. noise"
        } else {
            "which would be worth a look IF the attributes were real"
        }
    );
    println!();
    println!("  cost, by stage:");
    println!(
        "    stage 1 ingest     : {:.1} ms for {} records",
        r.ingest_us as f64 / 1000.0,
        r.records
    );
    println!(
        "    stage 1 selection  : {:.1} ms",
        r.select_us as f64 / 1000.0
    );
    println!(
        "    stage 4 inference  : {:.0} ms per image, encrypted  ({} run, {}/{} matched plaintext, {} bootstraps)",
        r.inference_ms_per_image,
        r.encrypted_evaluated,
        r.encrypted_matches_plaintext,
        r.encrypted_evaluated,
        r.bootstraps
    );
    let ratio = if r.select_us > 0 {
        r.inference_ms_per_image * 1000.0 / r.select_us as f64
    } else {
        0.0
    };
    println!();
    println!(
        "Selecting the cohort costs {:.0}x LESS than classifying one member of it.  The",
        ratio
    );
    println!("agenda's headline — that the price of privacy is wildly non-uniform across a");
    println!("pipeline — is visible inside a single query, not just across separate ones.");
    println!();
    println!("ATTRIBUTES ARE SYNTHETIC.  The images are real biomedical scans; the ages and");
    println!("HbA1c values are generated, because MedMNIST carries no demographics.  That");
    println!("licenses the COST claim above and licenses NO clinical or fairness claim: any");
    println!("cohort-vs-overall accuracy gap here is an artifact of the generator.  Real");
    println!("demographics need CheXpert or MIMIC-CXR (credentialed access).");

    if !r.cohorts_agree {
        return Err("the encrypted and plaintext cohorts disagree".to_string());
    }
    if r.encrypted_matches_plaintext != r.encrypted_evaluated {
        return Err("an encrypted prediction diverged from the plaintext one".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic generator must not correlate attributes with labels, or the
    /// benchmark would manufacture a subgroup gap and invite exactly the clinical
    /// reading the docs forbid.
    #[test]
    fn synthetic_attributes_are_independent_of_the_label() {
        let data = medmnist::load("pneumonia").unwrap();
        let in_cohort: Vec<usize> = (0..data.images.len())
            .filter(|i| matches(&attrs_for(*i), 40, 60, 7.0))
            .collect();
        assert!(
            !in_cohort.is_empty(),
            "the default predicate selects nobody"
        );

        // Positive-label rate inside the cohort vs overall: a generator that
        // correlated with the label would skew this.
        let rate = |ids: &[usize]| -> f64 {
            ids.iter().filter(|&&i| data.labels[i] == 1).count() as f64 / ids.len() as f64
        };
        let all: Vec<usize> = (0..data.images.len()).collect();
        assert!(
            (rate(&in_cohort) - rate(&all)).abs() < 0.15,
            "cohort label rate {:.2} vs overall {:.2} — the generator is correlating \
             with the label, which would fabricate a subgroup gap",
            rate(&in_cohort),
            rate(&all)
        );
    }

    /// The predicate must be evaluated identically by the local filter and by the
    /// range semantics BaruDB implements — inclusive on age, exclusive on HbA1c.
    #[test]
    fn the_predicate_boundaries_are_what_the_sql_says() {
        let on_low = Attrs {
            age: 40,
            hba1c: 7.01,
        };
        let on_high = Attrs {
            age: 60,
            hba1c: 7.01,
        };
        let below = Attrs {
            age: 39,
            hba1c: 7.01,
        };
        let at_hba1c = Attrs {
            age: 50,
            hba1c: 7.00,
        };
        assert!(
            matches(&on_low, 40, 60, 7.0),
            "age lower bound is inclusive"
        );
        assert!(
            matches(&on_high, 40, 60, 7.0),
            "age upper bound is inclusive"
        );
        assert!(!matches(&below, 40, 60, 7.0));
        assert!(!matches(&at_hba1c, 40, 60, 7.0), "hba1c bound is exclusive");
    }
}
