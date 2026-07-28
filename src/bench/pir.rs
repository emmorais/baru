//! `baru bench-pir` — what it costs to hide *which* record you fetched.
//!
//! Stage 2 of the healthcare workflow. Stage 1 (`bench-cohort`) showed that
//! encrypting records at rest is nearly free: ~1.0× on queries. This stage asks
//! the next question, and gets a much less comfortable answer.
//!
//! # What is measured
//!
//! [`crate::pir`]'s single-server BFV scheme, against the only two other ways to
//! serve the same request:
//!
//! | | hides `i`? | bytes moved |
//! |---|---|---|
//! | plain fetch | no | the record |
//! | download the whole database | yes | `N × record` |
//! | **PIR** | yes | one query ct + one ct per shard |
//!
//! Every run also checks the retrieved record against the plaintext database,
//! so the benchmark is a correctness test as well as a cost measurement.
//!
//! # The result, stated against our own interest (2026-07-28)
//!
//! **This scheme loses to downloading the whole database, at every size.** The
//! reason is structural, not a tuning problem, and it is worth writing down
//! precisely because it is exactly the kind of answer the cost model exists to
//! produce.
//!
//! A response ciphertext costs `2 · n · ⌈log₂q/8⌉` bytes and delivers one
//! record. The shard it was computed from holds `n · ⌊log₂t/8⌋` bytes of
//! database. So the ratio
//!
//! ```text
//! expansion = 2·⌈log₂q/8⌉ / ⌊log₂t/8⌋      ( = 13× at n=4096, q=2^100, t=2^20 )
//! ```
//!
//! is the factor by which one response exceeds the *entire* plaintext content
//! of the shard that produced it. Since a database of `N` records needs
//! `⌈N/per_shard⌉` shards and this scheme returns one ciphertext per shard, its
//! total traffic is `expansion ×` the whole database — independent of `N`. There
//! is no crossover to find.
//!
//! What fixes it is **recursion with server-side query expansion** (SealPIR):
//! fold the shard dimension on the server so the response is `O(1)` ciphertexts
//! while `N` grows, which needs Galois automorphisms and a substitution
//! operation `src/fhe/bfv.rs` does not have. That is the work this benchmark
//! justifies, and it is a real piece of work — not the "few hundred lines" the
//! textbook scheme took.
//!
//! So by the research agenda's own admission rule — a primitive enters when it
//! is *the cheapest way* to serve a stage — PIR has **not** earned stage 2 yet.
//! What it has earned is a measured cost baseline and a specific next step. The
//! honest current recommendation for stage 2 at these record sizes is the dull
//! one: download the shard, or accept the access-pattern leak and revisit after
//! expansion lands.
//!
//! Where the scheme is already strong is everything other than bandwidth: the
//! query is `O(1)` regardless of `N`, the client keeps no state beyond its key,
//! the server needs no key at all, it is one round, and it consumes **zero**
//! multiplicative depth — so it composes with the rest of the depth-1 pipeline
//! without a bootstrap, like everything else in this repo.

use std::time::Instant;

use serde::Serialize;

use crate::fhe::bfv::BfvParams;
use crate::pir::Pir;

/// One database shape, measured.
#[derive(Debug, Clone, Serialize)]
pub struct PirPoint {
    pub records: usize,
    pub record_bytes: usize,
    pub words_per_record: usize,
    pub records_per_shard: usize,
    pub shards: usize,
    pub queries: usize,
    pub correct: usize,
    pub query_bytes: usize,
    pub answer_bytes: usize,
    pub plain_fetch_bytes: usize,
    pub whole_db_bytes: usize,
    /// PIR traffic ÷ a non-private fetch.
    pub overhead_vs_plain: f64,
    /// Whole-database download ÷ PIR traffic. Above 1.0 means PIR wins.
    pub vs_whole_db: f64,
    pub encode_ms: f64,
    pub query_ms: f64,
    pub answer_ms: f64,
    pub decode_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PirReport {
    pub ring_n: usize,
    pub log2_q: u32,
    pub log2_t: u32,
    pub keygen_ms: f64,
    /// `2·⌈log₂q/8⌉ / ⌊log₂t/8⌋` — the structural factor explained in the
    /// module docs. Below 1.0 would mean PIR can beat a whole-database download.
    pub expansion: f64,
    pub shard_capacity_bytes: usize,
    pub points: Vec<PirPoint>,
}

fn synthetic(records: usize, record_bytes: usize) -> Vec<Vec<u8>> {
    (0..records)
        .map(|j| {
            (0..record_bytes)
                .map(|b| ((j * 131 + b * 17 + 7) % 251) as u8)
                .collect()
        })
        .collect()
}

pub fn run(
    params: BfvParams,
    shapes: &[(usize, usize)],
    queries: usize,
) -> Result<PirReport, String> {
    let per_coeff = (params.log2_q as usize).div_ceil(8);
    let bytes_per_word = (params.log2_t / 8) as usize;
    let expansion = (2 * per_coeff) as f64 / bytes_per_word.max(1) as f64;

    let mut points = Vec::new();
    let mut keygen_ms = 0.0;
    let mut shard_capacity_bytes = 0;

    for &(records, record_bytes) in shapes {
        let pir = Pir::new(params, records, record_bytes)?;
        shard_capacity_bytes = pir.shard_capacity_bytes();

        let t0 = Instant::now();
        let (ck, pk) = pir.bfv.keygen();
        keygen_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let data = synthetic(records, record_bytes);
        let t0 = Instant::now();
        let encoded = pir.encode_db(&data)?;
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Spread the sampled indices over the whole database so every shard,
        // including the last partial one, is exercised.
        let n_q = queries.min(records);
        let idxs: Vec<usize> = (0..n_q).map(|k| k * records / n_q).collect();

        let (mut query_ms, mut answer_ms, mut decode_ms) = (0.0, 0.0, 0.0);
        let (mut correct, mut traffic) = (0usize, None);
        for &i in &idxs {
            let t0 = Instant::now();
            let q = pir.query(&pk, i)?;
            query_ms += t0.elapsed().as_secs_f64() * 1000.0;

            let t0 = Instant::now();
            let a = pir.answer(&encoded, &q);
            answer_ms += t0.elapsed().as_secs_f64() * 1000.0;

            let t0 = Instant::now();
            let got = pir.decode(&ck, &a, i)?;
            decode_ms += t0.elapsed().as_secs_f64() * 1000.0;

            if got == data[i] {
                correct += 1;
            }
            traffic = Some(pir.traffic(&q, &a));
        }
        let tr = traffic.ok_or("no queries were run")?;
        let n = n_q as f64;
        points.push(PirPoint {
            records,
            record_bytes,
            words_per_record: pir.layout.words,
            records_per_shard: pir.layout.per_shard,
            shards: pir.layout.shards,
            queries: n_q,
            correct,
            query_bytes: tr.query_bytes,
            answer_bytes: tr.answer_bytes,
            plain_fetch_bytes: tr.plain_fetch_bytes,
            whole_db_bytes: tr.whole_db_bytes,
            overhead_vs_plain: tr.overhead_vs_plain(),
            vs_whole_db: tr.saving_vs_whole_db(),
            encode_ms,
            query_ms: query_ms / n,
            answer_ms: answer_ms / n,
            decode_ms: decode_ms / n,
        });
    }

    Ok(PirReport {
        ring_n: params.n,
        log2_q: params.log2_q,
        log2_t: params.log2_t,
        keygen_ms,
        expansion,
        shard_capacity_bytes,
        points,
    })
}

fn human(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

const USAGE: &str = "usage: baru bench-pir [options]

The cost of hiding *which* record was fetched: single-server BFV PIR against a
plain fetch and against downloading the whole database.

options:
  --records N,N,...   database sizes to sweep   (default 32,256,2048)
  --record-bytes B    bytes per record          (default 256)
  --queries Q         indices sampled per size  (default 4)
  --json              machine-readable output
  -h, --help          this message
";

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let mut sizes = vec![32usize, 256, 2048];
    let mut record_bytes = 256usize;
    let mut queries = 4usize;
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--records" => {
                let v = it.next().ok_or("--records needs a value")?;
                sizes = v
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad size {s}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--record-bytes" => {
                record_bytes = it
                    .next()
                    .ok_or("--record-bytes needs a value")?
                    .parse()
                    .map_err(|_| "bad --record-bytes")?;
            }
            "--queries" => {
                queries = it
                    .next()
                    .ok_or("--queries needs a value")?
                    .parse()
                    .map_err(|_| "bad --queries")?;
            }
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}\n\n{USAGE}")),
        }
    }

    let shapes: Vec<(usize, usize)> = sizes.iter().map(|&r| (r, record_bytes)).collect();
    let r = run(BfvParams::DEFAULT, &shapes, queries)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Private information retrieval — single-server BFV, one round, zero depth");
    println!(
        "ring n={}, q=2^{}, t=2^{}   keygen {:.0} ms   shard holds {}",
        r.ring_n,
        r.log2_q,
        r.log2_t,
        r.keygen_ms,
        human(r.shard_capacity_bytes)
    );
    println!();
    println!(
        "{:>8} {:>7} {:>7} {:>10} {:>10} {:>10} {:>9} {:>9}  server ms",
        "records", "shards", "correct", "query", "answer", "whole db", "vs plain", "vs whole"
    );
    for p in &r.points {
        println!(
            "{:>8} {:>7} {:>4}/{:<2} {:>10} {:>10} {:>10} {:>8.0}x {:>8.2}x  {:.0}",
            p.records,
            p.shards,
            p.correct,
            p.queries,
            human(p.query_bytes),
            human(p.answer_bytes),
            human(p.whole_db_bytes),
            p.overhead_vs_plain,
            p.vs_whole_db,
            p.answer_ms
        );
    }
    println!();
    println!(
        "ciphertext expansion = 2*ceil(log2 q/8) / floor(log2 t/8) = {:.0}x",
        r.expansion
    );
    println!(
        "One response ciphertext costs {:.0}x the entire plaintext content of the shard",
        r.expansion
    );
    println!("that produced it, and this scheme returns one per shard — so its traffic is");
    println!(
        "~{:.0}x a whole-database download regardless of N.  There is no crossover.",
        r.expansion
    );
    println!();
    println!("Verdict: PIR does NOT yet earn stage 2 on cost.  What fixes it is recursion");
    println!("with server-side query expansion (SealPIR), which needs Galois automorphisms");
    println!("and a substitution op BFV here lacks.  What already holds: O(1) query, one");
    println!("round, no server key, and zero multiplicative depth.");

    if r.points.iter().any(|p| p.correct != p.queries) {
        return Err("a retrieved record did not match the plaintext database".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correctness at real parameters, on a database that has to shard.
    #[test]
    fn retrieval_is_correct_at_secure_parameters() {
        let r = run(BfvParams::DEFAULT, &[(64, 128)], 3).unwrap();
        let p = &r.points[0];
        assert_eq!(p.correct, p.queries, "every sampled record must round-trip");
        assert!(p.shards >= 1);
    }

    /// The finding itself is worth pinning: if a future change made PIR beat a
    /// whole-database download, that would be a genuine result — and this test
    /// failing is how we would find out, rather than by not noticing.
    #[test]
    fn the_expansion_factor_is_what_the_docs_claim() {
        let r = run(BfvParams::DEFAULT, &[(256, 256)], 1).unwrap();
        assert!(
            (r.expansion - 13.0).abs() < 0.5,
            "expansion {} — the module docs explain the 13x; update them together",
            r.expansion
        );
        let p = &r.points[0];
        assert!(
            p.vs_whole_db < 1.0,
            "this scheme is expected to lose to a whole-db download ({:.2}x); if it \
             now wins, the docs and the agenda's stage-2 verdict need rewriting",
            p.vs_whole_db
        );
    }
}
