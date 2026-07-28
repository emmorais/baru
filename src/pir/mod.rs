//! **Private information retrieval** — fetch record `i` without revealing `i`.
//!
//! Stage 2 of the healthcare workflow in `docs/research-agenda.md`: a clinician
//! pulls one patient's record for review. Encrypting the records at rest (stage
//! 1, [`crate::db`]) does not help here, because *which* record is fetched is
//! itself the disclosure — an access pattern over an encrypted store still says
//! "this is the patient we are interested in". PIR is the primitive that closes
//! that gap, and it is admitted under the agenda's rule: it is the cheapest way
//! to serve this stage, and it arrives with a benchmark ([`crate::bench::pir`]).
//!
//! # The construction
//!
//! Single-server, single-round, built on the BFV already in the tree
//! ([`crate::fhe::bfv`]) — no new cryptographic assumption, and reusing the
//! parameters whose security is argued in that module.
//!
//! The whole scheme is one observation about the ring `Z_t[x]/(xⁿ + 1)`. Lay
//! the database out as a single plaintext polynomial, record `j` occupying `w`
//! consecutive coefficients:
//!
//! ```text
//! D(x) = Σ_j Σ_{c<w} d[j][c] · x^(j·w + c)
//! ```
//!
//! and let the client send an encryption of a *monomial*: `Q(x) = −x^(n − i·w)`
//! (and `+x⁰` when `i = 0`). Because `xⁿ = −1`, the product `Q·D` puts
//! `d[i][c]` — and nothing else — in coefficient `c` for every `c < w`:
//!
//! * `j = i` contributes `−x^(n+c) = +x^c`, the record we want;
//! * `j > i` lands at `(j−i)w + c ≥ w`;
//! * `j < i` lands at `n − (i−j)w + c ≥ w`, given `N·w ≤ n`.
//!
//! So the server does **one plaintext-ciphertext polynomial multiplication**
//! over the whole database and returns one ciphertext. The client reads the
//! first `w` coefficients. Upload and download are one ciphertext each,
//! independent of the number of records.
//!
//! Note what this is *not*: it is the textbook one-dimensional scheme, not
//! SealPIR. The costs it does not pay are query expansion (SealPIR sends one
//! ciphertext and expands it server-side with automorphisms) and recursion
//! (folding the database as a `d`-dimensional hypercube to shrink the response).
//! Where those matter is stated in the sharding note below and measured in the
//! benchmark.
//!
//! # Security, stated precisely
//!
//! * **What the server learns about `i`: nothing.** The query is a BFV
//!   ciphertext and the server sees only that; the guarantee is exactly BFV's
//!   semantic security, so it is computational, not information-theoretic.
//! * **The server must touch every record.** That is inherent to single-server
//!   PIR rather than an artifact of this construction: a server that skipped a
//!   record would learn the query was not for it. Hence linear server work, and
//!   hence why "just fetch it" is asymptotically better at everything except
//!   privacy.
//! * **Sizes leak.** The record length and record count are public here, and
//!   the database is padded to a fixed record size. A deployment where record
//!   length is itself sensitive needs padding decided outside this module.
//! * **No integrity.** A malicious server can return a wrong or malformed
//!   answer and the client cannot tell. This is the semi-honest setting the
//!   rest of Baru's MPC assumes; verifiable PIR is out of scope.
//! * The client key never leaves the client, and the server needs only the
//!   public key — it is not even given that here, since the query arrives
//!   already encrypted.

use crate::fhe::bfv::{Bfv, BfvCt, BfvParams};

/// How a database is laid out across coefficients and shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Ring degree — the number of plaintext coefficients per shard.
    pub n: usize,
    /// Bytes per plaintext coefficient. A `log₂t`-bit coefficient holds
    /// `⌊log₂t / 8⌋` whole bytes; the spare bits are left unused rather than
    /// bit-packed across coefficients, which would buy a few percent of
    /// capacity for a much easier off-by-one.
    pub bytes_per_word: usize,
    /// Coefficients per record.
    pub words: usize,
    pub record_bytes: usize,
    pub records: usize,
    /// Records that fit in one shard (`⌊n / words⌋`).
    pub per_shard: usize,
    pub shards: usize,
}

impl Layout {
    pub fn new(params: &BfvParams, records: usize, record_bytes: usize) -> Result<Self, String> {
        if records == 0 || record_bytes == 0 {
            return Err("PIR needs at least one record of at least one byte".to_string());
        }
        let bytes_per_word = (params.log2_t / 8) as usize;
        if bytes_per_word == 0 {
            return Err(format!(
                "log2_t = {} leaves less than a byte per coefficient",
                params.log2_t
            ));
        }
        let words = record_bytes.div_ceil(bytes_per_word);
        if words > params.n {
            return Err(format!(
                "a {record_bytes}-byte record needs {words} coefficients, more than the \
                 ring degree {}; raise n or split the record",
                params.n
            ));
        }
        let per_shard = params.n / words;
        Ok(Layout {
            n: params.n,
            bytes_per_word,
            words,
            record_bytes,
            records,
            per_shard,
            shards: records.div_ceil(per_shard),
        })
    }

    /// `(shard, index within shard)` for a record.
    fn locate(&self, index: usize) -> (usize, usize) {
        (index / self.per_shard, index % self.per_shard)
    }
}

/// A PIR instance: the ring, and the layout both sides agree on.
pub struct Pir {
    pub bfv: Bfv,
    pub layout: Layout,
}

impl Pir {
    pub fn new(params: BfvParams, records: usize, record_bytes: usize) -> Result<Self, String> {
        let layout = Layout::new(&params, records, record_bytes)?;
        Ok(Pir {
            bfv: Bfv::new(params),
            layout,
        })
    }

    /// Encode the database into one plaintext polynomial per shard.
    ///
    /// Records shorter than `record_bytes` are zero-padded; the padding is part
    /// of the public layout, so it leaks nothing that the record size did not
    /// already.
    pub fn encode_db(&self, records: &[Vec<u8>]) -> Result<Vec<Vec<u64>>, String> {
        if records.len() != self.layout.records {
            return Err(format!(
                "expected {} records, got {}",
                self.layout.records,
                records.len()
            ));
        }
        let l = &self.layout;
        let mut shards = vec![vec![0u64; l.n]; l.shards];
        for (j, rec) in records.iter().enumerate() {
            if rec.len() > l.record_bytes {
                return Err(format!(
                    "record {j} is {} bytes, over the {} the layout allows",
                    rec.len(),
                    l.record_bytes
                ));
            }
            let (shard, within) = l.locate(j);
            for c in 0..l.words {
                let mut word = 0u64;
                for b in 0..l.bytes_per_word {
                    let at = c * l.bytes_per_word + b;
                    if let Some(byte) = rec.get(at) {
                        word |= u64::from(*byte) << (8 * b);
                    }
                }
                shards[shard][within * l.words + c] = word;
            }
        }
        Ok(shards)
    }

    /// The client's query: an encrypted monomial selecting `index`.
    pub fn query(&self, pk: &[u8], index: usize) -> Result<BfvCt, String> {
        if index >= self.layout.records {
            return Err(format!(
                "index {index} is past the {} records",
                self.layout.records
            ));
        }
        let l = &self.layout;
        let (_, within) = l.locate(index);
        let offset = within * l.words;
        let mut m = vec![0u64; l.n];
        if offset == 0 {
            m[0] = 1;
        } else {
            // −x^(n − offset): the sign cancels the one x^n = −1 introduces, so
            // the answer comes back positive.
            m[l.n - offset] = self.bfv.params.t() - 1;
        }
        Ok(self.bfv.encrypt_poly(pk, &m))
    }

    /// The server's answer: one ciphertext per shard, each a single
    /// plaintext-ciphertext multiplication over that shard's whole polynomial.
    ///
    /// Every shard is answered because answering only the shard containing the
    /// target would reveal which one that is — the same reason the server
    /// cannot skip records.
    pub fn answer(&self, db: &[Vec<u64>], query: &BfvCt) -> Vec<BfvCt> {
        db.iter()
            .map(|shard| self.bfv.mul_plain_poly(query, shard))
            .collect()
    }

    /// Recover the record from the answers.
    pub fn decode(&self, ck: &[u8], answers: &[BfvCt], index: usize) -> Result<Vec<u8>, String> {
        let l = &self.layout;
        if index >= l.records {
            return Err(format!("index {index} is past the {} records", l.records));
        }
        let (shard, _) = l.locate(index);
        let ct = answers
            .get(shard)
            .ok_or_else(|| format!("answer for shard {shard} is missing"))?;
        let coeffs = self.bfv.decrypt_poly(ck, ct);
        let mut out = Vec::with_capacity(l.record_bytes);
        for &word in coeffs.iter().take(l.words) {
            for b in 0..l.bytes_per_word {
                if out.len() < l.record_bytes {
                    out.push(((word >> (8 * b)) & 0xff) as u8);
                }
            }
        }
        Ok(out)
    }

    /// Serialized size of a ciphertext, counting each coefficient as the
    /// `⌈log₂q / 8⌉` bytes it actually needs.
    ///
    /// Deliberately *not* [`Bfv::ct_bytes`], which charges the 16 bytes a
    /// `u128` occupies in memory. For a communication claim that would
    /// overstate the cost by 16/13 and flatter nothing — but this module's whole
    /// output is a communication comparison, so it has to be the wire size.
    pub fn wire_bytes(&self, ct: &BfvCt) -> usize {
        let per_coeff = (self.bfv.params.log2_q as usize).div_ceil(8);
        ct.c.len() * self.bfv.params.n * per_coeff
    }

    /// Bytes on the wire in each direction, and the two baselines worth
    /// comparing against: fetching the record in the clear, and downloading the
    /// whole database (the only other way to hide `i` outright).
    pub fn traffic(&self, query: &BfvCt, answers: &[BfvCt]) -> Traffic {
        let l = &self.layout;
        Traffic {
            query_bytes: self.wire_bytes(query),
            answer_bytes: answers.iter().map(|a| self.wire_bytes(a)).sum(),
            plain_fetch_bytes: l.record_bytes,
            whole_db_bytes: l.records * l.record_bytes,
        }
    }

    /// Plaintext bytes one shard's polynomial can hold — the denominator of the
    /// ciphertext-expansion factor that decides whether this scheme can ever
    /// beat downloading the database.
    pub fn shard_capacity_bytes(&self) -> usize {
        self.layout.n * self.layout.bytes_per_word
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Traffic {
    pub query_bytes: usize,
    pub answer_bytes: usize,
    /// What a non-private fetch would move.
    pub plain_fetch_bytes: usize,
    /// What hiding `i` by downloading everything would move.
    pub whole_db_bytes: usize,
}

impl Traffic {
    pub fn total(&self) -> usize {
        self.query_bytes + self.answer_bytes
    }
    /// Overhead against a non-private fetch — the price of hiding the index.
    pub fn overhead_vs_plain(&self) -> f64 {
        self.total() as f64 / self.plain_fetch_bytes.max(1) as f64
    }
    /// Saving against the trivial private scheme, downloading everything.
    pub fn saving_vs_whole_db(&self) -> f64 {
        self.whole_db_bytes as f64 / self.total().max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small ring, so the test is fast; correctness of the layout is
    /// independent of the ring being secure.
    const TEST: BfvParams = BfvParams::TEST_SMALL;

    fn db(records: usize, record_bytes: usize) -> Vec<Vec<u8>> {
        (0..records)
            .map(|j| {
                (0..record_bytes)
                    .map(|b| ((j * 7 + b * 31 + 1) % 251) as u8)
                    .collect()
            })
            .collect()
    }

    /// The property that matters: every index returns its own record, and the
    /// server never sees the index.
    #[test]
    fn every_record_is_retrievable() {
        let (records, record_bytes) = (8, 6);
        let pir = Pir::new(TEST, records, record_bytes).unwrap();
        let (ck, pk) = pir.bfv.keygen();
        let data = db(records, record_bytes);
        let encoded = pir.encode_db(&data).unwrap();

        for (i, want) in data.iter().enumerate() {
            let q = pir.query(&pk, i).unwrap();
            let a = pir.answer(&encoded, &q);
            let got = pir.decode(&ck, &a, i).unwrap();
            assert_eq!(&got, want, "record {i} came back wrong");
        }
    }

    /// A database too big for one polynomial has to shard, and retrieval must
    /// still work across the shard boundary.
    #[test]
    fn sharding_preserves_retrieval() {
        // n=64, 2 bytes/word, 8-byte records ⇒ 4 words ⇒ 16 records per shard.
        let (records, record_bytes) = (40, 8);
        let pir = Pir::new(TEST, records, record_bytes).unwrap();
        assert!(pir.layout.shards > 1, "this case must actually shard");
        let (ck, pk) = pir.bfv.keygen();
        let data = db(records, record_bytes);
        let encoded = pir.encode_db(&data).unwrap();

        for i in [0, 1, 15, 16, 17, 31, 32, 39] {
            let q = pir.query(&pk, i).unwrap();
            let a = pir.answer(&encoded, &q);
            assert_eq!(a.len(), pir.layout.shards, "every shard must be answered");
            assert_eq!(pir.decode(&ck, &a, i).unwrap(), data[i], "record {i}");
        }
    }

    /// Two queries for the same index must not be byte-identical, or the server
    /// could link repeated lookups even without learning the index.
    #[test]
    fn queries_are_randomized() {
        let pir = Pir::new(TEST, 8, 4).unwrap();
        let (_, pk) = pir.bfv.keygen();
        let a = pir.query(&pk, 3).unwrap();
        let b = pir.query(&pk, 3).unwrap();
        assert_ne!(a.c, b.c, "BFV encryption must be randomized");
    }

    /// A plaintext multiply must not consume the depth-1 budget: the answer has
    /// to be an ordinary degree-1 ciphertext, decryptable under `s`.
    #[test]
    fn the_answer_stays_depth_zero() {
        let pir = Pir::new(TEST, 4, 4).unwrap();
        let (_, pk) = pir.bfv.keygen();
        let q = pir.query(&pk, 1).unwrap();
        let encoded = pir.encode_db(&db(4, 4)).unwrap();
        let a = pir.answer(&encoded, &q);
        assert_eq!(q.key_pow(), 1);
        assert_eq!(
            a[0].key_pow(),
            1,
            "plaintext multiply must not raise degree"
        );
    }

    #[test]
    fn out_of_range_and_oversized_inputs_are_refused() {
        let pir = Pir::new(TEST, 4, 4).unwrap();
        let (_, pk) = pir.bfv.keygen();
        assert!(pir.query(&pk, 4).is_err());
        assert!(pir.encode_db(&db(3, 4)).is_err(), "wrong record count");
        let mut too_long = db(4, 4);
        too_long[2] = vec![0u8; 5];
        assert!(pir.encode_db(&too_long).is_err(), "oversized record");
        // A record wider than the ring cannot be laid out at all.
        assert!(Pir::new(TEST, 1, 8 * TEST.n).is_err());
    }

    /// Short records are padded, and padding must not corrupt the answer.
    #[test]
    fn short_records_round_trip_through_padding() {
        let pir = Pir::new(TEST, 4, 5).unwrap();
        let (ck, pk) = pir.bfv.keygen();
        let data: Vec<Vec<u8>> = vec![vec![1], vec![2, 3], vec![4, 5, 6, 7, 8], vec![]];
        let encoded = pir.encode_db(&data).unwrap();
        for (i, rec) in data.iter().enumerate() {
            let q = pir.query(&pk, i).unwrap();
            let a = pir.answer(&encoded, &q);
            let got = pir.decode(&ck, &a, i).unwrap();
            assert_eq!(&got[..rec.len()], &rec[..], "record {i} prefix");
            assert!(
                got[rec.len()..].iter().all(|&b| b == 0),
                "padding must be zero"
            );
        }
    }
}
