//! **LowGear** — Beaver triples from somewhat-homomorphic encryption, the
//! SHE half of the preprocessing trade-off.
//!
//! # Why this exists
//!
//! [`crate::mpcc::semi2k`] exists because the dishonest-majority arm of the
//! research agenda "was a **prediction** ... and the prediction rests entirely
//! on the claim that a Beaver triple costs ~3.1 KB of preprocessing." It made
//! that measurable for the **OT** route: 3104 B/triple measured against 3136
//! predicted, 1% error.
//!
//! LowGear was the other half of the same claim, and it was *not* measured.
//! `mpcc::protocols` carried it as `Status::Planned("P3 — reuses fhe::bfv")`
//! with a literature figure of ~384 B/triple, and `mpcc::cost` attached an
//! explicit caveat: *"LowGear trades bandwidth for CPU: its preprocessing is
//! compute-bound, which this bandwidth-only offline term does not capture."*
//! This module removes both the estimate and the caveat by running the
//! protocol.
//!
//! # The protocol (Overdrive/LowGear, semi-honest, two parties)
//!
//! A Beaver triple is `a·b = c` with `a = a₀+a₁`, `b = b₀+b₁` additively shared.
//! Expanding,
//!
//! ```text
//! c = a₀b₀ + a₁b₁ + a₀b₁ + a₁b₀
//!     └ local ┘   └── needs OLE ──┘
//! ```
//!
//! — structurally identical to [`crate::mpcc::semi2k`]. The *only* difference
//! is how the two cross terms are computed. Semi2k uses Gilboa's OLE over
//! `2k` correlated OTs per triple. LowGear uses one **ciphertext × plaintext**
//! multiplication:
//!
//! ```text
//! P_A → P_B :  ct = Enc_{pk_A}(x)                      (x in n CRT slots)
//! P_B       :  ct' = ct ⊙ y  +  (−r)  +  flooding      (y, r in n slots)
//! P_B → P_A :  ct'
//! P_A       :  decrypt ct' → x∘y − r
//! ```
//!
//! P_A holds `x∘y − r`, P_B holds `r`: an additive sharing of `x∘y`, `n` slots
//! at a time. Note what is *not* here — no relinearization, no bootstrapping,
//! no evaluation keys, and **no ciphertext×ciphertext multiply**: one operand
//! is a plaintext, so this costs *zero* multiplicative depth. Each party uses
//! its own key pair (this is what makes it Low-, not High-, Gear: pairwise
//! keys rather than a shared threshold key, so there is no distributed key
//! generation).
//!
//! # Why slots, and why this could not have been built before
//!
//! Everything else in this crate packs plaintexts into *coefficients*, where a
//! ring product is a convolution. An OLE needs `n` **independent** products, so
//! it needs [`crate::fhe::slots`] — CRT packing under a prime `t ≡ 1 (mod 2n)`,
//! which is why `BgvParams::t` had to stop being a power of two. With
//! coefficient packing one ciphertext pair yields *one* useful product, i.e.
//! ~131 KB per triple: **40× worse than the OT it replaces**. Slot packing is
//! the difference between this protocol winning by 16× and losing by 40×.
//!
//! # Parameters, and the flooding budget that fixes them
//!
//! The responder's returned ciphertext has noise that depends on its private
//! `y`, and the requester holds the secret key and can read that noise. So `y`
//! must be hidden by **flooding** ([`crate::fhe::bgv::Bgv::add_flooding_noise`]), and the
//! flooded noise must still decrypt. That single requirement pins the
//! plaintext modulus:
//!
//! ```text
//! ‖noise after ct ⊙ y‖ ≈ t · n · ‖y‖ · ‖e‖ ≈ t · n · (t/2) · 3n
//! flooded by 2^σ, must stay under q/2:
//!     t² · n² · 3 · 2^(σ−1) < q/2
//! ```
//!
//! At `n = 8192`, `q = 2^190`, `σ = 40` this gives `t < 2^61`. That is the whole
//! reason the parameters are as large as they are: the field has to carry
//! `bit_width + KAPPA` bits or the engine cannot bit-decompose, cannot compare,
//! and cannot run a private layer.
//!
//! It ran at `n = 4096, q = 2^100` until 2026-07-30, where the same inequality
//! gives a ~17-bit field — enough to multiply, not enough to compare. Widening
//! `q` past `u128` needed [`crate::fhe::bigint`]'s width-generic integers,
//! `ntt`'s wide profile, and [`crate::fhe::bgv_wide`]. Overdrive itself runs
//! `q` of 250–350 bits at `n = 16384–32768` for a 128-bit field.
//!
//! **Two bugs lived in this budget, so treat it as load bearing.** The flooding
//! mask was `log2_bound + 1` bits rather than `log2_bound`, doubling the noise
//! past `q/2` and silently invalidating *every* triple; and the parameter search
//! accepted a set with 0.29 bits of slack. `LowGearParams::validate` now
//! demands 2 bits, and `deployed_parameters_produce_valid_triples` is not
//! `#[ignore]`d — while it was, the first symptom was a wrong ReLU three layers
//! up.
//!
//! # Scope, stated
//!
//! * **Two parties.** Like Semi2k, and for the same reason: the OLE is
//!   pairwise, and two is the dishonest-majority case the hybrid literature
//!   works in.
//! * **Semi-honest.** No MACs, no sacrifice, and in particular **no
//!   zero-knowledge proofs of plaintext knowledge**. A malicious party can send
//!   a malformed ciphertext (e.g. with huge noise) and learn the other party's
//!   input from the decryption failure pattern. ZKPoPK is the bulk of the
//!   Overdrive paper and is deliberately *not* here — this is the LowGear
//!   analogue of Semi2k, not of SPDZ.
//! * **Local simulation.** Both parties run in this process; bytes are counted
//!   as they would go on a wire (`OleStats`), exactly as Semi2k counts OT
//!   traffic. No socket is involved, so this measures communication *volume*
//!   and local CPU, not latency.

use rand::Rng;

use crate::fhe::bgv_wide::{BgvWide, WideParams};
use crate::fhe::slots;

use super::vm::{Counters, Modulus, ShareEngine};

// ─── Parameters ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowGearParams {
    /// Ring dimension — also the number of triples produced per OLE batch.
    pub n: usize,
    /// log2 of the ciphertext modulus.
    pub log2_q: u32,
    /// Prime plaintext modulus, `t ≡ 1 (mod 2n)`. This is the field the
    /// triples live in.
    pub t: u64,
    /// Statistical security parameter for noise flooding.
    pub sec: u32,
}

impl LowGearParams {
    /// The deployed set: `n = 8192`, `q = 2^190`, `σ = 40`, and the largest
    /// prime the flooding budget allows — which comes out ~2^61.
    ///
    /// **Why so large.** The field has to carry `bit_width + KAPPA` bits or the
    /// engine cannot bit-decompose, and without bit-decomposition it cannot run
    /// a comparison, and without a comparison it cannot run a private layer.
    /// Working back through the budget below, a ~61-bit field needs
    /// `q ≈ 2^190`; `q = 2^190` needs 256-bit coefficients
    /// ([`crate::fhe::bgv_wide`]) and `n = 8192` to stay inside the ~218-bit
    /// RLWE ceiling at 128-bit security.
    ///
    /// This was `n = 4096, q = 2^100` until 2026-07-30, giving a ~17-bit field
    /// that could multiply but not compare — so `bench-layer`'s SHE row had to
    /// be *composed* from two measurements rather than measured. That is the
    /// gap these parameters close.
    pub fn default_params() -> LowGearParams {
        Self::largest_usable(8192, 190, 40)
            .expect("a slot prime fits the flooding budget at n=8192, q=2^190, sec=40")
    }

    /// The largest plaintext modulus that both admits slots **and** passes
    /// [`validate`](Self::validate).
    ///
    /// [`max_plaintext_modulus`] is a closed-form starting point, but it and
    /// the validator round differently (the flooding bound takes a `ceil`), so
    /// the analytic cap can be off by a bit. Rather than pad it with a fudge
    /// constant, step down through slot primes until the *actual* check
    /// passes — the two can then never disagree.
    pub fn largest_usable(n: usize, log2_q: u32, sec: u32) -> Option<LowGearParams> {
        let mut cap = max_plaintext_modulus(n, log2_q, sec);
        // **Halve** on failure, never decrement. The constraint is on `log2 t`,
        // so stepping the cap down by 1 barely moves it: an earlier version did
        // exactly that and scanned ~10^13 candidates before anyone noticed it
        // had not hung. Halving drops `log2 t` by one per iteration, so this
        // terminates in at most `log2 cap` rounds.
        while cap > 1 {
            if let Some(t) = slots::largest_slot_prime(n, cap) {
                let candidate = LowGearParams { n, log2_q, t, sec };
                if candidate.validate().is_ok() {
                    return Some(candidate);
                }
            }
            cap /= 2;
        }
        None
    }

    /// Small parameters for fast tests. **Not secure** — the ring is tiny.
    pub fn test_small() -> LowGearParams {
        LowGearParams {
            n: 64,
            log2_q: 190,
            t: 3329,
            sec: 20,
        }
    }

    /// Bits per field element on the wire — **exactly**
    /// `Modulus::Prime(t).order_bits()`, so the registry's `field_bits`, the
    /// domain programs are lowered for, and the bytes the engine charges all
    /// agree. `mpcc::protocols`' spec must carry this same number, and
    /// `tests::registry_field_bits_matches_the_engine` enforces it.
    pub fn field_bits(&self) -> u32 {
        64 - self.t.leading_zeros()
    }

    fn bgv(&self) -> WideParams {
        WideParams {
            n: self.n,
            log2_q: self.log2_q,
            t: self.t,
        }
    }

    /// log2 of the flooding bound: `σ` bits above the input-dependent noise
    /// `n · ‖y‖ · ‖e‖ ≈ n · (t/2) · 3n`.
    pub fn log2_flood_bound(&self) -> u32 {
        let n = self.n as f64;
        let input_dependent = n * (self.t as f64 / 2.0) * (3.0 * n);
        self.sec + input_dependent.log2().ceil() as u32
    }

    /// Reject parameters whose flooded noise would not decrypt, or that cannot
    /// carry slots. Returns the spare bits in the noise budget.
    pub fn validate(&self) -> Result<f64, String> {
        if !slots::supports_slots(self.n, self.t) {
            return Err(format!(
                "t = {} does not admit {} slots (needs a prime ≡ 1 mod {})",
                self.t,
                self.n,
                2 * self.n
            ));
        }
        // Total magnitude that must stay under q/2 = 2^(log2_q − 1).
        let used = (self.t as f64).log2() + self.log2_flood_bound() as f64;
        let budget = self.log2_q as f64 - 1.0;
        // Demand real margin, not just `<`. The first wide parameter set passed
        // this check with **0.29 bits** to spare and produced silently invalid
        // triples once a one-bit slip elsewhere ate the margin. Two bits is
        // cheap here — it costs one bit of field, and the field has ~30 bits
        // more than bit-decomposition needs.
        const MIN_SLACK_BITS: f64 = 2.0;
        if used + MIN_SLACK_BITS >= budget {
            return Err(format!(
                "flooded noise 2^{used:.1} leaves under {MIN_SLACK_BITS} bits against the \
                 correctness bound q/2 = 2^{budget:.0}; lower t or sec, or raise log2_q"
            ));
        }
        Ok(budget - used)
    }

    /// Serialized bytes of one ciphertext on the wire: 2 ring elements at
    /// `⌈log₂q/8⌉` bytes per coefficient — what a real implementation sends,
    /// not the 32 bytes an `I256` occupies in memory.
    pub fn ct_wire_bytes(&self) -> usize {
        2 * self.n * self.log2_q.div_ceil(8) as usize
    }
}

/// The largest plaintext modulus whose flooded OLE noise still decrypts.
///
/// Solves `t² · n² · 3 · 2^(σ−1) < 2^(log2_q − 1)` for `t` — the inequality
/// derived in the module docs.
pub fn max_plaintext_modulus(n: usize, log2_q: u32, sec: u32) -> u64 {
    let n = n as f64;
    // `validate` compares `log2 t + flood_bound` against `log2_q − 1`, and
    // `flood_bound` takes a `ceil` — so allow a whole bit for that rounding or
    // the closed form sits just above what the check accepts.
    let log2_rest = (n * n * 3.0 / 2.0).log2() + sec as f64 + 1.0;
    let log2_t = ((log2_q as f64 - 1.0) - log2_rest) / 2.0;
    if log2_t <= 1.0 {
        return 0;
    }
    2f64.powf(log2_t) as u64
}

// ─── Wire accounting ─────────────────────────────────────────────────────────

/// Communication actually moved, counted as it would go on a wire.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct OleStats {
    /// Ciphertexts transmitted (both directions).
    pub ciphertexts: usize,
    pub bytes: usize,
    /// Sequential message rounds. One OLE is 2 (send, return); the two cross
    /// terms are independent and batch into the same 2.
    pub rounds: usize,
    pub triples: usize,
}

impl OleStats {
    pub fn bytes_per_triple(&self) -> f64 {
        if self.triples == 0 {
            0.0
        } else {
            self.bytes as f64 / self.triples as f64
        }
    }
}

// ─── Triples ─────────────────────────────────────────────────────────────────

/// One party's additive shares of a batch of triples, slot-wise mod `t`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TripleShares {
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub c: Vec<u64>,
}

fn add_mod(x: u64, y: u64, t: u64) -> u64 {
    let s = x as u128 + y as u128;
    (s % t as u128) as u64
}

fn mul_mod(x: u64, y: u64, t: u64) -> u64 {
    ((x as u128 * y as u128) % t as u128) as u64
}

fn random_vec(n: usize, t: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0..t)).collect()
}

/// The engine. Holds one BGV instance and each party's key material.
pub struct LowGear {
    pub params: LowGearParams,
    bgv: BgvWide,
    /// Per-party `(client_key, public_key)`. LowGear is pairwise-keyed: each
    /// party has its own, and there is no distributed key generation.
    keys: [(Vec<u8>, Vec<u8>); 2],
}

impl LowGear {
    pub fn new(params: LowGearParams) -> Result<LowGear, String> {
        params.validate()?;
        let bgv = BgvWide::new(params.bgv())?;
        let k0 = bgv.keygen();
        let k1 = bgv.keygen();
        Ok(LowGear {
            params,
            bgv,
            keys: [k0, k1],
        })
    }

    /// One **oblivious linear evaluation** over `n` slots.
    ///
    /// `requester` holds `x` and ends up with `x∘y − r`; the responder holds
    /// `y`, picks `r`, and keeps it. Returns `(requester_share, responder_share)`
    /// and the traffic.
    ///
    /// The responder's steps are all depth-free: a plaintext multiply, a
    /// plaintext addition, and a noise addition. Nothing here raises the
    /// ciphertext degree.
    pub fn ole(
        &self,
        requester: usize,
        x: &[u64],
        y: &[u64],
    ) -> Result<(Vec<u64>, Vec<u64>, OleStats), String> {
        let p = &self.params;
        let (n, t) = (p.n, p.t);
        if x.len() != n || y.len() != n {
            return Err(format!("OLE operands must have {n} slots"));
        }
        let (ck, pk) = &self.keys[requester];

        // Requester: encrypt x in slots, send.
        let ct = self.bgv.encrypt_poly(pk, &slots::encode(x, n, t)?);

        // Responder: multiply by its plaintext y, subtract a fresh mask r,
        // then flood so the returned noise cannot reveal y.
        let r = random_vec(n, t);
        let neg_r: Vec<u64> = r.iter().map(|v| (t - v) % t).collect();
        let prod = self.bgv.mul_plain_poly(&ct, &slots::encode(y, n, t)?);
        let masked = self
            .bgv
            .add_plain_poly(&prod, &slots::encode(&neg_r, n, t)?);
        let flooded = self.bgv.add_flooding_noise(&masked, p.log2_flood_bound());

        // Requester: decrypt to its share.
        let share = slots::decode(&self.bgv.decrypt_poly(ck, &flooded), n, t)?;

        let stats = OleStats {
            ciphertexts: 2,
            bytes: 2 * p.ct_wire_bytes(),
            rounds: 2,
            triples: 0, // counted by the caller, which knows the batch shape
        };
        Ok((share, r, stats))
    }

    /// Generate one batch of `n` Beaver triples, returning both parties'
    /// shares and the traffic.
    ///
    /// Two OLEs, one per cross term. They are independent, so a real
    /// deployment issues them together — hence 2 rounds, not 4.
    pub fn generate_batch(&self) -> Result<(TripleShares, TripleShares, OleStats), String> {
        let p = &self.params;
        let (n, t) = (p.n, p.t);

        let (a0, b0) = (random_vec(n, t), random_vec(n, t));
        let (a1, b1) = (random_vec(n, t), random_vec(n, t));

        // Cross term a₀∘b₁: party 0 requests (holds a₀), party 1 responds.
        let (u0, v1, s1) = self.ole(0, &a0, &b1)?;
        // Cross term a₁∘b₀: party 1 requests (holds a₁), party 0 responds.
        let (u1, v0, s2) = self.ole(1, &a1, &b0)?;

        // c_i = a_i∘b_i + (this party's shares of both cross terms)
        let c0: Vec<u64> = (0..n)
            .map(|i| add_mod(add_mod(mul_mod(a0[i], b0[i], t), u0[i], t), v0[i], t))
            .collect();
        let c1: Vec<u64> = (0..n)
            .map(|i| add_mod(add_mod(mul_mod(a1[i], b1[i], t), u1[i], t), v1[i], t))
            .collect();

        let stats = OleStats {
            ciphertexts: s1.ciphertexts + s2.ciphertexts,
            bytes: s1.bytes + s2.bytes,
            rounds: 2, // the two OLEs are independent and batch together
            triples: n,
        };
        Ok((
            TripleShares {
                a: a0,
                b: b0,
                c: c0,
            },
            TripleShares {
                a: a1,
                b: b1,
                c: c1,
            },
            stats,
        ))
    }

    /// Generate at least `count` triples, batching `n` at a time.
    pub fn generate(&self, count: usize) -> Result<(TripleShares, TripleShares, OleStats), String> {
        let mut p0 = TripleShares {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
        };
        let mut p1 = TripleShares {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
        };
        let mut total = OleStats::default();
        while p0.a.len() < count {
            let (s0, s1, st) = self.generate_batch()?;
            p0.a.extend(s0.a);
            p0.b.extend(s0.b);
            p0.c.extend(s0.c);
            p1.a.extend(s1.a);
            p1.b.extend(s1.b);
            p1.c.extend(s1.c);
            total.ciphertexts += st.ciphertexts;
            total.bytes += st.bytes;
            total.rounds += st.rounds;
            total.triples += st.triples;
        }
        Ok((p0, p1, total))
    }

    /// Check a batch: `(a₀+a₁)·(b₀+b₁) ≡ c₀+c₁ (mod t)` in every slot.
    pub fn triples_are_valid(&self, p0: &TripleShares, p1: &TripleShares) -> bool {
        let t = self.params.t;
        (0..p0.a.len()).all(|i| {
            let a = add_mod(p0.a[i], p1.a[i], t);
            let b = add_mod(p0.b[i], p1.b[i], t);
            let c = add_mod(p0.c[i], p1.c[i], t);
            mul_mod(a, b, t) == c
        })
    }
}

/// Per-triple bytes for a `field_bits`-wide field, built by CRT over as many
/// `t`-sized primes as it takes.
///
/// The flooding budget caps a single plaintext modulus at ~17 bits here, so a
/// 64-bit field needs `⌈64/log₂t⌉` independent instances. This is the standard
/// construction and the only honest way to compare against Semi2k's `Z_2^64`.
pub fn wider_field_cost(
    params: &LowGearParams,
    field_bits: u32,
    one_limb_bytes: f64,
) -> (u32, f64) {
    let limb_bits = (params.t as f64).log2().floor() as u32;
    let limbs = field_bits.div_ceil(limb_bits.max(1));
    (limbs, one_limb_bytes * limbs as f64)
}

// ─── The online phase: a real, runnable protocol ─────────────────────────────

/// LowGear as a **VM-executable protocol**, not just a triple factory.
///
/// The online phase is Beaver's, identical to [`crate::mpcc::semi2k`]'s and
/// `DealerEngine`'s — which is the point. Everything that distinguishes these
/// three protocols lives in *where the triples come from*:
///
/// | | triples from | domain |
/// |---|---|---|
/// | `dealer` | a trusted party, free | `Z_2^k` |
/// | `semi2k` | Gilboa OLE over real OT | `Z_2^k` |
/// | **`lowgear`** | **SHE, real BGV** | **`F_t`** |
///
/// So a `mpcc calibrate` row for `lowgear` measures exactly the thing the cost
/// model was guessing at, with the online phase held constant against two
/// protocols whose predictions are already validated.
///
/// # Field width, and what it now permits
///
/// The field is `t ≈ 2^61`, fixed by the flooding budget at `q = 2^190` (see
/// the module docs). That is wide enough for `mpcc::vm`'s masked
/// bit-decomposition, which needs [`crate::mpcc::vm::KAPPA`] = 26 bits of
/// statistical headroom above the value width — so **comparison-shaped programs
/// run**, and with them whole private layers.
///
/// Until 2026-07-30 the field was ~17 bits (`q = 2^100`, capped by `u128`
/// coefficients) and this engine could multiply but not compare;
/// `mpcc::engine_can_run_program` refused those programs rather than answer
/// them wrongly. Widening the coefficient type
/// ([`crate::fhe::bigint`], [`crate::fhe::bgv_wide`]) is what removed the
/// restriction. Overdrive itself runs `q` of 250–350 bits at
/// `n = 16384–32768` for a 128-bit field; 61 bits is enough for this repo's
/// 32-bit values.
pub struct LowGearEngine {
    gen: LowGear,
    m: Modulus,
    counters: Counters,
    /// Triples not yet consumed, refilled a whole OLE batch at a time.
    pool: Vec<(Additive, Additive, Additive)>,
}

/// One party's additive share; the simulation holds both parties' views.
pub type Additive = Vec<u64>;

impl LowGearEngine {
    pub fn new(parties: usize, params: LowGearParams) -> Result<Self, String> {
        if parties != 2 {
            return Err(format!(
                "lowgear here is two-party (the OLE is pairwise, as in semi2k); got {parties}"
            ));
        }
        let gen = LowGear::new(params)?;
        Ok(LowGearEngine {
            m: Modulus::Prime(params.t),
            gen,
            counters: Counters::default(),
            pool: Vec::new(),
        })
    }

    /// Bits of statistical headroom the field leaves above a `value_bits`-wide
    /// value — what masked opening needs and this field mostly does not have.
    pub fn masking_headroom(&self, value_bits: u32) -> i32 {
        self.gen.params.field_bits() as i32 - value_bits as i32
    }

    fn charge(&mut self, elems: u64) {
        // One peer, matching `DealerEngine` and `Semi2kEngine` so all three are
        // comparable against the same prediction.
        self.counters.online_bytes += elems * self.m.element_bytes();
    }

    fn refill(&mut self) -> Result<(), String> {
        let (p0, p1, stats) = self.gen.generate_batch()?;
        self.counters.offline_bytes += stats.bytes as u64;
        self.counters.triples_generated += stats.triples as u64;
        for i in 0..p0.a.len() {
            self.pool.push((
                vec![p0.a[i], p1.a[i]],
                vec![p0.b[i], p1.b[i]],
                vec![p0.c[i], p1.c[i]],
            ));
        }
        Ok(())
    }

    fn triple(&mut self) -> (Additive, Additive, Additive) {
        if self.pool.is_empty() {
            self.refill()
                .expect("lowgear: SHE triple generation failed");
        }
        self.counters.triples += 1;
        self.pool.pop().expect("pool refilled")
    }

    fn reconstruct(&self, s: &Additive) -> u64 {
        s.iter().fold(0u64, |acc, x| self.m.add(acc, *x))
    }
    fn add_pure(&self, a: &Additive, b: &Additive) -> Additive {
        a.iter().zip(b).map(|(x, y)| self.m.add(*x, *y)).collect()
    }
    fn sub_pure(&self, a: &Additive, b: &Additive) -> Additive {
        a.iter().zip(b).map(|(x, y)| self.m.sub(*x, *y)).collect()
    }
    fn mul_const_pure(&self, a: &Additive, c: u64) -> Additive {
        a.iter().map(|x| self.m.mul(*x, c)).collect()
    }
    fn split(&self, value: u64) -> Additive {
        let mut rng = rand::thread_rng();
        let t = self.gen.params.t;
        let r = rng.gen_range(0..t);
        vec![r, self.m.sub(value % t, r)]
    }
}

impl ShareEngine for LowGearEngine {
    type Share = Additive;

    fn modulus(&self) -> Modulus {
        self.m
    }
    fn parties(&self) -> usize {
        2
    }
    fn counters(&self) -> &Counters {
        &self.counters
    }
    fn counters_mut(&mut self) -> &mut Counters {
        &mut self.counters
    }

    fn input(&mut self, value: u64, _party: usize) -> Additive {
        self.charge(1);
        self.split(value)
    }
    fn constant(&mut self, value: u64) -> Additive {
        vec![value % self.gen.params.t, 0]
    }

    fn add(&mut self, a: &Additive, b: &Additive) -> Additive {
        self.add_pure(a, b)
    }
    fn sub(&mut self, a: &Additive, b: &Additive) -> Additive {
        self.sub_pure(a, b)
    }
    fn add_const(&mut self, a: &Additive, c: u64) -> Additive {
        let mut out = a.clone();
        out[0] = self.m.add(out[0], c % self.gen.params.t);
        out
    }
    fn mul_const(&mut self, a: &Additive, c: u64) -> Additive {
        self.mul_const_pure(a, c % self.gen.params.t)
    }

    /// Beaver, identical to Semi2k's and Dealer's. One round for the whole
    /// batch: both maskings are opened together.
    fn mul_batch(&mut self, pairs: &[(Additive, Additive)]) -> Vec<Additive> {
        self.charge(2 * pairs.len() as u64);
        self.counters.muls += pairs.len() as u64;

        let mut out = Vec::with_capacity(pairs.len());
        for (x, y) in pairs {
            let (a, b, ab) = self.triple();
            let d = self.reconstruct(&self.sub_pure(x, &a));
            let e = self.reconstruct(&self.sub_pure(y, &b));
            let db = self.mul_const_pure(&b, d);
            let ea = self.mul_const_pure(&a, e);
            let mut z = self.add_pure(&ab, &db);
            z = self.add_pure(&z, &ea);
            z[0] = self.m.add(z[0], self.m.mul(d, e));
            out.push(z);
        }
        out
    }

    fn open_batch(&mut self, values: &[Additive]) -> Vec<u64> {
        self.charge(values.len() as u64);
        self.counters.opens += values.len() as u64;
        values.iter().map(|s| self.reconstruct(s)).collect()
    }

    /// `n` shared uniform bits.
    ///
    /// Same simplification as `DealerEngine` and `Semi2kEngine`: drawn locally
    /// and counted, never executed (see `mpcc::vm`). What changed on
    /// 2026-07-30 is that this is *reachable* at all — at the old ~17-bit field
    /// it panicked, because masked opening needs `KAPPA` bits of headroom above
    /// the value width and there were none. The wide parameters give ~61 bits,
    /// so bit-decomposition, comparison and therefore a whole private layer now
    /// run on this engine.
    ///
    /// Generated rather than dealt (`vm::random_bits_batch`): each party
    /// contributes a private bit and the shared bit is their XOR, which costs
    /// `n` real SHE-generated triples and one round for the whole batch.
    fn random_bits(&mut self, n: usize) -> Vec<Additive> {
        super::vm::random_bits_batch(self, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> LowGear {
        LowGear::new(LowGearParams::test_small()).expect("test params validate")
    }

    #[test]
    fn default_parameters_validate_with_slack() {
        let p = LowGearParams::default_params();
        let slack = p.validate().expect("default params must be usable");
        assert!(p.t > 1000, "field should not be degenerate, got t={}", p.t);
        assert!(slots::supports_slots(p.n, p.t));
        assert!(slack > 0.0, "no noise budget left: {slack}");
        // The bound in the module docs: ~61 bits at n=8192, q=2^190, sec=40.
        let log2_t = (p.t as f64).log2();
        assert!((58.0..62.0).contains(&log2_t), "log2 t = {log2_t}");
        // …and that is the whole point: wide enough to bit-decompose.
        assert!(
            p.field_bits() >= 32 + crate::mpcc::vm::KAPPA,
            "field must carry bit_width + KAPPA to run a comparison, got {} bits",
            p.field_bits()
        );
    }

    #[test]
    fn parameters_that_break_decryption_are_rejected() {
        // A plaintext modulus far above the flooding budget must not validate.
        let bad = LowGearParams {
            n: 4096,
            log2_q: 100,
            t: 1 + 8192 * 100_000,
            sec: 40,
        };
        assert!(bad.validate().is_err());
        // A non-slot prime must not validate either.
        let no_slots = LowGearParams {
            n: 4096,
            log2_q: 100,
            t: 3329,
            sec: 40,
        };
        assert!(no_slots.validate().is_err());
    }

    /// The OLE's defining relation: the two shares sum to the componentwise
    /// product.
    #[test]
    fn ole_shares_reconstruct_the_product() {
        let lg = small();
        let (n, t) = (lg.params.n, lg.params.t);
        let x = random_vec(n, t);
        let y = random_vec(n, t);
        let (share_req, share_resp, stats) = lg.ole(0, &x, &y).unwrap();
        for i in 0..n {
            assert_eq!(
                add_mod(share_req[i], share_resp[i], t),
                mul_mod(x[i], y[i], t),
                "slot {i}"
            );
        }
        assert_eq!(stats.ciphertexts, 2);
        assert_eq!(stats.rounds, 2);
    }

    /// Both directions must work — the two cross terms use opposite roles.
    #[test]
    fn ole_works_with_either_party_as_requester() {
        let lg = small();
        let (n, t) = (lg.params.n, lg.params.t);
        let (x, y) = (random_vec(n, t), random_vec(n, t));
        for requester in [0usize, 1] {
            let (u, v, _) = lg.ole(requester, &x, &y).unwrap();
            assert!((0..n).all(|i| add_mod(u[i], v[i], t) == mul_mod(x[i], y[i], t)));
        }
    }

    /// The headline: real Beaver triples, checked by reconstruction.
    #[test]
    fn generated_triples_are_valid() {
        let lg = small();
        let (p0, p1, stats) = lg.generate_batch().unwrap();
        assert_eq!(stats.triples, lg.params.n);
        assert!(
            lg.triples_are_valid(&p0, &p1),
            "a·b must equal c in every slot"
        );
    }

    /// Shares must individually look random — a share equal to the value it
    /// hides would be a catastrophic masking bug.
    #[test]
    fn shares_are_masked_not_plaintext() {
        let lg = small();
        let (p0, p1, _) = lg.generate_batch().unwrap();
        let t = lg.params.t;
        let products: Vec<u64> = (0..p0.a.len())
            .map(|i| {
                mul_mod(
                    add_mod(p0.a[i], p1.a[i], t),
                    add_mod(p0.b[i], p1.b[i], t),
                    t,
                )
            })
            .collect();
        // Party 0's c-share must not simply be the product.
        assert_ne!(p0.c, products);
        assert_ne!(p1.c, products);
    }

    #[test]
    fn generate_batches_until_the_count_is_met() {
        let lg = small();
        let want = lg.params.n + 5;
        let (p0, p1, stats) = lg.generate(want).unwrap();
        assert!(p0.a.len() >= want);
        assert_eq!(stats.triples, p0.a.len());
        assert_eq!(stats.ciphertexts, 4 * (p0.a.len() / lg.params.n));
        assert!(lg.triples_are_valid(&p0, &p1));
    }

    #[test]
    fn wrong_slot_counts_are_rejected() {
        let lg = small();
        assert!(lg.ole(0, &[1, 2, 3], &[1, 2, 3]).is_err());
    }

    /// The registry's `field_bits` must equal what the engine actually charges
    /// per element, or `calibrate` compares a prediction against a measurement
    /// taken in a different field — which is exactly the class of silent error
    /// this whole layer exists to prevent.
    #[test]
    fn registry_field_bits_matches_the_engine() {
        let p = LowGearParams::default_params();
        let spec = crate::mpcc::find("lowgear").expect("lowgear is in the registry");
        // `cost.rs` prices `field_bits / 8` bytes per element; the VM charges
        // `ceil(order_bits / 8)`. Those must come out equal or `calibrate`
        // reports a byte error that is an accounting artifact, not a finding.
        let predicted_bytes = spec.cost.field_bits as u64 / 8;
        let charged_bytes = Modulus::Prime(p.t).element_bytes();
        assert_eq!(
            predicted_bytes,
            charged_bytes,
            "registry field_bits={} predicts {predicted_bytes} B/element, but the VM \
             charges {charged_bytes} B for t={} ({} bits)",
            spec.cost.field_bits,
            p.t,
            p.field_bits()
        );
        // The wire width must be byte-aligned and actually hold the modulus.
        assert_eq!(
            spec.cost.field_bits % 8,
            0,
            "wire width must be byte-aligned"
        );
        assert!(spec.cost.field_bits >= p.field_bits());
    }

    /// The online phase is a real protocol: it multiplies, opens, and gets the
    /// right answer, with the triples coming from SHE.
    #[test]
    fn engine_multiplies_correctly_from_she_triples() {
        let mut e = LowGearEngine::new(2, LowGearParams::test_small()).unwrap();
        let t = 3329u64;
        for (x, y) in [(6u64, 7u64), (0, 5), (100, 33), (t - 1, 2)] {
            let sx = e.input(x, 0);
            let sy = e.input(y, 1);
            let sz = e.mul_batch(&[(sx, sy)]);
            assert_eq!(e.open_batch(&sz)[0], (x * y) % t, "{x}*{y} mod {t}");
        }
        assert!(
            e.counters().offline_bytes > 0,
            "SHE preprocessing must be charged"
        );
        assert!(e.counters().triples_generated >= e.counters().triples);
    }

    #[test]
    fn engine_local_operations_are_free_and_correct() {
        let mut e = LowGearEngine::new(2, LowGearParams::test_small()).unwrap();
        let before = e.counters().online_bytes;
        let a = e.constant(10);
        let b = e.constant(4);
        let s = e.add(&a, &b);
        let d = e.sub(&a, &b);
        let m = e.mul_const(&a, 3);
        let k = e.add_const(&a, 5);
        assert_eq!(
            e.counters().online_bytes,
            before,
            "local ops must not charge"
        );
        assert_eq!(e.open_batch(&[s, d, m, k]), vec![14, 6, 30, 15]);
    }

    #[test]
    fn engine_rejects_more_than_two_parties() {
        assert!(LowGearEngine::new(3, LowGearParams::test_small()).is_err());
    }

    /// `random_bits` used to panic — the ~17-bit field had no masking headroom.
    /// At the wide parameters it works, which is what unlocks comparisons.
    #[test]
    fn random_bits_produces_valid_shared_bits() {
        let mut e = LowGearEngine::new(2, LowGearParams::default_params()).unwrap();
        let bits = e.random_bits(64);
        assert_eq!(bits.len(), 64);
        let opened = e.open_batch(&bits);
        assert!(
            opened.iter().all(|&b| b == 0 || b == 1),
            "every share must open to a bit"
        );
        assert_eq!(e.counters().random_bits, 64);
    }

    /// The headroom that makes bit-decomposition legal, stated as a test rather
    /// than a comment: a 32-bit value in a ~61-bit field leaves ~29 bits, above
    /// `KAPPA = 26`.
    #[test]
    fn the_field_has_masking_headroom_for_comparisons() {
        let e = LowGearEngine::new(2, LowGearParams::default_params()).unwrap();
        let headroom = e.masking_headroom(32);
        assert!(
            headroom >= crate::mpcc::vm::KAPPA as i32,
            "need ≥ KAPPA = {} bits above a 32-bit value, got {headroom}",
            crate::mpcc::vm::KAPPA
        );
    }

    #[test]
    fn wider_field_cost_scales_by_the_number_of_limbs() {
        let p = LowGearParams::default_params();
        let (limbs, bytes) = wider_field_cost(&p, 64, 100.0);
        // A ~61-bit prime needs 2 limbs to cover 64 bits (it used to need 4).
        assert_eq!(
            limbs, 2,
            "expected 2 CRT limbs for a 64-bit field, got {limbs}"
        );
        assert!((bytes - 100.0 * limbs as f64).abs() < 1e-9);
        // The layer's own values are 32-bit, which one limb already covers.
        assert_eq!(wider_field_cost(&p, 32, 100.0).0, 1);
    }

    /// At the **deployed** parameters, where the measurement is taken.
    ///
    /// **Not `#[ignore]`d, deliberately.** This is the only test that exercises
    /// the real noise budget, and while it was ignored a one-bit flooding slip
    /// shipped parameters whose triples were all invalid — caught only by a
    /// wrong answer three layers up. A parameter change must fail here.
    #[test]
    fn deployed_parameters_produce_valid_triples() {
        let lg = LowGear::new(LowGearParams::default_params()).unwrap();
        let (p0, p1, stats) = lg.generate_batch().unwrap();
        assert!(
            lg.triples_are_valid(&p0, &p1),
            "SHE triples must reconstruct at the real budget"
        );
        assert_eq!(stats.triples, lg.params.n);
        // The claim this module exists to check: far under the OT route's
        // measured 3104 B/triple.
        assert!(
            stats.bytes_per_triple() < 3104.0 / 4.0,
            "expected a large win over OT, got {} B/triple",
            stats.bytes_per_triple()
        );
    }
}
