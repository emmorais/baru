//! The bytecode VM — the piece that turns `mpcc` from an analysis tool into a
//! runtime, and therefore the piece that lets [`crate::mpcc::cost`] be
//! *validated* instead of merely believed.
//!
//! MP-SPDZ's structure is one bytecode and many virtual machines, one per
//! protocol.  Here the split is a little finer, because almost everything above
//! the sharing scheme is protocol-independent: a [`ShareEngine`] supplies the
//! four things that genuinely differ between protocols — how a value is shared,
//! opened, multiplied, and where correlated randomness comes from — and the
//! driver in [`execute`] composes the bytecode out of those.  Adding a protocol
//! means implementing one trait, not one VM.
//!
//! ## What the simulator is honest about
//!
//! Every engine runs all parties in one process.  That means:
//!
//! * **Results are real.**  Shares are actually split, actually multiplied
//!   through the protocol's own multiplication procedure, and actually
//!   reconstructed.  A bug in the protocol shows up as a wrong answer.
//! * **Online communication is real.**  Bytes and rounds are counted from the
//!   messages the protocol would have sent, at the point it would have sent
//!   them.
//! * **Preprocessing is real where the engine makes it real.**  Beaver triples
//!   come from the engine: [`crate::mpcc::semi2k`] generates them over real
//!   IKNP correlated OT and [`crate::mpcc::lowgear`] over real BGV slot-OLE, so
//!   their `offline_bytes` is a measurement.  [`crate::mpcc::sim`]'s
//!   `DealerEngine` deals them for free *by definition* — a trusted dealer is
//!   the online lower bound, not a threat model.  Random bits are built from
//!   the same triples by [`random_bits_batch`].  Offline **time** is still not
//!   measured anywhere.
//!
//! That is the split the roadmap needs: P1 calibrates the online half of the
//! cost model against these numbers, and the offline half is measured in bytes
//! for the two engines that generate their own material.
//!
//! ## Coverage
//!
//! The linear and multiplicative core (`Input`, `Add`, `Sub`, `Neg`, `MulC`,
//! `AddC`, `ShlC`, `Mul`, `Mux`, `Open`, `Output`) runs, and so do the
//! comparison-shaped instructions — `LtS`, `LtU`, `EqZ` and `TruncPr` — on top
//! of [`bit_decompose_batch`], which is generic over [`ShareEngine`] and
//! therefore available to *every* protocol rather than to a chosen one.  A
//! batch of them costs one decomposition's rounds, not one per value.
//!
//! What does **not** run: secret `Div` (needs a restoring long-division circuit
//! on top of the comparison machinery), the raw bit-level ops `Xor`, `Not`,
//! `And`, `BitDecomp`, `BitComp` and `PopCount` (no binary-domain engine
//! implements [`ShareEngine`] — see `protocols.rs`, where every binary row is
//! `Planned`), and `Convert` (cross-domain conversion, which needs two engines
//! at once and real daBits).  Each is refused by name rather than approximated,
//! because executing one by peeking at the plaintext would produce
//! communication counts that are quietly fictional.

use std::collections::HashMap;

use rand::Rng;
use serde::Serialize;

use super::ir::{Op, Program, Reg};
use super::schedule::Schedule;

// ─── Arithmetic ───────────────────────────────────────────────────────────────

/// The algebraic structure an engine computes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modulus {
    /// `Z_{2^k}`, `k <= 64`.
    Ring { k: u32 },
    /// `Z_p` for an odd prime `p < 2^63` (so products fit in `u128`).
    Prime(u64),
}

impl Modulus {
    /// `2^61 - 1`, a Mersenne prime: big enough for 32-bit values with room for
    /// statistical masking, small enough that products fit in a `u128`.
    pub const MERSENNE61: Modulus = Modulus::Prime((1u64 << 61) - 1);

    fn mask(k: u32) -> u64 {
        if k >= 64 {
            u64::MAX
        } else {
            (1u64 << k) - 1
        }
    }

    pub fn order_bits(&self) -> u32 {
        match self {
            Modulus::Ring { k } => *k,
            Modulus::Prime(p) => 64 - p.leading_zeros(),
        }
    }

    pub fn element_bytes(&self) -> u64 {
        u64::from(self.order_bits()).div_ceil(8)
    }

    fn reduce(&self, v: u128) -> u64 {
        match self {
            Modulus::Ring { k } => (v as u64) & Self::mask(*k),
            Modulus::Prime(p) => (v % u128::from(*p)) as u64,
        }
    }

    pub fn add(&self, a: u64, b: u64) -> u64 {
        self.reduce(u128::from(a) + u128::from(b))
    }

    pub fn sub(&self, a: u64, b: u64) -> u64 {
        match self {
            Modulus::Ring { k } => a.wrapping_sub(b) & Self::mask(*k),
            Modulus::Prime(p) => {
                if a >= b {
                    a - b
                } else {
                    p - (b - a)
                }
            }
        }
    }

    pub fn neg(&self, a: u64) -> u64 {
        self.sub(0, a)
    }

    pub fn mul(&self, a: u64, b: u64) -> u64 {
        self.reduce(u128::from(a) * u128::from(b))
    }

    pub fn pow(&self, mut base: u64, mut exp: u64) -> u64 {
        let mut acc = 1u64;
        base = self.reduce(u128::from(base));
        while exp > 0 {
            if exp & 1 == 1 {
                acc = self.mul(acc, base);
            }
            base = self.mul(base, base);
            exp >>= 1;
        }
        acc
    }

    /// Multiplicative inverse.  Prime moduli only — `Z_{2^k}` has no inverse for
    /// even elements, and no engine here needs one.
    pub fn inv(&self, a: u64) -> Result<u64, String> {
        match self {
            Modulus::Prime(p) => {
                if a == 0 {
                    Err("mpcc vm: division by zero in the field".to_string())
                } else {
                    Ok(self.pow(a, p - 2))
                }
            }
            Modulus::Ring { .. } => {
                Err("mpcc vm: Z_2^k has no general multiplicative inverse".to_string())
            }
        }
    }

    /// Encode a signed program value.
    pub fn from_i64(&self, v: i64) -> u64 {
        match self {
            Modulus::Ring { k } => (v as u64) & Self::mask(*k),
            Modulus::Prime(p) => {
                let m = (v.unsigned_abs()) % *p;
                if v < 0 && m != 0 {
                    *p - m
                } else {
                    m
                }
            }
        }
    }

    /// Decode back to a signed program value.  Ring elements sign-extend from
    /// `bit_width` (Baru's convention: 32-bit values carried in wider shares);
    /// field elements use the centred representative.
    pub fn to_i64(&self, v: u64, bit_width: u32) -> i64 {
        match self {
            Modulus::Ring { .. } => {
                let w = bit_width.clamp(1, 64);
                if w == 64 {
                    v as i64
                } else {
                    let low = v & Self::mask(w);
                    if low & (1u64 << (w - 1)) != 0 {
                        low as i64 - (1i64 << w)
                    } else {
                        low as i64
                    }
                }
            }
            Modulus::Prime(p) => {
                if v > p / 2 {
                    v as i64 - *p as i64
                } else {
                    v as i64
                }
            }
        }
    }

    fn random(&self, rng: &mut impl FnMut() -> u64) -> u64 {
        match self {
            Modulus::Ring { k } => rng() & Self::mask(*k),
            Modulus::Prime(p) => {
                // Rejection-free enough: p is close to 2^61, so masking to 61
                // bits and rejecting the rare overflow terminates immediately.
                loop {
                    let v = rng() & ((1u64 << 61) - 1);
                    if v < *p {
                        return v;
                    }
                }
            }
        }
    }
}

/// Draw a uniform element of `m`.
pub fn random_elem(m: Modulus) -> u64 {
    let mut rng = rand::random::<u64>;
    m.random(&mut rng)
}

// ─── Engines ──────────────────────────────────────────────────────────────────

/// Communication and preprocessing actually consumed by a run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Counters {
    /// Bytes sent by **one** party — the same convention [`crate::mpcc::cost`]
    /// predicts, so the two are directly comparable.
    pub online_bytes: u64,
    /// Secret×secret multiplications performed.
    pub muls: u64,
    /// Values opened.
    pub opens: u64,
    /// Beaver triples consumed (0 for protocols that multiply from replicated
    /// or degree-reduced randomness rather than triples).
    pub triples: u64,
    /// Shared random bits consumed.
    pub random_bits: u64,
    /// Triples *generated*, which can exceed [`Counters::triples`] consumed
    /// because generation is batched.
    ///
    /// Offline cost amortizes over the batch, so bytes-per-triple is only
    /// meaningful against this number. Dividing by the consumed count instead
    /// reports the cost of a half-used batch and looks like a 20× modelling miss.
    pub triples_generated: u64,
    /// **Preprocessing** bytes actually moved, for engines that generate their own
    /// correlated randomness rather than assuming a dealer.
    ///
    /// Zero for `DealerEngine` (a trusted dealer hands triples over for free — that
    /// is what makes it a lower bound) and for `ShamirEngine` (honest majority,
    /// multiplication needs no triples). Non-zero only for `Semi2kEngine`, which
    /// runs real OT — and that is what lets `mpcc::ot_cost`'s derivation be checked
    /// against a measurement instead of trusted.
    ///
    /// **Triple generation only.** Preprocessing that runs *on top of* the engine
    /// — random-bit generation — is counted separately in
    /// [`Counters::random_bit_bytes`], because `bench-layer` divides this field by
    /// [`Counters::triples_generated`] to get a bytes-per-triple figure and
    /// anything else landing here would silently corrupt it.
    pub offline_bytes: u64,
    /// Bytes moved **generating random bits**, via [`random_bits_batch`].
    ///
    /// Offline, but not triple generation: this is the input sharing plus the
    /// Beaver openings of the XOR tree. Kept apart from
    /// [`Counters::offline_bytes`] so that field stays a clean per-triple
    /// measurement, and kept out of [`Counters::online_bytes`] because random
    /// bits are **query-independent** — a deployment generates them before the
    /// query arrives, exactly as Falcon's preprocessing does (survey §7.3).
    ///
    /// The triples these consume *are* counted in [`Counters::triples`], so
    /// their share of triple-generation cost is charged the same way every other
    /// triple's is. Only the bit protocol's own traffic is here.
    pub random_bit_bytes: u64,
    /// Rounds spent generating random bits — `⌈log₂(parties)⌉` per batch,
    /// regardless of how many bits it carries. Not part of the program's online
    /// round count, for the same query-independence reason.
    pub random_bit_rounds: u64,
    /// Multiplications spent generating random bits — one per bit per tree level.
    ///
    /// Kept out of [`Counters::muls`] deliberately. That field is validated
    /// against `mpcc::cost`'s independent `bit_decompose_muls` formula, which
    /// counts the multiplications *the program's* algorithm needs; folding
    /// preprocessing into it would make the two disagree by the size of the
    /// preprocessing rather than by a modelling error, and destroy a check that
    /// exists to catch modelling errors.
    pub random_bit_muls: u64,
}

/// The four things that actually differ between secret-sharing protocols.
///
/// Batched by construction: `mul_batch` and `open_batch` each cost one round no
/// matter how many elements they carry, which is what makes the driver's round
/// count match the schedule's dependence depth.
pub trait ShareEngine {
    /// One party's view of a shared value.  In-process simulation holds every
    /// party's view at once, so this is the whole sharing.
    type Share: Clone;

    fn modulus(&self) -> Modulus;
    fn parties(&self) -> usize;
    fn counters(&self) -> &Counters;
    /// Needed by [`random_bits_batch`], which reattributes the traffic it spends
    /// from the online column to the preprocessing one. Without it every engine
    /// would have to repeat that bookkeeping.
    fn counters_mut(&mut self) -> &mut Counters;

    /// `party`'s private input becomes a sharing.  Input distribution is
    /// charged as communication; the driver charges its round.
    fn input(&mut self, value: u64, party: usize) -> Self::Share;
    /// A value every party already knows.  Free.
    fn constant(&mut self, value: u64) -> Self::Share;

    // Local operations — no communication, no round.
    fn add(&mut self, a: &Self::Share, b: &Self::Share) -> Self::Share;
    fn sub(&mut self, a: &Self::Share, b: &Self::Share) -> Self::Share;
    fn add_const(&mut self, a: &Self::Share, c: u64) -> Self::Share;
    fn mul_const(&mut self, a: &Self::Share, c: u64) -> Self::Share;

    // Interactive — one round per call.
    fn mul_batch(&mut self, pairs: &[(Self::Share, Self::Share)]) -> Vec<Self::Share>;
    fn open_batch(&mut self, values: &[Self::Share]) -> Vec<u64>;

    /// `n` shared uniform bits, each a sharing of 0 or 1.
    ///
    /// Everything above the sharing layer (comparison, truncation, bit
    /// decomposition) is built from these, so this is the one extra primitive a
    /// protocol has to supply to become fully featured.
    ///
    /// Engines with a real threat model implement this by delegating to
    /// [`random_bits_batch`], which *runs* a protocol.  `DealerEngine` draws
    /// them locally, which is not a simplification but its definition — a
    /// trusted dealer supplying correlated randomness for free is the online
    /// lower bound.  `Spdz2kEngine` also draws locally, and that **is** a gap:
    /// see its own documentation.
    fn random_bits(&mut self, n: usize) -> Vec<Self::Share>;
}

// ─── Composite primitives ─────────────────────────────────────────────────────

/// Statistical security parameter for masked openings: the mask carries this
/// many bits more than the value, so the opened sum hides it to within
/// `2^-KAPPA`.  With a 33-bit decomposition that needs 59 bits of headroom,
/// which both `Z_2^64` and the 61-bit Mersenne field have.
pub const KAPPA: u32 = 26;

/// The bits of each input, LSB first, and the rounds the decomposition took.
pub type Decomposed<E> = (Vec<Vec<<E as ShareEngine>::Share>>, u64);

/// `n` shared uniform random bits, **generated by running a protocol** rather
/// than drawn locally and counted.
///
/// # Why this is one generic function and not one per engine
///
/// The construction needs nothing an engine does not already provide, so
/// writing it per protocol would be three copies of the same identity. Each
/// party contributes a private random bit `b_p`, and the shared bit is their
/// XOR. Over a ring or a field,
///
/// ```text
/// x ⊕ y  =  x + y − 2xy
/// ```
///
/// so one XOR is one multiplication, and the whole batch of `n` bits at one
/// tree level is **one `mul_batch`** — one round no matter how many bits. The
/// parties' contributions are XOR-reduced as a balanced tree, so a batch costs
/// `⌈log₂(parties)⌉` rounds: **one** at the two parties that
/// [`crate::mpcc::semi2k`] and [`crate::mpcc::lowgear`] are gated to.
///
/// Uniformity needs only *one* honest contributor: `b_p` is uniform and
/// independent of everything else, so `⊕_p b_p` is uniform to anyone missing
/// even a single `b_p`. That is why this is sound at a dishonest majority, and
/// it is the same argument the zero-sum masking terms elsewhere in the repo use.
///
/// # Where the cost lands, and why not in the online column
///
/// Random bits are **query-independent** — a deployment generates them before
/// the query arrives. Charging them to `online_bytes` would make the online
/// figure mean something else, and comparing that against an FHE total is
/// precisely the error survey §7.3 records making three times. So the traffic
/// this spends is moved to [`Counters::random_bit_bytes`] and its rounds to
/// [`Counters::random_bit_rounds`]; a caller that wants the total adds them.
///
/// The triples consumed are left in [`Counters::triples`], so they are charged
/// triple-generation cost exactly like every other triple.
///
/// # Security caveat, stated
///
/// This is sound against a **semi-honest** adversary. A malicious party can
/// contribute a value that is not a bit, and nothing here detects it — the
/// result is then not a sharing of 0 or 1 and
/// [`bit_decompose_batch`] silently returns garbage. Validating it costs one
/// batched `b·(b−1) = 0` check plus an opening; until that exists,
/// [`crate::mpcc::spdz2k`] deliberately does **not** call this.
pub fn random_bits_batch<E: ShareEngine>(engine: &mut E, n: usize) -> Vec<E::Share> {
    if n == 0 {
        return Vec::new();
    }

    // Snapshot the three *online-phase* counters, so everything below can be
    // reattributed to preprocessing. `triples`, `triples_generated` and
    // `offline_bytes` are deliberately left to grow: the triples these
    // multiplications consume are real, and charging them triple-generation cost
    // the same way every other triple is charged is the point.
    let (bytes_before, muls_before, opens_before) = {
        let c = engine.counters();
        (c.online_bytes, c.muls, c.opens)
    };

    let mut rng = rand::thread_rng();
    let mut layers: Vec<Vec<E::Share>> = Vec::with_capacity(engine.parties());
    for p in 0..engine.parties() {
        let mut layer = Vec::with_capacity(n);
        for _ in 0..n {
            layer.push(engine.input(u64::from(rng.gen::<bool>()), p));
        }
        layers.push(layer);
    }

    let mut rounds = 0u64;
    while layers.len() > 1 {
        // Every pair at this level is independent, so they share one round.
        let mut pairs: Vec<(E::Share, E::Share)> = Vec::new();
        let mut lhs: Vec<Vec<E::Share>> = Vec::new();
        let mut rhs: Vec<Vec<E::Share>> = Vec::new();
        let mut odd: Option<Vec<E::Share>> = None;
        for chunk in layers.chunks(2) {
            if let [a, b] = chunk {
                for (x, y) in a.iter().zip(b.iter()) {
                    pairs.push((x.clone(), y.clone()));
                }
                lhs.push(a.clone());
                rhs.push(b.clone());
            } else {
                odd = Some(chunk[0].clone());
            }
        }

        let prods = engine.mul_batch(&pairs);
        rounds += 1;

        let mut next: Vec<Vec<E::Share>> = Vec::with_capacity(lhs.len() + 1);
        let mut k = 0usize;
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            let mut combined = Vec::with_capacity(n);
            for (x, y) in a.iter().zip(b.iter()) {
                let sum = engine.add(x, y);
                let twice = engine.mul_const(&prods[k], 2);
                combined.push(engine.sub(&sum, &twice));
                k += 1;
            }
            next.push(combined);
        }
        next.extend(odd);
        layers = next;
    }

    let counters = engine.counters_mut();
    let bytes = counters.online_bytes.saturating_sub(bytes_before);
    let muls = counters.muls.saturating_sub(muls_before);
    let opens = counters.opens.saturating_sub(opens_before);
    counters.online_bytes = bytes_before;
    counters.muls = muls_before;
    counters.opens = opens_before;
    counters.random_bit_bytes += bytes;
    counters.random_bit_muls += muls;
    counters.random_bit_rounds += rounds;
    counters.random_bits += n as u64;
    debug_assert_eq!(opens, 0, "random-bit generation should not open anything");

    layers.pop().unwrap_or_default()
}

/// Bit-decompose several shared values at once, LSB first.
///
/// This is the primitive everything comparison-shaped reduces to, and the
/// reason phase P2 exists: without it the VM can multiply but cannot branch.
///
/// The protocol is the standard masked-opening construction:
///
/// 1. Draw `nbits + KAPPA` shared random bits per value and assemble
///    `r = Σ 2ⁱ rᵢ` locally.
/// 2. Open `c = x + 2^(nbits-1) + r`.  The offset makes the value non-negative
///    so its `nbits`-bit representation is well defined; the mask hides it.
/// 3. Recover the bits of `u = x + 2^(nbits-1)` as the binary difference
///    `c − r` truncated to `nbits` bits, using a **Kogge–Stone parallel-prefix
///    borrow chain** — `log₂(nbits)` rounds rather than the `nbits` rounds a
///    ripple-borrow chain would take.
/// 4. `x`'s two's-complement bits are `u`'s with the top bit flipped, because
///    adding `2^(nbits-1)` modulo `2^nbits` *is* flipping that bit.
///
/// Two details make the prefix chain cheap.  Because `c` is public, the
/// generate/propagate pair is computed **locally** (`gᵢ = ¬cᵢ ∧ rᵢ`,
/// `pᵢ = ¬(cᵢ ⊕ rᵢ)` each collapse to `rᵢ` or `1 − rᵢ`).  And because
/// `gᵢ · pᵢ = 0` always, the `OR` in the prefix combine is a plain addition —
/// so each combine is two multiplications, all independent, one round per
/// level.
///
/// **Precondition:** every `x` must lie in `[-2^(nbits-1), 2^(nbits-1))`.
/// Outside that range the offset overflows and the result is silently wrong;
/// callers use `nbits = bit_width + 1` so that Baru's 32-bit convention has a
/// bit of headroom.
///
/// Returns the bits and the number of communication rounds consumed — which is
/// independent of how many values were passed, since every step is batched.
pub fn bit_decompose_batch<E: ShareEngine>(
    engine: &mut E,
    xs: &[E::Share],
    nbits: u32,
) -> Result<Decomposed<E>, String> {
    if xs.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let m = engine.modulus();
    let width = nbits + KAPPA;
    if width + 1 > m.order_bits() {
        return Err(format!(
            "mpcc vm: {}-bit decomposition needs {} bits of headroom, but the              protocol computes in a {}-bit domain",
            nbits,
            width + 1,
            m.order_bits()
        ));
    }
    let n = xs.len();
    let mut rounds = 0u64;

    // 1. Masks.  Only the low `nbits` of each mask take part in the
    //    subtraction; the rest exist purely to hide the opened value.
    let bits = engine.random_bits(n * width as usize);
    let mut masked = Vec::with_capacity(n);
    let mut low_bits: Vec<Vec<E::Share>> = Vec::with_capacity(n);
    for (i, x) in xs.iter().enumerate() {
        let rb = &bits[i * width as usize..(i + 1) * width as usize];
        let mut r = engine.mul_const(&rb[0], 1);
        for (j, b) in rb.iter().enumerate().skip(1) {
            let term = engine.mul_const(b, 1u64 << j);
            r = engine.add(&r, &term);
        }
        let offset = engine.add_const(x, 1u64 << (nbits - 1));
        masked.push(engine.add(&offset, &r));
        low_bits.push(rb[..nbits as usize].to_vec());
    }

    // 2. One opening for the whole batch.
    let opened = engine.open_batch(&masked);
    rounds += 1;

    // 3. Generate/propagate — local, because the minuend is public.
    let nb = nbits as usize;
    let mut gen: Vec<Vec<E::Share>> = Vec::with_capacity(n);
    let mut prop: Vec<Vec<E::Share>> = Vec::with_capacity(n);
    for (i, c) in opened.iter().enumerate() {
        let mut g = Vec::with_capacity(nb);
        let mut p = Vec::with_capacity(nb);
        for (j, rj) in low_bits[i].iter().enumerate() {
            if (c >> j) & 1 == 0 {
                g.push(rj.clone());
                let one = engine.constant(1);
                p.push(engine.sub(&one, rj));
            } else {
                g.push(engine.constant(0));
                p.push(rj.clone());
            }
        }
        gen.push(g);
        prop.push(p);
    }

    // 4. Kogge–Stone: one batched round per level.
    let mut d = 1usize;
    while d < nb {
        let mut pairs = Vec::with_capacity(n * (nb - d) * 2);
        for i in 0..n {
            for j in d..nb {
                pairs.push((prop[i][j].clone(), gen[i][j - d].clone()));
                pairs.push((prop[i][j].clone(), prop[i][j - d].clone()));
            }
        }
        let products = engine.mul_batch(&pairs);
        rounds += 1;
        let mut k = 0;
        let mut new_gen = gen.clone();
        let mut new_prop = prop.clone();
        for i in 0..n {
            for j in d..nb {
                // `gᵢ · pᵢ = 0`, so the OR in the prefix combine is an add.
                new_gen[i][j] = engine.add(&gen[i][j], &products[k]);
                new_prop[i][j] = products[k + 1].clone();
                k += 2;
            }
        }
        gen = new_gen;
        prop = new_prop;
        d *= 2;
    }

    // 5. uⱼ = cⱼ ⊕ rⱼ ⊕ borrowⱼ₋₁, one batched round for all the XORs.
    let mut xor_pairs = Vec::new();
    let mut t_vals: Vec<Vec<E::Share>> = Vec::with_capacity(n);
    for (i, c) in opened.iter().enumerate() {
        let mut ts = Vec::with_capacity(nb);
        for (j, rj) in low_bits[i].iter().enumerate() {
            ts.push(if (c >> j) & 1 == 0 {
                rj.clone()
            } else {
                let one = engine.constant(1);
                engine.sub(&one, rj)
            });
        }
        for j in 1..nb {
            xor_pairs.push((ts[j].clone(), gen[i][j - 1].clone()));
        }
        t_vals.push(ts);
    }
    let xor_products = if xor_pairs.is_empty() {
        Vec::new()
    } else {
        let p = engine.mul_batch(&xor_pairs);
        rounds += 1;
        p
    };

    let mut out = Vec::with_capacity(n);
    let mut k = 0;
    for i in 0..n {
        let mut ubits = Vec::with_capacity(nb);
        ubits.push(t_vals[i][0].clone()); // no borrow into bit 0
        for j in 1..nb {
            // t ⊕ b = t + b − 2·t·b
            let sum = engine.add(&t_vals[i][j], &gen[i][j - 1]);
            let twice = engine.add(&xor_products[k], &xor_products[k]);
            ubits.push(engine.sub(&sum, &twice));
            k += 1;
        }
        // Undo the offset: adding 2^(nbits-1) mod 2^nbits flips the top bit.
        let one = engine.constant(1);
        let top = engine.sub(&one, &ubits[nb - 1]);
        ubits[nb - 1] = top;
        out.push(ubits);
    }
    Ok((out, rounds))
}

/// Fold `bits` with an AND tree, returning `1` iff every bit is 1.
/// Log-depth, one batched round per level.
fn all_ones<E: ShareEngine>(engine: &mut E, groups: &[Vec<E::Share>]) -> (Vec<E::Share>, u64) {
    let mut layers: Vec<Vec<E::Share>> = groups.to_vec();
    let mut rounds = 0u64;
    loop {
        let widest = layers.iter().map(|l| l.len()).max().unwrap_or(0);
        if widest <= 1 {
            break;
        }
        let mut pairs = Vec::new();
        for l in &layers {
            for c in l.chunks(2) {
                if c.len() == 2 {
                    pairs.push((c[0].clone(), c[1].clone()));
                }
            }
        }
        let products = engine.mul_batch(&pairs);
        rounds += 1;
        let mut k = 0;
        for l in layers.iter_mut() {
            let mut next = Vec::with_capacity(l.len().div_ceil(2));
            for c in l.chunks(2) {
                if c.len() == 2 {
                    next.push(products[k].clone());
                    k += 1;
                } else {
                    next.push(c[0].clone());
                }
            }
            *l = next;
        }
    }
    (layers.into_iter().map(|l| l[0].clone()).collect(), rounds)
}

// ─── Execution ────────────────────────────────────────────────────────────────

/// What a run actually cost, alongside what it computed.
#[derive(Debug, Clone, Serialize)]
pub struct Measured {
    pub protocol: String,
    pub parties: usize,
    /// Communication rounds, counted as levels of the schedule that contained
    /// at least one interactive instruction (plus one for input distribution).
    pub rounds: u64,
    pub online_bytes: u64,
    pub muls: u64,
    pub opens: u64,
    pub triples: u64,
    pub random_bits: u64,
    /// Preprocessing bytes actually moved — non-zero only for engines that
    /// generate their own correlated randomness (Semi2k's OT).
    pub offline_bytes: u64,
    /// Triples generated (batched), the denominator offline cost amortizes over.
    pub triples_generated: u64,
    /// Bytes and rounds spent *generating* the random bits, which is
    /// query-independent preprocessing and so is reported apart from
    /// [`Measured::online_bytes`] and [`Measured::rounds`].  Zero on
    /// `DealerEngine`, by definition — see [`Counters::random_bit_bytes`].
    pub random_bit_bytes: u64,
    pub random_bit_rounds: u64,
    /// Multiplications spent generating random bits — kept out of
    /// [`Measured::muls`] so that stays comparable to `mpcc::cost`'s prediction.
    pub random_bit_muls: u64,
    /// Wall clock, microseconds.  In-process, so this measures local compute
    /// only — there is no network to wait for.
    pub micros: u128,
    /// `(variable, value)` for each program output.
    pub outputs: Vec<(String, i64)>,
}

/// Run `prog` under `engine`.
///
/// Executes level by level: all local instructions ready at a level, then that
/// level's interactive instructions as a single batch.  That is what makes the
/// measured round count equal the schedule's depth rather than the instruction
/// count — and therefore what makes it comparable to the prediction.
pub fn execute<E: ShareEngine>(
    prog: &Program,
    sched: &Schedule,
    engine: &mut E,
    protocol: &str,
    inputs: &HashMap<String, i64>,
) -> Result<Measured, String> {
    let t0 = std::time::Instant::now();
    let m = engine.modulus();
    let mut shares: HashMap<Reg, E::Share> = HashMap::new();
    let mut clears: HashMap<Reg, u64> = HashMap::new();
    let mut rounds: u64 = 0;

    // ── Input distribution: one round for all of them ─────────────────────
    //
    // Which party owns an input register is found by **one pass** over the code,
    // not by a scan per input. It used to be `prog.code.iter().find(..)` inside
    // this loop, which is quadratic in exactly the dimension a private network
    // grows in: one input per weight. Measured on `784x256x10` — 203 000 inputs
    // against 611 694 instructions — that nested scan was ~10^11 operations and
    // dominated the whole run.
    let mut owner: HashMap<Reg, usize> = HashMap::with_capacity(prog.inputs.len());
    for instr in &prog.code {
        if let Op::Input { party } = instr.op {
            if let Some(reg) = instr.dst.first() {
                owner.entry(*reg).or_insert(party);
            }
        }
    }
    for (name, reg) in &prog.inputs {
        let v = *inputs
            .get(name)
            .ok_or_else(|| format!("mpcc vm: no value supplied for input `{}`", name))?;
        let party = owner.get(reg).copied().unwrap_or(0);
        shares.insert(*reg, engine.input(m.from_i64(v), party));
    }
    if !prog.inputs.is_empty() {
        rounds += 1;
    }

    // Instructions grouped by the level at which they are ready.
    let depth = sched.rounds.len();
    let mut locals_at: Vec<Vec<usize>> = vec![Vec::new(); depth + 2];
    for (idx, instr) in prog.code.iter().enumerate() {
        if !prog.is_interactive(instr) {
            let lvl = sched.level[idx].min(depth + 1);
            locals_at[lvl].push(idx);
        }
    }

    // Levels 0..=depth interleave with the rounds; anything beyond consumes the
    // last round's results (the final `Output`).
    let tail = locals_at.split_off(depth + 1);
    for (level, locals) in locals_at.iter().enumerate() {
        for &idx in locals {
            exec_local(prog, idx, engine, &mut shares, &mut clears)?;
        }
        if level < depth {
            rounds += exec_round(
                prog,
                &sched.rounds[level].instrs,
                engine,
                &mut shares,
                &mut clears,
            )?;
        }
    }
    for idx in tail.iter().flatten() {
        exec_local(prog, *idx, engine, &mut shares, &mut clears)?;
    }

    let outputs = prog
        .outputs
        .iter()
        .map(|(name, reg)| {
            let v = clears
                .get(reg)
                .ok_or_else(|| format!("mpcc vm: output `{}` was never opened", name))?;
            Ok((name.clone(), m.to_i64(*v, prog.bit_width)))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let c = engine.counters();
    Ok(Measured {
        protocol: protocol.to_string(),
        parties: engine.parties(),
        rounds,
        online_bytes: c.online_bytes,
        muls: c.muls,
        opens: c.opens,
        triples: c.triples,
        random_bits: c.random_bits,
        offline_bytes: c.offline_bytes,
        triples_generated: c.triples_generated,
        random_bit_bytes: c.random_bit_bytes,
        random_bit_rounds: c.random_bit_rounds,
        random_bit_muls: c.random_bit_muls,
        micros: t0.elapsed().as_micros(),
        outputs,
    })
}

/// Fetch an operand as a sharing, promoting a clear register if needed.
fn as_share<E: ShareEngine>(
    engine: &mut E,
    shares: &HashMap<Reg, E::Share>,
    clears: &HashMap<Reg, u64>,
    r: Reg,
) -> Result<E::Share, String> {
    if let Some(s) = shares.get(&r) {
        return Ok(s.clone());
    }
    match clears.get(&r) {
        Some(v) => Ok(engine.constant(*v)),
        None => Err(format!(
            "mpcc vm: register r{} read before it was written",
            r
        )),
    }
}

fn exec_local<E: ShareEngine>(
    prog: &Program,
    idx: usize,
    engine: &mut E,
    shares: &mut HashMap<Reg, E::Share>,
    clears: &mut HashMap<Reg, u64>,
) -> Result<(), String> {
    let instr = &prog.code[idx];
    let m = engine.modulus();
    // Every operand clear ⇒ the whole instruction is public computation.
    let all_clear = instr.src.iter().all(|r| clears.contains_key(r));
    let dst = instr.dst.first().copied();

    macro_rules! put_clear {
        ($v:expr) => {{
            if let Some(d) = dst {
                clears.insert(d, $v);
            }
        }};
    }
    macro_rules! put_share {
        ($v:expr) => {{
            if let Some(d) = dst {
                shares.insert(d, $v);
            }
        }};
    }

    match &instr.op {
        Op::LdI(v) => put_clear!(m.from_i64(*v)),
        // Inputs were distributed in the prelude.
        Op::Input { .. } => {}
        Op::Output => {}
        Op::Copy | Op::Add | Op::Sub | Op::Neg | Op::MulC(_) | Op::AddC(_) | Op::ShlC(_)
            if all_clear =>
        {
            let a = clears[&instr.src[0]];
            let v = match &instr.op {
                Op::Copy => a,
                Op::Add => m.add(a, clears[&instr.src[1]]),
                Op::Sub => m.sub(a, clears[&instr.src[1]]),
                Op::Neg => m.neg(a),
                Op::MulC(c) => m.mul(a, m.from_i64(*c)),
                Op::AddC(c) => m.add(a, m.from_i64(*c)),
                Op::ShlC(n) => m.mul(a, m.from_i64(1i64 << n.min(&62))),
                _ => unreachable!(),
            };
            put_clear!(v);
        }
        Op::Copy => {
            let a = as_share(engine, shares, clears, instr.src[0])?;
            put_share!(a);
        }
        Op::Add => {
            let a = as_share(engine, shares, clears, instr.src[0])?;
            let b = as_share(engine, shares, clears, instr.src[1])?;
            put_share!(engine.add(&a, &b));
        }
        Op::Sub => {
            let a = as_share(engine, shares, clears, instr.src[0])?;
            let b = as_share(engine, shares, clears, instr.src[1])?;
            put_share!(engine.sub(&a, &b));
        }
        Op::Neg => {
            let a = as_share(engine, shares, clears, instr.src[0])?;
            let z = engine.constant(0);
            put_share!(engine.sub(&z, &a));
        }
        Op::MulC(c) => {
            let a = as_share(engine, shares, clears, instr.src[0])?;
            put_share!(engine.mul_const(&a, m.from_i64(*c)));
        }
        Op::AddC(c) => {
            let a = as_share(engine, shares, clears, instr.src[0])?;
            put_share!(engine.add_const(&a, m.from_i64(*c)));
        }
        Op::ShlC(n) => {
            let a = as_share(engine, shares, clears, instr.src[0])?;
            put_share!(engine.mul_const(&a, m.from_i64(1i64 << n.min(&62))));
        }
        // `Mul`/`Mux` reach here only when an operand is clear, which makes them
        // local scalar operations.
        Op::Mul => {
            let (x, y) = (instr.src[0], instr.src[1]);
            let (secret, konst) = if let Some(c) = clears.get(&y) {
                (x, *c)
            } else {
                (y, clears[&x])
            };
            let a = as_share(engine, shares, clears, secret)?;
            put_share!(engine.mul_const(&a, konst));
        }
        Op::Mux => {
            // Clear condition: select at compile time.
            let cond = clears[&instr.src[0]];
            let chosen = if cond != 0 {
                instr.src[1]
            } else {
                instr.src[2]
            };
            if let Some(v) = clears.get(&chosen) {
                put_clear!(*v);
            } else {
                let s = as_share(engine, shares, clears, chosen)?;
                put_share!(s);
            }
        }
        other => {
            return Err(unsupported(other));
        }
    }
    Ok(())
}

/// What a comparison-shaped instruction wants from a bit decomposition.
enum BitUse {
    /// The sign bit, i.e. the result of a `<`.
    Sign(Reg),
    /// `1` iff every bit is zero.
    Zero(Reg),
    /// Recompose bits `shift..bit_width` — a logical right shift.
    Truncate { dst: Reg, shift: u32 },
}

/// Execute one dependence level: every interactive instruction in it, batched
/// by kind.
///
/// Independent groups (multiplications, openings, bit decompositions) would run
/// concurrently in a real deployment, so the level costs the **maximum** of the
/// groups' round counts, not their sum.  Composite instructions are batched
/// with each other for the same reason: ten comparisons at the same dependence
/// level cost one comparison's worth of rounds.
///
/// Returns the rounds consumed.
fn exec_round<E: ShareEngine>(
    prog: &Program,
    idxs: &[usize],
    engine: &mut E,
    shares: &mut HashMap<Reg, E::Share>,
    clears: &mut HashMap<Reg, u64>,
) -> Result<u64, String> {
    let mut mul_pairs: Vec<(E::Share, E::Share)> = Vec::new();
    let mut mul_dst: Vec<(Reg, Option<E::Share>)> = Vec::new();
    let mut open_vals: Vec<E::Share> = Vec::new();
    let mut open_dst: Vec<Reg> = Vec::new();
    let mut decomp: Vec<E::Share> = Vec::new();
    let mut decomp_use: Vec<BitUse> = Vec::new();

    for &idx in idxs {
        let instr = &prog.code[idx];
        match &instr.op {
            Op::Mul => {
                let a = as_share(engine, shares, clears, instr.src[0])?;
                let b = as_share(engine, shares, clears, instr.src[1])?;
                mul_pairs.push((a, b));
                mul_dst.push((instr.dst[0], None));
            }
            // `t + c*(e - t)`: the subtraction and the addition are local, so
            // the multiplication joins this level's batch like any other.
            Op::Mux => {
                let c = as_share(engine, shares, clears, instr.src[0])?;
                let t = as_share(engine, shares, clears, instr.src[1])?;
                let e = as_share(engine, shares, clears, instr.src[2])?;
                let diff = engine.sub(&e, &t);
                mul_pairs.push((c, diff));
                mul_dst.push((instr.dst[0], Some(t)));
            }
            Op::Open => {
                let a = as_share(engine, shares, clears, instr.src[0])?;
                open_vals.push(a);
                open_dst.push(instr.dst[0]);
            }
            // Every comparison-shaped instruction is one bit decomposition
            // plus local arithmetic, so they all join one batch.
            Op::LtS | Op::LtU => {
                let a = as_share(engine, shares, clears, instr.src[0])?;
                let b = as_share(engine, shares, clears, instr.src[1])?;
                decomp.push(engine.sub(&a, &b));
                decomp_use.push(BitUse::Sign(instr.dst[0]));
            }
            Op::EqZ => {
                let a = as_share(engine, shares, clears, instr.src[0])?;
                decomp.push(a);
                decomp_use.push(BitUse::Zero(instr.dst[0]));
            }
            Op::TruncPr { shift } => {
                let a = as_share(engine, shares, clears, instr.src[0])?;
                decomp.push(a);
                decomp_use.push(BitUse::Truncate {
                    dst: instr.dst[0],
                    shift: *shift,
                });
            }
            other => return Err(unsupported(other)),
        }
    }

    if !mul_pairs.is_empty() {
        let out = engine.mul_batch(&mul_pairs);
        for (v, (dst, base)) in out.into_iter().zip(mul_dst) {
            let v = match base {
                Some(t) => engine.add(&t, &v),
                None => v,
            };
            shares.insert(dst, v);
        }
    }
    if !open_vals.is_empty() {
        let out = engine.open_batch(&open_vals);
        for (v, dst) in out.into_iter().zip(open_dst) {
            clears.insert(dst, v);
        }
    }

    let mut composite_rounds = 0u64;
    if !decomp.is_empty() {
        // One extra bit of width so a difference of two `bit_width` values
        // still has a meaningful sign bit.
        let nbits = prog.bit_width + 1;
        let (bits, r) = bit_decompose_batch(engine, &decomp, nbits)?;
        composite_rounds += r;

        // Equality needs an AND tree over the complements; gather them so the
        // tree is one batch across every `EqZ` at this level.
        let mut zero_groups: Vec<Vec<E::Share>> = Vec::new();
        let mut zero_dst: Vec<Reg> = Vec::new();
        for (b, u) in bits.iter().zip(&decomp_use) {
            match u {
                BitUse::Sign(dst) => {
                    shares.insert(*dst, b[nbits as usize - 1].clone());
                }
                BitUse::Truncate { dst, shift } => {
                    // **Arithmetic** right shift: recompose the bits above
                    // `shift`, then restore the sign.  Free once the bits exist —
                    // the mirror image of a garbled circuit, where the shift is
                    // free and the bits are what you already have.
                    //
                    // The sign term is what this used to be missing, and it was
                    // wrong on every negative input. `b` holds `nbits = bit_width
                    // + 1` two's-complement bits, so
                    //
                    //     x = −2^bit_width·b[bit_width] + Σ_{j<bit_width} 2^j·b[j]
                    //
                    // and `floor(x / 2^s)` — which is what an arithmetic shift
                    // means, and what Rust's `>>` and the interpreter's
                    // `share_ashr16` both compute — is
                    //
                    //     −2^(bit_width−s)·b[bit_width] + Σ_{s≤j<bit_width} 2^(j−s)·b[j].
                    //
                    // Recomposing only the sum left the sign bit contributing
                    // `+2^(bit_width−s)` instead of `−2^(bit_width−s)`, so `−1 >> 3`
                    // came out as `2^29 − 1`. Positive values were unaffected,
                    // which is why the one existing test (`x = 1000`) passed.
                    let one = engine.constant(1);
                    let mut acc = engine.sub(&one, &one); // a shared zero
                    let lo = *shift as usize;
                    for (j, bit) in b.iter().enumerate().take(prog.bit_width as usize).skip(lo) {
                        let term = engine.mul_const(bit, 1u64 << (j - lo));
                        acc = engine.add(&acc, &term);
                    }
                    let sign = &b[prog.bit_width as usize];
                    let sign_weight = 1u64 << prog.bit_width.saturating_sub(*shift);
                    let sign_term = engine.mul_const(sign, sign_weight);
                    acc = engine.sub(&acc, &sign_term);
                    shares.insert(*dst, acc);
                }
                BitUse::Zero(dst) => {
                    let one = engine.constant(1);
                    zero_groups.push(b.iter().map(|bit| engine.sub(&one, bit)).collect());
                    zero_dst.push(*dst);
                }
            }
        }
        if !zero_groups.is_empty() {
            let (results, r) = all_ones(engine, &zero_groups);
            composite_rounds += r;
            for (v, dst) in results.into_iter().zip(zero_dst) {
                shares.insert(dst, v);
            }
        }
    }

    // Independent groups run concurrently.
    let simple_rounds = u64::from(!mul_pairs.is_empty() || !open_vals.is_empty());
    Ok(simple_rounds.max(composite_rounds))
}

/// Refuse an instruction by name, with the reason *that* instruction is refused.
///
/// This used to name secret division for every unsupported op, which was
/// misleading for the six that have nothing to do with division: the reason a
/// `Xor` cannot run is that no binary-domain engine exists, and no amount of
/// long-division circuitry would change that.
fn unsupported(op: &Op) -> String {
    let why = match op {
        Op::Div => {
            "secret division needs a restoring long-division circuit on top of the \
             comparison machinery — roadmap phase P3 (docs/mpc-compiler-roadmap.md)"
        }
        Op::Xor
        | Op::Not
        | Op::And
        | Op::BitDecomp { .. }
        | Op::BitComp { .. }
        | Op::PopCount { .. } => {
            "this is a binary-domain instruction and no binary protocol implements \
             `ShareEngine` — every binary row in `mpcc::protocols` is `Planned`.  \
             Comparisons on an arithmetic engine do not go through it: they use \
             `bit_decompose_batch`, which needs only `random_bits` and `mul_batch`"
        }
        Op::Convert => {
            "cross-domain conversion needs two engines live at once and real daBits; \
             `mpcc::cost` prices it (`assign` charges crossings as `Convert` on the \
             arithmetic protocol) and `mpcc::mixed` executes MPC<->FHE crossings, but \
             no arithmetic<->binary conversion is executed anywhere"
        }
        _ => "no execution arm is implemented for it",
    };
    format!(
        "mpcc vm: `{}` is not executable — {why}.  Costing it works today \
         (`baru mpcc compare`); only running it does not.",
        op.mnemonic()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_arithmetic_wraps_and_round_trips_signed_values() {
        let m = Modulus::Ring { k: 64 };
        assert_eq!(m.to_i64(m.from_i64(-7), 32), -7);
        assert_eq!(m.to_i64(m.from_i64(i32::MIN as i64), 32), i32::MIN as i64);
        assert_eq!(m.mul(m.from_i64(-3), m.from_i64(5)), m.from_i64(-15));
        assert_eq!(m.sub(m.from_i64(3), m.from_i64(5)), m.from_i64(-2));
    }

    #[test]
    fn field_arithmetic_round_trips_and_inverts() {
        let m = Modulus::MERSENNE61;
        assert_eq!(m.to_i64(m.from_i64(-7), 32), -7);
        assert_eq!(m.mul(m.from_i64(-3), m.from_i64(5)), m.from_i64(-15));
        for a in [1u64, 2, 12345, (1u64 << 60) + 1] {
            assert_eq!(m.mul(a, m.inv(a).unwrap()), 1, "inverse of {}", a);
        }
        assert!(m.inv(0).is_err());
        assert!(Modulus::Ring { k: 64 }.inv(3).is_err());
    }

    #[test]
    fn element_sizes_match_what_the_cost_model_charges() {
        assert_eq!(Modulus::Ring { k: 64 }.element_bytes(), 8);
        assert_eq!(Modulus::Ring { k: 32 }.element_bytes(), 4);
        assert_eq!(Modulus::MERSENNE61.element_bytes(), 8);
    }

    /// The bits must actually be bits, and the XOR of one contribution per party
    /// must be uniform — which for a *sampled* value can only be checked as a
    /// sanity bound, so this asserts both values occur over a large batch rather
    /// than asserting a distribution.
    #[test]
    fn generated_random_bits_are_bits_and_take_both_values() {
        let mut e = crate::mpcc::semi2k::Semi2kEngine::new(2, 64).expect("two-party ring engine");
        let bits = random_bits_batch(&mut e, 512);
        assert_eq!(bits.len(), 512);
        let opened = e.open_batch(&bits);
        assert!(
            opened.iter().all(|&b| b == 0 || b == 1),
            "every generated value must open to 0 or 1"
        );
        let ones = opened.iter().filter(|&&b| b == 1).count();
        assert!(
            ones > 100 && ones < 412,
            "512 uniform bits should not be this lopsided: {ones} ones"
        );
    }

    /// **The cost of a random bit is a Beaver triple.**
    ///
    /// This is the measurement the offline term in `mpcc::cost` is fitted to, and
    /// the reason that term changed: it used to charge one ring element (8 bytes)
    /// per bit, against a real cost of one triple — ~3.1 kB of OT traffic at
    /// 64-bit shares, some **400x** more. Pinning the structure here is what stops
    /// the model drifting back.
    ///
    /// The byte figure decomposes exactly, which is the check that the counter is
    /// measuring the protocol and not something incidental. At `p` parties a batch
    /// of `n` bits sends, per party: `n * p` input shares (one per party per bit)
    /// plus `2 * n` elements for the Beaver openings at each of the
    /// `ceil(log2 p)` tree levels. At `p = 2` that is `n * (2 + 2) = 4n`
    /// elements.
    #[test]
    fn the_cost_of_a_random_bit_is_a_triple() {
        let n = 64usize;
        let mut e = crate::mpcc::semi2k::Semi2kEngine::new(2, 64).expect("two-party ring engine");
        let before = e.counters().triples;
        let _ = random_bits_batch(&mut e, n);
        let c = e.counters();

        assert_eq!(
            c.triples - before,
            n as u64,
            "one triple per bit at two parties (one XOR tree level)"
        );
        assert_eq!(c.random_bits, n as u64);
        assert_eq!(
            c.random_bit_muls, n as u64,
            "one multiplication per bit per tree level"
        );
        assert_eq!(
            c.random_bit_rounds, 1,
            "the whole batch is one round, however many bits it carries"
        );
        assert_eq!(
            c.random_bit_bytes,
            (4 * n as u64) * Modulus::Ring { k: 64 }.element_bytes(),
            "2 input shares + 2 Beaver elements per bit, at 8 bytes an element"
        );

        // And none of it is charged to the online phase, because random bits are
        // query-independent — a deployment generates them before the query
        // arrives. Charging them online and then comparing against an FHE total
        // is the error survey §7.3 records making three times.
        assert_eq!(c.online_bytes, 0, "generation is preprocessing, not online");
        assert_eq!(c.muls, 0, "and its multiplications are not the program's");
    }

    /// The tree is balanced, so party count enters logarithmically: three parties
    /// cost two rounds and two multiplications per bit, not three.
    #[test]
    fn more_parties_cost_log_rounds_not_linear() {
        let mut e = crate::mpcc::sim::ShamirEngine::new(3).expect("three-party Shamir");
        let _ = random_bits_batch(&mut e, 16);
        let c = e.counters();
        assert_eq!(c.random_bit_rounds, 2, "ceil(log2 3) = 2 levels");
        assert_eq!(c.random_bit_muls, 32, "16 bits x 2 levels");
    }
}
