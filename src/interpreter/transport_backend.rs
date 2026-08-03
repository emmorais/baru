//! TransportBackend: distributed MPC backend using Transport for communication.
//!
//! This module implements the Backend trait for a real distributed MPC setting
//! where parties communicate with each other via a Transport abstraction. Each
//! operation (mul, a2b, reveal, …) runs the distributed protocol from
//! `crypto::aby3_protocol`.

use std::collections::HashMap;

use crate::crypto::aby3_protocol::PartyShare;
use crate::crypto::net::{Mailbox, PartyId, Transport};
use crate::crypto::types::Fr;

use super::backend::{Backend, ShareRef};

/// A `Backend` implementation that communicates with peer parties via a
/// `Transport`.  Each operation (mul, a2b, reveal, …) runs the distributed
/// protocol from `crypto::protocol`.
///
/// Share data is stored in a local `HashMap<share_id, PartyShare>`.  The
/// `ShareRef` handed back to the evaluator still carries a dummy
/// `local: None`; the actual per-party components live in `shares`.
pub struct TransportBackend<T: Transport> {
    me: PartyId,
    /// Buffering demultiplexer over the transport: receive helpers wait for a
    /// specific message while out-of-order arrivals are retained, not dropped.
    mailbox: Mailbox<T>,
    /// Map from share id to this party's two local components.
    shares: HashMap<u64, PartyShare>,
    /// Per-instance monotonic counter — all parties executing the same
    /// expression in lockstep will generate the same sequence of IDs.
    next_id: u64,
    /// Route batched comparisons through the **Yao leg** ([`Self::lt_gc_batch`])
    /// instead of the log-round bit decomposition. Off by default so the a2b
    /// path stays the control in every measurement; when on, every gadget that
    /// reduces to `lt_batch` — max-pooling, argmax, `max2_batch` — becomes
    /// constant-round without being rewritten. All parties must agree on the
    /// setting: it changes which protocol runs, hence the id-allocation
    /// sequence the parties execute in lockstep.
    nonlin: NonlinMode,
    /// Batched interactive protocol invocations — every point at which this
    /// party stopped computing and talked to its peers.
    ///
    /// Exactly comparable between any two routes that share a backend, which
    /// is what makes "hand-written versus compiled" a fair question: both go
    /// through the same primitives, so this counts *how often each route had
    /// to synchronise*, independently of how it was written.
    interactive_calls: std::sync::atomic::AtomicUsize,
}

/// Id space one `lt_falcon_batch` reserves: the inner `Π_WA`, its `Π_PC` tree
/// and the lifting select each need disjoint operation ids.
const FALCON_ID_BLOCK: u64 = 4096;

/// Which protocol backs every comparison — and therefore ReLU, max-pool and
/// argmax, all of which reduce to `lt_batch`.
///
/// The three are interchangeable *functionally* and very different in cost, and
/// which one wins depends on the deployment rather than on the workload:
///
/// | mode | rounds | bytes | wins when |
/// |---|---|---|---|
/// | `A2b` | `~log₂ nbits` | high | never, on current measurements |
/// | `Gc` | 1 | ~30× `Falcon` | latency dominates |
/// | `Falcon` | `5 + log₂ ℓ` | lowest | bandwidth dominates |
///
/// Falcon here means the `wrap3` sign extraction of Wagh et al. (PETS 2021),
/// reproduced in [`crate::crypto::falcon_mpc`]. It shares this backend's
/// replicated sharing and linear layer unchanged — the *only* thing it replaces
/// is the comparison, which is also the only place the two systems ever
/// differed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NonlinMode {
    /// Batched bit decomposition.
    #[default]
    A2b,
    /// Three-party garbled circuits, constant round.
    Gc,
    /// Falcon's `wrap3` DReLU over `Z_p`.
    Falcon,
}

impl NonlinMode {
    pub fn label(self) -> &'static str {
        match self {
            NonlinMode::A2b => "a2b (batched bit decomposition)",
            NonlinMode::Gc => "gc (3-party garbled, constant-round)",
            NonlinMode::Falcon => "falcon (wrap3 over Z_p)",
        }
    }

    /// Parse a CLI spelling.
    pub fn parse(s: &str) -> Option<NonlinMode> {
        match s {
            "a2b" => Some(NonlinMode::A2b),
            "gc" => Some(NonlinMode::Gc),
            "falcon" => Some(NonlinMode::Falcon),
            _ => None,
        }
    }
}

impl<T: Transport> TransportBackend<T> {
    pub fn new(me: PartyId, transport: T) -> Self {
        TransportBackend {
            me,
            mailbox: Mailbox::new(transport),
            shares: HashMap::new(),
            next_id: 1,
            nonlin: NonlinMode::A2b,
            interactive_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Batched interactive protocol invocations so far — see the field docs.
    pub fn interactive_calls(&self) -> usize {
        self.interactive_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn note_interactive(&self) {
        self.interactive_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Enable/disable the garbled-circuit route for batched comparisons.
    /// Retained as the historical two-way switch; [`Self::set_nonlin`] is the
    /// general form.
    pub fn set_gc_nonlinearities(&mut self, on: bool) {
        self.nonlin = if on { NonlinMode::Gc } else { NonlinMode::A2b };
    }

    /// Whether batched comparisons currently take the Yao leg.
    pub fn gc_nonlinearities(&self) -> bool {
        self.nonlin == NonlinMode::Gc
    }

    /// Choose which comparison protocol backs every `lt`-derived operation.
    ///
    /// Must be set identically on every party, before any operation that would
    /// diverge: the modes execute different id-allocation sequences.
    pub fn set_nonlin(&mut self, mode: NonlinMode) {
        self.nonlin = mode;
    }

    pub fn nonlin(&self) -> NonlinMode {
        self.nonlin
    }

    /// Falcon's `wrap3` DReLU as a drop-in for `lt_batch`.
    ///
    /// `lt(a,b)` is `MSB(a−b)`, i.e. the complement of `DReLU(a−b)`, so this is
    /// one `drelu_batch` plus a local inversion.
    ///
    /// # Two things this has to reconcile
    ///
    /// **Output encoding.** `drelu_batch` yields a `Z_2` bit, while every
    /// consumer of `lt_batch` — `max2_batch`, ReLU, argmax — expects an
    /// *arithmetic* share it can multiply by a wide operand. Lifting the bit
    /// into the ring is exactly `Π_SS`, so that select is a genuine cost of
    /// making Falcon interchangeable here, not an artefact.
    ///
    /// **Preprocessing.** `Π_WA` needs a random mask `x` with its bits shared
    /// over `Z_p` and its own `wrap3` — and `δ = wrap3(x₁,x₂,x₃)` is not
    /// derivable by any party, since each is missing a summand. All of it is now
    /// generated by a real interactive offline phase
    /// ([`random_wrap_masks`](crate::crypto::falcon_mpc::random_wrap_masks)),
    /// by bit composition as in Falcon's Fig. 6: the mask is *built from* shared
    /// bits rather than sampled and decomposed, so its bits come for free and its
    /// wrap falls out of one carry chain. No party learns the mask.
    ///
    /// This closes the last dealt piece. Until it did, the triple came from a
    /// seed every party could evaluate, so all three could reconstruct the mask
    /// — the online costs were real but a `NonlinMode::Falcon` run was not
    /// private. The cost of closing it is one `a2b` per element, offline; see
    /// `random_wrap_masks` for why that does not undo the online win.
    fn lt_falcon_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        use crate::crypto::falcon_mpc::{drelu_batch, select_batch};
        let nbits = pairs
            .iter()
            .map(|(a, b)| a.nbits.max(b.nbits))
            .max()
            .unwrap() as u32;
        let mask: Fr = if nbits >= 128 {
            Fr::MAX
        } else {
            (1u128 << nbits) - 1
        };
        let n = pairs.len();

        let diffs: Vec<PartyShare> = pairs
            .iter()
            .map(|(a, b)| {
                let (pa, pb) = (self.get_cloned(a), self.get_cloned(b));
                PartyShare {
                    left: pa.left.wrapping_sub(pb.left) & mask,
                    right: pa.right.wrapping_sub(pb.right) & mask,
                }
            })
            .collect();

        // ── Offline phase ───────────────────────────────────────────────────
        // Run here rather than in a separate pass because this backend has no
        // offline/online split to hang it on; the *costs* are still attributed
        // to preprocessing in `docs/falcon-reproduction.md`, which is what
        // Falcon does too.
        //
        // Every id range below is reserved from the protocol's own span
        // function. The previous fixed offsets (`base + 1_000`, `+ 2_000`,
        // `+ 3_000`) overlapped once `n` passed ~333, because the `Z₂` draw
        // consumes one PRF counter *per bit*; a CNN ReLU layer is well past
        // that, so two offline protocols were deriving randomness from the
        // same counters.
        use crate::crypto::falcon_mpc::{
            dabit_id_span, nonzero_mask_id_span, random_dabits_fp, random_dabits_ring,
            random_nonzero_masks, random_wrap_masks, wrap_mask_id_span,
        };
        let offline_span = wrap_mask_id_span(n, nbits)
            + dabit_id_span(n) // β
            + nonzero_mask_id_span(n) // Π_PC masks
            + dabit_id_span(n); // Π_SS selector
        let base = self.alloc_block(FALCON_ID_BLOCK + offline_span);
        let mut off = base + FALCON_ID_BLOCK; // online ids live in the first block

        // `Π_WA`'s masks: a uniform `x`, its bits over `Z_p`, and its own wrap
        // `δ` — generated by bit composition, so no party learns `x`.
        let wm = random_wrap_masks(n, nbits, off, self.me, &self.mailbox);
        off += wrap_mask_id_span(n, nbits);
        let (beta_z2, beta_fp) = random_dabits_fp(n, off, self.me, &self.mailbox);
        off += dabit_id_span(n);
        let masks = random_nonzero_masks(n, off, self.me, &self.mailbox);
        off += nonzero_mask_id_span(n);
        // `Π_SS`'s selector is a daBit over `Z₂` and the ring — fully interactive.
        let (sel_z2, sel_ring) =
            random_dabits_ring(n, nbits as usize, off, self.me, &self.mailbox);

        let wp: Vec<_> = wm
            .into_iter()
            .enumerate()
            .map(|(i, (x, bits, delta))| crate::crypto::falcon_mpc::WrapPreproc {
                x,
                bits,
                delta,
                beta_fp: beta_fp[i],
                beta_bit: beta_z2[i],
                m: masks[i],
            })
            .collect();
        let sp: Vec<_> = (0..n)
            .map(|i| crate::crypto::falcon_mpc::SelectPreproc {
                c_bit: sel_z2[i],
                c_ring: sel_ring[i].clone(),
            })
            .collect();
        let bits = drelu_batch(&diffs, &wp, nbits, base, self.me, &self.mailbox);

        // Lift `Z_2 → Z_L` so the result composes like an `a2b` output.
        let zeros = vec![PartyShare { left: 0, right: 0 }; n];
        let ones: Vec<PartyShare> = (0..n)
            .map(|_| PartyShare {
                left: Fr::from(self.me == 0),
                right: Fr::from(self.me == 2),
            })
            .collect();
        let sp2: Vec<_> = (0..n)
            .map(|i| crate::crypto::falcon_mpc::SelectPreproc {
                c_bit: sp[i].c_bit,
                c_ring: sp[i].c_ring.clone(),
            })
            .collect();
        let drelu = select_batch(
            &zeros,
            &ones,
            &bits,
            &sp2,
            nbits,
            base + FALCON_ID_BLOCK / 2,
            self.me,
            &self.mailbox,
        );

        // lt = 1 − DReLU, with the public 1 in a single component.
        drelu
            .into_iter()
            .map(|d| {
                let ps = PartyShare {
                    left: Fr::from(self.me == 0).wrapping_sub(d.left) & mask,
                    right: Fr::from(self.me == 2).wrapping_sub(d.right) & mask,
                };
                let id = self.alloc_id();
                self.store(id, ps, 1, 0)
            })
            .collect()
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Reserve `n` consecutive operation ids, returning the base. Sub-protocols
    /// that run several communication rounds derive their ids from this block
    /// so they cannot collide with other operations.
    fn alloc_block(&mut self, n: u64) -> u64 {
        let id = self.next_id;
        self.next_id += n;
        id
    }

    /// Store a share.
    ///
    /// # Why the components are *not* reduced to `nbits`
    ///
    /// Tempting, and it would let the wire encoding narrow — worth 4× on the
    /// `a2b` path (7 184 → 1 796 B per sign extraction, measured). It is wrong,
    /// and the reason is a conflation in [`ShareRef`] worth naming.
    ///
    /// `nbits` records the width of the **value**, not the modulus of the
    /// **sharing**. Those differ: `lt` yields a one-bit value and stores it with
    /// `nbits = 1`, but that share is then multiplied by a 20-bit operand. Its
    /// components must therefore be valid modulo `2^20`, not `2^1`. Reduce them
    /// to one bit and the sharing sums to `1 + 2k`, so the product comes out as
    /// `v + 2kv` — silently wrong, and only at the *consumer*.
    ///
    /// Both experiments hit this: narrowing inside `a2b` broke `max2(9,6)`
    /// (22 020 105 instead of 9), and narrowing here broke the CNN pipeline
    /// (`encrypted class 5 != plaintext 7`). Same root cause both times.
    ///
    /// **The packing win is real but needs `ShareRef` to carry the sharing
    /// modulus separately from the value width**, and every operation to respect
    /// it. That is a deliberate refactor, not a masking tweak.
    fn store(&mut self, id: u64, ps: PartyShare, nbits: usize, owner: usize) -> ShareRef {
        self.shares.insert(id, ps);
        ShareRef {
            id,
            nbits,
            owner,
            fixed: false,
        }
    }

    fn get(&self, sr: &ShareRef) -> &PartyShare {
        self.shares
            .get(&sr.id)
            .unwrap_or_else(|| panic!("TransportBackend: unknown share id {}", sr.id))
    }

    fn get_cloned(&self, sr: &ShareRef) -> PartyShare {
        self.get(sr).clone()
    }

    /// This party's canonical additive component of `a`: in the replicated
    /// layout party i holds `(s_i, s_{i+1})`, so each party contributing its
    /// `left` yields the three distinct components whose wrapping sum is the
    /// secret — the per-party half of an MPC→FHE scheme switch
    /// (`crate::hybrid`), with no communication at all. This is
    /// deliberately NOT `Backend::export_additive`: that trait method's
    /// contract is "return ALL components" (satisfiable only by an
    /// in-process simulator); a real party can only ever produce its own.
    pub fn my_additive_component(&self, a: &ShareRef) -> Fr {
        self.get(a).left
    }

    fn store_public(&mut self, value: Fr, nbits: usize) -> ShareRef {
        let id = self.alloc_id();
        let ps = crate::crypto::aby3_protocol::share_public(value, self.me);
        self.store(id, ps, nbits, 0)
    }

    /// Batched [`Backend::share_secret`]: distribute `values`, all owned by
    /// `owner`, in one round instead of one per element. Ids are allocated as
    /// a consecutive block, so every party stays in lockstep exactly as the
    /// scalar path does.
    pub fn share_secret_batch(
        &mut self,
        values: &[Fr],
        nbits: usize,
        owner: usize,
    ) -> Vec<ShareRef> {
        if values.is_empty() {
            return Vec::new();
        }
        self.note_interactive();
        let base = self.alloc_block(values.len() as u64);
        let shares = crate::crypto::aby3_protocol::share_input_batch(
            values,
            base,
            owner,
            self.me,
            &self.mailbox,
        );
        shares
            .into_iter()
            .enumerate()
            .map(|(k, ps)| self.store(base + k as u64, ps, nbits, owner))
            .collect()
    }

    /// Garbled-circuit batched comparison — the **Yao leg** of ABY3
    /// (`crate::crypto::gc_protocol`). Computes `aᵢ < bᵢ` (signed, i.e.
    /// `MSB(aᵢ−bᵢ)`) for the whole batch in **one** garble round + one
    /// resharing round, regardless of bit width, versus the `~log(nbits)`
    /// rounds of the `a2b`-based [`Backend::lt_batch`]. Requires the 3-party
    /// honest-majority roles (garblers 0/1, evaluator 2); OT-free because every
    /// input wire is a garbler input. Returns RSS arithmetic shares of the 0/1
    /// results, drop-in interoperable with the rest of the pipeline.
    pub fn lt_gc_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        if pairs.is_empty() {
            return Vec::new();
        }
        let nbits = pairs
            .iter()
            .map(|(a, b)| a.nbits.max(b.nbits))
            .max()
            .unwrap();
        let n = pairs.len();
        let a: Vec<PartyShare> = pairs.iter().map(|(a, _)| self.get_cloned(a)).collect();
        let b: Vec<PartyShare> = pairs.iter().map(|(_, b)| self.get_cloned(b)).collect();
        // Reserve op_id (garble messages) + the Y2A id block, in lockstep.
        let op_id = self.alloc_block(1 + crate::crypto::gc_protocol::gc_id_block(n, 1));
        let out =
            crate::crypto::gc_protocol::gc_lt_batch(&a, &b, nbits, op_id, self.me, &self.mailbox);
        out.into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.store(id, ps, 1, 0)
            })
            .collect()
    }

    /// Garbled fused ReLU∘rescale for a batch — the whole nonlinearity in one
    /// garble round: `outᵢ = (vᵢ>=0) ? (vᵢ>>shift) : 0`. Same Yao-leg protocol
    /// and roles as [`Self::lt_gc_batch`]; returns RSS arithmetic shares of the
    /// activations (`nbits−shift`-bit results carried in `nbits`).
    pub fn relu_gc_batch(
        &mut self,
        vals: &[ShareRef],
        nbits: usize,
        shift: usize,
    ) -> Vec<ShareRef> {
        if vals.is_empty() {
            return Vec::new();
        }
        let n = vals.len();
        let width = nbits - shift;
        let vs: Vec<PartyShare> = vals.iter().map(|v| self.get_cloned(v)).collect();
        let op_id = self.alloc_block(1 + crate::crypto::gc_protocol::gc_id_block(n, width));
        let out = crate::crypto::gc_protocol::gc_relu_batch(
            &vs,
            nbits,
            shift,
            op_id,
            self.me,
            &self.mailbox,
        );
        out.into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.store(id, ps, nbits, 0)
            })
            .collect()
    }
}

impl<T: Transport + 'static> Backend for TransportBackend<T> {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef {
        self.note_interactive();
        let id = self.alloc_id();
        let ps =
            crate::crypto::aby3_protocol::share_input(value, id, owner, self.me, &self.mailbox);
        self.store(id, ps, nbits, owner)
    }

    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = self.alloc_id();
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = crate::crypto::aby3_protocol::add_party(&pa, &pb);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let id = self.alloc_id();
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = crate::crypto::aby3_protocol::sub_party(&pa, &pb);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        self.note_interactive();
        let id = self.alloc_id();
        let pa = self.get_cloned(a);
        let pb = self.get_cloned(b);
        let ps = crate::crypto::aby3_protocol::mul_party(&pa, &pb, id, self.me, &self.mailbox);
        self.store(id, ps, a.nbits.max(b.nbits), a.owner)
    }

    /// Real batched multiplication: one communication round for the whole
    /// batch (`aby3_protocol::mul_batch_party`), instead of one round per
    /// pair via the trait's default loop-over-`mul` implementation.
    /// One resharing per dot product rather than per product — see
    /// [`crate::crypto::aby3_protocol::dot_batch_party`].
    fn dot_batch(&mut self, dots: &[Vec<(ShareRef, ShareRef)>], nbits: usize) -> Vec<ShareRef> {
        self.note_interactive();
        if dots.is_empty() {
            return Vec::new();
        }
        let base_op_id = self.alloc_id();
        let owned: Vec<Vec<(PartyShare, PartyShare)>> = dots
            .iter()
            .map(|d| {
                d.iter()
                    .map(|(a, b)| (self.get_cloned(a), self.get_cloned(b)))
                    .collect()
            })
            .collect();
        let refs: Vec<&[(PartyShare, PartyShare)]> = owned.iter().map(|v| v.as_slice()).collect();
        let results = crate::crypto::aby3_protocol::dot_batch_party(
            &refs,
            base_op_id,
            self.me,
            &self.mailbox,
        );
        results
            .into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.store(id, ps, nbits, 0)
            })
            .collect()
    }

    fn mul_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        self.note_interactive();
        if pairs.is_empty() {
            return Vec::new();
        }
        let base_op_id = self.alloc_id();
        let party_pairs: Vec<_> = pairs
            .iter()
            .map(|(a, b)| (self.get_cloned(a), self.get_cloned(b)))
            .collect();
        let results = crate::crypto::aby3_protocol::mul_batch_party(
            &party_pairs,
            base_op_id,
            self.me,
            &self.mailbox,
        );
        pairs
            .iter()
            .zip(results)
            .map(|((a, b), ps)| {
                let id = self.alloc_id();
                self.store(id, ps, a.nbits.max(b.nbits), a.owner)
            })
            .collect()
    }

    // divmod: previously attempted here via bitlen-normalization, which needs
    // `shift_left(b, diff)` for a *secret* `diff = bitlen(a)-bitlen(b)` —
    // `shift_left` only takes a public `usize` amount, so that approach was a
    // dead end. Uses the Backend trait's default restoring-division
    // implementation instead (composed from a2b/mul/add/sub/le/b2a below),
    // which only ever shifts by the public constant 1.

    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        println!(
            "TransportBackend: computing eq for share ids {} and {}",
            a.id, b.id
        );
        // eq(a,b) = NOT(OR(bits of a-b))
        let nbits = a.nbits.max(b.nbits);
        let diff = self.sub(a, b);
        let bit_refs = self.a2b(&diff, nbits);
        // OR all bits via a+b-a*b
        let zero = crate::crypto::aby3_protocol::share_public(0, self.me);
        let mut acc_id = self.alloc_id();
        println!("TransportBackend: inserting zero share id {}", acc_id);
        self.shares.insert(acc_id, zero);
        for br in &bit_refs {
            let pa = self.get_cloned(&ShareRef {
                id: acc_id,
                nbits: 1,
                owner: a.owner,
                fixed: false,
            });
            let pb = self.get_cloned(br);
            let sum = crate::crypto::aby3_protocol::add_party(&pa, &pb);
            let mul_id = self.alloc_id();
            let _mul_id_triple = self.alloc_id();
            let _mul_id_triple2 = self.alloc_id();
            let _mul_id_triple3 = self.alloc_id();
            println!("TransportBackend: computing mul for OR, mul_id={}", mul_id);
            let prod =
                crate::crypto::aby3_protocol::mul_party(&pa, &pb, mul_id, self.me, &self.mailbox);
            let or_ps = crate::crypto::aby3_protocol::sub_party(&sum, &prod);
            let new_id = self.alloc_id();
            println!("TransportBackend: inserting OR share id {}", new_id);
            self.shares.insert(new_id, or_ps);
            acc_id = new_id;
        }
        // NOT = 1 - acc
        let one = crate::crypto::aby3_protocol::share_public(1, self.me);
        let one_id = self.alloc_id();
        self.shares.insert(one_id, one);
        let acc_ps = self.get_cloned(&ShareRef {
            id: acc_id,
            nbits: 1,
            owner: a.owner,
            fixed: false,
        });
        let one_ps = self.get_cloned(&ShareRef {
            id: one_id,
            nbits: 1,
            owner: a.owner,
            fixed: false,
        });
        let not_ps = crate::crypto::aby3_protocol::sub_party(&one_ps, &acc_ps);
        let id = self.alloc_id();
        self.store(id, not_ps, 1, a.owner)
    }

    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        self.note_interactive();
        // lt(a,b) = MSB of (a - b) under two's-complement
        let nbits = a.nbits.max(b.nbits);
        let diff = self.sub(a, b);
        let bit_refs = self.a2b(&diff, nbits);
        bit_refs.last().cloned().expect("a2b produced no bits")
    }

    fn lt_batch(&mut self, pairs: &[(ShareRef, ShareRef)]) -> Vec<ShareRef> {
        self.note_interactive();
        if pairs.is_empty() {
            return Vec::new();
        }
        match self.nonlin {
            NonlinMode::Gc => return self.lt_gc_batch(pairs),
            NonlinMode::Falcon => return self.lt_falcon_batch(pairs),
            NonlinMode::A2b => {}
        }
        // Each comparison is the MSB of `a − b`; batch all the decompositions
        // into one `a2b_batch` so the whole set costs ~9 rounds, not `9·len`.
        let nbits = pairs
            .iter()
            .map(|(a, b)| a.nbits.max(b.nbits))
            .max()
            .unwrap();
        let diffs: Vec<ShareRef> = pairs.iter().map(|(a, b)| self.sub(a, b)).collect();
        self.a2b_batch(&diffs, nbits)
            .into_iter()
            .map(|bits| bits.into_iter().last().expect("a2b produced no bits"))
            .collect()
    }

    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let lt = self.lt(a, b);
        let eq = self.eq(a, b);
        // OR(lt, eq)
        let pa = self.get_cloned(&lt);
        let pb = self.get_cloned(&eq);
        let sum = crate::crypto::aby3_protocol::add_party(&pa, &pb);
        let mul_id = self.alloc_id();
        let prod =
            crate::crypto::aby3_protocol::mul_party(&pa, &pb, mul_id, self.me, &self.mailbox);
        let ps = crate::crypto::aby3_protocol::sub_party(&sum, &prod);
        let id = self.alloc_id();
        self.store(id, ps, 1, a.owner)
    }

    fn reveal(&mut self, a: &ShareRef, reveal_to: Option<usize>) -> Fr {
        self.note_interactive();
        let ps = self.get_cloned(a);
        crate::crypto::aby3_protocol::reveal(&ps, a.id, reveal_to, self.me, &self.mailbox)
    }

    fn a2b(&mut self, a: &ShareRef, nbits: usize) -> Vec<ShareRef> {
        self.note_interactive();
        let pa = self.get_cloned(a);
        // Reserve a block: the bit decomposition runs ~9 batched
        // multiplication rounds internally, each consuming two ids.
        let op_id = self.alloc_block(crate::crypto::aby3_protocol::A2B_ID_BLOCK);
        let bit_shares =
            crate::crypto::aby3_protocol::a2b_party(&pa, nbits, op_id, self.me, &self.mailbox);
        bit_shares
            .into_iter()
            .map(|ps| {
                let id = self.alloc_id();
                self.shares.insert(id, ps);
                ShareRef {
                    id,
                    nbits: 1,
                    owner: a.owner,
                    fixed: false,
                }
            })
            .collect()
    }

    fn a2b_batch(&mut self, values: &[ShareRef], nbits: usize) -> Vec<Vec<ShareRef>> {
        self.note_interactive();
        if values.is_empty() {
            return Vec::new();
        }
        let party_vals: Vec<_> = values.iter().map(|v| self.get_cloned(v)).collect();
        // One id block for the whole batch: `a2b_batch_party` runs the same
        // ~9 rounds as a single decomposition, just wider.
        let op_id = self.alloc_block(crate::crypto::aby3_protocol::A2B_ID_BLOCK);
        let bit_sets = crate::crypto::aby3_protocol::a2b_batch_party(
            &party_vals,
            nbits,
            op_id,
            self.me,
            &self.mailbox,
        );
        values
            .iter()
            .zip(bit_sets)
            .map(|(v, bits)| {
                bits.into_iter()
                    .map(|ps| {
                        let id = self.alloc_id();
                        self.shares.insert(id, ps);
                        ShareRef {
                            id,
                            nbits: 1,
                            owner: v.owner,
                            fixed: false,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        self.note_interactive();
        // b2a: sum_i bit_i * 2^i  (local, no communication needed)
        let id = self.alloc_id();
        let zero = crate::crypto::aby3_protocol::share_public(0, self.me);
        let mut acc = zero;
        for (i, br) in bits.iter().enumerate() {
            let bp = self.get_cloned(br);
            let term = crate::crypto::aby3_protocol::scalar_mul_party(&bp, 1u128 << i);
            acc = crate::crypto::aby3_protocol::add_party(&acc, &term);
        }
        self.store(
            id,
            acc,
            bits.first().map(|s| s.nbits).unwrap_or(0),
            bits.first().map(|s| s.owner).unwrap_or(0),
        )
    }

    fn reset_per_expression(&mut self) {
        self.shares.clear();
        self.next_id = 1;
    }

    fn share_public(&mut self, value: Fr, nbits: usize) -> ShareRef {
        self.store_public(value, nbits)
    }

    fn and(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let bit_refs_a = self.a2b(a, a.nbits);
        let bit_refs_b = self.a2b(b, b.nbits);
        let mut and_bits = Vec::new();
        for (bit_a, bit_b) in bit_refs_a.iter().zip(bit_refs_b.iter()) {
            let and_bit = self.mul(bit_a, bit_b);
            and_bits.push(and_bit);
        }
        self.b2a(&and_bits)
    }

    fn or(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let bit_refs_a = self.a2b(a, a.nbits);
        let bit_refs_b = self.a2b(b, b.nbits);
        let mut or_bits = Vec::new();
        for (bit_a, bit_b) in bit_refs_a.iter().zip(bit_refs_b.iter()) {
            let sum = self.add(bit_a, bit_b);
            let prod = self.mul(bit_a, bit_b);
            let ps = self.sub(&sum, &prod);
            or_bits.push(ps);
        }
        self.b2a(&or_bits)
    }

    fn bitlen(&mut self, a: &ShareRef) -> ShareRef {
        let zero: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            owner: 0,
            fixed: false,
        };
        let one: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            owner: 0,
            fixed: false,
        };
        let bit_refs_a = self.a2b(a, a.nbits);
        // Compute bit_refs_b such that bit_refs_b[i] = OR of bit_refs_a[i..]
        let mut bit_refs_b = Vec::new();
        for i in 0..bit_refs_a.len() {
            let or = bit_refs_a[i..]
                .iter()
                .cloned()
                .reduce(|a, b| self.or(&a, &b))
                .unwrap_or(zero.clone());
            bit_refs_b.push(or);
        }
        // Compute bitlen as count of 1s in (not(a_i) and not(b_i)) = 1 - OR(a_i, b_i), from MSB to LSB
        let mut bitlen: ShareRef = ShareRef {
            id: 0,
            nbits: 0,
            owner: 0,
            fixed: false,
        };
        for (a, b) in bit_refs_a.iter().zip(bit_refs_b.iter()) {
            let not_a = self.sub(&one, a);
            let not_b = self.sub(&one, b);
            let and = self.and(&not_a, &not_b);
            bitlen = self.add(&bitlen, &and);
        }
        self.b2a(&bit_refs_b)
    }

    fn shift_left(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        let mut result = a.clone();
        // shift left is easier, and more efficient,
        // since we just need to add the element to itself n times
        for _ in 0..n {
            result = self.add(&result, &result);
        }
        result
    }

    fn shift_right(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        // Initialize with zeros
        let mut result = Vec::new();
        for _ in 0..n {
            let zero_share = crate::crypto::aby3_protocol::share_public(0, self.me);
            let id = self.alloc_id();
            self.shares.insert(id, zero_share);
            result.push(ShareRef::new(id, 1, a.owner));
        }
        let bit_refs_a = self.a2b(a, a.nbits);
        // copy the bits shifted to the right
        for item in bit_refs_a.iter().take(bit_refs_a.len().saturating_sub(n)) {
            result.push(item.clone());
        }
        self.b2a(&result)
    }
}
