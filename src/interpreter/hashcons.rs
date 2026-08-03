//! Content-addressed hash-consing store.
//!
//! Every value — runtime datum, AST node, environment — is interned into a
//! tagged node keyed by the hash of its canonical content.  Two things follow:
//!
//! * **O(1) structural equality.** Constructing a value that already exists
//!   returns the same [`NodeId`]/[`ZHash`], so comparing arbitrarily complex
//!   objects is a 32-byte hash compare.  A "Merkle root" of some data is just
//!   the [`ZHash`] of its root node.
//! * **Programs are data.** `Expr` interns into the same store as values, so a
//!   program is just another node — storable in, and committed from, the same
//!   tree as ordinary data.
//!
//! The store is a DAG (children referenced by hash), so shared subterms are
//! stored once.  A participant's private/public data tree, and the group's
//! program tree, are all just rooted regions of one store.
//!
//! The hash is SHA-256 for now; it is the single swap point if a ZK-friendly
//! hash (e.g. Keccak/Poseidon for the binius64 backend) is wanted later.
//!
//! Limitation: a [`Types::Closure`] is committed by its *code* (params + body),
//! not its captured environment — that keeps interning acyclic in the presence
//! of recursive closures.  Committing captured environments (cycle-aware) is a
//! follow-up.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::interpreter::ast::{Env, Expr, Types};
use crate::interpreter::parser::{BinOp, UnOp};

/// A 32-byte content commitment — the "Merkle root" of whatever it names.
pub type ZHash = [u8; 32];

/// A local handle into a [`Store`] (fast array index).  Distinct from the
/// interpreter's runtime `Ptr`; this one is an immutable interned reference.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(u32);

/// Node kinds.  The discriminant is part of the content hash — **do not
/// reorder** existing variants (append new ones at the end).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tag {
    // ── values ──
    VNum,
    VBool,
    VChar,
    VStr,
    VNil,
    VCons,
    VQuoted,
    VClosure,
    VShare,
    VNative,
    // ── expressions ──
    ENum,
    EVar,
    EStr,
    EChar,
    EBool,
    EQuote,
    EEval,
    EBinOp,
    EUnOp,
    ELet,
    ELetRec,
    ESeq,
    ECar,
    ECdr,
    ECons,
    ELambda,
    ECall,
    EIf,
    EPrivate,
    // ── structural ──
    Sym,
    List,
    Binding,
    EnvNode,
    State,
    // ── appended after the initial release (see comment above: do not
    //    reorder existing variants above this line) ──
    /// Q16.16 fixed-point value (`Types::Fixed`).
    VFixed,
    /// Opaque FHE ciphertext handle (`Types::FheCiphertext`).
    VFheCiphertext,
    /// Q16.16 fixed-point literal (`Expr::Fixed`).
    EFixed,
    /// `(define name value)` (`Expr::Define`).
    EDefine,
    /// `(set! name value)` (`Expr::Set`).
    ESet,
}

/// An interned node: a tag, an inline byte payload for leaves, and links to
/// child nodes *by hash* (canonical, so the preimage is location-independent).
#[derive(Clone, Debug)]
pub struct Node {
    pub tag: Tag,
    pub atom: Vec<u8>,
    pub children: Vec<ZHash>,
}

/// Canonical node hash: `SHA-256(tag ‖ len(atom) ‖ atom ‖ len(children) ‖ child hashes)`.
fn hash_node(tag: Tag, atom: &[u8], children: &[ZHash]) -> ZHash {
    let mut h = Sha256::new();
    h.update([tag as u8]);
    h.update((atom.len() as u64).to_le_bytes());
    h.update(atom);
    h.update((children.len() as u64).to_le_bytes());
    for c in children {
        h.update(c);
    }
    h.finalize().into()
}

/// A content-addressed store with structural interning.
#[derive(Default)]
pub struct Store {
    nodes: Vec<Node>,
    hashes: Vec<ZHash>,
    intern: HashMap<ZHash, NodeId>,
}

impl Store {
    pub fn new() -> Self {
        Store::default()
    }

    /// Number of *distinct* interned nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Intern a node, deduplicating by content hash.
    pub fn intern(&mut self, tag: Tag, atom: Vec<u8>, children: Vec<ZHash>) -> NodeId {
        let h = hash_node(tag, &atom, &children);
        if let Some(&id) = self.intern.get(&h) {
            return id;
        }
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            tag,
            atom,
            children,
        });
        self.hashes.push(h);
        self.intern.insert(h, id);
        id
    }

    /// The content commitment of a node.
    pub fn hash(&self, id: NodeId) -> ZHash {
        self.hashes[id.0 as usize]
    }

    /// The node behind a handle.
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Look up the handle for a content hash, if interned.
    pub fn id_of(&self, h: &ZHash) -> Option<NodeId> {
        self.intern.get(h).copied()
    }

    // ── leaf / structural constructors ────────────────────────────────────
    pub fn num(&mut self, n: i64) -> NodeId {
        self.intern(Tag::VNum, n.to_le_bytes().to_vec(), vec![])
    }
    pub fn boolean(&mut self, b: bool) -> NodeId {
        self.intern(Tag::VBool, vec![b as u8], vec![])
    }
    pub fn ch(&mut self, c: char) -> NodeId {
        self.intern(Tag::VChar, (c as u32).to_le_bytes().to_vec(), vec![])
    }
    pub fn string(&mut self, s: &str) -> NodeId {
        self.intern(Tag::VStr, s.as_bytes().to_vec(), vec![])
    }
    pub fn sym(&mut self, s: &str) -> NodeId {
        self.intern(Tag::Sym, s.as_bytes().to_vec(), vec![])
    }
    pub fn nil(&mut self) -> NodeId {
        self.intern(Tag::VNil, vec![], vec![])
    }
    /// A cons cell over two already-interned nodes.
    pub fn cons(&mut self, car: NodeId, cdr: NodeId) -> NodeId {
        let (a, b) = (self.hash(car), self.hash(cdr));
        self.intern(Tag::VCons, vec![], vec![a, b])
    }
    /// The `car` of a cons cell, if this node is one.
    pub fn car(&self, id: NodeId) -> Option<NodeId> {
        let n = self.get(id);
        if n.tag == Tag::VCons {
            self.id_of(&n.children[0])
        } else {
            None
        }
    }
    /// The `cdr` of a cons cell, if this node is one.
    pub fn cdr(&self, id: NodeId) -> Option<NodeId> {
        let n = self.get(id);
        if n.tag == Tag::VCons {
            self.id_of(&n.children[1])
        } else {
            None
        }
    }

    /// A proper list `(a b c … nil)` from already-interned items.
    pub fn list(&mut self, items: &[NodeId]) -> NodeId {
        let mut acc = self.nil();
        for &item in items.iter().rev() {
            acc = self.cons(item, acc);
        }
        acc
    }

    /// A flat tagged tuple over child hashes (used for structural aggregates).
    fn tuple(&mut self, tag: Tag, children: &[NodeId]) -> NodeId {
        let hs: Vec<ZHash> = children.iter().map(|&c| self.hash(c)).collect();
        self.intern(tag, vec![], hs)
    }

    // ── interning runtime values ──────────────────────────────────────────
    /// Intern a runtime [`Types`] value, returning its handle.
    pub fn intern_value(&mut self, v: &Types) -> NodeId {
        match v {
            // Interned like any other opaque builtin: it has no structure.
            Types::NativeApply => self.intern(Tag::VNative, vec![], vec![]),
            // By message: two conditions with the same text are the same node.
            Types::Error(e, _) => self.intern(Tag::VStr, e.message.as_bytes().to_vec(), vec![]),
            // Interned by *contents*, like a list. Two vectors with equal
            // elements therefore share a handle even though `eq?` distinguishes
            // them — hash-consing is about structure, `eq?` about identity.
            Types::Vector(v) => {
                let ids: Vec<NodeId> = v.borrow().iter().map(|x| self.intern_value(x)).collect();
                self.list(&ids)
            }
            Types::Num(n) => self.num(*n),
            Types::Fixed(v) => self.intern(Tag::VFixed, v.to_le_bytes().to_vec(), vec![]),
            Types::Bool(b) => self.boolean(*b),
            Types::Str(s) => self.string(s),
            Types::Char(c) => self.ch(*c),
            Types::List(items) => {
                let ids: Vec<NodeId> = items.iter().map(|x| self.intern_value(x)).collect();
                self.list(&ids)
            }
            Types::Quoted(e) => {
                let eh = self.intern_expr(e);
                self.tuple(Tag::VQuoted, &[eh])
            }
            // Closures are committed by their *code* only (see module note).
            Types::Closure(params, _rest, body, _env) => {
                let p = self.sym_list(params);
                let b = self.intern_expr(body);
                self.tuple(Tag::VClosure, &[p, b])
            }
            Types::Share(sref) => {
                let mut atom = Vec::with_capacity(24);
                atom.extend_from_slice(&(sref.id).to_le_bytes());
                atom.extend_from_slice(&(sref.nbits as u64).to_le_bytes());
                atom.extend_from_slice(&(sref.owner as u64).to_le_bytes());
                self.intern(Tag::VShare, atom, vec![])
            }
            // Native builtins are opaque: their behavior is not content-addressable.
            Types::NativeFunc(_) => self.intern(Tag::VNative, vec![], vec![]),
            // FHE ciphertexts commit to their backend id + serialised bytes;
            // the `fixed` flag is folded into the atom so Q16.16 and integer
            // ciphertexts with identical bytes (impossible in practice, but
            // not structurally guaranteed) still intern distinctly.
            Types::FheCiphertext(h) => {
                let mut atom = Vec::with_capacity(h.backend_id.len() + h.bytes.len() + 9);
                atom.extend_from_slice(&(h.backend_id.len() as u64).to_le_bytes());
                atom.extend_from_slice(h.backend_id.as_bytes());
                atom.push(h.fixed as u8);
                atom.extend_from_slice(&h.bytes);
                self.intern(Tag::VFheCiphertext, atom, vec![])
            }
        }
    }

    // ── interning programs (Expr) ─────────────────────────────────────────
    fn sym_list(&mut self, names: &[String]) -> NodeId {
        let ids: Vec<NodeId> = names.iter().map(|n| self.sym(n)).collect();
        self.list(&ids)
    }

    fn expr_list(&mut self, es: &[Expr]) -> NodeId {
        let ids: Vec<NodeId> = es.iter().map(|e| self.intern_expr(e)).collect();
        self.list(&ids)
    }

    /// A `(name . value-expr)` binding node.
    fn binding(&mut self, name: &str, value: NodeId) -> NodeId {
        let vh = self.hash(value);
        self.intern(Tag::Binding, name.as_bytes().to_vec(), vec![vh])
    }

    fn binding_list(&mut self, binds: &[(String, Expr)]) -> NodeId {
        let ids: Vec<NodeId> = binds
            .iter()
            .map(|(name, e)| {
                let v = self.intern_expr(e);
                self.binding(name, v)
            })
            .collect();
        self.list(&ids)
    }

    /// Intern an [`Expr`], returning its handle.  Programs are data.
    pub fn intern_expr(&mut self, e: &Expr) -> NodeId {
        match e {
            Expr::Num(n) => self.intern(Tag::ENum, n.to_le_bytes().to_vec(), vec![]),
            Expr::Fixed(v) => self.intern(Tag::EFixed, v.to_le_bytes().to_vec(), vec![]),
            Expr::Var(s) => self.intern(Tag::EVar, s.as_bytes().to_vec(), vec![]),
            Expr::Str(s) => self.intern(Tag::EStr, s.as_bytes().to_vec(), vec![]),
            Expr::Char(c) => self.intern(Tag::EChar, (*c as u32).to_le_bytes().to_vec(), vec![]),
            Expr::Bool(b) => self.intern(Tag::EBool, vec![*b as u8], vec![]),
            Expr::Quote(inner) => {
                let ih = self.intern_expr(inner);
                self.tuple(Tag::EQuote, &[ih])
            }
            Expr::Eval(inner) => {
                let ih = self.intern_expr(inner);
                self.tuple(Tag::EEval, &[ih])
            }
            Expr::BinOp(op, l, r) => {
                let (lh, rh) = (self.intern_expr(l), self.intern_expr(r));
                let lz = self.hash(lh);
                let rz = self.hash(rh);
                self.intern(Tag::EBinOp, vec![binop_code(op)], vec![lz, rz])
            }
            Expr::UnOp(op, inner) => {
                let ih = self.intern_expr(inner);
                let iz = self.hash(ih);
                self.intern(Tag::EUnOp, vec![unop_code(op)], vec![iz])
            }
            Expr::Let(binds, body) => {
                let bl = self.binding_list(binds);
                let bd = self.expr_list(body);
                self.tuple(Tag::ELet, &[bl, bd])
            }
            Expr::LetRec(binds, body) => {
                let bl = self.binding_list(binds);
                let bd = self.expr_list(body);
                self.tuple(Tag::ELetRec, &[bl, bd])
            }
            Expr::Seq(es) => {
                let l = self.expr_list(es);
                self.tuple(Tag::ESeq, &[l])
            }
            Expr::Car(inner) => {
                let ih = self.intern_expr(inner);
                self.tuple(Tag::ECar, &[ih])
            }
            Expr::Cdr(inner) => {
                let ih = self.intern_expr(inner);
                self.tuple(Tag::ECdr, &[ih])
            }
            Expr::Cons(a, b) => {
                let (ah, bh) = (self.intern_expr(a), self.intern_expr(b));
                self.tuple(Tag::ECons, &[ah, bh])
            }
            Expr::Guard(var, handler, body) => {
                let v = self.intern(Tag::EVar, var.as_bytes().to_vec(), vec![]);
                let h = self.expr_list(handler);
                let b = self.expr_list(body);
                self.tuple(Tag::ESeq, &[v, h, b])
            }
            Expr::Define(name, value) => {
                let n = self.intern(Tag::EVar, name.as_bytes().to_vec(), vec![]);
                let v = self.intern_expr(value);
                self.tuple(Tag::EDefine, &[n, v])
            }
            Expr::Set(name, value) => {
                let n = self.intern(Tag::EVar, name.as_bytes().to_vec(), vec![]);
                let v = self.intern_expr(value);
                self.tuple(Tag::ESet, &[n, v])
            }
            Expr::Lambda(params, _rest, body) => {
                let p = self.sym_list(params);
                let b = self.intern_expr(body);
                self.tuple(Tag::ELambda, &[p, b])
            }
            Expr::Call(f, args) => {
                let fh = self.intern_expr(f);
                let al = self.expr_list(args);
                self.tuple(Tag::ECall, &[fh, al])
            }
            Expr::If(c, t, els) => {
                let (ch, th, eh) = (
                    self.intern_expr(c),
                    self.intern_expr(t),
                    self.intern_expr(els),
                );
                self.tuple(Tag::EIf, &[ch, th, eh])
            }
            Expr::Private(inner) => {
                let ih = self.intern_expr(inner);
                self.tuple(Tag::EPrivate, &[ih])
            }
        }
    }

    // ── environment & state commitments ───────────────────────────────────
    /// Commit an environment as a canonical (name-sorted) set of bindings.
    pub fn intern_env(&mut self, env: &Env) -> NodeId {
        let mut pairs: Vec<(String, Types)> = env
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.borrow().clone()))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let binds: Vec<NodeId> = pairs
            .iter()
            .map(|(name, val)| {
                let v = self.intern_value(val);
                self.binding(name, v)
            })
            .collect();
        self.tuple(Tag::EnvNode, &binds)
    }

    /// Commit a participant's compute state.  The full CEK triple is
    /// `(env, expr, cont)`; the continuation is committed as a follow-up, so
    /// this currently commits `(env, expr)` — enough to agree on data + program.
    pub fn state_root(&mut self, env: &Env, expr: &Expr) -> ZHash {
        let e = self.intern_env(env);
        let x = self.intern_expr(expr);
        let node = self.tuple(Tag::State, &[e, x]);
        self.hash(node)
    }
}

/// Stable byte code for a binary operator (part of the content hash).
fn binop_code(op: &BinOp) -> u8 {
    *op as u8
}

/// Stable byte code for a unary operator (part of the content hash).
fn unop_code(op: &UnOp) -> u8 {
    *op as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::parse;

    #[test]
    fn interning_dedups_structural_equals() {
        let mut s = Store::new();
        let a = s.num(42);
        let b = s.num(42);
        assert_eq!(a, b, "equal values must intern to the same handle");
        assert_eq!(s.hash(a), s.hash(b));
        let c = s.num(43);
        assert_ne!(s.hash(a), s.hash(c));
        // Only two distinct nodes were stored (42 and 43).
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn shared_subterms_stored_once() {
        let mut s = Store::new();
        let leaf = s.num(7);
        let p1 = s.cons(leaf, leaf);
        let p2 = s.cons(leaf, leaf);
        assert_eq!(p1, p2);
        // nodes: num(7), nil?(no), cons -> 2 distinct (num, cons)
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn car_cdr_navigate() {
        let mut s = Store::new();
        let a = s.num(1);
        let b = s.num(2);
        let pair = s.cons(a, b);
        assert_eq!(s.car(pair), Some(a));
        assert_eq!(s.cdr(pair), Some(b));
        assert_eq!(s.car(a), None);
    }

    #[test]
    fn equal_programs_get_equal_roots() {
        let e1 = parse("(+ (* 2 3) 4)").unwrap();
        let e2 = parse("(+ (* 2 3) 4)").unwrap();
        let e3 = parse("(+ (* 2 3) 5)").unwrap();
        let mut s = Store::new();
        let (i1, i2, i3) = (s.intern_expr(&e1), s.intern_expr(&e2), s.intern_expr(&e3));
        let (h1, h2, h3) = (s.hash(i1), s.hash(i2), s.hash(i3));
        assert_eq!(h1, h2, "identical programs must share a root");
        assert_ne!(h1, h3, "different programs must differ");
    }

    #[test]
    fn values_and_programs_share_one_store() {
        // "programs are data": interning a value and a program into the same
        // store just produces different roots, never a conflict.
        let mut s = Store::new();
        let v = s.intern_value(&Types::List(vec![Types::Num(1), Types::Bool(true)]));
        let e = s.intern_expr(&parse("(cons 1 true)").unwrap());
        let (vh, eh) = (s.hash(v), s.hash(e));
        assert_ne!(vh, eh);
    }

    #[test]
    fn interning_is_deterministic_across_stores() {
        let e = parse("(let ((x 3)) (* x x))").unwrap();
        let mut s1 = Store::new();
        let mut s2 = Store::new();
        // Insert extra junk into s2 first — the hash must not depend on order.
        let _ = s2.num(999);
        let (i1, i2) = (s1.intern_expr(&e), s2.intern_expr(&e));
        let (h1, h2) = (s1.hash(i1), s2.hash(i2));
        assert_eq!(h1, h2);
    }
}
