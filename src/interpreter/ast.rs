use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Fixed-point scale factor: Q16.16 representation.
/// `42.5` is stored as `42.5 * 65536 = 2785280`.
/// Multiplication requires a right-shift by 16 to truncate back.
pub const FIXED_SCALE: i64 = 1 << 16; // 65_536

#[derive(Debug, Clone)]
pub enum Expr {
    Num(i64),
    /// Fixed-point literal, stored as `value * FIXED_SCALE`.
    Fixed(i64),
    Var(String),
    Quote(Box<Expr>),
    Eval(Box<Expr>),
    BinOp(super::parser::BinOp, Box<Expr>, Box<Expr>),
    UnOp(super::parser::UnOp, Box<Expr>),
    Let(Vec<(String, Expr)>, Vec<Expr>),
    LetRec(Vec<(String, Expr)>, Vec<Expr>),
    Seq(Vec<Expr>),
    Car(Box<Expr>),
    Cdr(Box<Expr>),
    Cons(Box<Expr>, Box<Expr>),
    /// `(lambda (a b) body)`, or `(lambda (a b . rest) body)` when the second
    /// field is `Some`. The rest parameter collects the remaining arguments into
    /// a list, so a procedure can be variadic.
    Lambda(Vec<String>, Option<String>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Str(String),
    Char(char),
    Bool(bool),
    Private(Box<Expr>), // NEW: (private expr) for ZKP witness marking
    /// `(define name value)` — bind in the current environment.
    ///
    /// Unlike the derived forms in the parser, this needs its own variant
    /// because it *mutates* an environment, which no combination of
    /// `let`/`if`/`lambda` expresses. The circuit backends reject it: a circuit
    /// has no environment to extend at proving time.
    Define(String, Box<Expr>),
    /// `(set! name value)` — assign to an existing binding.
    ///
    /// Bindings are `Rc<RefCell<Types>>` shared between an environment and the
    /// closures that captured it, so assignment is visible through every
    /// closure over the same variable — Scheme's semantics, and the reason
    /// `set!` is worth having rather than simulating with rebinding.
    Set(String, Box<Expr>),
    /// `(guard (var handler...) body...)` — R7RS-style condition handling.
    ///
    /// Installs a handler, evaluates `body`, and on a raise binds the condition
    /// to `var` and runs `handler` *in place of* the rest of `body`. A real form
    /// rather than a derived one because it must push and pop machine state.
    Guard(String, Vec<Expr>, Vec<Expr>),
}

// Implement to_string for Expr
impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Num(n) => write!(f, "{}", n),
            Expr::Fixed(v) => {
                // Display as decimal: divide scaled integer by FIXED_SCALE
                let whole = v / FIXED_SCALE;
                let frac = (v % FIXED_SCALE).abs();
                // 4 decimal digits (65536 / 10000 ≈ 6.5, so multiply frac by 10000)
                let frac_digits = (frac * 10_000) / FIXED_SCALE;
                write!(f, "{}.{:04}", whole, frac_digits)
            }
            Expr::Var(v) => write!(f, "{}", v),
            Expr::Define(n, v) => write!(f, "(define {} {})", n, v),
            Expr::Set(n, v) => write!(f, "(set! {} {})", n, v),
            Expr::Quote(e) => write!(f, "'{}", e),
            Expr::Eval(e) => write!(f, "(eval {})", e),
            Expr::BinOp(op, lhs, rhs) => match op {
                super::parser::BinOp::Add => write!(f, "(+ {} {})", lhs, rhs),
                super::parser::BinOp::Sub => write!(f, "(- {} {})", lhs, rhs),
                super::parser::BinOp::Mul => write!(f, "(* {} {})", lhs, rhs),
                super::parser::BinOp::DivExact => write!(f, "(/ {} {})", lhs, rhs),
                super::parser::BinOp::Mod => write!(f, "(mod {} {})", lhs, rhs),
                super::parser::BinOp::Eq => write!(f, "(= {} {})", lhs, rhs),
                super::parser::BinOp::Lt => write!(f, "(< {} {})", lhs, rhs),
                super::parser::BinOp::Gt => write!(f, "(> {} {})", lhs, rhs),
                super::parser::BinOp::Le => write!(f, "(<= {} {})", lhs, rhs),
                super::parser::BinOp::Ge => write!(f, "(>= {} {})", lhs, rhs),
                super::parser::BinOp::And => write!(f, "(and {} {})", lhs, rhs),
                super::parser::BinOp::Or => write!(f, "(or {} {})", lhs, rhs),
                super::parser::BinOp::AshLeft => write!(f, "(ashl {} {})", lhs, rhs),
                super::parser::BinOp::AshRight => write!(f, "(ashr {} {})", lhs, rhs),
            },
            Expr::UnOp(op, expr) => match op {
                super::parser::UnOp::Neg => write!(f, "(- {})", expr),
                super::parser::UnOp::Not => write!(f, "(not {})", expr),
                super::parser::UnOp::Atom => write!(f, "(atom {})", expr),
                super::parser::UnOp::BitLen => write!(f, "(bitlen {})", expr),
                super::parser::UnOp::IsNull => write!(f, "(null? {})", expr),
            },
            Expr::Let(bindings, body) => {
                write!(f, "(let (")?;
                for (i, (var, expr)) in bindings.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", var, expr)?;
                }
                write!(f, ") (")?;
                for (i, expr) in body.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, "))")
            }
            Expr::LetRec(bindings, body) => {
                write!(f, "(letrec (")?;
                for (i, (var, expr)) in bindings.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", var, expr)?;
                }
                write!(f, ") (")?;
                for (i, expr) in body.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, "))")
            }
            Expr::Seq(exprs) => {
                write!(f, "(seq")?;
                for expr in exprs {
                    write!(f, " {}", expr)?;
                }
                write!(f, ")")
            }
            Expr::Car(expr) => write!(f, "(car {})", expr),
            Expr::Cdr(expr) => write!(f, "(cdr {})", expr),
            Expr::Cons(lhs, rhs) => write!(f, "(cons {} {})", lhs, rhs),
            Expr::Lambda(params, rest, body) => {
                write!(f, "(lambda (")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", param)?;
                }
                // Round-trippable: a variadic lambda must print its dot, or
                // `Display` would render it as a fixed-arity one.
                if let Some(r) = rest {
                    if !params.is_empty() {
                        write!(f, " ")?;
                    }
                    write!(f, ". {}", r)?;
                }
                // The trailing paren closes the `lambda` form itself. It was
                // missing before 2026-08-03, so every lambda printed unbalanced.
                write!(f, ") {})", body)
            }
            Expr::Call(func, args) => {
                write!(f, "({}", func)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
            Expr::If(cond, then_branch, else_branch) => {
                write!(f, "(if {} {} {})", cond, then_branch, else_branch)
            }
            Expr::Str(s) => write!(f, "\"{}\"", s),
            Expr::Char(c) => write!(f, "'{}'", c),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::Private(expr) => write!(f, "(private {})", expr),
            Expr::Guard(var, handler, body) => {
                write!(f, "(guard ({var}")?;
                for h in handler {
                    write!(f, " {h}")?;
                }
                write!(f, ")")?;
                for b in body {
                    write!(f, " {b}")?;
                }
                write!(f, ")")
            }
        }
    }
}

pub type Ptr = Rc<RefCell<Types>>;
pub type Env = Rc<RefCell<HashMap<String, Ptr>>>;

/// A condition: a message plus the values that provoked it.
///
/// Deliberately minimal — this is not a condition *hierarchy*. The distinction
/// Scheme's `error-object?` draws is enough for the failures this language
/// actually has (unbound variable, arity, type, index, division by zero), and a
/// taxonomy nobody dispatches on would be decoration.
#[derive(Debug)]
pub struct ErrObj {
    pub message: String,
    pub irritants: Vec<Types>,
}

impl ErrObj {
    /// A fresh **in-flight** condition.
    ///
    /// Named `raise` rather than `new` because it does not return an `ErrObj`:
    /// it returns the [`Types`] value whose delivery to a continuation *is* the
    /// raise (see the error check in the `Apply` step).
    pub fn raise(message: impl Into<String>) -> Types {
        Self::raise_with(message, Vec::new())
    }

    /// As [`ErrObj::raise`], carrying the values that provoked the failure.
    pub fn raise_with(message: impl Into<String>, irritants: Vec<Types>) -> Types {
        Types::Error(
            Rc::new(ErrObj {
                message: message.into(),
                irritants,
            }),
            true,
        )
    }

    /// Rendered the way the REPL and a panic message would show it.
    pub fn display(&self) -> String {
        if self.irritants.is_empty() {
            self.message.clone()
        } else {
            let items: Vec<String> = self.irritants.iter().map(|i| format!("{i:?}")).collect();
            format!("{}: {}", self.message, items.join(", "))
        }
    }
}

use crate::interpreter::backend::ShareRef;
use std::fmt;
use std::sync::Arc;

pub enum Types {
    Num(i64),
    /// Fixed-point value (Q16.16 scaled integer).
    Fixed(i64),
    List(Vec<Types>),
    Quoted(Box<Expr>),
    Closure(Vec<String>, Option<String>, Box<Expr>, Env),
    Bool(bool),
    Str(String),
    Char(char),
    Share(ShareRef),
    NativeFunc(Arc<dyn Fn(Vec<Types>) -> Types + Send + Sync>),
    /// An FHE ciphertext handle (serialised as JSON for REPL round-trips).
    FheCiphertext(crate::fhe::backend::FheCiphertextHandle),
    /// A condition, plus whether it is currently **in flight**.
    ///
    /// In flight (`true`), it is not an ordinary value: the moment one is handed
    /// to a continuation the machine stops and unwinds to the nearest `guard`
    /// (see the error check in the `Apply` step). That is what lets a builtin
    /// signal failure by *returning* — a `NativeFunc` cannot alter control flow.
    ///
    /// A handler binds the condition **inert** (`false`), so the caught value can
    /// be inspected and passed around like any other. Without that distinction,
    /// merely mentioning the handler's variable would re-raise it. `raise` turns
    /// an inert condition back into an in-flight one.
    Error(Rc<ErrObj>, bool),
    /// A Scheme vector: fixed-length, mutable, with **reference semantics**.
    ///
    /// `Rc<RefCell<..>>` rather than `Vec<Types>` because `vector-set!` must be
    /// observable through every binding that shares the vector, which is what
    /// distinguishes a vector from this language's immutable [`Types::List`].
    /// Cloning a `Types::Vector` shares the storage, exactly as passing a vector
    /// in Scheme does.
    Vector(Rc<RefCell<Vec<Types>>>),
    /// `apply` — a procedure the *evaluator* must implement, not a
    /// [`Types::NativeFunc`].
    ///
    /// A `NativeFunc` receives values and returns a value; it has no way to
    /// *call* a closure, which is precisely what `apply` must do. So `apply` is
    /// a distinguished value that the call dispatcher recognises and expands.
    ///
    /// It is a value rather than a special form on purpose: in Scheme `apply` is
    /// an ordinary procedure, so `(let ((apply ...)) ...)` must shadow it. A
    /// parser-level special form would silently capture that name — which it did
    /// in the first version of this change, breaking a higher-order test that
    /// used `apply` as a user variable.
    NativeApply,
}

impl Clone for Types {
    fn clone(&self) -> Self {
        match self {
            Types::NativeApply => Types::NativeApply,
            Types::Error(e, flight) => Types::Error(Rc::clone(e), *flight),
            // Rc clone: the storage is *shared*, which is what makes
            // `vector-set!` visible through every binding of the vector.
            Types::Vector(v) => Types::Vector(Rc::clone(v)),
            Types::Num(n) => Types::Num(*n),
            Types::Fixed(v) => Types::Fixed(*v),
            Types::List(vs) => Types::List(vs.clone()),
            Types::Quoted(e) => Types::Quoted(e.clone()),
            Types::Closure(p, r, b, e) => {
                Types::Closure(p.clone(), r.clone(), b.clone(), e.clone())
            }
            Types::Bool(b) => Types::Bool(*b),
            Types::Str(s) => Types::Str(s.clone()),
            Types::Char(c) => Types::Char(*c),
            Types::Share(sref) => Types::Share(sref.clone()),
            Types::NativeFunc(f) => Types::NativeFunc(f.clone()),
            Types::FheCiphertext(h) => Types::FheCiphertext(h.clone()),
        }
    }
}

impl fmt::Debug for Types {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Types::NativeApply => write!(f, "NativeApply"),
            Types::Error(e, _) => write!(f, "Error({})", e.display()),
            Types::Vector(v) => write!(f, "Vector(len={})", v.borrow().len()),
            Types::Num(n) => write!(f, "Num({})", n),
            Types::Fixed(v) => {
                let whole = v / FIXED_SCALE;
                let frac = (v % FIXED_SCALE).abs();
                let frac_digits = (frac * 10_000) / FIXED_SCALE;
                write!(f, "Fixed({}.{:04})", whole, frac_digits)
            }
            Types::List(vs) => write!(f, "List(len={})", vs.len()),
            Types::Quoted(e) => write!(f, "Quoted({:?})", e),
            Types::Closure(_, _, _, _) => write!(f, "Closure"),
            Types::Bool(b) => write!(f, "Bool({})", b),
            Types::Str(s) => write!(f, "Str(len={})", s.len()),
            Types::Char(c) => write!(f, "Char({})", c),
            Types::Share(sref) => write!(f, "Share(id={}, nbits={})", sref.id, sref.nbits),
            Types::NativeFunc(_) => write!(f, "NativeFunc"),
            Types::FheCiphertext(h) => {
                write!(
                    f,
                    "FheCiphertext(backend={}, bytes={})",
                    h.backend_id,
                    h.bytes.len()
                )
            }
        }
    }
}
