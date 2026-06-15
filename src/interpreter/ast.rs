use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Expr {
    Num(i64),
    Var(String),
    BinOp(super::parser::BinOp, Box<Expr>, Box<Expr>),
    UnOp(super::parser::UnOp, Box<Expr>),
    Let(Vec<(String, Expr)>, Vec<Expr>),
    LetRec(Vec<(String, Expr)>, Vec<Expr>),
    Seq(Vec<Expr>),
    Car(Box<Expr>),
    Cdr(Box<Expr>),
    Cons(Box<Expr>, Box<Expr>),
    Lambda(Vec<String>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Str(String),
    Char(char),
    Bool(bool),
}

pub type Ptr = Rc<RefCell<Types>>;
pub type Env = Rc<RefCell<HashMap<String, Ptr>>>;

use crate::interpreter::backend::ShareRef;
use std::fmt;
use std::sync::Arc;

pub enum Types {
    Num(i64),
    List(Vec<Types>),
    Closure(Vec<String>, Box<Expr>, Env),
    Bool(bool),
    Str(String),
    Char(char),
    Share(ShareRef),
    NativeFunc(Arc<dyn Fn(Vec<Types>) -> Types + Send + Sync>),
}

impl Clone for Types {
    fn clone(&self) -> Self {
        match self {
            Types::Num(n) => Types::Num(*n),
            Types::List(vs) => Types::List(vs.clone()),
            Types::Closure(p, b, e) => Types::Closure(p.clone(), b.clone(), e.clone()),
            Types::Bool(b) => Types::Bool(*b),
            Types::Str(s) => Types::Str(s.clone()),
            Types::Char(c) => Types::Char(*c),
            Types::Share(sref) => Types::Share(sref.clone()),
            Types::NativeFunc(f) => Types::NativeFunc(f.clone()),
        }
    }
}

impl fmt::Debug for Types {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Types::Num(n) => write!(f, "Num({})", n),
            Types::List(vs) => write!(f, "List(len={})", vs.len()),
            Types::Closure(_, _, _) => write!(f, "Closure"),
            Types::Bool(b) => write!(f, "Bool({})", b),
            Types::Str(s) => write!(f, "Str(len={})", s.len()),
            Types::Char(c) => write!(f, "Char({})", c),
            Types::Share(sref) => write!(f, "Share(id={}, nbits={})", sref.id, sref.nbits),
            Types::NativeFunc(_) => write!(f, "NativeFunc"),
        }
    }
}
