use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

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

pub type Ptr = Rc<RefCell<SharedValue>>;
pub type Env = Rc<RefCell<HashMap<String, Ptr>>>;

use crate::interpreter::backend::ShareRef;
use std::sync::Arc;
use std::fmt;

pub enum SharedValue {
    Num(i64),
    List(Vec<SharedValue>),
    Closure(Vec<String>, Box<Expr>, Env),
    Bool(bool),
    Str(String),
    Char(char),
    Share(ShareRef),
    NativeFunc(Arc<dyn Fn(Vec<SharedValue>) -> SharedValue + Send + Sync>),
}

impl Clone for SharedValue {
    fn clone(&self) -> Self {
        match self {
            SharedValue::Num(n) => SharedValue::Num(*n),
            SharedValue::List(vs) => SharedValue::List(vs.clone()),
            SharedValue::Closure(p, b, e) => SharedValue::Closure(p.clone(), b.clone(), e.clone()),
            SharedValue::Bool(b) => SharedValue::Bool(*b),
            SharedValue::Str(s) => SharedValue::Str(s.clone()),
            SharedValue::Char(c) => SharedValue::Char(*c),
            SharedValue::Share(sref) => SharedValue::Share(sref.clone()),
            SharedValue::NativeFunc(f) => SharedValue::NativeFunc(f.clone()),
        }
    }
}

impl fmt::Debug for SharedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SharedValue::Num(n) => write!(f, "Num({})", n),
            SharedValue::List(vs) => write!(f, "List(len={})", vs.len()),
            SharedValue::Closure(_, _, _) => write!(f, "Closure"),
            SharedValue::Bool(b) => write!(f, "Bool({})", b),
            SharedValue::Str(s) => write!(f, "Str(len={})", s.len()),
            SharedValue::Char(c) => write!(f, "Char({})", c),
            SharedValue::Share(sref) => write!(f, "Share(id={}, nbits={})", sref.id, sref.nbits),
            SharedValue::NativeFunc(_) => write!(f, "NativeFunc"),
        }
    }
}
