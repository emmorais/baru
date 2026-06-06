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

#[derive(Debug, Clone)]
pub enum SharedValue {
    Num(i64),
    List(Vec<SharedValue>),
    Closure(Vec<String>, Box<Expr>, Env),
    Bool(bool),
    Str(String),
    Char(char),
}
