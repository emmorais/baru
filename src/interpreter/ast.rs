use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Expr {
    Num(i64),
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
    Lambda(Vec<String>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Str(String),
    Char(char),
    Bool(bool),
}

// Implement to_string for Expr
impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Num(n) => write!(f, "{}", n),
            Expr::Var(v) => write!(f, "{}", v),
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
            Expr::Lambda(params, body) => {
                write!(f, "(lambda (")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ") {}", body)?;
                Ok(())
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
        }
    }
}

pub type Ptr = Rc<RefCell<Types>>;
pub type Env = Rc<RefCell<HashMap<String, Ptr>>>;

use crate::interpreter::backend::ShareRef;
use std::fmt;
use std::sync::Arc;

pub enum Types {
    Num(i64),
    List(Vec<Types>),
    Quoted(Box<Expr>),
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
            Types::Quoted(e) => Types::Quoted(e.clone()),
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
            Types::Quoted(e) => write!(f, "Quoted({:?})", e),
            Types::Closure(_, _, _) => write!(f, "Closure"),
            Types::Bool(b) => write!(f, "Bool({})", b),
            Types::Str(s) => write!(f, "Str(len={})", s.len()),
            Types::Char(c) => write!(f, "Char({})", c),
            Types::Share(sref) => write!(f, "Share(id={}, nbits={})", sref.id, sref.nbits),
            Types::NativeFunc(_) => write!(f, "NativeFunc"),
        }
    }
}
