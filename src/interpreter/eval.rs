use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use crate::interpreter::ast::{Expr, Env, SharedValue};
use crate::interpreter::backend::{ShareRef, SimBackend, Backend};
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub enum Continuation {
    Halt,
    BinOpLeft {
        op: super::parser::BinOp,
        right: Box<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },
    BinOpRight {
        op: super::parser::BinOp,
        left_val: SharedValue,
        cont: Box<Continuation>,
        env: Env,
    },
    UnOp {
        op: super::parser::UnOp,
        cont: Box<Continuation>,
        env: Env,
    },

    LetTop {
        name: String,
        remaining: Vec<(String, Expr)>,
        env: Env,
    },

    LetLocal {
        name: String,
        remaining: Vec<(String, Expr)>,
        local_env: Env,
        body: Vec<Expr>,
        cont: Box<Continuation>,
    },

    LetRecInit {
        name: String,
        remaining: Vec<(String, Expr)>,
        local_env: Env,
        body: Vec<Expr>,
        cont: Box<Continuation>,
    },

    Seq {
        remaining: Vec<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },

    Car {
        cont: Box<Continuation>,
        env: Env,
    },
    Cdr {
        cont: Box<Continuation>,
        env: Env,
    },

    ConsLeft {
        right: Box<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },
    ConsRight {
        left_val: SharedValue,
        cont: Box<Continuation>,
        env: Env,
    },

    CallEvalOperator {
        args: Vec<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },
    CallEvalArgs {
        operator: SharedValue,
        remaining: Vec<Expr>,
        mut_collected: Vec<SharedValue>,
        cont: Box<Continuation>,
        env: Env,
    },

    If {
        then_e: Box<Expr>,
        else_e: Box<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },
}

enum Task {
    Eval(Expr, Continuation, Env),
    Apply(Continuation, SharedValue, Env),
}

const MAX_FRAMES: usize = 1_000_000;

pub fn reduce_expression(expr: Expr, cont: Continuation, env: Env) -> SharedValue {
    use Task::*;

    let mut stack: Vec<Task> = Vec::new();
    stack.push(Eval(expr.clone(), cont, env));
    let mut frames: usize = 0;

    // tracing setup: if top-level expr mentions "fib", enable trace capture
    let trace_enabled = expr_contains_name(&expr, "fib");
    const TRACE_LIMIT: usize = 500;
    let mut trace: Vec<String> = Vec::new();

    while let Some(task) = stack.pop() {
        frames = frames.saturating_add(1);
        if frames > MAX_FRAMES {
            if trace_enabled {
                eprintln!("--- trace (first {} events) ---", trace.len());
                for (i, r) in trace.iter().enumerate() {
                    eprintln!("{:05}: {}", i, r);
                }
            }
            panic!("evaluation exceeded max frames ({})", MAX_FRAMES);
        }
        if frames % 10000 == 0 {
            eprintln!("reduce_expression frames = {}", frames);
        }

        if trace_enabled && trace.len() < TRACE_LIMIT {
            match &task {
                Eval(e, k, _env) => {
                    trace.push(format!("Popped Eval({}, cont={})", expr_short(e), cont_short(k)));
                }
                Apply(k, v, _env) => {
                    let vdesc = match v {
                        SharedValue::Num(n) => format!("Num({})", n),
                        SharedValue::List(l) => format!("List(len={})", l.len()),
                        SharedValue::Closure(_, _, _) => "Closure".to_string(),
                        SharedValue::Bool(b) => format!("Bool({})", b),
                        SharedValue::Str(s) => format!("Str(len={})", s.len()),
                        SharedValue::Char(c) => format!("Char({})", c),
                        SharedValue::Share(sr) => format!("Share(id={}, nbits={})", sr.id, sr.nbits),
                    };
                    trace.push(format!("Popped Apply(cont={}, val={})", cont_short(k), vdesc));
                }
            }
        }

        match task {
            Eval(e, k, eenv) => {
                match e {
                    Expr::Num(n) => {
                        stack.push(Apply(k, SharedValue::Num(n), eenv.clone()));
                    }
                    Expr::Str(s) => {
                        stack.push(Apply(k, SharedValue::Str(s), eenv.clone()));
                    }
                    Expr::Char(c) => {
                        stack.push(Apply(k, SharedValue::Char(c), eenv.clone()));
                    }
                    Expr::Bool(b) => {
                        stack.push(Apply(k, SharedValue::Bool(b), eenv.clone()));
                    }
                    Expr::UnOp(op, inner) => {
                        let new_cont = Continuation::UnOp { op, cont: Box::new(k), env: eenv.clone() };
                        stack.push(Eval(*inner, new_cont, eenv.clone()));
                    }
                    Expr::Var(name) => {
                        let map = eenv.borrow();
                        let ptr = map.get(&name).cloned().unwrap_or_else(|| panic!("unbound variable: {}", name));
                        let v = ptr.borrow().clone();
                        stack.push(Apply(k, v, eenv.clone()));
                    }
                    Expr::BinOp(op, left, right) => {
                        let new_cont = Continuation::BinOpLeft { op, right, cont: Box::new(k), env: eenv.clone() };
                        stack.push(Eval(*left, new_cont, eenv.clone()));
                    }
                    Expr::Let(bindings, body) => {
                        if body.is_empty() {
                            if bindings.is_empty() {
                                stack.push(Apply(k, SharedValue::Num(0), eenv.clone()));
                            } else {
                                let mut it = bindings.into_iter();
                                let (name, be) = it.next().unwrap();
                                let remaining: Vec<(String, Expr)> = it.collect();
                                stack.push(Eval(be, Continuation::LetTop { name, remaining, env: eenv.clone() }, eenv.clone()));
                            }
                        } else {
                            let local = Rc::new(RefCell::new(eenv.borrow().clone()));
                            if bindings.is_empty() {
                                if body.len() == 1 {
                                    stack.push(Eval(body.into_iter().next().unwrap(), k, local));
                                } else {
                                    let mut rest = body;
                                    let last = rest.pop().unwrap();
                                    stack.push(Eval(last, k, local.clone()));
                                    while let Some(pre) = rest.pop() {
                                        stack.push(Eval(pre, Continuation::Halt, local.clone()));
                                    }
                                }
                            } else {
                                let mut it = bindings.into_iter();
                                let (name, be) = it.next().unwrap();
                                let remaining: Vec<(String, Expr)> = it.collect();
                                stack.push(Eval(be, Continuation::LetLocal { name, remaining, local_env: local.clone(), body: body.clone(), cont: Box::new(k) }, local.clone()));
                            }
                        }
                    }
                    Expr::LetRec(bindings, body) => {
                        let local = Rc::new(RefCell::new(eenv.borrow().clone()));
                        for (name, _) in &bindings {
                            local.borrow_mut().insert(name.clone(), Rc::new(RefCell::new(SharedValue::Num(0))));
                        }
                        if bindings.is_empty() {
                            if body.len() == 1 {
                                stack.push(Eval(body.into_iter().next().unwrap(), k, local));
                            } else {
                                let mut rest = body;
                                let last = rest.pop().unwrap();
                                stack.push(Eval(last, k, local.clone()));
                                while let Some(pre) = rest.pop() {
                                    stack.push(Eval(pre, Continuation::Halt, local.clone()));
                                }
                            }
                        } else {
                            let mut it = bindings.into_iter();
                            let (name, be) = it.next().unwrap();
                            let remaining: Vec<(String, Expr)> = it.collect();
                            stack.push(Eval(be, Continuation::LetRecInit { name, remaining, local_env: local.clone(), body: body.clone(), cont: Box::new(k) }, local.clone()));
                        }
                    }
                    Expr::Seq(exprs) => {
                        if exprs.is_empty() {
                            stack.push(Apply(k, SharedValue::List(Vec::new()), eenv.clone()));
                        } else if exprs.len() == 1 {
                            stack.push(Eval(exprs.into_iter().next().unwrap(), k, eenv));
                        } else {
                            let mut rest = exprs;
                            let last = rest.pop().unwrap();
                            stack.push(Eval(last, k, eenv.clone()));
                            while let Some(pre) = rest.pop() {
                                stack.push(Eval(pre, Continuation::Halt, eenv.clone()));
                            }
                        }
                    }
                    Expr::Car(inner) => {
                        stack.push(Eval(*inner, Continuation::Car { cont: Box::new(k), env: eenv.clone() }, eenv.clone()));
                    }
                    Expr::Cdr(inner) => {
                        stack.push(Eval(*inner, Continuation::Cdr { cont: Box::new(k), env: eenv.clone() }, eenv.clone()));
                    }
                    Expr::Cons(h, t) => {
                        stack.push(Eval(*h, Continuation::ConsLeft { right: t, cont: Box::new(k), env: eenv.clone() }, eenv.clone()));
                    }
                    Expr::Lambda(params, body) => {
                        let closure = SharedValue::Closure(params, body, eenv.clone());
                        stack.push(Apply(k, closure, eenv.clone()));
                    }
                    Expr::Call(head, args) => {
                        stack.push(Eval(*head, Continuation::CallEvalOperator { args, cont: Box::new(k), env: eenv.clone() }, eenv.clone()));
                    }
                    Expr::If(cond, then_e, else_e) => {
                        stack.push(Eval(*cond, Continuation::If { then_e, else_e, cont: Box::new(k), env: eenv.clone() }, eenv.clone()));
                    }
                }
            }
            Apply(k, val, _env) => {
                match k {
                    Continuation::Halt => {
                        if stack.is_empty() {
                            return val;
                        } else {
                            continue;
                        }
                    }
                    Continuation::BinOpLeft { op, right, cont, env } => {
                        let next = Continuation::BinOpRight { op, left_val: val, cont, env: env.clone() };
                        stack.push(Eval(*right, next, env.clone()));
                    }
                    Continuation::BinOpRight { op, left_val, cont, env } => {
                        // helper to view a value as an i64 when reasonable (Num or Bool)
                        let maybe_num = |v: &SharedValue| -> Option<i64> {
                            match v {
                                SharedValue::Num(n) => Some(*n),
                                SharedValue::Bool(b) => Some(if *b { 1 } else { 0 }),
                                _ => None,
                            }
                        };

                        let out = match op {
                            super::parser::BinOp::Add | super::parser::BinOp::Sub | super::parser::BinOp::Mul | super::parser::BinOp::DivExact | super::parser::BinOp::Mod => {
                                // if either operand is a shared value, dispatch to backend
                                match (&left_val, &val) {
                                    (SharedValue::Share(sa), SharedValue::Share(sb)) => {
                                        let mut be = SimBackend::new();
                                        let sr = match op {
                                            super::parser::BinOp::Add => be.add(&sa, &sb),
                                            super::parser::BinOp::Sub => be.sub(&sa, &sb),
                                            super::parser::BinOp::Mul => be.mul(&sa, &sb),
                                            _ => panic!("unsupported shared op"),
                                        };
                                        SharedValue::Share(sr)
                                    }
                                    (SharedValue::Share(sa), SharedValue::Num(bn)) => {
                                        let mut be = SimBackend::new();
                                        let sb = be.share_secret(*bn as u128, sa.nbits);
                                        let sr = match op {
                                            super::parser::BinOp::Add => be.add(&sa, &sb),
                                            super::parser::BinOp::Sub => be.sub(&sa, &sb),
                                            super::parser::BinOp::Mul => be.mul(&sa, &sb),
                                            _ => panic!("unsupported shared op"),
                                        };
                                        SharedValue::Share(sr)
                                    }
                                    (SharedValue::Num(an), SharedValue::Share(sb)) => {
                                        let mut be = SimBackend::new();
                                        let sa = be.share_secret(*an as u128, sb.nbits);
                                        let sr = match op {
                                            super::parser::BinOp::Add => be.add(&sa, &sb),
                                            super::parser::BinOp::Sub => be.sub(&sa, &sb),
                                            super::parser::BinOp::Mul => be.mul(&sa, &sb),
                                            _ => panic!("unsupported shared op"),
                                        };
                                        SharedValue::Share(sr)
                                    }
                                    _ => {
                                        let a = maybe_num(&left_val).expect("binary arithmetic left operand must be NUM or BOOL");
                                        let b = maybe_num(&val).expect("binary arithmetic right operand must be NUM or BOOL");
                                        let r = match op {
                                            super::parser::BinOp::Add => a.wrapping_add(b),
                                            super::parser::BinOp::Sub => a.wrapping_sub(b),
                                            super::parser::BinOp::Mul => a.wrapping_mul(b),
                                            super::parser::BinOp::DivExact => {
                                                if b == 0 { panic!("division by zero"); }
                                                if a % b != 0 { panic!("non-exact division: {} / {}", a, b); }
                                                a / b
                                            }
                                            super::parser::BinOp::Mod => {
                                                if b == 0 { panic!("modulo by zero"); }
                                                a % b
                                            }
                                            _ => unreachable!(),
                                        };
                                        SharedValue::Num(r)
                                    }
                                }
                            }
                            super::parser::BinOp::Eq => {
                                let eq = match (&left_val, &val) {
                                    (SharedValue::Num(a), SharedValue::Num(b)) => a == b,
                                    (SharedValue::Bool(a), SharedValue::Bool(b)) => a == b,
                                    (SharedValue::Str(a), SharedValue::Str(b)) => a == b,
                                    (SharedValue::Char(a), SharedValue::Char(b)) => a == b,
                                    _ => false,
                                };
                                SharedValue::Num(if eq { 1 } else { 0 })
                            }
                            super::parser::BinOp::Lt | super::parser::BinOp::Gt | super::parser::BinOp::Le | super::parser::BinOp::Ge => {
                                let a = maybe_num(&left_val).expect("comparison left operand must be NUM or BOOL");
                                let b = maybe_num(&val).expect("comparison right operand must be NUM or BOOL");
                                let r = match op {
                                    super::parser::BinOp::Lt => a < b,
                                    super::parser::BinOp::Gt => a > b,
                                    super::parser::BinOp::Le => a <= b,
                                    super::parser::BinOp::Ge => a >= b,
                                    _ => unreachable!(),
                                };
                                SharedValue::Num(if r { 1 } else { 0 })
                            }
                            super::parser::BinOp::And => {
                                match (&left_val, &val) {
                                    (SharedValue::Num(a), SharedValue::Num(b)) => SharedValue::Num(a & b),
                                    _ => {
                                        let a = maybe_num(&left_val).expect("and expects NUM/BOOL");
                                        let b = maybe_num(&val).expect("and expects NUM/BOOL");
                                        SharedValue::Num(if a != 0 && b != 0 { 1 } else { 0 })
                                    }
                                }
                            }
                            super::parser::BinOp::Or => {
                                match (&left_val, &val) {
                                    (SharedValue::Num(a), SharedValue::Num(b)) => SharedValue::Num(a | b),
                                    _ => {
                                        let a = maybe_num(&left_val).expect("or expects NUM/BOOL");
                                        let b = maybe_num(&val).expect("or expects NUM/BOOL");
                                        SharedValue::Num(if a != 0 || b != 0 { 1 } else { 0 })
                                    }
                                }
                            }
                        };
                        stack.push(Apply(*cont, out, env.clone()));
                    }
                    Continuation::UnOp { op, cont, env } => {
                        let out = match op {
                            super::parser::UnOp::Not => match val {
                                SharedValue::Num(n) => SharedValue::Num(if n == 0 { 1 } else { 0 }),
                                SharedValue::Bool(b) => SharedValue::Num(if !b { 1 } else { 0 }),
                                _ => panic!("not expects numeric or boolean value"),
                            },
                            super::parser::UnOp::Neg => match val {
                                SharedValue::Num(n) => SharedValue::Num(n.wrapping_neg()),
                                _ => panic!("negation expects numeric value"),
                            },
                        };
                        stack.push(Apply(*cont, out, env.clone()));
                    }
                    Continuation::LetTop { name, mut remaining, env } => {
                        env.borrow_mut().insert(name, Rc::new(RefCell::new(val)));
                        if remaining.is_empty() {
                            stack.push(Apply(Continuation::Halt, SharedValue::Num(0), env.clone()));
                        } else {
                            let (nxt_name, nxt_expr) = remaining.remove(0);
                            stack.push(Eval(nxt_expr, Continuation::LetTop { name: nxt_name, remaining, env: env.clone() }, env.clone()));
                        }
                    }
                    Continuation::LetLocal { name, mut remaining, local_env, body, cont } => {
                        local_env.borrow_mut().insert(name, Rc::new(RefCell::new(val)));
                        if remaining.is_empty() {
                            if body.is_empty() {
                                stack.push(Apply(*cont, SharedValue::Num(0), local_env.clone()));
                            } else if body.len() == 1 {
                                stack.push(Eval(body.into_iter().next().unwrap(), *cont, local_env.clone()));
                            } else {
                                let mut rest = body;
                                let last = rest.pop().unwrap();
                                stack.push(Eval(last, *cont, local_env.clone()));
                                while let Some(pre) = rest.pop() {
                                    stack.push(Eval(pre, Continuation::Halt, local_env.clone()));
                                }
                            }
                        } else {
                            let (nxt_name, nxt_expr) = remaining.remove(0);
                            stack.push(Eval(nxt_expr, Continuation::LetLocal { name: nxt_name, remaining, local_env: local_env.clone(), body, cont }, local_env.clone()));
                        }
                    }
                    Continuation::LetRecInit { name, mut remaining, local_env, body, cont } => {
                        if let Some(ptr) = local_env.borrow().get(&name) {
                            *ptr.borrow_mut() = val;
                        } else {
                            local_env.borrow_mut().insert(name.clone(), Rc::new(RefCell::new(val)));
                        }
                        if remaining.is_empty() {
                            if body.len() == 1 {
                                stack.push(Eval(body.into_iter().next().unwrap(), *cont, local_env.clone()));
                            } else {
                                let mut rest = body;
                                let last = rest.pop().unwrap();
                                stack.push(Eval(last, *cont, local_env.clone()));
                                while let Some(pre) = rest.pop() {
                                    stack.push(Eval(pre, Continuation::Halt, local_env.clone()));
                                }
                            }
                        } else {
                            let (nxt_name, nxt_expr) = remaining.remove(0);
                            stack.push(Eval(nxt_expr, Continuation::LetRecInit { name: nxt_name, remaining, local_env: local_env.clone(), body, cont: cont.clone() }, local_env.clone()));
                        }
                    }
                    Continuation::Seq { mut remaining, cont, env } => {
                        if remaining.is_empty() {
                            stack.push(Apply(*cont, val, env.clone()));
                        } else {
                            let next = remaining.remove(0);
                            if remaining.is_empty() {
                                stack.push(Eval(next, *cont, env.clone()));
                            } else {
                                stack.push(Eval(next, Continuation::Seq { remaining, cont, env: env.clone() }, env.clone()));
                            }
                        }
                    }
                    Continuation::Car { cont, env } => {
                        match val {
                            SharedValue::List(mut vs) => {
                                if vs.is_empty() { panic!("car of empty list"); }
                                let first = vs.remove(0);
                                stack.push(Apply(*cont, first, env.clone()));
                            }
                            _ => panic!("car expects a list"),
                        }
                    }
                    Continuation::Cdr { cont, env } => {
                        match val {
                            SharedValue::List(vs) => {
                                if vs.is_empty() { panic!("cdr of empty list"); }
                                let tail = SharedValue::List(vs.into_iter().skip(1).collect());
                                stack.push(Apply(*cont, tail, env.clone()));
                            }
                            _ => panic!("cdr expects a list"),
                        }
                    }
                    Continuation::ConsLeft { right, cont, env } => {
                        let next = Continuation::ConsRight { left_val: val, cont, env: env.clone() };
                        stack.push(Eval(*right, next, env.clone()));
                    }
                    Continuation::ConsRight { left_val, cont, env } => {
                        match val {
                            SharedValue::List(mut vs) => {
                                vs.insert(0, left_val);
                                stack.push(Apply(*cont, SharedValue::List(vs), env.clone()));
                            }
                            _ => panic!("cons expects second arg to be a list"),
                        }
                    }
                    Continuation::CallEvalOperator { args, cont, env } => {
                        if args.is_empty() {
                            match val {
                                SharedValue::Closure(params, body, closure_env) => {
                                    if params.len() != 0 {
                                        panic!("closure arity mismatch: expected {} got 0", params.len());
                                    }
                                    let call_env = Rc::new(RefCell::new(closure_env.borrow().clone()));
                                    stack.push(Eval(*body, *cont, call_env));
                                }
                                _ => panic!("attempt to call non-function: {:?}", val),
                            }
                        } else {
                            let mut rem = args;
                            let first = rem.remove(0);
                            let call_args_cont = Continuation::CallEvalArgs { operator: val, remaining: rem, mut_collected: Vec::new(), cont, env: env.clone() };
                            stack.push(Eval(first, call_args_cont, env.clone()));
                        }
                    }
                    Continuation::CallEvalArgs { operator, mut remaining, mut mut_collected, cont, env } => {
                        mut_collected.push(val);
                        if remaining.is_empty() {
                            match operator {
                                SharedValue::Closure(params, body, closure_env) => {
                                    if params.len() != mut_collected.len() {
                                        panic!("closure arity mismatch: expected {} got {}", params.len(), mut_collected.len());
                                    }
                                    let call_env = Rc::new(RefCell::new(closure_env.borrow().clone()));
                                    for (p, v) in params.into_iter().zip(mut_collected.into_iter()) {
                                        call_env.borrow_mut().insert(p, Rc::new(RefCell::new(v)));
                                    }
                                    stack.push(Eval(*body, *cont, call_env));
                                }
                                SharedValue::Share(_sref) => {
                                    // support calling (reconstruct shared) as (reconstruct s)
                                    // if the operator is the variable `reconstruct`, handle it
                                    // otherwise error
                                    // Note: previously we planned to register a 'reconstruct' builtin;
                                    // for now, detect when operator is a Var("reconstruct") wrapped as Closure sentinel.
                                    panic!("attempt to call a shared value as function");
                                }
                                _ => panic!("attempt to call non-function: {:?}", operator),
                            }
                        } else {
                            let next = remaining.remove(0);
                            let new_cont = Continuation::CallEvalArgs { operator, remaining, mut_collected, cont, env: env.clone() };
                            stack.push(Eval(next, new_cont, env.clone()));
                        }
                    }
                    Continuation::If { then_e, else_e, cont, env } => {
                        match val {
                            SharedValue::Num(n) => {
                                if n != 0 {
                                    stack.push(Eval(*then_e, *cont, env.clone()));
                                } else {
                                    stack.push(Eval(*else_e, *cont, env.clone()));
                                }
                            }
                            _ => panic!("if condition must be NUM"),
                        }
                    }
                 }
             }
         }
     }

     panic!("evaluation fell out of loop without producing a value");
 }
 
 // A simulated participant in the 3-party REPL.
 pub struct Participant {
     id: usize,
 }

 impl Participant {
     pub fn new(id: usize) -> Self {
         Participant { id }
     }

     /// Evaluate an expression locally and return the resulting SharedValue.
     pub fn eval(&self, expr: &Expr) -> SharedValue {
         let store: Env = Rc::new(RefCell::new(HashMap::new()));
         // register builtins: share and reconstruct and a2b/b2a
         let mut backend = SimBackend::new();
         // insert share as a native closure: (share value nbits)
         store.borrow_mut().insert("share".to_string(), Rc::new(RefCell::new(SharedValue::Closure(vec!["v".to_string(), "nb".to_string()], Box::new(Expr::Num(0)), store.clone()))));
         // reconstruct will be handled as a special var name 'reconstruct' in call dispatch below
         let cont = Continuation::Halt;
         let result = reduce_expression(expr.clone(), cont, store.clone());
         if !cfg!(test) {
             println!("participant {} evaluated expression -> {:?}", self.id, result);
         }
         result
     }
 }
 
 // Helpers
 fn expr_contains_name(e: &Expr, name: &str) -> bool {
     match e {
         Expr::UnOp(_, inner) => expr_contains_name(inner, name),
         Expr::Str(s) => s == name,
         Expr::Char(c) => c.to_string() == name,
         Expr::Bool(_) => false,
         Expr::Num(_) => false,
         Expr::Var(n) => n == name,
         Expr::BinOp(_, l, r) => expr_contains_name(l, name) || expr_contains_name(r, name),
         Expr::Let(bindings, body) | Expr::LetRec(bindings, body) => {
             bindings.iter().any(|(n, ex)| n == name || expr_contains_name(ex, name)) || body.iter().any(|ex| expr_contains_name(ex, name))
         }
         Expr::Seq(es) => es.iter().any(|ex| expr_contains_name(ex, name)),
         Expr::Car(i) | Expr::Cdr(i) => expr_contains_name(i, name),
         Expr::Cons(a, b) => expr_contains_name(a, name) || expr_contains_name(b, name),
         Expr::Lambda(params, body) => params.iter().any(|p| p == name) || expr_contains_name(body, name),
         Expr::Call(h, args) => expr_contains_name(h, name) || args.iter().any(|a| expr_contains_name(a, name)),
         Expr::If(c, t, e) => expr_contains_name(c, name) || expr_contains_name(t, name) || expr_contains_name(e, name),
     }
 }

 fn expr_short(e: &Expr) -> String {
     match e {
         Expr::UnOp(_, _) => "UnOp".to_string(),
         Expr::Str(s) => format!("Str(len={})", s.len()),
         Expr::Char(_) => "Char".to_string(),
         Expr::Bool(b) => format!("Bool({})", b),
         Expr::Num(n) => format!("Num({})", n),
         Expr::Var(s) => format!("Var({})", s),
         Expr::BinOp(_, _, _) => "BinOp".to_string(),
         Expr::Let(bs, _) => format!("Let(len={})", bs.len()),
         Expr::LetRec(bs, _) => format!("LetRec(len={})", bs.len()),
         Expr::Seq(es) => format!("Seq(len={})", es.len()),
         Expr::Car(_) => "Car".to_string(),
         Expr::Cdr(_) => "Cdr".to_string(),
         Expr::Cons(_, _) => "Cons".to_string(),
         Expr::Lambda(ps, _) => format!("Lambda(arity={})", ps.len()),
         Expr::Call(_, args) => format!("Call(arity={})", args.len()),
         Expr::If(_, _, _) => "If".to_string(),
     }
 }

 fn cont_short(c: &Continuation) -> String {
     match c {
         Continuation::Halt => "Halt".to_string(),
         Continuation::BinOpLeft { .. } => "BinOpLeft".to_string(),
         Continuation::BinOpRight { .. } => "BinOpRight".to_string(),
         Continuation::UnOp { .. } => "UnOp".to_string(),
         Continuation::LetTop { .. } => "LetTop".to_string(),
         Continuation::LetLocal { .. } => "LetLocal".to_string(),
         Continuation::LetRecInit { .. } => "LetRecInit".to_string(),
         Continuation::Seq { .. } => "SeqCont".to_string(),
         Continuation::Car { .. } => "CarCont".to_string(),
         Continuation::Cdr { .. } => "CdrCont".to_string(),
         Continuation::ConsLeft { .. } => "ConsLeft".to_string(),
         Continuation::ConsRight { .. } => "ConsRight".to_string(),
         Continuation::CallEvalOperator { .. } => "CallEvalOperator".to_string(),
         Continuation::CallEvalArgs { .. } => "CallEvalArgs".to_string(),
         Continuation::If { .. } => "IfCont".to_string(),
     }
 }
