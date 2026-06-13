use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use serde::{Serialize, Deserialize};

use crate::interpreter::ast::{Expr, Env, SharedValue};
use crate::interpreter::backend::{Aby3Backend, Backend, SimBackend};
use colored::*;

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
    /// After evaluating a branch of an `if` whose condition was a shared boolean,
    /// normalize a numeric result into a share with the provided (owner, nbits).
    IfResultNormalize {
        cont: Box<Continuation>,
        normalize: Option<(usize, usize)>, // Some(owner, nbits) -> convert NUM -> SHARE
        env: Env,
    },
}

enum Task {
    Eval(Expr, Continuation, Env),
    Apply(Continuation, SharedValue, Env),
}

const MAX_FRAMES: usize = 1_000_000;

// Note: added `owner` parameter so reduce_expression can pass participant id
// into share_secret and reveal operations.
pub fn reduce_expression(expr: Expr, cont: Continuation, env: Env, backend: Arc<Mutex<Box<dyn Backend>>>, _owner: usize, verbose: bool) -> SharedValue {
    use Task::*;

    let mut stack: Vec<Task> = Vec::new();
    stack.push(Eval(expr.clone(), cont, env));
    let mut frames: usize = 0;

    // tracing setup: enable when verbose is requested or top-level expr mentions "fib"
    let trace_enabled = verbose || expr_contains_name(&expr, "fib");
    const TRACE_LIMIT: usize = 500;
    let mut trace: Vec<String> = Vec::new();

    while let Some(task) = stack.pop() {
        frames = frames.saturating_add(1);
        if frames > MAX_FRAMES {
            if trace_enabled {
                eprintln!("{}", format!("--- trace (first {} events) ---", trace.len()).bright_red());
                for (i, r) in trace.iter().enumerate() {
                    eprintln!("{:05}: {}", i, r.bright_blue());
                }
            }
            panic!("{}", format!("evaluation exceeded max frames ({})", MAX_FRAMES).bright_red());
        }
        if frames % 10000 == 0 {
            eprintln!("{}", format!("reduce_expression frames = {}", frames).bright_yellow());
        }

        if trace_enabled && trace.len() < TRACE_LIMIT {
            // Pretty / colored printers for environment, Expr and SharedValue used in traces
            fn env_pretty(env: &Env, max_entries: usize) -> String {
                use std::cmp::min;
                let map = env.borrow();
                if map.is_empty() { return "{}".to_string(); }
                // collect and sort keys for stable output
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut parts: Vec<String> = Vec::new();
                for k in keys.iter().take(min(max_entries, keys.len())) {
                    if let Some(vptr) = map.get(*k) {
                        // borrow the inner SharedValue for read-only display
                        let vref = vptr.borrow();
                        let sval = colored_value(&*vref);
                        parts.push(format!("{}: {}", k.bright_blue(), sval));
                    }
                }
                let mut s = String::new();
                s.push('{');
                if !parts.is_empty() {
                    s.push('\n');
                    for p in parts.iter() {
                        s.push_str("  ");
                        s.push_str(p);
                        s.push_str(",\n");
                    }
                    s.push('}');
                } else {
                    s.push('}');
                }
                if map.len() > max_entries {
                    s.push_str(&format!("  ... {} more fields", map.len() - max_entries));
                }
                s
            }

            fn colored_expr(e: &Expr) -> String {
                match e {
                    Expr::Num(n) => format!("{}", format!("Num({})", n).green()),
                    Expr::Var(s) => format!("{}", format!("Var({})", s).cyan()),
                    Expr::Str(s) => format!("{}", format!("Str(\"{}\")", s).magenta()),
                    Expr::Char(c) => format!("{}", format!("Char({})", c).magenta()),
                    Expr::Bool(b) => format!("{}", format!("Bool({})", b).yellow()),
                    Expr::BinOp(_, _, _) => format!("{}", expr_short(e).blue()),
                    Expr::UnOp(_, _) => format!("{}", expr_short(e).blue()),
                    Expr::Let(bs, _) => format!("{}", format!("Let(len={})", bs.len()).blue()),
                    Expr::LetRec(bs, _) => format!("{}", format!("LetRec(len={})", bs.len()).blue()),
                    Expr::Seq(es) => format!("{}", format!("Seq(len={})", es.len()).blue()),
                    Expr::Car(_) => "Car".blue().to_string(),
                    Expr::Cdr(_) => "Cdr".blue().to_string(),
                    Expr::Cons(_, _) => "Cons".blue().to_string(),
                    Expr::Lambda(ps, _) => format!("{}", format!("Lambda(arity={})", ps.len()).magenta()),
                    Expr::Call(_, args) => format!("{}", format!("Call(arity={})", args.len()).magenta()),
                    Expr::If(_, _, _) => "If".blue().to_string(),
                }
            }

            fn colored_value(v: &SharedValue) -> String {
                match v {
                    SharedValue::Num(n) => format!("{}", format!("Num({})", n).green()),
                    SharedValue::Bool(b) => format!("{}", format!("Bool({})", b).yellow()),
                    SharedValue::Str(s) => format!("{}", format!("Str(\"{}\")", s).magenta()),
                    SharedValue::Char(c) => format!("{}", format!("Char({})", c).magenta()),
                    SharedValue::List(l) => format!("{}", format!("List(len={})", l.len()).blue()),
                    SharedValue::Closure(_, _, _) => "Closure".magenta().to_string(),
                    SharedValue::Share(sr) => format!("{}", format!("Share(id={},nbits={},owner={})", sr.id, sr.nbits, sr.owner).red()),
                    SharedValue::NativeFunc(_) => "NativeFunc".cyan().to_string(),
                }
            }

            match &task {
                Eval(e, _k, env_snapshot) => {
                    // quieter eval trace: show expression and a short pretty env
                    let envs = env_pretty(env_snapshot, 6);
                    trace.push(format!("{}: {}  env={}",
                        "Eval".green(),
                        colored_expr(e),
                        envs
                    ));
                }
                Apply(k, v, _env_snapshot) => {
                    // quieter apply trace: show continuation and the single colored value
                    let vcol = colored_value(v);
                    trace.push(format!("{}: cont={} val={}",
                        "Apply".yellow(),
                        cont_short(k).bright_black(),
                        vcol
                    ));
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
                            if trace_enabled && !trace.is_empty() {
                                eprintln!("{}", format!("--- verbose trace (most recent {} events) ---", trace.len()).bright_yellow());
                                for (i, r) in trace.iter().enumerate() {
                                    eprintln!("{:05}: {}", i, r.bright_blue());
                                }
                            }
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
                                        let mut be = backend.lock().unwrap();
                                        let sr = match op {
                                            super::parser::BinOp::Add => be.add(&sa, &sb),
                                            super::parser::BinOp::Sub => be.sub(&sa, &sb),
                                            super::parser::BinOp::Mul => be.mul(&sa, &sb),
                                            _ => panic!("unsupported shared op"),
                                        };
                                        SharedValue::Share(sr)
                                    }
                                    (SharedValue::Share(sa), SharedValue::Num(bn)) => {
                                        let mut be = backend.lock().unwrap();
                                        // convert signed immediate into two's-complement field element for sa.nbits
                                        let imm = int_to_fr_signed(*bn as i64, sa.nbits);
                                        let sb = be.share_secret(imm, sa.nbits, 0);
                                        let sr = match op {
                                            super::parser::BinOp::Add => be.add(&sa, &sb),
                                            super::parser::BinOp::Sub => be.sub(&sa, &sb),
                                            super::parser::BinOp::Mul => be.mul(&sa, &sb),
                                            _ => panic!("unsupported shared op"),
                                        };
                                        SharedValue::Share(sr)
                                    }
                                    (SharedValue::Num(an), SharedValue::Share(sb)) => {
                                        let mut be = backend.lock().unwrap();
                                        // convert signed immediate into two's-complement field element for sb.nbits
                                        let imm = int_to_fr_signed(*an as i64, sb.nbits);
                                        let sa = be.share_secret(imm, sb.nbits, 0);
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
                                match (&left_val, &val) {
                                    (SharedValue::Num(a), SharedValue::Num(b)) => SharedValue::Num(if a == b { 1 } else { 0 }),
                                    (SharedValue::Bool(a), SharedValue::Bool(b)) => SharedValue::Num(if a == b { 1 } else { 0 }),
                                    (SharedValue::Str(a), SharedValue::Str(b)) => SharedValue::Num(if a == b { 1 } else { 0 }),
                                    (SharedValue::Char(a), SharedValue::Char(b)) => SharedValue::Num(if a == b { 1 } else { 0 }),
                                    (SharedValue::Share(sa), SharedValue::Share(sb)) => {
                                        let mut be = backend.lock().unwrap();
                                        SharedValue::Share(be.eq(&sa, &sb))
                                    }
                                    (SharedValue::Share(sa), SharedValue::Num(bn)) => {
                                        let mut be = backend.lock().unwrap();
                                        let imm = int_to_fr_signed(*bn as i64, sa.nbits);
                                        let sb = be.share_secret(imm, sa.nbits, 0);
                                        SharedValue::Share(be.eq(&sa, &sb))
                                    }
                                    (SharedValue::Num(an), SharedValue::Share(sb)) => {
                                        let mut be = backend.lock().unwrap();
                                        let imm = int_to_fr_signed(*an as i64, sb.nbits);
                                        let sa = be.share_secret(imm, sb.nbits, 0);
                                        SharedValue::Share(be.eq(&sa, &sb))
                                    }
                                    _ => SharedValue::Num(0),
                                }
                            }
                            super::parser::BinOp::Lt | super::parser::BinOp::Gt | super::parser::BinOp::Le | super::parser::BinOp::Ge => {
                                match (&left_val, &val) {
                                    (SharedValue::Num(a), SharedValue::Num(b)) => {
                                        let r = match op {
                                            super::parser::BinOp::Lt => a < b,
                                            super::parser::BinOp::Gt => a > b,
                                            super::parser::BinOp::Le => a <= b,
                                            super::parser::BinOp::Ge => a >= b,
                                            _ => unreachable!(),
                                        };
                                        SharedValue::Num(if r { 1 } else { 0 })
                                    }
                                    (SharedValue::Share(sa), SharedValue::Share(sb)) => {
                                        let mut be = backend.lock().unwrap();
                                        let out = match op {
                                            super::parser::BinOp::Lt => be.lt(&sa, &sb),
                                            super::parser::BinOp::Gt => be.lt(&sb, &sa),
                                            super::parser::BinOp::Le => be.le(&sa, &sb),
                                            super::parser::BinOp::Ge => be.le(&sb, &sa),
                                            _ => unreachable!(),
                                        };
                                        SharedValue::Share(out)
                                    }
                                    (SharedValue::Share(sa), SharedValue::Num(bn)) => {
                                        let mut be = backend.lock().unwrap();
                                        let imm = int_to_fr_signed(*bn as i64, sa.nbits);
                                        let sb = be.share_secret(imm, sa.nbits, 0);
                                        let out = match op {
                                            super::parser::BinOp::Lt => be.lt(&sa, &sb),
                                            super::parser::BinOp::Gt => be.lt(&sb, &sa),
                                            super::parser::BinOp::Le => be.le(&sa, &sb),
                                            super::parser::BinOp::Ge => be.le(&sb, &sa),
                                            _ => unreachable!(),
                                        };
                                        SharedValue::Share(out)
                                    }
                                    (SharedValue::Num(an), SharedValue::Share(sb)) => {
                                        let mut be = backend.lock().unwrap();
                                        let imm = int_to_fr_signed(*an as i64, sb.nbits);
                                        let sa = be.share_secret(imm, sb.nbits, 0);
                                        let out = match op {
                                            super::parser::BinOp::Lt => be.lt(&sa, &sb),
                                            super::parser::BinOp::Gt => be.lt(&sb, &sa),
                                            super::parser::BinOp::Le => be.le(&sa, &sb),
                                            super::parser::BinOp::Ge => be.le(&sb, &sa),
                                            _ => unreachable!(),
                                        };
                                        SharedValue::Share(out)
                                    }
                                    _ => SharedValue::Num(0),
                                }
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
                                    panic!("attempt to call a shared value as function");
                                }
                                SharedValue::NativeFunc(f) => {
                                    // native function: apply with left-to-right arguments
                                    let all_args = mut_collected;
                                    match f(all_args) {
                                        v @ SharedValue::Num(_) => {
                                            stack.push(Apply(*cont, v, env.clone()));
                                        }
                                        SharedValue::Share(sr) => {
                                            stack.push(Apply(*cont, SharedValue::Share(sr), env.clone()));
                                        }
                                        other => {
                                            // allow NativeFunc to return other SharedValue variants
                                            stack.push(Apply(*cont, other, env.clone()));
                                        }
                                    }
                                }
                                _ => panic!("attempt to call non-function"),
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
                            SharedValue::Bool(b) => {
                                if b {
                                    stack.push(Eval(*then_e, *cont, env.clone()));
                                } else {
                                    stack.push(Eval(*else_e, *cont, env.clone()));
                                }
                            }
                            SharedValue::Share(sr) => {
                                // reveal the boolean share to all parties so they all
                                // take the same control-flow branch (leaks 1 bit)
                                let revealed = {
                                    let mut be = backend.lock().unwrap();
                                    be.reveal(&sr, None)
                                };
                                let cond = revealed != 0;

                                // When condition is shared, normalize branch results into a share
                                // so downstream computations keep operating on shared values.
                                let normalize = Some((sr.owner, sr.nbits));
                                let new_cont = Continuation::IfResultNormalize {
                                    cont,
                                    normalize,
                                    env: env.clone(),
                                };

                                if cond {
                                    stack.push(Eval(*then_e, new_cont, env.clone()));
                                } else {
                                    stack.push(Eval(*else_e, new_cont, env.clone()));
                                }
                            }
                            _ => panic!("if condition must be NUM or shared boolean"),
                        }
                    }

                    Continuation::IfResultNormalize { cont, normalize, env } => {
                        // When a branch result is produced, optionally normalize NUM->SHARE
                        match normalize {
                            Some((_norm_owner, norm_nbits)) => {
                                match val {
                                    SharedValue::Share(_) => {
                                        // already a share, just continue
                                        stack.push(Apply(*cont, val, env.clone()));
                                    }
                                    SharedValue::Num(n) => {
                                        let mut be = backend.lock().unwrap();
                                        let fr = int_to_fr_signed(n as i64, norm_nbits);
                                        let sref = be.share_secret(fr, norm_nbits, 0);
                                        stack.push(Apply(*cont, SharedValue::Share(sref), env.clone()));
                                    }
                                    SharedValue::Bool(b) => {
                                        let mut be = backend.lock().unwrap();
                                        let fr = if b { 1u128 } else { 0u128 };
                                        let sref = be.share_secret(fr, norm_nbits, 0);
                                        stack.push(Apply(*cont, SharedValue::Share(sref), env.clone()));
                                    }
                                    _ => {
                                        // other types not supported for normalization
                                        panic!("cannot normalize branch result to share: {:?}", val);
                                    }
                                }
                            }
                            None => {
                                // no normalization needed, just continue
                                stack.push(Apply(*cont, val, env.clone()));
                            }
                        }
                    }
                 }
             }
         }
     }

     panic!("evaluation fell out of loop without producing a value");
 }
 
// A participant in the 3-party REPL.
 pub struct Participant {
    id: usize,
    backend: Arc<Mutex<Box<dyn Backend>>>,
    store: Env,
}

#[derive(Serialize, Deserialize)]
enum PersistValue {
    Num(i64),
    Bool(bool),
    Str(String),
    Char(char),
    List(Vec<PersistValue>),
}

 impl Participant {
     pub fn new(id: usize) -> Self {
         let backend: Arc<Mutex<Box<dyn Backend>>> = Arc::new(Mutex::new(Box::new(Aby3Backend::new())));
         let store: Env = Rc::new(RefCell::new(HashMap::new()));
         let participant = Participant { id, backend: backend.clone(), store: store.clone() };
         // register builtins once into the persistent store
         participant.register_builtins();
         participant
     }

     pub fn new_sim(id: usize) -> Self {
         let backend: Arc<Mutex<Box<dyn Backend>>> = Arc::new(Mutex::new(Box::new(SimBackend::new())));
         let store: Env = Rc::new(RefCell::new(HashMap::new()));
         let participant = Participant { id, backend: backend.clone(), store: store.clone() };
         // register builtins once into the persistent store
         participant.register_builtins();
         participant
     }

     /// Create a participant backed by a real `TransportBackend` using an
     /// existing `Transport` connection (e.g. `TcpTransport` or `InprocHandle`).
     pub fn new_transport<T: crate::crypto::net::Transport + 'static>(id: usize, transport: T) -> Self {
         use crate::interpreter::backend::TransportBackend;
         let backend: Arc<Mutex<Box<dyn Backend>>> = Arc::new(Mutex::new(Box::new(TransportBackend::new(id, transport))));
         let store: Env = Rc::new(RefCell::new(HashMap::new()));
         let participant = Participant { id, backend: backend.clone(), store: store.clone() };
         participant.register_builtins();
         participant
     }

     fn register_builtins(&self) {
        let backend = self.backend.clone();
        let _owner = self.id;

        // share builtin
        let share_backend = backend.clone();
        let share_fn = Arc::new(move |args: Vec<SharedValue>| -> SharedValue {
            if args.len() != 2 { panic!("share expects 2 args"); }
            match (&args[0], &args[1]) {
                (SharedValue::Num(n), SharedValue::Num(nb)) => {
                    let mut be = share_backend.lock().unwrap();
                    // convert signed immediate into two's-complement field element for nb bits
                    let imm = int_to_fr_signed(*n as i64, *nb as usize);
                    // Always pass the actual value. SimBackend uses it directly.
                    // TransportBackend passes it to protocol::share_input which ignores
                    // it for non-owner parties, so the owner check is enforced there.
                    let canonical_owner = 0usize;
                    let sref = be.share_secret(imm, *nb as usize, canonical_owner);
                    SharedValue::Share(sref)
                }
                (SharedValue::Share(s), SharedValue::Num(_nb)) => SharedValue::Share(s.clone()),
                _ => panic!("share expects (num nbits) or (share nbits)"),
            }
        });
        self.store.borrow_mut().insert("share".to_string(), Rc::new(RefCell::new(SharedValue::NativeFunc(share_fn))));

        // share-from builtin: (share-from owner-id value nbits)
        // Like (share value nbits) but the owner is explicitly specified, so each
        // party can provide their own private input.
        let share_from_backend = backend.clone();
        let share_from_fn = Arc::new(move |args: Vec<SharedValue>| -> SharedValue {
            if args.len() != 3 { panic!("share-from expects 3 args: (share-from owner-id value nbits)"); }
            match (&args[0], &args[1], &args[2]) {
                (SharedValue::Num(owner_id), SharedValue::Num(n), SharedValue::Num(nb)) => {
                    let mut be = share_from_backend.lock().unwrap();
                    let explicit_owner = *owner_id as usize;
                    let imm = int_to_fr_signed(*n as i64, *nb as usize);
                    // Same as share: pass the actual value; protocol::share_input enforces
                    // the owner check in distributed mode.
                    let sref = be.share_secret(imm, *nb as usize, explicit_owner);
                    SharedValue::Share(sref)
                }
                _ => panic!("share-from expects (owner-id value nbits) all as numbers"),
            }
        });
        self.store.borrow_mut().insert("share-from".to_string(), Rc::new(RefCell::new(SharedValue::NativeFunc(share_from_fn))));

        // reconstruct builtin
        let reconstruct_backend = backend.clone();
        let reconstruct_fn = Arc::new(move |args: Vec<SharedValue>| -> SharedValue {
            if args.len() != 1 { panic!("reconstruct expects 1 arg"); }
            match &args[0] {
                SharedValue::Share(sref) => {
                    let mut be = reconstruct_backend.lock().unwrap();
                    // reveal_to = None means broadcast: all parties participate and
                    // everyone reconstructs.  This is correct for both SimBackend
                    // and TransportBackend (in the single-party REPL, SimBackend
                    // ignores reveal_to anyway).
                    let plain = be.reveal(&sref, None);
                    SharedValue::Num(plain as i64)
                }
                SharedValue::Num(n) => SharedValue::Num(*n),
                _ => panic!("reconstruct expects a shared value"),
            }
        });
        self.store.borrow_mut().insert("reconstruct".to_string(), Rc::new(RefCell::new(SharedValue::NativeFunc(reconstruct_fn))));

        // a2b builtin
        let a2b_backend = backend.clone();
        let a2b_fn = Arc::new(move |args: Vec<SharedValue>| -> SharedValue {
            if args.len() != 2 { panic!("a2b expects 2 args: (a nbits)"); }
            match (&args[0], &args[1]) {
                (SharedValue::Share(sref), SharedValue::Num(nbits)) => {
                    let mut be = a2b_backend.lock().unwrap();
                    let bits = be.a2b(&sref, *nbits as usize);
                    let svs: Vec<SharedValue> = bits.into_iter().map(|b| SharedValue::Share(b)).collect();
                    SharedValue::List(svs)
                }
                _ => panic!("a2b expects (share nbits)"),
            }
        });
        self.store.borrow_mut().insert("a2b".to_string(), Rc::new(RefCell::new(SharedValue::NativeFunc(a2b_fn))));

        // b2a builtin
        let b2a_backend = backend.clone();
        let b2a_fn = Arc::new(move |args: Vec<SharedValue>| -> SharedValue {
            if args.len() != 1 { panic!("b2a expects 1 arg: (bits-list)"); }
            match &args[0] {
                SharedValue::List(list) => {
                    let bits_refs: Vec<crate::interpreter::backend::ShareRef> = list.iter().map(|v| {
                        if let SharedValue::Share(sr) = v { sr.clone() } else { panic!("b2a expects list of share bits") }
                    }).collect();
                    let mut be = b2a_backend.lock().unwrap();
                    let out = be.b2a(&bits_refs);
                    SharedValue::Share(out)
                }
                _ => panic!("b2a expects a list of bit shares"),
            }
        });
        self.store.borrow_mut().insert("b2a".to_string(), Rc::new(RefCell::new(SharedValue::NativeFunc(b2a_fn))));
    }

    /// Reveal a shared value to the caller (simulated reveal via backend).
    pub fn reveal(&self, sref: &crate::interpreter::backend::ShareRef) -> crate::crypto::types::Fr {
        let mut be = self.backend.lock().unwrap();
        be.reveal(sref, None)
    }

    /// Reset the persistent store (clears user bindings, re-registers builtins)
    pub fn reset_store(&self) {
        self.store.borrow_mut().clear();
        self.register_builtins();
    }

    /// Save the current store to a JSON file at `path`. Only simple values (num/bool/str/char/list)
    /// are persisted; closures and native functions are skipped.
    pub fn save_store<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let map = self.store.borrow();
        let mut out: std::collections::HashMap<String, PersistValue> = HashMap::new();
        for (k, vptr) in map.iter() {
            let v = vptr.borrow();
            if let Some(pv) = to_persist_value(&*v) {
                out.insert(k.clone(), pv);
            }
        }
        match serde_json::to_string_pretty(&out) {
            Ok(s) => {
                match File::create(path) {
                    Ok(mut f) => {
                        if let Err(e) = f.write_all(s.as_bytes()) { return Err(format!("write error: {}", e)); }
                        Ok(())
                    }
                    Err(e) => Err(format!("create file error: {}", e)),
                }
            }
            Err(e) => Err(format!("serialize error: {}", e)),
        }
    }

    /// Load a store from a JSON file, merging persisted bindings into the current store.
    /// Existing user bindings with the same name will be overwritten; builtins are preserved.
    pub fn load_store<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let mut s = String::new();
        match File::open(path) {
            Ok(mut f) => {
                if let Err(e) = f.read_to_string(&mut s) { return Err(format!("read error: {}", e)); }
            }
            Err(e) => return Err(format!("open error: {}", e)),
        }
        let parsed: std::result::Result<std::collections::HashMap<String, PersistValue>, _> = serde_json::from_str(&s);
        match parsed {
            Ok(m) => {
                for (k, pv) in m.into_iter() {
                    if let Some(sv) = from_persist_value(pv) {
                        self.store.borrow_mut().insert(k, Rc::new(RefCell::new(sv)));
                    }
                }
                Ok(())
            }
            Err(e) => Err(format!("parse error: {}", e)),
        }
    }

    /// Evaluate an expression using the participant's persistent store and return the resulting SharedValue.
    pub fn eval(&self, expr: &Expr) -> SharedValue {
        let owner = self.id;
        let cont = Continuation::Halt;
        let result = reduce_expression(expr.clone(), cont, self.store.clone(), self.backend.clone(), owner, false);
        if !cfg!(test) {
            println!("participant {} evaluated expression -> {:?}", self.id, result);
        }
        result
    }

    /// Like `eval` but never prints anything.  Used by the TCP party-0 REPL where
    /// output is handled by the caller with colours.
    pub fn eval_quiet(&self, expr: &Expr) -> SharedValue {
        let owner = self.id;
        let cont = Continuation::Halt;
        reduce_expression(expr.clone(), cont, self.store.clone(), self.backend.clone(), owner, false)
    }

    /// Like `eval_quiet` but also emits verbose frame-level tracing to stderr.
    pub fn eval_quiet_verbose(&self, expr: &Expr) -> SharedValue {
        let owner = self.id;
        let cont = Continuation::Halt;
        reduce_expression(expr.clone(), cont, self.store.clone(), self.backend.clone(), owner, true)
    }

    /// Evaluate with optional verbose tracing. When verbose=true the REPL will print
    /// per-frame environment, expression and continuation snapshots.
    pub fn eval_verbose(&self, expr: &Expr, verbose: bool) -> SharedValue {
        if verbose {
            println!("[participant {}] starting eval (verbose)", self.id);
        }
        let cont = Continuation::Halt;
        let result = reduce_expression(expr.clone(), cont, self.store.clone(), self.backend.clone(), self.id, verbose);
        if verbose {
            println!("[participant {}] finished eval -> {:?}", self.id, result);
        }
        result
    }
}

// helpers to convert between SharedValue and PersistValue
fn to_persist_value(v: &SharedValue) -> Option<PersistValue> {
    match v {
        SharedValue::Num(n) => Some(PersistValue::Num(*n)),
        SharedValue::Bool(b) => Some(PersistValue::Bool(*b)),
        SharedValue::Str(s) => Some(PersistValue::Str(s.clone())),
        SharedValue::Char(c) => Some(PersistValue::Char(*c)),
        SharedValue::List(lst) => {
            let mut out: Vec<PersistValue> = Vec::new();
            for item in lst.iter() {
                if let Some(iv) = to_persist_value(item) { out.push(iv); } else { return None; }
            }
            Some(PersistValue::List(out))
        }
        _ => None,
    }
}

fn from_persist_value(p: PersistValue) -> Option<SharedValue> {
    match p {
        PersistValue::Num(n) => Some(SharedValue::Num(n)),
        PersistValue::Bool(b) => Some(SharedValue::Bool(b)),
        PersistValue::Str(s) => Some(SharedValue::Str(s)),
        PersistValue::Char(c) => Some(SharedValue::Char(c)),
        PersistValue::List(vs) => {
            let mut out: Vec<SharedValue> = Vec::new();
            for iv in vs.into_iter() {
                if let Some(sv) = from_persist_value(iv) { out.push(sv); } else { return None; }
            }
            Some(SharedValue::List(out))
        }
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
         Continuation::IfResultNormalize { .. } => "IfResultNormalize".to_string(),
     }
 }
 
// Convert a signed i64 value into an unsigned field element representing two's complement with `nbits` bits
fn int_to_fr_signed(n: i64, nbits: usize) -> u128 {
    if nbits >= 128 {
        return n as i128 as u128;
    }
    let modulus = 1i128.checked_shl(nbits as u32).unwrap_or(0);
    if modulus == 0 { return n as i128 as u128; }
    let v = (n as i128).rem_euclid(modulus);
    v as u128
}
