use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::interpreter::aby3_simulator::Aby3Simulator;
use crate::interpreter::ast::{Env, Expr, Types};
use crate::interpreter::backend::Backend;
use crate::interpreter::parse;
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
        left_val: Types,
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
        left_val: Types,
        cont: Box<Continuation>,
        env: Env,
    },

    CallEvalOperator {
        args: Vec<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },
    CallEvalArgs {
        operator: Types,
        remaining: Vec<Expr>,
        mut_collected: Vec<Types>,
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
    Apply(Continuation, Types, Env),
}

const MAX_FRAMES: usize = 1_000_000;

// Note: added `owner` parameter so reduce_expression can pass participant id
// into share_secret and reveal operations.
pub fn reduce_expression(
    expr: Expr,
    cont: Continuation,
    env: Env,
    backend: Arc<Mutex<Box<dyn Backend>>>,
    _owner: usize,
    verbose: bool,
) -> Types {
    use Task::*;

    let mut stack: Vec<Task> = Vec::new();
    stack.push(Eval(expr.clone(), cont, env));
    let mut frames: usize = 0;

    // tracing setup: enable when verbose is requested
    let trace_enabled = verbose;
    const TRACE_LIMIT: usize = 500;
    let mut trace: Vec<String> = Vec::new();

    while let Some(task) = stack.pop() {
        frames = frames.saturating_add(1);
        if frames > MAX_FRAMES {
            if trace_enabled {
                eprintln!(
                    "{}",
                    format!("--- trace (first {} events) ---", trace.len()).bright_red()
                );
                for (i, r) in trace.iter().enumerate() {
                    eprintln!("{:05}: {}", i, r.bright_blue());
                }
            }
            panic!(
                "{}",
                format!("evaluation exceeded max frames ({})", MAX_FRAMES).bright_red()
            );
        }
        if frames.is_multiple_of(10000) {
            eprintln!(
                "{}",
                format!("reduce_expression frames = {}", frames).bright_yellow()
            );
        }

        if trace_enabled && trace.len() < TRACE_LIMIT {
            // Pretty / colored printers for environment, Expr and SharedValue used in traces

            fn colored_expr(e: &Expr) -> String {
                match e {
                    Expr::Num(n) => format!("{}", format!("Num({})", n).green()),
                    Expr::Fixed(v) => {
                        use crate::interpreter::ast::FIXED_SCALE;
                        let whole = v / FIXED_SCALE;
                        let frac = ((v % FIXED_SCALE).abs() * 10_000) / FIXED_SCALE;
                        format!("{}", format!("Fixed({}.{:04})", whole, frac).green())
                    }
                    Expr::Var(s) => format!("{}", format!("Var({})", s).cyan()),
                    Expr::Str(s) => format!("{}", format!("Str(\"{}\")", s).magenta()),
                    Expr::Char(c) => format!("{}", format!("Char({})", c).magenta()),
                    Expr::Bool(b) => format!("{}", format!("Bool({})", b).yellow()),
                    Expr::BinOp(_, _, _) => format!("{}", expr_short(e).blue()),
                    Expr::Quote(_) => format!("{}", expr_short(e).blue()),
                    Expr::Eval(_) => format!("{}", expr_short(e).blue()),
                    Expr::UnOp(_, _) => format!("{}", expr_short(e).blue()),
                    Expr::Let(bs, _) => format!("{}", format!("Let(len={})", bs.len()).blue()),
                    Expr::LetRec(bs, _) => {
                        format!("{}", format!("LetRec(len={})", bs.len()).blue())
                    }
                    Expr::Seq(es) => format!("{}", format!("Seq(len={})", es.len()).blue()),
                    Expr::Car(_) => "Car".blue().to_string(),
                    Expr::Cdr(_) => "Cdr".blue().to_string(),
                    Expr::Cons(_, _) => "Cons".blue().to_string(),
                    Expr::Lambda(ps, _) => {
                        format!("{}", format!("Lambda(arity={})", ps.len()).magenta())
                    }
                    Expr::Call(_, args) => {
                        format!("{}", format!("Call(arity={})", args.len()).magenta())
                    }
                    Expr::If(_, _, _) => "If".blue().to_string(),
                    Expr::Private(_) => format!("{}", expr_short(e).blue()),
                }
            }

            fn colored_value(v: &Types) -> String {
                match v {
                    Types::Num(n) => format!("{}", format!("Num({})", n).green()),
                    Types::Fixed(val) => {
                        use crate::interpreter::ast::FIXED_SCALE;
                        let whole = val / FIXED_SCALE;
                        let frac = ((val % FIXED_SCALE).abs() * 10_000) / FIXED_SCALE;
                        format!("{}", format!("{}.{:04}", whole, frac).green())
                    }
                    Types::Bool(b) => format!("{}", format!("Bool({})", b).yellow()),
                    Types::Str(s) => format!("{}", format!("Str(\"{}\")", s).magenta()),
                    Types::Char(c) => format!("{}", format!("Char({})", c).magenta()),
                    Types::List(l) => format!("{}", format!("List(len={})", l.len()).blue()),
                    Types::Quoted(e) => format!("{}", format!("Quoted({:?})", e).blue()),
                    Types::Closure(_, _, _) => "Closure".magenta().to_string(),
                    Types::Share(sr) => format!(
                        "{}",
                        format!("Share(id={},nbits={},owner={})", sr.id, sr.nbits, sr.owner).red()
                    ),
                    Types::NativeFunc(_) => "NativeFunc".cyan().to_string(),
                    Types::FheCiphertext(h) => {
                        format!("{}", format!("FheCt({})", h.backend_id).red())
                    }
                }
            }

            match &task {
                Eval(e, _k, _env) => {
                    // quieter eval trace: show expression and a short pretty env
                    trace.push(format!("{}: {}", "Eval".green(), colored_expr(e),));
                }
                Apply(k, v, _env_snapshot) => {
                    // quieter apply trace: show continuation and the single colored value
                    let vcol = colored_value(v);
                    trace.push(format!(
                        "{}: cont={} val={}",
                        "Apply".yellow(),
                        cont_short(k).bright_black(),
                        vcol
                    ));
                }
            }
        }

        match task {
            Eval(e, k, eenv) => match e {
                Expr::Num(n) => {
                    stack.push(Apply(k, Types::Num(n), eenv.clone()));
                }
                Expr::Fixed(v) => {
                    stack.push(Apply(k, Types::Fixed(v), eenv.clone()));
                }
                Expr::Str(s) => {
                    stack.push(Apply(k, Types::Str(s), eenv.clone()));
                }
                Expr::Char(c) => {
                    stack.push(Apply(k, Types::Char(c), eenv.clone()));
                }
                Expr::Bool(b) => {
                    stack.push(Apply(k, Types::Bool(b), eenv.clone()));
                }
                Expr::Quote(inner) => {
                    // Literal data lists like '(1 2 3) evaluate to Types::List so
                    // car/cdr/null? work on them.  Anything else (quoted code such
                    // as '(+ x y), or string lists used by zkp/fhe builtins) stays
                    // Quoted.
                    if let Some(vals) = quoted_literal_list(&inner) {
                        stack.push(Apply(k, Types::List(vals), eenv.clone()));
                    } else {
                        stack.push(Apply(k, Types::Quoted(Box::new(*inner)), eenv.clone()));
                    }
                }
                Expr::Private(inner) => {
                    // Private annotation is metadata for ZKP witness marking.
                    // At runtime, we evaluate the inner expression directly.
                    stack.push(Eval(*inner, k, eenv.clone()));
                }
                Expr::Eval(inner) => {
                    // if inner is quoted, we can evaluate it directly without going through the Eval continuation
                    if let Expr::Quote(inner_inner) = *inner {
                        stack.push(Eval(*inner_inner, k, eenv.clone()));
                    } else {
                        stack.push(Eval(*inner, k, eenv.clone()));
                    }
                }
                Expr::UnOp(op, inner) => {
                    let new_cont = Continuation::UnOp {
                        op,
                        cont: Box::new(k),
                        env: eenv.clone(),
                    };
                    stack.push(Eval(*inner, new_cont, eenv.clone()));
                }
                Expr::Var(name) => {
                    let map = eenv.borrow();
                    let ptr = map
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| panic!("unbound variable: {}", name));
                    let v = ptr.borrow().clone();
                    stack.push(Apply(k, v, eenv.clone()));
                }
                Expr::BinOp(op, left, right) => {
                    let new_cont = Continuation::BinOpLeft {
                        op,
                        right,
                        cont: Box::new(k),
                        env: eenv.clone(),
                    };
                    stack.push(Eval(*left, new_cont, eenv.clone()));
                }
                Expr::Let(bindings, body) => {
                    if body.is_empty() {
                        if bindings.is_empty() {
                            stack.push(Apply(k, Types::Num(0), eenv.clone()));
                        } else {
                            let mut it = bindings.into_iter();
                            let (name, be) = it.next().unwrap();
                            let remaining: Vec<(String, Expr)> = it.collect();
                            stack.push(Eval(
                                be,
                                Continuation::LetTop {
                                    name,
                                    remaining,
                                    env: eenv.clone(),
                                },
                                eenv.clone(),
                            ));
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
                            stack.push(Eval(
                                be,
                                Continuation::LetLocal {
                                    name,
                                    remaining,
                                    local_env: local.clone(),
                                    body: body.clone(),
                                    cont: Box::new(k),
                                },
                                local.clone(),
                            ));
                        }
                    }
                }
                Expr::LetRec(bindings, body) => {
                    let local = Rc::new(RefCell::new(eenv.borrow().clone()));
                    for (name, _) in &bindings {
                        local
                            .borrow_mut()
                            .insert(name.clone(), Rc::new(RefCell::new(Types::Num(0))));
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
                        stack.push(Eval(
                            be,
                            Continuation::LetRecInit {
                                name,
                                remaining,
                                local_env: local.clone(),
                                body: body.clone(),
                                cont: Box::new(k),
                            },
                            local.clone(),
                        ));
                    }
                }
                Expr::Seq(exprs) => {
                    if exprs.is_empty() {
                        stack.push(Apply(k, Types::List(Vec::new()), eenv.clone()));
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
                    stack.push(Eval(
                        *inner,
                        Continuation::Car {
                            cont: Box::new(k),
                            env: eenv.clone(),
                        },
                        eenv.clone(),
                    ));
                }
                Expr::Cdr(inner) => {
                    stack.push(Eval(
                        *inner,
                        Continuation::Cdr {
                            cont: Box::new(k),
                            env: eenv.clone(),
                        },
                        eenv.clone(),
                    ));
                }
                Expr::Cons(h, t) => {
                    stack.push(Eval(
                        *h,
                        Continuation::ConsLeft {
                            right: t,
                            cont: Box::new(k),
                            env: eenv.clone(),
                        },
                        eenv.clone(),
                    ));
                }
                Expr::Lambda(params, body) => {
                    let closure = Types::Closure(params, body, eenv.clone());
                    stack.push(Apply(k, closure, eenv.clone()));
                }
                Expr::Call(head, args) => {
                    stack.push(Eval(
                        *head,
                        Continuation::CallEvalOperator {
                            args,
                            cont: Box::new(k),
                            env: eenv.clone(),
                        },
                        eenv.clone(),
                    ));
                }
                Expr::If(cond, then_e, else_e) => {
                    stack.push(Eval(
                        *cond,
                        Continuation::If {
                            then_e,
                            else_e,
                            cont: Box::new(k),
                            env: eenv.clone(),
                        },
                        eenv.clone(),
                    ));
                }
            },
            Apply(k, val, _env) => {
                match k {
                    Continuation::Halt => {
                        if stack.is_empty() {
                            if trace_enabled && !trace.is_empty() {
                                eprintln!(
                                    "{}",
                                    format!(
                                        "--- verbose trace (most recent {} events) ---",
                                        trace.len()
                                    )
                                    .bright_yellow()
                                );
                                for (i, r) in trace.iter().enumerate() {
                                    eprintln!("{:05}: {}", i, r.bright_blue());
                                }
                            }
                            return val;
                        } else {
                            continue;
                        }
                    }
                    Continuation::BinOpLeft {
                        op,
                        right,
                        cont,
                        env,
                    } => {
                        let next = Continuation::BinOpRight {
                            op,
                            left_val: val,
                            cont,
                            env: env.clone(),
                        };
                        stack.push(Eval(*right, next, env.clone()));
                    }
                    Continuation::BinOpRight {
                        op,
                        left_val,
                        cont,
                        env,
                    } => {
                        // helper to view a value as an i64 when reasonable (Num or Bool)
                        let maybe_num = |v: &Types| -> Option<i64> {
                            match v {
                                Types::Num(n) => Some(*n),
                                Types::Bool(b) => Some(if *b { 1 } else { 0 }),
                                _ => None,
                            }
                        };
                        // helper: is a value a Fixed-point number?
                        let is_fixed = |v: &Types| matches!(v, Types::Fixed(_));
                        // coerce Num → Fixed by scaling; Fixed stays as-is
                        let to_fixed = |v: &Types| -> i64 {
                            use crate::interpreter::ast::FIXED_SCALE;
                            match v {
                                Types::Fixed(f) => *f,
                                Types::Num(n) => n * FIXED_SCALE,
                                _ => panic!("cannot coerce to fixed-point"),
                            }
                        };

                        // Q16.16 shares (or a share mixed with a plain Fixed)
                        // need scale-aware handling; the plain-share arms
                        // below would silently drop the scaling.
                        let is_share = |v: &Types| matches!(v, Types::Share(_));
                        let fixed_share_mix = matches!(&left_val, Types::Share(s) if s.fixed)
                            || matches!(&val, Types::Share(s) if s.fixed)
                            || (is_share(&left_val) && is_fixed(&val))
                            || (is_fixed(&left_val) && is_share(&val));

                        let out = if fixed_share_mix {
                            fixed_share_binop(&backend, op, &left_val, &val)
                        } else {
                            match op {
                                super::parser::BinOp::Add
                                | super::parser::BinOp::Sub
                                | super::parser::BinOp::Mul
                                | super::parser::BinOp::DivExact
                                | super::parser::BinOp::Mod => {
                                    // if either operand is a shared value, dispatch to backend
                                    match (&left_val, &val) {
                                        (Types::Share(sa), Types::Share(sb)) => {
                                            let mut be =
                                                backend.lock().unwrap_or_else(|e| e.into_inner());
                                            let sr = match op {
                                                super::parser::BinOp::Add => be.add(sa, sb),
                                                super::parser::BinOp::Sub => be.sub(sa, sb),
                                                super::parser::BinOp::Mul => be.mul(sa, sb),
                                                super::parser::BinOp::DivExact => {
                                                    be.divmod(sa, sb).0
                                                }
                                                super::parser::BinOp::Mod => be.divmod(sa, sb).1,
                                                _ => panic!("unsupported shared op"),
                                            };
                                            Types::Share(sr)
                                        }
                                        (Types::Share(sa), Types::Num(bn)) => {
                                            let mut be =
                                                backend.lock().unwrap_or_else(|e| e.into_inner());
                                            // convert signed immediate into two's-complement field element for sa.nbits
                                            let imm = int_to_fr_signed(*bn, sa.nbits);
                                            let sb = be.share_secret(imm, sa.nbits, 0);
                                            let sr = match op {
                                                super::parser::BinOp::Add => be.add(sa, &sb),
                                                super::parser::BinOp::Sub => be.sub(sa, &sb),
                                                super::parser::BinOp::Mul => be.mul(sa, &sb),
                                                super::parser::BinOp::DivExact => {
                                                    be.divmod(sa, &sb).0
                                                }
                                                super::parser::BinOp::Mod => be.divmod(sa, &sb).1,
                                                _ => panic!("unsupported shared op"),
                                            };
                                            Types::Share(sr)
                                        }
                                        (Types::Num(an), Types::Share(sb)) => {
                                            let mut be =
                                                backend.lock().unwrap_or_else(|e| e.into_inner());
                                            // convert signed immediate into two's-complement field element for sb.nbits
                                            let imm = int_to_fr_signed(*an, sb.nbits);
                                            let sa = be.share_secret(imm, sb.nbits, 0);
                                            let sr = match op {
                                                super::parser::BinOp::Add => be.add(&sa, sb),
                                                super::parser::BinOp::Sub => be.sub(&sa, sb),
                                                super::parser::BinOp::Mul => be.mul(&sa, sb),
                                                super::parser::BinOp::DivExact => {
                                                    be.divmod(&sa, sb).0
                                                }
                                                super::parser::BinOp::Mod => be.divmod(&sa, sb).1,
                                                _ => panic!("unsupported shared op"),
                                            };
                                            Types::Share(sr)
                                        }
                                        _ => {
                                            // If either operand is Fixed, perform fixed-point arithmetic
                                            if is_fixed(&left_val) || is_fixed(&val) {
                                                let a = to_fixed(&left_val);
                                                let b = to_fixed(&val);
                                                let r = match op {
                                                    super::parser::BinOp::Add => a.wrapping_add(b),
                                                    super::parser::BinOp::Sub => a.wrapping_sub(b),
                                                    super::parser::BinOp::Mul => {
                                                        // (a * b) >> 16 using i128 to avoid overflow
                                                        ((a as i128 * b as i128) >> 16) as i64
                                                    }
                                                    super::parser::BinOp::DivExact => {
                                                        if b == 0 {
                                                            panic!("division by zero");
                                                        }
                                                        // (a << 16) / b
                                                        ((a as i128) << 16) as i64 / b
                                                    }
                                                    super::parser::BinOp::Mod => {
                                                        if b == 0 {
                                                            panic!("modulo by zero");
                                                        }
                                                        a % b
                                                    }
                                                    _ => unreachable!(),
                                                };
                                                Types::Fixed(r)
                                            } else {
                                                let a = maybe_num(&left_val).expect(
                                                "binary arithmetic left operand must be NUM or BOOL",
                                            );
                                                let b = maybe_num(&val).expect(
                                                "binary arithmetic right operand must be NUM or BOOL",
                                            );
                                                let r = match op {
                                                    super::parser::BinOp::Add => a.wrapping_add(b),
                                                    super::parser::BinOp::Sub => a.wrapping_sub(b),
                                                    super::parser::BinOp::Mul => a.wrapping_mul(b),
                                                    super::parser::BinOp::DivExact => {
                                                        if b == 0 {
                                                            panic!("division by zero");
                                                        }
                                                        if a % b != 0 {
                                                            panic!(
                                                                "non-exact division: {} / {}",
                                                                a, b
                                                            );
                                                        }
                                                        a / b
                                                    }
                                                    super::parser::BinOp::Mod => {
                                                        if b == 0 {
                                                            panic!("modulo by zero");
                                                        }
                                                        a % b
                                                    }
                                                    _ => unreachable!(),
                                                };
                                                Types::Num(r)
                                            }
                                        }
                                    }
                                }
                                super::parser::BinOp::Eq => match (&left_val, &val) {
                                    (Types::Num(a), Types::Num(b)) => {
                                        Types::Num(if a == b { 1 } else { 0 })
                                    }
                                    (Types::Fixed(a), Types::Fixed(b)) => {
                                        Types::Num(if a == b { 1 } else { 0 })
                                    }
                                    (Types::Fixed(a), Types::Num(b)) => {
                                        use crate::interpreter::ast::FIXED_SCALE;
                                        Types::Num(if *a == b * FIXED_SCALE { 1 } else { 0 })
                                    }
                                    (Types::Num(a), Types::Fixed(b)) => {
                                        use crate::interpreter::ast::FIXED_SCALE;
                                        Types::Num(if a * FIXED_SCALE == *b { 1 } else { 0 })
                                    }
                                    (Types::Bool(a), Types::Bool(b)) => {
                                        Types::Num(if a == b { 1 } else { 0 })
                                    }
                                    (Types::Str(a), Types::Str(b)) => {
                                        Types::Num(if a == b { 1 } else { 0 })
                                    }
                                    (Types::Char(a), Types::Char(b)) => {
                                        Types::Num(if a == b { 1 } else { 0 })
                                    }
                                    (Types::Share(sa), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        Types::Share(be.eq(sa, sb))
                                    }
                                    (Types::Share(sa), Types::Num(bn)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*bn, sa.nbits);
                                        let sb = be.share_secret(imm, sa.nbits, 0);
                                        Types::Share(be.eq(sa, &sb))
                                    }
                                    (Types::Num(an), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*an, sb.nbits);
                                        let sa = be.share_secret(imm, sb.nbits, 0);
                                        Types::Share(be.eq(&sa, sb))
                                    }
                                    _ => Types::Num(0),
                                },
                                super::parser::BinOp::Lt
                                | super::parser::BinOp::Gt
                                | super::parser::BinOp::Le
                                | super::parser::BinOp::Ge => match (&left_val, &val) {
                                    (Types::Num(a), Types::Num(b)) => {
                                        let r = match op {
                                            super::parser::BinOp::Lt => a < b,
                                            super::parser::BinOp::Gt => a > b,
                                            super::parser::BinOp::Le => a <= b,
                                            super::parser::BinOp::Ge => a >= b,
                                            _ => unreachable!(),
                                        };
                                        Types::Num(if r { 1 } else { 0 })
                                    }
                                    // Fixed vs Fixed (or Fixed vs Num via coercion)
                                    (Types::Fixed(_), Types::Fixed(_))
                                    | (Types::Fixed(_), Types::Num(_))
                                    | (Types::Num(_), Types::Fixed(_)) => {
                                        let a = to_fixed(&left_val);
                                        let b = to_fixed(&val);
                                        let r = match op {
                                            super::parser::BinOp::Lt => a < b,
                                            super::parser::BinOp::Gt => a > b,
                                            super::parser::BinOp::Le => a <= b,
                                            super::parser::BinOp::Ge => a >= b,
                                            _ => unreachable!(),
                                        };
                                        Types::Num(if r { 1 } else { 0 })
                                    }
                                    (Types::Share(sa), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let out = match op {
                                            super::parser::BinOp::Lt => be.lt(sa, sb),
                                            super::parser::BinOp::Gt => be.lt(sb, sa),
                                            super::parser::BinOp::Le => be.le(sa, sb),
                                            super::parser::BinOp::Ge => be.le(sb, sa),
                                            _ => unreachable!(),
                                        };
                                        Types::Share(out)
                                    }
                                    (Types::Share(sa), Types::Num(bn)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*bn, sa.nbits);
                                        let sb = be.share_secret(imm, sa.nbits, 0);
                                        let out = match op {
                                            super::parser::BinOp::Lt => be.lt(sa, &sb),
                                            super::parser::BinOp::Gt => be.lt(&sb, sa),
                                            super::parser::BinOp::Le => be.le(sa, &sb),
                                            super::parser::BinOp::Ge => be.le(&sb, sa),
                                            _ => unreachable!(),
                                        };
                                        Types::Share(out)
                                    }
                                    (Types::Num(an), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*an, sb.nbits);
                                        let sa = be.share_secret(imm, sb.nbits, 0);
                                        let out = match op {
                                            super::parser::BinOp::Lt => be.lt(&sa, sb),
                                            super::parser::BinOp::Gt => be.lt(sb, &sa),
                                            super::parser::BinOp::Le => be.le(&sa, sb),
                                            super::parser::BinOp::Ge => be.le(sb, &sa),
                                            _ => unreachable!(),
                                        };
                                        Types::Share(out)
                                    }
                                    _ => Types::Num(0),
                                },
                                super::parser::BinOp::And => match (&left_val, &val) {
                                    (Types::Share(sa), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        Types::Share(be.and(sa, sb))
                                    }
                                    (Types::Share(sa), Types::Num(b)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*b, sa.nbits);
                                        let sb = be.share_secret(imm, sa.nbits, 0);
                                        Types::Share(be.and(sa, &sb))
                                    }
                                    (Types::Num(a), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*a, sb.nbits);
                                        let sa = be.share_secret(imm, sb.nbits, 0);
                                        Types::Share(be.and(&sa, sb))
                                    }
                                    (Types::Num(a), Types::Num(b)) => Types::Num(a & b),
                                    _ => {
                                        let a = maybe_num(&left_val).expect("and expects NUM/BOOL");
                                        let b = maybe_num(&val).expect("and expects NUM/BOOL");
                                        Types::Num(if a != 0 && b != 0 { 1 } else { 0 })
                                    }
                                },
                                super::parser::BinOp::Or => match (&left_val, &val) {
                                    (Types::Share(sa), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        Types::Share(be.or(sa, sb))
                                    }
                                    (Types::Share(sa), Types::Num(b)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*b, sa.nbits);
                                        let sb = be.share_secret(imm, sa.nbits, 0);
                                        Types::Share(be.or(sa, &sb))
                                    }
                                    (Types::Num(a), Types::Share(sb)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let imm = int_to_fr_signed(*a, sb.nbits);
                                        let sa = be.share_secret(imm, sb.nbits, 0);
                                        Types::Share(be.or(&sa, sb))
                                    }
                                    (Types::Num(a), Types::Num(b)) => Types::Num(a | b),
                                    _ => {
                                        let a = maybe_num(&left_val).expect("or expects NUM/BOOL");
                                        let b = maybe_num(&val).expect("or expects NUM/BOOL");
                                        Types::Num(if a != 0 || b != 0 { 1 } else { 0 })
                                    }
                                },
                                super::parser::BinOp::AshRight => match (&left_val, &val) {
                                    (Types::Share(sa), Types::Num(bn)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        // convert bn to usize
                                        let b = usize::try_from(*bn).unwrap_or(0);
                                        Types::Share(be.shift_right(sa, b))
                                    }
                                    (Types::Num(an), Types::Num(bn)) => {
                                        let r = an.wrapping_shr(*bn as u32);
                                        Types::Num(r)
                                    }
                                    _ => panic!("ash-right expects (SHARE, NUM) or (NUM, NUM)"),
                                },
                                super::parser::BinOp::AshLeft => match (&left_val, &val) {
                                    (Types::Share(sa), Types::Num(bn)) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        // convert bn to usize
                                        let b = usize::try_from(*bn).unwrap_or(0);
                                        Types::Share(be.shift_left(sa, b))
                                    }
                                    (Types::Num(an), Types::Num(bn)) => {
                                        let r = an.wrapping_shl(*bn as u32);
                                        Types::Num(r)
                                    }
                                    _ => panic!("ash-left expects (SHARE, NUM) or (NUM, NUM)"),
                                },
                            }
                        };
                        stack.push(Apply(*cont, out, env.clone()));
                    }
                    Continuation::UnOp { op, cont, env } => {
                        let out = match op {
                            super::parser::UnOp::Atom => match val {
                                Types::Num(_) | Types::Bool(_) | Types::Str(_) | Types::Char(_) => {
                                    Types::Num(1)
                                }
                                _ => Types::Num(0),
                            },
                            super::parser::UnOp::Not => match val {
                                Types::Num(n) => Types::Num(if n == 0 { 1 } else { 0 }),
                                Types::Bool(b) => Types::Num(if !b { 1 } else { 0 }),
                                _ => panic!("not expects numeric or boolean value"),
                            },
                            super::parser::UnOp::Neg => match val {
                                Types::Num(n) => Types::Num(n.wrapping_neg()),
                                _ => panic!("negation expects numeric value"),
                            },
                            super::parser::UnOp::IsNull => match val {
                                Types::List(ref vs) => {
                                    Types::Num(if vs.is_empty() { 1 } else { 0 })
                                }
                                Types::Quoted(ref e) => match e.as_ref() {
                                    Expr::Seq(items) => {
                                        Types::Num(if items.is_empty() { 1 } else { 0 })
                                    }
                                    _ => Types::Num(0),
                                },
                                _ => panic!("null? expects a list"),
                            },
                            super::parser::UnOp::BitLen => match val {
                                Types::Num(n) => {
                                    let bit_length = |n: u32| {
                                        // ilog2(0) is undefined, so checked_ilog2 returns None for 0
                                        n.checked_ilog2().map_or(0, |log| log + 1)
                                    };
                                    Types::Num(bit_length(n as u32) as i64)
                                }
                                Types::Share(s) => {
                                    let mut be = backend.lock().unwrap_or_else(|e| e.into_inner());
                                    let result = be.bitlen(&s);
                                    Types::Share(result)
                                }
                                _ => panic!("bitlen expects numeric or shared value"),
                            },
                        };
                        stack.push(Apply(*cont, out, env.clone()));
                    }
                    Continuation::LetTop {
                        name,
                        mut remaining,
                        env,
                    } => {
                        env.borrow_mut().insert(name, Rc::new(RefCell::new(val)));
                        if remaining.is_empty() {
                            stack.push(Apply(Continuation::Halt, Types::Num(0), env.clone()));
                        } else {
                            let (nxt_name, nxt_expr) = remaining.remove(0);
                            stack.push(Eval(
                                nxt_expr,
                                Continuation::LetTop {
                                    name: nxt_name,
                                    remaining,
                                    env: env.clone(),
                                },
                                env.clone(),
                            ));
                        }
                    }
                    Continuation::LetLocal {
                        name,
                        mut remaining,
                        local_env,
                        body,
                        cont,
                    } => {
                        local_env
                            .borrow_mut()
                            .insert(name, Rc::new(RefCell::new(val)));
                        if remaining.is_empty() {
                            if body.is_empty() {
                                stack.push(Apply(*cont, Types::Num(0), local_env.clone()));
                            } else if body.len() == 1 {
                                stack.push(Eval(
                                    body.into_iter().next().unwrap(),
                                    *cont,
                                    local_env.clone(),
                                ));
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
                            stack.push(Eval(
                                nxt_expr,
                                Continuation::LetLocal {
                                    name: nxt_name,
                                    remaining,
                                    local_env: local_env.clone(),
                                    body,
                                    cont,
                                },
                                local_env.clone(),
                            ));
                        }
                    }
                    Continuation::LetRecInit {
                        name,
                        mut remaining,
                        local_env,
                        body,
                        cont,
                    } => {
                        if let Some(ptr) = local_env.borrow().get(&name) {
                            *ptr.borrow_mut() = val;
                        } else {
                            local_env
                                .borrow_mut()
                                .insert(name.clone(), Rc::new(RefCell::new(val)));
                        }
                        if remaining.is_empty() {
                            if body.len() == 1 {
                                stack.push(Eval(
                                    body.into_iter().next().unwrap(),
                                    *cont,
                                    local_env.clone(),
                                ));
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
                            stack.push(Eval(
                                nxt_expr,
                                Continuation::LetRecInit {
                                    name: nxt_name,
                                    remaining,
                                    local_env: local_env.clone(),
                                    body,
                                    cont: cont.clone(),
                                },
                                local_env.clone(),
                            ));
                        }
                    }
                    Continuation::Seq {
                        mut remaining,
                        cont,
                        env,
                    } => {
                        if remaining.is_empty() {
                            stack.push(Apply(*cont, val, env.clone()));
                        } else {
                            let next = remaining.remove(0);
                            if remaining.is_empty() {
                                stack.push(Eval(next, *cont, env.clone()));
                            } else {
                                stack.push(Eval(
                                    next,
                                    Continuation::Seq {
                                        remaining,
                                        cont,
                                        env: env.clone(),
                                    },
                                    env.clone(),
                                ));
                            }
                        }
                    }
                    Continuation::Car { cont, env } => match val {
                        Types::List(mut vs) => {
                            if vs.is_empty() {
                                panic!("car of empty list");
                            }
                            let first = vs.remove(0);
                            stack.push(Apply(*cont, first, env.clone()));
                        }
                        _ => panic!("car expects a list"),
                    },
                    Continuation::Cdr { cont, env } => match val {
                        Types::List(vs) => {
                            if vs.is_empty() {
                                panic!("cdr of empty list");
                            }
                            let tail = Types::List(vs.into_iter().skip(1).collect());
                            stack.push(Apply(*cont, tail, env.clone()));
                        }
                        _ => panic!("cdr expects a list"),
                    },
                    Continuation::ConsLeft { right, cont, env } => {
                        let next = Continuation::ConsRight {
                            left_val: val,
                            cont,
                            env: env.clone(),
                        };
                        stack.push(Eval(*right, next, env.clone()));
                    }
                    Continuation::ConsRight {
                        left_val,
                        cont,
                        env,
                    } => match val {
                        Types::List(mut vs) => {
                            vs.insert(0, left_val);
                            stack.push(Apply(*cont, Types::List(vs), env.clone()));
                        }
                        _ => panic!("cons expects second arg to be a list"),
                    },
                    Continuation::CallEvalOperator { args, cont, env } => {
                        if args.is_empty() {
                            match val {
                                Types::Closure(params, body, closure_env) => {
                                    if !params.is_empty() {
                                        panic!(
                                            "closure arity mismatch: expected {} got 0",
                                            params.len()
                                        );
                                    }
                                    let call_env =
                                        Rc::new(RefCell::new(closure_env.borrow().clone()));
                                    stack.push(Eval(*body, *cont, call_env));
                                }
                                Types::NativeFunc(f) => {
                                    let result = f(vec![]);
                                    stack.push(Apply(*cont, result, env.clone()));
                                }
                                _ => panic!("attempt to call non-function: {:?}", val),
                            }
                        } else {
                            let mut rem = args;
                            let first = rem.remove(0);
                            let call_args_cont = Continuation::CallEvalArgs {
                                operator: val,
                                remaining: rem,
                                mut_collected: Vec::new(),
                                cont,
                                env: env.clone(),
                            };
                            stack.push(Eval(first, call_args_cont, env.clone()));
                        }
                    }
                    Continuation::CallEvalArgs {
                        operator,
                        mut remaining,
                        mut mut_collected,
                        cont,
                        env,
                    } => {
                        mut_collected.push(val);
                        if remaining.is_empty() {
                            match operator {
                                Types::Closure(params, body, closure_env) => {
                                    if params.len() != mut_collected.len() {
                                        panic!(
                                            "closure arity mismatch: expected {} got {}",
                                            params.len(),
                                            mut_collected.len()
                                        );
                                    }
                                    let call_env =
                                        Rc::new(RefCell::new(closure_env.borrow().clone()));
                                    for (p, v) in params.into_iter().zip(mut_collected) {
                                        call_env.borrow_mut().insert(p, Rc::new(RefCell::new(v)));
                                    }
                                    stack.push(Eval(*body, *cont, call_env));
                                }
                                Types::Share(_sref) => {
                                    panic!("attempt to call a shared value as function");
                                }
                                Types::NativeFunc(f) => {
                                    // native function: apply with left-to-right arguments
                                    let all_args = mut_collected;
                                    match f(all_args) {
                                        v @ Types::Num(_) => {
                                            stack.push(Apply(*cont, v, env.clone()));
                                        }
                                        Types::Share(sr) => {
                                            stack.push(Apply(*cont, Types::Share(sr), env.clone()));
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
                            let new_cont = Continuation::CallEvalArgs {
                                operator,
                                remaining,
                                mut_collected,
                                cont,
                                env: env.clone(),
                            };
                            stack.push(Eval(next, new_cont, env.clone()));
                        }
                    }
                    Continuation::If {
                        then_e,
                        else_e,
                        cont,
                        env,
                    } => {
                        match val {
                            Types::Num(n) => {
                                if n != 0 {
                                    stack.push(Eval(*then_e, *cont, env.clone()));
                                } else {
                                    stack.push(Eval(*else_e, *cont, env.clone()));
                                }
                            }
                            Types::Bool(b) => {
                                if b {
                                    stack.push(Eval(*then_e, *cont, env.clone()));
                                } else {
                                    stack.push(Eval(*else_e, *cont, env.clone()));
                                }
                            }
                            Types::Share(sr) => {
                                // reveal the boolean share to all parties so they all
                                // take the same control-flow branch (leaks 1 bit)
                                let revealed = {
                                    let mut be = backend.lock().unwrap_or_else(|e| e.into_inner());
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

                    Continuation::IfResultNormalize {
                        cont,
                        normalize,
                        env,
                    } => {
                        // When a branch result is produced, optionally normalize NUM->SHARE
                        match normalize {
                            Some((_norm_owner, norm_nbits)) => {
                                match val {
                                    Types::Share(_) => {
                                        // already a share, just continue
                                        stack.push(Apply(*cont, val, env.clone()));
                                    }
                                    Types::Num(n) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let fr = int_to_fr_signed(n, norm_nbits);
                                        let sref = be.share_secret(fr, norm_nbits, 0);
                                        stack.push(Apply(*cont, Types::Share(sref), env.clone()));
                                    }
                                    Types::Bool(b) => {
                                        let mut be =
                                            backend.lock().unwrap_or_else(|e| e.into_inner());
                                        let fr = if b { 1u128 } else { 0u128 };
                                        let sref = be.share_secret(fr, norm_nbits, 0);
                                        stack.push(Apply(*cont, Types::Share(sref), env.clone()));
                                    }
                                    _ => {
                                        // other types not supported for normalization
                                        panic!(
                                            "cannot normalize branch result to share: {:?}",
                                            val
                                        );
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
    /// A participant backed by the in-process
    /// [`Aby3Simulator`](crate::interpreter::aby3_simulator::Aby3Simulator).
    ///
    /// This is the right constructor for evaluating an expression on its own —
    /// a correctness baseline, a test, a plaintext-vs-MPC comparison — and it is
    /// the *only* one that can, because a single party of the real protocol has
    /// no peers to exchange with and would block forever. For anything whose
    /// timings or message counts are reported, use [`Participant::new_transport`]
    /// with three parties running concurrently.
    pub fn new_simulated(id: usize) -> Self {
        let backend: Arc<Mutex<Box<dyn Backend>>> =
            Arc::new(Mutex::new(Box::new(Aby3Simulator::new())));
        let store: Env = Rc::new(RefCell::new(HashMap::new()));
        let participant = Participant {
            id,
            backend: backend.clone(),
            store: store.clone(),
        };
        // register builtins once into the persistent store
        participant.register_builtins();
        participant
    }

    /// Create a participant backed by a real `TransportBackend` using an
    /// existing `Transport` connection (e.g. `TcpTransport` or `InprocHandle`).
    pub fn new_transport<T: crate::crypto::net::Transport + 'static>(
        id: usize,
        transport: T,
    ) -> Self {
        use crate::interpreter::transport_backend::TransportBackend;
        let backend: Arc<Mutex<Box<dyn Backend>>> =
            Arc::new(Mutex::new(Box::new(TransportBackend::new(id, transport))));
        let store: Env = Rc::new(RefCell::new(HashMap::new()));
        let participant = Participant {
            id,
            backend: backend.clone(),
            store: store.clone(),
        };
        participant.register_builtins();
        participant
    }

    fn register_builtins(&self) {
        let backend = self.backend.clone();
        let _owner = self.id;

        // share builtin
        let share_backend = backend.clone();
        let share_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("share expects 2 args");
            }
            match (&args[0], &args[1]) {
                (Types::Num(n), Types::Num(nb)) => {
                    let mut be = share_backend.lock().unwrap_or_else(|e| e.into_inner());
                    // convert signed immediate into two's-complement field element for nb bits
                    let imm = int_to_fr_signed(*n, *nb as usize);
                    // Always pass the actual value. SimBackend uses it directly.
                    // TransportBackend passes it to protocol::share_input which ignores
                    // it for non-owner parties, so the owner check is enforced there.
                    let canonical_owner = 0usize;
                    let sref = be.share_secret(imm, *nb as usize, canonical_owner);
                    Types::Share(sref)
                }
                (Types::Fixed(f), Types::Num(nb)) => {
                    // Q16.16 value: share the scaled integer, flag the share.
                    let mut be = share_backend.lock().unwrap_or_else(|e| e.into_inner());
                    let imm = int_to_fr_signed(*f, *nb as usize);
                    let mut sref = be.share_secret(imm, *nb as usize, 0);
                    sref.fixed = true;
                    Types::Share(sref)
                }
                (Types::Share(s), Types::Num(_nb)) => Types::Share(s.clone()),
                _ => panic!("share expects (num nbits) or (share nbits)"),
            }
        });
        self.store.borrow_mut().insert(
            "share".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(share_fn))),
        );

        // share-from builtin: (share-from owner-id value nbits)
        // Like (share value nbits) but the owner is explicitly specified, so each
        // party can provide their own private input.
        let share_from_backend = backend.clone();
        let share_from_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 3 {
                panic!("share-from expects 3 args: (share-from owner-id value nbits)");
            }
            match (&args[0], &args[1], &args[2]) {
                (Types::Num(owner_id), Types::Num(n), Types::Num(nb)) => {
                    let mut be = share_from_backend.lock().unwrap_or_else(|e| e.into_inner());
                    let explicit_owner = *owner_id as usize;
                    let imm = int_to_fr_signed(*n, *nb as usize);
                    // Same as share: pass the actual value; protocol::share_input enforces
                    // the owner check in distributed mode.
                    let sref = be.share_secret(imm, *nb as usize, explicit_owner);
                    Types::Share(sref)
                }
                (Types::Num(owner_id), Types::Fixed(f), Types::Num(nb)) => {
                    let mut be = share_from_backend.lock().unwrap_or_else(|e| e.into_inner());
                    let imm = int_to_fr_signed(*f, *nb as usize);
                    let mut sref = be.share_secret(imm, *nb as usize, *owner_id as usize);
                    sref.fixed = true;
                    Types::Share(sref)
                }
                _ => panic!("share-from expects (owner-id value nbits) all as numbers"),
            }
        });
        self.store.borrow_mut().insert(
            "share-from".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(share_from_fn))),
        );

        // reconstruct builtin
        let reconstruct_backend = backend.clone();
        let reconstruct_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 1 {
                panic!("reconstruct expects 1 arg");
            }
            match &args[0] {
                Types::Share(sref) => {
                    let mut be = reconstruct_backend
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    // reveal_to = None means broadcast: all parties participate and
                    // everyone reconstructs.  This is correct for both SimBackend
                    // and TransportBackend (in the single-party REPL, SimBackend
                    // ignores reveal_to anyway).
                    let plain = be.reveal(sref, None);
                    if sref.fixed {
                        // Q16.16 share: sign-extend the raw scaled value.
                        Types::Fixed(fr_to_i64_signed(plain, sref.nbits))
                    } else {
                        Types::Num(plain as i64)
                    }
                }
                Types::Num(n) => Types::Num(*n),
                _ => panic!("reconstruct expects a shared value"),
            }
        });
        self.store.borrow_mut().insert(
            "reconstruct".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(reconstruct_fn))),
        );

        // ── Vector builtins: dot, vsum, matvec ────────────────────────────────
        // Native replacements for the recursive car/cdr dot-product pattern.
        // Elements may be Num, Fixed, or Share; a user lambda named `dot`
        // shadows the builtin (letrec bindings overwrite store entries).
        let dot_backend = backend.clone();
        let dot_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("dot expects 2 args: (dot vec-a vec-b)");
            }
            match (&args[0], &args[1]) {
                (Types::List(a), Types::List(b)) => vec_dot(&dot_backend, a, b),
                _ => panic!("dot expects two lists"),
            }
        });
        self.store.borrow_mut().insert(
            "dot".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(dot_fn))),
        );

        let vsum_backend = backend.clone();
        let vsum_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 1 {
                panic!("vsum expects 1 arg: (vsum vec)");
            }
            match &args[0] {
                Types::List(items) => {
                    let mut acc: Option<Types> = None;
                    for item in items {
                        acc = Some(match acc {
                            None => item.clone(),
                            Some(s) => vec_add(&vsum_backend, &s, item),
                        });
                    }
                    acc.unwrap_or(Types::Num(0))
                }
                _ => panic!("vsum expects a list"),
            }
        });
        self.store.borrow_mut().insert(
            "vsum".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(vsum_fn))),
        );

        let matvec_backend = backend.clone();
        let matvec_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("matvec expects 2 args: (matvec matrix vec)");
            }
            match (&args[0], &args[1]) {
                (Types::List(rows), Types::List(v)) => {
                    Types::List(matvec_impl(&matvec_backend, rows, v))
                }
                _ => panic!("matvec expects a list of lists and a list"),
            }
        });
        self.store.borrow_mut().insert(
            "matvec".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(matvec_fn))),
        );

        // nn-load-model builtin: (nn-load-model "path.json") -> list of
        // layers, each a 3-element list (weights bias activation) with
        // weights/bias as nested Fixed lists — see `crate::ml`.
        let nn_load_model_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 1 {
                panic!("nn-load-model expects 1 arg: (nn-load-model path)");
            }
            let path = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => panic!("nn-load-model expects a string path"),
            };
            let net = crate::ml::QuantizedNetwork::load_json(&path)
                .unwrap_or_else(|e| panic!("nn-load-model: {e}"));
            if net.scale_bits != 16 {
                panic!(
                    "nn-load-model: model was quantized at {} scale bits, but Baru's \
                     Types::Fixed is always Q16.16 — re-run `baru quantize` with \
                     --scale-bits 16",
                    net.scale_bits
                );
            }
            let layers: Vec<Types> = net
                .layers
                .iter()
                .map(|layer| {
                    let weights = Types::List(
                        layer
                            .weights
                            .iter()
                            .map(|row| Types::List(row.iter().map(|&w| Types::Fixed(w)).collect()))
                            .collect(),
                    );
                    let bias =
                        Types::List(layer.bias.iter().map(|&b| Types::Fixed(b)).collect());
                    let activation = Types::Str(
                        match layer.activation {
                            crate::ml::Activation::Relu => "relu",
                            crate::ml::Activation::Identity => "identity",
                        }
                        .to_string(),
                    );
                    Types::List(vec![weights, bias, activation])
                })
                .collect();
            Types::List(layers)
        });
        self.store.borrow_mut().insert(
            "nn-load-model".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(nn_load_model_fn))),
        );

        // nn-forward builtin: (nn-forward layers input) -> output vector.
        // Runs a full multi-layer forward pass (matvec + bias-add +
        // activation per layer, chained) over whatever `nn-load-model`
        // produced. Works for public (Num/Fixed) and private (Share) input
        // alike, since matvec_impl/vec_add/vec_relu all dispatch per-element
        // on the actual runtime type — a private input against a public
        // model, or an all-private evaluation, both fall out of the same
        // code path with no separate "private mode" to opt into.
        let nn_forward_backend = backend.clone();
        let nn_forward_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("nn-forward expects 2 args: (nn-forward layers input)");
            }
            let (layers, mut cur) = match (&args[0], &args[1]) {
                (Types::List(layers), Types::List(input)) => (layers.clone(), input.clone()),
                _ => panic!("nn-forward expects (list-of-layers input-list)"),
            };
            for layer in &layers {
                let parts = match layer {
                    Types::List(parts) if parts.len() == 3 => parts,
                    _ => panic!(
                        "nn-forward: each layer must be a 3-element list \
                         (weights bias activation), as produced by nn-load-model"
                    ),
                };
                let weights = match &parts[0] {
                    Types::List(w) => w,
                    _ => panic!("nn-forward: layer weights must be a list"),
                };
                let bias = match &parts[1] {
                    Types::List(b) => b,
                    _ => panic!("nn-forward: layer bias must be a list"),
                };
                let activation = match &parts[2] {
                    Types::Str(s) => s.as_str(),
                    _ => panic!("nn-forward: layer activation must be a string"),
                };
                let pre = matvec_impl(&nn_forward_backend, weights, &cur);
                let biased: Vec<Types> = pre
                    .iter()
                    .zip(bias.iter())
                    .map(|(p, b)| vec_add(&nn_forward_backend, p, b))
                    .collect();
                cur = match activation {
                    "relu" => biased
                        .iter()
                        .map(|t| vec_relu(&nn_forward_backend, t))
                        .collect(),
                    "identity" => biased,
                    other => panic!("nn-forward: unknown activation '{other}'"),
                };
            }
            Types::List(cur)
        });
        self.store.borrow_mut().insert(
            "nn-forward".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(nn_forward_fn))),
        );

        // a2b builtin
        let a2b_backend = backend.clone();
        let a2b_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("a2b expects 2 args: (a nbits)");
            }
            match (&args[0], &args[1]) {
                (Types::Share(sref), Types::Num(nbits)) => {
                    let mut be = a2b_backend.lock().unwrap_or_else(|e| e.into_inner());
                    let bits = be.a2b(sref, *nbits as usize);
                    let svs: Vec<Types> = bits.into_iter().map(Types::Share).collect();
                    Types::List(svs)
                }
                _ => panic!("a2b expects (share nbits)"),
            }
        });
        self.store.borrow_mut().insert(
            "a2b".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(a2b_fn))),
        );

        // b2a builtin
        let b2a_backend = backend.clone();
        let b2a_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 1 {
                panic!("b2a expects 1 arg: (bits-list)");
            }
            match &args[0] {
                Types::List(list) => {
                    let bits_refs: Vec<crate::interpreter::backend::ShareRef> = list
                        .iter()
                        .map(|v| {
                            if let Types::Share(sr) = v {
                                sr.clone()
                            } else {
                                panic!("b2a expects list of share bits")
                            }
                        })
                        .collect();
                    let mut be = b2a_backend.lock().unwrap_or_else(|e| e.into_inner());
                    let out = be.b2a(&bits_refs);
                    Types::Share(out)
                }
                _ => panic!("b2a expects a list of bit shares"),
            }
        });
        self.store.borrow_mut().insert(
            "b2a".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(b2a_fn))),
        );

        // list builtin: (list v1 v2 ...) → Types::List
        // Collects all arguments into a list.  Useful for building ciphertext
        // lists to pass to fhe-eval's enc-cts argument.
        let list_fn = Arc::new(move |args: Vec<Types>| -> Types { Types::List(args) });
        self.store.borrow_mut().insert(
            "list".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(list_fn))),
        );

        // zkp-compile builtin: (zkp-compile 'expr '("x" "y"))
        // Compiles a quoted expression to a Binius64 circuit.
        // Returns a JSON string representing the compiled Circuit.
        let zkp_compile_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("zkp-compile expects 2 args: (zkp-compile 'expr '(\"x\" \"y\"))");
            }
            // First arg: quoted expression
            let expr = match &args[0] {
                Types::Quoted(e) => *e.clone(),
                _ => panic!("zkp-compile: first arg must be a quoted expression (use ')"),
            };
            // Second arg: list of private variable names
            // Accepts either Types::List (from evaluated list) or Types::Quoted
            // (from quote shorthand '("x" "y")), or an empty quoted seq.
            let private_vars: Vec<String> = match &args[1] {
                Types::List(items) => items
                    .iter()
                    .map(|v| match v {
                        Types::Str(s) => s.clone(),
                        _ => panic!("zkp-compile: private-vars must be a list of strings"),
                    })
                    .collect(),
                Types::Quoted(inner) => {
                    // '("x" "y") parses as Quote(Call(Str("x"), [Str("y"), ...]))
                    // or Quote(Seq([...])) for zero/multiple items
                    match inner.as_ref() {
                        Expr::Call(head, rest) => {
                            let mut vars = Vec::new();
                            if let Expr::Str(s) = head.as_ref() {
                                vars.push(s.clone());
                            } else {
                                panic!("zkp-compile: private-vars must be a list of strings");
                            }
                            for e in rest {
                                if let Expr::Str(s) = e {
                                    vars.push(s.clone());
                                } else {
                                    panic!("zkp-compile: private-vars must be a list of strings");
                                }
                            }
                            vars
                        }
                        Expr::Str(s) => vec![s.clone()],
                        Expr::Seq(items) if items.is_empty() => vec![],
                        _ => panic!("zkp-compile: private-vars must be a quoted list of strings like '(\"x\" \"y\")"),
                    }
                }
                _ => panic!("zkp-compile: second arg must be a list of strings or a quoted list like '(\"x\")"),
            };
            let backend = crate::zkp::binius64_backend::Binius64Backend::new();
            use crate::zkp::ZKPBackend;
            let circuit = backend
                .compile_circuit(&expr, &private_vars)
                .unwrap_or_else(|e| panic!("zkp-compile failed: {}", e));
            let json = serde_json::to_string(&circuit)
                .unwrap_or_else(|e| panic!("zkp-compile: failed to serialize circuit: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "zkp-compile".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(zkp_compile_fn))),
        );

        // zkp-prove builtin: (zkp-prove circuit-str '("x") '(5) '("y") '(6))
        // circuit-str: string returned by zkp-compile (store with let)
        // private-keys: quoted list of witness variable names  e.g. '("x" "z")
        // private-vals: quoted list of witness values          e.g. '(5 10)
        // public-keys:  quoted list of public output names     e.g. '("y")
        // public-vals:  quoted list of public output values    e.g. '(6)
        // Returns a JSON string representing the ZKProof.
        let zkp_prove_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 5 {
                panic!("zkp-prove expects 5 args: (zkp-prove circuit-str '(priv-keys) '(priv-vals) '(pub-keys) '(pub-vals))");
            }
            let circuit_json = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => {
                    panic!("zkp-prove: first arg must be a circuit JSON string (from zkp-compile)")
                }
            };
            let circuit: crate::zkp::Circuit = serde_json::from_str(&circuit_json)
                .unwrap_or_else(|e| panic!("zkp-prove: invalid circuit JSON: {}", e));

            let private_keys = decode_quoted_str_list(&args[1], "zkp-prove private-keys");
            let private_nums = decode_quoted_num_list(&args[2], "zkp-prove private-vals");
            let public_keys = decode_quoted_str_list(&args[3], "zkp-prove public-keys");
            let public_nums = decode_quoted_num_list(&args[4], "zkp-prove public-vals");

            if private_keys.len() != private_nums.len() {
                panic!("zkp-prove: private-keys and private-vals must have the same length");
            }
            if public_keys.len() != public_nums.len() {
                panic!("zkp-prove: public-keys and public-vals must have the same length");
            }

            let private_vals: std::collections::HashMap<String, i64> =
                private_keys.into_iter().zip(private_nums).collect();
            let public_vals: std::collections::HashMap<String, i64> =
                public_keys.into_iter().zip(public_nums).collect();

            let backend = crate::zkp::binius64_backend::Binius64Backend::new();
            use crate::zkp::ZKPBackend;
            let proof = backend
                .prove(&circuit, &private_vals, &public_vals)
                .unwrap_or_else(|e| panic!("zkp-prove failed: {}", e));
            let json = serde_json::to_string(&proof)
                .unwrap_or_else(|e| panic!("zkp-prove: failed to serialize proof: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "zkp-prove".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(zkp_prove_fn))),
        );

        // zkp-verify builtin: (zkp-verify circuit-str proof-str '("y") '(6))
        // Returns 1 if valid, 0 if invalid, panics on structural errors.
        let zkp_verify_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 4 {
                panic!("zkp-verify expects 4 args: (zkp-verify circuit-str proof-str '(pub-keys) '(pub-vals))");
            }
            let circuit_json = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => panic!("zkp-verify: first arg must be a circuit JSON string"),
            };
            let proof_json = match &args[1] {
                Types::Str(s) => s.clone(),
                _ => panic!("zkp-verify: second arg must be a proof JSON string"),
            };
            let circuit: crate::zkp::Circuit = serde_json::from_str(&circuit_json)
                .unwrap_or_else(|e| panic!("zkp-verify: invalid circuit JSON: {}", e));
            let proof: crate::zkp::ZKProof = serde_json::from_str(&proof_json)
                .unwrap_or_else(|e| panic!("zkp-verify: invalid proof JSON: {}", e));

            let public_keys = decode_quoted_str_list(&args[2], "zkp-verify public-keys");
            let public_nums = decode_quoted_num_list(&args[3], "zkp-verify public-vals");

            if public_keys.len() != public_nums.len() {
                panic!("zkp-verify: public-keys and public-vals must have the same length");
            }
            let public_vals: std::collections::HashMap<String, i64> =
                public_keys.into_iter().zip(public_nums).collect();

            let backend = crate::zkp::binius64_backend::Binius64Backend::new();
            use crate::zkp::ZKPBackend;
            match backend.verify(&circuit, &proof, &public_vals) {
                Ok(true) => Types::Num(1),
                Ok(false) => Types::Num(0),
                Err(e) => panic!("zkp-verify failed: {}", e),
            }
        });
        self.store.borrow_mut().insert(
            "zkp-verify".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(zkp_verify_fn))),
        );

        // proof-save builtin: (proof-save proof-json filename)
        // Saves a proof JSON string to a file using the serialization module.
        let proof_save_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("proof-save expects 2 args: (proof-save proof-json filename)");
            }
            match (&args[0], &args[1]) {
                (Types::Str(proof_json), Types::Str(filename)) => {
                    let proof: crate::zkp::ZKProof = serde_json::from_str(proof_json)
                        .unwrap_or_else(|e| panic!("proof-save: invalid proof JSON: {}", e));
                    crate::zkp::serialization::save_proof(&proof, filename)
                        .unwrap_or_else(|e| panic!("proof-save failed: {}", e));
                    Types::Str(format!("proof saved to {}", filename))
                }
                _ => panic!("proof-save expects (proof-json-string filename-string)"),
            }
        });
        self.store.borrow_mut().insert(
            "proof-save".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(proof_save_fn))),
        );

        // proof-load builtin: (proof-load filename)
        // Loads a proof from a JSON file, returns proof JSON string.
        let proof_load_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 1 {
                panic!("proof-load expects 1 arg: (proof-load filename)");
            }
            match &args[0] {
                Types::Str(filename) => {
                    let proof = crate::zkp::serialization::load_proof(filename)
                        .unwrap_or_else(|e| panic!("proof-load failed: {}", e));
                    let json = serde_json::to_string(&proof)
                        .unwrap_or_else(|e| panic!("proof-load: failed to serialize: {}", e));
                    Types::Str(json)
                }
                _ => panic!("proof-load expects a filename string"),
            }
        });
        self.store.borrow_mut().insert(
            "proof-load".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(proof_load_fn))),
        );

        // ── Fixed-point conversion builtins ───────────────────────────────────
        //
        // (int->fixed n)   → Q16.16 fixed-point from integer
        // (fixed->int f)   → truncate fixed-point to integer (floor toward zero)
        // (fixed->str f)   → decimal string representation, e.g. "3.1416"

        let int_to_fixed_fn = Arc::new(move |args: Vec<Types>| -> Types {
            use crate::interpreter::ast::FIXED_SCALE;
            if args.len() != 1 {
                panic!("int->fixed expects 1 arg: (int->fixed n)");
            }
            match &args[0] {
                Types::Num(n) => Types::Fixed(n * FIXED_SCALE),
                Types::Fixed(v) => Types::Fixed(*v),
                _ => panic!("int->fixed: argument must be a number"),
            }
        });
        self.store.borrow_mut().insert(
            "int->fixed".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(int_to_fixed_fn))),
        );

        let fixed_to_int_fn = Arc::new(move |args: Vec<Types>| -> Types {
            use crate::interpreter::ast::FIXED_SCALE;
            if args.len() != 1 {
                panic!("fixed->int expects 1 arg: (fixed->int f)");
            }
            match &args[0] {
                Types::Fixed(v) => Types::Num(v / FIXED_SCALE),
                Types::Num(n) => Types::Num(*n),
                _ => panic!("fixed->int: argument must be a fixed-point or integer"),
            }
        });
        self.store.borrow_mut().insert(
            "fixed->int".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fixed_to_int_fn))),
        );

        let fixed_to_str_fn = Arc::new(move |args: Vec<Types>| -> Types {
            use crate::interpreter::ast::FIXED_SCALE;
            if args.len() != 1 {
                panic!("fixed->str expects 1 arg: (fixed->str f)");
            }
            let v = match &args[0] {
                Types::Fixed(v) => *v,
                Types::Num(n) => n * FIXED_SCALE,
                _ => panic!("fixed->str: argument must be a fixed-point or integer"),
            };
            let whole = v / FIXED_SCALE;
            let frac = (v % FIXED_SCALE).abs();
            let frac_digits = (frac * 10_000) / FIXED_SCALE;
            Types::Str(format!("{}.{:04}", whole, frac_digits))
        });
        self.store.borrow_mut().insert(
            "fixed->str".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fixed_to_str_fn))),
        );

        // ── FHE builtins ─────────────────────────────────────────────────────
        //
        // (fhe-keygen)
        //   → returns a JSON string containing the key-pair (FheKeyPair).
        //     Store it with let: (let ((keys (fhe-keygen))) ...)
        //
        // (fhe-enc keys value)
        //   → encrypts a Num (or Fixed) using the client key embedded in `keys`
        //   → returns Types::FheCiphertext
        //
        // (fhe-dec keys ciphertext)
        //   → decrypts a Types::FheCiphertext → Types::Num
        //
        // (fhe-compile 'expr '("enc-var1" "enc-var2"))
        //   → compiles the expression; the second arg names which variables will
        //     be supplied as ciphertexts at eval time (analogous to zkp-compile's
        //     private-vars list).
        //   → returns a JSON string (FheCircuit)
        //
        // (fhe-eval circuit-str keys '("pub-x") '(3) '("enc-y") '(ct))
        //   → evaluates circuit homomorphically
        //   → public-keys / public-vals: plaintext inputs
        //   → enc-keys / enc-cts:        ciphertext inputs (Types::FheCiphertext)
        //   → returns Types::FheCiphertext (the encrypted result)

        // fhe-keygen  (mock plaintext backend — instant, no crypto)
        let fhe_keygen_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if !args.is_empty() {
                panic!("fhe-keygen takes no arguments");
            }
            let be = crate::fhe::MockFheBackend;
            use crate::fhe::FheBackend;
            let (ck, sk) = be.keygen();
            let kp = crate::fhe::backend::FheKeyPair {
                backend_id: be.backend_id().to_string(),
                client_key_bytes: ck,
                server_key_bytes: sk,
            };
            let json = serde_json::to_string(&kp)
                .unwrap_or_else(|e| panic!("fhe-keygen: serialise error: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "fhe-keygen".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_keygen_fn))),
        );

        // fhe-keygen-tfhe  (real TFHE-rs backend — slow first call ~3 s)
        // Also installs the server key into the TFHE-rs global thread-local so
        // subsequent fhe-enc / fhe-eval calls work without extra setup.
        let fhe_keygen_tfhe_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if !args.is_empty() {
                panic!("fhe-keygen-tfhe takes no arguments");
            }
            use crate::fhe::FheBackend;
            let be = crate::fhe::TfheBackend::new();
            let (ck, sk) = be.keygen();
            // Install server key globally so all subsequent gates work.
            crate::fhe::TfheBackend::install_server_key(&sk);
            let kp = crate::fhe::backend::FheKeyPair {
                backend_id: be.backend_id().to_string(),
                client_key_bytes: ck,
                server_key_bytes: sk,
            };
            let json = serde_json::to_string(&kp)
                .unwrap_or_else(|e| panic!("fhe-keygen-tfhe: serialise error: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "fhe-keygen-tfhe".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_keygen_tfhe_fn))),
        );

        // fhe-keygen-ckks  (SEAL CKKS backend — approximate real arithmetic;
        // pair with circuits from fhe-compile-ckks)
        let fhe_keygen_ckks_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if !args.is_empty() {
                panic!("fhe-keygen-ckks takes no arguments");
            }
            use crate::fhe::RealFheBackend;
            let be = crate::fhe::CkksBackend::new();
            let (ck, sk) = be.keygen();
            let kp = crate::fhe::backend::FheKeyPair {
                backend_id: be.backend_id().to_string(),
                client_key_bytes: ck,
                server_key_bytes: sk,
            };
            let json = serde_json::to_string(&kp)
                .unwrap_or_else(|e| panic!("fhe-keygen-ckks: serialise error: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "fhe-keygen-ckks".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_keygen_ckks_fn))),
        );

        // fhe-keygen-ckks-mock  (exact f64 twin of the CKKS backend)
        let fhe_keygen_ckks_mock_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if !args.is_empty() {
                panic!("fhe-keygen-ckks-mock takes no arguments");
            }
            use crate::fhe::RealFheBackend;
            let be = crate::fhe::MockCkksBackend;
            let (ck, sk) = be.keygen();
            let kp = crate::fhe::backend::FheKeyPair {
                backend_id: be.backend_id().to_string(),
                client_key_bytes: ck,
                server_key_bytes: sk,
            };
            let json = serde_json::to_string(&kp)
                .unwrap_or_else(|e| panic!("fhe-keygen-ckks-mock: serialise error: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "fhe-keygen-ckks-mock".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_keygen_ckks_mock_fn))),
        );

        // fhe-enc
        let fhe_enc_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("fhe-enc expects 2 args: (fhe-enc keys value)");
            }
            let keys_json = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => panic!("fhe-enc: first arg must be a key-pair JSON string (from fhe-keygen)"),
            };
            let kp: crate::fhe::backend::FheKeyPair = serde_json::from_str(&keys_json)
                .unwrap_or_else(|e| panic!("fhe-enc: invalid key JSON: {}", e));
            let (plaintext, is_fixed) = match &args[1] {
                Types::Num(n) => (*n as u64, false),
                Types::Fixed(v) => (*v as u64, true),
                Types::Bool(b) => {
                    if *b {
                        (1u64, false)
                    } else {
                        (0u64, false)
                    }
                }
                _ => panic!("fhe-enc: second arg must be a number or boolean"),
            };
            use crate::fhe::FheBackend;
            match kp.backend_id.as_str() {
                "tfhe-rs-v1.6" => {
                    let be = crate::fhe::TfheBackend::new();
                    let ct = be.encrypt(&kp.client_key_bytes, plaintext);
                    let handle = crate::fhe::backend::FheCiphertextHandle {
                        backend_id: be.backend_id().to_string(),
                        bytes: be.serialize_ct(&ct),
                        fixed: is_fixed,
                        nbits: None,
                    };
                    Types::FheCiphertext(handle)
                }
                "seal-ckks" | "ckks-mock" => {
                    // Real-arithmetic backends encrypt the true value: fixed
                    // Q16.16 integers cross the boundary as v / 2^16.
                    let real = if is_fixed {
                        plaintext as i64 as f64 / 65536.0
                    } else {
                        plaintext as i64 as f64
                    };
                    fn enc<B: crate::fhe::RealFheBackend>(
                        be: &B,
                        ck: &[u8],
                        v: f64,
                        fixed: bool,
                    ) -> Types {
                        let ct = be.encrypt(ck, v);
                        Types::FheCiphertext(crate::fhe::backend::FheCiphertextHandle {
                            backend_id: be.backend_id().to_string(),
                            bytes: be.serialize_ct(&ct),
                            fixed,
                            nbits: None,
                        })
                    }
                    if kp.backend_id == "seal-ckks" {
                        enc(
                            &crate::fhe::CkksBackend::new(),
                            &kp.client_key_bytes,
                            real,
                            is_fixed,
                        )
                    } else {
                        enc(
                            &crate::fhe::MockCkksBackend,
                            &kp.client_key_bytes,
                            real,
                            is_fixed,
                        )
                    }
                }
                _ => {
                    let be = crate::fhe::MockFheBackend;
                    let ct = be.encrypt(&kp.client_key_bytes, plaintext);
                    let handle = crate::fhe::backend::FheCiphertextHandle {
                        backend_id: be.backend_id().to_string(),
                        bytes: be.serialize_ct(&ct),
                        fixed: is_fixed,
                        nbits: None,
                    };
                    Types::FheCiphertext(handle)
                }
            }
        });
        self.store.borrow_mut().insert(
            "fhe-enc".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_enc_fn))),
        );

        // fhe-dec
        let fhe_dec_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("fhe-dec expects 2 args: (fhe-dec keys ciphertext)");
            }
            let keys_json = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => panic!("fhe-dec: first arg must be a key-pair JSON string"),
            };
            let kp: crate::fhe::backend::FheKeyPair = serde_json::from_str(&keys_json)
                .unwrap_or_else(|e| panic!("fhe-dec: invalid key JSON: {}", e));
            let handle = match &args[1] {
                Types::FheCiphertext(h) => h.clone(),
                _ => panic!("fhe-dec: second arg must be a FheCiphertext"),
            };
            // Scheme-switched ciphertexts (share->fhe) carry the share's bit
            // width: the plaintext is a residue class mod 2^nbits (possibly
            // with share-sum carries above it), so reduce and sign-extend to
            // recover the signed value the share represented.
            let reduce_signed = |raw: i64, nbits: usize| -> i64 {
                if nbits >= 64 {
                    return raw;
                }
                let m = 1u128 << nbits;
                let v = (raw as u64 as u128) % m;
                if v >= m / 2 {
                    (v as i128 - m as i128) as i64
                } else {
                    v as i64
                }
            };
            use crate::fhe::FheBackend;
            let plain = match kp.backend_id.as_str() {
                "seal-ckks" | "ckks-mock" => {
                    // Real-arithmetic decrypt: recover the approximate real
                    // and re-enter the Q16.16 domain (or round to int).
                    use crate::fhe::RealFheBackend;
                    let v = if kp.backend_id == "seal-ckks" {
                        let be = crate::fhe::CkksBackend::new();
                        let ct = be.deserialize_ct(&handle.bytes);
                        be.decrypt(&kp.client_key_bytes, &ct)
                    } else {
                        let be = crate::fhe::MockCkksBackend;
                        let ct = be.deserialize_ct(&handle.bytes);
                        be.decrypt(&kp.client_key_bytes, &ct)
                    };
                    if let Some(nb) = handle.nbits {
                        // Converted share: v is the (carried) raw residue,
                        // already in the scaled domain if fixed.
                        let raw = reduce_signed(v.round() as i64, nb);
                        return if handle.fixed {
                            Types::Fixed(raw)
                        } else {
                            Types::Num(raw)
                        };
                    }
                    return if handle.fixed {
                        Types::Fixed((v * 65536.0).round() as i64)
                    } else {
                        Types::Num(v.round() as i64)
                    };
                }
                "tfhe-rs-v1.6" => {
                    let be = crate::fhe::TfheBackend::new();
                    let ct = be.deserialize_ct(&handle.bytes);
                    be.decrypt(&kp.client_key_bytes, &ct) as i64
                }
                _ => {
                    let be = crate::fhe::MockFheBackend;
                    let ct = be.deserialize_ct(&handle.bytes);
                    be.decrypt(&kp.client_key_bytes, &ct) as i64
                }
            };
            let plain = match handle.nbits {
                Some(nb) => reduce_signed(plain, nb),
                None => plain,
            };
            if handle.fixed {
                Types::Fixed(plain)
            } else {
                Types::Num(plain)
            }
        });
        self.store.borrow_mut().insert(
            "fhe-dec".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_dec_fn))),
        );

        // fhe-compile
        let fhe_compile_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("fhe-compile expects 2 args: (fhe-compile 'expr '(\"enc-var\"))");
            }
            let expr = match &args[0] {
                Types::Quoted(e) => *e.clone(),
                _ => panic!("fhe-compile: first arg must be a quoted expression"),
            };
            let enc_vars: Vec<String> = match &args[1] {
                Types::List(items) => items
                    .iter()
                    .map(|v| match v {
                        Types::Str(s) => s.clone(),
                        _ => panic!("fhe-compile: enc-vars must be a list of strings"),
                    })
                    .collect(),
                Types::Quoted(inner) => match inner.as_ref() {
                    Expr::Str(s) => vec![s.clone()],
                    Expr::Seq(items) if items.is_empty() => vec![],
                    Expr::Call(head, rest) => {
                        let mut vars = Vec::new();
                        if let Expr::Str(s) = head.as_ref() {
                            vars.push(s.clone());
                        }
                        for e in rest {
                            if let Expr::Str(s) = e {
                                vars.push(s.clone());
                            }
                        }
                        vars
                    }
                    _ => panic!("fhe-compile: enc-vars must be a quoted list of strings"),
                },
                _ => panic!("fhe-compile: second arg must be a list of strings"),
            };
            let mut cc = crate::zkp::circuit_compiler::CircuitCompiler::new();
            let mut circuit = cc
                .compile_fhe(&expr)
                .unwrap_or_else(|e| panic!("fhe-compile failed: {}", e));
            // Move declared enc-vars from public_inputs to encrypted_inputs
            circuit.encrypted_inputs = enc_vars.clone();
            circuit.public_inputs.retain(|v| !enc_vars.contains(v));
            let json = serde_json::to_string(&circuit)
                .unwrap_or_else(|e| panic!("fhe-compile: serialise error: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "fhe-compile".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_compile_fn))),
        );

        // fhe-compile-ckks — like fhe-compile, but in real-arithmetic mode:
        // fixed-point literals keep their true value and no Q16.16 gadgets
        // are emitted.  Required for evaluation on CKKS backends.
        let fhe_compile_ckks_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("fhe-compile-ckks expects 2 args: (fhe-compile-ckks 'expr '(\"enc-var\"))");
            }
            let expr = match &args[0] {
                Types::Quoted(e) => *e.clone(),
                _ => panic!("fhe-compile-ckks: first arg must be a quoted expression"),
            };
            let enc_vars: Vec<String> = match &args[1] {
                Types::List(items) => items
                    .iter()
                    .map(|v| match v {
                        Types::Str(s) => s.clone(),
                        _ => panic!("fhe-compile-ckks: enc-vars must be a list of strings"),
                    })
                    .collect(),
                Types::Quoted(inner) => match inner.as_ref() {
                    Expr::Str(s) => vec![s.clone()],
                    Expr::Seq(items) if items.is_empty() => vec![],
                    Expr::Call(head, rest) => {
                        let mut vars = Vec::new();
                        if let Expr::Str(s) = head.as_ref() {
                            vars.push(s.clone());
                        }
                        for e in rest {
                            if let Expr::Str(s) = e {
                                vars.push(s.clone());
                            }
                        }
                        vars
                    }
                    _ => panic!("fhe-compile-ckks: enc-vars must be a quoted list of strings"),
                },
                _ => panic!("fhe-compile-ckks: second arg must be a list of strings"),
            };
            let mut cc = crate::zkp::circuit_compiler::CircuitCompiler::new_real();
            let mut circuit = cc
                .compile_fhe(&expr)
                .unwrap_or_else(|e| panic!("fhe-compile-ckks failed: {}", e));
            circuit.encrypted_inputs = enc_vars.clone();
            circuit.public_inputs.retain(|v| !enc_vars.contains(v));
            let json = serde_json::to_string(&circuit)
                .unwrap_or_else(|e| panic!("fhe-compile-ckks: serialise error: {}", e));
            Types::Str(json)
        });
        self.store.borrow_mut().insert(
            "fhe-compile-ckks".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_compile_ckks_fn))),
        );

        // fhe-eval
        // (fhe-eval circuit-str keys '("pub-x") '(3) '("enc-y") '(ct))
        let fhe_eval_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 6 {
                panic!(
                    "fhe-eval expects 6 args: \
                     (fhe-eval circuit-str keys '(pub-keys) '(pub-vals) '(enc-keys) '(enc-cts))"
                );
            }
            let circuit_json = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => panic!("fhe-eval: first arg must be a circuit JSON string (from fhe-compile)"),
            };
            let mut circuit: crate::fhe::backend::FheCircuit = serde_json::from_str(&circuit_json)
                .unwrap_or_else(|e| panic!("fhe-eval: invalid circuit JSON: {}", e));
            let keys_json = match &args[1] {
                Types::Str(s) => s.clone(),
                _ => panic!("fhe-eval: second arg must be a key-pair JSON string"),
            };
            let kp: crate::fhe::backend::FheKeyPair = serde_json::from_str(&keys_json)
                .unwrap_or_else(|e| panic!("fhe-eval: invalid key JSON: {}", e));

            let pub_keys = decode_quoted_str_list(&args[2], "fhe-eval pub-keys");
            let pub_nums = decode_quoted_num_list(&args[3], "fhe-eval pub-vals");
            let enc_keys = decode_quoted_str_list(&args[4], "fhe-eval enc-keys");
            // enc-cts: list of FheCiphertext values
            // Accepts Types::List (from `list` builtin) or an empty quoted seq '()
            let enc_cts: Vec<crate::fhe::backend::FheCiphertextHandle> = match &args[5] {
                Types::List(items) => items
                    .iter()
                    .map(|v| match v {
                        Types::FheCiphertext(h) => h.clone(),
                        _ => panic!("fhe-eval: enc-cts must be a list of FheCiphertext values"),
                    })
                    .collect(),
                Types::Quoted(inner) => match inner.as_ref() {
                    Expr::Seq(items) if items.is_empty() => vec![],
                    _ => panic!("fhe-eval: enc-cts must be a list of FheCiphertext values (use (list ct1 ct2 ...) or '() for empty)"),
                },
                _ => panic!("fhe-eval: enc-cts must be a list of FheCiphertext values (use (list ct1 ct2 ...) or '() for empty)"),
            };

            if pub_keys.len() != pub_nums.len() {
                panic!("fhe-eval: pub-keys and pub-vals must have the same length");
            }
            if enc_keys.len() != enc_cts.len() {
                panic!("fhe-eval: enc-keys and enc-cts must have the same length");
            }

            // Update circuit's input classification to match caller's intent
            circuit.public_inputs = pub_keys.clone();
            circuit.encrypted_inputs = enc_keys.clone();

            use crate::fhe::FheBackend;
            let pub_map: std::collections::HashMap<String, i64> =
                pub_keys.into_iter().zip(pub_nums).collect();

            match kp.backend_id.as_str() {
                "seal-ckks" | "ckks-mock" => {
                    // Real-arithmetic path: public inputs cross as true reals.
                    let pub_map_f: std::collections::HashMap<String, f64> = pub_map
                        .iter()
                        .map(|(k, v)| (k.clone(), *v as f64))
                        .collect();
                    fn run<B: crate::fhe::RealFheBackend>(
                        be: &B,
                        circuit: &crate::fhe::backend::FheCircuit,
                        pub_map: &std::collections::HashMap<String, f64>,
                        enc_keys: Vec<String>,
                        enc_cts: &[crate::fhe::backend::FheCiphertextHandle],
                    ) -> Types {
                        let enc_map: std::collections::HashMap<String, B::Ct> = enc_keys
                            .into_iter()
                            .zip(enc_cts.iter().map(|h| be.deserialize_ct(&h.bytes)))
                            .collect();
                        let result_ct = crate::fhe::circuit_emitter::evaluate_fhe_real(
                            circuit, pub_map, &enc_map, be,
                        );
                        Types::FheCiphertext(crate::fhe::backend::FheCiphertextHandle {
                            backend_id: be.backend_id().to_string(),
                            bytes: be.serialize_ct(&result_ct),
                            fixed: circuit.output_fixed,
                            nbits: None,
                        })
                    }
                    if kp.backend_id == "seal-ckks" {
                        let be = crate::fhe::CkksBackend::with_server_keys(&kp.server_key_bytes);
                        run(&be, &circuit, &pub_map_f, enc_keys, &enc_cts)
                    } else {
                        run(
                            &crate::fhe::MockCkksBackend,
                            &circuit,
                            &pub_map_f,
                            enc_keys,
                            &enc_cts,
                        )
                    }
                }
                "tfhe-rs-v1.6" => {
                    // Ensure the server key is installed (caller may have used
                    // fhe-keygen-tfhe which installs it, but re-install to be safe).
                    crate::fhe::TfheBackend::install_server_key(&kp.server_key_bytes);
                    let be = crate::fhe::TfheBackend::new();
                    let enc_map: std::collections::HashMap<String, _> = enc_keys
                        .into_iter()
                        .zip(enc_cts.iter().map(|h| be.deserialize_ct(&h.bytes)))
                        .collect();
                    let result_ct = crate::fhe::circuit_emitter::evaluate_fhe(
                        &circuit, &pub_map, &enc_map, &be,
                    );
                    let handle = crate::fhe::backend::FheCiphertextHandle {
                        backend_id: be.backend_id().to_string(),
                        bytes: be.serialize_ct(&result_ct),
                        fixed: circuit.output_fixed,
                        nbits: None,
                    };
                    Types::FheCiphertext(handle)
                }
                _ => {
                    let be = crate::fhe::MockFheBackend;
                    let enc_map: std::collections::HashMap<String, u64> = enc_keys
                        .into_iter()
                        .zip(enc_cts.iter().map(|h| be.deserialize_ct(&h.bytes)))
                        .collect();
                    let result_ct = crate::fhe::circuit_emitter::evaluate_fhe(
                        &circuit, &pub_map, &enc_map, &be,
                    );
                    let handle = crate::fhe::backend::FheCiphertextHandle {
                        backend_id: be.backend_id().to_string(),
                        bytes: be.serialize_ct(&result_ct),
                        fixed: circuit.output_fixed,
                        nbits: None,
                    };
                    Types::FheCiphertext(handle)
                }
            }
        });
        self.store.borrow_mut().insert(
            "fhe-eval".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_eval_fn))),
        );

        // ── Scheme switching: MPC shares <-> FHE ciphertexts ─────────────────
        //
        // (share->fhe keys share)
        //   → converts a live MPC share into an FHE ciphertext of the same
        //     value: each party encrypts its additive component, the sum is
        //     taken homomorphically.  The value is never reconstructed.
        //
        // (fhe->share keys ciphertext nbits)
        //   → converts an FHE ciphertext into a live MPC share via
        //     mask-and-open (only a uniformly-masked value is ever decrypted).
        //
        // See `crate::hybrid` for the protocol, trust model, and literature
        // (Gazelle / CHIMERA / threshold FHE).

        let share_to_fhe_backend = backend.clone();
        let share_to_fhe_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 2 {
                panic!("share->fhe expects 2 args: (share->fhe keys share)");
            }
            let keys_json = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => panic!("share->fhe: first arg must be a key-pair JSON string"),
            };
            let kp: crate::fhe::backend::FheKeyPair = serde_json::from_str(&keys_json)
                .unwrap_or_else(|e| panic!("share->fhe: invalid key JSON: {}", e));
            let sref = match &args[1] {
                Types::Share(s) => s.clone(),
                _ => panic!("share->fhe: second arg must be a shared value"),
            };
            let mut be = share_to_fhe_backend
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            use crate::fhe::{FheBackend, RealFheBackend};
            use crate::hybrid::{mpc_share_to_fhe, FheAdapter, IntFhe, RealFhe};
            fn pack<A: FheAdapter>(
                a: &A,
                ct: &A::Ct,
                ser: impl Fn(&A::Ct) -> Vec<u8>,
                sref: &crate::interpreter::backend::ShareRef,
            ) -> Types {
                Types::FheCiphertext(crate::fhe::backend::FheCiphertextHandle {
                    backend_id: a.backend_id().to_string(),
                    bytes: ser(ct),
                    fixed: sref.fixed,
                    nbits: Some(sref.nbits),
                })
            }
            match kp.backend_id.as_str() {
                "tfhe-rs-v1.6" => {
                    crate::fhe::TfheBackend::install_server_key(&kp.server_key_bytes);
                    let f = crate::fhe::TfheBackend::new();
                    let a = IntFhe(&f);
                    let ct = mpc_share_to_fhe(&mut **be, &sref, &a, &kp.client_key_bytes);
                    pack(&a, &ct, |c| f.serialize_ct(c), &sref)
                }
                "seal-ckks" => {
                    let f = crate::fhe::CkksBackend::new();
                    let a = RealFhe(&f);
                    let ct = mpc_share_to_fhe(&mut **be, &sref, &a, &kp.client_key_bytes);
                    pack(&a, &ct, |c| f.serialize_ct(c), &sref)
                }
                "ckks-mock" => {
                    let f = crate::fhe::MockCkksBackend;
                    let a = RealFhe(&f);
                    let ct = mpc_share_to_fhe(&mut **be, &sref, &a, &kp.client_key_bytes);
                    pack(&a, &ct, |c| f.serialize_ct(c), &sref)
                }
                _ => {
                    let f = crate::fhe::MockFheBackend;
                    let a = IntFhe(&f);
                    let ct = mpc_share_to_fhe(&mut **be, &sref, &a, &kp.client_key_bytes);
                    pack(&a, &ct, |c| f.serialize_ct(c), &sref)
                }
            }
        });
        self.store.borrow_mut().insert(
            "share->fhe".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(share_to_fhe_fn))),
        );

        let fhe_to_share_backend = backend.clone();
        let fhe_to_share_fn = Arc::new(move |args: Vec<Types>| -> Types {
            if args.len() != 3 {
                panic!("fhe->share expects 3 args: (fhe->share keys ciphertext nbits)");
            }
            let keys_json = match &args[0] {
                Types::Str(s) => s.clone(),
                _ => panic!("fhe->share: first arg must be a key-pair JSON string"),
            };
            let kp: crate::fhe::backend::FheKeyPair = serde_json::from_str(&keys_json)
                .unwrap_or_else(|e| panic!("fhe->share: invalid key JSON: {}", e));
            let handle = match &args[1] {
                Types::FheCiphertext(h) => h.clone(),
                _ => panic!("fhe->share: second arg must be a FheCiphertext"),
            };
            let nbits = match &args[2] {
                Types::Num(n) => *n as usize,
                _ => panic!("fhe->share: third arg must be a bit width (number)"),
            };
            let mut be = fhe_to_share_backend
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            use crate::fhe::{FheBackend, RealFheBackend};
            use crate::hybrid::{fhe_to_mpc_share, IntFhe, RealFhe};
            let mut sref = match kp.backend_id.as_str() {
                "tfhe-rs-v1.6" => {
                    crate::fhe::TfheBackend::install_server_key(&kp.server_key_bytes);
                    let f = crate::fhe::TfheBackend::new();
                    let ct = f.deserialize_ct(&handle.bytes);
                    fhe_to_mpc_share(&IntFhe(&f), &kp.client_key_bytes, &ct, &mut **be, nbits)
                }
                "seal-ckks" => {
                    let f = crate::fhe::CkksBackend::new();
                    let ct = f.deserialize_ct(&handle.bytes);
                    fhe_to_mpc_share(&RealFhe(&f), &kp.client_key_bytes, &ct, &mut **be, nbits)
                }
                "ckks-mock" => {
                    let f = crate::fhe::MockCkksBackend;
                    let ct = f.deserialize_ct(&handle.bytes);
                    fhe_to_mpc_share(&RealFhe(&f), &kp.client_key_bytes, &ct, &mut **be, nbits)
                }
                _ => {
                    let f = crate::fhe::MockFheBackend;
                    let ct = f.deserialize_ct(&handle.bytes);
                    fhe_to_mpc_share(&IntFhe(&f), &kp.client_key_bytes, &ct, &mut **be, nbits)
                }
            };
            sref.fixed = handle.fixed;
            Types::Share(sref)
        });
        self.store.borrow_mut().insert(
            "fhe->share".to_string(),
            Rc::new(RefCell::new(Types::NativeFunc(fhe_to_share_fn))),
        );
    }

    /// Reveal a shared value to the caller (simulated reveal via backend).
    pub fn reveal(&self, sref: &crate::interpreter::backend::ShareRef) -> crate::crypto::types::Fr {
        let mut be = self.backend.lock().unwrap_or_else(|e| e.into_inner());
        be.reveal(sref, None)
    }

    /// Print env
    pub fn print_env(&self) {
        let map = self.store.borrow();
        println!("=== participant {} environment ===", self.id);
        for (k, vptr) in map.iter() {
            let v = vptr.borrow();
            println!("{} => {:?}", k, *v);
        }
        println!("=== end environment ===");
    }

    /// Reset per-expression backend state (share map + ID counter).
    /// Call this before evaluating each new top-level expression so that all
    /// parties start with matching ID counters, even after an error/abort.
    pub fn reset_per_expression(&self) {
        let mut be = self.backend.lock().unwrap_or_else(|e| e.into_inner());
        be.reset_per_expression();
    }

    /// Content-addressed commitment to this participant's current top-level
    /// environment, via the hash-consed store: a 32-byte state root that two
    /// parties can compare to decide whether they hold the same environment,
    /// without exchanging it.
    pub fn env_root(&self) -> [u8; 32] {
        let mut store = crate::interpreter::hashcons::Store::new();
        let root = store.intern_env(&self.store);
        store.hash(root)
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
            if let Some(pv) = to_persist_value(&v) {
                out.insert(k.clone(), pv);
            }
        }
        match serde_json::to_string_pretty(&out) {
            Ok(s) => match File::create(path) {
                Ok(mut f) => {
                    if let Err(e) = f.write_all(s.as_bytes()) {
                        return Err(format!("write error: {}", e));
                    }
                    Ok(())
                }
                Err(e) => Err(format!("create file error: {}", e)),
            },
            Err(e) => Err(format!("serialize error: {}", e)),
        }
    }

    /// Load a store from a JSON file, merging persisted bindings into the current store.
    /// Existing user bindings with the same name will be overwritten; builtins are preserved.
    pub fn load_store<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let mut s = String::new();
        match File::open(path) {
            Ok(mut f) => {
                if let Err(e) = f.read_to_string(&mut s) {
                    return Err(format!("read error: {}", e));
                }
            }
            Err(e) => return Err(format!("open error: {}", e)),
        }
        let parsed: std::result::Result<std::collections::HashMap<String, PersistValue>, _> =
            serde_json::from_str(&s);
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
    pub fn eval(&self, expr: &Expr) -> Types {
        let owner = self.id;
        let cont = Continuation::Halt;
        let result = reduce_expression(
            expr.clone(),
            cont,
            self.store.clone(),
            self.backend.clone(),
            owner,
            false,
        );
        if !cfg!(test) {
            println!(
                "participant {} evaluated expression -> {:?}",
                self.id, result
            );
        }
        result
    }

    /// Like `eval` but never prints anything.  Used by the TCP party-0 REPL where
    /// output is handled by the caller with colours.
    pub fn eval_quiet(&self, expr: &Expr) -> Types {
        let owner = self.id;
        let cont = Continuation::Halt;
        reduce_expression(
            expr.clone(),
            cont,
            self.store.clone(),
            self.backend.clone(),
            owner,
            false,
        )
    }

    /// Like `eval_quiet` but also emits verbose frame-level tracing to stderr.
    pub fn eval_quiet_verbose(&self, expr: &Expr) -> Types {
        let owner = self.id;
        let cont = Continuation::Halt;
        reduce_expression(
            expr.clone(),
            cont,
            self.store.clone(),
            self.backend.clone(),
            owner,
            true,
        )
    }

    /// Evaluate with optional verbose tracing. When verbose=true the REPL will print
    /// per-frame environment, expression and continuation snapshots.
    pub fn eval_verbose(&self, expr: &Expr, verbose: bool) -> Types {
        if verbose {
            println!("[participant {}] starting eval (verbose)", self.id);
        }
        let cont = Continuation::Halt;
        let result = reduce_expression(
            expr.clone(),
            cont,
            self.store.clone(),
            self.backend.clone(),
            self.id,
            verbose,
        );
        if verbose {
            println!("[participant {}] finished eval -> {:?}", self.id, result);
        }
        result
    }
}

// helpers to convert between SharedValue and PersistValue
fn to_persist_value(v: &Types) -> Option<PersistValue> {
    match v {
        Types::Num(n) => Some(PersistValue::Num(*n)),
        // Persist Fixed as a specially-prefixed string "fixed:<scaled_i64>"
        Types::Fixed(f) => Some(PersistValue::Str(format!("fixed:{}", f))),
        Types::Bool(b) => Some(PersistValue::Bool(*b)),
        Types::Str(s) => Some(PersistValue::Str(s.clone())),
        Types::Char(c) => Some(PersistValue::Char(*c)),
        Types::List(lst) => {
            let mut out: Vec<PersistValue> = Vec::new();
            for item in lst.iter() {
                let iv = to_persist_value(item)?;
                out.push(iv);
            }
            Some(PersistValue::List(out))
        }
        _ => None,
    }
}

fn from_persist_value(p: PersistValue) -> Option<Types> {
    match p {
        PersistValue::Num(n) => Some(Types::Num(n)),
        PersistValue::Bool(b) => Some(Types::Bool(b)),
        PersistValue::Str(s) => {
            // Restore Fixed values from persist format
            if let Some(rest) = s.strip_prefix("fixed:") {
                if let Ok(v) = rest.parse::<i64>() {
                    return Some(Types::Fixed(v));
                }
            }
            Some(Types::Str(s))
        }
        PersistValue::Char(c) => Some(Types::Char(c)),
        PersistValue::List(vs) => {
            let mut out: Vec<Types> = Vec::new();
            for iv in vs.into_iter() {
                let sv = from_persist_value(iv)?;
                out.push(sv);
            }
            Some(Types::List(out))
        }
    }
}

fn expr_short(e: &Expr) -> String {
    match e {
        Expr::Quote(_) => "Quote".to_string(),
        Expr::Eval(_) => "Eval".to_string(),
        Expr::UnOp(_, _) => "UnOp".to_string(),
        Expr::Str(s) => format!("Str(len={})", s.len()),
        Expr::Char(_) => "Char".to_string(),
        Expr::Bool(b) => format!("Bool({})", b),
        Expr::Num(n) => format!("Num({})", n),
        Expr::Fixed(v) => {
            use crate::interpreter::ast::FIXED_SCALE;
            let whole = v / FIXED_SCALE;
            let frac = ((v % FIXED_SCALE).abs() * 10_000) / FIXED_SCALE;
            format!("Fixed({}.{:04})", whole, frac)
        }
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
        Expr::Private(_) => "Private".to_string(),
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

/// Decode a quoted list of strings from a Types value.
/// Accepts '("a" "b") → Quote(Call(Str("a"), [Str("b"), ...])) or Quote(Seq([...])) for zero/multiple items
/// Convert a quoted expression into a literal data list, if it is one.
///
/// A quote shorthand like '(1 2 3) parses as Quote(Call(Num(1), [Num(2), Num(3)]))
/// and '() as Quote(Seq([])).  When every element is a literal (number, fixed,
/// bool, char, negated number, or a nested literal list) the quote denotes
/// *data*, and we materialise it as a Vec<Types> so car/cdr/null? work on it.
/// Quoted code ('(+ x y)) and string lists ('("x") — used as argument lists by
/// the zkp/fhe builtins) return None and stay Quoted.
fn quoted_literal_list(e: &Expr) -> Option<Vec<Types>> {
    fn literal_value(e: &Expr) -> Option<Types> {
        match e {
            Expr::Num(n) => Some(Types::Num(*n)),
            Expr::Fixed(v) => Some(Types::Fixed(*v)),
            Expr::Bool(b) => Some(Types::Bool(*b)),
            Expr::Char(c) => Some(Types::Char(*c)),
            Expr::UnOp(super::parser::UnOp::Neg, inner) => match inner.as_ref() {
                Expr::Num(n) => Some(Types::Num(-n)),
                Expr::Fixed(v) => Some(Types::Fixed(-v)),
                _ => None,
            },
            Expr::Call(_, _) | Expr::Seq(_) => quoted_literal_list(e).map(Types::List),
            _ => None,
        }
    }
    match e {
        Expr::Seq(items) if items.is_empty() => Some(vec![]),
        Expr::Call(head, rest) => {
            let mut out = Vec::with_capacity(1 + rest.len());
            out.push(literal_value(head)?);
            for item in rest {
                out.push(literal_value(item)?);
            }
            Some(out)
        }
        _ => None,
    }
}

fn decode_quoted_str_list(v: &Types, label: &str) -> Vec<String> {
    use crate::interpreter::ast::Expr;
    match v {
        Types::List(items) => items
            .iter()
            .map(|x| match x {
                Types::Str(s) => s.clone(),
                _ => panic!("{}: expected string item", label),
            })
            .collect(),
        Types::Quoted(inner) => match inner.as_ref() {
            Expr::Call(head, rest) => {
                let mut out = Vec::new();
                if let Expr::Str(s) = head.as_ref() {
                    out.push(s.clone());
                } else {
                    panic!("{}: expected string in quoted list, got {:?}", label, head);
                }
                for e in rest {
                    if let Expr::Str(s) = e {
                        out.push(s.clone());
                    } else {
                        panic!("{}: expected string in quoted list", label);
                    }
                }
                out
            }
            Expr::Str(s) => vec![s.clone()],
            Expr::Seq(items) if items.is_empty() => vec![],
            _ => panic!(
                "{}: expected a quoted list of strings like '(\"x\" \"y\")",
                label
            ),
        },
        _ => panic!("{}: expected a quoted list of strings like '(\"x\")", label),
    }
}

/// Decode a quoted list of numbers from a Types value.
/// Accepts '(5 10) → Quote(Call(Num(5), [Num(10)])) or Quote(Num(5)) or Quote(Seq([])).
fn decode_quoted_num_list(v: &Types, label: &str) -> Vec<i64> {
    use crate::interpreter::ast::Expr;
    match v {
        Types::List(items) => items
            .iter()
            .map(|x| match x {
                Types::Num(n) => *n,
                _ => panic!("{}: expected number item", label),
            })
            .collect(),
        Types::Quoted(inner) => match inner.as_ref() {
            Expr::Call(head, rest) => {
                let mut out = Vec::new();
                if let Expr::Num(n) = head.as_ref() {
                    out.push(*n);
                } else {
                    panic!("{}: expected number in quoted list, got {:?}", label, head);
                }
                for e in rest {
                    if let Expr::Num(n) = e {
                        out.push(*n);
                    } else {
                        panic!("{}: expected number in quoted list", label);
                    }
                }
                out
            }
            Expr::Num(n) => vec![*n],
            Expr::Seq(items) if items.is_empty() => vec![],
            _ => panic!("{}: expected a quoted list of numbers like '(5 10)", label),
        },
        _ => panic!("{}: expected a quoted list of numbers like '(5)", label),
    }
}

// ─── Vector builtin helpers (dot / vsum / matvec) ────────────────────────────

use crate::interpreter::backend::ShareRef;

/// Sign-extend an nbits two's-complement field value to i64.
fn fr_to_i64_signed(v: u128, nbits: usize) -> i64 {
    if nbits >= 64 {
        v as i64
    } else {
        let m = 1u128 << nbits;
        let v = v & (m - 1);
        if v & (m >> 1) != 0 {
            (v as i128 - m as i128) as i64
        } else {
            v as i64
        }
    }
}

/// Arithmetic >>16 on a two's-complement share — the Q16.16 rescale after a
/// fixed×fixed multiply.  Decompose to bits, drop the low 16, replicate the
/// sign bit into the top 16, recompose.
fn share_ashr16(be: &mut dyn Backend, a: &ShareRef) -> ShareRef {
    assert!(
        a.nbits > 16,
        "fixed-point shares need more than 16 bits, got {}",
        a.nbits
    );
    let bits = be.a2b(a, a.nbits);
    let sign = bits[a.nbits - 1].clone();
    let mut out_bits: Vec<ShareRef> = bits[16..].to_vec();
    while out_bits.len() < a.nbits {
        out_bits.push(sign.clone());
    }
    let mut r = be.b2a(&out_bits);
    // b2a inherits the 1-bit width of its inputs; restore the real width.
    r.nbits = a.nbits;
    r
}

/// Lift an integer share into the Q16.16 domain (×2^16 via doubling).
fn share_to_fixed(be: &mut dyn Backend, a: &ShareRef) -> ShareRef {
    let mut r = be.shift_left(a, 16);
    r.nbits = a.nbits;
    r.fixed = true;
    r
}

/// Share a plain operand at face value: Fixed keeps its scaled
/// representation (share marked fixed), Num/Bool stay integers.
fn share_imm_raw(be: &mut dyn Backend, v: &Types, nbits: usize) -> ShareRef {
    match v {
        Types::Num(n) => be.share_secret(int_to_fr_signed(*n, nbits), nbits, 0),
        Types::Bool(b) => be.share_secret(if *b { 1 } else { 0 }, nbits, 0),
        Types::Fixed(f) => {
            let mut s = be.share_secret(int_to_fr_signed(*f, nbits), nbits, 0);
            s.fixed = true;
            s
        }
        _ => panic!("cannot share operand {:?}", v),
    }
}

/// Bring an operand into the share domain; when `scaled` is set, integer
/// operands are lifted to Q16.16 (mirrors the interpreter's Num→Fixed
/// coercion).
fn share_operand(be: &mut dyn Backend, v: &Types, nbits: usize, scaled: bool) -> ShareRef {
    use crate::interpreter::ast::FIXED_SCALE;
    match v {
        Types::Share(s) => {
            if scaled && !s.fixed {
                share_to_fixed(be, s)
            } else {
                s.clone()
            }
        }
        Types::Num(n) if scaled => {
            let mut s = be.share_secret(int_to_fr_signed(n * FIXED_SCALE, nbits), nbits, 0);
            s.fixed = true;
            s
        }
        _ => share_imm_raw(be, v, nbits),
    }
}

fn operand_nbits(a: &Types, b: &Types) -> usize {
    match (a, b) {
        (Types::Share(s), _) => s.nbits,
        (_, Types::Share(s)) => s.nbits,
        _ => panic!("fixed-share op needs at least one share operand"),
    }
}

fn operand_is_fixed(v: &Types) -> bool {
    matches!(v, Types::Fixed(_)) || matches!(v, Types::Share(s) if s.fixed)
}

/// Multiply in the share domain with Q16.16 awareness: a fixed×int product
/// is already correctly scaled, so the (expensive) rescale runs only for
/// fixed×fixed.
fn share_mul_typed(be: &mut dyn Backend, a: &Types, b: &Types) -> ShareRef {
    let nbits = operand_nbits(a, b);
    let (fa, fb) = (operand_is_fixed(a), operand_is_fixed(b));
    let sa = share_operand(be, a, nbits, false);
    let sb = share_operand(be, b, nbits, false);
    let prod = be.mul(&sa, &sb);
    let mut r = if fa && fb {
        share_ashr16(be, &prod)
    } else {
        prod
    };
    r.fixed = fa || fb;
    r
}

/// Add/sub in the share domain with Q16.16 awareness (mixed int operands
/// are lifted to the scaled domain first).
fn share_addsub_typed(be: &mut dyn Backend, a: &Types, b: &Types, sub: bool) -> ShareRef {
    let nbits = operand_nbits(a, b);
    let any_fixed = operand_is_fixed(a) || operand_is_fixed(b);
    let sa = share_operand(be, a, nbits, any_fixed);
    let sb = share_operand(be, b, nbits, any_fixed);
    let mut r = if sub {
        be.sub(&sa, &sb)
    } else {
        be.add(&sa, &sb)
    };
    r.fixed = any_fixed;
    r
}

/// Binary op where at least one operand is a Q16.16 share (or a share mixed
/// with a plain Fixed).  Comparisons compare in the scaled domain and yield
/// plain bit shares.
fn fixed_share_binop(
    backend: &Arc<Mutex<Box<dyn Backend>>>,
    op: crate::interpreter::parser::BinOp,
    a: &Types,
    b: &Types,
) -> Types {
    use crate::interpreter::parser::BinOp;
    let mut guard = backend.lock().unwrap_or_else(|e| e.into_inner());
    let be: &mut dyn Backend = &mut **guard;
    match op {
        BinOp::Mul => Types::Share(share_mul_typed(be, a, b)),
        BinOp::Add => Types::Share(share_addsub_typed(be, a, b, false)),
        BinOp::Sub => Types::Share(share_addsub_typed(be, a, b, true)),
        BinOp::Eq | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            let nbits = operand_nbits(a, b);
            let sa = share_operand(be, a, nbits, true);
            let sb = share_operand(be, b, nbits, true);
            let r = match op {
                BinOp::Eq => be.eq(&sa, &sb),
                BinOp::Lt => be.lt(&sa, &sb),
                BinOp::Gt => be.lt(&sb, &sa),
                BinOp::Le => be.le(&sa, &sb),
                BinOp::Ge => be.le(&sb, &sa),
                _ => unreachable!(),
            };
            Types::Share(r)
        }
        other => panic!(
            "operation {:?} is not supported on fixed-point shares",
            other
        ),
    }
}

/// Element-wise product for the vector builtins.  Handles Num, Fixed and
/// Share operands (including Q16.16 shares) following the interpreter's
/// scalar coercion rules.
fn vec_mul(backend: &Arc<Mutex<Box<dyn Backend>>>, a: &Types, b: &Types) -> Types {
    use crate::interpreter::ast::FIXED_SCALE;
    let to_fixed = |v: &Types| -> i64 {
        match v {
            Types::Fixed(f) => *f,
            Types::Num(n) => n * FIXED_SCALE,
            _ => unreachable!(),
        }
    };
    match (a, b) {
        (Types::Num(x), Types::Num(y)) => Types::Num(x.wrapping_mul(*y)),
        (Types::Fixed(_), Types::Fixed(_))
        | (Types::Fixed(_), Types::Num(_))
        | (Types::Num(_), Types::Fixed(_)) => {
            let prod = (to_fixed(a) as i128 * to_fixed(b) as i128) >> 16;
            Types::Fixed(prod as i64)
        }
        _ if matches!(a, Types::Share(_)) || matches!(b, Types::Share(_)) => {
            let mut be = backend.lock().unwrap_or_else(|e| e.into_inner());
            Types::Share(share_mul_typed(&mut **be, a, b))
        }
        _ => panic!("vector builtins: cannot multiply {:?} and {:?}", a, b),
    }
}

/// Element-wise sum for the vector builtins.  Same operand rules as `vec_mul`.
fn vec_add(backend: &Arc<Mutex<Box<dyn Backend>>>, a: &Types, b: &Types) -> Types {
    use crate::interpreter::ast::FIXED_SCALE;
    let to_fixed = |v: &Types| -> i64 {
        match v {
            Types::Fixed(f) => *f,
            Types::Num(n) => n * FIXED_SCALE,
            _ => unreachable!(),
        }
    };
    match (a, b) {
        (Types::Num(x), Types::Num(y)) => Types::Num(x.wrapping_add(*y)),
        (Types::Fixed(_), Types::Fixed(_))
        | (Types::Fixed(_), Types::Num(_))
        | (Types::Num(_), Types::Fixed(_)) => Types::Fixed(to_fixed(a).wrapping_add(to_fixed(b))),
        _ if matches!(a, Types::Share(_)) || matches!(b, Types::Share(_)) => {
            let mut be = backend.lock().unwrap_or_else(|e| e.into_inner());
            Types::Share(share_addsub_typed(&mut **be, a, b, false))
        }
        _ => panic!("vector builtins: cannot add {:?} and {:?}", a, b),
    }
}

/// Shared matvec dispatch, used by both the `matvec` builtin and
/// `nn-forward` (one row of a network layer at a time). When every weight
/// and input element is a live Share, dispatch to the registered MatVec
/// precompile (see `crate::precompile::mpc_matvec`) instead of the naive
/// per-element loop below — a hand-optimized "gadget" substituted in
/// transparently. Any Num/Fixed mix, or a public-weights/private-input
/// combination (the precompile only accepts bare `ShareRef`s on both
/// sides), falls through to the per-row `vec_dot` loop unchanged.
fn matvec_impl(backend: &Arc<Mutex<Box<dyn Backend>>>, rows: &[Types], v: &[Types]) -> Vec<Types> {
    let all_shares = |vs: &[Types]| vs.iter().all(|t| matches!(t, Types::Share(_)));
    let rows_are_share_lists = rows
        .iter()
        .all(|row| matches!(row, Types::List(r) if all_shares(r)));
    if rows_are_share_lists && all_shares(v) && !rows.is_empty() {
        let x_refs: Vec<ShareRef> = v
            .iter()
            .map(|t| match t {
                Types::Share(sr) => sr.clone(),
                _ => unreachable!(),
            })
            .collect();
        let w_refs: Vec<Vec<ShareRef>> = rows
            .iter()
            .map(|row| match row {
                Types::List(r) => r
                    .iter()
                    .map(|t| match t {
                        Types::Share(sr) => sr.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                _ => unreachable!(),
            })
            .collect();
        let mut be = backend.lock().unwrap_or_else(|e| e.into_inner());
        let out = crate::precompile::mpc_matvec::registry()
            .champion()
            .matvec(&mut **be, &w_refs, &x_refs);
        return out.into_iter().map(Types::Share).collect();
    }
    rows.iter()
        .map(|row| match row {
            Types::List(r) => vec_dot(backend, r, v),
            _ => panic!("matvec: matrix rows must be lists"),
        })
        .collect()
}

/// Elementwise ReLU for `nn-forward`. Num/Fixed clamp locally; Share follows
/// the mux-via-mul pattern from `bench::nn`'s Layer workload: `mask = (0 <
/// x)`, then `x * mask` — no fixed-point rescale needed since `mask` is a
/// plain 0/1 bit, not Q16.16-scaled.
fn vec_relu(backend: &Arc<Mutex<Box<dyn Backend>>>, x: &Types) -> Types {
    match x {
        Types::Num(n) => Types::Num((*n).max(0)),
        Types::Fixed(f) => Types::Fixed((*f).max(0)),
        Types::Share(sr) => {
            let mut be = backend.lock().unwrap_or_else(|e| e.into_inner());
            let zero = be.sub(sr, sr);
            let mask = be.lt(&zero, sr);
            let mut r = be.mul(sr, &mask);
            r.fixed = sr.fixed;
            Types::Share(r)
        }
        _ => panic!("nn-forward: relu expects Num, Fixed, or Share elements"),
    }
}

/// Dot product of two element vectors (used by both `dot` and `matvec`).
fn vec_dot(backend: &Arc<Mutex<Box<dyn Backend>>>, a: &[Types], b: &[Types]) -> Types {
    if a.len() != b.len() {
        panic!("dot: length mismatch ({} vs {})", a.len(), b.len());
    }
    let mut acc: Option<Types> = None;
    for (x, y) in a.iter().zip(b.iter()) {
        let p = vec_mul(backend, x, y);
        acc = Some(match acc {
            None => p,
            Some(s) => vec_add(backend, &s, &p),
        });
    }
    acc.unwrap_or(Types::Num(0))
}

// Convert a signed i64 value into an unsigned field element representing two's complement with `nbits` bits
fn int_to_fr_signed(n: i64, nbits: usize) -> u128 {
    if nbits >= 128 {
        return n as i128 as u128;
    }
    let modulus = 1i128.checked_shl(nbits as u32).unwrap_or(0);
    if modulus == 0 {
        return n as i128 as u128;
    }
    let v = (n as i128).rem_euclid(modulus);
    v as u128
}

/// Evaluate a plain (non-shared) expression and return the numeric result.
pub fn eval_plain(src: &str) -> i64 {
    let expr = parse(src).expect("parse failed");
    let p = Participant::new_simulated(0);
    match p.eval(&expr) {
        Types::Num(n) => n,
        other => panic!("expected Num, got: {:?}", other),
    }
}

/// Evaluate a plain (non-shared) expression and return the numeric result.
pub fn eval_plain_string(src: &str) -> String {
    let expr = parse(src).expect("parse failed");
    let p = Participant::new_simulated(0);
    match p.eval(&expr) {
        Types::Num(n) => {
            println!("eval_plain: numeric value: {}", n);
            n.to_string()
        }
        Types::Quoted(v) => {
            println!("eval_plain: quoted value: {:?}", v);
            // Return the string representation of the quoted value as string
            v.to_string()
        }
        other => panic!("expected Num, got: {:?}", other),
    }
}

/// Evaluate an expression, reveal any resulting Share, and return the
/// reconstructed value interpreted as a signed 32-bit integer.
pub fn eval_shared(src: &str) -> i64 {
    let expr = parse(src).expect("parse failed");
    let p = Participant::new_simulated(0);
    match p.eval(&expr) {
        // Plain Num may carry a raw u128 field value stored as i64 bits; reinterpret as signed 32.
        Types::Num(n) => (n as u32) as i32 as i64,
        Types::Share(sr) => {
            let raw = p.reveal(&sr);
            // sign-extend from 32 bits
            (raw as u32) as i32 as i64
        }
        other => panic!("expected Num or Share, got: {:?}", other),
    }
}

/// Evaluate a Baru expression that uses FHE builtins (fhe-keygen, fhe-enc,
/// fhe-dec, fhe-eval, fhe-keygen-tfhe).  Returns the numeric result.
/// The expression must produce a `Types::Num` at the top level; use
/// `fhe-dec` inside the expression to convert a ciphertext to a number.
pub fn eval_fhe(src: &str) -> i64 {
    let expr = parse(src).expect("parse failed");
    let p = Participant::new_simulated(0);
    match p.eval(&expr) {
        Types::Num(n) => n,
        other => panic!("eval_fhe: expected Num, got: {:?}", other),
    }
}

#[cfg(test)]
mod fixed_point_tests {
    use super::*;
    use crate::interpreter::ast::{Expr, FIXED_SCALE};
    use crate::interpreter::parse;

    /// Evaluate a Baru source string through a fresh Participant and return the result.
    fn eval_str(src: &str) -> Types {
        let expr = parse(src).expect("parse failed");
        let p = Participant::new_simulated(0);
        p.eval(&expr)
    }

    // ── Parser ────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_fixed_literal() {
        let expr = parse("3.5").unwrap();
        assert!(
            matches!(expr, Expr::Fixed(v) if v == (3 * FIXED_SCALE + FIXED_SCALE / 2)),
            "3.5 should parse to Fixed(3.5 * 65536)"
        );
    }

    #[test]
    fn test_parse_negative_fixed() {
        let expr = parse("-1.25").unwrap();
        // -1.25 * 65536 = -81920
        assert!(
            matches!(expr, Expr::Fixed(v) if v == (-FIXED_SCALE - FIXED_SCALE / 4)),
            "got: {:?}",
            expr
        );
    }

    #[test]
    fn test_parse_zero_fixed() {
        let expr = parse("0.0").unwrap();
        assert!(matches!(expr, Expr::Fixed(0)));
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[test]
    fn test_fixed_add() {
        // 1.5 + 2.25 = 3.75
        let result = eval_str("(+ 1.5 2.25)");
        match result {
            Types::Fixed(v) => {
                let expected = (3 * FIXED_SCALE + 3 * FIXED_SCALE / 4) as f64 / FIXED_SCALE as f64;
                let got = v as f64 / FIXED_SCALE as f64;
                assert!(
                    (got - expected).abs() < 0.0002,
                    "expected ~{}, got {}",
                    expected,
                    got
                );
            }
            _ => panic!("expected Fixed, got {:?}", result),
        }
    }

    #[test]
    fn test_fixed_sub() {
        // 3.0 - 1.5 = 1.5
        let result = eval_str("(- 3.0 1.5)");
        match result {
            Types::Fixed(v) => {
                let got = v as f64 / FIXED_SCALE as f64;
                assert!((got - 1.5).abs() < 0.0002, "expected 1.5, got {}", got);
            }
            _ => panic!("expected Fixed, got {:?}", result),
        }
    }

    #[test]
    fn test_fixed_mul() {
        // 2.0 * 3.5 = 7.0
        let result = eval_str("(* 2.0 3.5)");
        match result {
            Types::Fixed(v) => {
                let got = v as f64 / FIXED_SCALE as f64;
                assert!((got - 7.0).abs() < 0.0002, "expected 7.0, got {}", got);
            }
            _ => panic!("expected Fixed, got {:?}", result),
        }
    }

    #[test]
    fn test_fixed_div() {
        // 7.0 / 2.0 = 3.5
        let result = eval_str("(/ 7.0 2.0)");
        match result {
            Types::Fixed(v) => {
                let got = v as f64 / FIXED_SCALE as f64;
                assert!((got - 3.5).abs() < 0.0002, "expected 3.5, got {}", got);
            }
            _ => panic!("expected Fixed, got {:?}", result),
        }
    }

    #[test]
    fn test_fixed_num_coercion_add() {
        // 1 + 0.5 = 1.5  (Num coerced to Fixed)
        let result = eval_str("(+ 1 0.5)");
        match result {
            Types::Fixed(v) => {
                let got = v as f64 / FIXED_SCALE as f64;
                assert!((got - 1.5).abs() < 0.0002, "expected 1.5, got {}", got);
            }
            _ => panic!("expected Fixed, got {:?}", result),
        }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────

    #[test]
    fn test_fixed_eq() {
        assert!(matches!(eval_str("(== 1.5 1.5)"), Types::Num(1)));
        assert!(matches!(eval_str("(== 1.5 2.0)"), Types::Num(0)));
    }

    #[test]
    fn test_fixed_lt() {
        assert!(matches!(eval_str("(< 1.5 2.0)"), Types::Num(1)));
        assert!(matches!(eval_str("(< 2.0 1.5)"), Types::Num(0)));
    }

    #[test]
    fn test_fixed_mixed_comparison() {
        // 2 (int) < 2.5 (fixed) should be true
        assert!(matches!(eval_str("(< 2 2.5)"), Types::Num(1)));
    }
}
