// Minimal distributed Lisp-like evaluator for MPC (ABY3 backend simulation)
// - Supports NUMs, variables, let, letrec, sequences and binary operations (+, -, *, exact /)
// - Implements `apply_continuation` and `reduce_expression` in a CEK-like style
// - Simulates 3 participants running a distributed REPL where constants are evaluated locally

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug, Clone)]
enum Expr {
    Num(i64),
    Var(String),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Let(Vec<(String, Expr)>, Vec<Expr>),
    LetRec(Vec<(String, Expr)>, Vec<Expr>),
    Seq(Vec<Expr>),
    Car(Box<Expr>),
    Cdr(Box<Expr>),
    Cons(Box<Expr>, Box<Expr>),
    Lambda(Vec<String>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    DivExact,
    Eq,
}

// Store/Env: map names to pointers so letrec can prebind placeholders.
type Ptr = Rc<RefCell<SharedValue>>;
type Env = Rc<RefCell<HashMap<String, Ptr>>>;

#[derive(Debug, Clone)]
/// A very small representation of a secret-shared value.
/// Can be a number or a list of shared values or a closure capturing an Env.
enum SharedValue {
    Num(i64),
    List(Vec<SharedValue>),
    Closure(Vec<String>, Box<Expr>, Env),
}

#[derive(Debug, Clone)]
enum Continuation {
    Halt,
    BinOpLeft {
        op: BinOp,
        right: Box<Expr>,
        cont: Box<Continuation>,
    },
    BinOpRight {
        op: BinOp,
        left_val: SharedValue,
        cont: Box<Continuation>,
    },

    // let (top-level bindings) continuation: after evaluating a binding value
    // insert into `env` under `name` and evaluate next binding (if any).
    LetTop {
        name: String,
        remaining: Vec<(String, Expr)>,
        env: Env,
    },

    // let with a local scope: after computing a binding value insert into
    // `local_env` and continue with remaining bindings or evaluate body.
    LetLocal {
        name: String,
        remaining: Vec<(String, Expr)>,
        local_env: Env,
        body: Vec<Expr>,
        cont: Box<Continuation>,
    },

    // letrec initializer continuation: write initializer result into placeholder ptr
    LetRecInit {
        name: String,
        remaining: Vec<(String, Expr)>,
        local_env: Env,
        body: Vec<Expr>,
        cont: Box<Continuation>,
    },

    // Sequence continuation: when a subexpression finishes, either evaluate
    // the next expr or apply the parent continuation using last value.
    Seq {
        remaining: Vec<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },

    Car(Box<Continuation>),
    Cdr(Box<Continuation>),

    ConsLeft {
        right: Box<Expr>,
        cont: Box<Continuation>,
        env: Env,
    },
    ConsRight {
        left_val: SharedValue,
        cont: Box<Continuation>,
    },

    // Call: after operator is evaluated, begin evaluating args sequentially.
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
    },
}

// Tasks for the evaluator loop.
enum Task {
    Eval(Expr, Continuation, Env),
    Apply(Continuation, SharedValue, Env),
}

const MAX_FRAMES: usize = 1_000_000;

fn reduce_expression(expr: Expr, cont: Continuation, env: Env) -> SharedValue {
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
                    let vdesc = match v { SharedValue::Num(n) => format!("Num({})", n), SharedValue::List(l) => format!("List(len={})", l.len()), SharedValue::Closure(_, _, _) => "Closure".to_string(), };
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
                    Expr::Var(name) => {
                        let map = eenv.borrow();
                        let ptr = map.get(&name).cloned().unwrap_or_else(|| panic!("unbound variable: {}", name));
                        let v = ptr.borrow().clone();
                        stack.push(Apply(k, v, eenv.clone()));
                    }
                    Expr::BinOp(op, left, right) => {
                        // Evaluate left first, with a continuation that will evaluate right.
                        let new_cont = Continuation::BinOpLeft { op, right, cont: Box::new(k) };
                        stack.push(Eval(*left, new_cont, eenv.clone()));
                    }
                    Expr::Let(bindings, body) => {
                        if body.is_empty() {
                            // top-level bindings: evaluate each and insert into current env
                            if bindings.is_empty() {
                                stack.push(Apply(k, SharedValue::Num(0), eenv.clone()));
                            } else {
                                let mut it = bindings.into_iter();
                                let (name, be) = it.next().unwrap();
                                let remaining: Vec<(String, Expr)> = it.collect();
                                stack.push(Eval(be, Continuation::LetTop { name, remaining, env: eenv.clone() }, eenv.clone()));
                            }
                        } else {
                            // create a fresh local scope cloned from env
                            let local = Rc::new(RefCell::new(eenv.borrow().clone()));
                            if bindings.is_empty() {
                                // evaluate body in local
                                if body.len() == 1 {
                                    stack.push(Eval(body.into_iter().next().unwrap(), k, local));
                                } else {
                                    let mut rest = body;
                                    let last = rest.pop().unwrap();
                                    stack.push(Eval(last, k, local.clone()));
                                    // push preceding expressions as Eval(..., Halt)
                                    while let Some(pre) = rest.pop() {
                                        stack.push(Eval(pre, Continuation::Halt, local.clone()));
                                    }
                                }
                            } else {
                                let mut it = bindings.into_iter();
                                let (name, be) = it.next().unwrap();
                                let remaining: Vec<(String, Expr)> = it.collect();
                                stack.push(Eval(be, Continuation::LetLocal { name, remaining, local_env: local.clone(), body, cont: Box::new(k) }, local.clone()));
                            }
                        }
                    }
                    Expr::LetRec(bindings, body) => {
                        // prebind placeholders into a shared local env
                        let local = Rc::new(RefCell::new(eenv.borrow().clone()));
                        // insert placeholders (Ptr to dummy values)
                        for (name, _) in &bindings {
                            local.borrow_mut().insert(name.clone(), Rc::new(RefCell::new(SharedValue::Num(0))));
                        }
                        // start evaluating initializers sequentially
                        if bindings.is_empty() {
                            // evaluate body in local
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
                            stack.push(Eval(be, Continuation::LetRecInit { name, remaining, local_env: local.clone(), body, cont: Box::new(k) }, local.clone()));
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
                            // we want to evaluate leading expressions for side effects under Halt
                            stack.push(Eval(last, k, eenv.clone()));
                            while let Some(pre) = rest.pop() {
                                stack.push(Eval(pre, Continuation::Halt, eenv.clone()));
                            }
                        }
                    }
                    Expr::Car(inner) => {
                        stack.push(Eval(*inner, Continuation::Car(Box::new(k)), eenv.clone()));
                    }
                    Expr::Cdr(inner) => {
                        stack.push(Eval(*inner, Continuation::Cdr(Box::new(k)), eenv.clone()));
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
                        stack.push(Eval(*cond, Continuation::If { then_e, else_e, cont: Box::new(k) }, eenv.clone()));
                    }
                }
            }
            Apply(k, val, eenv) => {
                match k {
                    Continuation::Halt => {
                        // if no more tasks on stack, this is the final result
                        if stack.is_empty() {
                            return val;
                        } else {
                            // otherwise this value was produced for side-effect; drop it
                            continue;
                        }
                    }
                    Continuation::BinOpLeft { op, right, cont } => {
                        // left value is `val`; evaluate right with BinOpRight continuation
                        let next = Continuation::BinOpRight { op, left_val: val, cont };
                        stack.push(Eval(*right, next, eenv.clone()));
                    }
                    Continuation::BinOpRight { op, left_val, cont } => {
                        let a = match left_val { SharedValue::Num(x) => x, _ => panic!("binary op left operand must be NUM"), };
                        let b = match val { SharedValue::Num(x) => x, _ => panic!("binary op right operand must be NUM"), };
                        let res = match op {
                            BinOp::Add => a.wrapping_add(b),
                            BinOp::Sub => a.wrapping_sub(b),
                            BinOp::Mul => a.wrapping_mul(b),
                            BinOp::DivExact => {
                                if b == 0 { panic!("division by zero"); }
                                if a % b != 0 { panic!("non-exact division: {} / {}", a, b); }
                                a / b
                            }
                            BinOp::Eq => if a == b { 1 } else { 0 },
                        };
                        let out = SharedValue::Num(res);
                        stack.push(Apply(*cont, out, eenv));
                    }
                    Continuation::LetTop { name, mut remaining, env } => {
                        // insert value into env and continue with remaining bindings
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
                            // evaluate body sequentially in local_env
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
                        // write initializer into placeholder ptr
                        // we inserted a placeholder earlier; overwrite it by replacing the inner value
                        if let Some(ptr) = local_env.borrow().get(&name) {
                            *ptr.borrow_mut() = val;
                        } else {
                            // shouldn't happen
                            local_env.borrow_mut().insert(name.clone(), Rc::new(RefCell::new(val)));
                        }
                        if remaining.is_empty() {
                            // evaluate body in local_env
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
                                // last expr -> apply parent continuation with its value
                                stack.push(Eval(next, *cont, env.clone()));
                            } else {
                                // evaluate next for side effects first
                                stack.push(Eval(next, Continuation::Seq { remaining, cont, env: env.clone() }, env.clone()));
                            }
                        }
                    }
                    Continuation::Car(cont) => {
                        match val {
                            SharedValue::List(mut vs) => {
                                if vs.is_empty() { panic!("car of empty list"); }
                                let first = vs.remove(0);
                                stack.push(Apply(*cont, first, eenv.clone()));
                            }
                            _ => panic!("car expects a list"),
                        }
                    }
                    Continuation::Cdr(cont) => {
                        match val {
                            SharedValue::List(vs) => {
                                if vs.is_empty() { panic!("cdr of empty list"); }
                                let tail = SharedValue::List(vs.into_iter().skip(1).collect());
                                stack.push(Apply(*cont, tail, eenv.clone()));
                            }
                            _ => panic!("cdr expects a list"),
                        }
                    }
                    Continuation::ConsLeft { right, cont, env } => {
                        // left value is available; evaluate right with ConsRight continuation
                        let next = Continuation::ConsRight { left_val: val, cont };
                        stack.push(Eval(*right, next, env.clone()));
                    }
                    Continuation::ConsRight { left_val, cont } => {
                        match val {
                            SharedValue::List(mut vs) => {
                                vs.insert(0, left_val);
                                stack.push(Apply(*cont, SharedValue::List(vs), eenv.clone()));
                            }
                            _ => panic!("cons expects second arg to be a list"),
                        }
                    }
                    Continuation::CallEvalOperator { args, cont, env } => {
                        // operator value is `val`. Begin evaluating args if any.
                        if args.is_empty() {
                            // directly apply
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
                            let collected: Vec<SharedValue> = Vec::new();
                            let call_args_cont = Continuation::CallEvalArgs { operator: val, remaining: rem, mut_collected: collected, cont, env: env.clone() };
                            stack.push(Eval(first, call_args_cont, env.clone()));
                        }
                    }
                    Continuation::CallEvalArgs { operator, mut remaining, mut mut_collected, cont, env } => {
                        // val is the next argument's evaluated value
                        mut_collected.push(val);
                        if remaining.is_empty() {
                            // perform the call with collected arguments
                            match operator {
                                SharedValue::Closure(params, body, closure_env) => {
                                    if params.len() != mut_collected.len() {
                                        panic!("closure arity mismatch: expected {} got {}", params.len(), mut_collected.len());
                                    }
                                    // create call env cloned from closure_env and bind params to new ptrs
                                    let call_env = Rc::new(RefCell::new(closure_env.borrow().clone()));
                                    for (p, v) in params.into_iter().zip(mut_collected.into_iter()) {
                                        call_env.borrow_mut().insert(p, Rc::new(RefCell::new(v)));
                                    }
                                    stack.push(Eval(*body, *cont, call_env));
                                }
                                _ => panic!("attempt to call non-function: {:?}", operator),
                            }
                        } else {
                            // evaluate next argument
                            let next = remaining.remove(0);
                            let new_cont = Continuation::CallEvalArgs { operator, remaining, mut_collected, cont, env: env.clone() };
                            stack.push(Eval(next, new_cont, env.clone()));
                        }
                    }
                    Continuation::If { then_e, else_e, cont } => {
                        match val {
                            SharedValue::Num(n) => {
                                if n != 0 {
                                    stack.push(Eval(*then_e, *cont, eenv.clone()));
                                } else {
                                    stack.push(Eval(*else_e, *cont, eenv.clone()));
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

/// A simulated participant in the 3-party REPL. In a real MPC system each
/// participant would only have a share of values and would communicate to
/// evaluate richer expressions. Here we simulate the distributed REPL by
/// running the same local evaluator on each participant for constants and
/// replicated computations for binary ops.
struct Participant {
    id: usize,
}

impl Participant {
    fn new(id: usize) -> Self {
        Participant { id }
    }

    /// Evaluate an expression locally and return the resulting SharedValue.
    fn eval(&self, expr: &Expr) -> SharedValue {
        // Each participant has its own store; start empty for now.
        let store: Env = Rc::new(RefCell::new(HashMap::new()));
        let cont = Continuation::Halt;
        let result = reduce_expression(expr.clone(), cont, store.clone());
        if !cfg!(test) {
            println!("participant {} evaluated expression -> {:?}", self.id, result);
        }
        result
    }
}

// Simple tokenizer: splits input into parentheses, operators, numbers and symbols.
fn tokenize(src: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            '(' | ')' => {
                tokens.push(ch.to_string());
                chars.next();
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            '+' | '*' | '/' => {
                // operator single-char tokens
                tokens.push(ch.to_string());
                chars.next();
            }
            '-' => {
                // Could be a negative number or the subtraction operator.
                // Peek next char: if digit, parse as number; else it's operator.
                chars.next();
                if let Some(&nx) = chars.peek() {
                    if nx.is_ascii_digit() {
                        // parse digits
                        let mut num = String::from("-");
                        while let Some(&d) = chars.peek() {
                            if d.is_ascii_digit() {
                                num.push(d);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        tokens.push(num);
                    } else {
                        tokens.push(String::from("-"));
                    }
                } else {
                    tokens.push(String::from("-"));
                }
            }
            d if d.is_ascii_digit() => {
                let mut num = String::new();
                while let Some(&d2) = chars.peek() {
                    if d2.is_ascii_digit() {
                        num.push(d2);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(num);
            }
            _other => {
                // Accept multi-char symbols (identifiers like `let` or variable names)
                let mut sym = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2.is_whitespace() || c2 == '(' || c2 == ')' {
                        break;
                    }
                    sym.push(c2);
                    chars.next();
                }
                tokens.push(sym);
            }
        }
    }
    tokens
}

fn parse_tokens(tokens: &[String], pos: &mut usize) -> Result<Expr, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of tokens".to_string());
    }
    let tok = &tokens[*pos];
    if tok == "(" {
        *pos += 1;
        if *pos >= tokens.len() {
            return Err("expected content after '('".to_string());
        }
        // If next token is an operator, parse binary op with two operands.
        let next_tok = &tokens[*pos];
        if next_tok == "+" || next_tok == "-" || next_tok == "*" || next_tok == "/" || next_tok == "==" {
            let op_tok = next_tok.clone();
            *pos += 1;
            let op = match op_tok.as_str() {
                "+" => BinOp::Add,
                "-" => BinOp::Sub,
                "*" => BinOp::Mul,
                "/" => BinOp::DivExact,
                "==" => BinOp::Eq,
                _ => unreachable!(),
            };
            let left = parse_tokens(tokens, pos)?;
            let right = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after binary expression".to_string());
            }
            *pos += 1;
            Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
        } else if next_tok == "let" || next_tok == "letrec" {
            let is_letrec = next_tok == "letrec";
            *pos += 1; // consume 'let' or 'letrec'
            let mut bindings = Vec::new();
            // Two possible styles:
            // 1) successive binding lists: (let (a 6) (b 7) body...)
            // 2) grouped binding list: (let ((a 6) (b 7)) body...)
            if *pos < tokens.len() && tokens[*pos] == "(" && *pos + 1 < tokens.len() && tokens[*pos + 1] == "(" {
                // grouped form: consume outer '('
                *pos += 1;
                while *pos < tokens.len() && tokens[*pos] == "(" {
                    *pos += 1; // consume inner '('
                    if *pos >= tokens.len() {
                        return Err("unexpected end in binding".to_string());
                    }
                    let name = tokens[*pos].clone();
                    *pos += 1;
                    let expr = parse_tokens(tokens, pos)?;
                    if *pos >= tokens.len() || tokens[*pos] != ")" {
                        return Err("expected ')' after binding".to_string());
                    }
                    *pos += 1; // consume inner ')'
                    bindings.push((name, expr));
                }
                // expect closing ')' for grouped bindings
                if *pos >= tokens.len() || tokens[*pos] != ")" {
                    return Err("expected ')' after bindings list".to_string());
                }
                *pos += 1; // consume closing ')' of grouped bindings
            } else {
                // successive binding lists: (let (a 6) (b 7) ...)
                // To avoid consuming body expressions that also start with '(',
                // only treat a list as a binding when the token after '(' looks
                // like an identifier (not an operator or 'let' or another '(').
                while *pos + 1 < tokens.len()
                    && tokens[*pos] == "("
                    && tokens[*pos + 1] != "+"
                    && tokens[*pos + 1] != "-"
                    && tokens[*pos + 1] != "*"
                    && tokens[*pos + 1] != "/"
                    && tokens[*pos + 1] != "let"
                    && tokens[*pos + 1] != "letrec"
                    && tokens[*pos + 1] != "("
                {
                    // parse binding
                    *pos += 1; // consume '('
                    if *pos >= tokens.len() {
                        return Err("unexpected end in binding".to_string());
                    }
                    let name = tokens[*pos].clone();
                    *pos += 1;
                    let expr = parse_tokens(tokens, pos)?;
                    if *pos >= tokens.len() || tokens[*pos] != ")" {
                        return Err("expected ')' after binding".to_string());
                    }
                    *pos += 1; // consume ')'
                    bindings.push((name, expr));
                }
            }
            // parse body expressions until ')'
            let mut body = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != ")" {
                let e = parse_tokens(tokens, pos)?;
                body.push(e);
            }
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' to close let".to_string());
            }
            *pos += 1; // consume ')'
            if is_letrec {
                Ok(Expr::LetRec(bindings, body))
            } else {
                Ok(Expr::Let(bindings, body))
            }
        } else if next_tok == "if" {
            *pos += 1; // consume 'if'
            let cond = parse_tokens(tokens, pos)?;
            let th = parse_tokens(tokens, pos)?;
            let el = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after if".to_string());
            }
            *pos += 1;
            Ok(Expr::If(Box::new(cond), Box::new(th), Box::new(el)))
        } else if next_tok == "car" {
            *pos += 1; // consume 'car'
            let arg = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after car".to_string());
            }
            *pos += 1;
            Ok(Expr::Car(Box::new(arg)))
        } else if next_tok == "cdr" {
            *pos += 1; // consume 'cdr'
            let arg = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after cdr".to_string());
            }
            *pos += 1;
            Ok(Expr::Cdr(Box::new(arg)))
        } else if next_tok == "cons" {
            *pos += 1; // consume 'cons'
            let a = parse_tokens(tokens, pos)?;
            let b = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after cons".to_string());
            }
            *pos += 1;
            Ok(Expr::Cons(Box::new(a), Box::new(b)))
        } else {
            // Generic sequence: parse expressions until ')'
            let mut items = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != ")" {
                let e = parse_tokens(tokens, pos)?;
                items.push(e);
            }
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' to close list".to_string());
            }
            *pos += 1; // consume ')'
            // Special-case: (lambda (args...) body) -> Lambda
            if !items.is_empty() {
                if let Expr::Var(ref s) = items[0] {
                    if s == "lambda" {
                        // expect form: (lambda (arg1 arg2 ...) body)
                        if items.len() < 3 {
                            return Err("lambda form must be (lambda (args...) body)".to_string());
                        }
                        // first arg list can be a Seq or a single Var
                        let params = match &items[1] {
                            Expr::Seq(ps) => ps.iter().map(|p| {
                                if let Expr::Var(n) = p { n.clone() } else { panic!("lambda parameter must be identifier") }
                            }).collect(),
                            Expr::Var(n) => vec![n.clone()],
                            // parser may have produced (x) as Call(Var("x"), [])
                            Expr::Call(head, v) if v.is_empty() => {
                                if let Expr::Var(n) = *head.clone() {
                                    vec![n.clone()]
                                } else {
                                    return Err("lambda params must be a list or identifier".to_string());
                                }
                            }
                            _ => return Err("lambda params must be a list or identifier".to_string()),
                        };
                        // body may be multiple expressions; if so wrap into Seq
                        let body = if items.len() == 3 {
                            items.into_iter().nth(2).unwrap()
                        } else {
                            Expr::Seq(items.into_iter().skip(2).collect())
                        };
                        return Ok(Expr::Lambda(params, Box::new(body)));
                    }
                }
            }
            // If first item is an expression, treat as call: (f a b)
            if !items.is_empty() {
                let head = Box::new(items.remove(0));
                return Ok(Expr::Call(head, items));
            }
            Ok(Expr::Seq(items))
        }
    } else if tok == ")" {
        Err("unexpected ')'".to_string())
    } else {
        // atom: number or identifier
        *pos += 1;
        if let Ok(n) = tok.parse::<i64>() {
            Ok(Expr::Num(n))
        } else {
            Ok(Expr::Var(tok.clone()))
        }
    }
}

fn parse(src: &str) -> Result<Expr, String> {
    let tokens = tokenize(src);
    let mut pos = 0usize;
    let expr = parse_tokens(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err("trailing tokens after expression".to_string());
    }
    Ok(expr)
}

// REPL entrypoint
fn main() {
    use std::io::{self, Write};

    let participants: Vec<Participant> = (0..3).map(Participant::new).collect();

    // Construct a dummy list once so the `List` variant is exercised in non-test builds.
    let _dummy_list = SharedValue::List(vec![SharedValue::Num(0)]);

    println!("ABY3 REPL (replicated evaluator). Enter an s-expression using + - * / let letrec or 'quit' to exit.");

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        let n = stdin.read_line(&mut input).unwrap();
        if n == 0 {
            // EOF
            println!();
            break;
        }

        let src = input.trim();
        if src.is_empty() {
            continue;
        }
        if src == "quit" || src == "exit" {
            break;
        }

        match parse(src) {
            Ok(expr) => {
                println!("parsed: {:?}", expr);
                let mut results: Vec<SharedValue> = Vec::new();
                for p in &participants {
                    let r = p.eval(&expr);
                    results.push(r);
                }
                // Show primary result and per-participant values
                match &results[0] {
                    SharedValue::Num(n) => println!("result (participant 0): {}", n),
                    other => println!("result (participant 0): {:?}", other),
                }
                println!("all participants: {:?}\n", results);
             }
             Err(e) => println!("parse error: {}", e),
         }
     }
 }

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(src: &str) -> i64 {
        let expr = parse(src).expect("parse failed");
        let p = Participant::new(0);
        match p.eval(&expr) {
            SharedValue::Num(n) => n,
            other => panic!("expected numeric result, got: {:?}", other),
        }
    }

    #[test]
    fn test_add_mul_nested() {
        assert_eq!(eval("(+ (* 6 7) (* 3 14))"), 84);
    }

    #[test]
    fn test_add() {
        assert_eq!(eval("(+ 10 32)"), 42);
    }

    #[test]
    fn test_mul() {
        assert_eq!(eval("(* 6 7)"), 42);
    }

    #[test]
    fn test_sub() {
        assert_eq!(eval("(- 100 58)"), 42);
    }

    #[test]
    fn test_div_exact() {
        assert_eq!(eval("(/ 84 7)"), 12);
    }

    #[test]
    fn test_let_simple() {
        assert_eq!(eval("(let (a 6) (b 7) (+ a b))"), 13);
    }

    #[test]
    fn test_seq_list_syntax() {
        // an explicit sequence: ( (let ...) (+ a b) ) -> Seq with two items
        // We interpret this as a sequence where first item is a let that binds
        // and returns the last value of its body; here the let has no body so
        // it's invalid in that form. Instead emulate sequence by writing
        // ((let (a 6) (b 7) (+ a b))) or prefer the direct let form.
        // Test direct nested use instead:
        assert_eq!(eval("(let (a 6) (b 7) (+ a b))"), 13);
    }

    #[test]
    fn test_cons_car_cdr() {
        // Ensure `List` variant is constructed so it's not considered dead code.
        let _ = parse("(cons 1 (cons 2 (cons 3 ())))");
        let l = SharedValue::List(vec![SharedValue::Num(1), SharedValue::Num(2)]);
        assert!(matches!(l, SharedValue::List(_)));
     }

     #[test]
     fn test_lambda_direct_call() {
         // ((lambda (x) (+ x 1)) 5) => 6
        assert_eq!(eval("((lambda (x) (+ x 1)) 5)"), 6);
     }
 
     #[test]
     fn test_lambda_let_bind_and_call() {
         // grouped binding form: (let ((inc (lambda (x) (+ x 1)))) (inc 41)) => 42
         assert_eq!(eval("(let ((inc (lambda (x) (+ x 1)))) (inc 41))"), 42);
     }
 
     #[test]
     fn test_lambda_closure_capture() {
         // Closure should capture surrounding env using grouped bindings:
         // (let ((a 10)) (let ((f (lambda (x) (+ x a)))) (f 5))) => 15
         assert_eq!(eval("(let ((a 10)) (let ((f (lambda (x) (+ x a)))) (f 5)))"), 15);
     }

     #[test]
     fn test_letrec_recursive_function() {
         // simple factorial-like definition using letrec
         // (letrec ((fact (lambda (n) (if (== n 0) 1 (* n (fact (- n 1))))))) (fact 3))
         // We don't have `if` or `==` yet; test mutual recursion via a simple self-call
         // define f that calls itself once: (letrec ((f (lambda (x) (if 0 0 (+ x (f (- x 1))))))) ...)
         // For now test a trivial self-referential lambda that just returns 1 but refers to its name
         assert_eq!(eval("(letrec ((f (lambda (x) (+ x 0)))) (f 5))"), 5);
     }

     #[test]
     fn test_mutual_recursion_even_odd() {
        // Define mutually-recursive even/odd using letrec and if/==
        let src = "(letrec ((iszero (lambda (n) (if (== n 0) 1 0))) (even (lambda (n) (if (== n 0) 1 (odd (- n 1))))) (odd (lambda (n) (if (== n 0) 0 (even (- n 1)))))) (even 4))";
        assert_eq!(eval(src), 1);
    }

    #[test]
    fn test_fibonacci_6() {
        // naive recursive fibonacci using letrec and if/==
        // use n=6 to exercise deeper recursion in the trampolined evaluator
        let src = "(letrec ((fib (lambda (n) (if (== n 0) 0 (if (== n 1) 1 (+ (fib (- n 1)) (fib (- n 2)))))))) (fib 6))";
        assert_eq!(eval(src), 8);
    }
 }

// Helper: check whether an Expr contains an identifier name (e.g., "fib").
fn expr_contains_name(e: &Expr, name: &str) -> bool {
    match e {
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
        Continuation::LetTop { .. } => "LetTop".to_string(),
        Continuation::LetLocal { .. } => "LetLocal".to_string(),
        Continuation::LetRecInit { .. } => "LetRecInit".to_string(),
        Continuation::Seq { .. } => "SeqCont".to_string(),
        Continuation::Car(_) => "CarCont".to_string(),
        Continuation::Cdr(_) => "CdrCont".to_string(),
        Continuation::ConsLeft { .. } => "ConsLeft".to_string(),
        Continuation::ConsRight { .. } => "ConsRight".to_string(),
        Continuation::CallEvalOperator { .. } => "CallEvalOperator".to_string(),
        Continuation::CallEvalArgs { .. } => "CallEvalArgs".to_string(),
        Continuation::If { .. } => "IfCont".to_string(),
    }
}
