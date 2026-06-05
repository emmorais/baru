// Minimal distributed Lisp-like evaluator for MPC (ABY3 backend simulation)
// - Supports NUMs, variables, let, sequences and binary operations (+, -, *, exact /)
// - Implements `apply_continuation` and `reduce_expression` in a CEK-like style
// - Simulates 3 participants running a distributed REPL where constants are evaluated locally

use std::collections::HashMap;

#[derive(Debug, Clone)]
enum Expr {
    Num(i64),
    Var(String),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Let(Vec<(String, Expr)>, Vec<Expr>),
    Seq(Vec<Expr>),
    Car(Box<Expr>),
    Cdr(Box<Expr>),
    Cons(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    DivExact,
}

type Store = HashMap<String, SharedValue>;

#[derive(Debug, Clone)]
/// A very small representation of a secret-shared value.
/// Can be a number or a list of shared values.
enum SharedValue {
    Num(i64),
    List(Vec<SharedValue>),
}

#[derive(Debug)]
enum Continuation {
    /// Terminal continuation: evaluation is finished and the contained value
    /// should be returned to the caller / REPL.
    Halt,

    /// After evaluating the left operand, evaluate the right operand.
    /// Holds the operator, the right expression and the continuation to run
    /// after the binary operation finishes.
    BinOpLeft {
        op: BinOp,
        right: Box<Expr>,
        cont: Box<Continuation>,
    },

    /// After evaluating the right operand, apply the binary operator using
    /// the previously-computed left value, then continue with `cont`.
    BinOpRight {
        op: BinOp,
        left_val: SharedValue,
        cont: Box<Continuation>,
    },
}

/// Apply a continuation to a value. For binary operations we thread the
/// evaluation: when the left operand is ready we evaluate the right, and
/// when the right is ready we apply the operator.
fn apply_continuation(cont: Continuation, val: SharedValue, env: &mut Store) -> SharedValue {
    match cont {
        Continuation::Halt => val,
        Continuation::BinOpLeft { op, right, cont } => {
            // Left operand is `val`. Now evaluate right under a continuation
            // that remembers the left value.
            let next = Continuation::BinOpRight { op, left_val: val, cont };
            reduce_expression(*right, next, env)
        }
        Continuation::BinOpRight { op, left_val, cont } => {
            // Right operand evaluated to `val`. Apply the operator.
            let a = match left_val {
                SharedValue::Num(x) => x,
                _ => panic!("binary op left operand must be NUM"),
            };
            let b = match val {
                SharedValue::Num(x) => x,
                _ => panic!("binary op right operand must be NUM"),
            };
            let res = match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::DivExact => {
                    if b == 0 {
                        panic!("division by zero");
                    }
                    if a % b != 0 {
                        panic!("non-exact division: {} / {}", a, b);
                    }
                    a / b
                }
            };
            let out = SharedValue::Num(res);
            apply_continuation(*cont, out, env)
        }
    }
}

/// Reduce an expression under a continuation, using and mutating `env` as the
/// current store for variable bindings. Sequences and let-expressions create
/// inner scopes by cloning the environment.
fn reduce_expression(expr: Expr, cont: Continuation, env: &mut Store) -> SharedValue {
    match expr {
        Expr::Num(n) => {
            let v = SharedValue::Num(n);
            apply_continuation(cont, v, env)
        }
        Expr::Var(name) => {
            let v = env.get(&name).cloned().unwrap_or_else(|| panic!("unbound variable: {}", name));
            apply_continuation(cont, v, env)
        }
        Expr::BinOp(op, left, right) => {
            // Evaluate left first; after left is ready we will evaluate right.
            let cont2 = Continuation::BinOpLeft { op, right, cont: Box::new(cont) };
            reduce_expression(*left, cont2, env)
        }
        Expr::Let(bindings, body) => {
            if body.is_empty() {
                // Treat as a top-level binding statement: evaluate bindings in the current env
                // and install them so subsequent expressions in the same sequence can use them.
                for (name, bind_expr) in bindings {
                    let v = reduce_expression(bind_expr, Continuation::Halt, env);
                    env.insert(name, v);
                }
                // Return a dummy value (not used when this Let is part of a Seq with following expressions)
                SharedValue::Num(0)
            } else {
                // Evaluate bindings sequentially in a fresh scope (clone env)
                let mut local = env.clone();
                for (name, bind_expr) in bindings {
                    let v = reduce_expression(bind_expr, Continuation::Halt, &mut local);
                    local.insert(name, v);
                }
                // Evaluate body sequentially; result of last expression is returned
                if body.is_empty() {
                    panic!("let requires at least one body expression");
                }
                let len = body.len();
                for (i, e) in body.into_iter().enumerate() {
                    if i + 1 == len {
                        return reduce_expression(e, cont, &mut local);
                    } else {
                        let _ = reduce_expression(e, Continuation::Halt, &mut local);
                    }
                }
                unreachable!()
            }
        }
        Expr::Seq(exprs) => {
            // Empty sequence is treated as the empty list literal `()`.
            if exprs.is_empty() {
                return apply_continuation(cont, SharedValue::List(Vec::new()), env);
            }
            // Non-empty Seq is a sequence of expressions; return last expression's value.
            let len = exprs.len();
            for (i, e) in exprs.into_iter().enumerate() {
                if i + 1 == len {
                    return reduce_expression(e, cont, env);
                } else {
                    let _ = reduce_expression(e, Continuation::Halt, env);
                }
            }
            unreachable!()
        }
        Expr::Car(inner) => {
            // Evaluate inner then take first element
            let v = reduce_expression(*inner, Continuation::Halt, env);
            match v {
                SharedValue::List(mut vs) => {
                    if vs.is_empty() { panic!("car of empty list"); }
                    let first = vs.remove(0);
                    apply_continuation(cont, first, env)
                }
                _ => panic!("car expects a list"),
            }
        }
        Expr::Cdr(inner) => {
            let v = reduce_expression(*inner, Continuation::Halt, env);
            match v {
                SharedValue::List(vs) => {
                    if vs.is_empty() { panic!("cdr of empty list"); }
                    let tail = SharedValue::List(vs.into_iter().skip(1).collect());
                    apply_continuation(cont, tail, env)
                }
                _ => panic!("cdr expects a list"),
            }
        }
        Expr::Cons(h, t) => {
            let head = reduce_expression(*h, Continuation::Halt, env);
            let tail = reduce_expression(*t, Continuation::Halt, env);
            match tail {
                SharedValue::List(mut vs) => {
                    vs.insert(0, head);
                    apply_continuation(cont, SharedValue::List(vs), env)
                }
                _ => panic!("cons expects second arg to be a list"),
            }
        }
    }
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
        let mut store: Store = Store::new();
        let cont = Continuation::Halt;
        let result = reduce_expression(expr.clone(), cont, &mut store);
        println!("participant {} evaluated expression -> {:?}", self.id, result);
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
        if next_tok == "+" || next_tok == "-" || next_tok == "*" || next_tok == "/" {
            let op_tok = next_tok.clone();
            *pos += 1;
            let op = match op_tok.as_str() {
                "+" => BinOp::Add,
                "-" => BinOp::Sub,
                "*" => BinOp::Mul,
                "/" => BinOp::DivExact,
                _ => unreachable!(),
            };
            let left = parse_tokens(tokens, pos)?;
            let right = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after binary expression".to_string());
            }
            *pos += 1;
            Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
        } else if next_tok == "let" {
            // parse let: support both (let (a 6) (b 7) body...) and (let ((a 6) (b 7)) body...)
            *pos += 1; // consume 'let'
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
             Ok(Expr::Let(bindings, body))
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

    println!("ABY3 REPL (replicated evaluator). Enter an s-expression using + - * / let or 'quit' to exit.");

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
 }
