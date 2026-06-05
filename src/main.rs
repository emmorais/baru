// Minimal distributed Lisp-like evaluator for MPC (ABY3 backend simulation)
// - Supports NUMs and binary operations (+, -, *, exact /)
// - Implements `apply_continuation` and `reduce_expression` in a CEK-like style
// - Simulates 3 participants running a distributed REPL where constants are evaluated locally

#[derive(Debug, Clone)]
enum Expr {
    Num(i64),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
/// A very small representation of a secret-shared numeric value.
/// For this minimal example we keep it simple: the value is the same
/// on every participant (no communication required for constants).
struct SharedValue {
    value: i64,
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    DivExact,
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
fn apply_continuation(cont: Continuation, val: SharedValue) -> SharedValue {
    match cont {
        Continuation::Halt => val,
        Continuation::BinOpLeft { op, right, cont } => {
            // Left operand is `val`. Now evaluate right under a continuation
            // that remembers the left value.
            let next = Continuation::BinOpRight { op, left_val: val, cont };
            reduce_expression(*right, next)
        }
        Continuation::BinOpRight { op, left_val, cont } => {
            // Right operand evaluated to `val`. Apply the operator.
            let a = left_val.value;
            let b = val.value;
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
            let out = SharedValue { value: res };
            apply_continuation(*cont, out)
        }
    }
}

/// Reduce an expression under a continuation. For NUM constants and when both
/// operands are NUMs the computation happens locally and we return a
/// SharedValue. If operands are expressions we evaluate left then right,
/// threading continuations as above.
fn reduce_expression(expr: Expr, cont: Continuation) -> SharedValue {
    match expr {
        Expr::Num(n) => {
            let v = SharedValue { value: n };
            apply_continuation(cont, v)
        }
        Expr::BinOp(op, left, right) => {
            // Evaluate left first; after left is ready we will evaluate right.
            let cont2 = Continuation::BinOpLeft { op, right, cont: Box::new(cont) };
            reduce_expression(*left, cont2)
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
        // For this minimal example there is a single terminal continuation.
        let cont = Continuation::Halt;
        // Work with an owned expression so we can move boxes into continuations.
        let result = reduce_expression(expr.clone(), cont);
        println!("participant {} evaluated expression -> {}", self.id, result.value);
        result
    }
}

// Simple tokenizer: splits input into parentheses, operators and numbers.
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
                // Accept multi-char symbols (not needed now but harmless)
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
            return Err("expected operator after '('".to_string());
        }
        let op_tok = &tokens[*pos];
        *pos += 1;
        let op = match op_tok.as_str() {
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            "*" => BinOp::Mul,
            "/" => BinOp::DivExact,
            _other => return Err(format!("unknown operator: {}", _other)),
        };
        let left = parse_tokens(tokens, pos)?;
        let right = parse_tokens(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos] != ")" {
            return Err("expected ')' after binary expression".to_string());
        }
        *pos += 1;
        Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
    } else if tok == ")" {
        Err("unexpected ')'".to_string())
    } else {
        // number
        *pos += 1;
        match tok.parse::<i64>() {
            Ok(n) => Ok(Expr::Num(n)),
            Err(_) => Err(format!("invalid token: {}", tok)),
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

fn main() {
    // Parse and evaluate some example s-expressions
    let examples = [
        "(+ (* 6 7) (* 3 14))",
        "(+ 10 32)",
        "(* 6 7)",
        "(- 100 58)",
        "(/ 84 7)",
    ];

    let participants: Vec<Participant> = (0..3).map(Participant::new).collect();

    for src in &examples {
        match parse(src) {
            Ok(expr) => {
                println!("parsed '{}': {:?}", src, expr);
                let mut results: Vec<i64> = Vec::new();
                for p in &participants {
                    let r = p.eval(&expr);
                    results.push(r.value);
                }
                println!("final results from participants: {:?}\n", results);
            }
            Err(e) => println!("parse error for '{}': {}", src, e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(src: &str) -> i64 {
        let expr = parse(src).expect("parse failed");
        let p = Participant::new(0);
        p.eval(&expr).value
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
}
