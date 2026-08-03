#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    DivExact,
    Eq,
    Lt,
    Gt,
    Le,
    Ge,
    Mod,
    And,
    Or,
    AshLeft,
    AshRight,
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Atom,
    Not,
    Neg,
    BitLen,
    /// `(null? lst)` — 1 if the list is empty, 0 otherwise.
    IsNull,
}

// Very small tokenizer + parser extracted from main.rs (keeps same behavior)
pub fn tokenize(src: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            '(' | ')' => {
                tokens.push(ch.to_string());
                chars.next();
            }
            '"' => {
                // parse string literal
                chars.next();
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '"' {
                        break;
                    }
                    if c == '\\' {
                        if let Some(&esc) = chars.peek() {
                            chars.next();
                            match esc {
                                'n' => s.push('\n'),
                                't' => s.push('\t'),
                                '\\' => s.push('\\'),
                                '"' => s.push('"'),
                                _ => s.push(esc),
                            }
                        }
                    } else {
                        s.push(c);
                    }
                }
                tokens.push(format!("\"{}\"", s));
            }
            '\'' => {
                chars.next();
                // Peek at what follows the single quote to decide: char literal or quote shorthand.
                if let Some(&next_ch) = chars.peek() {
                    // Char literal: 'x' — single character followed by closing quote
                    // We need to look two chars ahead to distinguish 'a' from 'abc or '(...)
                    let is_char_literal = {
                        let mut it = chars.clone();
                        let first = it.next();
                        let second = it.next();
                        matches!((first, second), (Some(c), Some('\'')) if !c.is_whitespace() && c != '(')
                    };

                    if is_char_literal {
                        // Char literal: 'a' — consume the char and closing quote
                        let ch = next_ch;
                        chars.next();
                        if let Some(&closing) = chars.peek() {
                            if closing == '\'' {
                                chars.next();
                            }
                        }
                        tokens.push(format!("'{}'", ch));
                    } else if next_ch == '('
                        || next_ch.is_alphanumeric()
                        || next_ch == '"'
                        || next_ch == '#'
                        || next_ch == '\''
                    {
                        // Quote shorthand: 'expr  →  (quote expr)
                        tokens.push("(".to_string());
                        tokens.push("quote".to_string());
                        let mut depth = 0i32;
                        let mut saw_first = false;
                        let mut inner_tokens: Vec<String> = Vec::new();
                        while let Some(&c) = chars.peek() {
                            if c.is_whitespace() && !saw_first {
                                chars.next();
                                continue;
                            }
                            match c {
                                '(' => {
                                    depth += 1;
                                    saw_first = true;
                                    inner_tokens.push("(".to_string());
                                    chars.next();
                                }
                                ')' => {
                                    depth -= 1;
                                    inner_tokens.push(")".to_string());
                                    chars.next();
                                    if depth <= 0 {
                                        break;
                                    }
                                }
                                '"' => {
                                    chars.next();
                                    let mut s = String::new();
                                    while let Some(&sc) = chars.peek() {
                                        chars.next();
                                        if sc == '"' {
                                            break;
                                        }
                                        if sc == '\\' {
                                            if let Some(&esc) = chars.peek() {
                                                chars.next();
                                                match esc {
                                                    'n' => s.push('\n'),
                                                    't' => s.push('\t'),
                                                    '\\' => s.push('\\'),
                                                    '"' => s.push('"'),
                                                    _ => s.push(esc),
                                                }
                                            }
                                        } else {
                                            s.push(sc);
                                        }
                                    }
                                    inner_tokens.push(format!("\"{}\"", s));
                                    saw_first = true;
                                    if depth == 0 {
                                        break;
                                    }
                                }
                                ';' => {
                                    // Line comment inside a quoted form.
                                    for cc in chars.by_ref() {
                                        if cc == '\n' {
                                            break;
                                        }
                                    }
                                }
                                c if c.is_whitespace() => {
                                    chars.next();
                                    if saw_first && depth == 0 {
                                        break;
                                    }
                                }
                                _ => {
                                    let mut sym = String::new();
                                    while let Some(&sc) = chars.peek() {
                                        if sc.is_whitespace() || sc == '(' || sc == ')' {
                                            break;
                                        }
                                        sym.push(sc);
                                        chars.next();
                                    }
                                    inner_tokens.push(sym);
                                    saw_first = true;
                                    if depth == 0 {
                                        break;
                                    }
                                }
                            }
                        }
                        tokens.extend(inner_tokens);
                        tokens.push(")".to_string());
                    }
                }
            }
            '#' => {
                // boolean #t or #f
                chars.next();
                if let Some(&c) = chars.peek() {
                    chars.next();
                    let tok = format!("#{}", c);
                    tokens.push(tok);
                }
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            '+' | '*' | '/' | '%' => {
                tokens.push(ch.to_string());
                chars.next();
            }
            '<' | '>' => {
                // possibly <= or >=
                chars.next();
                if let Some(&'=') = chars.peek() {
                    chars.next();
                    tokens.push(format!("{}=", ch));
                } else {
                    tokens.push(ch.to_string());
                }
            }
            '-' => {
                chars.next();
                if let Some(&nx) = chars.peek() {
                    if nx.is_ascii_digit() {
                        let mut num = String::from("-");
                        while let Some(&d) = chars.peek() {
                            if d.is_ascii_digit() {
                                num.push(d);
                                chars.next();
                            } else if d == '.' {
                                num.push(d);
                                chars.next();
                                while let Some(&d2) = chars.peek() {
                                    if d2.is_ascii_digit() {
                                        num.push(d2);
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                                break;
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
            ';' => {
                // comment until end of line
                chars.next();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '\n' {
                        break;
                    }
                }
            }
            d if d.is_ascii_digit() => {
                let mut num = String::new();
                while let Some(&d2) = chars.peek() {
                    if d2.is_ascii_digit() {
                        num.push(d2);
                        chars.next();
                    } else if d2 == '.' {
                        // could be a decimal literal like 42.0
                        num.push(d2);
                        chars.next();
                        while let Some(&d3) = chars.peek() {
                            if d3.is_ascii_digit() {
                                num.push(d3);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        break;
                    } else {
                        break;
                    }
                }
                tokens.push(num);
            }
            _ => {
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

// We keep parse_tokens and parse in parser.rs for now, but they refer to Ast types in ast.rs
use crate::interpreter::ast::Expr;

// ─── Derived forms ───────────────────────────────────────────────────────────
//
// `cond`, `case`, `when`, `unless`, `let*`, named `let` and `begin` are all
// *desugared here* into the core AST rather than given evaluator support.
//
// That is a deliberate choice, and the reason is the circuit backends. The ZKP
// and FHE compilers (`zkp::circuit_compiler`, `mpcc`) flatten a fixed set of
// `Expr` variants; anything new would have to be taught to each of them, and
// `CVal`-based flattening resolves `if` statically to terminate its unrolling.
// A form that lowers to `If`/`Let`/`LetRec`/`Seq` needs no compiler change and
// cannot desynchronise the interpreter from the compilers, because after
// parsing there is nothing left to disagree about.

/// Unique suffix for parser-generated temporaries, so a desugared binding can
/// never capture a user variable or a temporary from an enclosing form.
fn gensym(stem: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    // A space keeps the name unwritable in source: the tokenizer splits on it.
    format!("{stem} {}", N.fetch_add(1, Ordering::Relaxed))
}

/// Parse expressions up to (and consuming) the closing paren.
fn parse_body(tokens: &[String], pos: &mut usize, form: &str) -> Result<Vec<Expr>, String> {
    let mut out = Vec::new();
    while *pos < tokens.len() && tokens[*pos] != ")" {
        out.push(parse_tokens(tokens, pos)?);
    }
    if *pos >= tokens.len() {
        return Err(format!("expected ')' to close {form}"));
    }
    *pos += 1;
    Ok(out)
}

/// Collapse a body into one expression, `Seq` when there is more than one.
fn body_expr(mut body: Vec<Expr>, form: &str) -> Result<Expr, String> {
    match body.len() {
        0 => Err(format!("{form} needs at least one body expression")),
        1 => Ok(body.pop().unwrap()),
        _ => Ok(Expr::Seq(body)),
    }
}

/// `((name init) ...)` — the binding list shared by `let`, `let*` and `letrec`.
/// Assumes `*pos` is at the opening paren of the whole list.
fn parse_bindings(
    tokens: &[String],
    pos: &mut usize,
    form: &str,
) -> Result<Vec<(String, Expr)>, String> {
    if *pos >= tokens.len() || tokens[*pos] != "(" {
        return Err(format!("{form} expects a binding list"));
    }
    *pos += 1;
    let mut out = Vec::new();
    while *pos < tokens.len() && tokens[*pos] == "(" {
        *pos += 1;
        if *pos >= tokens.len() {
            return Err(format!("unterminated binding in {form}"));
        }
        let name = tokens[*pos].clone();
        if name == "(" || name == ")" {
            return Err(format!("{form} binding needs a name"));
        }
        *pos += 1;
        // `:sintN` / `:sintN-from-P` sugar is handled by the core `let`, so a
        // derived form that wants it should lower to `let` rather than reparse
        // the annotation here.
        if *pos < tokens.len() && tokens[*pos].starts_with(':') {
            return Err(format!(
                "{form} does not support the `{}` share annotation; use `let`",
                tokens[*pos]
            ));
        }
        let init = parse_tokens(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos] != ")" {
            return Err(format!("expected ')' after binding in {form}"));
        }
        *pos += 1;
        out.push((name, init));
    }
    if *pos >= tokens.len() || tokens[*pos] != ")" {
        return Err(format!("expected ')' to close {form} bindings"));
    }
    *pos += 1;
    Ok(out)
}

/// One `cond` / `case` clause: `None` test means `else`.
struct Clause {
    test: Option<Expr>,
    body: Vec<Expr>,
}

/// Fold clauses right-to-left into nested `If`. A `cond` that falls off the end
/// yields `0`, matching how this language already spells "no value" (`when`
/// without a taken branch, and the `1`/`0` comparisons).
fn clauses_to_if(clauses: Vec<Clause>, form: &str) -> Result<Expr, String> {
    let mut out: Option<Expr> = None;
    for c in clauses.into_iter().rev() {
        let body = body_expr(c.body, form)?;
        out = Some(match c.test {
            None => {
                if out.is_some() {
                    return Err(format!("{form}: `else` must be the last clause"));
                }
                body
            }
            Some(t) => Expr::If(
                Box::new(t),
                Box::new(body),
                Box::new(out.unwrap_or(Expr::Num(0))),
            ),
        });
    }
    out.ok_or_else(|| format!("{form} needs at least one clause"))
}

/// Parse `(test body...)` clauses until the closing paren of the whole form.
fn parse_clauses(tokens: &[String], pos: &mut usize, form: &str) -> Result<Vec<Clause>, String> {
    let mut clauses = Vec::new();
    while *pos < tokens.len() && tokens[*pos] == "(" {
        *pos += 1;
        if *pos >= tokens.len() {
            return Err(format!("unterminated clause in {form}"));
        }
        let test = if tokens[*pos] == "else" {
            *pos += 1;
            None
        } else {
            Some(parse_tokens(tokens, pos)?)
        };
        let body = parse_body(tokens, pos, &format!("{form} clause"))?;
        clauses.push(Clause { test, body });
    }
    if *pos >= tokens.len() || tokens[*pos] != ")" {
        return Err(format!("expected ')' to close {form}"));
    }
    *pos += 1;
    Ok(clauses)
}

pub fn parse_tokens(tokens: &[String], pos: &mut usize) -> Result<Expr, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of tokens".to_string());
    }
    let tok = &tokens[*pos];
    if tok == "(" {
        *pos += 1;
        if *pos >= tokens.len() {
            return Err("expected content after '('".to_string());
        }
        let next_tok = &tokens[*pos];
        // quoting
        if next_tok == "quote" {
            *pos += 1;
            let quoted = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after quote".to_string());
            }
            *pos += 1;
            return Ok(Expr::Quote(Box::new(quoted)));
        }
        // private annotation for ZKP witness marking
        if next_tok == "private" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after private".to_string());
            }
            *pos += 1;
            return Ok(Expr::Private(Box::new(inner)));
        }
        if next_tok == "eval" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after eval".to_string());
            }
            *pos += 1;
            return Ok(Expr::Eval(Box::new(inner)));
        }
        // ─── Mutation and application (real AST variants) ───────────────────
        if next_tok == "define" || next_tok == "set!" {
            let is_define = next_tok == "define";
            let form = if is_define { "define" } else { "set!" };
            *pos += 1;
            if *pos >= tokens.len() {
                return Err(format!("{form} needs a name"));
            }
            // `(define (f a b) body...)` is sugar for `(define f (lambda (a b) body...))`.
            if is_define && tokens[*pos] == "(" {
                *pos += 1;
                if *pos >= tokens.len() {
                    return Err("define: empty function signature".to_string());
                }
                let fname = tokens[*pos].clone();
                *pos += 1;
                let mut params = Vec::new();
                while *pos < tokens.len() && tokens[*pos] != ")" {
                    params.push(tokens[*pos].clone());
                    *pos += 1;
                }
                if *pos >= tokens.len() {
                    return Err("define: unterminated parameter list".to_string());
                }
                *pos += 1; // consume ')' of the signature
                let body = body_expr(parse_body(tokens, pos, "define")?, "define")?;
                return Ok(Expr::Define(
                    fname,
                    Box::new(Expr::Lambda(params, Box::new(body))),
                ));
            }
            let name = tokens[*pos].clone();
            if name == ")" {
                return Err(format!("{form} needs a name"));
            }
            *pos += 1;
            let value = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err(format!("expected ')' after {form}"));
            }
            *pos += 1;
            return Ok(if is_define {
                Expr::Define(name, Box::new(value))
            } else {
                Expr::Set(name, Box::new(value))
            });
        }
        // ─── Derived forms (desugared; see the note above `gensym`) ─────────
        if next_tok == "begin" {
            *pos += 1;
            let body = parse_body(tokens, pos, "begin")?;
            return body_expr(body, "begin");
        }
        if next_tok == "when" || next_tok == "unless" {
            let is_when = next_tok == "when";
            let form = if is_when { "when" } else { "unless" };
            *pos += 1;
            let cond = parse_tokens(tokens, pos)?;
            let body = body_expr(parse_body(tokens, pos, form)?, form)?;
            let zero = Expr::Num(0);
            return Ok(if is_when {
                Expr::If(Box::new(cond), Box::new(body), Box::new(zero))
            } else {
                Expr::If(Box::new(cond), Box::new(zero), Box::new(body))
            });
        }
        if next_tok == "cond" {
            *pos += 1;
            let clauses = parse_clauses(tokens, pos, "cond")?;
            return clauses_to_if(clauses, "cond");
        }
        if next_tok == "case" {
            // (case key ((d ...) body...) ... (else body...))
            //   => (let ((tmp key)) (if (or (== tmp d) ...) body ... ))
            // The key is bound once so it is evaluated once, as Scheme requires.
            *pos += 1;
            let key = parse_tokens(tokens, pos)?;
            let raw = parse_clauses(tokens, pos, "case")?;
            let tmp = gensym("case-key");
            let mut clauses = Vec::with_capacity(raw.len());
            for c in raw {
                let test = match c.test {
                    None => None,
                    Some(datums) => {
                        // The clause head parsed as an expression; recover the
                        // datum list from whatever shape it took.
                        let items: Vec<Expr> = match datums {
                            Expr::Call(head, rest) => {
                                let mut v = vec![*head];
                                v.extend(rest);
                                v
                            }
                            Expr::Seq(v) => v,
                            single => vec![single],
                        };
                        if items.is_empty() {
                            return Err("case clause needs at least one datum".to_string());
                        }
                        // `==` yields 1/0 and `or` is bitwise, so a chain of
                        // `or`s over them is a correct logical disjunction.
                        let mut acc: Option<Expr> = None;
                        for d in items {
                            let eq = Expr::BinOp(
                                BinOp::Eq,
                                Box::new(Expr::Var(tmp.clone())),
                                Box::new(d),
                            );
                            acc = Some(match acc {
                                None => eq,
                                Some(a) => {
                                    Expr::BinOp(BinOp::Or, Box::new(a), Box::new(eq))
                                }
                            });
                        }
                        acc
                    }
                };
                clauses.push(Clause { test, body: c.body });
            }
            let inner = clauses_to_if(clauses, "case")?;
            return Ok(Expr::Let(vec![(tmp, key)], vec![inner]));
        }
        if next_tok == "let*" {
            // (let* ((a x) (b y)) body...) => (let ((a x)) (let ((b y)) body...))
            *pos += 1;
            let bindings = parse_bindings(tokens, pos, "let*")?;
            let body = parse_body(tokens, pos, "let*")?;
            if bindings.is_empty() {
                return body_expr(body, "let*");
            }
            let mut out = body;
            for b in bindings.into_iter().rev() {
                out = vec![Expr::Let(vec![b], out)];
            }
            return body_expr(out, "let*");
        }
        // Named `let`: (let loop ((i 0)) body...). Detected by a name where the
        // core `let` expects its binding list. Lowers to `letrec` + a call, so
        // the loop variable is in scope inside the body exactly as in Scheme.
        if next_tok == "let"
            && *pos + 1 < tokens.len()
            && tokens[*pos + 1] != "("
            && tokens[*pos + 1] != ")"
        {
            *pos += 1;
            let name = tokens[*pos].clone();
            *pos += 1;
            let bindings = parse_bindings(tokens, pos, "named let")?;
            let body = body_expr(parse_body(tokens, pos, "named let")?, "named let")?;
            let (params, inits): (Vec<String>, Vec<Expr>) = bindings.into_iter().unzip();
            let lam = Expr::Lambda(params, Box::new(body));
            return Ok(Expr::LetRec(
                vec![(name.clone(), lam)],
                vec![Expr::Call(Box::new(Expr::Var(name)), inits)],
            ));
        }
        // unary operators: atom, not, bitlen
        if next_tok == "atom" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after atom".to_string());
            }
            *pos += 1;
            return Ok(Expr::UnOp(super::parser::UnOp::Atom, Box::new(inner)));
        }
        if next_tok == "not" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after not".to_string());
            }
            *pos += 1;
            return Ok(Expr::UnOp(super::parser::UnOp::Not, Box::new(inner)));
        }
        if next_tok == "null?" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after null?".to_string());
            }
            *pos += 1;
            return Ok(Expr::UnOp(super::parser::UnOp::IsNull, Box::new(inner)));
        }
        if next_tok == "bitlen" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after bitlen".to_string());
            }
            *pos += 1;
            return Ok(Expr::UnOp(super::parser::UnOp::BitLen, Box::new(inner)));
        }
        if next_tok == "+"
            || next_tok == "-"
            || next_tok == "*"
            || next_tok == "/"
            || next_tok == "=="
            || next_tok == "<"
            || next_tok == ">"
            || next_tok == "mod"
            || next_tok == "and"
            || next_tok == "or"
            || next_tok == "ash-left"
            || next_tok == "ash-right"
            || next_tok == "<="
            || next_tok == ">="
        {
            let op_tok = next_tok.clone();
            *pos += 1;
            let op = match op_tok.as_str() {
                "+" => BinOp::Add,
                "-" => BinOp::Sub,
                "*" => BinOp::Mul,
                "/" => BinOp::DivExact,
                "==" => BinOp::Eq,
                "<" => BinOp::Lt,
                ">" => BinOp::Gt,
                "<=" => BinOp::Le,
                ">=" => BinOp::Ge,
                "mod" => BinOp::Mod,
                "and" => BinOp::And,
                "or" => BinOp::Or,
                "ash-left" => BinOp::AshLeft,
                "ash-right" => BinOp::AshRight,
                _ => unreachable!(),
            };
            // Two operands take the core `BinOp` path, which is the only one
            // that dispatches MPC shares and FHE ciphertexts through the
            // backend. Any other arity becomes a call to the same operator's
            // *procedure* binding, which is variadic but public-values-only —
            // so `(+ 1 2 3)` and `(- 5)` work, and the fast path is untouched.
            let operands = parse_body(tokens, pos, &format!("`{op_tok}` expression"))?;
            if operands.len() == 2 {
                let mut it = operands.into_iter();
                let left = it.next().expect("2 operands");
                let right = it.next().expect("2 operands");
                Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
            } else {
                Ok(Expr::Call(Box::new(Expr::Var(op_tok)), operands))
            }
        } else if next_tok == "let" || next_tok == "letrec" {
            let is_letrec = next_tok == "letrec";
            *pos += 1;
            let mut bindings = Vec::new();
            if *pos < tokens.len()
                && tokens[*pos] == "("
                && *pos + 1 < tokens.len()
                && tokens[*pos + 1] == "("
            {
                *pos += 1;
                while *pos < tokens.len() && tokens[*pos] == "(" {
                    *pos += 1;
                    if *pos >= tokens.len() {
                        return Err("unexpected end in binding".to_string());
                    }
                    let name = tokens[*pos].clone();
                    *pos += 1;
                    // optional type annotation like :sint32 or :sint32-from-1
                    let mut wrap_with_share = None;
                    let mut share_owner_override: Option<i64> = None;
                    if *pos < tokens.len() && tokens[*pos].starts_with(":") {
                        let tkn = tokens[*pos].clone();
                        *pos += 1;
                        if tkn.starts_with(":sint") {
                            if let Some(rest) = tkn.strip_prefix(":sint") {
                                if let Some(dash) = rest.find("-from-") {
                                    if let Ok(nbits) = rest[..dash].parse::<i64>() {
                                        wrap_with_share = Some(nbits);
                                        share_owner_override = rest[dash + 6..].parse::<i64>().ok();
                                    }
                                } else if let Ok(nbits) = rest.parse::<i64>() {
                                    wrap_with_share = Some(nbits);
                                }
                            }
                        }
                    }
                    let mut expr = parse_tokens(tokens, pos)?;
                    if let Some(nbits) = wrap_with_share {
                        // transform into (share expr nbits) or (share-from owner expr nbits)
                        expr = if let Some(oid) = share_owner_override {
                            Expr::Call(
                                Box::new(Expr::Var("share-from".to_string())),
                                vec![Expr::Num(oid), expr, Expr::Num(nbits)],
                            )
                        } else {
                            Expr::Call(
                                Box::new(Expr::Var("share".to_string())),
                                vec![expr, Expr::Num(nbits)],
                            )
                        };
                    }
                    if *pos >= tokens.len() || tokens[*pos] != ")" {
                        return Err("expected ')' after binding".to_string());
                    }
                    *pos += 1;
                    bindings.push((name, expr));
                }
                if *pos >= tokens.len() || tokens[*pos] != ")" {
                    return Err("expected ')' after bindings list".to_string());
                }
                *pos += 1;
            } else {
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
                    *pos += 1;
                    if *pos >= tokens.len() {
                        return Err("unexpected end in binding".to_string());
                    }
                    let name = tokens[*pos].clone();
                    *pos += 1;
                    // optional type annotation
                    let mut wrap_with_share = None;
                    let mut share_owner_override: Option<i64> = None;
                    if *pos < tokens.len() && tokens[*pos].starts_with(":") {
                        let tkn = tokens[*pos].clone();
                        *pos += 1;
                        if let Some(rest) = tkn.strip_prefix(":sint") {
                            if let Some(dash) = rest.find("-from-") {
                                if let Ok(nbits) = rest[..dash].parse::<i64>() {
                                    wrap_with_share = Some(nbits);
                                    share_owner_override = rest[dash + 6..].parse::<i64>().ok();
                                }
                            } else if let Ok(nbits) = rest.parse::<i64>() {
                                wrap_with_share = Some(nbits);
                            }
                        }
                    }
                    let mut expr = parse_tokens(tokens, pos)?;
                    if let Some(nbits) = wrap_with_share {
                        expr = if let Some(oid) = share_owner_override {
                            Expr::Call(
                                Box::new(Expr::Var("share-from".to_string())),
                                vec![Expr::Num(oid), expr, Expr::Num(nbits)],
                            )
                        } else {
                            Expr::Call(
                                Box::new(Expr::Var("share".to_string())),
                                vec![expr, Expr::Num(nbits)],
                            )
                        };
                    }
                    if *pos >= tokens.len() || tokens[*pos] != ")" {
                        return Err("expected ')' after binding".to_string());
                    }
                    *pos += 1;
                    bindings.push((name, expr));
                }
            }
            let mut body = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != ")" {
                let e = parse_tokens(tokens, pos)?;
                body.push(e);
            }
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' to close let".to_string());
            }
            *pos += 1;
            if is_letrec {
                Ok(Expr::LetRec(bindings, body))
            } else {
                Ok(Expr::Let(bindings, body))
            }
        } else if next_tok == "if" {
            *pos += 1;
            let cond = parse_tokens(tokens, pos)?;
            let th = parse_tokens(tokens, pos)?;
            let el = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after if".to_string());
            }
            *pos += 1;
            Ok(Expr::If(Box::new(cond), Box::new(th), Box::new(el)))
        } else if next_tok == "car" {
            *pos += 1;
            let arg = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after car".to_string());
            }
            *pos += 1;
            Ok(Expr::Car(Box::new(arg)))
        } else if next_tok == "cdr" {
            *pos += 1;
            let arg = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after cdr".to_string());
            }
            *pos += 1;
            Ok(Expr::Cdr(Box::new(arg)))
        } else if next_tok == "cons" {
            *pos += 1;
            let a = parse_tokens(tokens, pos)?;
            let b = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after cons".to_string());
            }
            *pos += 1;
            Ok(Expr::Cons(Box::new(a), Box::new(b)))
        } else {
            let mut items = Vec::new();
            while *pos < tokens.len() && tokens[*pos] != ")" {
                let e = parse_tokens(tokens, pos)?;
                items.push(e);
            }
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' to close list".to_string());
            }
            *pos += 1;
            if !items.is_empty() {
                if let Expr::Var(ref s) = items[0] {
                    if s == "lambda" {
                        if items.len() < 3 {
                            return Err("lambda form must be (lambda (args...) body)".to_string());
                        }
                        let params = match &items[1] {
                            Expr::Seq(ps) => ps
                                .iter()
                                .map(|p| {
                                    if let Expr::Var(n) = p {
                                        n.clone()
                                    } else {
                                        panic!("lambda parameter must be identifier")
                                    }
                                })
                                .collect(),
                            Expr::Var(n) => vec![n.clone()],
                            // (lambda (f) ...) — zero or one arg wrapped in a Call
                            Expr::Call(head, v) if v.is_empty() => {
                                if let Expr::Var(n) = *head.clone() {
                                    vec![n.clone()]
                                } else {
                                    return Err(
                                        "lambda params must be a list or identifier".to_string()
                                    );
                                }
                            }
                            // (lambda (f x y ...) ...) — multiple params parsed as a Call
                            Expr::Call(head, args) => {
                                let mut params = Vec::new();
                                if let Expr::Var(n) = *head.clone() {
                                    params.push(n);
                                } else {
                                    return Err("lambda params must be identifiers".to_string());
                                }
                                for a in args {
                                    if let Expr::Var(n) = a {
                                        params.push(n.clone());
                                    } else {
                                        return Err("lambda params must be identifiers".to_string());
                                    }
                                }
                                params
                            }
                            _ => {
                                return Err("lambda params must be a list or identifier".to_string())
                            }
                        };
                        let body = if items.len() == 3 {
                            items.into_iter().nth(2).unwrap()
                        } else {
                            Expr::Seq(items.into_iter().skip(2).collect())
                        };
                        return Ok(Expr::Lambda(params, Box::new(body)));
                    }
                }
            }
            if !items.is_empty() {
                let head = Box::new(items.remove(0));
                return Ok(Expr::Call(head, items));
            }
            Ok(Expr::Seq(items))
        }
    } else if tok == ")" {
        Err("unexpected ')'".to_string())
    } else {
        // atom handling: numbers, strings, chars, booleans, identifiers
        *pos += 1;
        if tok.starts_with('"') && tok.ends_with('"') && tok.len() >= 2 {
            let inner = tok[1..tok.len() - 1].to_string();
            Ok(Expr::Str(inner))
        } else if tok.starts_with('\'') && tok.ends_with('\'') && tok.len() == 3 {
            let c = tok.chars().nth(1).unwrap();
            Ok(Expr::Char(c))
        } else if tok == "#t" {
            Ok(Expr::Bool(true))
        } else if tok == "#f" {
            Ok(Expr::Bool(false))
        } else if tok.contains('.') {
            // Decimal literal: parse as f64 then scale to Q16.16
            match tok.parse::<f64>() {
                Ok(f) => {
                    use crate::interpreter::ast::FIXED_SCALE;
                    let scaled = (f * FIXED_SCALE as f64).round() as i64;
                    Ok(Expr::Fixed(scaled))
                }
                Err(_) => Ok(Expr::Var(tok.clone())),
            }
        } else if let Ok(n) = tok.parse::<i64>() {
            Ok(Expr::Num(n))
        } else {
            Ok(Expr::Var(tok.clone()))
        }
    }
}

pub fn parse(src: &str) -> Result<Expr, String> {
    let tokens = tokenize(src);
    let mut pos = 0usize;
    let expr = parse_tokens(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err("trailing tokens after expression".to_string());
    }
    Ok(expr)
}
