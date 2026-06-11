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
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Not,
    Neg,
}

// Very small tokenizer + parser extracted from main.rs (keeps same behavior)
pub fn tokenize(src: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            '(' | ')' => { tokens.push(ch.to_string()); chars.next(); }
            '"' => {
                // parse string literal
                chars.next();
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '"' { break; }
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
                // char literal like 'a'
                // consume opening '\''
                chars.next();
                if let Some(&c) = chars.peek() {
                    // consume the char
                    let ch = c;
                    chars.next();
                    // if there's a closing quote, consume it
                    if let Some(&closing) = chars.peek() {
                        if closing == '\'' {
                            chars.next();
                        }
                    }
                    tokens.push(format!("'{}'", ch));
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
            c if c.is_whitespace() => { chars.next(); }
            '+' | '*' | '/' | '%' => { tokens.push(ch.to_string()); chars.next(); }
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
                            if d.is_ascii_digit() { num.push(d); chars.next(); } else { break; }
                        }
                        tokens.push(num);
                    } else {
                        tokens.push(String::from("-"));
                    }
                } else { tokens.push(String::from("-")); }
            }
            d if d.is_ascii_digit() => {
                let mut num = String::new();
                while let Some(&d2) = chars.peek() {
                    if d2.is_ascii_digit() { num.push(d2); chars.next(); } else { break; }
                }
                tokens.push(num);
            }
            _ => {
                let mut sym = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2.is_whitespace() || c2 == '(' || c2 == ')' { break; }
                    sym.push(c2); chars.next();
                }
                tokens.push(sym);
            }
        }
    }
    tokens
}

// We keep parse_tokens and parse in parser.rs for now, but they refer to Ast types in ast.rs
use crate::interpreter::ast::{Expr};

pub fn parse_tokens(tokens: &[String], pos: &mut usize) -> Result<Expr, String> {
    if *pos >= tokens.len() { return Err("unexpected end of tokens".to_string()); }
    let tok = &tokens[*pos];
    if tok == "(" {
        *pos += 1;
        if *pos >= tokens.len() { return Err("expected content after '('".to_string()) }
        let next_tok = &tokens[*pos];
        // unary operator `not`
        if next_tok == "not" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after not".to_string()); }
            *pos += 1;
            return Ok(Expr::UnOp(super::parser::UnOp::Not, Box::new(inner)));
        }
        if next_tok == "+" || next_tok == "-" || next_tok == "*" || next_tok == "/" || next_tok == "==" || next_tok == "<" || next_tok == ">" || next_tok == "<=" || next_tok == ">=" || next_tok == "mod" || next_tok == "and" || next_tok == "or" {
            let op_tok = next_tok.clone(); *pos += 1;
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
                _ => unreachable!(),
            };
            let left = parse_tokens(tokens, pos)?;
            let right = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after binary expression".to_string()); }
            *pos += 1;
            Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
        } else if next_tok == "let" || next_tok == "letrec" {
             let is_letrec = next_tok == "letrec"; *pos += 1;
             let mut bindings = Vec::new();
             if *pos < tokens.len() && tokens[*pos] == "(" && *pos + 1 < tokens.len() && tokens[*pos + 1] == "(" {
                 *pos += 1;
                 while *pos < tokens.len() && tokens[*pos] == "(" {
                     *pos += 1; if *pos >= tokens.len() { return Err("unexpected end in binding".to_string()); }
                     let name = tokens[*pos].clone(); *pos += 1;
                     // optional type annotation like :sint32
                     let mut wrap_with_share = None;
                     if *pos < tokens.len() && tokens[*pos].starts_with(":") {
                         let tkn = tokens[*pos].clone(); *pos += 1;
                         if tkn.starts_with(":sint") {
                             if let Ok(nbits) = tkn[5..].parse::<i64>() {
                                wrap_with_share = Some(nbits as i64);
                             }
                         }
                     }
                     let mut expr = parse_tokens(tokens, pos)?;
                     if let Some(nbits) = wrap_with_share {
                         // transform into (share expr nbits)
                         expr = Expr::Call(Box::new(Expr::Var("share".to_string())), vec![expr, Expr::Num(nbits)]);
                     }
                     if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after binding".to_string()); }
                     *pos += 1;
                     bindings.push((name, expr));
                 }
                 if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after bindings list".to_string()); }
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
                    *pos += 1; if *pos >= tokens.len() { return Err("unexpected end in binding".to_string()); }
                    let name = tokens[*pos].clone(); *pos += 1;
                    // optional type annotation
                    let mut wrap_with_share = None;
                    if *pos < tokens.len() && tokens[*pos].starts_with(":") {
                        let tkn = tokens[*pos].clone(); *pos += 1;
                        if tkn.starts_with(":sint") {
                            if let Ok(nbits) = tkn[5..].parse::<i64>() {
                                wrap_with_share = Some(nbits as i64);
                            }
                        }
                    }
                    let mut expr = parse_tokens(tokens, pos)?;
                    if let Some(nbits) = wrap_with_share {
                        expr = Expr::Call(Box::new(Expr::Var("share".to_string())), vec![expr, Expr::Num(nbits)]);
                    }
                     if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after binding".to_string()); }
                     *pos += 1;
                     bindings.push((name, expr));
                 }
             }
             let mut body = Vec::new();
             while *pos < tokens.len() && tokens[*pos] != ")" {
                let e = parse_tokens(tokens, pos)?; body.push(e);
             }
             if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' to close let".to_string()); }
             *pos += 1;
             if is_letrec { Ok(Expr::LetRec(bindings, body)) } else { Ok(Expr::Let(bindings, body)) }
        } else if next_tok == "if" {
            *pos += 1; let cond = parse_tokens(tokens, pos)?; let th = parse_tokens(tokens, pos)?; let el = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after if".to_string()); }
            *pos += 1; Ok(Expr::If(Box::new(cond), Box::new(th), Box::new(el)))
        } else if next_tok == "car" { *pos += 1; let arg = parse_tokens(tokens, pos)?; if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after car".to_string()); } *pos += 1; Ok(Expr::Car(Box::new(arg))) }
        else if next_tok == "cdr" { *pos += 1; let arg = parse_tokens(tokens, pos)?; if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after cdr".to_string()); } *pos += 1; Ok(Expr::Cdr(Box::new(arg))) }
        else if next_tok == "cons" { *pos += 1; let a = parse_tokens(tokens, pos)?; let b = parse_tokens(tokens, pos)?; if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' after cons".to_string()); } *pos += 1; Ok(Expr::Cons(Box::new(a), Box::new(b))) }
        else {
            let mut items = Vec::new(); while *pos < tokens.len() && tokens[*pos] != ")" { let e = parse_tokens(tokens, pos)?; items.push(e); }
            if *pos >= tokens.len() || tokens[*pos] != ")" { return Err("expected ')' to close list".to_string()); }
            *pos += 1;
            if !items.is_empty() {
                if let Expr::Var(ref s) = items[0] {
                    if s == "lambda" {
                        if items.len() < 3 { return Err("lambda form must be (lambda (args...) body)".to_string()); }
                        let params = match &items[1] { Expr::Seq(ps) => ps.iter().map(|p| { if let Expr::Var(n) = p { n.clone() } else { panic!("lambda parameter must be identifier") } }).collect(), _ => return Err("lambda args must be a list".to_string()) };
                        let body = if items.len() == 3 { items.into_iter().nth(2).unwrap() } else { Expr::Seq(items.into_iter().skip(2).collect()) };
                        return Ok(Expr::Lambda(params, Box::new(body)));
                    }
                }
            }
            if !items.is_empty() { let head = Box::new(items.remove(0)); return Ok(Expr::Call(head, items)); }
            Ok(Expr::Seq(items))
        }
    } else if tok == ")" { Err("unexpected ')'".to_string()) } else {
        // atom handling: numbers, strings, chars, booleans, identifiers
        *pos += 1;
        if tok.starts_with('"') && tok.ends_with('"') && tok.len() >= 2 {
            let inner = tok[1..tok.len()-1].to_string();
            Ok(Expr::Str(inner))
        } else if tok.starts_with('\'') && tok.ends_with('\'') && tok.len() == 3 {
            let c = tok.chars().nth(1).unwrap();
            Ok(Expr::Char(c))
        } else if tok == "#t" {
            Ok(Expr::Bool(true))
        } else if tok == "#f" {
            Ok(Expr::Bool(false))
        } else if let Ok(n) = tok.parse::<i64>() { Ok(Expr::Num(n)) } else { Ok(Expr::Var(tok.clone())) }
    }
}

pub fn parse(src: &str) -> Result<Expr, String> {
    let tokens = tokenize(src);
    let mut pos = 0usize;
    let expr = parse_tokens(&tokens, &mut pos)?;
    if pos != tokens.len() { return Err("trailing tokens after expression".to_string()); }
    Ok(expr)
}
