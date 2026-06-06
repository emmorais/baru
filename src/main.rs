use std::io::{self, Write};

use baru::interpreter::{parse, Participant, SharedValue};

fn main() {
    let participants: Vec<Participant> = (0..3).map(Participant::new).collect();

    println!("ABY3 REPL (replicated evaluator). Enter an s-expression using + - * / let letrec or 'quit' to exit.");

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        let n = stdin.read_line(&mut input).unwrap();
        if n == 0 {
            println!();
            break;
        }

        let src = input.trim();
        if src.is_empty() { continue; }
        if src == "quit" || src == "exit" { break; }

        match parse(src) {
            Ok(expr) => {
                println!("parsed: {:?}", expr);
                let mut results: Vec<SharedValue> = Vec::new();
                for p in &participants {
                    let r = p.eval(&expr);
                    results.push(r);
                }
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
