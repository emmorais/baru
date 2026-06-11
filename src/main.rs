use std::io::{self, Write};

use baru::interpreter::{parse, Participant, SharedValue};

fn main() {
    let participants: Vec<Participant> = (0..3).map(Participant::new).collect();

    println!("ABY3 REPL (replicated evaluator). Enter an s-expression using + - * / let letrec or 'quit' to exit.");
    println!("Special commands:");
    println!("  :verbose [on|off]   - enable/disable verbose participant eval traces (default off)");
    println!("  :verbose            - toggle verbose");
    println!("  :last [pid]         - print last result for participant pid (or all if omitted)");
    println!("  :results            - print last results for all participants");
    println!("  :reveal <pid>       - reveal last result for participant pid (if it's a share or list of shares)");
    println!("  :evalfor <pid> EXPR - evaluate EXPR only for participant pid");

    let stdin = io::stdin();
    let mut verbose = false;

    // store last evaluation result per participant for simple inspection
    let mut last_results: Vec<Option<SharedValue>> = vec![None; participants.len()];

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

        // handle special REPL commands starting with ':'
        if src.starts_with(':') {
            let mut parts = src.split_whitespace();
            let cmd = parts.next().unwrap();
            match cmd {
                ":verbose" => {
                    match parts.next() {
                        Some(arg) => match arg {
                            "on" => { verbose = true; println!("verbose: on"); }
                            "off" => { verbose = false; println!("verbose: off"); }
                            _ => println!("usage: :verbose [on|off]"),
                        },
                        None => { verbose = !verbose; println!("verbose: {}", if verbose { "on" } else { "off" }); }
                    }
                }
                ":last" => {
                    match parts.next() {
                        Some(pid_s) => match pid_s.parse::<usize>() {
                            Ok(pid) if pid < participants.len() => {
                                match &last_results[pid] {
                                    Some(v) => println!("last[{}] = {:?}", pid, v),
                                    None => println!("last[{}] = <none>", pid),
                                }
                            }
                            _ => println!("invalid participant id"),
                        },
                        None => {
                            for (i, r) in last_results.iter().enumerate() {
                                match r {
                                    Some(v) => println!("last[{}] = {:?}", i, v),
                                    None => println!("last[{}] = <none>", i),
                                }
                            }
                        }
                    }
                }
                ":results" => {
                    for (i, r) in last_results.iter().enumerate() {
                        match r {
                            Some(v) => println!("participant[{}] -> {:?}", i, v),
                            None => println!("participant[{}] -> <none>", i),
                        }
                    }
                }
                ":reveal" => {
                    match parts.next() {
                        Some(pid_s) => match pid_s.parse::<usize>() {
                            Ok(pid) if pid < participants.len() => {
                                match &last_results[pid] {
                                    Some(v) => match v {
                                        SharedValue::Share(sr) => {
                                            let val = participants[pid].reveal(sr);
                                            println!("reveal[{}] -> {}", pid, val);
                                        }
                                        SharedValue::List(list) => {
                                            // try to reveal each share in the list
                                            let mut revealed: Vec<String> = Vec::new();
                                            for item in list {
                                                match item {
                                                    SharedValue::Share(sr) => {
                                                        let val = participants[pid].reveal(sr);
                                                        revealed.push(format!("{}", val));
                                                    }
                                                    SharedValue::Num(n) => revealed.push(format!("Num({})", n)),
                                                    other => revealed.push(format!("{:?}", other)),
                                                }
                                            }
                                            println!("reveal[{}] -> {:?}", pid, revealed);
                                        }
                                        SharedValue::Num(n) => println!("reveal[{}] -> Num({})", pid, n),
                                        other => println!("reveal[{}] -> {:?}", pid, other),
                                    },
                                    None => println!("no last result for participant {}", pid),
                                }
                            }
                            _ => println!("invalid participant id"),
                        },
                        None => println!("usage: :reveal <participant-id>"),
                    }
                }
                ":evalfor" => {
                    match parts.next() {
                        Some(pid_s) => match pid_s.parse::<usize>() {
                            Ok(pid) if pid < participants.len() => {
                                // rest of the input is the expression to evaluate
                                let rest = parts.collect::<Vec<&str>>().join(" ");
                                if rest.is_empty() {
                                    println!("usage: :evalfor <pid> EXPR");
                                } else {
                                    match parse(&rest) {
                                        Ok(expr) => {
                                            let r = if verbose { participants[pid].eval_verbose(&expr, verbose) } else { participants[pid].eval(&expr) };
                                            last_results[pid] = Some(r.clone());
                                            println!("evalfor[{}] -> {:?}", pid, r);
                                        }
                                        Err(e) => println!("parse error: {}", e),
                                    }
                                }
                            }
                            _ => println!("invalid participant id"),
                        },
                        None => println!("usage: :evalfor <pid> EXPR"),
                    }
                }
                _ => println!("unknown command: {}", cmd),
            }
            continue;
        }

        match parse(src) {
            Ok(expr) => {
                println!("parsed: {:?}", expr);
                let mut results: Vec<SharedValue> = Vec::new();
                for (i, p) in participants.iter().enumerate() {
                    let r = if verbose { p.eval_verbose(&expr, verbose) } else { p.eval(&expr) };
                    last_results[i] = Some(r.clone());
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
