use std::path::PathBuf;
use std::env;
use std::error::Error;

use rustyline::error::ReadlineError;
use rustyline::completion::{Completer, Pair};
use rustyline::{Editor, Helper};
use rustyline::hint::Hinter;
use rustyline::highlight::Highlighter;
use rustyline::validate::Validator;

use baru::interpreter::{parse, Participant, SharedValue};

struct ReplHelper {
    words: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        let words = vec![
            ":verbose".to_string(),
            ":last".to_string(),
            ":results".to_string(),
            ":reveal".to_string(),
            ":evalfor".to_string(),
            ":inspect".to_string(),
            "quit".to_string(),
            "exit".to_string(),
            "share".to_string(),
            "reconstruct".to_string(),
            "a2b".to_string(),
            "b2a".to_string(),
        ];
        ReplHelper { words }
    }
}

impl Helper for ReplHelper {}
impl Hinter for ReplHelper {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}
impl Highlighter for ReplHelper {}
impl Validator for ReplHelper {}

impl Completer for ReplHelper {
    type Candidate = Pair;
    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let start = match line[..pos].rfind(|c: char| c.is_whitespace()) {
            Some(i) => i + 1,
            None => 0,
        };
        let word = &line[start..pos];
        let mut matches: Vec<Pair> = Vec::new();
        if word.is_empty() {
            for w in &self.words {
                matches.push(Pair {
                    display: w.clone(),
                    replacement: w.clone(),
                });
            }
        } else {
            for w in &self.words {
                if w.starts_with(word) {
                    matches.push(Pair {
                        display: w.clone(),
                        replacement: w.clone(),
                    });
                }
            }
        }
        Ok((start, matches))
    }
}

fn history_path() -> Option<PathBuf> {
    if let Ok(home) = env::var("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".baru_history");
        Some(p)
    } else {
        None
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let participants: Vec<Participant> = (0..3).map(Participant::new).collect();

    println!("ABY3 REPL (replicated evaluator). Enter an s-expression using + - * / let letrec or 'quit' to exit.");
    println!("Special commands:");
    println!("  :verbose [on|off]   - enable/disable verbose participant eval traces (default off)");
    println!("  :verbose            - toggle verbose");
    println!("  :last [pid]         - print last result for participant pid (or all if omitted)");
    println!("  :results            - print last results for all participants");
    println!("  :reveal <pid>       - reveal last result for participant pid (if it's a share or list of shares)");
    println!("  :evalfor <pid> EXPR - evaluate EXPR only for participant pid");
    println!("  :inspect <pid>      - evaluate '(reconstruct VAR)' for participant pid");

    let mut verbose = false;
    let mut last_results: Vec<Option<SharedValue>> = vec![None; participants.len()];

    let helper = ReplHelper::new();
    let mut rl = Editor::new()?;
    rl.set_helper(Some(helper));

    if let Some(p) = history_path() {
        if p.exists() {
            let _ = rl.load_history(&p);
        }
    }

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let src = line.trim();
                if src.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(src);

                if src == "quit" || src == "exit" {
                    break;
                }

                if src.starts_with(':') {
                    let mut parts = src.split_whitespace();
                    let cmd = parts.next().unwrap();
                    match cmd {
                        ":verbose" => {
                            match parts.next() {
                                Some(arg) => match arg {
                                    "on" => {
                                        verbose = true;
                                        println!("verbose: on");
                                    }
                                    "off" => {
                                        verbose = false;
                                        println!("verbose: off");
                                    }
                                    _ => println!("usage: :verbose [on|off]"),
                                },
                                None => {
                                    verbose = !verbose;
                                    println!("verbose: {}", if verbose { "on" } else { "off" });
                                }
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
                        ":inspect" => {
                            match parts.next() {
                                Some(pid_s) => match pid_s.parse::<usize>() {
                                    Ok(pid) if pid < participants.len() => {
                                        match parts.next() {
                                            Some(varname) => {
                                                // construct (reconstruct varname)
                                                let expr_src = format!("(reconstruct {} )", varname);
                                                match parse(&expr_src) {
                                                    Ok(expr) => {
                                                        let r = if verbose { participants[pid].eval_verbose(&expr, verbose) } else { participants[pid].eval(&expr) };
                                                        println!(":inspect[{}] {} -> {:?}", pid, varname, r);
                                                    }
                                                    Err(e) => println!("parse error: {}", e),
                                                }
                                            }
                                            None => println!("usage: :inspect <pid> <varname>"),
                                        }
                                    }
                                    _ => println!("invalid participant id"),
                                },
                                None => println!("usage: :inspect <pid> <varname>"),
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
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(err) => {
                eprintln!("Error reading line: {:?}", err);
                break;
            }
        }
    }

    if let Some(p) = history_path() {
        let _ = rl.save_history(&p);
    }

    Ok(())
}
