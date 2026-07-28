use std::env;
use std::error::Error;
use std::path::PathBuf;

use rustyline::error::ReadlineError;
use rustyline::Editor;

use baru::interpreter::{parse, Participant, Types};
use baru::repl::{print_examples_for, print_examples_list, print_help, ReplHelper, EXAMPLES};
use colored::*;

// ─────────────────────────────────────────────────────────────────────────────

fn history_path() -> Option<PathBuf> {
    if let Ok(home) = env::var("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".baru_history");
        Some(p)
    } else {
        None
    }
}

#[allow(clippy::format_in_format_args)]
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("bench") {
        return baru::bench::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-nn") {
        return baru::bench::nn::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("quantize") {
        return baru::ml::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-ckks-compare") {
        return baru::bench::ckks_compare::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-hybrid-nn") {
        return baru::bench::hybrid_nn::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("demo-private-mnist") {
        return baru::demo::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-deep-nn") {
        return baru::bench::deep_nn::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-cnn") {
        return baru::bench::cnn::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-mnist") {
        return baru::bench::mnist::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-medmnist") {
        return baru::bench::medmnist::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-gc-compare") {
        return baru::bench::gc_compare::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-fhe-innerprod") {
        return baru::bench::fhe_innerprod::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-train") {
        return baru::bench::train::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("bench-cohort") {
        return baru::bench::cohort::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("mpcc") {
        return baru::mpcc::run_cli(&args[1..]).map_err(|e| e.into());
    }
    if args.first().map(String::as_str) == Some("db") {
        return baru::db::run_cli(&args[1..]).map_err(|e| e.into());
    }

    let participants: Vec<Participant> = (0..3).map(Participant::new).collect();

    println!(
        "{}",
        "Baru REPL — Secure Multi-Party Computation (replicated 3-party evaluation)."
            .bright_green()
    );
    println!("{}", "Type ':help' for language reference, ':examples' to browse built-in examples, or 'quit' to exit.".bright_yellow());

    let mut verbose = false;
    let mut last_results: Vec<Option<Types>> = vec![None; participants.len()];

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
                        ":help" => {
                            print_help(false);
                        }
                        ":examples" => match parts.next() {
                            None => print_examples_list(),
                            Some(cat) => print_examples_for(cat),
                        },
                        ":example" => {
                            match parts.next() {
                                None => println!("usage: :example <n>  (use ':examples' to list)"),
                                Some(n_s) => match n_s.parse::<usize>() {
                                    Ok(n) if n >= 1 && n <= EXAMPLES.len() => {
                                        let ex = &EXAMPLES[n - 1];
                                        println!();
                                        println!(
                                            "{} [{}] {} ({})",
                                            "Running example".bright_yellow(),
                                            n.to_string().bright_white(),
                                            ex.name.bright_cyan(),
                                            ex.category
                                        );
                                        println!(
                                            "{} {}",
                                            "Code:".bright_yellow(),
                                            ex.code.bright_green()
                                        );
                                        println!(
                                            "{} {}",
                                            "Expected:".bright_yellow(),
                                            ex.expected.bright_magenta()
                                        );
                                        println!();
                                        match parse(ex.code) {
                                            Ok(expr) => {
                                                let mut results: Vec<Types> = Vec::new();
                                                for (i, p) in participants.iter().enumerate() {
                                                    let r = p.eval(&expr);
                                                    last_results[i] = Some(r.clone());
                                                    results.push(r);
                                                }
                                                match &results[0] {
                                                    Types::Share(sr) => {
                                                        let raw = participants[0].reveal(sr);
                                                        let val = (raw as u32) as i32 as i64;
                                                        println!(
                                                            "{} {} (revealed, signed)",
                                                            "result:".bright_cyan(),
                                                            val.to_string().bright_green()
                                                        );
                                                    }
                                                    Types::Num(n) => {
                                                        // sign-extend through 32 bits so MPC field values display correctly
                                                        let val = (*n as u32) as i32 as i64;
                                                        println!(
                                                            "{} {}",
                                                            "result:".bright_cyan(),
                                                            val.to_string().bright_green()
                                                        );
                                                    }
                                                    other => println!(
                                                        "{} {:?}",
                                                        "result:".bright_cyan(),
                                                        other
                                                    ),
                                                }
                                                println!("all participants: {:?}\n", results);
                                            }
                                            Err(e) => println!("parse error: {}", e),
                                        }
                                    }
                                    _ => println!("invalid example number (1–{})", EXAMPLES.len()),
                                },
                            }
                        }
                        ":verbose" => match parts.next() {
                            Some(arg) => match arg {
                                "on" => {
                                    verbose = true;
                                    println!("{}", "verbose: on".bright_green());
                                }
                                "off" => {
                                    verbose = false;
                                    println!("{}", "verbose: off".bright_yellow());
                                }
                                _ => println!("usage: :verbose [on|off]"),
                            },
                            None => {
                                verbose = !verbose;
                                println!("verbose: {}", if verbose { "on" } else { "off" });
                            }
                        },
                        ":last" => match parts.next() {
                            Some(pid_s) => match pid_s.parse::<usize>() {
                                Ok(pid) if pid < participants.len() => match &last_results[pid] {
                                    Some(v) => println!(
                                        "{} {}",
                                        format!("last[{}] =", pid).bright_magenta(),
                                        format!("{:?}", v)
                                    ),
                                    None => println!(
                                        "{}",
                                        format!("last[{}] = <none>", pid).bright_red()
                                    ),
                                },
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
                        },
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
                                                Types::Share(sr) => {
                                                    let val = participants[pid].reveal(sr);
                                                    println!(
                                                        "{} {}",
                                                        format!("reveal[{}] ->", pid)
                                                            .bright_magenta(),
                                                        format!("{}", val).bright_green()
                                                    );
                                                }
                                                Types::List(list) => {
                                                    // try to reveal each share in the list
                                                    let mut revealed: Vec<String> = Vec::new();
                                                    for item in list {
                                                        match item {
                                                            Types::Share(sr) => {
                                                                let val =
                                                                    participants[pid].reveal(sr);
                                                                revealed.push(format!("{}", val));
                                                            }
                                                            Types::Num(n) => {
                                                                revealed.push(format!("Num({})", n))
                                                            }
                                                            other => revealed
                                                                .push(format!("{:?}", other)),
                                                        }
                                                    }
                                                    println!("reveal[{}] -> {:?}", pid, revealed);
                                                }
                                                Types::Num(n) => {
                                                    println!("reveal[{}] -> Num({})", pid, n)
                                                }
                                                other => println!("reveal[{}] -> {:?}", pid, other),
                                            },
                                            None => {
                                                println!("no last result for participant {}", pid)
                                            }
                                        }
                                    }
                                    _ => println!("invalid participant id"),
                                },
                                None => println!("usage: :reveal <participant-id>"),
                            }
                        }
                        ":evalfor" => match parts.next() {
                            Some(pid_s) => match pid_s.parse::<usize>() {
                                Ok(pid) if pid < participants.len() => {
                                    let rest = parts.collect::<Vec<&str>>().join(" ");
                                    if rest.is_empty() {
                                        println!("usage: :evalfor <pid> EXPR");
                                    } else {
                                        match parse(&rest) {
                                            Ok(expr) => {
                                                let r = if verbose {
                                                    participants[pid].eval_verbose(&expr, verbose)
                                                } else {
                                                    participants[pid].eval(&expr)
                                                };
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
                        },
                        ":inspect" => {
                            match parts.next() {
                                Some(pid_s) => match pid_s.parse::<usize>() {
                                    Ok(pid) if pid < participants.len() => {
                                        match parts.next() {
                                            Some(varname) => {
                                                // construct (reconstruct varname)
                                                let expr_src =
                                                    format!("(reconstruct {} )", varname);
                                                match parse(&expr_src) {
                                                    Ok(expr) => {
                                                        let r = if verbose {
                                                            participants[pid]
                                                                .eval_verbose(&expr, verbose)
                                                        } else {
                                                            participants[pid].eval(&expr)
                                                        };
                                                        println!(
                                                            ":inspect[{}] {} -> {:?}",
                                                            pid, varname, r
                                                        );
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
                        ":save" => match parts.next() {
                            Some(pid_s) => match pid_s.parse::<usize>() {
                                Ok(pid) if pid < participants.len() => {
                                    let filename = parts
                                        .next()
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| format!("store_{}.json", pid));
                                    let path = std::path::PathBuf::from(filename);
                                    match participants[pid].save_store(&path) {
                                        Ok(_) => println!(
                                            "saved store for participant {} -> {:?}",
                                            pid, path
                                        ),
                                        Err(e) => println!("error saving store: {}", e),
                                    }
                                }
                                _ => println!("invalid participant id"),
                            },
                            None => println!("usage: :save <pid> [filename]"),
                        },
                        ":load" => match parts.next() {
                            Some(pid_s) => match pid_s.parse::<usize>() {
                                Ok(pid) if pid < participants.len() => {
                                    let filename = parts
                                        .next()
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| format!("store_{}.json", pid));
                                    let path = std::path::PathBuf::from(filename);
                                    match participants[pid].load_store(&path) {
                                        Ok(_) => println!(
                                            "loaded store for participant {} <- {:?}",
                                            pid, path
                                        ),
                                        Err(e) => println!("error loading store: {}", e),
                                    }
                                }
                                _ => println!("invalid participant id"),
                            },
                            None => println!("usage: :load <pid> [filename]"),
                        },
                        ":reset" => match parts.next() {
                            Some(pid_s) => match pid_s.parse::<usize>() {
                                Ok(pid) if pid < participants.len() => {
                                    participants[pid].reset_store();
                                    println!("reset store for participant {}", pid);
                                }
                                _ => println!("invalid participant id"),
                            },
                            None => {
                                for p in &participants {
                                    p.reset_store();
                                }
                                println!("reset stores for all participants");
                            }
                        },
                        ":env" => match parts.next() {
                            Some(pid_s) => match pid_s.parse::<usize>() {
                                Ok(pid) if pid < participants.len() => {
                                    participants[pid].print_env();
                                }
                                _ => println!("invalid participant id"),
                            },
                            None => println!("usage: :env <participant-id>"),
                        },
                        _ => println!(
                            "unknown command: {}  (type ':help' for command reference)",
                            cmd
                        ),
                    }
                    continue;
                }

                match parse(src) {
                    Ok(expr) => {
                        println!("parsed: {:?}", expr);
                        let mut results: Vec<Types> = Vec::new();
                        for (i, p) in participants.iter().enumerate() {
                            let r = if verbose {
                                p.eval_verbose(&expr, verbose)
                            } else {
                                p.eval(&expr)
                            };
                            last_results[i] = Some(r.clone());
                            results.push(r);
                        }
                        match &results[0] {
                            Types::Num(n) => println!("result (participant 0): {}", n),
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
