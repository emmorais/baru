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
use colored::*;

// ── Examples catalog ─────────────────────────────────────────────────────────

struct Example {
    category: &'static str,
    name:     &'static str,
    code:     &'static str,
    expected: &'static str,
}

#[rustfmt::skip]
const EXAMPLES: &[Example] = &[
    // ── Arithmetic ───────────────────────────────────────────────────────────
    Example { category: "arithmetic", name: "add",          code: "(+ 10 32)",                          expected: "42"      },
    Example { category: "arithmetic", name: "multiply",     code: "(* 6 7)",                             expected: "42"      },
    Example { category: "arithmetic", name: "subtract",     code: "(- 100 58)",                          expected: "42"      },
    Example { category: "arithmetic", name: "divide",       code: "(/ 84 7)",                            expected: "12"      },
    Example { category: "arithmetic", name: "modulo",       code: "(mod 10 3)",                          expected: "1"       },
    Example { category: "arithmetic", name: "nested",       code: "(+ (* 6 7) (* 3 14))",                expected: "84"      },

    // ── Comparisons ──────────────────────────────────────────────────────────
    Example { category: "comparisons", name: "less-than",      code: "(< 3 4)",          expected: "1" },
    Example { category: "comparisons", name: "greater-than",   code: "(> 5 2)",          expected: "1" },
    Example { category: "comparisons", name: "equal",          code: "(== 7 7)",         expected: "1" },
    Example { category: "comparisons", name: "less-or-equal",  code: "(<= 3 3)",         expected: "1" },
    Example { category: "comparisons", name: "neg-lt-zero",    code: "(< -5 0)",         expected: "1" },

    // ── Booleans / Bitwise ───────────────────────────────────────────────────
    Example { category: "booleans", name: "not-true",    code: "(not #t)",    expected: "0" },
    Example { category: "booleans", name: "not-false",   code: "(not #f)",    expected: "1" },
    Example { category: "booleans", name: "and",         code: "(and #t #t)", expected: "1" },
    Example { category: "booleans", name: "or",          code: "(or #t #f)",  expected: "1" },
    Example { category: "booleans", name: "bitwise-and", code: "(and 6 3)",   expected: "2" },
    Example { category: "booleans", name: "bitwise-or",  code: "(or 6 3)",    expected: "7" },

    // ── Let / Lambda / Closures ───────────────────────────────────────────────
    Example { category: "let-lambda", name: "let-binding",
              code: "(let ((a 6) (b 7)) (+ a b))",               expected: "13" },
    Example { category: "let-lambda", name: "nested-shadow",
              code: "(let ((x 1)) (let ((x 2)) x))",             expected: "2"  },
    Example { category: "let-lambda", name: "lambda-call",
              code: "((lambda (x) (+ x 1)) 5)",                  expected: "6"  },
    Example { category: "let-lambda", name: "closure-capture",
              code: "(let ((a 10)) (let ((f (lambda (x) (+ x a)))) (f 5)))", expected: "15" },
    Example { category: "let-lambda", name: "currying",
              code: "(let ((add-n (lambda (n) (lambda (x) (+ x n))))) ((add-n 5) 37))", expected: "42" },

    // ── Higher-Order Functions ────────────────────────────────────────────────
    Example { category: "higher-order", name: "apply",
              code: "(let ((apply (lambda (f x) (f x)))) (apply (lambda (n) (* n n)) 7))", expected: "49" },
    Example { category: "higher-order", name: "compose",
              code: "(let ((compose (lambda (f g) (lambda (x) (f (g x)))))) (let ((double (lambda (x) (* x 2))) (inc (lambda (x) (+ x 1)))) ((compose double inc) 3)))", expected: "8" },
    Example { category: "higher-order", name: "mutual-recursion-even",
              code: "(letrec ((even (lambda (n) (if (== n 0) 1 (odd (- n 1))))) (odd (lambda (n) (if (== n 0) 0 (even (- n 1)))))) (even 10))", expected: "1" },

    // ── Lists (cons / car / cdr) ─────────────────────────────────────────────
    Example { category: "lists", name: "car",
              code: "(car (cons 1 (cons 2 ())))",                      expected: "1" },
    Example { category: "lists", name: "cdr-car",
              code: "(car (cdr (cons 1 (cons 2 ()))))",                expected: "2" },
    Example { category: "lists", name: "recursive-sum-1-10",
              code: "(letrec ((sum (lambda (n acc) (if (== n 0) acc (sum (- n 1) (+ acc n)))))) (sum 10 0))", expected: "55" },

    // ── Classic CS: Fibonacci ─────────────────────────────────────────────────
    Example { category: "recursion", name: "fibonacci-10",
              code: "(letrec ((fib (lambda (n) (if (== n 0) 0 (if (== n 1) 1 (+ (fib (- n 1)) (fib (- n 2)))))))) (fib 10))", expected: "55" },
    Example { category: "recursion", name: "fibonacci-20",
              code: "(letrec ((fib (lambda (n) (if (== n 0) 0 (if (== n 1) 1 (+ (fib (- n 1)) (fib (- n 2)))))))) (fib 20))", expected: "6765" },

    // ── Classic CS: Factorial ─────────────────────────────────────────────────
    Example { category: "recursion", name: "factorial-10",
              code: "(letrec ((fact (lambda (n) (if (== n 0) 1 (* n (fact (- n 1))))))) (fact 10))", expected: "3628800" },

    // ── Classic CS: Fast Exponentiation ──────────────────────────────────────
    Example { category: "recursion", name: "fast-pow-2^10",
              code: "(letrec ((pow (lambda (b e) (if (== e 0) 1 (if (== (mod e 2) 0) (let ((h (pow b (/ e 2)))) (* h h)) (* b (pow b (- e 1)))))))) (pow 2 10))", expected: "1024" },

    // ── Classic CS: GCD (Euclidean) ───────────────────────────────────────────
    Example { category: "recursion", name: "gcd-48-18",
              code: "(letrec ((gcd (lambda (a b) (if (== b 0) a (gcd b (mod a b)))))) (gcd 48 18))", expected: "6" },
    Example { category: "recursion", name: "gcd-coprime-17-5",
              code: "(letrec ((gcd (lambda (a b) (if (== b 0) a (gcd b (mod a b)))))) (gcd 17 5))", expected: "1" },

    // ── Classic CS: Integer Square Root ──────────────────────────────────────
    Example { category: "recursion", name: "isqrt-100",
              code: "(letrec ((isqrt (lambda (x) (letrec ((go (lambda (n) (if (> (* (+ n 1) (+ n 1)) x) n (go (+ n 1)))))) (go 0))))) (isqrt 100))", expected: "10" },
    Example { category: "recursion", name: "isqrt-50",
              code: "(letrec ((isqrt (lambda (x) (letrec ((go (lambda (n) (if (> (* (+ n 1) (+ n 1)) x) n (go (+ n 1)))))) (go 0))))) (isqrt 50))", expected: "7" },

    // ── Classic CS: Collatz ───────────────────────────────────────────────────
    Example { category: "recursion", name: "collatz-27",
              code: "(letrec ((c (lambda (n s) (if (== n 1) s (if (== (mod n 2) 0) (c (/ n 2) (+ s 1)) (c (+ (* 3 n) 1) (+ s 1))))))) (c 27 0))", expected: "111" },

    // ── Classic CS: Tail-Recursive Sum ───────────────────────────────────────
    Example { category: "recursion", name: "tail-sum-1-100",
              code: "(letrec ((sum (lambda (n acc) (if (== n 0) acc (sum (- n 1) (+ acc n)))))) (sum 100 0))", expected: "5050" },

    // ── Classic CS: Ackermann ─────────────────────────────────────────────────
    Example { category: "recursion", name: "ackermann-3-3",
              code: "(letrec ((ack (lambda (m n) (if (== m 0) (+ n 1) (if (== n 0) (ack (- m 1) 1) (ack (- m 1) (ack m (- n 1)))))))) (ack 3 3))", expected: "61" },

    // ── Church Numerals ───────────────────────────────────────────────────────
    Example { category: "church", name: "church-add-2+3",
              code: "(let ((two (lambda (f) (lambda (x) (f (f x))))) (three (lambda (f) (lambda (x) (f (f (f x)))))) (church+ (lambda (m) (lambda (n) (lambda (f) (lambda (x) ((m f) ((n f) x))))))) (to-int (lambda (n) ((n (lambda (x) (+ x 1))) 0)))) (to-int ((church+ two) three)))",
              expected: "5" },
    Example { category: "church", name: "church-mul-2*3",
              code: "(let ((two (lambda (f) (lambda (x) (f (f x))))) (three (lambda (f) (lambda (x) (f (f (f x)))))) (church* (lambda (m) (lambda (n) (lambda (f) (m (n f)))))) (to-int (lambda (n) ((n (lambda (x) (+ x 1))) 0)))) (to-int ((church* two) three)))",
              expected: "6" },

    // ── MPC Basics ───────────────────────────────────────────────────────────
    Example { category: "mpc-basics", name: "share-add",
              code: "(reconstruct (+ (share 10 32) (share 32 32)))",    expected: "42" },
    Example { category: "mpc-basics", name: "share-mul",
              code: "(reconstruct (* (share 6 32) (share 7 32)))",      expected: "42" },
    Example { category: "mpc-basics", name: "share-sub",
              code: "(reconstruct (- (share 100 32) (share 58 32)))",   expected: "42" },
    Example { category: "mpc-basics", name: "scalar-add",
              code: "(reconstruct (+ (share 40 32) 2))",                expected: "42" },
    Example { category: "mpc-basics", name: "scalar-mul",
              code: "(reconstruct (* (share 6 32) 7))",                 expected: "42" },

    // ── MPC Signed Values ────────────────────────────────────────────────────
    Example { category: "mpc-signed", name: "negative-input",
              code: "(reconstruct (share -42 32))",                           expected: "-42" },
    Example { category: "mpc-signed", name: "signed-add",
              code: "(reconstruct (+ (share -10 32) (share 52 32)))",         expected: "42"  },
    Example { category: "mpc-signed", name: "signed-mul-neg-neg",
              code: "(reconstruct (* (share -6 32) (share -7 32)))",          expected: "42"  },
    Example { category: "mpc-signed", name: "signed-sub-negative-result",
              code: "(reconstruct (- (share 3 32) (share 10 32)))",           expected: "-7"  },

    // ── MPC Comparisons / Oblivious Select ───────────────────────────────────
    Example { category: "mpc-compare", name: "oblivious-max",
              code: "(let ((a (share 30 32)) (b (share 20 32)) (one (share 1 32))) (let ((m (> a b))) (+ (* a m) (* b (- one m)))))",
              expected: "30" },
    Example { category: "mpc-compare", name: "oblivious-min",
              code: "(let ((a (share 30 32)) (b (share 20 32)) (one (share 1 32))) (let ((m (< a b))) (+ (* a m) (* b (- one m)))))",
              expected: "20" },
    Example { category: "mpc-compare", name: "equality-select",
              code: "(let ((a (share 42 32)) (b (share 42 32)) (one (share 1 32))) (let ((m (== a b))) (+ (* a m) (* b (- one m)))))",
              expected: "42" },

    // ── ReLU & Neural Networks ───────────────────────────────────────────────
    Example { category: "relu-network", name: "relu-positive",
              code: "(let ((z (share 5 32))) (if (> z 0) z 0))",
              expected: "5" },
    Example { category: "relu-network", name: "relu-negative",
              code: "(let ((z (share -4 32))) (if (> z 0) z 0))",
              expected: "0" },
    Example { category: "relu-network", name: "relu-zero",
              code: "(let ((z (share 0 32))) (if (> z 0) z 0))",
              expected: "0" },
    Example { category: "relu-network", name: "single-neuron-relu",
              code: "(let ((w1 (share 2 32)) (w2 (share -3 32)) (x (share 4 32))) (let ((z (+ (* w1 x) (* w2 x)))) (if (> z 0) z 0)))",
              expected: "0  (z = 2*4 + -3*4 = -4, ReLU -> 0)" },
    Example { category: "relu-network", name: "two-neuron-layer",
              code: "(let ((x (share 4 32)) (w11 (share 2 32)) (w12 (share -3 32)) (w21 (share 1 32)) (w22 (share 2 32))) (let ((z1 (+ (* w11 x) (* w12 x))) (z2 (+ (* w21 x) (* w22 x)))) (let ((n1 (if (> z1 0) z1 0)) (n2 (if (> z2 0) z2 0))) (+ n1 n2))))",
              expected: "12  (n1=ReLU(-4)=0, n2=ReLU(12)=12)" },
    Example { category: "relu-network", name: "single-neuron-sint-sugar",
              code: "(let ((w1 :sint32 5) (w2 :sint32 -2) (x :sint32 3)) (let ((z (+ (* w1 x) (* w2 x)))) (if (> z 0) z 0)))",
              expected: "9  (z = 5*3 + -2*3 = 9, ReLU -> 9)" },

    // ── Oblivious Algorithms ─────────────────────────────────────────────────
    Example { category: "mpc-oblivious", name: "abs-positive",
              code: "(let ((x (share 7 32))) (if (>= x 0) x (- (share 0 32) x)))",
              expected: "7" },
    Example { category: "mpc-oblivious", name: "abs-negative",
              code: "(let ((x (share -7 32))) (if (>= x 0) x (- (share 0 32) x)))",
              expected: "7" },
    Example { category: "mpc-oblivious", name: "clamp-below-range",
              code: "(let ((x (share -5 32)) (lo (share 0 32)) (hi (share 10 32)) (one (share 1 32))) (let ((ch (let ((m (< x hi))) (+ (* x m) (* hi (- one m)))))) (let ((m2 (> ch lo))) (+ (* ch m2) (* lo (- one m2))))))",
              expected: "0" },
    Example { category: "mpc-oblivious", name: "clamp-in-range",
              code: "(let ((x (share 5 32)) (lo (share 0 32)) (hi (share 10 32)) (one (share 1 32))) (let ((ch (let ((m (< x hi))) (+ (* x m) (* hi (- one m)))))) (let ((m2 (> ch lo))) (+ (* ch m2) (* lo (- one m2))))))",
              expected: "5" },
    Example { category: "mpc-oblivious", name: "oblivious-min-of-pair",
              code: "(let ((a (share 7 32)) (b (share 3 32)) (one (share 1 32))) (let ((m (< a b))) (+ (* a m) (* b (- one m)))))",
              expected: "3" },
];

fn distinct_categories() -> Vec<&'static str> {
    let mut seen = std::collections::HashSet::new();
    let mut cats: Vec<&'static str> = Vec::new();
    for ex in EXAMPLES {
        if seen.insert(ex.category) {
            cats.push(ex.category);
        }
    }
    cats
}

fn print_help() {
    println!();
    println!("{}", "╔══════════════════════════════════════════════════════════════╗".bright_cyan());
    println!("{}", "║      baru — a Lisp-like Secure Multi-Party Computation REPL  ║".bright_cyan());
    println!("{}", "╚══════════════════════════════════════════════════════════════╝".bright_cyan());
    println!();

    println!("{}", "LITERALS".bright_yellow());
    println!("  {}   integers (positive and negative)", "42   -7   0".bright_green());
    println!("  {}      boolean true / false", "#t   #f".bright_green());
    println!("  {}  empty list / nil", "()".bright_green());
    println!();

    println!("{}", "ARITHMETIC".bright_yellow());
    println!("  {}   addition / subtraction", "(+ a b)  (- a b)".bright_green());
    println!("  {}   multiplication / integer division", "(* a b)  (/ a b)".bright_green());
    println!("  {}           remainder (modulo)", "(mod a b)".bright_green());
    println!();

    println!("{}", "COMPARISONS  — return 1 (true) or 0 (false)".bright_yellow());
    println!("  {}",
        "(== a b)  (< a b)  (<= a b)  (> a b)  (>= a b)".bright_green());
    println!();

    println!("{}", "BOOLEAN / BITWISE".bright_yellow());
    println!("  {}              logical / bitwise not", "(not x)".bright_green());
    println!("  {}        logical / bitwise and, or", "(and a b)  (or a b)".bright_green());
    println!();

    println!("{}", "CONDITIONALS".bright_yellow());
    println!("  {}  evaluates then or else branch", "(if cond then else)".bright_green());
    println!();

    println!("{}", "LET BINDINGS".bright_yellow());
    println!("  {}  parallel bindings, then body",
        "(let ((x 1) (y 2)) body)".bright_green());
    println!("  {}  auto-shares the value as an N-bit signed integer",
        "(let ((z :sint32 -5)) body)".bright_green());
    println!();

    println!("{}", "LAMBDAS — first-class functions".bright_yellow());
    println!("  {}      single parameter", "(lambda (x) body)".bright_green());
    println!("  {}   multiple parameters", "(lambda (x y) body)".bright_green());
    println!("  {}    direct application", "((lambda (x) (* x x)) 5)  =>  25".bright_green());
    println!();

    println!("{}", "RECURSIVE BINDINGS".bright_yellow());
    println!("  {}", "(letrec ((fact (lambda (n)".bright_green());
    println!("  {}",               "           (if (== n 0) 1 (* n (fact (- n 1)))))))".bright_green());
    println!("  {}",     "(fact 10))".bright_green());
    println!("  Mutually recursive bindings are supported.");
    println!();

    println!("{}", "LISTS".bright_yellow());
    println!("  {}   build a pair / list", "(cons head tail)".bright_green());
    println!("  {}         first element", "(car lst)".bright_green());
    println!("  {}         rest of list", "(cdr lst)".bright_green());
    println!();

    println!("{}", "SECURE MPC OPERATIONS".bright_yellow());
    println!("  {}  secret-share value v in an nbits-wide field",
        "(share v nbits)".bright_green());
    println!("  {}   reconstruct / reveal a shared value",
        "(reconstruct x)".bright_green());
    println!("  {}    auto-share as N-bit signed integer (in let bindings)",
        ":sintN".bright_green());
    println!();
    println!("  Shared arithmetic:");
    println!("    {}   local — zero communication rounds", "(+ s1 s2)  (- s1 s2)".bright_green());
    println!("    {}          Beaver triple — 1 round", "(* s1 s2)".bright_green());
    println!("    {}  comparison via a2b — ~1 round", "(< s1 s2)  (> s1 s2)".bright_green());
    println!("    {}        equality test via zero-check", "(== s1 s2)".bright_green());
    println!("    {}     branching on shared bool (leaks 1 bit!)", "(if s ...)".bright_green());
    println!();

    println!("{}", "REPL COMMANDS".bright_yellow());
    let cmds = [
        (":help",                       "show this help message"),
        (":examples",                   "list all example categories"),
        (":examples <category>",        "show examples for a category"),
        (":example <n>",                "run example number n"),
        (":verbose [on|off]",           "toggle per-participant eval traces"),
        (":last [pid]",                 "print last result for participant pid"),
        (":results",                    "print all participants' last results"),
        (":reveal <pid>",               "reveal last shared result for participant pid"),
        (":evalfor <pid> EXPR",         "evaluate EXPR for a specific participant"),
        (":inspect <pid> <var>",        "reveal a variable for a participant"),
        (":save <pid> [file]",          "save participant store to JSON file"),
        (":load <pid> [file]",          "load participant store from JSON file"),
        (":reset [pid]",                "clear participant store(s)"),
        ("quit / exit",                 "exit the REPL"),
    ];
    for (cmd, desc) in &cmds {
        println!("  {:<32} {}", cmd.bright_cyan(), desc);
    }
    println!();
    println!("Tip: use ':examples mpc-basics' to jump straight to MPC examples.");
    println!();
}

fn print_examples_list() {
    println!();
    println!("{}", "=== Example Categories ===".bright_yellow());
    println!("  Use '{}' to see examples  |  '{}' to run one",
        ":examples <category>".bright_cyan(), ":example <n>".bright_cyan());
    println!();

    let cats = distinct_categories();
    let mut start = 1usize;
    for cat in &cats {
        let count = EXAMPLES.iter().filter(|e| e.category == *cat).count();
        let end = start + count - 1;
        println!("  {:<20} [{:>2}-{:>2}]  {} examples",
            cat.bright_cyan(),
            start.to_string().bright_white(),
            end.to_string().bright_white(),
            count);
        start += count;
    }
    println!();
}

fn print_examples_for(filter: &str) {
    let filter_lc = filter.to_lowercase();
    let matched: Vec<(usize, &Example)> = EXAMPLES.iter()
        .enumerate()
        .filter(|(_, e)| e.category.to_lowercase().contains(&filter_lc))
        .map(|(i, e)| (i + 1, e))   // 1-based global index
        .collect();

    if matched.is_empty() {
        println!("No examples found matching '{}'.", filter);
        let cats = distinct_categories();
        println!("Available categories: {}", cats.join(", "));
        return;
    }

    let cat_name = matched[0].1.category;
    println!();
    println!("{}", format!("=== {} ===", cat_name).bright_yellow());
    println!("  Run with '{}' e.g. ':example {}'",
        ":example <n>".bright_cyan(), matched[0].0);
    println!();
    for (global_idx, ex) in &matched {
        println!("  [{}] {}",
            global_idx.to_string().bright_white(),
            ex.name.bright_cyan());
        println!("      {}", ex.code.bright_green());
        println!("      {} {}", "=>".bright_yellow(), ex.expected.bright_magenta());
        println!();
    }
}

// ─────────────────────────────────────────────────────────────────────────────

struct ReplHelper {
    words: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        let words = vec![
            ":help".to_string(),
            ":examples".to_string(),
            ":example".to_string(),
            ":verbose".to_string(),
            ":last".to_string(),
            ":results".to_string(),
            ":reveal".to_string(),
            ":evalfor".to_string(),
            ":inspect".to_string(),
            ":save".to_string(),
            ":load".to_string(),
            ":reset".to_string(),
            "quit".to_string(),
            "exit".to_string(),
            "share".to_string(),
            "reconstruct".to_string(),
            "a2b".to_string(),
            "b2a".to_string(),
            // category names for :examples tab-completion
            "arithmetic".to_string(),
            "comparisons".to_string(),
            "booleans".to_string(),
            "let-lambda".to_string(),
            "higher-order".to_string(),
            "lists".to_string(),
            "recursion".to_string(),
            "church".to_string(),
            "mpc-basics".to_string(),
            "mpc-signed".to_string(),
            "mpc-compare".to_string(),
            "relu-network".to_string(),
            "mpc-oblivious".to_string(),
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

    println!("{}", "ABY3 REPL — Secure Multi-Party Computation (replicated 3-party evaluation).".bright_green());
    println!("{}", "Type ':help' for language reference, ':examples' to browse built-in examples, or 'quit' to exit.".bright_yellow());

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
                        ":help" => {
                            print_help();
                        }
                        ":examples" => {
                            match parts.next() {
                                None => print_examples_list(),
                                Some(cat) => print_examples_for(cat),
                            }
                        }
                        ":example" => {
                            match parts.next() {
                                None => println!("usage: :example <n>  (use ':examples' to list)"),
                                Some(n_s) => match n_s.parse::<usize>() {
                                    Ok(n) if n >= 1 && n <= EXAMPLES.len() => {
                                        let ex = &EXAMPLES[n - 1];
                                        println!();
                                        println!("{} [{}] {} ({})",
                                            "Running example".bright_yellow(),
                                            n.to_string().bright_white(),
                                            ex.name.bright_cyan(),
                                            ex.category);
                                        println!("{} {}", "Code:".bright_yellow(), ex.code.bright_green());
                                        println!("{} {}", "Expected:".bright_yellow(), ex.expected.bright_magenta());
                                        println!();
                                        match parse(ex.code) {
                                            Ok(expr) => {
                                                let mut results: Vec<SharedValue> = Vec::new();
                                                for (i, p) in participants.iter().enumerate() {
                                                    let r = p.eval(&expr);
                                                    last_results[i] = Some(r.clone());
                                                    results.push(r);
                                                }
                                                match &results[0] {
                                                    SharedValue::Share(sr) => {
                                                        let raw = participants[0].reveal(sr);
                                                        let val = (raw as u32) as i32 as i64;
                                                        println!("{} {} (revealed, signed)", "result:".bright_cyan(), val.to_string().bright_green());
                                                    }
                                                    SharedValue::Num(n) => {
                                                        // sign-extend through 32 bits so MPC field values display correctly
                                                        let val = (*n as u32) as i32 as i64;
                                                        println!("{} {}", "result:".bright_cyan(), val.to_string().bright_green());
                                                    }
                                                    other => println!("{} {:?}", "result:".bright_cyan(), other),
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
                        ":verbose" => {
                            match parts.next() {
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
                            }
                        }
                        ":last" => {
                            match parts.next() {
                                Some(pid_s) => match pid_s.parse::<usize>() {
                                    Ok(pid) if pid < participants.len() => {
                                        match &last_results[pid] {
                                            Some(v) => println!("{} {}", format!("last[{}] =", pid).bright_magenta(), format!("{:?}", v)),
                                            None => println!("{}", format!("last[{}] = <none>", pid).bright_red()),
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
                                                    println!("{} {}", format!("reveal[{}] ->", pid).bright_magenta(), format!("{}", val).bright_green());
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
                        ":save" => {
                            match parts.next() {
                                Some(pid_s) => match pid_s.parse::<usize>() {
                                    Ok(pid) if pid < participants.len() => {
                                        let filename = parts.next().map(|s| s.to_string()).unwrap_or_else(|| format!("store_{}.json", pid));
                                        let path = std::path::PathBuf::from(filename);
                                        match participants[pid].save_store(&path) {
                                            Ok(_) => println!("saved store for participant {} -> {:?}", pid, path),
                                            Err(e) => println!("error saving store: {}", e),
                                        }
                                    }
                                    _ => println!("invalid participant id"),
                                },
                                None => println!("usage: :save <pid> [filename]"),
                            }
                        }
                        ":load" => {
                            match parts.next() {
                                Some(pid_s) => match pid_s.parse::<usize>() {
                                    Ok(pid) if pid < participants.len() => {
                                        let filename = parts.next().map(|s| s.to_string()).unwrap_or_else(|| format!("store_{}.json", pid));
                                        let path = std::path::PathBuf::from(filename);
                                        match participants[pid].load_store(&path) {
                                            Ok(_) => println!("loaded store for participant {} <- {:?}", pid, path),
                                            Err(e) => println!("error loading store: {}", e),
                                        }
                                    }
                                    _ => println!("invalid participant id"),
                                },
                                None => println!("usage: :load <pid> [filename]"),
                            }
                        }
                        ":reset" => {
                            match parts.next() {
                                Some(pid_s) => match pid_s.parse::<usize>() {
                                    Ok(pid) if pid < participants.len() => {
                                        participants[pid].reset_store();
                                        println!("reset store for participant {}", pid);
                                    }
                                    _ => println!("invalid participant id"),
                                },
                                None => {
                                    for pid in 0..participants.len() {
                                        participants[pid].reset_store();
                                    }
                                    println!("reset stores for all participants");
                                }
                            }
                        }
                        _ => println!("unknown command: {}  (type ':help' for command reference)", cmd),
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
