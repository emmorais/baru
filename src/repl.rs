//! Shared REPL utilities: examples catalog, help text, tab-completion.
//! Used by both the single-process REPL (`main.rs`) and the TCP multi-party
//! binary (`bin/tcp_party.rs`).

use colored::*;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::Helper;

// ── Examples catalog ─────────────────────────────────────────────────────────

pub struct Example {
    pub category: &'static str,
    pub name: &'static str,
    pub code: &'static str,
    pub expected: &'static str,
}

#[rustfmt::skip]
pub const EXAMPLES: &[Example] = &[
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

    // ── Lists ─────────────────────────────────────────────────────────────────
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

    // ── Classic CS: GCD ───────────────────────────────────────────────────────
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

    // ── MPC Private Inputs (share-from) ──────────────────────────────────────
    // In single-process REPL: party 0 supplies all inputs (owner id is symbolic).
    // In tcp_party mode: each party supplies only their own value; others pass 0.
    Example { category: "mpc-private-inputs", name: "two-party-sum",
              code: "(reconstruct (+ (share-from 0 10 32) (share-from 1 32 32)))",
              expected: "42  (party 0 contributes 10, party 1 contributes 32)" },
    Example { category: "mpc-private-inputs", name: "two-party-mul",
              code: "(reconstruct (* (share-from 0 6 32) (share-from 1 7 32)))",
              expected: "42  (party 0 contributes 6, party 1 contributes 7)" },
    Example { category: "mpc-private-inputs", name: "salary-threshold",
              code: "(let ((salary :sint32-from-0 75000) (threshold :sint32-from-1 80000)) (> salary threshold))",
              expected: "0  (salary < threshold — party 0 earns less, no reveal of actual values)" },
    Example { category: "mpc-private-inputs", name: "private-max",
              code: "(let ((a :sint32-from-0 30) (b :sint32-from-1 42) (one (share-from 0 1 32))) (let ((m (> a b))) (reconstruct (+ (* a m) (* b (- one m))))))",
              expected: "42  (private max: neither party learns the other's input)" },
    Example { category: "mpc-private-inputs", name: "three-party-sum",
              code: "(reconstruct (+ (share-from 0 10 32) (+ (share-from 1 20 32) (share-from 2 30 32))))",
              expected: "60  (each party contributes their own value)" },
];

pub fn distinct_categories() -> Vec<&'static str> {
    let mut seen = std::collections::HashSet::new();
    let mut cats: Vec<&'static str> = Vec::new();
    for ex in EXAMPLES {
        if seen.insert(ex.category) {
            cats.push(ex.category);
        }
    }
    cats
}

// ── Help text ─────────────────────────────────────────────────────────────────

/// Print the language reference + command list.
/// `tcp_mode = true` shows TCP-party-0 commands instead of single-REPL commands.
pub fn print_help(tcp_mode: bool) {
    println!();
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════╗".bright_cyan()
    );
    println!(
        "{}",
        "║      Baru — a Lisp-like Secure Multi-Party Computation REPL  ║".bright_cyan()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════╝".bright_cyan()
    );
    println!();

    println!("{}", "LITERALS".bright_yellow());
    println!(
        "  {}   integers (positive and negative)",
        "42   -7   0".bright_green()
    );
    println!("  {}      boolean true / false", "#t   #f".bright_green());
    println!("  {}  empty list / nil", "()".bright_green());
    println!();

    println!("{}", "ARITHMETIC".bright_yellow());
    println!(
        "  {}   addition / subtraction",
        "(+ a b)  (- a b)".bright_green()
    );
    println!(
        "  {}   multiplication / integer division",
        "(* a b)  (/ a b)".bright_green()
    );
    println!(
        "  {}           remainder (modulo)",
        "(mod a b)".bright_green()
    );
    println!();

    println!(
        "{}",
        "COMPARISONS  — return 1 (true) or 0 (false)".bright_yellow()
    );
    println!(
        "  {}",
        "(== a b)  (< a b)  (<= a b)  (> a b)  (>= a b)".bright_green()
    );
    println!();

    println!("{}", "BOOLEAN / BITWISE".bright_yellow());
    println!(
        "  {}              logical / bitwise not",
        "(not x)".bright_green()
    );
    println!(
        "  {}        logical / bitwise and, or",
        "(and a b)  (or a b)".bright_green()
    );
    println!();

    println!("{}", "CONDITIONALS".bright_yellow());
    println!(
        "  {}  evaluates then or else branch",
        "(if cond then else)".bright_green()
    );
    println!();

    println!("{}", "LET BINDINGS".bright_yellow());
    println!(
        "  {}  parallel bindings, then body",
        "(let ((x 1) (y 2)) body)".bright_green()
    );
    println!(
        "  {}  auto-shares as N-bit signed integer",
        "(let ((z :sint32 -5)) body)".bright_green()
    );
    println!();

    println!("{}", "LAMBDAS — first-class functions".bright_yellow());
    println!(
        "  {}      single parameter",
        "(lambda (x) body)".bright_green()
    );
    println!(
        "  {}   multiple parameters",
        "(lambda (x y) body)".bright_green()
    );
    println!(
        "  {}    direct application",
        "((lambda (x) (* x x)) 5)  =>  25".bright_green()
    );
    println!();

    println!("{}", "RECURSIVE BINDINGS".bright_yellow());
    println!("  {}", "(letrec ((fact (lambda (n)".bright_green());
    println!(
        "  {}",
        "           (if (== n 0) 1 (* n (fact (- n 1))))))".bright_green()
    );
    println!("  {}", "  (fact 10))".bright_green());
    println!("  Mutually recursive bindings are supported.");
    println!();

    println!("{}", "LISTS".bright_yellow());
    println!(
        "  {}   build a pair / list",
        "(cons head tail)".bright_green()
    );
    println!("  {}         first element", "(car lst)".bright_green());
    println!("  {}          rest of list", "(cdr lst)".bright_green());
    println!();

    println!("{}", "SECURE MPC OPERATIONS".bright_yellow());
    println!(
        "  {}  secret-share value v in an nbits-wide field (party 0 is owner)",
        "(share v nbits)".bright_green()
    );
    println!(
        "  {}  secret-share v with explicit owner party id",
        "(share-from owner v nbits)".bright_green()
    );
    println!(
        "  {}   reconstruct / reveal a shared value",
        "(reconstruct x)".bright_green()
    );
    println!(
        "  {}    auto-share as N-bit signed integer (party 0 owner)",
        ":sintN".bright_green()
    );
    println!(
        "  {}  auto-share with explicit owner (e.g. :sint32-from-1)",
        ":sintN-from-M".bright_green()
    );
    println!();
    println!("  Shared arithmetic:");
    println!(
        "    {}   local — zero communication rounds",
        "(+ s1 s2)  (- s1 s2)".bright_green()
    );
    println!(
        "    {}          Beaver triple — 1 round",
        "(* s1 s2)".bright_green()
    );
    println!(
        "    {}  comparison via a2b — ~1 round",
        "(< s1 s2)  (> s1 s2)".bright_green()
    );
    println!(
        "    {}        equality test via zero-check",
        "(== s1 s2)".bright_green()
    );
    println!(
        "    {}     branching on shared bool (leaks 1 bit!)",
        "(if s ...)".bright_green()
    );
    println!();

    if tcp_mode {
        println!(
            "{}",
            "REPL COMMANDS (party 0 — distributed mode)".bright_yellow()
        );
        let cmds: &[(&str, &str)] = &[
            (":help", "show this help message"),
            (":examples", "list all example categories"),
            (":examples <category>", "show examples for a category"),
            (
                ":example <n>",
                "broadcast and run example jointly with all parties",
            ),
            (":verbose [on|off]", "toggle eval traces"),
            (":quit", "signal all parties to quit and exit"),
        ];
        for (cmd, desc) in cmds {
            println!("  {:<32} {}", cmd.bright_cyan(), desc);
        }
    } else {
        println!("{}", "REPL COMMANDS".bright_yellow());
        let cmds: &[(&str, &str)] = &[
            (":help", "show this help message"),
            (":examples", "list all example categories"),
            (":examples <category>", "show examples for a category"),
            (":example <n>", "run example number n"),
            (":verbose [on|off]", "toggle per-participant eval traces"),
            (":last [pid]", "print last result for participant pid"),
            (":results", "print all participants' last results"),
            (
                ":reveal <pid>",
                "reveal last shared result for participant pid",
            ),
            (
                ":evalfor <pid> EXPR",
                "evaluate EXPR for a specific participant",
            ),
            (
                ":inspect <pid> <var>",
                "reveal a variable for a participant",
            ),
            (":save <pid> [file]", "save participant store to JSON file"),
            (
                ":load <pid> [file]",
                "load participant store from JSON file",
            ),
            (":reset [pid]", "clear participant store(s)"),
            ("quit / exit", "exit the REPL"),
        ];
        for (cmd, desc) in cmds {
            println!("  {:<32} {}", cmd.bright_cyan(), desc);
        }
    }
    println!();
    println!("Tip: use ':examples mpc-basics' to jump straight to MPC examples.");
    println!();
}

pub fn print_examples_list() {
    println!();
    println!("{}", "=== Example Categories ===".bright_yellow());
    println!(
        "  Use '{}' to see examples  |  '{}' to run one",
        ":examples <category>".bright_cyan(),
        ":example <n>".bright_cyan()
    );
    println!();

    let cats = distinct_categories();
    let mut start = 1usize;
    for cat in &cats {
        let count = EXAMPLES.iter().filter(|e| e.category == *cat).count();
        let end = start + count - 1;
        println!(
            "  {:<20} [{:>2}-{:>2}]  {} examples",
            cat.bright_cyan(),
            start.to_string().bright_white(),
            end.to_string().bright_white(),
            count
        );
        start += count;
    }
    println!();
}

pub fn print_examples_for(filter: &str) {
    let filter_lc = filter.to_lowercase();
    let matched: Vec<(usize, &Example)> = EXAMPLES
        .iter()
        .enumerate()
        .filter(|(_, e)| e.category.to_lowercase().contains(&filter_lc))
        .map(|(i, e)| (i + 1, e))
        .collect();

    if matched.is_empty() {
        println!("No examples found matching '{}'.", filter);
        println!("Available categories: {}", distinct_categories().join(", "));
        return;
    }

    let cat_name = matched[0].1.category;
    println!();
    println!("{}", format!("=== {} ===", cat_name).bright_yellow());
    println!(
        "  Run with '{}' e.g. ':example {}'",
        ":example <n>".bright_cyan(),
        matched[0].0
    );
    println!();
    for (global_idx, ex) in &matched {
        println!(
            "  [{}] {}",
            global_idx.to_string().bright_white(),
            ex.name.bright_cyan()
        );
        println!("      {}", ex.code.bright_green());
        println!(
            "      {} {}",
            "=>".bright_yellow(),
            ex.expected.bright_magenta()
        );
        println!();
    }
}

// ── Tab-completion helper ─────────────────────────────────────────────────────

const BASE_WORDS: &[&str] = &[
    ":help",
    ":examples",
    ":example",
    ":verbose",
    "share",
    "share-from",
    "reconstruct",
    "a2b",
    "b2a",
    "arithmetic",
    "comparisons",
    "booleans",
    "let-lambda",
    "higher-order",
    "lists",
    "recursion",
    "church",
    "mpc-basics",
    "mpc-signed",
    "mpc-compare",
    "relu-network",
    "mpc-oblivious",
    "mpc-private-inputs",
];

const REPL_ONLY_WORDS: &[&str] = &[
    ":last", ":results", ":reveal", ":evalfor", ":inspect", ":save", ":load", ":reset", "quit",
    "exit",
];

const TCP_ONLY_WORDS: &[&str] = &[":quit"];

pub struct ReplHelper {
    pub words: Vec<String>,
}

impl ReplHelper {
    /// Tab-completion helper for the single-process REPL.
    pub fn new() -> Self {
        let mut words: Vec<String> = BASE_WORDS
            .iter()
            .chain(REPL_ONLY_WORDS.iter())
            .map(|&s| s.to_string())
            .collect();
        words.sort();
        ReplHelper { words }
    }

    /// Tab-completion helper for the TCP party-0 REPL.
    pub fn new_tcp() -> Self {
        let mut words: Vec<String> = BASE_WORDS
            .iter()
            .chain(TCP_ONLY_WORDS.iter())
            .map(|&s| s.to_string())
            .collect();
        words.sort();
        ReplHelper { words }
    }
}

impl Default for ReplHelper {
    fn default() -> Self {
        Self::new()
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
        let matches: Vec<Pair> = self
            .words
            .iter()
            .filter(|w| word.is_empty() || w.starts_with(word))
            .map(|w| Pair {
                display: w.clone(),
                replacement: w.clone(),
            })
            .collect();
        Ok((start, matches))
    }
}
