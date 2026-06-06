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

#[cfg(test)]
mod tests {
    use super::parse;
    use baru::interpreter::Participant;
    use baru::interpreter::SharedValue;

    fn eval(src: &str) -> i64 {
        let expr = parse(src).expect("parse failed");
        let p = Participant::new(0);
        match p.eval(&expr) {
            SharedValue::Num(n) => n,
            other => panic!("expected numeric result, got: {:?}", other),
        }
    }

    fn eval_val(src: &str) -> SharedValue {
        let expr = parse(src).expect("parse failed");
        let p = Participant::new(0);
        p.eval(&expr)
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
    fn test_cons_car_cdr() {
        let _ = parse("(cons 1 (cons 2 (cons 3 ())))");
        let l = SharedValue::List(vec![SharedValue::Num(1), SharedValue::Num(2)]);
        assert!(matches!(l, SharedValue::List(_)));
    }

    #[test]
    fn test_lambda_direct_call() {
        assert_eq!(eval("((lambda (x) (+ x 1)) 5)"), 6);
    }

    #[test]
    fn test_lambda_let_bind_and_call() {
        assert_eq!(eval("(let ((inc (lambda (x) (+ x 1)))) (inc 41))"), 42);
    }

    #[test]
    fn test_lambda_closure_capture() {
        assert_eq!(eval("(let ((a 10)) (let ((f (lambda (x) (+ x a)))) (f 5)))"), 15);
    }

    #[test]
    fn test_letrec_recursive_function() {
        assert_eq!(eval("(letrec ((f (lambda (x) (+ x 0)))) (f 5))"), 5);
    }

    #[test]
    fn test_mutual_recursion_even_odd() {
        let src = "(letrec ((iszero (lambda (n) (if (== n 0) 1 0))) (even (lambda (n) (if (== n 0) 1 (odd (- n 1)))) ) (odd (lambda (n) (if (== n 0) 0 (even (- n 1)))))) (even 4))";
        assert_eq!(eval(src), 1);
    }

    fn fib_src(n: i64) -> String {
        format!("(letrec 
                    ((fib (lambda (n) 
                        (if (== n 0) 0 
                            (if (== n 1) 1 
                                (+ 
                                    (fib (- n 1)) 
                                    (fib (- n 2))
                                )
                            )
                        )
                    ))) 
                    (fib {})
                  )", n)
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(eval(&fib_src(0)), 0);
        assert_eq!(eval(&fib_src(1)), 1);
        assert_eq!(eval(&fib_src(6)), 8);
        assert_eq!(eval(&fib_src(10)), 55);
        assert_eq!(eval(&fib_src(20)), 6765);
    }

    #[test]
    fn test_strings_and_chars_eq() {
        assert_eq!(eval("(== \"hi\" \"hi\")"), 1);
        assert_eq!(eval("(== \"hi\" \"ho\")"), 0);
        assert_eq!(eval("(== 'a' 'a')"), 1);
        assert_eq!(eval("(== 'a' 'b')"), 0);
    }

    #[test]
    fn test_booleans_not_and_or() {
        assert_eq!(eval("(not #t)"), 0);
        assert_eq!(eval("(not #f)"), 1);
        assert_eq!(eval("(and #t #f)"), 0);
        assert_eq!(eval("(and #t #t)"), 1);
        assert_eq!(eval("(or #t #f)"), 1);
        assert_eq!(eval("(or #f #f)"), 0);
    }

    #[test]
    fn test_comparisons() {
        assert_eq!(eval("(< 3 4)"), 1);
        assert_eq!(eval("(<= 3 3)"), 1);
        assert_eq!(eval("(> 5 2)"), 1);
        assert_eq!(eval("(>= 2 3)"), 0);
    }

    #[test]
    fn test_mod_and_bitwise() {
        assert_eq!(eval("(mod 10 3)"), 1);
        assert_eq!(eval("(and 6 3)"), 2); // bitwise AND for numeric operands
        assert_eq!(eval("(or 6 3)"), 7);  // bitwise OR for numeric operands
    }
}
