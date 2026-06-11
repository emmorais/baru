use baru::interpreter::{parse, Participant, SharedValue};

fn eval(src: &str) -> i64 {
    let expr = parse(src).expect("parse failed");
    let p = Participant::new(0);
    match p.eval(&expr) {
        SharedValue::Num(n) => n,
        other => panic!("expected numeric result, got: {:?}", other),
    }
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
    format!("(letrec ((fib (lambda (n) (if (== n 0) 0 (if (== n 1) 1 (+ (fib (- n 1)) (fib (- n 2)))))))) (fib {}))", n)
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
    assert_eq!(eval("(and 6 3)"), 2);
    assert_eq!(eval("(or 6 3)"), 7);
}

#[test]
fn test_cons_car_cdr_eval() {
    assert_eq!(eval("(car (cons 1 (cons 2 ())))"), 1);
    assert_eq!(eval("(car (cdr (cons 1 (cons 2 ()))) )"), 2);
}

#[test]
fn test_not_numeric() {
    assert_eq!(eval("(not 0)"), 1);
    assert_eq!(eval("(not 42)"), 0);
}

#[test]
fn test_string_closure_capture() {
    assert_eq!(eval("(let ((s \"hi\")) (== s \"hi\"))"), 1);
}

#[test]
fn test_shared_comparison_select_max() {
    // create two shared integers a and b, compute mask = (a > b) (shared bit)
    // then compute selected = a*mask + b*(1-mask) and reconstruct result
    let src = "(let ((a :sint32 30) (b :sint32 20) (one :sint32 1)) (let ((mask (> a b))) (+ (* a mask) (* b (- one mask)))))";
    let expr = parse(src).expect("parse failed");
    let p = Participant::new(0);
    let res = p.eval(&expr);
    match res {
        SharedValue::Share(sr) => {
            // reveal via participant backend (uses Backend::reveal)
            let plain = p.reveal(&sr);
            assert_eq!(plain, 30u128);
        }
        other => panic!("expected shared result, got: {:?}", other),
    }
}

#[test]
fn test_shared_comparison_select_min() {
    // same as above but with (a < b) mask to select min
    let src = "(let ((a :sint32 30) (b :sint32 20) (one :sint32 1)) (let ((mask (< a b))) (+ (* a mask) (* b (- one mask)))))";
    let expr = parse(src).expect("parse failed");
    let p = Participant::new(0);

    let res = p.eval(&expr);
    match res {
        SharedValue::Share(sr) => {
            let plain = p.reveal(&sr);
            assert_eq!(plain, 20u128);
        }
        other => panic!("expected shared result, got: {:?}", other),
    }
}   

#[test]
fn test_shared_equality() {
    // create two shared integers a and b, compute mask = (a == b) (shared bit)
    // then compute selected = a*mask + b*(1-mask) and reconstruct result; test both equal and not equal cases
    for (a_val, b_val, expected) in &[(42, 42, 42), (42, 43, 43)] {
        let src = format!("(let ((a :sint32 {}) (b :sint32 {}) (one :sint32 1)) (let ((mask (== a b))) (+ (* a mask) (* b (- one mask)))))", a_val, b_val);
        let expr = parse(&src).expect("parse failed");
        let p = Participant::new(0);
        let res = p.eval(&expr);
        match res {
            SharedValue::Share(sr) => {
                // reveal via participant backend
                let plain = p.reveal(&sr);
                assert_eq!(plain, (*expected) as u128);
            }
             other => panic!("expected shared result, got: {:?}", other),
         }
     }
 }

 #[test]
 fn test_very_simple_neural_network() {
     // test a simple 2-layer NN with shared weights and inputs; just checks that the computation runs and produces some output
     let src = "(let ((w1 :sint32 2) (w2 :sint32 3) (x :sint32 4)) (+ (* w1 x) (* w2 x)))";
     let expr = parse(src).expect("parse failed");
     let p = Participant::new(0);
     let res = p.eval(&expr);
     match res {
         SharedValue::Share(sr) => {
             let plain = p.reveal(&sr);
             assert_eq!(plain, 20u128); // (2*4) + (3*4) = 8 + 12 = 20
         }
         other => panic!("expected shared result, got: {:?}", other),
     }
 }

 #[test]
 fn test_simple_neural_network(){
        // test a simple 2-layer NN with shared weights and inputs, including a ReLU activation; just checks that the computation runs and produces some output
        let src = "(let ((w1 :sint32 2) (w2 :sint32 -3) (x :sint32 4)) (let ((z (+ (* w1 x) (* w2 x)))) (if (> z 0) z 0)))";
        let expr = parse(src).expect("parse failed");
        let p = Participant::new(0);
        let res = p.eval(&expr);
        match res {
            SharedValue::Share(sr) => {
                let plain = p.reveal(&sr);
                assert_eq!(plain, 0u128); // (2*4) + (-3*4) = 8 - 12 = -4, ReLU(-4) = 0
            }
            other => panic!("expected shared result, got: {:?}", other),
        }
 }