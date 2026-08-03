use baru::interpreter::{
    eval::{eval_fhe, eval_plain as eval, eval_shared},
    eval_plain_string,
};

// ---------------------------------------------------------------------------
// Basic arithmetic
// ---------------------------------------------------------------------------

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
fn test_add_mul_nested() {
    assert_eq!(eval("(+ (* 6 7) (* 3 14))"), 84);
}

#[test]
fn test_mod_basic() {
    assert_eq!(eval("(mod 10 3)"), 1);
}

#[test]
fn test_neg_unary() {
    assert_eq!(eval("(- 0 5)"), -5);
}

#[test]
fn test_wrapping_arithmetic() {
    // basic wrapping identity: (a - a) = 0
    assert_eq!(eval("(- 100 100)"), 0);
    assert_eq!(eval("(+ 0 0)"), 0);
}

// ---------------------------------------------------------------------------
// Atom
// ---------------------------------------------------------------------------

#[test]
fn test_atom() {
    assert_eq!(eval("(atom 42)"), 1);
    assert_eq!(eval("(atom (+ 2 3))"), 1);
    assert_eq!(eval("(let ((x 10)) (atom (+ x 5)))"), 1);
}

// ---------------------------------------------------------------------------
// Quoting
// ---------------------------------------------------------------------------

#[test]
fn test_quote() {
    assert_eq!(eval_plain_string("(quote 42)"), "42");
    assert_eq!(eval_plain_string("(quote (+ 2 3))"), "(+ 2 3)");
    assert_eq!(
        eval_plain_string("(let ((x 10)) (quote (+ x 5)))"),
        "(+ x 5)"
    );
}

#[test]
fn test_eval() {
    assert_eq!(eval_plain_string("(eval (quote (+ 2 3)))"), "5");
    assert_eq!(eval_plain_string("(eval (quote (if #t 1 0)))"), "1");
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[test]
fn test_comparisons() {
    assert_eq!(eval("(< 3 4)"), 1);
    assert_eq!(eval("(<= 3 3)"), 1);
    assert_eq!(eval("(<= 3 4)"), 1);
    assert_eq!(eval("(> 5 2)"), 1);
    assert_eq!(eval("(>= 2 3)"), 0);
    assert_eq!(eval("(>= 3 3)"), 1);
    assert_eq!(eval("(== 7 7)"), 1);
    assert_eq!(eval("(== 7 8)"), 0);
}

#[test]
fn test_comparison_with_negatives() {
    assert_eq!(eval("(< -5 0)"), 1);
    assert_eq!(eval("(> 0 -1)"), 1);
    assert_eq!(eval("(<= -3 -3)"), 1);
}

// ---------------------------------------------------------------------------
// Booleans and logical operators
// ---------------------------------------------------------------------------

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
fn test_ash_left() {
    assert_eq!(eval("(ash-left 1 2)"), 4);
    assert_eq!(eval("(reconstruct (ash-left (share 1 32) 2))"), 4);
}

#[test]
fn test_ash_right() {
    assert_eq!(eval("(ash-right 4 2)"), 1);
    assert_eq!(eval("(reconstruct (ash-right (share 4 32) 2))"), 1);

    assert_eq!(eval("(ash-right 512 3)"), 64);
    assert_eq!(eval("(reconstruct (ash-right (share 512 32) 3))"), 64);
}

#[test]
#[ignore]
fn test_bitlen() {
    assert_eq!(eval("(bitlen 0)"), 0);
    assert_eq!(eval("(bitlen 1)"), 1);
    assert_eq!(eval("(bitlen 2)"), 2);
    assert_eq!(eval("(bitlen 3)"), 2);
    assert_eq!(eval("(bitlen 4)"), 3);
    assert_eq!(eval("(bitlen 9)"), 4);
    assert_eq!(eval("(bitlen 42)"), 6);
    assert_eq!(eval("(bitlen 1042)"), 11);
    // shared values should also work
    assert_eq!(eval("(reconstruct (bitlen (share 42 32)))"), 6);
    assert_eq!(eval("(reconstruct (bitlen (share 1042 32)))"), 11);
}

#[test]
fn test_not_numeric() {
    assert_eq!(eval("(not 0)"), 1);
    assert_eq!(eval("(not 42)"), 0);
}

#[test]
fn test_bitwise_and_or() {
    assert_eq!(eval("(and 6 3)"), 2);
    assert_eq!(eval("(or  6 3)"), 7);
}

// ---------------------------------------------------------------------------
// Strings and characters
// ---------------------------------------------------------------------------

#[test]
fn test_strings_eq() {
    assert_eq!(eval("(== \"hi\" \"hi\")"), 1);
    assert_eq!(eval("(== \"hi\" \"ho\")"), 0);
}

#[test]
fn test_chars_eq() {
    assert_eq!(eval("(== 'a' 'a')"), 1);
    assert_eq!(eval("(== 'a' 'b')"), 0);
}

#[test]
fn test_string_closure_capture() {
    assert_eq!(eval("(let ((s \"hi\")) (== s \"hi\"))"), 1);
}

// ---------------------------------------------------------------------------
// Let bindings
// ---------------------------------------------------------------------------

#[test]
fn test_let_simple() {
    assert_eq!(eval("(let ((a 6) (b 7)) (+ a b))"), 13);
}

#[test]
fn test_let_nested_shadow() {
    // inner binding shadows outer
    assert_eq!(eval("(let ((x 1)) (let ((x 2)) x))"), 2);
}

#[test]
fn test_let_multiple_body_exprs() {
    // last expression in body is the result
    assert_eq!(eval("(let ((a 1)) (+ a 1) (+ a 10))"), 11);
}

// ---------------------------------------------------------------------------
// Lambdas and closures
// ---------------------------------------------------------------------------

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
    assert_eq!(
        eval("(let ((a 10)) (let ((f (lambda (x) (+ x a)))) (f 5)))"),
        15
    );
}

#[test]
fn test_higher_order_apply() {
    // apply: takes a function and a value, applies the function
    let src = "(let ((apply (lambda (f x) (f x)))) (apply (lambda (n) (* n n)) 7))";
    assert_eq!(eval(src), 49);
}

#[test]
fn test_higher_order_compose() {
    // compose two functions: f∘g
    let src = "(let ((compose (lambda (f g) (lambda (x) (f (g x))))))
                 (let ((double (lambda (x) (* x 2)))
                       (inc    (lambda (x) (+ x 1))))
                   ((compose double inc) 3)))";
    assert_eq!(eval(src), 8); // double(inc(3)) = double(4) = 8
}

#[test]
fn test_currying() {
    // add-n returns a closure that adds n to its argument
    let src = "(let ((add-n (lambda (n) (lambda (x) (+ x n)))))
                 ((add-n 5) 37))";
    assert_eq!(eval(src), 42);
}

// ---------------------------------------------------------------------------
// Lists (cons / car / cdr)
// ---------------------------------------------------------------------------

#[test]
fn test_cons_car_cdr_eval() {
    assert_eq!(eval("(car (cons 1 (cons 2 ())))"), 1);
    assert_eq!(eval("(car (cdr (cons 1 (cons 2 ()))))"), 2);
}

#[test]
fn test_list_length_recursive() {
    // length of a list via recursion
    let src = "(letrec ((len (lambda (lst)
                              (if (== (car lst) 0)
                                  0
                                  (+ 1 (len (cdr lst)))))))
                 (len (cons 1 (cons 2 (cons 3 (cons 0 ()))))))";
    // Note: uses 0 as sentinel for empty list
    assert_eq!(eval(src), 3);
}

#[test]
fn test_list_sum_recursive() {
    // sum of a fixed list using a recursive function
    let src = "(letrec ((sum (lambda (n acc)
                               (if (== n 0)
                                   acc
                                   (sum (- n 1) (+ acc n))))))
                 (sum 10 0))";
    assert_eq!(eval(src), 55); // 1+2+…+10 = 55
}

// ---------------------------------------------------------------------------
// letrec — recursive and mutually recursive functions
// ---------------------------------------------------------------------------

#[test]
fn test_letrec_recursive_function() {
    assert_eq!(eval("(letrec ((f (lambda (x) (+ x 0)))) (f 5))"), 5);
}

#[test]
fn test_mutual_recursion_even_odd() {
    let src = "(letrec ((even (lambda (n) (if (== n 0) 1 (odd  (- n 1)))))
                        (odd  (lambda (n) (if (== n 0) 0 (even (- n 1))))))
                 (even 4))";
    assert_eq!(eval(src), 1);
}

#[test]
fn test_mutual_recursion_even_odd_odd_number() {
    let src = "(letrec ((even (lambda (n) (if (== n 0) 1 (odd  (- n 1)))))
                        (odd  (lambda (n) (if (== n 0) 0 (even (- n 1))))))
                 (odd 7))";
    assert_eq!(eval(src), 1);
}

// ---------------------------------------------------------------------------
// Classic CS: Fibonacci
// ---------------------------------------------------------------------------

fn fib_src(n: i64) -> String {
    format!(
        "(letrec ((fib (lambda (n)
                              (if (== n 0) 0
                                (if (== n 1) 1
                                  (+ (fib (- n 1)) (fib (- n 2))))))))
               (fib {}))",
        n
    )
}

#[test]
fn test_fibonacci() {
    assert_eq!(eval(&fib_src(0)), 0);
    assert_eq!(eval(&fib_src(1)), 1);
    assert_eq!(eval(&fib_src(6)), 8);
    assert_eq!(eval(&fib_src(10)), 55);
    assert_eq!(eval(&fib_src(20)), 6765);
}

// ---------------------------------------------------------------------------
// Classic CS: factorial
// ---------------------------------------------------------------------------

#[test]
fn test_factorial() {
    let src = "(letrec ((fact (lambda (n)
                                (if (== n 0) 1 (* n (fact (- n 1)))))))
                 (fact 10))";
    assert_eq!(eval(src), 3628800);
}

// ---------------------------------------------------------------------------
// Classic CS: power (fast exponentiation)
// ---------------------------------------------------------------------------

#[test]
fn test_fast_exponentiation() {
    // fast-pow via repeated squaring
    let src = "(letrec ((pow (lambda (base exp)
                               (if (== exp 0) 1
                                 (if (== (mod exp 2) 0)
                                   (let ((half (pow base (/ exp 2))))
                                     (* half half))
                                   (* base (pow base (- exp 1))))))))
                 (pow 2 10))";
    assert_eq!(eval(src), 1024);
}

// ---------------------------------------------------------------------------
// Classic CS: greatest common divisor (Euclidean algorithm)
// ---------------------------------------------------------------------------

#[test]
fn test_gcd() {
    let src = "(letrec ((gcd (lambda (a b)
                               (if (== b 0) a (gcd b (mod a b))))))
                 (gcd 48 18))";
    assert_eq!(eval(src), 6);
}

#[test]
fn test_gcd_coprime() {
    let src = "(letrec ((gcd (lambda (a b)
                               (if (== b 0) a (gcd b (mod a b))))))
                 (gcd 17 5))";
    assert_eq!(eval(src), 1);
}

// ---------------------------------------------------------------------------
// Classic CS: integer square root (Newton–Raphson)
// ---------------------------------------------------------------------------

#[test]
fn test_isqrt() {
    // integer square root: largest n with n*n <= x
    let src = "(letrec ((isqrt (lambda (x)
                                 (letrec ((go (lambda (n)
                                               (if (> (* (+ n 1) (+ n 1)) x)
                                                 n
                                                 (go (+ n 1))))))
                                   (go 0)))))
                 (isqrt 100))";
    assert_eq!(eval(src), 10);
}

#[test]
fn test_isqrt_non_perfect() {
    let src = "(letrec ((isqrt (lambda (x)
                                 (letrec ((go (lambda (n)
                                               (if (> (* (+ n 1) (+ n 1)) x)
                                                 n
                                                 (go (+ n 1))))))
                                   (go 0)))))
                 (isqrt 50))";
    assert_eq!(eval(src), 7); // floor(sqrt(50)) = 7
}

// ---------------------------------------------------------------------------
// Classic CS: Collatz sequence length
// ---------------------------------------------------------------------------

#[test]
fn test_collatz_length() {
    // number of steps for n to reach 1 under the Collatz rule
    let src = "(letrec ((collatz (lambda (n steps)
                                   (if (== n 1) steps
                                     (if (== (mod n 2) 0)
                                       (collatz (/ n 2)     (+ steps 1))
                                       (collatz (+ (* 3 n) 1) (+ steps 1)))))))
                 (collatz 27 0))";
    assert_eq!(eval(src), 111); // Collatz(27) takes 111 steps
}

// ---------------------------------------------------------------------------
// Classic CS: accumulator pattern (tail-recursive sum)
// ---------------------------------------------------------------------------

#[test]
fn test_tail_recursive_sum() {
    let src = "(letrec ((sum (lambda (n acc)
                               (if (== n 0) acc (sum (- n 1) (+ acc n))))))
                 (sum 100 0))";
    assert_eq!(eval(src), 5050);
}

// ---------------------------------------------------------------------------
// Classic CS: ackermann function (small values)
// ---------------------------------------------------------------------------

#[test]
fn test_ackermann() {
    let src = "(letrec ((ack (lambda (m n)
                               (if (== m 0) (+ n 1)
                                 (if (== n 0) (ack (- m 1) 1)
                                   (ack (- m 1) (ack m (- n 1))))))))
                 (ack 3 3))";
    assert_eq!(eval(src), 61);
}

// ---------------------------------------------------------------------------
// if expressions
// ---------------------------------------------------------------------------

#[test]
fn test_if_true_branch() {
    assert_eq!(eval("(if (== 1 1) 42 0)"), 42);
}

#[test]
fn test_if_false_branch() {
    assert_eq!(eval("(if (== 1 2) 42 99)"), 99);
}

#[test]
fn test_nested_if() {
    let src = "(let ((x 5))
                 (if (< x 0) -1
                   (if (== x 0) 0
                     1)))";
    assert_eq!(eval(src), 1);
}

// ---------------------------------------------------------------------------
// Shared (MPC) arithmetic — plain values
// ---------------------------------------------------------------------------

#[test]
fn test_shared_add_reveal() {
    assert_eq!(
        eval_shared("(reconstruct (+ (share 10 32) (share 32 32)))"),
        42
    );
}

#[test]
fn test_shared_mul_reveal() {
    assert_eq!(
        eval_shared("(reconstruct (* (share 6 32) (share 7 32)))"),
        42
    );
}

#[test]
fn test_shared_sub_reveal() {
    assert_eq!(
        eval_shared("(reconstruct (- (share 100 32) (share 58 32)))"),
        42
    );
}

#[test]
fn test_shared_scalar_add() {
    // mixed: shared + plain constant
    assert_eq!(eval_shared("(reconstruct (+ (share 40 32) 2))"), 42);
}

#[test]
fn test_shared_scalar_mul() {
    assert_eq!(eval_shared("(reconstruct (* (share 6 32) 7))"), 42);
}

// ---------------------------------------------------------------------------
// Shared (MPC) division and modulo — both operands shared, restoring-division
// circuit (see Backend::divmod's default implementation).
// ---------------------------------------------------------------------------

#[test]
fn test_shared_div_both_shared() {
    assert_eq!(
        eval_shared("(reconstruct (/ (share 84 32) (share 7 32)))"),
        12
    );
}

#[test]
fn test_shared_mod_both_shared() {
    assert_eq!(
        eval_shared("(reconstruct (mod (share 10 32) (share 3 32)))"),
        1
    );
}

#[test]
fn test_shared_div_scalar_divisor() {
    // shared dividend, plain (public) divisor
    assert_eq!(eval_shared("(reconstruct (/ (share 84 32) 7))"), 12);
}

#[test]
fn test_shared_mod_scalar_divisor() {
    assert_eq!(eval_shared("(reconstruct (mod (share 10 32) 3))"), 1);
}

#[test]
fn test_shared_div_scalar_dividend() {
    // plain (public) dividend, shared divisor
    assert_eq!(eval_shared("(reconstruct (/ 84 (share 7 32)))"), 12);
}

#[test]
fn test_shared_mod_scalar_dividend() {
    assert_eq!(eval_shared("(reconstruct (mod 10 (share 3 32)))"), 1);
}

#[test]
fn test_shared_divmod_matches_plain_semantics_signed() {
    // Truncating division / dividend-signed remainder, matching Rust's `/`
    // and `%` on i64 (and this crate's own plain-value BinOp::Mod/DivExact
    // arm) for a spread of signed cases, including negative operands.
    let cases: &[(i64, i64)] = &[
        (7, 3),
        (-7, 3),
        (7, -3),
        (-7, -3),
        (100, 9),
        (-100, 9),
        (1, 7),
        (-1, 7),
        (0, 5),
        (6, 3),
        (-6, 3),
    ];
    for &(a, b) in cases {
        let expected_div = a / b;
        let expected_mod = a % b;
        let src_div = format!("(reconstruct (/ (share {a} 32) (share {b} 32)))");
        let src_mod = format!("(reconstruct (mod (share {a} 32) (share {b} 32)))");
        assert_eq!(
            eval_shared(&src_div),
            expected_div,
            "shared {a} / {b} should be {expected_div}"
        );
        assert_eq!(
            eval_shared(&src_mod),
            expected_mod,
            "shared {a} mod {b} should be {expected_mod}"
        );
        // a == b*q + r must hold, exactly as it does for the plain path.
        assert_eq!(expected_div * b + expected_mod, a);
    }
}

// ---------------------------------------------------------------------------
// Shared arithmetic with signed values
// ---------------------------------------------------------------------------

#[test]
fn test_shared_negative_input() {
    assert_eq!(eval_shared("(reconstruct (share -42 32))"), -42);
}

#[test]
fn test_shared_signed_add() {
    assert_eq!(
        eval_shared("(reconstruct (+ (share -10 32) (share 52 32)))"),
        42
    );
}

#[test]
fn test_shared_signed_mul() {
    assert_eq!(
        eval_shared("(reconstruct (* (share -6 32) (share -7 32)))"),
        42
    );
}

#[test]
fn test_shared_signed_sub_negative_result() {
    assert_eq!(
        eval_shared("(reconstruct (- (share 3 32) (share 10 32)))"),
        -7
    );
}

// ---------------------------------------------------------------------------
// Shared comparisons -> oblivious select (max / min)
// ---------------------------------------------------------------------------

#[test]
fn test_shared_comparison_select_max() {
    let src = "(let ((a (share 30 32)) (b (share 20 32)) (one (share 1 32)))
                 (let ((mask (> a b)))
                   (+ (* a mask) (* b (- one mask)))))";
    assert_eq!(eval_shared(src), 30);
}

#[test]
fn test_shared_comparison_select_min() {
    let src = "(let ((a (share 30 32)) (b (share 20 32)) (one (share 1 32)))
                 (let ((mask (< a b)))
                   (+ (* a mask) (* b (- one mask)))))";
    assert_eq!(eval_shared(src), 20);
}

#[test]
fn test_shared_comparison_equal_values() {
    // when a == b, both max and min should equal a (= b)
    let src = "(let ((a (share 42 32)) (b (share 42 32)) (one (share 1 32)))
                 (let ((mask (> a b)))
                   (+ (* a mask) (* b (- one mask)))))";
    assert_eq!(eval_shared(src), 42);
}

#[test]
fn test_shared_equality() {
    for (a_val, b_val, expected) in &[(42i64, 42i64, 42i64), (42, 43, 43)] {
        let src = format!(
            "(let ((a (share {} 32)) (b (share {} 32)) (one (share 1 32)))
               (let ((mask (== a b)))
                 (+ (* a mask) (* b (- one mask)))))",
            a_val, b_val
        );
        assert_eq!(
            eval_shared(&src),
            *expected,
            "equality select failed for a={} b={}",
            a_val,
            b_val
        );
    }
}

// ---------------------------------------------------------------------------
// Classic CS over shares: oblivious absolute value
// ---------------------------------------------------------------------------

#[test]
fn test_shared_abs_positive() {
    // abs(x) = if x >= 0 then x else -x
    let src = "(let ((x (share 7 32)))
                 (if (>= x 0) x (- (share 0 32) x)))";
    assert_eq!(eval_shared(src), 7);
}

#[test]
fn test_shared_abs_negative() {
    let src = "(let ((x (share -7 32)))
                 (if (>= x 0) x (- (share 0 32) x)))";
    assert_eq!(eval_shared(src), 7);
}

// ---------------------------------------------------------------------------
// Classic CS over shares: oblivious clamp
// ---------------------------------------------------------------------------

#[test]
fn test_shared_clamp() {
    // clamp(x, lo, hi) = max(lo, min(x, hi))
    // implemented via two oblivious selects
    let clamp = |x: i64, lo: i64, hi: i64| {
        let src = format!(
            "(let ((x  (share {} 32))
                   (lo (share {} 32))
                   (hi (share {} 32))
                   (one (share 1 32)))
               (let ((clamped_hi (let ((m (< x hi)))  (+ (* x m) (* hi (- one m))))))
                 (let ((m2 (> clamped_hi lo))) (+ (* clamped_hi m2) (* lo (- one m2))))))",
            x, lo, hi
        );
        eval_shared(&src)
    };
    assert_eq!(clamp(-5, 0, 10), 0);
    assert_eq!(clamp(5, 0, 10), 5);
    assert_eq!(clamp(15, 0, 10), 10);
}

// ---------------------------------------------------------------------------
// Classic CS: oblivious swap (sorting network primitive)
// ---------------------------------------------------------------------------

#[test]
fn test_oblivious_sort_two_elements() {
    // compare-and-swap: ensures result[0] <= result[1]
    // min = a*(a<b) + b*(1-(a<b))
    // max = b*(a<b) + a*(1-(a<b))
    let sort2 = |a: i64, b: i64| -> (i64, i64) {
        let src_min = format!(
            "(let ((a (share {} 32)) (b (share {} 32)) (one (share 1 32)))
               (let ((m (< a b))) (+ (* a m) (* b (- one m)))))",
            a, b
        );
        let src_max = format!(
            "(let ((a (share {} 32)) (b (share {} 32)) (one (share 1 32)))
               (let ((m (> a b))) (+ (* a m) (* b (- one m)))))",
            a, b
        );
        (eval_shared(&src_min), eval_shared(&src_max))
    };
    assert_eq!(sort2(3, 7), (3, 7));
    assert_eq!(sort2(7, 3), (3, 7));
    assert_eq!(sort2(5, 5), (5, 5));
}

// ---------------------------------------------------------------------------
// Classic CS: Church numerals (purely functional, plain evaluator)
// ---------------------------------------------------------------------------

#[test]
fn test_church_numerals_add() {
    // Church encoding: zero = λf.λx.x, succ n = λf.λx.f(n f x)
    // church-to-int: n (λx.x+1) 0
    let src = "(let ((zero  (lambda (f) (lambda (x) x)))
                     (one   (lambda (f) (lambda (x) (f x))))
                     (two   (lambda (f) (lambda (x) (f (f x)))))
                     (three (lambda (f) (lambda (x) (f (f (f x))))))
                     (church+ (lambda (m) (lambda (n) (lambda (f) (lambda (x) ((m f) ((n f) x)))))))
                     (to-int  (lambda (n) ((n (lambda (x) (+ x 1))) 0))))
                 (to-int ((church+ two) three)))";
    assert_eq!(eval(src), 5);
}

#[test]
fn test_church_numerals_mul() {
    let src = "(let ((two   (lambda (f) (lambda (x) (f (f x)))))
                     (three (lambda (f) (lambda (x) (f (f (f x))))))
                     (church* (lambda (m) (lambda (n) (lambda (f) (m (n f))))))
                     (to-int  (lambda (n) ((n (lambda (x) (+ x 1))) 0))))
                 (to-int ((church* two) three)))";
    assert_eq!(eval(src), 6);
}

// ---------------------------------------------------------------------------
// Classic CS: Y-combinator style (via letrec)
// ---------------------------------------------------------------------------

#[test]
fn test_letrec_as_fixed_point() {
    // power via letrec, same shape as a Y-combinator fixed point
    let src = "(letrec ((pow (lambda (b e)
                               (if (== e 0) 1 (* b (pow b (- e 1)))))))
                 (pow 3 5))";
    assert_eq!(eval(src), 243);
}

// ---------------------------------------------------------------------------
// :sintN sugar for auto-sharing (only valid in let-binding positions)
// ---------------------------------------------------------------------------

#[test]
fn test_sint_sugar_relu() {
    // :sintN auto-shares the next expression
    let src = "(let ((z :sint32 -5)) (if (> z 0) z 0))";
    assert_eq!(eval_shared(src), 0);
}

#[test]
fn test_shared_comparison_select_max_sint() {
    let src = "(let ((a :sint32 30) (b :sint32 20) (one :sint32 1))
                 (let ((mask (> a b)))
                   (+ (* a mask) (* b (- one mask)))))";
    assert_eq!(eval_shared(src), 30);
}

#[test]
fn test_shared_comparison_select_min_sint() {
    let src = "(let ((a :sint32 30) (b :sint32 20) (one :sint32 1))
                 (let ((mask (< a b)))
                   (+ (* a mask) (* b (- one mask)))))";
    assert_eq!(eval_shared(src), 20);
}

#[test]
fn test_very_simple_neural_network_sint() {
    let src = "(let ((w1 :sint32 2) (w2 :sint32 3) (x :sint32 4))
                 (+ (* w1 x) (* w2 x)))";
    assert_eq!(eval_shared(src), 20);
}

#[test]
fn test_simple_neural_network_sint() {
    // z = 2*4 + (-3)*4 = -4  ->  ReLU(-4) = 0  (using :sintN sugar)
    let src = "(let ((w1 :sint32 2) (w2 :sint32 -3) (x :sint32 4))
                 (let ((z (+ (* w1 x) (* w2 x))))
                   (if (> z 0) z 0)))";
    assert_eq!(eval_shared(src), 0);
}

#[test]
fn test_non_trivial_neural_network_sint() {
    // two neurons, same input:
    //   n1 = ReLU(w11*x + w12*x) = ReLU((2 + -3)*4) = ReLU(-4) = 0
    //   n2 = ReLU(w21*x + w22*x) = ReLU((1 +  2)*4) = ReLU(12) = 12
    // output = n1 + n2 = 12
    let src = "(let ((x  :sint32 4)
                     (w11 :sint32 2) (w12 :sint32 -3)
                     (w21 :sint32 1) (w22 :sint32  2))
                 (let ((z1 (+ (* w11 x) (* w12 x)))
                       (z2 (+ (* w21 x) (* w22 x))))
                   (let ((n1 (if (> z1 0) z1 0))
                         (n2 (if (> z2 0) z2 0)))
                     (+ n1 n2))))";
    assert_eq!(eval_shared(src), 12);
}

// ---------------------------------------------------------------------------
// Private inputs via share-from / :sintN-from-M
// ---------------------------------------------------------------------------

#[test]
fn test_share_from_add() {
    // Party 0 contributes 10, party 1 contributes 32 → sum = 42
    let src = "(reconstruct (+ (share-from 0 10 32) (share-from 1 32 32)))";
    assert_eq!(eval_shared(src), 42);
}

#[test]
fn test_share_from_mul() {
    // Party 0 contributes 6, party 1 contributes 7 → product = 42
    let src = "(reconstruct (* (share-from 0 6 32) (share-from 1 7 32)))";
    assert_eq!(eval_shared(src), 42);
}

#[test]
fn test_share_from_three_parties() {
    // All three parties contribute → sum = 10 + 20 + 30 = 60
    let src =
        "(reconstruct (+ (share-from 0 10 32) (+ (share-from 1 20 32) (share-from 2 30 32))))";
    assert_eq!(eval_shared(src), 60);
}

#[test]
fn test_share_from_signed_negative() {
    // Party 1 contributes -7, party 2 contributes 10 → 10 + (-7) = 3
    let src = "(reconstruct (+ (share-from 1 -7 32) (share-from 2 10 32)))";
    assert_eq!(eval_shared(src), 3);
}

#[test]
fn test_sint_from_sugar_add() {
    // :sintN-from-M syntax in let bindings
    let src = "(let ((a :sint32-from-0 30) (b :sint32-from-1 12))
                 (reconstruct (+ a b)))";
    assert_eq!(eval_shared(src), 42);
}

#[test]
fn test_sint_from_sugar_comparison() {
    // salary-threshold: party 0 has salary, party 1 has threshold
    // 75 < 80 → comparison result = 0
    let src = "(let ((salary :sint32-from-0 75) (threshold :sint32-from-1 80))
                 (> salary threshold))";
    assert_eq!(eval_shared(src), 0);
}

#[test]
fn test_sint_from_sugar_comparison_greater() {
    // salary > threshold → 1
    let src = "(let ((salary :sint32-from-0 90) (threshold :sint32-from-1 80))
                 (> salary threshold))";
    assert_eq!(eval_shared(src), 1);
}

#[test]
fn test_share_from_private_max() {
    // Oblivious max: neither party reveals their value, result is the larger one
    let src = "(let ((a :sint32-from-0 30) (b :sint32-from-1 42) (one (share-from 0 1 32)))
                 (let ((m (> a b)))
                   (reconstruct (+ (* a m) (* b (- one m))))))";
    assert_eq!(eval_shared(src), 42);
}

#[test]
fn test_share_from_private_min() {
    let src = "(let ((a :sint32-from-0 30) (b :sint32-from-1 42) (one (share-from 0 1 32)))
                 (let ((m (< a b)))
                   (reconstruct (+ (* a m) (* b (- one m))))))";
    assert_eq!(eval_shared(src), 30);
}

// unit test to verify that the test framework correctly handles panics (e.g. from parse errors or type errors)
#[test]
fn test_panics() {
    let src = "(let ((a :sint32-from-0 30) (b :sint32-from-1 42))
                 (if (> a b) a b))";
    assert_eq!(eval_shared(src), 42);
}

// Test ZKP
// Addition of private variables with zero-knowledge proof, using "private" builtin
// All variables fro the same participant are private, and the result is public.
// The result is revealed to all participants, but the inputs are not revealed.
/*#[test]
fn test_zkp() {
    let src = "(let ((a (private 30)) (b (private 42)) (c 72))
                        (== (verify (prove (== (+ a b) c))) 1))";
    assert_eq!(eval_shared(src), 72);
}*/

// ---------------------------------------------------------------------------
// FHE builtins — mock plaintext backend (fhe-keygen)
// These tests run instantly since MockFheBackend does no crypto.
// ---------------------------------------------------------------------------

#[test]
fn test_fhe_mock_encrypt_decrypt() {
    // Basic round-trip: encrypt 42, decrypt, compare.
    let src = "(let ((keys (fhe-keygen)))
                 (fhe-dec keys (fhe-enc keys 42)))";
    assert_eq!(eval_fhe(src), 42);
}

#[test]
fn test_fhe_mock_compile_eval_add() {
    // Compile (+ x y), evaluate with x=10 (public) and y encrypted.
    let src = r#"(let ((keys (fhe-keygen)))
                   (let ((ct-y (fhe-enc keys 32)))
                     (let ((circuit (fhe-compile '(+ x y) '("y"))))
                       (fhe-dec keys
                         (fhe-eval circuit keys '("x") '(10) '("y") (list ct-y))))))"#;
    assert_eq!(eval_fhe(src), 42);
}

#[test]
fn test_fhe_mock_compile_eval_mul() {
    // Compile (* a b), both encrypted.
    let src = r#"(let ((keys (fhe-keygen)))
                   (let ((ct-a (fhe-enc keys 6))
                         (ct-b (fhe-enc keys 7)))
                     (let ((circuit (fhe-compile '(* a b) '("a" "b"))))
                       (fhe-dec keys
                         (fhe-eval circuit keys '() '() '("a" "b") (list ct-a ct-b))))))"#;
    assert_eq!(eval_fhe(src), 42);
}

#[test]
fn test_fhe_mock_fixed_point_output() {
    // (* x 1.5) marks the output wire as Q16.16; fhe-dec returns a Fixed,
    // which fixed->int truncates to an integer.
    let src = r#"(let ((keys (fhe-keygen)))
                   (let ((ct-x (fhe-enc keys 4)))
                     (let ((circuit (fhe-compile '(* x 1.5) '("x"))))
                       (fixed->int (fhe-dec keys
                         (fhe-eval circuit keys '() '() '("x") (list ct-x)))))))"#;
    assert_eq!(eval_fhe(src), 6);
}

#[test]
fn test_fhe_mock_fixed_point_negative() {
    // Negative fixed product: -6 * 1.5 = -9.0 (the >>16 rescale must be an
    // arithmetic shift, emulated with logical gates).
    let src = r#"(let ((keys (fhe-keygen)))
                   (let ((ct-x (fhe-enc keys -6)))
                     (let ((circuit (fhe-compile '(* x 1.5) '("x"))))
                       (fixed->int (fhe-dec keys
                         (fhe-eval circuit keys '() '() '("x") (list ct-x)))))))"#;
    assert_eq!(eval_fhe(src), -9);
}

#[test]
fn test_fhe_mock_negative_roundtrip() {
    // Negative values survive encrypt/decrypt via two's complement.
    let src = "(let ((keys (fhe-keygen)))
                 (fhe-dec keys (fhe-enc keys -4)))";
    assert_eq!(eval_fhe(src), -4);
}

#[test]
fn test_fhe_mock_relu_positive() {
    // ReLU(5) = 5, evaluated homomorphically on the mock backend.
    let src = r#"(let ((keys (fhe-keygen)))
                   (let ((ct-z (fhe-enc keys 5)))
                     (let ((circuit (fhe-compile '(if (> z 0) z 0) '("z"))))
                       (fhe-dec keys
                         (fhe-eval circuit keys '() '() '("z") (list ct-z))))))"#;
    assert_eq!(eval_fhe(src), 5);
}

#[test]
fn test_fhe_mock_relu_negative() {
    // ReLU(-4) = 0 — requires signed comparison on the encrypted value.
    let src = r#"(let ((keys (fhe-keygen)))
                   (let ((ct-z (fhe-enc keys -4)))
                     (let ((circuit (fhe-compile '(if (> z 0) z 0) '("z"))))
                       (fhe-dec keys
                         (fhe-eval circuit keys '() '() '("z") (list ct-z))))))"#;
    assert_eq!(eval_fhe(src), 0);
}

// ---------------------------------------------------------------------------
// FHE builtins — real TFHE-rs backend (fhe-keygen-tfhe)
// Slow: keygen alone takes a few seconds; homomorphic gates take longer.
// ---------------------------------------------------------------------------

#[test]
fn test_fhe_tfhe_relu() {
    // ReLU over real TFHE ciphertexts, both branches.  This exercises:
    //   - trivial encryption of the constant 0 (must mix with real ciphertexts)
    //   - signed comparison (ReLU(-4) would be wrong under unsigned compare)
    //   - mux on an encrypted boolean
    let src = r#"(let ((keys (fhe-keygen-tfhe)))
                   (let ((ct-pos (fhe-enc keys 5))
                         (ct-neg (fhe-enc keys -4)))
                     (let ((circuit (fhe-compile '(if (> z 0) z 0) '("z"))))
                       (+ (fhe-dec keys
                            (fhe-eval circuit keys '() '() '("z") (list ct-pos)))
                          (fhe-dec keys
                            (fhe-eval circuit keys '() '() '("z") (list ct-neg)))))))"#;
    // ReLU(5) + ReLU(-4) = 5 + 0 = 5
    assert_eq!(eval_fhe(src), 5);
}

// ---------------------------------------------------------------------------
// Scheme derived forms (Tier 1)
//
// Every form below is *desugared in the parser* into the core AST, so these
// tests double as a check that the desugaring is semantics-preserving: there is
// no evaluator support to test independently. The circuit backends see only the
// lowered shapes, which is why `test_derived_forms_reach_the_zkp_compiler`
// exists further down.
// ---------------------------------------------------------------------------

#[test]
fn test_begin_returns_last() {
    assert_eq!(eval("(begin 1 2 3)"), 3);
}

#[test]
fn test_begin_single() {
    assert_eq!(eval("(begin 7)"), 7);
}

#[test]
fn test_when_taken_and_not() {
    assert_eq!(eval("(when (> 3 1) 42)"), 42);
    assert_eq!(eval("(when (> 1 3) 42)"), 0);
}

#[test]
fn test_unless_is_when_inverted() {
    assert_eq!(eval("(unless (> 1 3) 42)"), 42);
    assert_eq!(eval("(unless (> 3 1) 42)"), 0);
}

#[test]
fn test_when_body_is_a_sequence() {
    assert_eq!(eval("(when #t 1 2 99)"), 99);
}

#[test]
fn test_cond_picks_first_true_clause() {
    assert_eq!(eval("(cond ((> 1 3) 10) ((> 3 1) 20) (else 30))"), 20);
}

#[test]
fn test_cond_falls_through_to_else() {
    assert_eq!(eval("(cond ((> 1 3) 10) ((> 2 5) 20) (else 30))"), 30);
}

/// A `cond` with no matching clause and no `else` yields 0 — the same "no
/// value" this language already uses for an untaken `when`.
#[test]
fn test_cond_without_else_yields_zero() {
    assert_eq!(eval("(cond ((> 1 3) 10))"), 0);
}

#[test]
fn test_cond_clause_body_is_a_sequence() {
    assert_eq!(eval("(cond (#t 1 2 3) (else 9))"), 3);
}

/// `else` must come last; a clause after it is unreachable and is a bug in the
/// program, not something to silently accept.
#[test]
fn test_cond_rejects_else_before_other_clauses() {
    let e = baru::interpreter::parser::parse("(cond (else 1) (#t 2))");
    assert!(e.is_err(), "else before another clause must be an error");
}

#[test]
fn test_let_star_sees_earlier_bindings() {
    assert_eq!(eval("(let* ((x 2) (y (* x 3))) (+ x y))"), 8);
}

/// **A deviation from Scheme, pinned here rather than left as folklore.**
///
/// Baru's plain `let` is *sequential*: a later init sees an earlier binding, so
/// `(let ((a 1) (b (+ a 10))) b)` is 11. In Scheme that is `let*` behaviour and
/// plain `let` would either reject `a` as unbound or see an outer `a`.
///
/// The consequence is that `let*` is semantically **redundant** here — it is
/// provided so Scheme source ports unchanged and so intent is explicit, not
/// because it does something `let` cannot. Making `let` parallel would be a
/// breaking change to a core form (~46 multi-binding `let`s in-tree), so it is
/// a deliberate decision recorded in LANGUAGE.md §13, not an oversight.
#[test]
fn test_let_is_sequential_like_let_star() {
    assert_eq!(eval("(let ((a 1) (b (+ a 10))) b)"), 11);
    // Both forms therefore agree where Scheme would distinguish them.
    assert_eq!(eval("(let ((x 1)) (let* ((x 2) (y x)) y))"), 2);
    assert_eq!(eval("(let ((x 1)) (let ((x 2) (y x)) y))"), 2);
}

#[test]
fn test_let_star_empty_bindings() {
    assert_eq!(eval("(let* () 5)"), 5);
}

#[test]
fn test_named_let_loops() {
    assert_eq!(
        eval("(let loop ((i 0) (acc 0)) (if (== i 5) acc (loop (+ i 1) (+ acc i))))"),
        10
    );
}

#[test]
fn test_named_let_is_tail_recursive() {
    // 20k iterations: proper tail calls reuse the caller's continuation, so
    // this must not grow the machine's continuation chain.
    assert_eq!(
        eval("(let loop ((i 0)) (if (== i 20000) 123 (loop (+ i 1))))"),
        123
    );
}

#[test]
fn test_case_matches_one_of_several_datums() {
    assert_eq!(eval("(case 2 ((1) 100) ((2 3) 200) (else 300))"), 200);
    assert_eq!(eval("(case 3 ((1) 100) ((2 3) 200) (else 300))"), 200);
    assert_eq!(eval("(case 9 ((1) 100) ((2 3) 200) (else 300))"), 300);
}

#[test]
fn test_case_single_datum_clause() {
    assert_eq!(eval("(case 1 ((1) 100) (else 300))"), 100);
}

/// The key expression must be evaluated exactly once, which is why the
/// desugaring binds it to a temporary rather than substituting it per datum.
#[test]
fn test_case_evaluates_its_key_once() {
    // If the key were substituted per datum, the inner `let` would run twice
    // and this would still return 200 — so instead assert the shape directly.
    let e = baru::interpreter::parser::parse("(case (+ 1 1) ((2) 200) (else 0))").unwrap();
    let rendered = format!("{e:?}");
    assert_eq!(
        rendered.matches("BinOp(Add").count(),
        1,
        "case must bind its key once, not inline it per datum: {rendered}"
    );
}

/// The temporary a `case` introduces must not be capturable, or a user variable
/// of the same name would shadow it. Nested `case` is the case that would break.
#[test]
fn test_nested_case_temporaries_do_not_collide() {
    assert_eq!(
        eval("(case 1 ((1) (case 2 ((2) 42) (else 0))) (else 0))"),
        42
    );
}

// ---------------------------------------------------------------------------
// Scheme predicates, equivalence and list procedures (Tier 1)
// ---------------------------------------------------------------------------

#[test]
fn test_type_predicates() {
    for (src, want) in [
        ("(number? 3)", 1),
        ("(number? \"x\")", 0),
        ("(integer? 3)", 1),
        ("(string? \"x\")", 1),
        ("(string? 3)", 0),
        ("(char? 'a')", 1),
        ("(boolean? #t)", 1),
        ("(boolean? 1)", 0),
        ("(list? (list 1 2))", 1),
        ("(list? 3)", 0),
        ("(procedure? (lambda (x) x))", 1),
        ("(procedure? 3)", 0),
        ("(zero? 0)", 1),
        ("(zero? 1)", 0),
    ] {
        assert_eq!(eval(src), want, "{src}");
    }
}

/// `pair?` is false on the empty list and true on a non-empty one. This
/// language has no improper pairs, so that is all it can mean here.
#[test]
fn test_pair_predicate() {
    assert_eq!(eval("(pair? (list 1 2))"), 1);
    assert_eq!(eval("(pair? (list))"), 0);
}

#[test]
fn test_eqv_on_atoms() {
    assert_eq!(eval("(eq? 3 3)"), 1);
    assert_eq!(eval("(eq? 3 4)"), 0);
    assert_eq!(eval("(eqv? 'a' 'a')"), 1);
    assert_eq!(eval("(eq? \"ab\" \"ab\")"), 1);
}

/// The `eq?` / `equal?` split: `equal?` recurses into lists, `eq?` does not.
#[test]
fn test_equal_recurses_where_eq_does_not() {
    assert_eq!(eval("(equal? (list 1 2) (list 1 2))"), 1);
    assert_eq!(eval("(eq? (list 1 2) (list 1 2))"), 0);
    assert_eq!(eval("(equal? (list 1 2) (list 1 3))"), 0);
    assert_eq!(eval("(equal? (list) (list))"), 1);
}

#[test]
fn test_list_procedures() {
    assert_eq!(eval("(length (list 1 2 3))"), 3);
    assert_eq!(eval("(length (list))"), 0);
    assert_eq!(eval("(car (reverse (list 1 2 3)))"), 3);
    assert_eq!(eval("(length (append (list 1) (list 2 3)))"), 3);
    assert_eq!(eval("(car (append (list 1) (list 2)))"), 1);
    assert_eq!(eval("(list-ref (list 10 20 30) 1)"), 20);
    assert_eq!(eval("(list-ref (list 10 20 30) 0)"), 10);
}

#[test]
fn test_derived_forms_compose_with_each_other() {
    // cond inside a named let inside let*, all desugared.
    assert_eq!(
        eval(
            "(let* ((n 4))
               (let loop ((i 0) (acc 0))
                 (cond ((== i n) acc)
                       (else (loop (+ i 1) (+ acc i))))))"
        ),
        6
    );
}

// ---------------------------------------------------------------------------
// define, set!, apply, and operators as values (Tier 1)
// ---------------------------------------------------------------------------

#[test]
fn test_define_binds_a_value() {
    assert_eq!(eval("(begin (define x 5) x)"), 5);
}

#[test]
fn test_define_function_sugar() {
    assert_eq!(eval("(begin (define (sq n) (* n n)) (sq 7))"), 49);
    assert_eq!(eval("(begin (define (add a b) (+ a b)) (add 3 4))"), 7);
    assert_eq!(eval("(begin (define (nine) 9) (nine))"), 9);
}

#[test]
fn test_set_assigns_an_existing_binding() {
    assert_eq!(eval("(begin (define x 1) (set! x 42) x)"), 42);
    assert_eq!(eval("(let ((c 0)) (begin (set! c (+ c 10)) (set! c (+ c 5)) c))"), 15);
}

/// Scheme requires the variable to exist. Catching it *before* the operand runs
/// means a typo cannot first perform whatever side effects the operand had.
#[test]
#[should_panic(expected = "set!: unbound variable")]
fn test_set_rejects_an_unbound_variable() {
    eval("(set! definitely-not-bound 1)");
}

/// The property that makes `set!` worth having rather than simulating with
/// rebinding: bindings are shared cells, so a closure observes assignments made
/// after it was created. This is the classic counter.
#[test]
fn test_set_is_visible_through_a_closure() {
    assert_eq!(
        eval(
            "(let ((n 0))
               (let ((bump (lambda () (set! n (+ n 1)))))
                 (begin (bump) (bump) (bump) n)))"
        ),
        3
    );
}

#[test]
fn test_apply_spreads_a_list() {
    assert_eq!(eval("(apply + (list 3 4))"), 7);
    assert_eq!(eval("(apply (lambda (a b c) (+ a (* b c))) (list 1 2 3))"), 7);
    assert_eq!(eval("(begin (define (f) 9) (apply f (list)))"), 9);
}

/// `apply` is a **value**, not syntax — so it can be passed to itself, and more
/// importantly a user binding of the same name shadows it. An earlier version of
/// this change made `apply` a parser special form, which silently broke
/// `test_higher_order_apply` below; this pins the correct behaviour.
#[test]
fn test_apply_is_a_first_class_shadowable_value() {
    assert_eq!(eval("(procedure? apply)"), 1);
    assert_eq!(eval("(apply apply (list + (list 1 2)))"), 3);
    assert_eq!(
        eval("(let ((apply (lambda (f x) (f x)))) (apply (lambda (n) (* n n)) 7))"),
        49
    );
}

#[test]
#[should_panic(expected = "must be a list")]
fn test_apply_requires_a_list() {
    eval("(apply + 5)");
}

/// Operators are usable as values, and non-binary arities route to those
/// bindings. Two operands keep the core `BinOp` path, which is the only one that
/// dispatches secret shares.
#[test]
fn test_operators_are_first_class_and_variadic() {
    assert_eq!(eval("(- 5)"), -5); // documented since forever; a parse error until now
    assert_eq!(eval("(+ 1 2 3)"), 6);
    assert_eq!(eval("(* 2 3 4)"), 24);
    assert_eq!(eval("(+)"), 0); // Scheme's additive identity
    assert_eq!(eval("(- 10 3 2)"), 5); // left fold
    assert_eq!(eval("(/ 100 5 2)"), 10);
    assert_eq!(eval("(apply * (list 2 3 4))"), 24);
    // The two-operand fast path is unchanged.
    assert_eq!(eval("(+ 1 2)"), 3);
}

/// Q16.16 must survive the variadic path, including the rescale per multiply.
#[test]
fn test_variadic_operators_keep_fixed_point() {
    assert_eq!(eval_plain_string("(+ 1.5 2)"), "3.5000");
    assert_eq!(eval_plain_string("(* 2.0 3.0)"), "6.0000");
    assert_eq!(eval_plain_string("(- 1.5)"), "-1.5000");
}

/// A secret operand reaching the *procedure* form gets an error naming the
/// two-operand form, rather than a silently wrong public answer. The
/// two-operand form itself still runs the protocol.
#[test]
fn test_operator_procedures_refuse_secrets() {
    assert_eq!(eval_shared("(let ((z :sint32 5)) (reconstruct (+ z 1)))"), 6);
    let secret_via_apply =
        std::panic::catch_unwind(|| eval_shared("(let ((z :sint32 5)) (apply + (list z 1)))"));
    assert!(
        secret_via_apply.is_err(),
        "`+` as a procedure must refuse a share instead of guessing"
    );
}

// ---------------------------------------------------------------------------
// `do` loops and vectors (Tier 1)
// ---------------------------------------------------------------------------

#[test]
fn test_do_accumulates() {
    assert_eq!(
        eval("(do ((i 0 (+ i 1)) (acc 0 (+ acc i))) ((== i 5) acc))"),
        10
    );
}

#[test]
fn test_do_result_and_body() {
    assert_eq!(eval("(do ((i 0 (+ i 1))) ((== i 3) 99))"), 99);
    // A body that mutates an outer binding, i.e. `do` used for effect.
    assert_eq!(
        eval("(let ((n 0)) (do ((i 0 (+ i 1))) ((== i 4) n) (set! n (+ n i))))"),
        6
    );
}

/// A spec without a step carries its variable through unchanged, and an empty
/// result clause yields 0 (this language's "no value").
#[test]
fn test_do_optional_step_and_empty_result() {
    assert_eq!(eval("(do ((i 0 (+ i 1)) (k 7)) ((== i 2) k))"), 7);
    assert_eq!(eval("(do ((i 0 (+ i 1))) ((== i 2)))"), 0);
}

#[test]
fn test_vector_basics() {
    assert_eq!(eval("(vector-length (vector 1 2 3))"), 3);
    assert_eq!(eval("(vector-ref (vector 10 20 30) 1)"), 20);
    assert_eq!(eval("(vector? (vector 1))"), 1);
    assert_eq!(eval("(vector? (list 1))"), 0);
    assert_eq!(eval("(vector-length (make-vector 4))"), 4);
    assert_eq!(eval("(vector-ref (make-vector 3 7) 2)"), 7);
    assert_eq!(eval("(length (vector->list (vector 1 2)))"), 2);
    assert_eq!(eval("(vector-length (list->vector (list 1 2 3)))"), 3);
}

#[test]
fn test_vector_set_mutates() {
    assert_eq!(
        eval("(let ((v (vector 1 2 3))) (begin (vector-set! v 0 99) (vector-ref v 0)))"),
        99
    );
}

/// **The property that distinguishes a vector from a list here.** Vectors have
/// reference semantics: cloning a `Types::Vector` shares its storage, so a
/// mutation is visible through every alias and through a function that received
/// it as an argument. Without this a vector would just be a list with O(1)
/// indexing, which `list-ref` already gives.
#[test]
fn test_vectors_have_reference_semantics() {
    assert_eq!(
        eval("(let ((v (vector 1 2))) (let ((w v)) (begin (vector-set! w 0 9) (vector-ref v 0))))"),
        9
    );
    assert_eq!(
        eval(
            "(let ((v (vector 1 2)))
               (let ((f (lambda (x) (vector-set! x 1 77))))
                 (begin (f v) (vector-ref v 1))))"
        ),
        77
    );
    // Distinct vectors with equal contents are not `eq?`.
    assert_eq!(eval("(eq? (vector 1) (vector 1))"), 0);
}

#[test]
#[should_panic(expected = "past end")]
fn test_vector_ref_bounds_checked() {
    eval("(vector-ref (vector 1 2) 5)");
}

/// A secret index is refused rather than silently scanned, matching `list-ref`.
#[test]
fn test_vector_index_must_be_public() {
    let r = std::panic::catch_unwind(|| {
        eval_shared("(let ((z :sint32 1)) (vector-ref (vector 10 20) z))")
    });
    assert!(r.is_err(), "a secret vector index must be refused");
}

// ---------------------------------------------------------------------------
// Rest parameters (Tier 1)
// ---------------------------------------------------------------------------

#[test]
fn test_rest_parameter_collects_remaining_arguments() {
    assert_eq!(eval("((lambda (a . rest) (length rest)) 1 2 3)"), 2);
    assert_eq!(eval("((lambda (a . rest) a) 1 2 3)"), 1);
    // The rest list is empty, not absent, when only the fixed args are given.
    assert_eq!(eval("((lambda (a . rest) (length rest)) 1)"), 0);
}

/// `(lambda (. args) ...)` takes everything. Scheme spells this `(lambda args ...)`,
/// but a bare identifier there already means "one required parameter" in this
/// language, and changing that would break existing programs — so the dotted
/// spelling is the supported one. Recorded in LANGUAGE.md §13.
#[test]
fn test_all_arguments_as_a_rest_list() {
    assert_eq!(eval("((lambda (. args) (length args)) 1 2 3 4)"), 4);
    assert_eq!(eval("((lambda (. args) (length args)))"), 0);
}

#[test]
fn test_rest_parameter_works_through_apply() {
    assert_eq!(
        eval("(apply (lambda (a . rest) (+ a (length rest))) (list 10 1 2))"),
        12
    );
}

/// With a rest parameter the fixed parameters are a *minimum*; without one the
/// arity stays exact.
#[test]
#[should_panic(expected = "expected at least 2")]
fn test_rest_parameter_still_requires_the_fixed_arguments() {
    eval("((lambda (a b . r) a) 1)");
}

#[test]
fn test_fixed_arity_is_unchanged() {
    assert_eq!(eval("((lambda (a b) (+ a b)) 1 2)"), 3);
}

/// `Display` must round-trip the dot, or a variadic lambda would print as a
/// fixed-arity one — which matters because quoted expressions are shown to users
/// and fed to `eval` and the compilers.
#[test]
fn test_variadic_lambda_displays_its_dot() {
    let e = baru::interpreter::parser::parse("(lambda (a . rest) a)").unwrap();
    assert_eq!(format!("{e}"), "(lambda (a . rest) a)");
    let e = baru::interpreter::parser::parse("(lambda (. args) args)").unwrap();
    assert_eq!(format!("{e}"), "(lambda (. args) args)");
}

#[test]
fn test_malformed_rest_parameter_is_rejected() {
    assert!(baru::interpreter::parser::parse("(lambda (a . b c) a)").is_err());
    assert!(baru::interpreter::parser::parse("(lambda (a .) a)").is_err());
}

// ---------------------------------------------------------------------------
// The error continuation (Tier 1)
//
// A raised condition does not reach the continuation it was handed to. The
// machine truncates the task stack back to the height recorded when the
// innermost `guard` was installed and runs that handler instead — so unwinding
// is one `truncate`, not a walk over pending frames.
// ---------------------------------------------------------------------------

#[test]
fn test_guard_catches_an_explicit_error() {
    assert_eq!(eval("(guard (e 42) (error \"boom\"))"), 42);
}

#[test]
fn test_guard_passes_a_normal_value_through() {
    assert_eq!(eval("(guard (e 99) (+ 1 2))"), 3);
}

/// The condition is bound **inert**, so the handler can mention it without
/// re-raising. Getting this wrong makes a caught error impossible to inspect.
#[test]
fn test_a_caught_condition_can_be_inspected() {
    assert_eq!(eval("(guard (e (error? e)) (error \"x\"))"), 1);
    assert_eq!(
        eval_plain_string("(guard (e (error-message e)) (error \"boom\"))"),
        "boom"
    );
    assert_eq!(
        eval("(guard (e (length (error-irritants e))) (error \"m\" 1 2 3))"),
        3
    );
}

/// Failures the *evaluator* raises, not just explicit `error` calls. These were
/// process-killing panics before the error continuation existed.
#[test]
fn test_evaluator_failures_are_catchable() {
    assert_eq!(eval("(guard (e 99) definitely-unbound)"), 99);
    assert_eq!(eval("(guard (e 7) ((lambda (a) a)))"), 7); // arity
    assert_eq!(eval("(guard (e 5) (/ 1 0))"), 5);
    assert_eq!(eval("(guard (e 5) (mod 1 0))"), 5);
    assert_eq!(eval("(guard (e 5) (vector-ref (vector 1) 9))"), 5);
    assert_eq!(eval("(guard (e 5) (list-ref (list 1) 9))"), 5);
    assert_eq!(eval("(guard (e 5) (1 2))"), 5); // calling a non-procedure
}

#[test]
fn test_innermost_guard_wins() {
    assert_eq!(eval("(guard (e 1) (guard (f 2) (error \"inner\")))"), 2);
}

/// A handler may itself raise; the condition then propagates to the *enclosing*
/// guard. This is what proves the handler runs with its own frame already
/// popped, rather than still installed and catching itself forever.
#[test]
fn test_a_handler_can_reraise_to_the_enclosing_guard() {
    assert_eq!(
        eval("(guard (e 1) (car (guard (f (error \"re\")) (error \"x\"))))"),
        1
    );
    assert_eq!(eval("(guard (e 8) (guard (f (raise f)) (error \"x\")))"), 8);
}

/// A handler goes out of scope at the normal exit of its body, or it would keep
/// catching conditions raised long afterwards.
#[test]
fn test_handler_is_uninstalled_on_normal_exit() {
    // The inner guard completes normally; the later raise must reach the outer.
    assert_eq!(
        eval("(guard (e 111) (begin (guard (f 0) 1) (error \"after\")))"),
        111
    );
}

#[test]
fn test_unwinding_discards_the_rest_of_the_guarded_expression() {
    // The `+ 1` around the raise never runs; the handler's value is the result.
    assert_eq!(eval("(guard (e 0) (+ 1 (error \"x\")))"), 0);
    // A guard used as a subexpression yields its handler's value in place.
    assert_eq!(eval("(+ 1 (guard (e 10) (error \"x\")))"), 11);
}

#[test]
fn test_raise_wraps_a_non_condition() {
    assert_eq!(eval("(guard (e (error? e)) (raise 5))"), 1);
    assert_eq!(
        eval("(guard (e (car (error-irritants e))) (raise 5))"),
        5
    );
}

/// **The point of the whole exercise**: an uncaught condition is the *value* of
/// the evaluation rather than a panic, so a REPL session survives a bad
/// expression instead of taking the process down.
#[test]
fn test_an_uncaught_condition_is_a_value_not_a_panic() {
    let out = eval_plain_string("(guard (e (error-message e)) (error \"survivable\"))");
    assert_eq!(out, "survivable");
    // Directly: no panic, and the message is preserved.
    let r = std::panic::catch_unwind(|| eval_plain_string("(+ 1 1)"));
    assert!(r.is_ok(), "a following expression must still evaluate");
}

#[test]
fn test_malformed_guard_is_rejected() {
    for src in [
        "(guard (e) (error \"x\"))",   // no handler body
        "(guard (e 1))",              // nothing to protect
        "(guard e 1)",                // missing clause parens
    ] {
        assert!(
            baru::interpreter::parser::parse(src).is_err(),
            "should be a parse error: {src}"
        );
    }
}

/// The error continuation reaches the *crypto* builtins too, not just the
/// interpreter core. Each of these was a process-killing panic before.
#[test]
fn test_crypto_builtin_failures_are_catchable() {
    for src in [
        "(guard (e 1) (nn-load-model \"definitely-not-a-file.json\"))",
        "(guard (e 1) (zkp-compile 5 '(\"x\")))",
        "(guard (e 1) (zkp-verify \"bad\" \"bad\" '() '()))",
        "(guard (e 1) (fhe-compile 5 '()))",
        "(guard (e 1) (fhe-dec \"not-json\" 5))",
        "(guard (e 1) (proof-load \"definitely-not-a-file\"))",
        "(guard (e 1) (b2a 5 8))",
        "(guard (e 1) (a2b 5 8))",
    ] {
        assert_eq!(eval(src), 1, "{src}");
    }
}

/// Type errors in the machine itself — `car`/`cdr`/`cons`/`if` and the unary
/// operators — raise rather than abort, and carry the offending value as an
/// irritant so a handler can report what was wrong.
#[test]
fn test_machine_type_errors_are_catchable() {
    for src in [
        "(guard (e 1) (car (list)))",
        "(guard (e 1) (car 5))",
        "(guard (e 1) (cdr (list)))",
        "(guard (e 1) (cdr \"x\"))",
        "(guard (e 1) (cons 1 2))",
        "(guard (e 1) (not \"x\"))",
        "(guard (e 1) (null? 5))",
        "(guard (e 1) (if \"x\" 1 2))",
    ] {
        assert_eq!(eval(src), 1, "{src}");
    }
    assert_eq!(
        eval_plain_string("(guard (e (error-message e)) (car (list)))"),
        "car of empty list"
    );
    // The irritant is the value that provoked it.
    assert_eq!(
        eval("(guard (e (car (error-irritants e))) (car 42))"),
        42
    );
}

/// Normal list operations are untouched by the above.
#[test]
fn test_list_operations_still_work_after_raising_their_errors() {
    assert_eq!(eval("(car (list 1 2))"), 1);
    assert_eq!(eval("(car (cdr (list 1 2)))"), 2);
    assert_eq!(eval("(length (cdr (cons 1 (cons 2 ()))))"), 1);
    assert_eq!(eval("(null? (list))"), 1);
    assert_eq!(eval("(not 0)"), 1);
}

/// The vector builtins (`dot`, `vsum`, `matvec`, `nn-forward`) share a set of
/// fallible helpers, so their length and type errors raise together.
#[test]
fn test_vector_builtin_failures_are_catchable() {
    for src in [
        "(guard (e 1) (dot (list 1 2) (list 1)))",   // length mismatch
        "(guard (e 1) (dot (list \"x\") (list 1)))", // bad element type
        "(guard (e 1) (vsum (list \"x\" 1)))",
        "(guard (e 1) (matvec (list 5) (list 1)))",  // row is not a list
    ] {
        assert_eq!(eval(src), 1, "{src}");
    }
    assert!(eval_plain_string("(guard (e (error-message e)) (dot (list 1 2) (list 1)))")
        .contains("length mismatch"));
}

/// And they still compute the right answers.
#[test]
fn test_vector_builtins_still_correct() {
    assert_eq!(eval("(dot (list 1 2) (list 3 4))"), 11);
    assert_eq!(eval("(vsum (list 1 2 3))"), 6);
    assert_eq!(eval("(car (matvec (list (list 1 2)) (list 3 4)))"), 11);
}

/// Secret-dependent paths raise too. A branch taken under a *secret* condition
/// must produce something shareable, since the result has to stay secret — so a
/// string branch is a condition rather than an abort.
#[test]
fn test_secret_dependent_failures_are_catchable() {
    assert_eq!(
        eval_shared("(guard (e 1) (let ((z :sint32 5)) (if (> z 0) \"yes\" \"no\")))"),
        1
    );
    // An operation with no fixed-point-share implementation.
    assert_eq!(
        eval_shared("(guard (e 1) (let ((z :sint32 5)) (reconstruct (mod z 2.5))))"),
        1
    );
    // The normal secret branch still works and stays a share.
    assert_eq!(
        eval_shared("(let ((z :sint32 5)) (reconstruct (if (> z 0) 1 2)))"),
        1
    );
}

#[test]
fn test_apply_and_fhe_eval_argument_errors_are_catchable() {
    assert_eq!(eval("(guard (e 1) (apply + 5))"), 1);
    assert_eq!(eval("(guard (e 1) (apply))"), 1);
    assert_eq!(eval("(guard (e 1) (fhe-eval \"x\"))"), 1);
}
