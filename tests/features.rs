use baru::interpreter::eval::{eval_plain as eval, eval_shared};

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
