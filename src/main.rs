// Minimal distributed Lisp-like evaluator for MPC (ABY3 backend simulation)
// - Supports only NUMs for now
// - Implements `apply_continuation` and `reduce_expression`
// - Simulates 3 participants running a distributed REPL where constants are evaluated locally

#[derive(Debug, Clone, Copy)]
enum Expr {
    Num(i64),
}

#[derive(Debug, Clone, Copy)]
/// A very small representation of a secret-shared numeric value.
/// For this minimal example we keep it simple: the value is the same
/// on every participant (no communication required for constants).
struct SharedValue {
    value: i64,
}

#[derive(Debug)]
enum Continuation {
    /// Terminal continuation: evaluation is finished and the contained value
    /// should be returned to the caller / REPL.
    Halt,
}

/// Apply a continuation to a value. In a full implementation this would
/// thread the control flow (and potentially schedule further reductions).
/// For NUM constants and the `Halt` continuation we simply return the value.
fn apply_continuation(cont: &Continuation, val: SharedValue) -> SharedValue {
    match cont {
        Continuation::Halt => val,
    }
}

/// Reduce an expression under a continuation. For this minimal example
/// NUM constants evaluate to themselves wrapped in a SharedValue and then
/// the continuation is applied.
fn reduce_expression(expr: &Expr, cont: &Continuation) -> SharedValue {
    match *expr {
        Expr::Num(n) => {
            let v = SharedValue { value: n };
            apply_continuation(cont, v)
        }
    }
}

/// A simulated participant in the 3-party REPL. In a real MPC system each
/// participant would only have a share of values and would communicate to
/// evaluate richer expressions. Here we simulate the distributed REPL by
/// running the same local evaluator on each participant for constants.
struct Participant {
    id: usize,
}

impl Participant {
    fn new(id: usize) -> Self {
        Participant { id }
    }

    /// Evaluate an expression locally and return the resulting SharedValue.
    fn eval(&self, expr: &Expr) -> SharedValue {
        // For this minimal example there is a single terminal continuation.
        let cont = Continuation::Halt;
        let result = reduce_expression(expr, &cont);
        // In a real MPC run we'd return the participant's share here. For
        // constants we keep the same concrete value so participants agree.
        // Use the concrete field so the compiler knows it's read (avoids dead_code warning).
        println!("participant {} evaluated expression -> {}", self.id, result.value);
        result
    }
}

fn main() {
    // Example: evaluate the constant 42 on three participants
    let expr = Expr::Num(42);

    let participants: Vec<Participant> = (0..3).map(Participant::new).collect();

    // Collect concrete values (reads SharedValue.value to avoid dead_code warning)
    let mut results: Vec<i64> = Vec::new();
    for p in &participants {
        let r = p.eval(&expr);
        results.push(r.value);
    }

    // In this toy example all participants produce the same numeric value.
    println!("final results from participants: {:?}", results);
}
