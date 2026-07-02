// Circuit Compiler for Baru
// Converts Expr AST into ZKP circuit constraints
// Handles variable extraction, witness classification, and gate generation

use crate::interpreter::ast::Expr;

/// Represents a single constraint in the circuit
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Linear constraint: a*x + b*y + c = 0
    Linear {
        var_x: String,
        coeff_x: i64,
        var_y: String,
        coeff_y: i64,
        constant: i64,
    },
    /// Multiplication constraint: x * y = z
    Multiply {
        var_x: String,
        var_y: String,
        var_z: String,
    },
    /// Addition gate: x + y = z
    Add {
        var_x: String,
        var_y: String,
        var_z: String,
    },
    /// Assignment: x = constant
    Assign { var: String, value: i64 },
}

/// Result of circuit compilation
#[derive(Debug)]
pub struct CompiledCircuit {
    pub constraints: Vec<Constraint>,
    pub witnesses: Vec<String>,
    pub public_inputs: Vec<String>,
    pub num_gates: usize,
}

/// Circuit compiler
pub struct CircuitCompiler {
    var_counter: usize,
    witnesses: Vec<String>,
    public_inputs: Vec<String>,
}

impl Default for CircuitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitCompiler {
    pub fn new() -> Self {
        CircuitCompiler {
            var_counter: 0,
            witnesses: Vec::new(),
            public_inputs: Vec::new(),
        }
    }

    /// Generate a unique temporary variable name
    fn fresh_var(&mut self) -> String {
        let name = format!("_t{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    /// Compile an expression to constraints
    pub fn compile(
        &mut self,
        expr: &Expr,
        private_vars: &[String],
    ) -> Result<CompiledCircuit, String> {
        let mut constraints = Vec::new();

        // Extract all variables from the expression
        let all_vars = self.extract_variables(expr);

        // Classify variables: private vs public
        for var in &all_vars {
            if private_vars.contains(var) {
                self.witnesses.push(var.clone());
            } else {
                self.public_inputs.push(var.clone());
            }
        }

        // Compile expression to constraints
        let _result_var = self.compile_expr(expr, &mut constraints)?;

        Ok(CompiledCircuit {
            constraints,
            witnesses: self.witnesses.clone(),
            public_inputs: self.public_inputs.clone(),
            num_gates: 0, // TODO: count gates
        })
    }

    /// Extract all variable names from an expression
    fn extract_variables(&self, expr: &Expr) -> Vec<String> {
        let mut vars = Vec::new();
        self.extract_vars_helper(expr, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn extract_vars_helper(&self, expr: &Expr, vars: &mut Vec<String>) {
        match expr {
            Expr::Var(name) => {
                vars.push(name.clone());
            }
            Expr::Let(bindings, body) => {
                for (name, e) in bindings {
                    vars.push(name.clone());
                    self.extract_vars_helper(e, vars);
                }
                for e in body {
                    self.extract_vars_helper(e, vars);
                }
            }
            Expr::LetRec(bindings, body) => {
                for (name, e) in bindings {
                    vars.push(name.clone());
                    self.extract_vars_helper(e, vars);
                }
                for e in body {
                    self.extract_vars_helper(e, vars);
                }
            }
            Expr::BinOp(_, left, right) => {
                self.extract_vars_helper(left, vars);
                self.extract_vars_helper(right, vars);
            }
            Expr::UnOp(_, expr) => {
                self.extract_vars_helper(expr, vars);
            }
            Expr::Call(func, args) => {
                self.extract_vars_helper(func, vars);
                for arg in args {
                    self.extract_vars_helper(arg, vars);
                }
            }
            Expr::If(cond, then_e, else_e) => {
                self.extract_vars_helper(cond, vars);
                self.extract_vars_helper(then_e, vars);
                self.extract_vars_helper(else_e, vars);
            }
            Expr::Cons(left, right) => {
                self.extract_vars_helper(left, vars);
                self.extract_vars_helper(right, vars);
            }
            Expr::Car(expr) | Expr::Cdr(expr) => {
                self.extract_vars_helper(expr, vars);
            }
            Expr::Quote(expr) | Expr::Eval(expr) | Expr::Private(expr) => {
                self.extract_vars_helper(expr, vars);
            }
            Expr::Seq(exprs) => {
                for e in exprs {
                    self.extract_vars_helper(e, vars);
                }
            }
            // Terminal expressions
            Expr::Num(_) | Expr::Str(_) | Expr::Char(_) | Expr::Bool(_) | Expr::Lambda(_, _) => {
                // No variables
            }
        }
    }

    /// Compile an expression to constraints, returning the result variable
    fn compile_expr(
        &mut self,
        expr: &Expr,
        constraints: &mut Vec<Constraint>,
    ) -> Result<String, String> {
        match expr {
            // Numbers: constant assignments
            Expr::Num(n) => {
                let var = self.fresh_var();
                constraints.push(Constraint::Assign {
                    var: var.clone(),
                    value: *n,
                });
                Ok(var)
            }

            // Variables: just return the name
            Expr::Var(name) => Ok(name.clone()),

            // Binary operations
            Expr::BinOp(op, left, right) => {
                let left_var = self.compile_expr(left, constraints)?;
                let right_var = self.compile_expr(right, constraints)?;

                use crate::interpreter::parser::BinOp;
                match op {
                    BinOp::Add => {
                        let result = self.fresh_var();
                        constraints.push(Constraint::Add {
                            var_x: left_var,
                            var_y: right_var,
                            var_z: result.clone(),
                        });
                        Ok(result)
                    }
                    BinOp::Mul => {
                        let result = self.fresh_var();
                        constraints.push(Constraint::Multiply {
                            var_x: left_var,
                            var_y: right_var,
                            var_z: result.clone(),
                        });
                        Ok(result)
                    }
                    // Other operations would be compiled to multiple gates
                    _ => Err(format!("Operation {:?} not yet supported in ZKP", op)),
                }
            }

            // Private annotations: compile inner expression (annotation is metadata)
            Expr::Private(inner) => self.compile_expr(inner, constraints),

            // Not yet implemented
            _ => Err("Expression type not yet supported in ZKP circuit compilation".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_variables() {
        let compiler = CircuitCompiler::new();
        let expr = Expr::BinOp(
            crate::interpreter::parser::BinOp::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );

        let vars = compiler.extract_variables(&expr);
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_compile_simple_num() {
        let mut compiler = CircuitCompiler::new();
        let expr = Expr::Num(42);

        let result = compiler.compile(&expr, &[]);
        assert!(result.is_ok());

        let circuit = result.unwrap();
        assert_eq!(circuit.constraints.len(), 1);
    }

    #[test]
    fn test_compile_add() {
        let mut compiler = CircuitCompiler::new();
        let expr = Expr::BinOp(
            crate::interpreter::parser::BinOp::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );

        let result = compiler.compile(&expr, &["x".to_string()]);
        assert!(result.is_ok());

        let circuit = result.unwrap();
        assert_eq!(circuit.witnesses, vec!["x"]);
        assert_eq!(circuit.public_inputs, vec!["y"]);
    }
}
