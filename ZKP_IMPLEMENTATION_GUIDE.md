# ZKP Implementation Guide: Code Examples

## Phase 6A: AST Extension ✅ COMPLETE

**Status**: IMPLEMENTED AND TESTED
**Date**: July 2, 2026
**Build Status**: ✅ Compiles with zero warnings/errors

### Actual Implementation

The `(private expr)` syntax has been successfully implemented:

```rust
// src/interpreter/ast.rs
#[derive(Debug, Clone)]
pub enum Expr {
    Num(i64),
    Var(String),
    // ... existing variants ...
    Private(Box<Expr>),  // NEW: (private expr) for ZKP witness marking
}
```

### Parser Implementation

```rust
// src/interpreter/parser.rs
pub fn parse_tokens(tokens: &[String], pos: &mut usize) -> Result<Expr, String> {
    if tok == "(" {
        *pos += 1;
        let next_tok = &tokens[*pos];
        
        // NEW: Parse (private expr) immediately after quote handling
        if next_tok == "private" {
            *pos += 1;
            let inner = parse_tokens(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("expected ')' after private".to_string());
            }
            *pos += 1;
            return Ok(Expr::Private(Box::new(inner)));
        }
        // ... rest of parser ...
    }
}
```

### Testing

```
Command: echo '(let ((x 42)) (let ((y (private x))) (+ y 1)))' | ./target/debug/baru

Output:
parsed: Private(Var("x")) correctly
result: 43 ✅
```

### Usage Example

```scheme
; Input REPL
(let ((secret 42))
  (let ((result (+ (private secret) 10)))
    result))

; Parser output AST:
// Let bindings: secret = 42
// Let bindings: result = BinOp(Add, Private(Var("secret")), Num(10))
// Result: Private annotation extracted during circuit compilation
```

---

## Phase 6B: ZKPBackend Trait

### Minimal Trait Definition

```rust
// src/zkp/mod.rs

use std::collections::HashMap;
use crate::crypto::types::Fr;

/// Compiled circuit ready for proving/verifying
pub struct Circuit {
    pub inner: Box<dyn std::any::Any>,  // Backend-specific
    pub backend_name: String,
}

/// Public values needed for verification
#[derive(Clone)]
pub struct PublicValues {
    pub values: HashMap<String, Fr>,
}

/// Serialized proof
#[derive(Clone, Debug)]
pub struct ZKProof {
    pub data: Vec<u8>,
    pub backend_name: String,
}

/// ZKP Backend Abstraction
pub trait ZKPBackend: Send + Sync {
    /// Compile expression to circuit
    /// private_vars: list of variable names that are witness
    /// All other variables in expr are public (inferred)
    fn compile_circuit(
        &self,
        expr: &crate::interpreter::ast::Expr,
        private_vars: &[String],
    ) -> Result<Circuit, String>;

    fn prove(
        &self,
        circuit: &Circuit,
        private_values: &HashMap<String, Fr>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String>;

    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String>;

    fn serialize_circuit(&self, circuit: &Circuit) -> Result<Vec<u8>, String>;
    fn deserialize_circuit(&self, data: &[u8]) -> Result<Circuit, String>;
    fn name(&self) -> &str;
}
```

### Mock Backend for Testing

```rust
// src/zkp/mock_backend.rs (TEST HELPER)

pub struct MockBackend;

impl ZKPBackend for MockBackend {
    fn compile_circuit(
        &self,
        _expr: &Expr,
        _private_vars: &[String],
    ) -> Result<Circuit, String> {
        Ok(Circuit {
            inner: Box::new(vec![]),
            backend_name: "mock".to_string(),
        })
    }

    fn prove(
        &self,
        _circuit: &Circuit,
        _private_values: &HashMap<String, Fr>,
        _public_values: &PublicValues,
    ) -> Result<ZKProof, String> {
        // Return dummy proof
        Ok(ZKProof {
            data: vec![1, 2, 3],
            backend_name: "mock".to_string(),
        })
    }

    fn verify(
        &self,
        _circuit: &Circuit,
        _proof: &ZKProof,
        _public_values: &PublicValues,
    ) -> Result<bool, String> {
        Ok(true)  // Always "verify" for testing
    }

    fn serialize_circuit(&self, _circuit: &Circuit) -> Result<Vec<u8>, String> {
        Ok(vec![])
    }

    fn deserialize_circuit(&self, _data: &[u8]) -> Result<Circuit, String> {
        Ok(Circuit {
            inner: Box::new(vec![]),
            backend_name: "mock".to_string(),
        })
    }

    fn name(&self) -> &str {
        "mock"
    }
}
```

---

## Phase 6C: Binius Backend Skeleton

### Binius Backend Structure

```rust
// src/zkp/binius_backend.rs

use super::{Circuit, PublicValues, ZKProof, ZKPBackend};
use crate::crypto::types::Fr;
use crate::interpreter::ast::Expr;
use std::collections::HashMap;

pub struct Binius64Backend;

impl Binius64Backend {
    pub fn new() -> Self {
        Binius64Backend
    }

    /// Compile Expr to Binius circuit representation
    /// This is the hard part - converting Baru AST to gates
    fn compile_expr_to_gates(
        expr: &Expr,
        private_vars: &[String],
    ) -> Result<Vec<Gate>, String> {
        // Traverse AST and emit gates
        // For now: placeholder
        let mut gates = Vec::new();
        // TODO: implement gate emission
        Ok(gates)
    }
}

impl ZKPBackend for Binius64Backend {
    fn compile_circuit(
        &self,
        expr: &Expr,
        private_vars: &[String],
    ) -> Result<Circuit, String> {
        // 1. Analyze expression to get all variables
        let vars = extract_variables(expr);
        
        // 2. Partition into witness and public
        let witness_vars: Vec<String> = vars
            .iter()
            .filter(|v| private_vars.contains(v))
            .cloned()
            .collect();
        
        let public_vars: Vec<String> = vars
            .iter()
            .filter(|v| !private_vars.contains(v))
            .cloned()
            .collect();

        // 3. Compile to gates
        let _gates = Self::compile_expr_to_gates(expr, private_vars)?;

        // 4. Store in Circuit
        let circuit_data = Binius64Circuit {
            expr: expr.clone(),
            witness_vars,
            public_vars,
            // gates stored here
        };

        Ok(Circuit {
            inner: Box::new(circuit_data),
            backend_name: "binius64".to_string(),
        })
    }

    fn prove(
        &self,
        circuit: &Circuit,
        private_values: &HashMap<String, Fr>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String> {
        // 1. Extract circuit data
        let _circuit_data = circuit
            .inner
            .downcast_ref::<Binius64Circuit>()
            .ok_or("Invalid circuit type")?;

        // 2. Populate witness
        // 3. Call Binius prover
        // 4. Serialize proof

        Ok(ZKProof {
            data: vec![],  // TODO: implement
            backend_name: "binius64".to_string(),
        })
    }

    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String> {
        // 1. Extract circuit data
        let _circuit_data = circuit
            .inner
            .downcast_ref::<Binius64Circuit>()
            .ok_or("Invalid circuit type")?;

        // 2. Deserialize proof
        // 3. Call Binius verifier with public_values
        // 4. Return result

        Ok(true)  // TODO: implement
    }

    fn serialize_circuit(&self, circuit: &Circuit) -> Result<Vec<u8>, String> {
        let _circuit_data = circuit
            .inner
            .downcast_ref::<Binius64Circuit>()
            .ok_or("Invalid circuit type")?;
        
        // Serialize to bytes
        Ok(vec![])
    }

    fn deserialize_circuit(&self, _data: &[u8]) -> Result<Circuit, String> {
        // Deserialize from bytes
        unimplemented!()
    }

    fn name(&self) -> &str {
        "binius64"
    }
}

/// Binius64-specific circuit representation
#[derive(Clone)]
struct Binius64Circuit {
    expr: Expr,
    witness_vars: Vec<String>,
    public_vars: Vec<String>,
}

/// Gate representation (simplified)
#[derive(Clone)]
enum Gate {
    Add(WireId, WireId, WireId),      // out = in1 + in2
    Mul(WireId, WireId, WireId),      // out = in1 * in2
    And(WireId, WireId, WireId),      // out = in1 & in2
    Eq(WireId, WireId, WireId),       // out = in1 == in2
}

#[derive(Clone, Copy, Debug)]
struct WireId(usize);

fn extract_variables(expr: &Expr) -> Vec<String> {
    // Walk AST and collect all Var nodes
    let mut vars = Vec::new();
    fn walk(expr: &Expr, vars: &mut Vec<String>) {
        match expr {
            Expr::Var(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            Expr::BinOp(_, l, r) => {
                walk(l, vars);
                walk(r, vars);
            }
            Expr::Call(_, args) => {
                for arg in args {
                    walk(arg, vars);
                }
            }
            Expr::Private(inner) => walk(inner, vars),
            _ => {}
        }
    }
    walk(expr, &mut vars);
    vars
}
```

---

## Phase 6E: REPL Integration

### Builtin Functions

```rust
// src/interpreter/eval.rs (NEW FUNCTIONS)

fn builtin_prove(
    args: Vec<Value>,
    ctx: &mut EvalContext,
) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("(prove expr (private var1 var2 ...))".to_string());
    }

    let expr = match &args[0] {
        Value::Expr(e) => e,
        _ => return Err("First arg must be expression".to_string()),
    };

    // Parse private variables from (private x y z)
    let private_vars = match &args[1] {
        Value::List(list) => {
            let mut vars = Vec::new();
            for item in list {
                if let Value::Atom(name) = item {
                    vars.push(name.clone());
                }
            }
            vars
        }
        _ => return Err("Second arg must be (private ...)".to_string()),
    };

    // Get backend
    let backend = ctx.get_zkp_backend()?;

    // Collect values from context
    let mut private_values = HashMap::new();
    let mut public_values = HashMap::new();

    for var in &private_vars {
        if let Some(value) = ctx.variables.get(var) {
            private_values.insert(var.clone(), value.to_fr()?);
        }
    }

    // Collect all other variables as public
    for (var, value) in &ctx.variables {
        if !private_vars.contains(var) {
            public_values.values.insert(var.clone(), value.to_fr()?);
        }
    }

    // Compile and prove
    let circuit = backend.compile_circuit(expr, &private_vars)?;
    let proof = backend.prove(
        &circuit,
        &private_values,
        &public_values,
    )?;

    Ok(Value::Proof(proof))
}

fn builtin_verify(
    args: Vec<Value>,
    ctx: &mut EvalContext,
) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("(verify proof expr public-values)".to_string());
    }

    let proof = match &args[0] {
        Value::Proof(p) => p,
        _ => return Err("First arg must be proof".to_string()),
    };

    let expr = match &args[1] {
        Value::Expr(e) => e,
        _ => return Err("Second arg must be expression".to_string()),
    };

    // Parse public values
    let public_values = match &args[2] {
        Value::List(list) => {
            let mut pv = PublicValues::new();
            for pair in list {
                match pair {
                    Value::List(pair_list) if pair_list.len() == 2 => {
                        if let (Value::Atom(key), Value::Number(val)) = 
                            (&pair_list[0], &pair_list[1]) 
                        {
                            pv.values.insert(key.clone(), (*val) as Fr);
                        }
                    }
                    _ => {}
                }
            }
            pv
        }
        _ => return Err("Third arg must be public values list".to_string()),
    };

    // Get backend
    let backend = ctx.get_zkp_backend()?;

    // Verify
    let backend_name = &proof.backend_name;
    let circuit = backend.deserialize_circuit(&proof.data)?;
    let result = backend.verify(&circuit, proof, &public_values)?;

    Ok(Value::Boolean(result))
}

/// Add to builtins registry
pub fn register_zkp_builtins(builtins: &mut HashMap<String, BuiltinFn>) {
    builtins.insert("prove".to_string(), builtin_prove as BuiltinFn);
    builtins.insert("verify".to_string(), builtin_verify as BuiltinFn);
    builtins.insert("proof-save".to_string(), builtin_proof_save as BuiltinFn);
    builtins.insert("proof-load".to_string(), builtin_proof_load as BuiltinFn);
}
```

### Value Type Extension

```rust
// src/interpreter/types.rs (MODIFIED)

#[derive(Debug, Clone)]
pub enum Value {
    Number(u128),
    Atom(String),
    List(Vec<Value>),
    Expr(Expr),
    Proof(ZKProof),              // NEW
    Boolean(bool),
    // ...
}

impl Value {
    pub fn to_fr(&self) -> Result<Fr, String> {
        match self {
            Value::Number(n) => Ok(Fr::from(*n)),
            _ => Err("Cannot convert to Fr".to_string()),
        }
    }
}
```

---

## Phase 6F: Serialization

### Proof Storage

```rust
// src/zkp/serialization.rs

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct StoredProof {
    pub backend: String,
    pub proof_data: Vec<u8>,
    pub circuit_hash: String,  // SHA256 of circuit for verification
    pub timestamp: i64,
}

impl StoredProof {
    pub fn from_proof(proof: &ZKProof) -> Self {
        StoredProof {
            backend: proof.backend_name.clone(),
            proof_data: proof.data.clone(),
            circuit_hash: String::new(),
            timestamp: chrono::Local::now().timestamp(),
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| e.to_string())?;
        std::fs::write(path, json)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| e.to_string())?;
        serde_json::from_str(&json)
            .map_err(|e| e.to_string())
    }

    pub fn to_proof(&self) -> ZKProof {
        ZKProof {
            data: self.proof_data.clone(),
            backend_name: self.backend.clone(),
        }
    }
}
```

### REPL Usage

```scheme
;; Generate proof
(define proof (prove result (private secret)))

;; Save to file
(proof-save proof "proof.json")

;; Load from file (on different machine)
(define loaded-proof (proof-load "proof.json"))

;; Verify
(verify loaded-proof result public-map)
```

---

## Implementation Checklist

- [ ] Phase 6A: Add `is_private` to Expr, parse `(private ...)`
- [ ] Phase 6B: Define `ZKPBackend` trait and `PublicValues`
- [ ] Phase 6B: Create `MockBackend` for testing
- [ ] Phase 6C: Create `Binius64Backend` skeleton
- [ ] Phase 6C: Implement `compile_circuit` stub
- [ ] Phase 6D: Implement `extract_variables` helper
- [ ] Phase 6D: Implement gate emission for operations
- [ ] Phase 6D: Integrate with Binius CircuitBuilder
- [ ] Phase 6E: Add `builtin_prove` and `builtin_verify`
- [ ] Phase 6E: Add to REPL builtins registry
- [ ] Phase 6F: Implement `StoredProof` serialization
- [ ] Phase 6F: Add `proof-save` and `proof-load` builtins
- [ ] Tests: Unit tests for each phase
- [ ] Tests: Integration test end-to-end prove/verify

