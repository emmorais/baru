# ZKP Integration Plan for Baru: Phase 6 - Zero-Knowledge Proofs

## Executive Summary

This document outlines a comprehensive plan to integrate Zero-Knowledge Proof (ZKP) capabilities into Baru using **Binius64** as the initial backend, with a designed architecture that enables future backends (Plonky3, others) without coupling.

**Key Vision:** Enable participants to prove that local/private computations in MPC were carried out correctly, using minimal type annotations to distinguish secrets from public values.

**Simplified Design Principle:**
- **Witness** = variables explicitly marked `(private ...)` + any shared variables from MPC
- **Public** = all other variables (default)
- **Result:** Minimal annotation syntax, automatic classification, cleaner proof interface

---

## Part 1: Analysis of Binius64

### 1.1 Binius64 Architecture Overview

**System Characteristics:**
- **zk-SNARK** system optimized for 64-bit word circuits
- **Binary field arithmetic** (GF(2^128) for Iron Spartan; 64-bit words for Binius64)
- **Constraint types:**
  - AND constraints: bitwise AND relations
  - MUL constraints: 64-bit unsigned integer multiplication
  - Linear constraints via XOR (free)

**Performance Profile:**
- SHA-512 preimage proof: ~128ms prove, ~600ms setup
- 64-fold reduction vs bit-level approaches
- SIMD optimizations for modern CPUs
- Optional Rayon multithreading

### 1.2 Binius64 API Structure

**Key Components:**
```
binius-frontend          → Circuit construction (CircuitBuilder)
                          Witness population
                          Wire definitions

binius-prover           → High-level prove() API
                          Accepts: circuit + witness
                          Returns: Proof

binius-verifier         → High-level verify() API
                          Accepts: proof + public values
                          Returns: bool (verified?)

binius-circuits         → Standard gadgets (SHA256, ECDSA, etc.)

binius-core             → Shared constraint system structures
```

**Proof Flow:**
1. **Define circuit** via CircuitBuilder DSL
2. **Populate witness** with secret/private values
3. **Extract public values** needed for verification
4. **Generate proof** using prover backend
5. **Verify proof** independently using public values

### 1.3 Integration Difficulty Assessment

| Aspect | Difficulty | Notes |
|--------|-----------|-------|
| **Dependency Management** | Low | Binius64 is a stable library; well-maintained |
| **API Integration** | Medium | Need to wrap prove/verify in Baru builtins |
| **Type System** | Medium | Must extend Baru to support public/witness annotations |
| **Performance** | Low | Binius64 already optimized; minimal overhead |
| **Backend Abstraction** | Medium | Design trait for multiple ZKP backends |
| **Serialization** | Medium | Must handle proof/public value serialization |
| **Circuit Compilation** | High | Converting Baru expressions → Binius64 circuits |
| **Witness Extraction** | High | Tracking which values are witness vs public |

**Overall Complexity:** **Medium-High** (mainly circuit compilation & witness extraction)

---

## Part 2: Integration with Baru Architecture

### 2.1 Type System Extensions

**Simplified Type Classification:**

The witness is determined automatically:
- **Witness variables** = Any variable marked with `(private ...)` annotation
- **Shared variables** = Automatically treated as witness (part of witness)
- **Public variables** = Everything else (all inputs not marked private)

**Add minimal annotation in AST:**

```rust
// Current: pub enum Expr { ... }
// New variant in Types enum - only need to mark EXPLICITLY private:

#[derive(Debug, Clone)]
pub struct AnnotatedExpr {
    expr: Expr,
    is_private: bool,  // Only this matters; if false → public, if true → witness
}
```

**Syntax Extension (Lisp-like) - Minimal:**
```scheme
; Mark as private/witness (explicit secret)
(private (sha256 data))
(private secret-key)

; No need to mark as public - it's the default
(+ x y)  ; x, y are public

; Shared from MPC is automatically witness
(define shares (share secret 1 0))  ; 'shares' becomes witness automatically
```

### 2.1.1 Design Principles: Simplified Witness Model

This design achieves simplicity through **principle of least annotation**:

1. **Default is Public:** All variables are public unless explicitly marked otherwise
   - Reason: Most computation inputs/outputs are meant to be public knowledge
   - User only annotates secrets, reducing cognitive load

2. **Shared Variables are Automatically Witness:**
   ```scheme
   (define secret-share (share 42 1 0))  ; Automatically witness
   (define result (+ secret-share 10))   ; '10' is public
   (prove result (private secret-share))  ; Redundant but explicit
   ```

3. **Single Annotation: (private ...)**
   - Not needed for shared variables (auto-detected)
   - Used for explicit secrets not from MPC
   - Example: `(private password-hash)`, `(private encryption-key)`

4. **Verification Requires Only Public Values:**
   ```scheme
   ; Prover: compute with secrets
   (define proof (prove expr (private secret)))
   
   ; Verifier: check without secrets
   (verify proof expr public-values)
   ; Verifier learns nothing about secret!
   ```

This model maps cleanly to real security requirements:
- Users protect secrets with `(private ...)`
- Computation integrity is publicly verifiable
- Backward compatible with existing MPC code

### 2.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Baru Interpreter                          │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │   ABY3Backend    │  │   TransportBE    │  ... (MPC)      │
│  └──────────────────┘  └──────────────────┘                 │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            ZKP Integration Layer (NEW)               │   │
│  │                                                        │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │  ZKPBackend Trait (abstraction)                 │ │   │
│  │  │  - prove(circuit, witness, public) → Proof     │ │   │
│  │  │  - verify(proof, public) → bool                │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │   │
│  │  │ Biniu sBackend│  │ Plonky3BE    │  │ FutureBE  │  │   │
│  │  └──────────────┘  └──────────────┘  └───────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Circuit Compiler (Expr → ZKP Circuit)               │   │
│  │  - traverse AST                                       │   │
│  │  - emit gates (AND, MUL, linear)                     │   │
│  │  - classify witness vs public                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  ZKP Backend Crates (NEW)                    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  src/zkp/mod.rs (backend trait + types)             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  src/zkp/binius_backend.rs (Binius64 wrapper)       │   │
│  │  - CircuitBuilder wrapping                          │   │
│  │  - prove() → Binius prover                          │   │
│  │  - verify() → Binius verifier                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  src/crypto/circuit_compiler.rs (NEW)               │   │
│  │  - Compile Expr to ZKP constraints                  │   │
│  │  - Emit gates for operations (add, mul, sha256)     │   │
│  │  - Extract witness and public values                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Module Structure

```
src/
├── interpreter/
│   ├── backend.rs              (unchanged)
│   ├── aby3_backend.rs         (unchanged)
│   ├── transport_backend.rs    (unchanged)
│   └── eval.rs                 (extend for ZKP annotation support)
│
├── zkp/                         (NEW)
│   ├── mod.rs                  (ZKPBackend trait, types)
│   ├── binius_backend.rs       (Binius64 wrapper)
│   ├── plonky3_backend.rs      (future)
│   └── circuit_compiler.rs     (Expr → Circuit compilation)
│
└── crypto/
    └── (unchanged - still has ABY3 protocol)
```

---

## Part 3: Detailed Implementation Plan

### Phase 6A: Type System Extension (Week 1)

**Goal:** Add minimal private annotation to Baru expressions

**Simplified Model:**
- Default: all variables are **public**
- Explicit: mark secrets with `(private ...)` → these become **witness**
- Automatic: shared variables from MPC are also **witness**

**Changes:**
1. **AST Extension** (`src/interpreter/ast.rs`)
   - Add single boolean flag to Expr: `is_private: bool`
   - Add parser rule for `(private expr)` 

2. **Parser Extension** (`src/interpreter/parser.rs`)
   - Add token: `private`
   - Parse `(private expr)` → mark expr as private

3. **Tests**
   - Parse `(private secret)` correctly
   - Unannotated expressions are public by default
   - Handle nested annotations

**Effort:** ~100-150 lines of code + tests

---

### Phase 6B: ZKP Backend Trait (Week 1)

**Goal:** Define abstract ZKP backend interface with simplified witness model

**Key Design Decision:**
- **Witness** = variables marked with `(private ...)` + any shared variables from MPC
- **Public** = all other variables in the expression
- **Automatic classification** = no need to explicitly list what's public

**New File:** `src/zkp/mod.rs`

```rust
use crate::crypto::types::Fr;

/// Represents a compiled circuit ready for proving/verifying
pub struct Circuit {
    /// Circuit definition (backend-specific)
    /// For Binius: CircuitBuilder handle
    pub inner: Box<dyn std::any::Any>,
    pub backend_name: String,
}

/// Public values needed for verification (only these are revealed)
#[derive(Clone)]
pub struct PublicValues {
    /// Map from public variable name to Fr value
    pub values: std::collections::HashMap<String, Fr>,
}

/// ZK Proof (serializable)
#[derive(Clone, Debug)]
pub struct ZKProof {
    /// Serialized proof (backend-specific)
    pub data: Vec<u8>,
    pub backend_name: String,
}

/// ZKP Backend Trait
pub trait ZKPBackend: Send + Sync {
    /// Compile an expression into a circuit
    /// Input: AST, private_var_names (witness), everything else is public
    /// Output: Compiled circuit
    fn compile_circuit(
        &self,
        expr: &crate::interpreter::ast::Expr,
        private_vars: &[String],  // Only EXPLICITLY marked private variables → witness
        // All other variables in the expression → public
    ) -> Result<Circuit, String>;

    /// Generate a proof for given values
    /// Input: circuit, all values (private + public), witness dict
    /// Output: ZKProof
    fn prove(
        &self,
        circuit: &Circuit,
        private_values: &std::collections::HashMap<String, Fr>,  // Witness
        public_values: &PublicValues,                              // Public inputs
    ) -> Result<ZKProof, String>;

    /// Verify a proof given only public values
    /// Input: circuit, proof, public values
    /// Output: verification result (bool)
    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String>;

    /// Serialize circuit for storage/transmission
    fn serialize_circuit(&self, circuit: &Circuit) -> Result<Vec<u8>, String>;

    /// Deserialize circuit from storage
    fn deserialize_circuit(&self, data: &[u8]) -> Result<Circuit, String>;

    /// Get backend name
    fn name(&self) -> &str;
}
```

**Tests:**
- Mock backend implementation
- Test trait object creation
- Verify private/public classification
- Serialization round-trip

**Effort:** ~150-200 lines + tests

---

### Phase 6C: Binius64 Backend Wrapper (Weeks 2-3)

**Goal:** Wrap Binius64 API in ZKP backend trait

**New File:** `src/zkp/binius_backend.rs`

```rust
use super::{Circuit, PublicValues, ZKProof, ZKPBackend};
use crate::crypto::types::Fr;

pub struct Binius64Backend {
    // Binius-specific configuration
    // e.g., field size, optimization flags
}

impl Binius64Backend {
    pub fn new() -> Self {
        Binius64Backend {}
    }

    /// Convert Baru Expr to Binius64 CircuitBuilder operations
    fn compile_expr_to_gates(
        &self,
        expr: &crate::interpreter::ast::Expr,
        builder: &mut binius_frontend::CircuitBuilder,
    ) -> Result<WireRef, String> {
        // Pattern match on expr type:
        // - Var → lookup wire
        // - NumLit → constant
        // - BinOp::Add → builder.add()
        // - BinOp::Mul → builder.mul()
        // - BinOp::BitAnd → builder.and() [native]
        // - BinOp::BitOr → decompose to AND/XOR
        // - Call(sha256, ...) → builder.sha256_gadget()
        // - Other → emit as constraints
        unimplemented!()
    }
}

impl ZKPBackend for Binius64Backend {
    fn compile_circuit(
        &self,
        expr: &crate::interpreter::ast::Expr,
        witness_vars: &[String],
        public_vars: &[String],
    ) -> Result<Circuit, String> {
        // 1. Create CircuitBuilder
        // 2. Register witness wires
        // 3. Register public input wires
        // 4. Compile expr to gates
        // 5. Return Circuit wrapping the builder
        unimplemented!()
    }

    fn prove(
        &self,
        circuit: &Circuit,
        witness: &std::collections::HashMap<String, Fr>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String> {
        // 1. Extract CircuitBuilder from Circuit::inner
        // 2. Populate witness values
        // 3. Call binius_prover::prove()
        // 4. Serialize proof to bytes
        // 5. Return ZKProof
        unimplemented!()
    }

    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String> {
        // 1. Deserialize proof
        // 2. Call binius_verifier::verify() with public values
        // 3. Return result
        unimplemented!()
    }

    // ... other methods
}
```

**Integration Points:**
- Depends on: `binius-frontend`, `binius-prover`, `binius-verifier`
- No changes needed to existing Baru code
- This is a purely additive module

**Challenges:**
1. **Automatic Private Variable Detection:** 
   - Solution: Track variables from `(share ...)` calls; mark them as witness automatically
2. **Circuit Representation:** Binius uses CircuitBuilder, need to store handle
   - Solution: Use `Box<dyn Any>` with downcast
3. **Wire References:** Mapping Baru vars to Binius wires
   - Solution: HashMap<String, WireRef>
4. **Operation Support:** Not all Baru ops map directly to Binius gates
   - Solution: Decompose complex ops to AND/MUL/XOR gates

**Effort:** ~400-600 lines (complex, high-value)

---

### Phase 6D: Circuit Compiler (Weeks 2-4)

**Goal:** Compile Baru Expr → ZKP constraints

**New File:** `src/crypto/circuit_compiler.rs`

**Core Functionality:**

```rust
pub struct CircuitCompiler {
    backend: Arc<dyn ZKPBackend>,
    private_vars: Vec<String>,  // Variables marked as (private ...)
    witness_map: HashMap<String, Fr>,
    public_map: HashMap<String, Fr>,
}

impl CircuitCompiler {
    pub fn new(backend: Arc<dyn ZKPBackend>) -> Self {
        CircuitCompiler {
            backend,
            private_vars: Vec::new(),
            witness_map: HashMap::new(),
            public_map: HashMap::new(),
        }
    }

    /// Analyze expression to extract explicitly private variables
    pub fn analyze_expr(&mut self, expr: &Expr) -> Vec<String> {
        // Traverse AST
        // Collect variables marked as (private x)
        // Return list of private_vars
        // Everything else is automatically public
        unimplemented!()
    }

    /// Compile and generate proof
    /// private_vars: list of variables to treat as witness
    /// All other variables in expr are automatically public
    pub fn prove(
        &mut self,
        expr: &Expr,
        private_vars: &[String],
        witness: &HashMap<String, Fr>,
        public: &HashMap<String, Fr>,
    ) -> Result<ZKProof, String> {
        // 1. Compile to circuit using private_vars list
        // 2. Populate witness with private_vars values
        // 3. Call backend.prove() with witness + public
        unimplemented!()
    }

    /// Verify proof
    /// Needs only the public values; witness is hidden in proof
    pub fn verify(
        &self,
        expr: &Expr,
        private_vars: &[String],
        proof: &ZKProof,
        public: &HashMap<String, Fr>,
    ) -> Result<bool, String> {
        // 1. Compile to circuit (deterministic) using same private_vars
        // 2. Call backend.verify() with public values only
        // 3. Proof is verification; cannot learn witness
        unimplemented!()
    }
}
```

**Simplified Workflow:**
```
Private variables (marked as (private ...)): kept secret in proof
↓
Compile Expr → Circuit
↓
Prove(circuit, private_values, public_values) → ZKProof
↓
Verify(circuit, ZKProof, public_values) → bool
         (no private values needed!)
```

**Operation Support Matrix:**

| Operation | Binius Support | Method |
|-----------|---|---------|
| `+` (add) | Native | Linear constraint |
| `*` (mul) | Native MUL gate | Direct |
| `-` (sub) | Native | Linear (add with negation) |
| `=` (eq) | Decompose | Convert to AND/XOR gates |
| `<` (lt) | Decompose | Bitwise comparison |
| `&` (AND) | Native AND gate | Direct |
| `\|` (OR) | Decompose | `a \| b = a + b - a*b` |
| `>>` (shift) | Decompose | Array indexing + padding |
| `<<` (shift) | Linear | Multiplication by power-of-2 |
| `sha256()` | Gadget | Use binius-circuits::sha256 |
| `sha512()` | Gadget | Use binius-circuits::sha512 |

**Effort:** ~800-1200 lines (complex, foundational)

---

### Phase 6E: Interpreter Integration (Week 4)

**Goal:** Add prove/verify builtins to Baru REPL

**Changes to `src/interpreter/eval.rs`:**

```rust
// Add to builtin functions registry:

fn builtin_prove(args: Vec<Types>, participant: &Participant) -> Types {
    // (prove <expr> <witness-vars> <public-vars>)
    // Returns: (proof <proof-bytes> <circuit-bytes>)
    unimplemented!()
}

fn builtin_verify(args: Vec<Types>, participant: &Participant) -> Types {
    // (verify <proof> <expr> <public-values>)
    // Returns: (verified? #t/#f)
    unimplemented!()
}

fn builtin_zkp_backend(args: Vec<Types>, participant: &Participant) -> Types {
    // (zkp-backend "binius64") or (zkp-backend "plonky3")
    // Sets active ZKP backend
    unimplemented!()
}
```

**REPL Usage:**
```scheme
;; Define a function to prove
(define secret 42)
(define public-input 10)
(define result (+ secret public-input))

;; Compile and prove
(define proof (prove result
                      (private secret)
                      (public public-input)))

;; Verify (can be done by another party with only proof + public-input)
(verify proof result (public public-input))
```

**Effort:** ~200-300 lines + tests

---

### Phase 6F: Serialization & Storage (Week 3)

**Goal:** Save/load proofs and circuits for cross-party verification

**New Module:** `src/zkp/serialization.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct StoredProof {
    pub backend: String,
    pub circuit_hash: [u8; 32],  // SHA256 of circuit
    pub proof_data: Vec<u8>,
    pub timestamp: i64,
    pub prover_id: usize,
}

#[derive(Serialize, Deserialize)]
pub struct StoredCircuit {
    pub backend: String,
    pub circuit_data: Vec<u8>,
    pub expr_source: String,  // Original Baru expression
    pub hash: [u8; 32],
}

impl StoredProof {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}
```

**Effort:** ~150-200 lines

---

## Part 4: Integration with Existing Architecture

### 4.1 How ZKPs Fit with MPC Backends

```
Scenario: 3-party MPC with ZKP verification

1. ABY3Backend computes shared values:
   - Party 0: (x0, x1)
   - Party 1: (x1, x2)
   - Party 2: (x2, x0)
   
2. Each party independently:
   - Takes their local components
   - Proves (in ZKP) that computation was correct
   - Shares proof with others
   
3. Other parties verify:
   - Use proof + public circuit metadata
   - Confirm local computation was honest
   - No need to trust other party's computation
```

**Architectural Benefit:**
- **Orthogonal to MPC:** ZKPs add integrity, MPC adds privacy
- **Composable:** Prove(MPC_computation) enables hybrid secure computation
- **No changes needed:** Existing Backend trait unchanged

### 4.2 Using ZKPs with TransportBackend

```
Distributed scenario:

1. Party 0 (garbler) sends garbled circuit to Party 1
2. Party 0 proves (in ZKP): "This garbled circuit is correctly formed"
3. Party 1 verifies proof before using garbled circuit
4. Result: Party 1 has cryptographic guarantee of correctness
```

### 4.3 REPL Example Flow

```lisp
; Setup: create MPC backend
(define p0 (participant-new 0))

; Define secrets and public inputs
(define secret-a 42)          ; witness
(define secret-b 10)          ; witness
(define public-multiplier 2)  ; public

; Compute result
(define result (* (+ secret-a secret-b) public-multiplier))

; Generate ZKP proof: mark which variables are private
(define proof (prove result
                      (private secret-a secret-b)  ; witness
                      ))
; Everything else (public-multiplier) is automatically public

; Save proof for other participants
(proof-save proof "proof.json")

; Other party: load and verify (only needs proof + public inputs)
(define proof-other (proof-load "proof.json"))
(define public-inputs (list public-multiplier))
(define verified? (verify proof-other result public-inputs))
```

---

## Part 5: Implementation Phases Summary

| Phase | Name | Duration | Priority | Effort |
|-------|------|----------|----------|--------|
| 6A | Type System | Week 1 | P1 | Low |
| 6B | Backend Trait | Week 1 | P1 | Low |
| 6C | Binius Wrapper | Weeks 2-3 | P1 | High |
| 6D | Circuit Compiler | Weeks 2-4 | P1 | High |
| 6E | REPL Integration | Week 4 | P2 | Medium |
| 6F | Serialization | Week 3 | P2 | Medium |

**Total Estimated Effort:** 4-5 weeks (can be parallelized)

---

## Part 6: Future Extensions

### 6.1 Plonky3 Backend (Phase 7)

```rust
pub struct Plonky3Backend {
    // Plonky3-specific config
}

impl ZKPBackend for Plonky3Backend {
    // Similar interface to Binius
    // Map Baru ops to Plonky3 AIR constraints
}
```

**Rationale:** Plonky3 offers different performance tradeoffs (smaller proofs, different field)

### 6.2 Multi-Backend Proofs

```scheme
; Generate proof with multiple backends and compare
(define proof-binius (prove-with "binius64" expr))
(define proof-plonky (prove-with "plonky3" expr))

; Require verification by all backends
(and (verify-with "binius64" proof-binius)
     (verify-with "plonky3" proof-plonky))
```

### 6.3 Recursive Proofs

```scheme
; Prove a proof (proof of proof)
(define inner-proof (prove computation))
(define outer-proof (prove (proof-is-valid inner-proof)))
```

### 6.4 Proof Aggregation

```scheme
; Combine multiple proofs into one
(define aggregated (aggregate-proofs proof1 proof2 proof3))
```

---

## Part 7: Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Binius API instability | Low | High | Pin version; add compatibility layer |
| Circuit compilation bugs | Medium | High | Extensive testing; formal verification |
| Witness leakage | Low | Critical | Careful variable scope management; audits |
| Performance regression | Medium | Medium | Benchmark early; optimize hot paths |
| Field mismatch | Low | High | Document Fr↔Binius field mapping |

---

## Part 8: Success Criteria

✅ **Phase 6A:** Parse public/private annotations correctly  
✅ **Phase 6B:** ZKPBackend trait compiles; mock impl passes tests  
✅ **Phase 6C:** Binius backend integrates; simple circuits prove/verify  
✅ **Phase 6D:** Compiler handles add/mul/and operations correctly  
✅ **Phase 6E:** REPL builtins work; end-to-end prove/verify flow complete  
✅ **Phase 6F:** Proofs serialize/deserialize without loss  

**Integration Test:**
```rust
#[test]
fn test_zkp_workflow() {
    // 1. Create ZKP backend
    // 2. Evaluate expression with annotations
    // 3. Generate proof
    // 4. Verify proof independently
    // 5. Assert verification succeeds
}
```

---

## Conclusion

The proposed ZKP integration provides:

1. **Clean Abstraction:** ZKPBackend trait enables multiple backends
2. **No Disruption:** Orthogonal to existing MPC architecture
3. **Strong Guarantees:** Participants can prove honest computation
4. **Extensible Design:** Easy to add Plonky3, FHE, other systems
5. **Practical:** Uses battle-tested Binius64 library

**Timeline:** 4-5 weeks to full implementation  
**Risk Level:** Medium (high complexity, established library reduces risk)  
**Expected Outcome:** Baru becomes hybrid: MPC + ZKP system enabling verified distributed computation

