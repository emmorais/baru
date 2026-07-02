# Phase 6B: ZKP Backend Trait Definition - Implementation Plan

**Status**: READY TO BEGIN  
**Prerequisite**: Phase 6A ✅ COMPLETE  
**Estimated Duration**: 3-4 hours  
**Target Completion**: July 2, 2026 (same day)  

---

## Overview

Phase 6B defines the ZKP backend abstraction layer. This layer enables different ZKP backends (Binius64, Circom, Plonk, etc.) to be swapped without changing core interpreter code.

The trait-based design mirrors the existing MPC backend abstraction currently in the codebase.

---

## Objectives

### Primary
1. Create `src/zkp/` module structure
2. Define `ZKPBackend` trait with minimal but sufficient API
3. Create core types: `Circuit`, `PublicValues`, `ZKProof`
4. Establish interface for circuit compilation, proving, and verification

### Secondary
1. Create mock backend for testing
2. Add basic error handling
3. Document trait requirements clearly

### Non-Objectives (Phase 6C+)
- Implement Binius64 backend (Phase 6C)
- Implement circuit compilation logic (Phase 6D)
- Integrate with REPL (Phase 6E)

---

## Design

### Module Structure

```
src/
├── interpreter/
│   ├── ast.rs (Expr with Private variant ✅)
│   ├── parser.rs (supports (private ...) ✅)
│   ├── eval.rs (CPS evaluator ✅)
│   ├── backend.rs (MPC backend trait)
│   └── ...
├── zkp/ (NEW)
│   ├── mod.rs (trait definitions)
│   ├── mock_backend.rs (test implementation)
│   └── (later: binius_backend.rs, compiler.rs)
└── ...
```

### Type Definitions

```rust
// src/zkp/mod.rs

use std::collections::HashMap;
use crate::interpreter::ast::Expr;

/// Backend-agnostic compiled circuit
pub struct Circuit {
    pub inner: Box<dyn std::any::Any>,  // Backend stores what it needs
    pub backend_name: String,
}

/// Public values for verification
#[derive(Clone, Debug)]
pub struct PublicValues {
    pub values: HashMap<String, i64>,  // Variable name → value
}

/// Serialized proof
#[derive(Clone, Debug)]
pub struct ZKProof {
    pub data: Vec<u8>,
    pub backend_name: String,
}

/// ZKP Backend Trait
pub trait ZKPBackend: Send + Sync {
    /// Compile expression to circuit with witness variables marked
    fn compile_circuit(
        &self,
        expr: &Expr,
        witness_vars: &[String],  // Variables marked (private ...) or from MPC
    ) -> Result<Circuit, String>;

    /// Generate proof
    fn prove(
        &self,
        circuit: &Circuit,
        witness_values: &HashMap<String, i64>,
        public_values: &PublicValues,
    ) -> Result<ZKProof, String>;

    /// Verify proof
    fn verify(
        &self,
        circuit: &Circuit,
        proof: &ZKProof,
        public_values: &PublicValues,
    ) -> Result<bool, String>;

    /// Serialize circuit for storage
    fn serialize_circuit(&self, circuit: &Circuit) -> Result<Vec<u8>, String>;

    /// Deserialize circuit from storage
    fn deserialize_circuit(&self, data: &[u8]) -> Result<Circuit, String>;

    /// Backend identifier
    fn name(&self) -> &str;
}
```

### API Design Rationale

| Method | Purpose | Notes |
|--------|---------|-------|
| `compile_circuit()` | AST → Circuit | Phase 6D: implement actual compilation |
| `prove()` | Generate proof | Given all values (witness + public) |
| `verify()` | Check proof | Only needs public values + proof |
| `serialize_circuit()` | Store circuit | For reuse across invocations |
| `deserialize_circuit()` | Load circuit | Companion to serialize |
| `name()` | Backend ID | For debug/logging |

---

## Implementation Plan

### Step 1: Create Module Structure

**File**: `src/zkp/mod.rs`

```rust
pub mod mock_backend;

pub use mock_backend::MockBackend;

use std::collections::HashMap;
use crate::interpreter::ast::Expr;

// Type definitions from above
pub struct Circuit { ... }
pub struct PublicValues { ... }
pub struct ZKProof { ... }
pub trait ZKPBackend { ... }
```

### Step 2: Define Core Types

**In**: `src/zkp/mod.rs`

```rust
#[derive(Clone, Debug)]
pub struct PublicValues {
    pub values: HashMap<String, i64>,
}

impl PublicValues {
    pub fn new() -> Self {
        PublicValues {
            values: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, value: i64) {
        self.values.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<i64> {
        self.values.get(name).copied()
    }
}

#[derive(Clone, Debug)]
pub struct Circuit {
    pub inner: Box<dyn std::any::Any>,
    pub backend_name: String,
}

#[derive(Clone, Debug)]
pub struct ZKProof {
    pub data: Vec<u8>,
    pub backend_name: String,
}
```

### Step 3: Define Trait

**In**: `src/zkp/mod.rs`

```rust
pub trait ZKPBackend: Send + Sync {
    fn compile_circuit(
        &self,
        expr: &Expr,
        witness_vars: &[String],
    ) -> Result<Circuit, String>;

    fn prove(
        &self,
        circuit: &Circuit,
        witness_values: &HashMap<String, i64>,
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

### Step 4: Create Mock Backend

**File**: `src/zkp/mock_backend.rs`

```rust
use super::{Circuit, PublicValues, ZKProof, ZKPBackend};
use crate::interpreter::ast::Expr;
use std::collections::HashMap;
use std::any::Any;

/// Mock backend for testing
/// Always "proves" successfully, never verifies correctly
/// Used for testing infrastructure before real backend
pub struct MockBackend;

impl ZKPBackend for MockBackend {
    fn compile_circuit(
        &self,
        _expr: &Expr,
        _witness_vars: &[String],
    ) -> Result<Circuit, String> {
        Ok(Circuit {
            inner: Box::new("mock circuit"),
            backend_name: "mock".to_string(),
        })
    }

    fn prove(
        &self,
        _circuit: &Circuit,
        _witness_values: &HashMap<String, i64>,
        _public_values: &PublicValues,
    ) -> Result<ZKProof, String> {
        Ok(ZKProof {
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],  // Mock proof
            backend_name: "mock".to_string(),
        })
    }

    fn verify(
        &self,
        _circuit: &Circuit,
        _proof: &ZKProof,
        _public_values: &PublicValues,
    ) -> Result<bool, String> {
        // Mock: always returns false to force real backend testing
        Ok(false)
    }

    fn serialize_circuit(&self, _circuit: &Circuit) -> Result<Vec<u8>, String> {
        Ok(vec![0])
    }

    fn deserialize_circuit(&self, _data: &[u8]) -> Result<Circuit, String> {
        Ok(Circuit {
            inner: Box::new("mock circuit"),
            backend_name: "mock".to_string(),
        })
    }

    fn name(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend_compile() {
        let backend = MockBackend;
        let expr = Expr::Num(42);
        let result = backend.compile_circuit(&expr, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_backend_prove() {
        let backend = MockBackend;
        let circuit = Circuit {
            inner: Box::new("mock"),
            backend_name: "mock".to_string(),
        };
        let witness = HashMap::new();
        let public = PublicValues::new();
        let result = backend.prove(&circuit, &witness, &public);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_backend_verify() {
        let backend = MockBackend;
        let circuit = Circuit {
            inner: Box::new("mock"),
            backend_name: "mock".to_string(),
        };
        let proof = ZKProof {
            data: vec![],
            backend_name: "mock".to_string(),
        };
        let public = PublicValues::new();
        let result = backend.verify(&circuit, &proof, &public);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);  // Mock always returns false
    }
}
```

### Step 5: Update Module Exports

**In**: `src/interpreter/mod.rs` or `src/lib.rs`

```rust
pub mod zkp;  // NEW: export ZKP module

// If needed, add to main module exports:
pub use zkp::{ZKPBackend, Circuit, PublicValues, ZKProof};
```

---

## Testing Strategy

### Unit Tests
- Mock backend compile test
- Mock backend prove test
- Mock backend verify test
- PublicValues insertion/retrieval

### Integration Tests (Phase 6B+)
- Circuit storage/retrieval round-trip
- Multiple witness variables
- Error handling

### Example Usage (Reference for Phase 6D)
```rust
// Phase 6D will look something like:
let backend = MockBackend;  // or Binius64Backend
let expr = Expr::BinOp(Add, Box::new(Expr::Var("a")), Box::new(Expr::Num(10)));
let circuit = backend.compile_circuit(&expr, &["a"])?;

let mut witness = HashMap::new();
witness.insert("a".to_string(), 32);

let mut public = PublicValues::new();
public.insert("output".to_string(), 42);

let proof = backend.prove(&circuit, &witness, &public)?;
let valid = backend.verify(&circuit, &proof, &public)?;
assert!(valid);
```

---

## Deliverables

### Files to Create
1. `src/zkp/mod.rs` - Type definitions and trait
2. `src/zkp/mock_backend.rs` - Mock implementation

### Files to Modify
1. `src/interpreter/mod.rs` or `src/lib.rs` - Add module export

### Documentation
1. Update `ZKP_IMPLEMENTATION_GUIDE.md` with trait details
2. Add rustdoc comments to trait methods

---

## Acceptance Criteria

### Build & Compile
- [ ] `cargo build` succeeds with zero errors
- [ ] No compiler warnings
- [ ] No clippy warnings

### Functionality
- [ ] MockBackend implements all trait methods
- [ ] Mock tests compile and pass
- [ ] Types are properly exported

### Documentation
- [ ] Trait methods documented with examples
- [ ] Design decisions explained
- [ ] README updated

### Backward Compatibility
- [ ] No changes to existing MPC code
- [ ] AST module still works as before
- [ ] Evaluator unchanged

---

## Estimated Effort

| Task | Time | Notes |
|------|------|-------|
| Create module structure | 10 min | Straightforward |
| Define types | 20 min | Trait design well-settled |
| Write trait | 15 min | Copy from design above |
| Mock backend | 30 min | Includes tests |
| Export setup | 10 min | Update main module |
| Testing & cleanup | 20 min | Run tests, fix issues |
| **Total** | **~2 hours** | **Conservative estimate** |

---

## Blockers & Risks

### No Blockers
- ✅ Phase 6A complete
- ✅ Design settled
- ✅ No external dependencies needed

### Low Risk
- Mock backend is intentionally simple
- Types are straightforward
- Trait is well-defined and minimal

---

## Success Metrics

By end of Phase 6B:
- ✅ ZKP module exists and exports types/trait
- ✅ MockBackend fully implements ZKPBackend
- ✅ Unit tests pass
- ✅ Ready for Phase 6C (Binius64 implementation)

---

## Next: Phase 6C

Once Phase 6B is complete:

```
Phase 6C: Binius64 Backend Implementation
├─ Create src/zkp/binius_backend.rs
├─ Implement ZKPBackend for Binius64
├─ Add CircuitBuilder wrapping
├─ Integrate binius_prover/binius_verifier
└─ Write comprehensive tests
```

---

## References

- **Phase 6A**: `PHASE_6A_COMPLETE.md` - AST extension ✅
- **Design Doc**: `ZKP_INTEGRATION_PLAN.md` - Overall plan
- **MPC Reference**: `src/interpreter/backend.rs` - Existing trait pattern

---

*Plan Created: July 2, 2026*  
*Status: Ready to Implement*  
*Expected Completion: Same day (4-5 hours after Phase 6A)*
