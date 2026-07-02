# ZKP Integration Quick Reference

## One-Sentence Summary
Enable Baru participants to prove computations are correct without revealing secrets, using automatic witness detection.

## Core Model

```
Witness     = (private var) + shared variables (auto)
Public      = all other variables
Proof       = proves constraints satisfied (zero-knowledge)
Verification= works with only public values
```

## Syntax

| Feature | Example | Meaning |
|---------|---------|---------|
| Define variable | `(define x 42)` | x = 42 (public) |
| Mark as secret | `(define secret (private key))` | secret is witness |
| Compute | `(define result (+ secret 10))` | Mix secret & public |
| Generate proof | `(prove result (private secret))` | Prove computation |
| Verify proof | `(verify proof result)` | Anyone can verify |

## Quick Examples

### Example 1: Simple Secret
```scheme
(define secret 42)
(define result (+ secret 100))
(define proof (prove result (private secret)))
(verify proof result)  ; → true
```

### Example 2: Shared Values (Auto-Witness)
```scheme
(define share (share secret-value 1 0))
(define result (mul share 2))
(define proof (prove result (private share)))  ; 'share' auto-detected
```

### Example 3: Multiple Secrets
```scheme
(define s1 10)
(define s2 20)
(define result (+ s1 s2))
(define proof (prove result (private s1 s2)))
```

### Example 4: Stored Proof
```scheme
(define proof (prove expr (private secret)))
(proof-save proof "proof.bin")

; Later...
(define loaded (proof-load "proof.bin"))
(verify loaded expr)  ; Works!
```

## Architecture

```
┌──────────────────────┐
│   Baru REPL          │
├──────────────────────┤
│ AST + is_private     │
├──────────────────────┤
│  ZKPBackend Trait    │
├────────────┬─────────┤
│ Binius64   │ Plonky3 │  (future)
└────────────┴─────────┘
```

## Implementation Phases

| Phase | Task | LOC | Time |
|-------|------|-----|------|
| 6A | AST: add is_private flag | 100 | 1w |
| 6B | Trait: ZKPBackend definition | 150 | 1w |
| 6C | Binius64Backend wrapper | 500 | 1w |
| 6D | Circuit compiler | 800 | 2w |
| 6E | REPL builtins | 200 | 1w |
| 6F | Serialization | 150 | 1w |

## Key Files to Modify

### New Modules
- `src/zkp/mod.rs` - Backend trait
- `src/zkp/binius_backend.rs` - Binius64
- `src/crypto/circuit_compiler.rs` - Compilation

### Existing Modules
- `src/interpreter/ast.rs` - Add is_private
- `src/interpreter/parser.rs` - Parse (private ...)
- `src/interpreter/eval.rs` - prove/verify builtins

## Common Tasks

### Add New Operation
1. In `circuit_compiler.rs`: Add gate emission
2. Map operation to Binius constraint
3. Add unit test

### Add New Backend
1. Implement `ZKPBackend` trait
2. Add backend module
3. Register in REPL

### Debug Witness Issues
```rust
// Check what's marked as witness
let private_vars: Vec<String> = extract_private_annotations(expr);
dbg!(private_vars);

// Check inferred public vars
let all_vars = extract_all_variables(expr);
let public_vars: Vec<String> = all_vars.iter()
    .filter(|v| !private_vars.contains(v))
    .cloned()
    .collect();
```

## API Quick Reference

```rust
// Backend abstraction
pub trait ZKPBackend {
    fn compile_circuit(&self, expr: &Expr, private_vars: &[String]) 
        -> Result<Circuit, String>;
    fn prove(&self, circuit: &Circuit, private_values: &HashMap<String, Fr>,
             public_values: &PublicValues) -> Result<ZKProof, String>;
    fn verify(&self, circuit: &Circuit, proof: &ZKProof,
              public_values: &PublicValues) -> Result<bool, String>;
}

// Types
pub struct PublicValues { pub values: HashMap<String, Fr> }
pub struct ZKProof { pub data: Vec<u8>, pub backend_name: String }
pub struct Circuit { pub inner: Box<dyn Any>, backend_name: String }
```

## Decision Tree

```
I want to...
│
├─ Prove computation is correct
│  └─ Use: (prove expr (private secrets))
│
├─ Verify someone's proof
│  └─ Use: (verify proof expr public-values)
│
├─ Save proof for later
│  └─ Use: (proof-save proof "file")
│
├─ Load proof from file
│  └─ Use: (proof-load "file")
│
├─ Add new ZKP backend
│  └─ Implement ZKPBackend trait
│
└─ Debug why proof fails
   └─ Check: public values match, expr unchanged
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Proof invalid | Circuit modified after proving | Use same expr for verify |
| Witness leaked | Secret not marked (private ...) | Mark with (private secret) |
| Compilation fails | Operation not supported | Map to supported gates |
| Type mismatch | Fr vs u128 mismatch | Convert values properly |

## Performance Estimates

| Operation | Time |
|-----------|------|
| Circuit compile | <100ms |
| Proof generation | 100-500ms |
| Proof verification | 50-100ms |
| Proof size | 1-10KB |

## Security Properties

| Property | What it means |
|----------|--------------|
| Zero-Knowledge | Verifier learns nothing about witness |
| Soundness | Cannot fake valid proof (unless break crypto) |
| Completeness | Honest prover can always prove |
| Non-interactive | No communication between prove/verify |

## Glossary

- **Witness** - Secret values kept hidden (from public)
- **Public Values** - Revealed in proof (not hidden)
- **Circuit** - Constraint system over variables
- **Proof** - Evidence satisfying constraints (hides witness)
- **Backend** - Implementation of ZKP system
- **Binius64** - ZKP system for 64-bit operations
- **Soundness** - Cannot prove false statements

## Resources

- **Full Spec** - `ZKP_INTEGRATION_PLAN.md`
- **Design** - `ZKP_DESIGN_SUMMARY.md`
- **Code** - `ZKP_IMPLEMENTATION_GUIDE.md`
- **System** - `SYSTEM_ARCHITECTURE.md`

## Next Steps

1. Review this document
2. Read `ZKP_DESIGN_SUMMARY.md` for details
3. Implement Phase 6A (AST changes)
4. Implement Phase 6B (trait definition)
5. Implement Phase 6C (Binius integration)
6. Add tests at each phase
7. Verify end-to-end workflow

---

**Status:** Design Complete ✅ | Ready for Implementation 🚀

