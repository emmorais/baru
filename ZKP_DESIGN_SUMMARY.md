# ZKP Integration: Simplified Design Summary

## Core Insight: Witness is Automatic

Instead of requiring users to manually specify what's public vs private, we use a **principle of least annotation**:

### Classification Rules (in order)

1. **Witness (Secret) = Automatically Detected**
   - Variables from `(share ...)` calls (MPC shared values)
   - Variables marked with `(private ...)` annotation
   
2. **Public = Everything Else**
   - All other variables in the expression
   - No annotation needed; it's the default

### Why This Works

```scheme
; Example: Prove a secret computation
; Using let for local bindings (define is not yet implemented)

(let ((secret 42)           ; Witness (marked private)
      (multiplier 2)        ; Public (not marked)
      (result (* secret multiplier)))
  
  ; Proof API: only specify the private variables
  (define proof (prove result (private secret)))
  ;                        ^^^^^^^^^^^^^^^^
  ;                        Only this needs marking!
  
  ; Verification: only public values needed
  (verify proof result (list multiplier))
  ; Verifier cannot learn 'secret', only that it satisfies constraints
)
```

**Note:** Future versions will support `(define ...)` for top-level bindings. 
For now, all computations use `let` for scoped variable binding.

### Advantages

✅ **Minimal Syntax** - Only mark secrets with `(private ...)`  
✅ **Intuitive** - Public is default, matches real-world thinking  
✅ **Automatic MPC Integration** - Shared variables auto-detected as witness  
✅ **Clean API** - prove/verify have simple, symmetric signatures  
✅ **Backward Compatible** - Doesn't break existing Baru code  

---

## API Design

### Type Definitions

```rust
/// Public values (revealed in proof)
pub struct PublicValues {
    pub values: HashMap<String, Fr>,
}

/// ZK Proof (hides witness, proves constraints)
pub struct ZKProof {
    pub data: Vec<u8>,
    pub backend_name: String,
}

/// Backend trait (abstraction for Binius, Plonky3, etc.)
pub trait ZKPBackend {
    fn compile_circuit(
        &self,
        expr: &Expr,
        private_vars: &[String],  // Only these are witness
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
        public_values: &PublicValues,  // Only public!
    ) -> Result<bool, String>;
}
```

### REPL Usage

```scheme
;; Define a computation using let for scoped bindings
(let ((secret-a 42)           ; Witness (marked private)
      (secret-b 10)           ; Witness (marked private)
      (public-factor 2)       ; Public (not marked)
      (result (+ (* secret-a secret-b) public-factor)))
  
  ;; Generate proof (mark which vars are secret)
  (let ((proof (prove result (private secret-a secret-b))))
    
    ;; Save for transmission
    (proof-save proof "proof.json")
    
    ;; Remote party: verify without learning secrets
    (let ((proof-loaded (proof-load "proof.json"))
          (public-map (list (list "public-factor" public-factor))))
      (verify proof-loaded result public-map))))
  
;; → #t if computation was correct
;; → #f if someone cheated
;; → Verifier learns NOTHING about secret-a or secret-b
```

**Note:** Uses `let` for scoped bindings (current system). 
Future versions will support `(define ...)` for top-level definitions.

---

## Architecture Changes

### Module Structure

```
src/
├── interpreter/
│   ├── ast.rs          (+ is_private: bool flag on Expr)
│   ├── parser.rs       (+ parse (private expr))
│   └── eval.rs         (+ prove/verify builtins)
│
├── zkp/                (NEW)
│   ├── mod.rs          (types + ZKPBackend trait)
│   ├── binius_backend.rs     (Binius64 implementation)
│   └── circuit_compiler.rs   (Expr → Circuit compilation)
│
└── crypto/             (unchanged)
```

### Integration with MPC

ZKPs are **orthogonal** to MPC:

```
MPC Backend (ABY3):      Ensures Privacy
                         (computation stays secret)
                         ↓
                    Compute shared result
                         ↓
ZKP Backend (Binius):   Ensures Integrity
                        (prove correct computation)
```

Example use case:
1. Party 0, 1, 2 use ABY3 to compute on shared values
2. Each party independently proves their local computation was honest
3. Other parties verify proofs, confirming no cheating
4. Result: Private computation that's cryptographically verified to be correct

---

## Why Simplified Design Matters

### Problem (Old Approach)
- "Mark every input/output as public or private"
- "List witness variables, list public variables"
- Cognitive load: users think about what NOT to protect
- Error-prone: accidentally marking secrets as public

### Solution (New Approach)
- "Mark only the secrets with (private ...)"
- "Everything else is public"
- Cognitive load: users think about what TO protect
- Safe: if you forget (private ...), it just becomes public (detection is possible)

### Analogy
Like file permissions: `chmod 644` (mark readable) vs `chmod 600` (mark private)
Better UX: "Mark sensitive files as private" instead of "Mark all files as readable"

---

## Implementation Phases

| Phase | Task | Effort | Status |
|-------|------|--------|--------|
| 6A | Add `is_private: bool` to Expr AST | ~100 LOC | Design ✓ |
| 6B | Define ZKPBackend trait | ~150 LOC | Design ✓ |
| 6C | Implement Binius64Backend | ~500 LOC | Next |
| 6D | Circuit compiler (Expr → gates) | ~800 LOC | Next |
| 6E | REPL builtins (prove/verify) | ~200 LOC | Next |
| 6F | Serialization (proof storage) | ~150 LOC | Next |

**Total: ~1900 LOC, 4-5 weeks**

---

## Key Design Decisions

1. **Witness is automatic** ← Everything derives from this
2. **Private annotation is optional for MPC variables** ← They're auto-detected
3. **Single annotation: (private ...)** ← No public annotation needed
4. **Backend abstraction via trait** ← Enables future systems (Plonky3, etc.)
5. **Proof is portable** ← Anyone can verify with just public values

---

## Security Model

### What Each Party Learns

**Prover (has secrets):**
- Computes result
- Generates proof using private values

**Verifier (no secrets):**
- Receives proof + public values
- Cannot learn private values from proof (zero-knowledge)
- Can verify: "The computation was done correctly"
- Cannot verify: "The computation was done honestly" (requires trust in prover code)

### What's Proven

✓ **Circuit Satisfaction:** Given these public inputs, these outputs satisfy the circuit  
✗ **Not Proven:** Private inputs match some real-world data (that's a social layer)

### Example

```
Circuit: result = secret * 2

Proof: "I have some secret where result = secret * 2"

Verifier sees:
  - public: result = 42
  - proof
  → Verifies: "OK, so secret could be 21"
  → Cannot verify: "Is 21 the actual secret?" (that requires trusting the prover)
```

---

## Next Steps

1. **Review & Feedback** - Does this design make sense?
2. **Phase 6A** - Implement AST changes (is_private flag)
3. **Phase 6B** - Implement ZKPBackend trait + mock backend
4. **Phase 6C** - Integrate Binius64 (main work)
5. **Phase 6D** - Build circuit compiler (main complexity)
6. **Phase 6E** - Add REPL builtins
7. **Phase 6F** - Add serialization for distributed proving

