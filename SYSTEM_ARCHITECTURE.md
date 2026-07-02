# Baru System Architecture: MPC + ZKP Integration

## System Overview

Baru is evolving into a **hybrid system** combining:
1. **MPC (Multi-Party Computation)** via ABY3/Transport backends
2. **ZKP (Zero-Knowledge Proofs)** via Binius64/Plonky3 backends

These are **orthogonal layers** that work together.

```
┌────────────────────────────────────────────────────────┐
│              Baru Interpreter (REPL)                   │
│  - Parse Scheme/Lisp syntax                            │
│  - Evaluate expressions                                │
│  - Manage backends                                     │
└────────────────────────────────────────────────────────┘
                         ▲
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ ABY3 BE  │   │Transport │   │   GC BE  │  ← MPC Backends
    │          │   │   BE     │   │          │    (Privacy)
    └──────────┘   └──────────┘   └──────────┘
        │ shares values among parties
        │ computes securely
        │ reveals only result
        ▼
    ┌──────────────────────────────────────┐
    │   ZKP Backend Integration            │
    │  ┌──────────────┐  ┌──────────────┐  │
    │  │Binius64 BE   │  │ Plonky3 BE   │  │  ← ZKP Backends
    │  │              │  │ (future)     │  │    (Integrity)
    │  └──────────────┘  └──────────────┘  │
    │   proves correct computation         │
    │   verifies with public values        │
    └──────────────────────────────────────┘
```

---

## Use Cases

### Use Case 1: Secure Computation with Integrity Verification

**Scenario:** 3 parties compute on secret data; want to verify honesty

```scheme
;; Party 0 (Garbler/Prover)
(define my-secret 42)
(define p0 (participant-new 0))
(define result (share my-secret 1 0))  ; Share with party 1

;; Compute in MPC (ABY3)
(define intermediate (add result 100))

;; Generate ZKP proof of correct computation
(define proof (prove intermediate 
                      (private my-secret)))  ; Witness: my-secret

;; Share proof with party 1
(proof-save proof "proof-of-computation.json")

;; Party 1 (Evaluator/Verifier)
(define loaded-proof (proof-load "proof-of-computation.json"))
(define verified? (verify loaded-proof intermediate))

;; Result: Party 1 can verify party 0 computed correctly
;; WITHOUT learning the secret!
```

### Use Case 2: Distributed Proving

**Scenario:** Multiple parties prove their local computations

```scheme
;; Party 0: proves their local computation
(define proof0 (prove local-result0 (private secret0)))

;; Party 1: proves their local computation
(define proof1 (prove local-result1 (private secret1)))

;; Party 2: verifies both proofs
(define valid? (and (verify proof0 ...)
                    (verify proof1 ...)))
```

### Use Case 3: Privacy-Preserving Audit Trail

**Scenario:** Prove computation without revealing data

```scheme
;; Compute secret function
(define f (lambda (x) (+ (* x x) 42)))
(define secret-input 7)
(define secret-output (f secret-input))

;; Prove "output = f(input)" without revealing either
(define proof (prove secret-output (private secret-input)))

;; Auditor can verify computation, learns nothing about data
(verify proof secret-output)  ; Works!
(verify proof 1000)           ; Fails!
```

---

## Integration Points

### 1. Shared Variables are Auto-Witness

```scheme
;; MPC creates shared variables
(define secret-share (share 42 1 0))  ; Creates shared value

;; In ZKP context, this is automatically witness
(prove expr (private secret-share))
;; → secret-share is marked as both shared AND private
;; → Automatically detected as witness by circuit compiler
```

### 2. Proof Storage Enables Distributed Verification

```scheme
;; Prover (Party 0) on Machine A
(define proof (prove result (private secret)))
(proof-save proof "proof.bin")
; Transmit proof.bin over network

;; Verifier (Party 1) on Machine B
(define proof (proof-load "proof.bin"))
(verify proof result public-values)
; Works even without knowing prover's secrets!
```

### 3. Orthogonal Error Handling

```
MPC Error:
  → Party disconnected
  → Communication failure
  → Reconstruction error

ZKP Error:
  → Circuit compilation failed
  → Proof generation failed
  → Verification failed (cheating detected!)

Both can occur independently
```

---

## Backend Architecture

### MPC Backend Trait (Existing)

```rust
pub trait Backend {
    fn share_secret(&mut self, value: Fr, nbits: usize, owner: usize) -> ShareRef;
    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef;
    fn reveal(&mut self, a: &ShareRef, ...) -> Fr;
    // ... 15+ methods
}

// Implementations: ABY3Backend, TransportBackend, GCBackend
```

### ZKP Backend Trait (New)

```rust
pub trait ZKPBackend {
    fn compile_circuit(
        &self,
        expr: &Expr,
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
}

// Implementations: Binius64Backend, Plonky3Backend (future)
```

**Key Difference:**
- **MPC Backend:** Manages shared state, protocols between parties
- **ZKP Backend:** Stateless; proves computation, verifies independently

---

## Data Flow Example

### End-to-End: MPC + ZKP Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Setup (using let for scoped bindings)               │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ (let ((p0 (participant-new 0))                        │   │
│ │       (secret 42))                                    │   │
│ │   ... computation ...                                 │   │
│ │ )                                                     │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: MPC Computation (ABY3)                              │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ (let ((shared (share secret 1 0))                     │   │
│ │       (result (add shared 100)))                      │   │
│ │   ... rest of computation ...                         │   │
│ │ )                                                     │   │
│ │ → secret distributed as shares:                       │   │
│ │   Party 0: (share0, mask0)                            │   │
│ │   Party 1: (share1, mask1)                            │   │
│ │   Reconstruction: share0 XOR share1 XOR ...           │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: ZKP Proof Generation                                │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ (let ((proof (prove result (private secret))))        │   │
│ │   ... use proof ...                                   │   │
│ │ )                                                     │   │
│ │ Circuit compiler:                                     │   │
│ │ ┌─────────────────────────────────────────────────┐   │   │
│ │ │ 1. Analyze AST: find all variables             │   │   │
│ │ │ 2. Mark 'secret' as witness                    │   │   │
│ │ │ 3. Mark 'result', '100' as public              │   │   │
│ │ │ 4. Emit gates:                                 │   │   │
│ │ │    - Witness input: secret (hidden in proof)   │   │   │
│ │ │    - Public input: result, 100                 │   │   │
│ │ │    - Constraint: result = secret + 100         │   │   │
│ │ │ 5. Return circuit definition                   │   │   │
│ │ └─────────────────────────────────────────────────┘   │   │
│ │ Prover:                                               │   │
│ │ ┌─────────────────────────────────────────────────┐   │   │
│ │ │ 1. Compile circuit to Binius format            │   │   │
│ │ │ 2. Generate Binius64 proof                     │   │   │
│ │ │ 3. Serialize proof to bytes                    │   │   │
│ │ │ → proof.data = [...]                           │   │   │
│ │ └─────────────────────────────────────────────────┘   │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Proof Storage & Transmission                        │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ (proof-save proof "computation.zkp")                  │   │
│ │ JSON:                                                 │   │
│ │ {                                                     │   │
│ │   "backend": "binius64",                              │   │
│ │   "proof_data": "0x1a2b3c...",                        │   │
│ │   "timestamp": 1719950400                             │   │
│ │ }                                                     │   │
│ │ → Transmit over network to verifier                   │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Verification (on different machine)                 │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ (let ((proof (proof-load "computation.zkp"))          │   │
│ │       (public-map '(("result" 142) ("100" 100))))     │   │
│ │   (verify proof result public-map)                    │   │
│ │   → true (if computation was correct)                   │   │
│ │   → false (if someone cheated)                          │   │
│ │                                                       │   │
│ │   Verifier learns:                                    │   │
│ │     ✓ Computation result (142)                        │   │
│ │     ✓ Computation is honest                           │   │
│ │     ✗ Original secret (42) - HIDDEN                   │   │
│ │ )                                                     │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Security Properties

### MPC Layer Provides
✓ **Privacy:** Each party's input is hidden from others  
✓ **Correctness:** Honest parties compute correct result  
✗ **Integrity:** Malicious party can cheat undetected  

### ZKP Layer Provides
✓ **Integrity:** Computation must satisfy constraints  
✓ **Verifiability:** Anyone can verify with public values  
✗ **Privacy:** If secrets are in public values, they're revealed  

### Combined: MPC + ZKP
✓ **Privacy:** MPC hides secrets from other parties  
✓ **Integrity:** ZKP proves correct computation  
✓ **Verifiability:** Public proof of honest computation  

---

## Configuration

### Select Active Backends

```scheme
;; REPL commands (future)
(set-mpc-backend "aby3")
(set-zkp-backend "binius64")

;; Or programmatic
(define mpc-be (aby3-backend-new 0))
(define zkp-be (binius64-backend-new))
```

### Future: Multiple Backends Simultaneously

```scheme
;; Generate proofs with multiple ZKP systems
(define proof-binius (prove-with "binius64" expr))
(define proof-plonky (prove-with "plonky3" expr))

;; Require both to verify (strong guarantee)
(and (verify-with "binius64" proof-binius)
     (verify-with "plonky3" proof-plonky))
```

---

## Migration Path

### Phase 1 (Current)
- ABY3 backend for 3-party MPC
- Transport backend for distributed setup

### Phase 2 (This Sprint: Phase 6)
- Add Binius64 backend for ZKP
- Integrate with existing expression evaluator
- Enable prove/verify builtins

### Phase 3 (Future)
- Add Plonky3 backend
- Support recursive proofs
- Aggregation and batching

### Phase 4 (Future)
- Add FHE backend (fully homomorphic encryption)
- Hybrid schemes (MPC + FHE + ZKP)
- Hardware acceleration (GPU proving)

---

## Performance Expectations

### MPC Operations (ABY3)
- Boolean gate: ~1ms (with communication)
- Arithmetic gate: ~5ms (with communication)
- 1000-gate circuit: ~1-5 seconds

### ZKP Operations (Binius64)
- Circuit compilation: ~10ms
- Proof generation: ~100-500ms (depending on circuit size)
- Proof verification: ~50-100ms
- Proof size: ~1-10KB

### Combined Workflow
```
Setup:           0ms (first time)
MPC compute:     1-5s
ZKP prove:       0.1-0.5s
Proof transmit:  <1ms (small proof)
Verification:    0.05-0.1s
───────────────────────
Total:           1-6s (dominated by MPC)
```

---

## Example: Auction System

### Problem
Participants bid in secret; highest bidder wins; want to verify no cheating

### Solution with MPC+ZKP

```scheme
; All operations wrapped in let for scoped bindings
(let ((p0 (participant-new 0))
      (p1 (participant-new 1))
      (p2 (participant-new 2))
      
      ; Each participant has a secret bid (secret to all others)
      (bid0 100)
      (bid1 150)
      (bid2 120)
      
      ; MPC: compute winner without revealing bids
      (bid0-shared (share bid0 1 0))
      (bid1-shared (share bid1 1 0))
      (bid2-shared (share bid2 1 0))
      
      (max-bid (max bid0-shared bid1-shared bid2-shared))
      (winner (if (eq max-bid bid0-shared) 0
                  (if (eq max-bid bid1-shared) 1 2)))
      
      ; ZKP: each participant proves their bid is honest
      (proof0 (prove bid0-shared (private bid0)))
      (proof1 (prove bid1-shared (private bid1)))
      (proof2 (prove bid2-shared (private bid2))))
  
  ; Verification: anyone can verify bids are "reasonable"
  (and (verify proof0 bid0-shared)  ; Checks bid0 in valid range
       (verify proof1 bid1-shared)  ; Checks bid1 in valid range
       (verify proof2 bid2-shared)) ; Checks bid2 in valid range
  
  ; Result:
  ; - Winner is determined (correct MPC)
  ; - Bids remain secret (MPC privacy)
  ; - No one can cheat (ZKP integrity)
)
```

---

## Conclusion

Baru's ZKP integration:
1. **Complements MPC** - adds integrity verification
2. **Maintains simplicity** - minimal annotation `(private ...)`
3. **Enables new use cases** - privacy + integrity + verifiability
4. **Extensible** - trait-based backends for future systems
5. **Production-ready** - uses proven Binius64 library

The combination creates a powerful system for secure distributed computation with cryptographic integrity guarantees.

