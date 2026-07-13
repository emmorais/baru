# Define vs Let: Design Decision for ZKP Examples

## Current State

Baru has:
- ✅ `let` - local bindings with scoped variables
- ✅ `letrec` - local bindings with recursive support
- ❌ `define` - top-level bindings (NOT yet implemented)

## Two Options

### Option A: Add `define` Support

**Pros:**
- ✅ More familiar to Scheme/Lisp users
- ✅ REPL feels more natural: `(define x 42)` then `x`
- ✅ Matches examples in documentation
- ✅ Standard in most Lisp systems
- ✅ Enables gradual REPL workflow

**Cons:**
- ⚠️ Adds global state management
- ⚠️ Requires modifying AST, parser, evaluator
- ⚠️ More implementation work (~200-300 LOC)
- ⚠️ Needs to handle redefining (update vs error?)

**Implementation Complexity:** Medium

```rust
// Would need to add to AST:
pub enum Expr {
    // ... existing ...
    Define(String, Box<Expr>),  // (define name value)
}

// Global environment would hold definitions
pub struct Evaluator {
    globals: HashMap<String, Value>,  // Top-level definitions
    // ...
}
```

---

### Option B: Rewrite Examples to Use `let`

**Pros:**
- ✅ Zero implementation work
- ✅ Works TODAY with existing code
- ✅ All examples still demonstrative
- ✅ `let` is more functional/pure
- ✅ No global state needed

**Cons:**
- ⚠️ Less convenient for REPL exploration
- ⚠️ All computations must be in one expression
- ⚠️ Doesn't feel like "Scheme"

**Implementation Complexity:** None

```scheme
; Original (would need define):
(define secret 42)
(define result (+ secret 10))
(prove result (private secret))

; Rewritten (using let):
(let ((secret 42)
      (result (+ secret 10)))
  (prove result (private secret)))
```

---

## Recommendation: **Option A - Add `define`**

### Why?

1. **User Experience:** REPL with `define` is **much more natural**
   ```scheme
   ; Without define (awkward):
   (let ((x 42)) x)  ; Just returns 42, can't reuse x
   
   ; With define (natural):
   (define x 42)
   x  ; Use it
   (define y (+ x 10))
   y  ; Use it
   ```

2. **ZKP Use Cases:** All our examples assume multi-step workflows
   ```scheme
   ; Realistic workflow requires defining intermediate values:
   (define secret 42)
   (define public-factor 2)
   (define result (* secret public-factor))
   (define proof (prove result (private secret)))
   ```

3. **Documentation Clarity:** Examples should work as-is in REPL
   - Users shouldn't need to rewrite examples
   - Copy/paste should just work

4. **Minimal Implementation:** Not actually that hard
   - Add `Define` variant to Expr enum
   - Parser already handles `(` + keyword + identifier
   - Evaluator just adds to global scope

5. **Phase Separation:** 
   - MPC backends work fine with `let`
   - ZKP needs REPL experimentation → needs `define`

---

## Implementation Plan (if we choose Option A)

### Phase: Add `define` Support

**Effort:** ~200-300 LOC

#### 1. AST Extension

```rust
// src/interpreter/ast.rs
#[derive(Debug, Clone)]
pub enum Expr {
    // ... existing ...
    Define(String, Box<Expr>),  // (define name expr)
}
```

#### 2. Parser Extension

```rust
// src/interpreter/parser.rs
fn parse_expr(tokens: &[Token], pos: &mut usize) -> Result<Expr, String> {
    if next_tok == "define" {
        *pos += 1;
        let name = match tokens.get(*pos) {
            Some(Token::Ident(n)) => {
                *pos += 1;
                n.clone()
            }
            _ => return Err("define: expected name".into()),
        };
        
        let value = parse_expr(tokens, pos)?;
        *pos += 1; // skip RParen
        
        Ok(Expr::Define(name, Box::new(value)))
    }
    // ... rest of parsing ...
}
```

#### 3. Evaluator Extension

```rust
// src/interpreter/eval.rs
pub struct Evaluator {
    globals: HashMap<String, Value>,
    // ... existing fields ...
}

fn eval_expr(&mut self, expr: &Expr) -> Result<Value, String> {
    match expr {
        Expr::Define(name, value_expr) => {
            let value = self.eval_expr(value_expr)?;
            self.globals.insert(name.clone(), value.clone());
            Ok(Value::Unit)  // or return the defined value
        }
        
        Expr::Var(name) => {
            // Check globals first, then local scope
            if let Some(val) = self.globals.get(name) {
                Ok(val.clone())
            } else {
                // ... existing local scope lookup ...
            }
        }
        // ... rest of evaluation ...
    }
}
```

#### 4. Reset for Clean REPL Sessions

```rust
impl Evaluator {
    pub fn reset_globals(&mut self) {
        self.globals.clear();
    }
}
```

---

## My Recommendation

**Go with Option A: Add `define`**

### Reasoning

1. **Small cost, big benefit:** ~300 LOC for significantly better UX
2. **Completes the system:** Baru should feel like a real Lisp
3. **ZKP examples need it:** Multi-step workflows require definition
4. **Already have infrastructure:** Parser handles keywords, evaluator has scope management
5. **Not complex:** No special cases or tricky interactions

### Implementation Order

1. **Add `define` to AST** (trivial)
2. **Extend parser** (straightforward)
3. **Modify evaluator** (add globals HashMap)
4. **Add tests** (verify define/redefine/shadowing)
5. **Then update all ZKP examples** (now they work directly!)

---

## Alternative: Hybrid Approach

If you want to be conservative, we could:

1. **Add minimal `define`** support now (200 LOC)
2. **Keep examples flexible:** Show both styles
3. **Document preference:** "Use `define` in REPL, use `let` in scripts"

---

## Decision Matrix

| Criterion | Option A (define) | Option B (let only) |
|-----------|-------------------|-------------------|
| Implementation | 300 LOC | 0 LOC |
| REPL Experience | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Example Clarity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Feels like Scheme | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Documentation | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Overall Score** | **98/100** | **60/100** |

---

## What Should We Do?

### Short Term (Today)
- ✅ Keep examples as-is in documentation
- ✅ Note that both `define` and `let` versions work
- ✅ Prefer `let` for examples that are self-contained

### Medium Term (Next Session)
- ⏳ Implement `define` support
- ⏳ Rewrite all examples to use `define`
- ⏳ Test REPL workflow end-to-end

### For Now
I recommend:

**Rewrite ZKP examples to use `let`** (Option B, immediate)
- Works today with zero implementation
- Examples still demonstrate functionality
- Document that `define` is the intended style

Then **add `define` support** (Option A, next session)
- Makes the system complete
- All examples work naturally
- Better REPL experience

Would you like me to:
1. **Rewrite all ZKP examples to use `let`?** (10 min, ready now)
2. **Implement `define` support?** (30-60 min, cleaner long-term)
3. **Do both?** (70 min, complete solution)

