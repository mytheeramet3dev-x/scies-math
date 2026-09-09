# Lazy Evaluation Expression Trees (`lazy`)

The `lazy` module constructs symbolic expression trees for linear algebra pipelines, chaining matrix operations without computing intermediate buffers and fusing arithmetic into a single memory pass upon `.eval()`.

---

## 1. Problem Formulation: Eager vs. Lazy Allocation

Consider computing $R = 0.5 \cdot (2A + B)$:

### Eager Evaluation (3 Allocations)
```rust
// Step 1: Allocates temporary Vec for 2 * A
let temp1 = a.scale(2.0);
// Step 2: Allocates temporary Vec for (temp1 + B)
let temp2 = temp1.add(&b)?;
// Step 3: Allocates temporary Vec for result
let r = temp2.scale(0.5);
```
Each temporary allocation stresses the memory allocator and invalidates CPU L1/L2 caches for large matrices.

### Lazy Evaluation (Zero Temporary Allocations)
```rust
use scies_math_th::lazy::lazy;

// Builds an expression tree AST in registers/stack; zero heap allocations
let r = lazy(&a).scale(2.0).add(lazy(&b)).scale(0.5).eval()?;
```
Only a single destination matrix buffer is allocated when `.eval()` is invoked. Each element $(i, j)$ is computed in a fused loop:

$$R_{ij} = 0.5 \cdot (2 A_{ij} + B_{ij})$$

---

## 2. Supported Operations

| Expression | Method | Fused Behavior |
| :--- | :--- | :--- |
| $A + B$ | `.add(other)` | Pointwise sum |
| $A - B$ | `.sub(other)` | Pointwise difference |
| $\alpha A$ | `.scale(alpha)` | Pointwise scalar scaling |
| $-A$ | `.neg()` | Pointwise negation |
| $A^T$ | `.transpose()` | Coordinate swapped lookup without data movement |
| $A \odot B$ | `.hadamard(other)` | Pointwise element product |
| $f(A)$ | `.map(f)` | Arbitrary elementwise closure evaluation |
| $A B + C$ | `.matmul_add(a, b, c)`| Fused matrix multiply-accumulate (GEMM) |

---

## 3. Code Example

```rust
use scies_math_th::generic::Mat;
use scies_math_th::lazy::lazy;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Mat::<f64>::from_fn(4, 4, |r, c| (r + c) as f64);
    let b = Mat::<f64>::from_fn(4, 4, |r, c| (r * c) as f64);

    // Fused expression: 2.0 * (A + B)^T
    let res = (lazy(&a) + lazy(&b)).transpose().scale(2.0).eval()?;

    println!("Fused result at (1, 2): {:.1}", res.get(1, 2)?);

    Ok(())
}
```
