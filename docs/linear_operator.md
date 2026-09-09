# Matrix-Free Linear Operators (`linear_operator`)

The `linear_operator` module defines abstractions for matrix-free linear algebra. It allows iterative methods (such as conjugate gradient or Lanczos) to operate without explicitly forming dense or sparse matrices in memory.

---

## 1. The `LinearOperator` Trait

```rust
pub trait LinearOperator {
    /// Dimension of the output space (number of rows).
    fn rows(&self) -> usize;

    /// Dimension of the input space (number of columns).
    fn cols(&self) -> usize;

    /// Evaluates the forward action y = A * x.
    fn apply(&self, x: &[f64]) -> SciResult<Vec<f64>>;

    /// Evaluates the adjoint action y = A^T * x. Default is self-adjoint.
    fn apply_adjoint(&self, x: &[f64]) -> SciResult<Vec<f64>> {
        self.apply(x)
    }
}
```

### Standard Implementations
- `DynamicMatrix`: Dense matrix-vector product $A x$ and $A^T x$.
- `SymmetricMatrix`: Packed lower-triangular symmetric matrix-vector product.
- `SparseMatrixCsr`: Compressed Sparse Row product $A x$ and transpose $A^T x$.

---

## 2. Composite Operators

- `SumOperator<'a, Op1, Op2>`: Represents the sum $(A + B) x = A x + B x$ without computing matrix additions in memory.

---

## 3. Code Example

```rust
use scies_math_th::linear_algebra::DynamicMatrix;
use scies_math_th::linear_operator::{LinearOperator, SumOperator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0])?;
    let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0])?;

    // Matrix-free sum operator: (A + B)
    let sum_op = SumOperator { op1: &a, op2: &b };
    let x = vec![1.0, 1.0];

    // Evaluates A*x + B*x without allocating matrix (A + B)
    let y = sum_op.apply(&x)?;
    assert_eq!(y, vec![14.0, 22.0]);

    Ok(())
}
```
