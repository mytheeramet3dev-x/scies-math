# Packed Symmetric Matrices (`symmetric`)

The `symmetric` module implements memory-efficient packed lower-triangular storage for symmetric matrices $A \in \mathcal{S}^n$.

---

## 1. Storage Layout

Because $A_{ij} = A_{ji}$, only the lower-triangular elements $i \ge j$ are stored in contiguous memory:

$$\text{index}(i, j) = \frac{i(i + 1)}{2} + j \quad (0 \le j \le i < n)$$

Total elements stored: $N_{\text{packed}} = \frac{n(n + 1)}{2}$, saving $\approx 50\%$ memory over dense storage.

---

## 2. Operations

- `trace_inner_product(&self, other: &Self) -> SciResult<f64>`: Computes Frobenius inner product $\langle A, B \rangle = \text{Tr}(A B) = \sum_i A_{ii} B_{ii} + 2 \sum_{i > j} A_{ij} B_{ij}$ directly in packed form.
- `quadratic_form(&self, x: &[f64]) -> SciResult<f64>`: Computes $x^T A x$.
- `mul_vector(&self, x: &[f64]) -> SciResult<Vec<f64>>`: Matrix-vector product $A x$.
- `to_dense(&self) -> SciResult<DynamicMatrix>`: Unpacks into full $n \times n$ dense representation.
- `from_dense(dense: &DynamicMatrix) -> SciResult<Self>`: Validates symmetry within tolerance and converts to packed format.

---

## 3. Code Example

```rust
use scies_math_th::symmetric::SymmetricMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3x3 matrix:
    // [ 4  1 -2 ]
    // [ 1  5  0 ]
    // [-2  0  6 ]
    let packed_data = vec![
        4.0,       // (0,0)
        1.0, 5.0,  // (1,0), (1,1)
       -2.0, 0.0, 6.0 // (2,0), (2,1), (2,2)
    ];

    let mat = SymmetricMatrix::new(3, packed_data)?;

    let x = vec![1.0, 2.0, 3.0];
    let y = mat.mul_vector(&x)?;
    println!("A * x = {:?}", y); // [0.0, 11.0, 16.0]

    let quad = mat.quadratic_form(&x)?;
    println!("x^T A x = {}", quad); // 70.0

    Ok(())
}
```
