# N-Dimensional Tensors and Decompositions (`tensor`)

The `tensor` module provides $N$-dimensional array storage, multidimensional indexing, tensor contractions (Einstein summation), Higher-Order SVD (HOSVD / Tucker decomposition), and CANDECOMP/PARAFAC (CP) decomposition via Alternating Least Squares (ALS).

---

## 1. Storage and Indexing

A `Tensor` stores elements in a contiguous `Vec<f64>` using row-major (C-order) strides:

$$\text{offset}(i_0, i_1, \dots, i_{d-1}) = \sum_{k=0}^{d-1} i_k \cdot s_k, \quad s_k = \prod_{j=k+1}^{d-1} n_j$$

### Core Methods
- `Tensor::zeros(shape: &[usize])`: Allocates zero-initialized tensor.
- `Tensor::from_fn(shape: &[usize], f: F)`: Fills tensor via index closure `Fn(&[usize]) -> f64`.
- `get(indices: &[usize]) -> Option<f64>` / `set(indices: &[usize], val: f64)`.
- `reshape(new_shape: &[usize])`: Re-interprets strides without copying memory.
- `transpose_axes(permutation: &[usize])`: Arbitrary dimensional axis permutation.
- `contract(other: &Tensor, self_axis: usize, other_axis: usize)`: Contraction along specified modes.

---

## 2. Tensor Decompositions

### Higher-Order SVD (HOSVD / Tucker Decomposition)
Decomposes a 3-way tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$ into orthogonal factor matrices and a core tensor $\mathcal{G}$:

$$\mathcal{X} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$$

where each factor matrix $U^{(n)}$ contains the left singular vectors of the mode-$n$ unfolding of $\mathcal{X}$.

### CANDECOMP/PARAFAC Decomposition (`cp_als`)
Approximates $\mathcal{X}$ as the sum of $R$ rank-one component tensors:

$$\mathcal{X} \approx \sum_{r=1}^R a_r \circ b_r \circ c_r$$

Optimized via Alternating Least Squares (ALS) solving linear subproblems iteratively for each factor matrix.

---

## 3. Code Example

```rust
use scies_math_th::tensor::{Tensor, hosvd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3D tensor of shape [4, 4, 4]
    let t = Tensor::from_fn(&[4, 4, 4], |idx| {
        (idx[0] + 2 * idx[1] + 3 * idx[2]) as f64
    });

    println!("Original shape: {:?}", t.shape());
    println!("Value at [1, 2, 3]: {:.1}", t.get(&[1, 2, 3]).unwrap()); // 1 + 4 + 9 = 14.0

    // Compute Higher-Order SVD
    let tucker = hosvd(&t)?;
    println!("Factor matrix 1 rows: {}", tucker.factors[0].rows());
    println!("Core tensor shape:   {:?}", tucker.core.shape());

    Ok(())
}
```
