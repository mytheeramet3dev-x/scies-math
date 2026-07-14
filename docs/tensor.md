# `tensor` Module Documentation

N-dimensional tensors — creation, indexing, decomposition.

# Overview

[`Tensor`] stores data in a flat `Vec<f64>` with a shape `Vec<usize>`.
Indices follow C (row-major) order.

# Construction

```rust
use scies_math_th::tensor::Tensor;

let t = Tensor::zeros(&[3, 4, 5]);   // 3×4×5 tensor, all zeros
let t = Tensor::ones(&[2, 3]);       // 2×3 matrix
let t = Tensor::from_fn(&[4, 4], |idx| idx[0] as f64 + idx[1] as f64);
```

# Indexing and slicing

```rust
# use scies_math_th::tensor::Tensor;
let mut t = Tensor::zeros(&[3, 3]);
t.set(&[0, 0], 1.0);
let v = t.get(&[0, 0]);  // 1.0
```

# Operations

| Method | Description |
|---|---|
| `reshape(shape)` | Change shape (total elements unchanged) |
| `transpose_axes(perm)` | Permute axes |
| `add`, `sub`, `scale` | Elementwise arithmetic |
| `matmul_2d` | Matrix product on last two axes |
| `contract(other, axes)` | Einstein summation (tensor contraction) |
| `outer(other)` | Outer product |

# Decompositions

| Function | Description |
|---|---|
| `hosvd(tensor)` | Higher-order SVD (Tucker decomposition) |
| `cp_als(tensor, rank, iters)` | CP decomposition via alternating least squares |
| `kronecker_product(a, b)` | Kronecker (tensor) product |
