# `linear_algebra` Module Documentation

Linear algebra — vectors, matrices, and decompositions.

# Overview

This module provides:

- Fixed-size types: [`Vector2`], [`Vector3`], [`Matrix2`], [`Matrix3`]
- Dynamic-size: [`DynamicMatrix`] (heap-allocated, row-major)
- Decompositions: [`LuDecomposition`], [`QrDecomposition`],
  [`SingularValueDecomposition`], [`CholeskyDecomposition`]
- Factorization-based solvers and norm / condition-number utilities

# Fixed-size types

[`Vector2`] and [`Vector3`] are `Copy` structs. All arithmetic uses
operator overloads (`+`, `-`, `*`).

```rust
use scies_math_th::linear_algebra::{Vector2, Vector3};

let a = Vector3::new(1.0, 0.0, 0.0);
let b = Vector3::new(0.0, 1.0, 0.0);
let c = a.cross(b);          // (0, 0, 1)
let d = a.dot(b);            // 0.0
```

[`Matrix2`] and [`Matrix3`] are likewise `Copy`:

```rust
use scies_math_th::linear_algebra::Matrix3;

let m = Matrix3::identity();
let v = scies_math_th::linear_algebra::Vector3::new(1.0, 2.0, 3.0);
let r = m.mul_vector(v);
```

# DynamicMatrix

The main workhorse for arbitrary-shaped computations.  Data is stored in
row-major order in a `Vec<f64>`.

## Construction

```rust
use scies_math_th::linear_algebra::DynamicMatrix;

let a = DynamicMatrix::zeros(4, 4).unwrap();
let b = DynamicMatrix::identity(4).unwrap();
let c = DynamicMatrix::new(2, 3, vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
```

## Basic operations

| Method | Description |
|---|---|
| `get(r, c)` | Element access |
| `set(r, c, v)` | Element mutation |
| `transpose()` | Returns a new transposed matrix |
| `trace()` | Sum of diagonal entries |
| `determinant()` | Computed via partial-pivot Gaussian elimination |
| `norm_fro()` | Frobenius norm √(ΣΣ aᵢⱼ²) |
| `norm_1()` | Maximum absolute column sum |
| `norm_inf()` | Maximum absolute row sum |

## Matrix multiply (auto-dispatched)

`mul_matrix` automatically selects the fastest backend based on size:

| Size (max dim) | Backend |
|---|---|
| < 64 | Naive i-k-j |
| 64–511 | Cache-tiled (64 × 64 blocks) |
| ≥ 512 | Strassen O(n^2.807) |

For parallel workloads use `mul_matrix_par(other, n_threads)`.

## Decompositions

### LU (partial pivoting)

```rust
# use scies_math_th::linear_algebra::DynamicMatrix;
let a = DynamicMatrix::new(3, 3, vec![
    2.0, 1.0, 1.0,
    4.0, 3.0, 3.0,
    8.0, 7.0, 9.0,
]).unwrap();
let lu = a.lu_decompose().unwrap();
let x  = lu.solve(&[1.0, 1.0, 1.0]).unwrap();
```

### QR (Gram-Schmidt)

Returns orthonormal Q (m×n) and upper-triangular R (n×n).
Used for least-squares regression via `least_squares(rhs)`.

### SVD

Power-iteration Golub-Reinsch variant.  Returns U, Σ (as `Vec<f64>`),
and Vᵀ sorted by descending singular values.

```rust
# use scies_math_th::linear_algebra::DynamicMatrix;
let a = DynamicMatrix::new(3, 3, vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]).unwrap();
let svd = a.singular_value_decompose(1e-10, 1000).unwrap();
println!("{:?}", svd.singular_values); // [16.12, 1.07, ~0]
```

### Cholesky (Banachiewicz)

Requires a symmetric positive-definite matrix.  Returns lower triangular
L such that A = L · Lᵀ.  Errors with `DomainError` when A is not
positive-definite.

## Advanced utilities

| Method | Description |
|---|---|
| `inverse()` | LU-based matrix inverse |
| `pseudoinverse(tol, iters)` | Moore-Penrose via SVD |
| `rank_estimate(tol, iters)` | Numerical rank (SVD thresholding) |
| `condition_number_2(tol, iters)` | Spectral κ₂ = σ_max / σ_min |
| `condition_number_1()` | 1-norm κ₁ = ‖A‖₁ · ‖A⁻¹‖₁ |
| `qr_eigenvalues(tol, iters)` | QR iteration for eigenvalues |
| `power_iteration(v0, tol, iters)` | Dominant eigenvalue/vector |
| `jacobi_eigendecomposition(tol, iters)` | Symmetric matrices only |

## Sparse and iterative solvers

See [`crate::sparse`] for CSR/CSC formats and iterative solvers
(Conjugate Gradient, GMRES, BiCGSTAB).
