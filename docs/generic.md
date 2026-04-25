# `generic` Module Documentation

Phase 2 — Generic scalar types: `Scalar` trait, `Mat<T>`, `SMatrix<T,R,C>`.

# Overview

| Type | Dims | Scalar | Use case |
|---|---|---|---|
| `Mat<T>` | Runtime | `T: Scalar` | General-purpose dynamic matrix |
| `SMatrix<T,R,C>` | Compile-time | `T: Scalar` | Zero-overhead fixed-size matrix |
| `Vec1<T>` | Runtime | `T: Scalar` | Dense column vector |

# Quick start
```rust
use scies_math::generic::{Mat, SMatrix, Scalar};

// Dynamic f32 matrix
let a: Mat<f32> = Mat::zeros(3, 3);

// Compile-time 4×4 f64 matrix (zero-cost, no heap alloc)
let eye: SMatrix<f64, 4, 4> = SMatrix::identity();
```