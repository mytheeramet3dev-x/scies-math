# scies-math-th

An extensible, dependency-free mathematical toolkit for Rust.

This repository now ships two full documents:
- [English documentation](README.en.md)
- [เอกสารภาษาไทย](README.th.md)

## Package

- Crates name: `scies-math-th`
- Rust import path: `scies_math_th`
- Default build: zero external dependencies

## What it provides

- Exact rational arithmetic and interval bounds (`exact`)
- Symbolic expression trees, canonical simplification, and calculus (`symbolic`)
- SI 7-base dimensional analysis and type-safe quantities (`units`)
- Mathematical provenance tracking and Lean 4 export bridge (`verification`)
- Semidefinite programming (ADMM and Mehrotra Primal-Dual IPM) (`sdp`, `sdp_ipm`)
- Moment-SOS hierarchies, CNF/3-SAT relaxation, and SDPA I/O (`moment`)
- Generic matrices, packed symmetric matrices, and linear operators (`linear_algebra`, `symmetric`, `linear_operator`)
- Statistics, inference, and distributions (`statistics`, `inference`, `distributions`)
- Numerical optimization, ODE/PDE solvers, and signal processing
- Random number generation and Monte Carlo tools
- Tensor, quaternions, geometry, and graph algorithms
- Automatic differentiation (forward and reverse AD)

## Feature flags

- `serde`: serialization support
- `parallel`: parallel implementations where available
- `bench`: benchmark-only helpers

## Documentation structure

- `README.en.md`: full English overview
- `README.th.md`: full Thai overview
- `docs/numerical_reliability.md`: mathematical validity hierarchy & KKT verification guidelines
- `docs/`: module-level reference pages

## Installation

```toml
[dependencies]
scies-math-th = "0.3.0"

# With serde support:
scies-math-th = { version = "0.3.0", features = ["serde"] }
```

## Quick example

```rust
use scies_math_th::generic::{Mat, SMatrix};
use scies_math_th::lazy::lazy;

fn main() {
    let a = Mat::<f64>::from_fn(2, 2, |r, c| (r + c) as f64);
    let b = SMatrix::<f64, 2, 2>::identity();
    let _result = (lazy(&a) + lazy(&a)).scale(0.5).eval().unwrap();
    let _ = b;
}
```

