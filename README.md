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

- Generic matrices and vectors
- Linear algebra
- Statistics and inference
- Probability and distributions
- Numerical optimization
- ODE, PDE, and signal processing
- Random number generation and Monte Carlo tools
- Tensor and time-series utilities
- Automatic differentiation
- Geometry, graph, and scientific I/O helpers

## Feature flags

- `serde`: serialization support
- `parallel`: parallel implementations where available
- `bench`: benchmark-only helpers

## Documentation structure

- `README.en.md`: full English overview
- `README.th.md`: full Thai overview
- `docs/`: module-level reference pages

## Installation

```toml
[dependencies]
scies-math-th = "0.2.3"

# With serde support:
scies-math-th = { version = "0.2.3", features = ["serde"] }
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

