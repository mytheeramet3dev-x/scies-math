# scies-math-th

`scies-math-th` is a dependency-free mathematical toolkit for Rust. It targets scientific and engineering workloads that need a broad, practical, and well-documented API without pulling in a large dependency tree by default.

The crate is designed as a long-lived foundation. Its job is not to solve one narrow numerical problem, but to provide the core numerical building blocks that other biological, scientific, and engineering tools can depend on safely.

## What this crate is for

- Core numeric primitives for scientific computing
- Linear algebra and matrix workflows
- Statistics, probability, and inference
- Numerical methods such as optimization, interpolation, ODE, and PDE solving
- Signal processing and time-series analysis
- Scientific support utilities such as random sampling, autodiff, geometry, and graph algorithms

## Design principles

- Default build is zero dependency.
- Public APIs should be generic where it matters.
- Fixed-size data should use const generics where possible.
- Large data workflows should favor lazy or streaming designs.
- Hot paths should be benchmarked.
- Documentation should be complete enough that the crate can be learned from the docs alone.
- Math-heavy helpers should be reused from `scies-math-th` itself rather than duplicated elsewhere in the ecosystem.

## Architecture overview

The crate is organized around a few stable layers:

### 1. Core types

The `generic`, `complex`, `tensor`, and `lazy` modules provide the basic numeric containers and expression-building blocks used throughout the crate.

### 2. Linear algebra layer

The `linear_algebra`, `transform`, `geometry_ext`, and `sparse` modules cover matrix operations, transforms, decompositions, and geometric helpers.

### 3. Statistical layer

The `statistics`, `probability`, `distributions`, `inference`, `regression`, and `monte_carlo` modules support common statistical workflows.

### 4. Numerical methods layer

The `optimization`, `interpolation`, `ode`, `pde`, and `special_functions` modules provide numerical algorithms and function families.

### 5. Scientific workflows layer

The `signal`, `timeseries`, `autodiff`, `rng_ext`, `graph`, `nn`, and `io` modules support broader scientific workflows and data handling.

## Feature flags

### `serde`

Adds `Serialize` and `Deserialize` support for selected public types.

### `parallel`

Enables parallel implementations where the crate provides them.

### `bench`

Includes benchmark-oriented helpers and benchmark-only paths.

The default feature set is empty.

## Crate identity

- Crates.io package: `scies-math-th`
- Rust import path: `scies_math_th`
- Repository: `https://github.com/mytheeramet3dev-x/scies-math-th`
- Documentation root: `docs/`

## Package layout

```text
scies_math_th::
├── generic         # Mat<T>, SMatrix<T, R, C>, scalar traits
├── lazy            # Lazy expression trees for matrix-style computation
├── complex         # Complex number arithmetic
├── tensor          # Tensor container and decomposition helpers
├── linear_algebra  # Matrices, vectors, decomposition, solvers
├── transform       # Quaternion, rotation, and rigid transform utilities
├── geometry_ext    # Extended geometry helpers
├── sparse          # Sparse matrix and sparse solver utilities
├── statistics      # Descriptive statistics
├── probability     # Probability helpers
├── distributions   # Probability distributions
├── inference       # Hypothesis testing and inference helpers
├── regression      # Linear and regularized regression helpers
├── monte_carlo     # Monte Carlo integration and sampling
├── optimization    # Scalar optimization
├── opt_multivar    # Multivariate optimization
├── interpolation   # Interpolation utilities
├── ode             # Ordinary differential equation solvers
├── ode_ext         # Extended ODE routines
├── pde             # Partial differential equation solvers
├── pde_ext         # Extended PDE routines
├── special_functions # Erf, Gamma, Beta, Bessel and related functions
├── signal          # FFT and signal processing utilities
├── timeseries      # ARIMA, smoothing, DTW
├── autodiff        # Forward-mode automatic differentiation
├── reverse_ad      # Reverse-mode automatic differentiation
├── rng             # Core RNG utilities
├── rng_ext         # RNG engines and sampler APIs
├── graph           # Graph algorithms
├── nn              # Neural-network primitives
└── io              # Scientific I/O helpers
```

## Examples

### Generic matrix creation

```rust
use scies_math_th::generic::{Mat, SMatrix};

fn main() {
    let a = Mat::<f64>::from_fn(2, 2, |r, c| (r + c) as f64);
    let eye: SMatrix<f64, 2, 2> = SMatrix::identity();
    let _ = (a, eye);
}
```

### Lazy evaluation

```rust
use scies_math_th::generic::Mat;
use scies_math_th::lazy::lazy;

fn main() {
    let a = Mat::<f64>::from_fn(4, 4, |r, c| (r + c) as f64);
    let result = (lazy(&a) + lazy(&a)).scale(0.5).eval().unwrap();
    let _ = result;
}
```

## Installation

```toml
[dependencies]
scies-math-th = "0.2.4"

# Optional serde support:
scies-math-th = { version = "0.2.4", features = ["serde"] }
```

## Benchmarks

Benchmarking is part of the project culture, not an afterthought.

Relevant benchmark suites live under `benches/` and are meant to track:

- matrix multiplication
- RNG and sampling
- ODE solvers
- transform and decomposition workflows

## Documentation

- [Module docs](docs/README.md)
- [Thai overview](README.th.md)
- [Root overview](README.md)

The module pages are intended to be self-contained and should include examples whenever the API is public-facing and nontrivial.

## Release notes

Version `0.2.4` keeps the crate in the zero-dependency default mode while expanding the public surface and documentation quality. Future releases should continue to grow by module family rather than by ad hoc additions.

