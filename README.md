# scies-math v0.2.0

[![crates.io](https://img.shields.io/crates/v/scies-math.svg)](https://crates.io/crates/scies-math)
[![docs.rs](https://docs.rs/scies-math/badge.svg)](https://docs.rs/scies-math)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A **high-performance, dependency-free** mathematical toolkit for Rust.  
Zero external dependencies in default mode — only the Rust standard library.

## Highlights

| Module | What's inside |
|---|---|
| `perf` | Cache-tiled GEMM, Strassen O(n^2.807), multithreaded matmul |
| `generic` | `Mat<T>`, `SMatrix<T,R,C>` (stack-alloc), `Vec1<T>` — generic over `f32`/`f64` |
| `transform` | `Quaternion`, `Rotation3`, `Isometry3`, `DualQuaternion`, perspective/look-at |
| `rng_ext` | Xoshiro256\*\*, PCG64, Wyrand, MT19937 + `Sampler` API + `AliasTable` |
| `lazy` | `LazyMat` expression tree — fused ops, zero temporaries |
| `linear_algebra` | `DynamicMatrix`: LU, QR, SVD, Cholesky, pseudoinverse, sparse solvers |
| `ode` / `ode_ext` | RK4, RK45, DOP853, BDF2, Rosenbrock2, Symplectic, Adams-Bashforth |
| `pde` / `pde_ext` | Heat, Wave, Laplace, Navier-Stokes (FD), Poisson |
| `signal` / `signal_ext` | FFT, IFFT, STFT, FIR/IIR filter, Wavelet CWT, Hilbert |
| `distributions` / `_ext` | 30+ distributions: PDF, CDF, inverse CDF |
| `inference` / `_ext` | t-test, ANOVA, KS, Mann-Whitney, bootstrap CI, FDR correction |
| `regression` / `_ext` | Linear, Ridge, Lasso, Elastic Net, k-NN, Decision Tree (CART) |
| `monte_carlo` / `_ext` | MCMC: MH, Gibbs, HMC, Slice; Particle Filter; Sobol; Latin Hypercube |
| `timeseries` | AR, MA, ARMA, ARIMA, Holt-Winters, DTW, rolling stats |
| `tensor` | N-dim arrays, HOSVD, CP decomposition, Kronecker |
| `autodiff` / `reverse_ad` | Forward + reverse autodiff, Jacobian, Hessian |

## Quick Start

```toml
[dependencies]
scies-math = "0.2"

# With JSON/serde support:
scies-math = { version = "0.2", features = ["serde"] }
```

```rust
use scies_math::generic::{Mat, SMatrix};
use scies_math::lazy::lazy;
use scies_math::transform::{Quaternion, Rotation3, Isometry3};
use scies_math::rng_ext::DefaultSampler;

fn main() {
    // Generic matrix (f32 or f64)
    let a = Mat::<f64>::from_fn(4, 4, |r, c| (r + c) as f64);
    let b = Mat::<f64>::identity(4);

    // Lazy evaluation — zero temporaries
    let result = (lazy(&a) + lazy(&b)).scale(0.5).eval().unwrap();

    // Compile-time fixed-size matrix (stack-allocated, no heap)
    let eye: SMatrix<f64, 4, 4> = SMatrix::identity();

    // Quaternion rotation
    let rot = Rotation3::from_axis_angle([0.0, 1.0, 0.0], 1.57).unwrap();
    let iso = Isometry3::new(rot, [1.0, 0.0, 0.0]);
    let p   = iso.transform_point([0.0, 0.0, 0.0]);

    // High-quality sampling
    let mut rng = DefaultSampler::seeded(42);
    let samples: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
}
```

## Running Benchmarks

```bash
# Full suite (HTML report in target/criterion/)
cargo bench

# Individual benchmarks
cargo bench --bench matmul
cargo bench --bench rng
cargo bench --bench ode
cargo bench --bench transform
cargo bench --bench sampling

# Open HTML report
open target/criterion/matmul/report/index.html
```

## Feature Flags

| Flag | Enables |
|---|---|
| `serde` | `Serialize`/`Deserialize` for `Mat<T>`, `Vec1<T>`, `Quaternion`, `Isometry3` |

## Modules at a Glance

```
scies_math::
├── generic      — Mat<T>, SMatrix<T,R,C>, Vec1<T>, Scalar trait
├── perf         — matmul (tiled, Strassen, threaded), Xoshiro256**
├── lazy         — LazyMat / LazyVec expression tree
├── transform    — Quaternion, Rotation3, Isometry3, Similarity3, DualQuaternion
├── rng_ext      — Xoshiro256**, PCG64, Wyrand, MT19937, Sampler, AliasTable
├── linear_algebra — DynamicMatrix, LU, QR, SVD, Cholesky, sparse solvers
├── ode / ode_ext  — RK4..DOP853, stiff, symplectic, event detection
├── pde / pde_ext  — Heat, Wave, Laplace, Navier-Stokes
├── signal / _ext  — FFT, STFT, filters, wavelet, Hilbert
├── distributions  — 30+ distributions (PDF/CDF/iCDF)
├── inference      — hypothesis tests, bootstrap, FDR
├── regression     — linear models, k-NN, decision trees
├── monte_carlo    — MCMC, SMC, quasi-random, variance reduction
├── timeseries     — ARIMA, Holt-Winters, DTW
├── tensor         — N-dim, HOSVD, CP, Kronecker
└── autodiff       — forward + reverse mode AD
```

## Design Goals

- **Zero dependencies** in default mode
- **Generic over scalar types** (`f32`, `f64`)
- **Compile-time dimension checking** via const generics (`SMatrix<T, R, C>`)
- **Lazy evaluation** to avoid unnecessary heap allocation
- **Benchmarked** with Criterion for regression-free releases

## License

MIT
