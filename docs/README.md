# scies-math Documentation

Reference documentation for every module in the `scies-math` crate.

## Modules

| File | Module | Topic |
|---|---|---|
| [linear_algebra.md](linear_algebra.md) | `linear_algebra` | Vectors, matrices, LU/QR/SVD/Cholesky |
| [generic.md](generic.md) | `generic` | Generic scalar types, Mat\<T\>, SMatrix\<T,R,C\> |
| [perf.md](perf.md) | `perf` | High-performance matmul backends, Xoshiro256** |
| [transform.md](transform.md) | `transform` | Quaternion, Rotation3, Isometry3, projections |
| [rng.md](rng.md) | `rng_ext` | RNG engines and Sampler API |
| [ode.md](ode.md) | `ode`, `ode_ext` | ODE solvers (fixed-step, adaptive, stiff, symplectic) |
| [pde.md](pde.md) | `pde`, `pde_ext` | PDE solvers (heat, wave, Laplace, Navier-Stokes) |
| [statistics.md](statistics.md) | `statistics` | Descriptive statistics |
| [distributions.md](distributions.md) | `distributions` | Probability distributions (PDF/CDF/iCDF) |
| [inference.md](inference.md) | `inference`, `inference_ext` | Hypothesis tests, bootstrap, FDR |
| [regression.md](regression.md) | `regression`, `regression_ext` | Linear models, Lasso, k-NN, decision tree |
| [monte_carlo.md](monte_carlo.md) | `monte_carlo`, `monte_carlo_ext` | MC integration, MCMC, particle filters |
| [signal.md](signal.md) | `signal`, `signal_ext` | FFT, filters, wavelets |
| [optimization.md](optimization.md) | `optimization`, `opt_multivar`, `opt_multivar_ext` | Scalar and multivariate optimization |
| [autodiff.md](autodiff.md) | `autodiff`, `reverse_ad` | Forward and reverse automatic differentiation |
| [tensor.md](tensor.md) | `tensor` | N-dimensional tensors, HOSVD, CP decomposition |
| [timeseries.md](timeseries.md) | `timeseries` | ARIMA, exponential smoothing, DTW |
| [complex.md](complex.md) | `complex` | Complex number arithmetic |
| [interpolation.md](interpolation.md) | `interpolation` | Splines, RBF, bilinear interpolation |
| [lazy.md](lazy.md) | `lazy` | Lazy matrix expression tree |

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
scies-math = "0.2"

# With serde support:
scies-math = { version = "0.2", features = ["serde"] }
```
| [special_functions.md](special_functions.md) | `special_functions` | Erf, Gamma, Beta, Bessel |
