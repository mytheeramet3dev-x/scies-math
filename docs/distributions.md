# `distributions` Module Documentation

Probability distributions — PDF, CDF, and inverse CDF.

# Distributions

| Distribution | PDF | CDF | Inverse CDF |
|---|---|---|---|
| `Normal` | Yes | Yes | Yes |
| `LogNormal` | Yes | Yes | Yes |
| `Exponential` | Yes | Yes | Yes |
| `Gamma` | Yes | Yes | — |
| `Beta` | Yes | Yes | Yes |
| `ChiSquared` | Yes | Yes | — |
| `StudentT` | Yes | Yes | — |
| `Weibull` | Yes | Yes | Yes |
| `Uniform` | Yes | Yes | Yes |
| `Triangular` | Yes | Yes | Yes |
| `Pareto` | Yes | Yes | Yes |

# Usage

```rust
use scies_math_th::distributions::{Normal, Distribution};

let n = Normal::new(0.0, 1.0).unwrap();
let p = n.pdf(1.96);   // ≈ 0.0584
let c = n.cdf(1.96);   // ≈ 0.975
let q = n.inverse_cdf(0.975).unwrap(); // ≈ 1.96
```

# Design

All distribution structs implement a common `Distribution` trait:

```text
trait Distribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
}
```

For sampling, use [`crate::rng_ext::Sampler`] which provides high-quality
samplers for every distribution above using rejection/transformation methods.
