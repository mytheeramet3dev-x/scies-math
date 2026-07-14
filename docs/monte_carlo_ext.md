# `monte_carlo_ext` Module Documentation

Advanced Monte Carlo — MCMC samplers, quasi-random, and particle filters.

# MCMC Samplers

| Sampler | Function | Best for |
|---|---|---|
| Metropolis-Hastings | `metropolis_hastings` | General log-density |
| Gibbs | `gibbs_sampler` | Conditionally conjugate models |
| HMC | `hamiltonian_mc` | High-dimensional, smooth posteriors |
| Slice | `slice_sampler` | Univariate, bounded support |
| NUTS (No-U-Turn) | `nuts_sampler` | HMC with automatic step tuning |

# Quasi-random sequences

| Function | Description |
|---|---|
| `sobol_sequence` | Sobol low-discrepancy sequence |
| `halton_sequence` | Halton sequence (coprime bases) |
| `latin_hypercube` | Latin hypercube sampling |

# Diagnostics

| Function | Description |
|---|---|
| `effective_sample_size(chain)` | ESS estimate |
| `gelman_rubin(chains)` | R̂ convergence diagnostic |
| `autocorrelation(chain, lag)` | Sample autocorrelation |

# Usage — Hamiltonian Monte Carlo

```rust
use scies_math_th::monte_carlo_ext::hamiltonian_mc;

// Sample from N(0,1): log p(x) = -x²/2
let samples = hamiltonian_mc(
    |x: &[f64]| -0.5 * x[0] * x[0],          // log_density
    |x: &[f64]| vec![-x[0]],                   // grad_log_density
    vec![0.0],                                  // initial state
    0.1,                                        // step size ε
    10,                                         // leapfrog steps L
    1000,                                       // num samples
    42,                                         // seed
);
```
