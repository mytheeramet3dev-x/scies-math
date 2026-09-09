# Monte Carlo Integration and Sampling (`monte_carlo`, `monte_carlo_ext`)

The `monte_carlo` and `monte_carlo_ext` modules provide stochastic multidimensional integration, variance reduction techniques, quasi-random low-discrepancy sequences, Markov Chain Monte Carlo (MCMC) samplers, and MCMC convergence diagnostics.

---

## 1. Multidimensional Integration

Evaluates high-dimensional definite integrals $I = \int_{\Omega} f(\mathbf{x}) d\mathbf{x}$ using statistical sampling:

$$\hat{I}_N = \frac{\text{Vol}(\Omega)}{N} \sum_{i=1}^N f(\mathbf{x}_i), \quad \mathbf{x}_i \sim \text{Uniform}(\Omega)$$

By the Central Limit Theorem, the standard error scales as $O(N^{-1/2})$, **immune to the curse of dimensionality** that degrades grid-based Newton-Cotes cubature rules.

### Variance Reduction Techniques
- **Stratified Sampling**: Partitions $\Omega$ into $K$ mutually disjoint sub-boxes, reducing sample clustering.
- **Importance Sampling**: Samples from proposal density $q(\mathbf{x})$ matching $|f(\mathbf{x})|$:
  $$\hat{I}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^N \frac{f(\mathbf{x}_i)}{q(\mathbf{x}_i)}, \quad \mathbf{x}_i \sim q(\mathbf{x})$$

---

## 2. Quasi-Monte Carlo (Low-Discrepancy Sequences)

Replaces pseudo-random numbers with deterministic space-filling sequences, accelerating convergence to $O(N^{-1} (\ln N)^d)$ via the Koksma-Hlawka inequality:

- `halton_sequence(n_points, dim)`: Generalized Van der Corput sequences across coprime integer bases.
- `sobol_sequence(n_points, dim)`: Base-2 digital net generated via primitive polynomials and Gray code ordering.
- `latin_hypercube(n_points, dim, seed)`: Guarantees that each axis has exactly one sample per 1D stratum.

---

## 3. Markov Chain Monte Carlo (MCMC)

Samples from unnormalized probability densities $p(\theta) \propto \pi(\theta)$:

| Sampler | Description | Target Modality |
| :--- | :--- | :--- |
| `metropolis_hastings` | Symmetric Gaussian random-walk proposal | General continuous posterior |
| `hamiltonian_mc` | Auxiliary momentum variables + leapfrog symplectic physics simulation | High-dimensional correlated posteriors |
| `nuts_sampler` | No-U-Turn Sampler (auto-tunes trajectory length $L$) | State-of-the-art Bayesian inference |
| `gibbs_sampler` | Iteratively samples full conditional distributions $p(\theta_i \mid \theta_{-i})$ | Conditionally conjugate Bayesian models |
| `slice_sampler` | Uniform sampling under the density curve | Univariate / multi-modal distributions |

### MCMC Diagnostics
- `gelman_rubin(&[chains])`: Computes the potential scale reduction factor $\hat{R} = \sqrt{\frac{\text{Var}^+(\theta)}{W}}$. Convergence is achieved when $\hat{R} < 1.05$.
- `effective_sample_size(&chain)`: Estimates the number of independent draws accounting for autocorrelation:
  $$\text{ESS} = \frac{N}{1 + 2 \sum_{k=1}^\infty \rho_k}$$

---

## 4. Code Example

```rust
use scies_math_th::monte_carlo::monte_carlo_integrate;
use scies_math_th::monte_carlo_ext::{halton_sequence, hamiltonian_mc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Monte Carlo: Estimate volume of 3D unit sphere: V = (4/3)*pi*r^3 ≈ 4.18879
    let sphere_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)];
    let volume_approx = monte_carlo_integrate(
        |x| if x[0]*x[0] + x[1]*x[1] + x[2]*x[2] <= 1.0 { 1.0 } else { 0.0 },
        &sphere_bounds,
        100_000,
        12345,
    );
    println!("Monte Carlo 3D sphere volume: {:.4}", volume_approx);

    // 2. Quasi-Monte Carlo: Halton 2D points in [0, 1]^2
    let qmc_points = halton_sequence(5, 2)?;
    println!("First Halton point: {:?}", qmc_points[0]);

    // 3. Hamiltonian Monte Carlo on Standard Normal distribution
    let hmc_samples = hamiltonian_mc(
        |x| -0.5 * x[0] * x[0], // log-posterior ln p(x)
        |x| vec![-x[0]],        // grad ln p(x)
        vec![0.0],              // initial position
        0.1,                    // step size epsilon
        10,                     // leapfrog steps L
        500,                    // sample count
        42,                     // seed
    )?;
    println!("HMC collected {} posterior samples.", hmc_samples.len());

    Ok(())
}
```
