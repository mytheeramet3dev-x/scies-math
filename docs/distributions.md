# Extended Probability Distributions (`distributions`, `distributions_ext`)

The `distributions` and `distributions_ext` modules provide exact probability density functions (PDF), cumulative distribution functions (CDF), quantile inverse CDFs (iCDF), and statistical moments across continuous and discrete distribution families.

---

## 1. Supported Distribution Families

### Continuous Distributions
| Distribution | Support | Key Parameters | PDF / PMF | CDF | Inverse CDF (Quantile) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Normal** ($\mathcal{N}$) | $\mathbb{R}$ | $\mu \in \mathbb{R}, \sigma > 0$ | Yes | Yes | Yes (Rational Chebyshev) |
| **Log-Normal** | $(0, \infty)$ | $\mu \in \mathbb{R}, \sigma > 0$ | Yes | Yes | Yes |
| **Student's $t$** | $\mathbb{R}$ | $\nu > 0$ (degrees of freedom) | Yes | Yes | Yes |
| **Chi-Squared** ($\chi^2$) | $(0, \infty)$ | $k \in \mathbb{N}^+$ | Yes | Yes (Incomplete Gamma) | Yes |
| **Fisher-Snedecor $F$** | $(0, \infty)$ | $d_1, d_2 > 0$ | Yes | Yes (Incomplete Beta) | Yes |
| **Gamma** | $(0, \infty)$ | $\alpha > 0$ (shape), $\beta > 0$ (rate) | Yes | Yes | Yes (Newton-Raphson) |
| **Beta** | $[0, 1]$ | $\alpha, \beta > 0$ | Yes | Yes | Yes (Newton-Raphson) |
| **Weibull** | $[0, \infty)$ | $\lambda > 0$ (scale), $k > 0$ (shape) | Yes | Yes | Yes |
| **Laplace** | $\mathbb{R}$ | $\mu \in \mathbb{R}, b > 0$ | Yes | Yes | Yes (Closed-form) |
| **Logistic** | $\mathbb{R}$ | $\mu \in \mathbb{R}, s > 0$ | Yes | Yes | Yes (Closed-form) |
| **Gumbel** | $\mathbb{R}$ | $\mu \in \mathbb{R}, \beta > 0$ | Yes | Yes | Yes (Extreme value) |
| **Pareto (Type I)** | $[x_m, \infty)$ | $x_m > 0, \alpha > 0$ | Yes | Yes | Yes (Power law) |
| **Rayleigh** | $[0, \infty)$ | $\sigma > 0$ | Yes | Yes | Yes |
| **Truncated Normal** | $[a, b]$ | $\mu, \sigma, a, b$ | Yes | Yes | Yes |
| **Von Mises** | $[-\pi, \pi]$ | $\mu, \kappa \ge 0$ (circular concentration)| Yes | — | — |
| **Maxwell-Boltzmann**| $[0, \infty)$ | $a > 0$ (molecular kinetic speed)| Yes | Yes | — |

### Discrete Distributions
| Distribution | Support | PMF Formula | Expected Mean |
| :--- | :--- | :--- | :--- |
| **Binomial** | $\{0, \dots, n\}$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | $n p$ |
| **Poisson** | $\mathbb{N}_0$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ |
| **Geometric** | $\{1, 2, \dots\}$ | $(1-p)^{k-1} p$ | $1/p$ |
| **Negative Binomial** | $\mathbb{N}_0$ | $\binom{k+r-1}{k} (1-p)^k p^r$ | $\frac{r(1-p)}{p}$ |
| **Multinomial** | $\sum x_i = n$ | $\frac{n!}{\prod x_i!} \prod p_i^{x_i}$ | $n p_i$ |
| **Beta-Binomial** | $\{0, \dots, n\}$ | $\binom{n}{k} \frac{B(k+\alpha, n-k+\beta)}{B(\alpha, \beta)}$ | $n \frac{\alpha}{\alpha + \beta}$ |

---

## 2. API Usage Example

```rust
use scies_math_th::distributions::{Distribution, Normal};
use scies_math_th::distributions_ext::{laplace_cdf, laplace_pdf, pareto_pdf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Standard Normal: N(0, 1)
    let norm = Normal::new(0.0, 1.0)?;
    println!("Normal PDF at x=0: {:.6}", norm.pdf(0.0)); // 0.398942
    println!("Two-tailed 95% critical value: {:.4}", norm.inverse_cdf(0.975)?); // 1.95996

    // 2. Heavy-tailed Pareto Distribution: x_min = 1.0, alpha = 2.0
    let p_pdf = pareto_pdf(2.0, 1.0, 2.0)?;
    println!("Pareto(1, 2) PDF at x=2.0: {:.4}", p_pdf); // 2 * 1^2 / 2^3 = 0.25

    // 3. Laplace Distribution (Double Exponential)
    let l_pdf = laplace_pdf(1.0, 0.0, 1.0)?;
    let l_cdf = laplace_cdf(1.0, 0.0, 1.0)?;
    println!("Laplace(0, 1) PDF at x=1: {:.4}", l_pdf);
    println!("Laplace(0, 1) CDF at x=1: {:.4}", l_cdf);

    Ok(())
}
```
