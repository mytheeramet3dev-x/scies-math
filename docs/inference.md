# Statistical Inference and Hypothesis Testing (`inference`, `inference_ext`)

The `inference` and `inference_ext` modules provide rigorous parametric and non-parametric hypothesis testing, bootstrap resampling, permutation tests, and multiple comparison correction.

---

## 1. Hypothesis Testing Guide

| Test | Module | Null Hypothesis ($H_0$) | Assumptions | Test Statistic |
| :--- | :--- | :--- | :--- | :--- |
| **One-Sample $t$-test** | `inference` | $\mu = \mu_0$ | Normal population | Student's $t$ ($n-1$ df) |
| **Welch's Two-Sample $t$-test** | `inference` | $\mu_1 = \mu_2$ | Unequal variances allowed | Satterthwaite $t$ |
| **Paired $t$-test** | `inference` | $\mu_{\text{diff}} = 0$ | Dependent paired samples | Student's $t$ |
| **One-Way ANOVA** | `inference` | $\mu_1 = \dots = \mu_k$ | Homoscedastic normality | Fisher's $F$ ratio |
| **$\chi^2$ Independence** | `inference` | Row and Column variables are independent | Contingency table counts | Pearson $\chi^2$ |
| **Mann-Whitney $U$** | `inference_ext`| Distributions of $X$ and $Y$ are identical | Non-parametric ordinal/continuous | Rank sum $U$ |
| **Wilcoxon Signed-Rank** | `inference_ext`| Median difference is zero | Non-parametric paired continuous | Signed rank $W$ |
| **Kruskal-Wallis** | `inference_ext`| Medians of $k$ groups are equal | Non-parametric ANOVA equivalent | $H$ statistic ($\chi^2_{k-1}$) |
| **Kolmogorov-Smirnov 2-Sample**| `inference_ext`| $F_1(x) = F_2(x)$ | Continuous cumulative distributions | Supremum distance $D = \sup_x \|F_1 - F_2\|$ |
| **Levene's Test** | `inference_ext`| Group variances $\sigma_1^2 = \dots = \sigma_k^2$ | Robust to non-normality | $F$-statistic on absolute deviations |

---

## 2. Resampling & Multiple Testing Correction

### Bootstrap Confidence Intervals (`bootstrap_ci`)
Computes empirical percentile confidence intervals for arbitrary statistics without parametric distribution assumptions:

```rust
use scies_math_th::inference_ext::bootstrap_ci;

let data = vec![1.2, 2.3, 1.8, 4.5, 3.1, 2.9, 1.7];
let (ci_lo, ci_hi) = bootstrap_ci(&data, 2000, 0.05, |sample| {
    sample.iter().sum::<f64>() / sample.len() as f64
})?;
println!("95% Bootstrap Mean CI: [{:.3}, {:.3}]", ci_lo, ci_hi);
```

### Multiple Testing Adjustment (`p_adjust`)
Controls the Family-Wise Error Rate (FWER) or False Discovery Rate (FDR):

- **Bonferroni**: $p_i^{\text{adj}} = \min(1, m \cdot p_i)$ (Strong FWER control).
- **Holm-Bonferroni**: Step-down sequentially rejective procedure.
- **Benjamini-Hochberg (BH)**: Controls the False Discovery Rate:
  $$p_{(i)}^{\text{adj}} = \min_{k \ge i} \min\left(1, \frac{m}{k} p_{(k)}\right)$$

---

## 3. Code Example

```rust
use scies_math_th::inference::two_sample_t_test;
use scies_math_th::inference_ext::{AdjustMethod, mann_whitney_u, p_adjust};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let group_a = vec![14.2, 15.1, 14.8, 15.4, 14.9];
    let group_b = vec![16.5, 17.2, 16.8, 16.1, 17.0];

    // Welch's t-test
    let t_res = two_sample_t_test(&group_a, &group_b, 0.05)?;
    println!("t-stat: {:.4}, p-value: {:.2e}, reject H0: {}",
        t_res.statistic, t_res.p_value, t_res.reject_h0);

    // Non-parametric Mann-Whitney U test
    let mw_res = mann_whitney_u(&group_a, &group_b, 0.05)?;
    println!("Mann-Whitney U: {:.1}, p-value: {:.2e}", mw_res.statistic, mw_res.p_value);

    // Multiple testing correction across 4 hypotheses
    let raw_p_values = vec![0.001, 0.012, 0.045, 0.120];
    let bh_adjusted = p_adjust(&raw_p_values, AdjustMethod::BenjaminiHochberg)?;
    println!("FDR Benjamini-Hochberg adjusted p-values: {:?}", bh_adjusted);

    Ok(())
}
```
