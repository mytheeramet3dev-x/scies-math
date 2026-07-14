# `inference` Module Documentation

Statistical inference — hypothesis tests and confidence intervals.

## Overview

This module covers common hypothesis tests and inference routines that sit on top of descriptive statistics.

## Tests

| Function | Test | H₀ |
|---|---|---|
| `one_sample_t_test` | One-sample t | μ = μ₀ |
| `two_sample_t_test` | Welch t-test | μ₁ = μ₂ |
| `paired_t_test` | Paired t | μ_diff = 0 |
| `one_way_anova` | One-way ANOVA | All group means equal |
| `chi_square_gof` | Goodness-of-fit χ² | Observed = Expected |
| `chi_square_independence` | χ² independence | Rows ⊥ Columns |
| `ks_test` | Kolmogorov-Smirnov | CDF = reference CDF |
| `mann_whitney_u` | Mann-Whitney U | Identical distributions |
| `wilcoxon_signed_rank` | Wilcoxon | Symmetric about 0 |

# Return type — `TestResult`

```text
pub struct TestResult {
    pub statistic: f64,
    pub p_value:   f64,
    pub reject_h0: bool,   // true when p_value < alpha
}
```

## Usage

```rust
use scies_math_th::inference::two_sample_t_test;

let a = vec![5.1, 4.9, 5.0, 5.2, 4.8];
let b = vec![5.5, 5.3, 5.4, 5.6, 5.2];
let result = two_sample_t_test(&a, &b, 0.05).unwrap();
println!("p = {:.4}, reject H0: {}", result.p_value, result.reject_h0);
```

For bootstrap CI, permutation tests, and FDR correction see [`crate::inference_ext`].

## Notes

- Use this module when you need a formal test decision and a p-value together.
- Keep sample sizes and assumptions in mind; tests do not replace domain judgment.
