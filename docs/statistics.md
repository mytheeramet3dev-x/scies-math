# `statistics` Module Documentation

Descriptive statistics — moments, order statistics, correlations.

# Functions

| Function | Description |
|---|---|
| `mean(data)` | Arithmetic mean |
| `variance(data)` | Sample variance (Bessel corrected) |
| `std_dev(data)` | Sample standard deviation |
| `median(data)` | Middle value (sorted) |
| `mode(data)` | Most frequent value |
| `quantile(data, p)` | p-th quantile (0 ≤ p ≤ 1) |
| `iqr(data)` | Inter-quartile range Q3 − Q1 |
| `skewness(data)` | Third standardised moment |
| `kurtosis(data)` | Fourth standardised moment − 3 |
| `covariance(x, y)` | Sample covariance |
| `pearson_r(x, y)` | Pearson correlation coefficient |
| `spearman_r(x, y)` | Spearman rank correlation |
| `histogram(data, bins)` | Bin counts and edges |
| `z_score(data)` | Standardised data (mean=0, std=1) |

# Usage

```rust
use scies_math_th::statistics::{mean, std_dev, pearson_r};

let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
assert!((mean(&data) - 5.0).abs() < 1e-10);
assert!((std_dev(&data) - 2.0).abs() < 1e-10);

let x = vec![1.0, 2.0, 3.0];
let y = vec![2.0, 4.0, 6.0];
assert!((pearson_r(&x, &y).unwrap() - 1.0).abs() < 1e-10);
```
