# `interpolation` Module Documentation

Interpolation and function approximation.

# Methods

| Function | Method | Notes |
|---|---|---|
| `linear_interp` | Piecewise linear | O(log n) with sorted xs |
| `lagrange_interp` | Lagrange polynomial | O(n²); avoid large n |
| `neville_interp` | Neville's algorithm | Numerically stable Lagrange |
| `cubic_spline` | Natural cubic spline | C² everywhere |
| `akima_spline` | Akima spline | Resists oscillation near outliers |
| `bilinear_interp` | 2D bilinear | Grid data |
| `rbf_interp` | Radial basis functions | Scattered data |

# Usage

```rust
use scies_math_th::interpolation::cubic_spline;

let xs = vec![0.0, 1.0, 2.0, 3.0];
let ys = vec![0.0, 1.0, 0.0, 1.0];
let spline = cubic_spline(&xs, &ys).unwrap();

let y = spline.eval(1.5);  // smooth interpolated value
```
