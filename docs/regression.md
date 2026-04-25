# `regression` Module Documentation

Linear regression models.

# Models

| Type | Function / Struct | Regularization |
|---|---|---|
| OLS | `linear_regression` | None |
| Ridge | `ridge_regression` | L2 (λ‖β‖₂²) |
| Weighted OLS | `weighted_linear_regression` | None |
| Polynomial | `polynomial_regression` | None |

# Ordinary Least Squares

Solves min ‖Xβ − y‖₂² via QR decomposition (numerically stable).

```rust
use scies_math::regression::linear_regression;

// X: design matrix (n_samples × n_features), y: targets
let x = vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![1.0, 4.0]];
let y = vec![5.0, 7.0, 9.0];
let beta = linear_regression(&x, &y).unwrap();
// beta ≈ [1.0, 2.0]  (intercept=1, slope=2)
```

# Ridge Regression

Adds an L2 penalty: min ‖Xβ − y‖₂² + λ‖β‖₂².
Useful when X is nearly singular or p >> n.

```rust
use scies_math::regression::ridge_regression;

let beta = ridge_regression(&x, &y, 0.1).unwrap(); // lambda = 0.1
```

# Return value

All regression functions return `Vec<f64>` of coefficients β.
The first element is the intercept when an intercept column is included in X.

See [`crate::regression_ext`] for Lasso, Elastic Net, k-NN, and Decision Tree.