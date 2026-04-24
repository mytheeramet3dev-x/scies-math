# SciesMath

`scies-math` is a high-performance, dependency-free mathematical toolkit for Rust. It provides extensive, production-ready modules for advanced mathematics, statistics, and numerical methods.

## Features
- **Linear Algebra**: Matrix operations, Cholesky, LU, QR, SVD, Eigenvalue decomposition, Sparse matrices (CSR/CSC), and iterative solvers (CG, GMRES, BiCGSTAB).
- **Tensor Algebra**: N-dimensional arrays, tensor contraction, Kronecker products, HOSVD (Tucker), CP Decomposition.
- **Calculus & Autodiff**: Forward and reverse-mode automatic differentiation, Jacobian/Hessian computation, vector calculus.
- **Optimization**: Gradient Descent, BFGS, Trust-Region, Augmented Lagrangian, PSO, Differential Evolution.
- **Differential Equations**: ODE solvers (RK4, RK45, DOP853, BDF2, Rosenbrock2, Symplectic integrators) and PDE solvers (Heat, Wave, Laplace).
- **Signal Processing**: FFT, IFFT, STFT, FIR/IIR filtering, Wavelet Transform.
- **Probability & Statistics**: Extensive distributions, Hypothesis testing, MCMC, Time Series (ARIMA, Holt-Winters).
- **Regression & Classification**: Linear, Ridge, Lasso, Logistic, Softmax, Naive Bayes, k-NN, Decision Trees.

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
scies-math = { path = "path/to/scies-math" } # Or from git/crates.io when published
```
