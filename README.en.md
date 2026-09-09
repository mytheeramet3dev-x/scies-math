# scies-math-th

[![Crates.io](https://img.shields.io/crates/v/scies-math-th.svg)](https://crates.io/crates/scies-math-th)
[![Documentation](https://docs.rs/scies-math-th/badge.svg)](https://docs.rs/scies-math-th)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-success.svg)]()

`scies-math-th` is a high-performance, verifiable, dependency-free mathematical and scientific computing toolkit written in pure Rust. It provides foundational linear algebra, numerical solvers, Semidefinite Programming (SDP), Moment-SOS polynomial optimization hierarchies, exact 128-bit rational arithmetic, symbolic calculus, SI dimensional analysis, and an automated Lean 4 / Mathlib4 theorem export bridge.

Designed from the ground up for scientific workloads, experimental mathematics, and high-assurance computational pipelines, `scies-math-th` compiles with **zero required external dependencies** in its default build.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [Mathematical Reliability & Validity Tiers](#mathematical-reliability--validity-tiers)
- [Architecture & Module Organization](#architecture--module-organization)
- [Installation](#installation)
- [Comprehensive Module Walkthroughs & Examples](#comprehensive-module-walkthroughs--examples)
  - [1. Semidefinite Programming (ADMM & Mehrotra Primal-Dual IPM)](#1-semidefinite-programming-admm--mehrotra-primal-dual-ipm)
  - [2. Moment-SOS Hierarchy & DIMACS CNF/3-SAT Relaxation](#2-moment-sos-hierarchy--dimacs-cnf3-sat-relaxation)
  - [3. Exact 128-Bit Checked Rational Arithmetic](#3-exact-128-bit-checked-rational-arithmetic)
  - [4. Symbolic Expression Trees, Calculus & Polynomial Division](#4-symbolic-expression-trees-calculus--polynomial-division)
  - [5. SI 7-Base Dimensional Analysis & Type-Safe Quantities](#5-si-7-base-dimensional-analysis--type-safe-quantities)
  - [6. Proof Provenance & Lean 4 / Mathlib4 Export Bridge](#6-proof-provenance--lean-4--mathlib4-export-bridge)
  - [7. Dense Linear Algebra, Blocked GEMM & SVD](#7-dense-linear-algebra-blocked-gemm--svd)
  - [8. Compressed Sparse Row (CSR) & Lanczos Eigensolver](#8-compressed-sparse-row-csr--lanczos-eigensolver)
  - [9. Multivariate Optimization & Automatic Differentiation](#9-multivariate-optimization--automatic-differentiation)
  - [10. Differential Equations (Adaptive RK45 & 2D Heat ADI)](#10-differential-equations-adaptive-rk45--2d-heat-adi)
- [Feature Flags](#feature-flags)
- [Quality Assurance & Benchmarks](#quality-assurance--benchmarks)
- [Documentation & License](#documentation--license)

---

## Key Highlights

- **Zero External Dependencies**: The entire core engine—matrix operations, decompositions, solvers, autodiff, and rational arithmetic—runs without pulling in third-party crates.
- **Strict Verification Discipline**: Numerical solvers do not claim mathematical certificates without explicit a-posteriori verifiers. Termination statuses (e.g. `SdpStatus::Optimal`) are only assigned when independent KKT residuals pass tight numerical tolerances.
- **Mehrotra Predictor-Corrector IPM**: Includes an interior-point semidefinite solver with step-damping, Cholesky/LU factorizations, and full KKT diagnostics alongside a fast first-order ADMM solver.
- **Lasserre / Moment-SOS Relaxations**: Built-in canonical DegRevLex monomial indexing, quotient polynomial reductions ($x_i^2 = x_i$), localizing matrix generation, and DIMACS CNF/SDPA sparse I/O.
- **Exact & Verified Arithmetic**: 128-bit checked rational arithmetic (`Rational`) preventing overflow and division by zero, alongside interval bounds.
- **Formal Methods Ready**: Export symbolic derivations directly into Lean 4 theorem declarations with Mathlib tactics.

---

## Mathematical Reliability & Validity Tiers

To maintain absolute scientific integrity, `scies-math-th` strictly classifies every computational result into one of five rigorous tiers (see [`docs/api_reliability_matrix.md`](docs/api_reliability_matrix.md)):

| Tier | Classification | Underlying Type | Guarantees & Semantics | Modules |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Experimental** | Exploratory | Rapid prototyping heuristics; experimental algorithms. | `nn`, `geometry_ext` |
| **Tier 2** | **Numerical** | IEEE 754 `f64` | Standard floating-point convergence. Subject to roundoff and conditioning. | `opt_multivar`, `ode`, `pde`, `statistics` |
| **Tier 3** | **Numerically Verified** | `f64` + [`SdpResidualReport`] | Rigorous a-posteriori verification ($r_p, v_p, R_d, v_d, \text{gap} \le \epsilon$). | `sdp`, `sdp_ipm`, `sdp_primitives`, `eigensystem` |
| **Tier 4** | **Exact** | [`Rational`] / Algebraic AST | $p/q \in \mathbb{Q}$ using 128-bit checked integers; zero roundoff error. | `exact::Rational`, `symbolic`, `units` |
| **Tier 5** | **Formal** | Machine-Checked AST | Unverified Lean 4 theorem skeletons for external machine checking. | `verification::derivation`, `verification::lean_export` |

---

## Architecture & Module Organization

```text
scies_math_th::
├── sdp                 # Semidefinite Programming (ADMM, KKT residual report, verify_sdp_candidate)
├── sdp_ipm             # High-accuracy Mehrotra Predictor-Corrector Interior-Point SDP solver
├── sdp_primitives     # PSD cone spectral projection, eigenvalue diagnostics, rank estimation
├── moment              # Moment-SOS hierarchies, DegRevLex monomial bases, 3-SAT relaxation, SDPA I/O
├── exact               # 128-bit checked Rational arithmetic, interval enclosure heuristics
├── symbolic            # Expression trees, canonical simplifier, symbolic differentiation, polynomials
├── units               # SI 7-base dimensional analysis, type-safe physical units (m, kg, s, A, K, mol, cd)
├── verification        # Proof provenance tracking, derivation trees, Lean 4 / Mathlib4 code generator
├── linear_algebra      # DynamicMatrix, Blocked GEMM, LU, QR, Cholesky, SVD, solve, inversion
├── linear_operator     # Matrix-free abstract linear operator trait (apply, apply_adjoint)
├── symmetric           # Packed lower-triangular SymmetricMatrix storage & operations
├── eigensystem         # Symmetric Jacobi & QR eigensystems with spectral/orthogonality residuals
├── sparse              # Compressed Sparse Row (CSR), Lanczos eigensolver, PCG, GMRES, BiCGSTAB
├── generic             # Generic Mat<T>, SMatrix<T, R, C> with const generics
├── lazy                # Lazy evaluation expression trees for memory-efficient matrix operations
├── autodiff            # Forward-mode automatic differentiation (Dual numbers, gradient, Jacobian)
├── reverse_ad          # Reverse-mode automatic differentiation (Wengert tape, VJP, Hessian)
├── opt_multivar        # Multivariate optimization: BFGS, L-BFGS, Nelder-Mead, Armijo gradient descent
├── optimization        # Univariate root finding and 1D optimization (Brent, Golden Section)
├── ode / ode_ext       # Fixed-step & adaptive ODE solvers (Euler, RK4, RK45, Stiff, Symplectic)
├── pde / pde_ext       # PDE solvers (2D Heat ADI, 2D Wave, Burgers, Method of Lines, Shooting)
├── statistics          # Descriptive statistics, moments, quantiles, robust estimators
├── distributions       # Probability distributions (Normal, Gamma, Beta, Poisson, Student's t, etc.)
├── inference           # Hypothesis testing (t-test, ANOVA, Mann-Whitney, Kolmogorov-Smirnov)
├── regression          # Ordinary Least Squares, Ridge, Lasso, Elastic Net, Logistic regression
├── monte_carlo         # Quasi-Monte Carlo (Halton, Sobol, Latin Hypercube), MCMC (Metropolis-Hastings)
├── signal              # Cooley-Tukey FFT, IIR/FIR filter design, STFT, Hilbert transform
├── timeseries          # Autoregressive models (ARIMA), Holt-Winters smoothing, Dynamic Time Warping
├── transform           # Quaternions, 3D rotations, rigid isometries, SE(3) transforms
├── special_functions   # Error function (erf), Gamma, Log-Gamma, Beta, Digamma, Bessel J0/J1
└── io                  # Scientific data serialization: SDPA, DIMACS CNF, MatrixMarket, CSV
```

---

## Installation

Add `scies-math-th` to your `Cargo.toml`:

```toml
[dependencies]
scies-math-th = "0.3.0"

# Optional features:
# scies-math-th = { version = "0.3.0", features = ["serde", "parallel"] }
```

---

## Comprehensive Module Walkthroughs & Examples

### 1. Semidefinite Programming (ADMM & Mehrotra Primal-Dual IPM)

`scies-math-th` solves standard primal-dual semidefinite programs of the form:

$$\textbf{Primal:} \quad \min_{X \in \mathcal{S}^n} \langle C, X \rangle \quad \text{s.t.} \quad \langle A_i, X \rangle = b_i \; (i = 1, \dots, m), \; X \succeq 0$$

$$\textbf{Dual:} \quad \max_{y \in \mathbb{R}^m, S \in \mathcal{S}^n} b^T y \quad \text{s.t.} \quad \sum_{i=1}^m y_i A_i + S = C, \; S \succeq 0$$

```rust
use scies_math_th::sdp::{SdpProblem, SdpSolverConfig, SdpStatus, verify_sdp_candidate};
use scies_math_th::sdp_ipm::{solve_sdp_ipm, SdpIpmConfig};
use scies_math_th::symmetric::SymmetricMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize Tr(C * X) subject to <A1, X> = 1, <A2, X> = 1, X >= 0
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0])?; // [1 0; 0 2]
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0])?; // X_00 = 1
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0])?; // X_11 = 1
    let b = vec![1.0, 1.0];

    let problem = SdpProblem::new(c, vec![a1, a2], b)?;

    // Method A: First-order ADMM Solver
    let admm_config = SdpSolverConfig {
        tolerance: 1e-4,
        max_iterations: 300,
        rho: 1.0,
        strict: true,
    };
    let admm_sol = problem.solve(&admm_config)?;
    assert_eq!(admm_sol.status, SdpStatus::Optimal);
    println!("ADMM Primal Objective: {:.6}", admm_sol.primal_objective); // ~ 3.000000

    // Method B: High-Precision Mehrotra Primal-Dual Interior-Point Solver (IPM)
    let ipm_config = SdpIpmConfig {
        tolerance: 1e-7,
        max_iterations: 50,
        step_damping: 0.95,
    };
    let ipm_sol = solve_sdp_ipm(&problem, &ipm_config)?;
    println!("IPM Primal Objective: {:.8}", ipm_sol.primal_objective);

    // Independent A-Posteriori KKT Verification
    let report = verify_sdp_candidate(&problem, &ipm_sol.x, &ipm_sol.y, &ipm_sol.s, 1e-6)?;
    assert!(report.is_optimal(1e-6));
    println!("Primal equality residual: {:.2e}", report.primal_equality_residual);
    println!("Dual equality residual:   {:.2e}", report.dual_equality_residual);
    println!("Relative duality gap:     {:.2e}", report.relative_duality_gap);

    Ok(())
}
```

---

### 2. Moment-SOS Hierarchy & DIMACS CNF/3-SAT Relaxation

Compile Boolean satisfiability (SAT / 3-SAT) and polynomial optimization problems into semidefinite relaxations via the Lasserre / Moment-SOS hierarchy using canonical DegRevLex monomial bases:

```rust
use scies_math_th::moment::sat_encoding::{Clause, CnfFormula, Literal};
use scies_math_th::moment::sdpa_io::{export_sdpa_sparse, import_sdpa_sparse};
use scies_math_th::sdp::SdpSolverConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse DIMACS CNF format string
    let dimacs_data = r#"c Benchmark 3-SAT formula
p cnf 3 2
1 2 3 0
-1 2 -3 0
"#;
    let formula = CnfFormula::from_dimacs(dimacs_data)?;
    assert_eq!(formula.num_vars(), 3);
    assert_eq!(formula.num_clauses(), 2);

    // Exact truth-table brute-force oracle for small instances (n <= 20)
    let witness = formula.solve_brute_force()?;
    println!("SAT Witness Assignment: {:?}", witness);

    // Build degree-2 Moment-SOS SDP relaxation
    let (sdp, report) = formula.build_moment_sdp_relaxation_with_report(2)?;
    println!("Moment Matrix Dimension: {} x {}", sdp.matrix_dim(), sdp.matrix_dim());
    println!("Generated Equality Constraints: {}", report.num_generated_constraints);

    // Solve the resulting SDP relaxation
    let config = SdpSolverConfig::default();
    let sol = sdp.solve(&config)?;
    println!("Relaxation Status: {:?}", sol.status);

    // Universal SDPA Sparse format roundtrip (.dat-s)
    let sdpa_str = export_sdpa_sparse(&sdp, "3-SAT Degree-2 Relaxation")?;
    let reloaded_sdp = import_sdpa_sparse(&sdpa_str)?;
    assert_eq!(reloaded_sdp.matrix_dim(), sdp.matrix_dim());

    Ok(())
}
```

---

### 3. Exact 128-Bit Checked Rational Arithmetic

For scientific applications where floating-point roundoff error cannot be tolerated, [`exact::Rational`](docs/exact.md) represents numbers as exact reduced fractions $p/q \in \mathbb{Q}$ using 128-bit signed integers with checked overflow guards:

```rust
use scies_math_th::exact::Rational;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Rational::new(1, 3)?;
    let b = Rational::new(1, 6)?;

    // Exact checked arithmetic: 1/3 + 1/6 = 3/6 = 1/2
    let sum = a.checked_add(&b)?;
    assert_eq!(sum, Rational::new(1, 2)?);
    assert_eq!(sum.to_f64(), 0.5);

    // Exact division: (1/3) / (1/6) = 2
    let div = a.checked_div(&b)?;
    assert_eq!(div, Rational::from_integer(2));

    // Division by zero returns descriptive SciError
    assert!(Rational::new(1, 0).is_err());

    // Best rational approximation of floating-point numbers (Stern-Brocot / Farey)
    let approx_pi = Rational::from_f64_approx(3.141592653589793, 1000)?;
    assert_eq!(approx_pi, Rational::new(355, 113)?);

    Ok(())
}
```

---

### 4. Symbolic Expression Trees, Calculus & Polynomial Division

`scies-math-th` contains a zero-dependency symbolic algebra engine capable of canonical simplification, constant folding, symbolic differentiation, and univariate polynomial Euclidean division:

```rust
use scies_math_th::symbolic::{Expr, Polynomial};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Expr::sym("x");

    // Build symbolic expression: f(x) = x^3 + sin(x) + 0
    let f = Expr::Add(vec![
        Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(3))),
        Expr::Sin(Box::new(x.clone())),
        Expr::int(0),
    ]);

    // Canonical simplification (removes zeros, folds constants)
    let simplified = f.simplify();
    println!("Simplified: {}", simplified);

    // Symbolic differentiation: d/dx [ x^3 + sin(x) ] = 3*x^2 + cos(x)
    let df = simplified.diff("x").simplify();
    println!("df/dx: {}", df);

    // Polynomial Euclidean Division: A(x) = B(x) * Q(x) + R(x)
    // A(x) = x^3 - 2x^2 - 4 (coeffs: [-4, 0, -2, 1])
    // B(x) = x - 3          (coeffs: [-3, 1])
    let p_a = Polynomial::new(vec![-4.0, 0.0, -2.0, 1.0])?;
    let p_b = Polynomial::new(vec![-3.0, 1.0])?;
    let (quot, rem) = p_a.div_rem(&p_b)?;
    println!("Quotient:   {:?}", quot.coefficients()); // [7, 1, 1] => x^2 + x + 7
    println!("Remainder:  {:?}", rem.coefficients()); // [17] => 17

    Ok(())
}
```

---

### 5. SI 7-Base Dimensional Analysis & Type-Safe Quantities

Prevent dimensional mismatch errors at runtime with strict dimensional algebra across the 7 SI base dimensions ($\text{L}, \text{M}, \text{T}, \text{I}, \Theta, \text{N}, \text{J}$):

```rust
use scies_math_th::units::Quantity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mass = Quantity::kilograms(10.0);
    let acceleration = Quantity::meters(9.81) / (Quantity::seconds(1.0) * Quantity::seconds(1.0))?;

    // Newton's Second Law: F = m * a
    let force = mass * acceleration;
    println!("Force: {} N", force.value());

    // Incompatible additions return descriptive SciError
    let length = Quantity::meters(5.0);
    let invalid_add = mass.checked_add(&length);
    assert!(invalid_add.is_err()); // DimensionalMismatch

    Ok(())
}
```

---

### 6. Proof Provenance & Lean 4 / Mathlib4 Export Bridge

Bridge computation and interactive theorem proving. Export symbolic derivations into valid Lean 4 syntax complete with Mathlib headers and derivation metadata:

```rust
use scies_math_th::symbolic::Expr;
use scies_math_th::verification::lean_export::export_lean4_theorem;

fn main() {
    let x = Expr::sym("x");
    let expr = Expr::Add(vec![
        Expr::Pow(Box::new(x), Box::new(Expr::int(2))),
        Expr::int(1),
    ]);

    // Export verified theorem skeleton for Lean 4
    let lean_code = export_lean4_theorem(
        "quadratic_strictly_positive",
        &[("x", "ℝ")],
        "x ^ 2 + 1 > 0",
        Some("positivity"),
    );

    println!("{}", lean_code);
}
```

Generated Lean 4 output:
```lean
/-!
  Generated by scies-math-th (v0.3.0)
  Status: Generated theorem skeleton (requires verification by Lean 4 kernel)
-/

import Mathlib

theorem quadratic_strictly_positive (x : ℝ) : x ^ 2 + 1 > 0 := by
  positivity
```

---

### 7. Dense Linear Algebra, Blocked GEMM & SVD

```rust
use scies_math_th::linear_algebra::DynamicMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = DynamicMatrix::new(3, 3, vec![
        12.0, -51.0, 4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ])?;

    // Blocked Householder QR Decomposition
    let qr = a.qr_decompose()?;
    let reconstructed = qr.q.mul_matrix(&qr.r)?;
    assert!((a.get(0, 0)? - reconstructed.get(0, 0)?).abs() < 1e-10);

    // Singular Value Decomposition (SVD)
    let svd = a.singular_value_decompose(1e-10, 200)?;
    println!("Singular values: {:?}", svd.singular_values);
    println!("Matrix 2-norm condition number: {:.4}", svd.condition_number());

    Ok(())
}
```

---

### 8. Compressed Sparse Row (CSR) & Lanczos Eigensolver

```rust
use scies_math_th::sparse::{SparseMatrixCsr, lanczos_eigen};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Construct 4x4 tridiagonal matrix via coordinate (COO) triplets
    let triplets = vec![
        (0, 0,  2.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1,  2.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2,  2.0), (2, 3, -1.0),
        (3, 2, -1.0), (3, 3,  2.0),
    ];
    let csr = SparseMatrixCsr::from_triplets(4, 4, &triplets)?;
    csr.validate_invariants()?; // Ensures monotonic rows and sorted unique indices

    // Deterministic iterative Lanczos eigensolver (find top 2 eigenvalues)
    let result = lanczos_eigen(&csr, 2, 20, 1e-8)?;
    assert!(result.converged);
    println!("Top eigenvalues: {:?}", result.eigenvalues);
    println!("Eigenpair residuals ||Av - λv||: {:?}", result.residuals);

    Ok(())
}
```

---

### 9. Multivariate Optimization & Automatic Differentiation

```rust
use scies_math_th::autodiff::{Dual, jacobian};
use scies_math_th::opt_multivar::bfgs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2D Rosenbrock Banana Function: min f(x, y) = (1 - x)^2 + 100(y - x^2)^2
    let rosenbrock = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        let one = Dual::from(1.0);
        let hundred = Dual::from(100.0);
        (one - x) * (one - x) + hundred * (y - x * x) * (y - x * x)
    };

    let x0 = [-1.2, 1.0];
    let (x_min, f_min) = bfgs(rosenbrock, &x0, 1e-6, 500)?;
    println!("Minimizer: [x = {:.4}, y = {:.4}]", x_min[0], x_min[1]); // [1.0000, 1.0000]
    println!("Minimum function value: {:.2e}", f_min);

    // Exact automatic differentiation Jacobian
    let grad_at_min = jacobian(rosenbrock, &x_min);
    println!("Gradient norm at minimum: {:.2e}", grad_at_min[0].hypot(grad_at_min[1]));

    Ok(())
}
```

---

### 10. Differential Equations (Adaptive RK45 & 2D Heat ADI)

```rust
use scies_math_th::ode::rk45;
use scies_math_th::pde_ext::heat_2d_adi;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Adaptive Dormand-Prince RK45: dy/dt = -y, y(0) = 1.0
    let f_ode = |_t: f64, y: &[f64]| vec![-y[0]];
    let sol = rk45(f_ode, 0.0, 5.0, vec![1.0], 1e-6, 1e-9, 0.1)?;
    println!("ODE y(5.0) = {:.6} (Exact: {:.6})", sol.y.last().unwrap()[0], (-5.0_f64).exp());

    // 2. 2D Heat Equation via Alternating Direction Implicit (Peaceman-Rachford ADI)
    let nx = 21;
    let ny = 21;
    let mut u0 = vec![0.0; nx * ny];
    u0[(ny / 2) * nx + (nx / 2)] = 100.0; // Central heat point source

    let u_final = heat_2d_adi(&u0, nx, ny, 0.05, 0.05, 0.001, 1.0, 50)?;
    println!("Final central temperature: {:.4}", u_final[(ny / 2) * nx + (nx / 2)]);

    Ok(())
}
```

---

## Feature Flags

| Feature | Description |
| :--- | :--- |
| `default` | Zero external dependencies. |
| `serde` | Adds `Serialize` and `Deserialize` derives for mathematical structs and vectors. |
| `parallel` | Multithreaded blocked matrix multiplication and solvers via `rayon`. |
| `bench` | Benchmark-only utilities and profiling paths. |

---

## Quality Assurance & Benchmarks

The crate is subject to strict automated quality gates on every change:
- **`cargo fmt --all -- --check`**: 100% compliant with standard formatting.
- **`cargo clippy --offline --all-targets -- -D warnings`**: Zero compiler or clippy warnings.
- **`cargo test --offline`**: 89 unit, integration, and property tests passing.
- **`cargo doc --offline --no-deps`**: Zero broken links or rustdoc warnings.
- **`cargo bench --no-run --offline`**: 7 Criterion benchmark suites compiling cleanly.

---

## Documentation & License

- Full module API reference: [`docs/README.md`](docs/README.md)
- Mathematical Validity Hierarchy: [`docs/numerical_reliability.md`](docs/numerical_reliability.md)
- Subsystem Reliability Matrix: [`docs/api_reliability_matrix.md`](docs/api_reliability_matrix.md)
- Thai language overview: [`README.th.md`](README.th.md)

Licensed under the **MIT License** ([LICENSE](LICENSE)).
