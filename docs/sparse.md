# Sparse Matrix Algebra and Iterative Solvers (`sparse`)

The `sparse` module provides Compressed Sparse Row (CSR) storage, preconditioners, iterative linear solvers, and an iterative Lanczos symmetric eigensolver.

---

## 1. Compressed Sparse Row (`SparseMatrixCsr`)

Stores an $m \times n$ matrix with $N_{\text{nz}}$ non-zeros using three compact contiguous vectors:
- `values: Vec<f64>` (length $N_{\text{nz}}$): Non-zero numerical values.
- `col_indices: Vec<usize>` (length $N_{\text{nz}}$): Column index of each non-zero.
- `row_pointers: Vec<usize>` (length $m + 1$): Offset into `values` where each row begins.

### Invariant Validation (`validate_invariants`)
Guarantees integrity before calling numerical solvers:
- `row_pointers[0] == 0` and monotonically non-decreasing.
- `col_indices[k] < cols`.
- Column indices strictly increasing within each row (no unsorted duplicates).
- All stored values are finite (no NaN / Inf).

---

## 2. Iterative Linear Solvers

| Method | System Type | Preconditioned | Typical Convergence Rate |
| :--- | :--- | :--- | :--- |
| `conjugate_gradient` | Symmetric Positive Definite (SPD) | No | $O(\sqrt{\kappa} \ln(1/\epsilon))$ |
| `preconditioned_cg` | Symmetric Positive Definite (SPD) | Yes (ILU(0) / Jacobi) | Accelerated via preconditioning |
| `gmres` | General Non-Symmetric | No | Krylov subspace minimization |
| `preconditioned_gmres`| General Non-Symmetric | Yes (ILU(0) / Jacobi) | Accelerated GMRES |
| `bicgstab` | General Non-Symmetric | No | Smooth Bi-Conjugate Gradient |
| `minres` | Symmetric Indefinite | No | Minimum residual |

### Preconditioners
- `DiagPreconditioner`: Diagonal / Jacobi scaling $M = \text{diag}(A)$.
- `IluPreconditioner`: Incomplete LU factorization preserving sparsity pattern $\text{ILU}(0)$.

---

## 3. Lanczos Symmetric Eigensolver (`lanczos_eigen`)

Computes the extreme eigenvalues and Ritz eigenvectors of large symmetric sparse matrices using the Lanczos iteration with full Gram-Schmidt reorthogonalization:

```rust
pub fn lanczos_eigen(
    matrix: &SparseMatrixCsr,
    k_eigenvalues: usize,
    max_iterations: usize,
    tolerance: f64,
) -> SciResult<LanczosEigenResult>
```

### Guarantees & Features
- **Deterministic Initial Vector**: $q_1 = [1/\sqrt{n}, \dots, 1/\sqrt{n}]^T$.
- **Full Reorthogonalization**: Safeguards against loss of orthogonality among Lanczos vectors.
- **Residual Tracking**: Reports $\|A v_i - \lambda_i v_i\|_2$ for each Ritz pair.
- **Canonical Sign Convention**: Enforces the first non-negligible component of each eigenvector to be positive.

---

## 4. Code Example

```rust
use scies_math_th::sparse::{SparseMatrixCsr, lanczos_eigen};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 5x5 Laplacian matrix
    let triplets = vec![
        (0, 0, 2.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2, 2.0), (2, 3, -1.0),
        (3, 2, -1.0), (3, 3, 2.0), (3, 4, -1.0),
        (4, 3, -1.0), (4, 4, 2.0),
    ];

    let csr = SparseMatrixCsr::from_triplets(5, 5, &triplets)?;
    csr.validate_invariants()?;

    let eig = lanczos_eigen(&csr, 2, 20, 1e-8)?;
    assert!(eig.converged);

    println!("Top 2 eigenvalues: {:?}", eig.eigenvalues);
    println!("Eigenpair residuals: {:?}", eig.residuals);

    Ok(())
}
```
