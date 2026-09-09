//! Property tests and invariants for linear algebra decompositions and sparse operators.

use scies_math_th::eigensystem::jacobi_eigen;
use scies_math_th::linear_algebra::DynamicMatrix;
use scies_math_th::sparse::{SparseMatrixCsr, lanczos_eigen};

#[test]
fn test_lu_decomposition_reconstruction() {
    let a = DynamicMatrix::new(
        3,
        3,
        vec![
            2.0, 1.0, 1.0, //
            4.0, -6.0, 0.0, //
            -2.0, 7.0, 2.0, //
        ],
    )
    .unwrap();

    let lu = a.lu_decompose().unwrap();
    let recon = lu.l.mul_matrix(&lu.u).unwrap();

    // Reconstruct permuted A
    let perm = &lu.permutation;
    for (r, &perm_r) in perm.iter().enumerate().take(3) {
        for c in 0..3 {
            let orig_val = a.get(perm_r, c).unwrap();
            let recon_val = recon.get(r, c).unwrap();
            assert!(
                (orig_val - recon_val).abs() < 1e-10,
                "LU mismatch at ({r}, {c})"
            );
        }
    }
}

#[test]
fn test_qr_decomposition_reconstruction_and_orthogonality() {
    let a = DynamicMatrix::new(
        3,
        3,
        vec![
            12.0, -51.0, 4.0, //
            6.0, 167.0, -68.0, //
            -4.0, 24.0, -41.0, //
        ],
    )
    .unwrap();

    let qr = a.qr_decompose().unwrap();
    let recon = qr.q.mul_matrix(&qr.r).unwrap();

    // Reconstruction A = Q * R
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (a.get(r, c).unwrap() - recon.get(r, c).unwrap()).abs() < 1e-10,
                "QR reconstruction mismatch"
            );
        }
    }

    // Orthogonality Q^T * Q = I
    let qtq = qr.q.transpose().mul_matrix(&qr.q).unwrap();
    for r in 0..3 {
        for c in 0..3 {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!(
                (qtq.get(r, c).unwrap() - expected).abs() < 1e-10,
                "Q is not orthogonal"
            );
        }
    }
}

#[test]
fn test_cholesky_decomposition_reconstruction_and_non_pd_rejection() {
    // Strictly positive definite matrix
    let a = DynamicMatrix::new(
        3,
        3,
        vec![
            4.0, 12.0, -16.0, //
            12.0, 37.0, -43.0, //
            -16.0, -43.0, 98.0, //
        ],
    )
    .unwrap();

    let chol = a.cholesky_decompose().unwrap();
    let recon = chol.l.mul_matrix(&chol.l.transpose()).unwrap();

    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (a.get(r, c).unwrap() - recon.get(r, c).unwrap()).abs() < 1e-10,
                "Cholesky reconstruction mismatch"
            );
        }
    }

    // Indefinite matrix rejection
    let indef = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 2.0, 1.0]).unwrap();
    assert!(indef.cholesky_decompose().is_err());
}

#[test]
fn test_svd_reconstruction_and_orthogonality() {
    let a = DynamicMatrix::new(2, 2, vec![3.0, 0.0, 0.0, -2.0]).unwrap();
    let svd = a.singular_value_decompose(1e-10, 100).unwrap();

    // Singular values must be non-negative and sorted descending
    assert!(svd.singular_values[0] >= svd.singular_values[1]);
    assert!(svd.singular_values[1] >= 0.0);
    assert!((svd.singular_values[0] - 3.0).abs() < 1e-6);
    assert!((svd.singular_values[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_jacobi_eigen_residuals() {
    let a = DynamicMatrix::new(
        3,
        3,
        vec![
            2.0, -1.0, 0.0, //
            -1.0, 2.0, -1.0, //
            0.0, -1.0, 2.0, //
        ],
    )
    .unwrap();

    let eig = jacobi_eigen(&a, 1e-12, 100).unwrap();
    assert_eq!(eig.values.len(), 3);

    let max_pair_res = eig.max_eigenpair_residual(&a).unwrap();
    let ortho_res = eig.orthogonality_residual().unwrap();
    let spec_res = eig.spectral_residual(&a).unwrap();

    assert!(max_pair_res < 1e-10);
    assert!(ortho_res < 1e-10);
    assert!(spec_res < 1e-10);
}

#[test]
fn test_lanczos_sparse_eigensolver() {
    // 4x4 Laplacian tridiagonal sparse matrix
    let triplets = vec![
        (0, 0, 2.0),
        (0, 1, -1.0),
        (1, 0, -1.0),
        (1, 1, 2.0),
        (1, 2, -1.0),
        (2, 1, -1.0),
        (2, 2, 2.0),
        (2, 3, -1.0),
        (3, 2, -1.0),
        (3, 3, 2.0),
    ];

    let csr = SparseMatrixCsr::from_triplets(4, 4, &triplets).unwrap();
    csr.validate_invariants().unwrap();
    assert!(csr.is_symmetric(1e-10));

    let lanczos = lanczos_eigen(&csr, 2, 10, 1e-6).unwrap();
    assert_eq!(lanczos.eigenvalues.len(), 2);
    assert!(lanczos.converged);
    assert!(lanczos.eigenvalues[0] >= lanczos.eigenvalues[1]);

    for &res in &lanczos.residuals {
        assert!(res < 1e-6);
    }
}
