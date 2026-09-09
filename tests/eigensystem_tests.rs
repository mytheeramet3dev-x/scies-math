use scies_math_th::eigensystem::jacobi_eigen;
use scies_math_th::linear_algebra::DynamicMatrix;

#[test]
fn test_jacobi_eigen_direct_api() {
    let mat =
        DynamicMatrix::new(3, 3, vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]).unwrap();

    let eig = jacobi_eigen(&mat, 1e-12, 100).unwrap();
    assert!(eig.converged, "Jacobi solver should converge");
    assert!(eig.sweeps > 0, "Sweeps count should be positive");
    assert!(eig.residual < 1e-12, "Residual should be below tolerance");

    // Exact eigenvalues of 1D Laplacian: 2 - 2*cos(k*pi/4) for k=1,2,3
    // lambda1 = 2 - sqrt(2) approx 0.585786
    // lambda2 = 2.0
    // lambda3 = 2 + sqrt(2) approx 3.414213
    let mut sorted_vals = eig.values.clone();
    sorted_vals.sort_by(|a, b| a.total_cmp(b));

    let exp1 = 2.0 - 2.0_f64.sqrt();
    let exp2 = 2.0;
    let exp3 = 2.0 + 2.0_f64.sqrt();

    assert!((sorted_vals[0] - exp1).abs() < 1e-10);
    assert!((sorted_vals[1] - exp2).abs() < 1e-10);
    assert!((sorted_vals[2] - exp3).abs() < 1e-10);

    let max_res = eig.max_eigenpair_residual(&mat).unwrap();
    assert!(max_res < 1e-10, "Max eigenpair residual is {max_res}");
}

#[test]
fn test_dynamic_matrix_jacobi_forwarder() {
    let mat = DynamicMatrix::new(2, 2, vec![3.0, 1.0, 1.0, 3.0]).unwrap();
    let eig = mat.jacobi_eigendecomposition(1e-12, 50).unwrap();
    assert!(eig.converged);
    assert!((eig.values[0] - 4.0).abs() < 1e-10);
    assert!((eig.values[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_hilbert_matrix_spectral_properties() {
    // 3x3 Hilbert matrix H_ij = 1 / (i + j - 1)
    let h3 = DynamicMatrix::new(
        3,
        3,
        vec![
            1.0,
            1.0 / 2.0,
            1.0 / 3.0,
            1.0 / 2.0,
            1.0 / 3.0,
            1.0 / 4.0,
            1.0 / 3.0,
            1.0 / 4.0,
            1.0 / 5.0,
        ],
    )
    .unwrap();

    let eig = jacobi_eigen(&h3, 1e-12, 100).unwrap();
    assert!(eig.converged);
    // All eigenvalues of Hilbert matrix must be strictly positive
    for &val in &eig.values {
        assert!(
            val > 0.0,
            "Hilbert matrix eigenvalue must be positive: {val}"
        );
    }

    let max_res = eig.max_eigenpair_residual(&h3).unwrap();
    assert!(
        max_res < 1e-10,
        "Residual of Hilbert eigenpairs should be tiny: {max_res}"
    );
}

#[test]
fn test_asymmetric_matrix_rejected() {
    let asym = DynamicMatrix::new(2, 2, vec![1.0, 5.0, 0.0, 1.0]).unwrap();
    let err = jacobi_eigen(&asym, 1e-10, 50);
    assert!(err.is_err());
}
