//! Comprehensive reliability, strict validation, and KKT verification integration tests.

use scies_math_th::eigensystem::jacobi_eigen;
use scies_math_th::errors::SciError;
use scies_math_th::exact::rational::Rational;
use scies_math_th::linear_algebra::DynamicMatrix;
use scies_math_th::moment::builder::MomentMatrixBuilder;
use scies_math_th::moment::monomial::{BooleanDomain, Monomial};
use scies_math_th::moment::sat_encoding::{Clause, CnfFormula, Literal};
use scies_math_th::moment::sdpa_io::{export_sdpa_sparse, import_sdpa_sparse};
use scies_math_th::sdp::{SdpProblem, SdpSolverConfig, SdpStatus};
use scies_math_th::sdp_ipm::{SdpIpmConfig, solve_sdp_ipm};
use scies_math_th::sdp_primitives::diagnose_psd;
use scies_math_th::symmetric::SymmetricMatrix;

#[test]
fn test_sdp_strict_kkt_verification() {
    // 2D SDP problem:
    // Min X_00 + 2 X_11
    // s.t. X_00 = 1.0, X_11 = 1.0, X >= 0
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0]).unwrap();
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap();
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0]).unwrap();
    let b = vec![1.0, 1.0];

    let problem = SdpProblem::new(c, vec![a1, a2], b).unwrap();

    // Verify candidate solution directly using independent verifier
    let x_opt = SymmetricMatrix::new(2, vec![1.0, 0.0, 1.0]).unwrap();
    let y_opt = vec![1.0, 2.0];
    let s_opt = SymmetricMatrix::new(2, vec![0.0, 0.0, 0.0]).unwrap();

    let report = problem
        .verify_solution(&x_opt, &y_opt, &s_opt, 1e-8)
        .unwrap();
    assert!(report.is_optimal(1e-6));
    assert_eq!(report.primal_equality_residual, 0.0);
    assert_eq!(report.dual_equality_residual, 0.0);
    assert!(report.relative_duality_gap < 1e-8);

    // Corrupted candidate with violated primal equality
    let x_bad = SymmetricMatrix::new(2, vec![2.0, 0.0, 1.0]).unwrap();
    let bad_report = problem
        .verify_solution(&x_bad, &y_opt, &s_opt, 1e-8)
        .unwrap();
    assert!(!bad_report.is_optimal(1e-6));
    assert!((bad_report.primal_equality_residual - 1.0).abs() < 1e-8);

    // Corrupted candidate with negative eigenvalue in S
    let s_neg = SymmetricMatrix::new(2, vec![-0.5, 0.0, 0.0]).unwrap();
    let bad_s_report = problem
        .verify_solution(&x_opt, &y_opt, &s_neg, 1e-8)
        .unwrap();
    assert!(!bad_s_report.is_optimal(1e-6));
    assert!((bad_s_report.dual_psd_violation - 0.5).abs() < 1e-6);
}

#[test]
fn test_sdp_ipm_strict_convergence() {
    let c = SymmetricMatrix::new(2, vec![2.0, 0.0, 1.0]).unwrap();
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap();
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0]).unwrap();
    let b = vec![3.0, 4.0];

    let problem = SdpProblem::new(c, vec![a1, a2], b).unwrap();
    let config = SdpIpmConfig {
        tolerance: 1e-5,
        max_iterations: 50,
        step_damping: 0.9,
    };

    let solution = solve_sdp_ipm(&problem, &config).unwrap();
    assert_eq!(solution.status, SdpStatus::Optimal);
    assert!(solution.residuals.is_optimal(1e-4));
    assert!((solution.primal_objective - (2.0 * 3.0 + 1.0 * 4.0)).abs() < 1e-3);
}

#[test]
fn test_sdp_redundant_constraints_handling() {
    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 1.0]).unwrap();
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap();
    let a2 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap(); // Duplicate constraint -> singular Gram matrix
    let b = vec![1.0, 1.0];

    let problem = SdpProblem::new(c, vec![a1, a2], b).unwrap();
    let config = SdpSolverConfig::default();

    // Solver must report SingularMatrix error instead of crashing or silent nonsense
    let result = problem.solve(&config);
    assert!(matches!(result, Err(SciError::SingularMatrix(_))));
}

#[test]
fn test_moment_insufficient_degree_returns_error() {
    // 3-variable clause (x0 OR x1 OR x2) has degree 3
    let formula = CnfFormula::new(
        3,
        vec![Clause::new(vec![
            Literal::positive(0),
            Literal::positive(1),
            Literal::positive(2),
        ])],
    );

    // Degree 1 cannot represent degree 3 violation polynomial in moment basis
    let res = formula.build_moment_sdp_relaxation_with_report(1);
    assert!(res.is_err());
    assert!(matches!(res, Err(SciError::InvalidParameter(_))));

    // Degree 2 succeeds and returns complete report
    let (sdp, report) = formula.build_moment_sdp_relaxation_with_report(2).unwrap();
    assert_eq!(report.num_variables, 3);
    assert_eq!(report.relaxation_degree, 2);
    assert_eq!(report.basis_size, 7);
    assert_eq!(report.num_moments, 8);
    assert_eq!(report.clause_degrees, vec![3]);
    assert_eq!(report.num_generated_constraints, sdp.num_constraints());
}

#[test]
fn test_localizing_matrix_constraint_generation() {
    let builder = MomentMatrixBuilder::new(2, 2, BooleanDomain::ZeroOne).unwrap();
    let x0 = Monomial::single_var(0);
    let x1 = Monomial::single_var(1);

    // Polynomial equality: x0 + x1 - 1.0 = 0
    let poly = vec![(x0, 1.0), (x1, 1.0), (Monomial::constant(), -1.0)];

    let (a_cons, b_vals) = builder.generate_localizing_constraints(&poly, 0.0).unwrap();
    assert_eq!(a_cons.len(), 1);
    assert_eq!(b_vals[0], 0.0);
    assert_eq!(a_cons[0].size(), builder.matrix_dim());
}

#[test]
fn test_sdpa_format_strict_parsing_and_rejection() {
    // Correct single block
    let valid_sdpa = "\"* Header\"\n1\n1\n2\n3.0\n0 1 1 1 1.0\n0 1 2 2 2.0\n1 1 1 1 1.0\n";
    let prob = import_sdpa_sparse(valid_sdpa).unwrap();
    assert_eq!(prob.matrix_dim(), 2);
    assert_eq!(prob.num_constraints(), 1);
    assert_eq!(prob.b, vec![3.0]);

    // Roundtrip verification
    let exported = export_sdpa_sparse(&prob, "Export Test").unwrap();
    let re_imported = import_sdpa_sparse(&exported).unwrap();
    assert_eq!(prob.c.raw_data(), re_imported.c.raw_data());

    // Rejection of multi-block
    let multi_block = "\"* Header\"\n1\n2\n2 2\n3.0\n0 1 1 1 1.0\n";
    assert!(import_sdpa_sparse(multi_block).is_err());

    // Rejection of NaN float
    let nan_sdpa = "\"* Header\"\n1\n1\n2\n3.0\n0 1 1 1 NaN\n";
    assert!(import_sdpa_sparse(nan_sdpa).is_err());

    // Rejection of out of bound index
    let oob_sdpa = "\"* Header\"\n1\n1\n2\n3.0\n0 1 3 3 1.0\n";
    assert!(import_sdpa_sparse(oob_sdpa).is_err());
}

#[test]
fn test_exact_rational_checked_safety() {
    // Division by zero
    assert_eq!(Rational::new(5, 0), Err(SciError::DivisionByZero));
    let r = Rational::new(3, 4).unwrap();
    assert_eq!(
        r.checked_div(&Rational::zero()),
        Err(SciError::DivisionByZero)
    );

    // Integer overflow detection
    let huge = Rational::new(i128::MAX, 1).unwrap();
    let huge2 = Rational::new(i128::MAX - 1, 1).unwrap();
    assert!(huge.checked_add(&huge2).is_err());
    assert!(huge.checked_mul(&huge2).is_err());
}

#[test]
fn test_psd_diagnostics_and_rejection() {
    // Valid PD matrix
    let mat = DynamicMatrix::new(2, 2, vec![2.0, 0.0, 0.0, 3.0]).unwrap();
    let diag = diagnose_psd(&mat, 1e-8).unwrap();
    assert!(diag.is_psd);
    assert!(diag.is_strictly_pd);
    assert_eq!(diag.numerical_rank, 2);

    // Asymmetric matrix rejection
    let asym = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 1.0]).unwrap();
    assert!(diagnose_psd(&asym, 1e-8).is_err());

    // NaN input rejection
    let nan_mat = DynamicMatrix::new(2, 2, vec![1.0, f64::NAN, f64::NAN, 1.0]).unwrap();
    assert!(diagnose_psd(&nan_mat, 1e-8).is_err());
}

#[test]
fn test_eigensystem_residuals() {
    // Symmetric matrix
    let mat = DynamicMatrix::new(2, 2, vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let eig = jacobi_eigen(&mat, 1e-12, 100).unwrap();

    let max_pair_res = eig.max_eigenpair_residual(&mat).unwrap();
    let ortho_res = eig.orthogonality_residual().unwrap();
    let spec_res = eig.spectral_residual(&mat).unwrap();

    assert!(max_pair_res < 1e-10);
    assert!(ortho_res < 1e-10);
    assert!(spec_res < 1e-10);
}
