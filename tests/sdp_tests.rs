use scies_math_th::moment::sdpa_io::{export_sdpa_sparse, import_sdpa_sparse};
use scies_math_th::sdp::{SdpProblem, SdpSolverConfig, SdpStatus};
use scies_math_th::symmetric::SymmetricMatrix;

#[test]
fn test_sdp_admm_solver_2d() {
    // Problem:
    // Min Tr(C * X)
    // s.t. <A1, X> = 1.0, <A2, X> = 1.0, X >= 0
    // C = [[1.0, 0.0], [0.0, 2.0]]
    // A1 = [[1.0, 0.0], [0.0, 0.0]] => X_00 = 1.0
    // A2 = [[0.0, 0.0], [0.0, 1.0]] => X_11 = 1.0
    // Expected optimal X = [[1.0, 0.0], [0.0, 1.0]], opt obj = 3.0

    let c = SymmetricMatrix::new(2, vec![1.0, 0.0, 2.0]).unwrap();
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap();
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0]).unwrap();

    let problem = SdpProblem::new(c, vec![a1, a2], vec![1.0, 1.0]).unwrap();
    let config = SdpSolverConfig {
        tolerance: 1e-4,
        max_iterations: 200,
        rho: 1.0,
        strict: false,
    };

    let solution = problem.solve(&config).unwrap();
    assert_eq!(solution.status, SdpStatus::Optimal);
    assert!((solution.primal_objective - 3.0).abs() < 1e-2);
    assert!((solution.x.get(0, 0).unwrap() - 1.0).abs() < 1e-2);
    assert!((solution.x.get(1, 1).unwrap() - 1.0).abs() < 1e-2);
    assert!(solution.residuals.primal_equality_residual < 1e-3);
}

#[test]
fn test_sdpa_export_import_integration() {
    let c = SymmetricMatrix::new(3, vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0]).unwrap();
    let a1 = SymmetricMatrix::identity(3).unwrap();
    let b = vec![5.0];

    let problem = SdpProblem::new(c, vec![a1], b).unwrap();
    let sdpa_str = export_sdpa_sparse(&problem, "Integration Test SDP").unwrap();
    assert!(sdpa_str.contains("\"* Integration Test SDP\""));

    let reloaded = import_sdpa_sparse(&sdpa_str).unwrap();
    assert_eq!(reloaded.matrix_dim(), 3);
    assert_eq!(reloaded.num_constraints(), 1);
    assert_eq!(reloaded.b[0], 5.0);
}

#[test]
fn test_unconstrained_sdp_solve() {
    // Min <C, X> s.t. X >= 0 with C >= 0 (identity) => optimal X = 0, obj = 0
    let c = SymmetricMatrix::identity(2).unwrap();
    let problem = SdpProblem::new(c, vec![], vec![]).unwrap();
    let config = SdpSolverConfig::default();

    let sol = problem.solve(&config).unwrap();
    assert_eq!(sol.status, SdpStatus::Optimal);
    assert!((sol.primal_objective - 0.0).abs() < 1e-10);
}

#[test]
fn test_verify_sdp_candidate_api() {
    let c = SymmetricMatrix::identity(2).unwrap();
    let a1 = SymmetricMatrix::identity(2).unwrap();
    let problem = SdpProblem::new(c, vec![a1], vec![2.0]).unwrap();

    let x = SymmetricMatrix::identity(2).unwrap(); // <I, I> = 2.0
    let y = vec![1.0];
    let s = SymmetricMatrix::zeros(2).unwrap(); // C - y A = I - 1*I = 0

    let report = scies_math_th::sdp::verify_sdp_candidate(&problem, &x, &y, &s, 1e-6).unwrap();
    assert!(report.is_optimal(1e-6));
    assert!(report.primal_equality_residual < 1e-10);
    assert!(report.dual_equality_residual < 1e-10);
}
