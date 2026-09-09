use scies_math_th::sdp::{SdpProblem, SdpStatus};
use scies_math_th::sdp_ipm::{SdpIpmConfig, solve_sdp_ipm};
use scies_math_th::symmetric::SymmetricMatrix;

#[test]
fn test_sdp_ipm_high_precision_convergence() {
    // Min Tr(C * X)
    // s.t. <A1, X> = 1.0, <A2, X> = 1.0, X >= 0
    // C = diag(2.0, 3.0)
    // A1 = diag(1.0, 0.0) -> X_00 = 1.0
    // A2 = diag(0.0, 1.0) -> X_11 = 1.0
    // Optimal X = diag(1.0, 1.0), optimal objective = 5.0
    let c = SymmetricMatrix::new(2, vec![2.0, 0.0, 3.0]).unwrap();
    let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 0.0]).unwrap();
    let a2 = SymmetricMatrix::new(2, vec![0.0, 0.0, 1.0]).unwrap();

    let problem = SdpProblem::new(c, vec![a1, a2], vec![1.0, 1.0]).unwrap();
    let config = SdpIpmConfig {
        tolerance: 1e-4,
        max_iterations: 50,
        step_damping: 0.95,
    };

    let solution = solve_sdp_ipm(&problem, &config).unwrap();
    assert_eq!(solution.status, SdpStatus::Optimal);
    assert!((solution.primal_objective - 5.0).abs() < 0.1);
    assert!((solution.x.get(0, 0).unwrap() - 1.0).abs() < 0.1);
    assert!((solution.x.get(1, 1).unwrap() - 1.0).abs() < 0.1);
}
