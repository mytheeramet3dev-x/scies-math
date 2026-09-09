use scies_math_th::moment::log::CalculationLog;
use scies_math_th::moment::monomial::{BooleanDomain, MonomialBasis};
use scies_math_th::moment::sat_encoding::{Clause, CnfFormula, Literal};
use scies_math_th::moment::sdpa_io::export_sdpa_sparse;
use scies_math_th::sdp::{SdpSolverConfig, SdpStatus};

#[test]
fn test_monomial_basis_consistency() {
    let basis = MonomialBasis::new(4, 2, BooleanDomain::ZeroOne).unwrap();
    // Degree 0: 1
    // Degree 1: 4 (x0, x1, x2, x3)
    // Degree 2: 6 (x0x1, x0x2, x0x3, x1x2, x1x3, x2x3)
    // Total = 11
    assert_eq!(basis.len(), 11);
}

#[test]
fn test_sat_relaxation_satisfiable_formula() {
    // Satisfiable 3-SAT formula on 3 variables:
    // (x0 OR x1 OR x2)
    // (~x0 OR x1 OR ~x2)
    let formula = CnfFormula::new(
        3,
        vec![
            Clause::new(vec![
                Literal::positive(0),
                Literal::positive(1),
                Literal::positive(2),
            ]),
            Clause::new(vec![
                Literal::negative(0),
                Literal::positive(1),
                Literal::negative(2),
            ]),
        ],
    );

    // With degree 1, 3-SAT clause of degree 3 must fail with Err
    assert!(formula.build_moment_sdp_relaxation(1).is_err());

    // Degree 2 tracks monomials up to degree 4, which is sufficient for degree-3 clauses
    let sdp = formula.build_moment_sdp_relaxation(2).unwrap();
    let config = SdpSolverConfig {
        tolerance: 1e-4,
        max_iterations: 300,
        rho: 1.0,
        strict: false,
    };

    let solution = sdp.solve(&config).unwrap();
    assert_eq!(solution.status, SdpStatus::Optimal);
    assert!(solution.residuals.primal_equality_residual < 1e-2);

    // Export SDPA format for external solver verification
    let sdpa_text = export_sdpa_sparse(&sdp, "SAT Relaxation Instance").unwrap();
    assert!(sdpa_text.len() > 100);

    // Record calculation log
    let mut log = CalculationLog::new("SAT_Instance_3Var", sdp.matrix_dim());
    log.num_constraints = sdp.num_constraints();
    log.tolerance = config.tolerance;
    log.iterations = solution.iterations;
    log.primal_residual = solution.residuals.primal_equality_residual;
    log.min_eigenvalue = solution.residuals.min_eigenvalue_x;
    log.status = format!("{:?}", solution.status);

    let summary = log.summary();
    assert!(summary.contains("SAT_Instance_3Var"));
    assert!(summary.contains("status=Optimal"));
}

#[test]
fn test_cnf_dimacs_roundtrip_and_brute_force() {
    let dimacs_src = r#"c A simple 3-SAT problem
p cnf 3 2
1 2 3 0
-1 2 -3 0
"#;

    let formula = CnfFormula::from_dimacs(dimacs_src).unwrap();
    assert_eq!(formula.num_vars(), 3);
    assert_eq!(formula.num_clauses(), 2);

    let exported = formula.to_dimacs();
    let reloaded = CnfFormula::from_dimacs(&exported).unwrap();
    assert_eq!(reloaded.num_vars(), 3);
    assert_eq!(reloaded.num_clauses(), 2);

    // Brute force solver check
    let sol = formula.solve_brute_force().unwrap();
    assert!(sol.is_some());
    let assignment = sol.unwrap();
    assert!(formula.evaluate_assignment(&assignment).unwrap());

    // Unsatisfiable formula: x0 AND ~x0
    let unsat = CnfFormula::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    assert!(unsat.solve_brute_force().unwrap().is_none());
}
