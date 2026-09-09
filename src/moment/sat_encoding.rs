//! CNF and 3-SAT clause polynomial encodings for Moment-SOS SDP relaxations.
//!
//! Encodes Boolean satisfiability problems (CNF / 3-SAT) into polynomial equality
//! constraints on the moment matrix.

use super::builder::MomentMatrixBuilder;
use super::monomial::{BooleanDomain, Monomial};
use crate::errors::{SciError, SciResult};
use crate::sdp::SdpProblem;
use crate::symmetric::SymmetricMatrix;

/// A Boolean literal representing either variable $x_i$ or its negation $\neg x_i$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Literal {
    /// 0-indexed variable identifier.
    pub variable: usize,
    /// `false` for positive literal $x_i$, `true` for negated literal $\neg x_i$.
    pub negated: bool,
}

impl Literal {
    /// Creates a positive literal $x_v$.
    pub fn positive(variable: usize) -> Self {
        Self {
            variable,
            negated: false,
        }
    }

    /// Creates a negated literal $\neg x_v$.
    pub fn negative(variable: usize) -> Self {
        Self {
            variable,
            negated: true,
        }
    }
}

/// A CNF clause representing the disjunction of literals $(\ell_1 \lor \dots \lor \ell_k)$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    /// Creates a clause from a list of literals.
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    /// Number of literals in the clause (clause width/degree).
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Whether the clause has no literals.
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Expands the clause violation polynomial $\prod_{i} e_i = 0$ in $\{0, 1\}$ basis.
    ///
    /// For literal $x_v$, $e_i = (1 - x_v)$. For literal $\neg x_v$, $e_i = x_v$.
    /// The clause is satisfied iff $\prod e_i = 0$.
    ///
    /// Returns a list of `(Monomial, coefficient)`.
    pub fn to_violation_polynomial(&self) -> Vec<(Monomial, f64)> {
        // Start with polynomial P = [ (1, 1.0) ]
        let mut poly: Vec<(Monomial, f64)> = vec![(Monomial::constant(), 1.0)];

        for lit in &self.literals {
            let mut next_poly = Vec::new();
            let var_mon = Monomial::single_var(lit.variable);

            for (m, coeff) in poly {
                if lit.negated {
                    // Multiply by x_v
                    let (prod, _) = m.mul_boolean(&var_mon, BooleanDomain::ZeroOne);
                    next_poly.push((prod, coeff));
                } else {
                    // Multiply by (1 - x_v) = 1*m - x_v*m
                    next_poly.push((m.clone(), coeff));
                    let (prod, _) = m.mul_boolean(&var_mon, BooleanDomain::ZeroOne);
                    next_poly.push((prod, -coeff));
                }
            }

            // Consolidate identical monomials
            poly = consolidate_polynomial(next_poly);
        }

        poly
    }
}

fn consolidate_polynomial(terms: Vec<(Monomial, f64)>) -> Vec<(Monomial, f64)> {
    let mut map: Vec<(Monomial, f64)> = Vec::new();
    for (mon, coeff) in terms {
        if let Some(entry) = map.iter_mut().find(|(m, _)| *m == mon) {
            entry.1 += coeff;
        } else {
            map.push((mon, coeff));
        }
    }
    map.retain(|(_, coeff)| coeff.abs() > 1e-12);
    map
}

/// Diagnostic report detailing the dimensions and structure of a Moment-SOS SDP relaxation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MomentRelaxationReport {
    /// Number of Boolean variables in the problem.
    pub num_variables: usize,
    /// Moment relaxation hierarchy level $d$.
    pub relaxation_degree: usize,
    /// Size $N$ of the moment matrix ($N \times N$).
    pub basis_size: usize,
    /// Total number of distinct moment variables tracked up to degree $2d$.
    pub num_moments: usize,
    /// Degrees (number of literals) for each encoded clause.
    pub clause_degrees: Vec<usize>,
    /// Total number of generated linear equality constraints in the SDP.
    pub num_generated_constraints: usize,
}

/// A CNF formula consisting of $n$ variables and $m$ clauses.
#[derive(Debug, Clone, PartialEq)]
pub struct CnfFormula {
    pub num_vars: usize,
    pub clauses: Vec<Clause>,
}

impl CnfFormula {
    /// Creates a new CNF formula.
    pub fn new(num_vars: usize, clauses: Vec<Clause>) -> Self {
        Self { num_vars, clauses }
    }

    /// Number of variables in the formula.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of clauses in the formula.
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Compiles this CNF formula into a standard Semidefinite Programming relaxation (`SdpProblem`)
    /// along with a detailed diagnostic report.
    ///
    /// Uses degree $d$ moment relaxation (e.g. $d = 2$ for standard Lovász-Schrijver / Lasserre level 2).
    ///
    /// # Errors
    /// Returns [`SciError::InvalidParameter`] if `degree == 0` or if the relaxation degree is insufficient
    /// to represent any clause violation polynomial in the moment matrix basis.
    pub fn build_moment_sdp_relaxation_with_report(
        &self,
        degree: usize,
    ) -> SciResult<(SdpProblem, MomentRelaxationReport)> {
        if degree == 0 {
            return Err(SciError::InvalidParameter("degree must be >= 1"));
        }

        let builder = MomentMatrixBuilder::new(self.num_vars, degree, BooleanDomain::ZeroOne)?;
        let n = builder.matrix_dim();

        // Base moment structure constraints
        let (mut a_constraints, mut b_values) = builder.generate_moment_sdp_constraints()?;

        let mut clause_degrees = Vec::with_capacity(self.clauses.len());

        // Clause equality constraints: E[ P_clause(x) ] = 0
        for (clause_idx, clause) in self.clauses.iter().enumerate() {
            clause_degrees.push(clause.len());
            let poly_terms = clause.to_violation_polynomial();

            let (clause_a, clause_b) = builder
                .generate_polynomial_equality_constraint(&poly_terms, 0.0)
                .map_err(|_| {
                    SciError::InvalidParameter(
                        "relaxation degree is insufficient to represent clause violation polynomial in moment basis (increase relaxation degree)",
                    )
                })?;

            a_constraints.push(clause_a);
            b_values.push(clause_b);
            let _ = clause_idx;
        }

        let report = MomentRelaxationReport {
            num_variables: self.num_vars,
            relaxation_degree: degree,
            basis_size: builder.matrix_dim(),
            num_moments: builder.num_moments(),
            clause_degrees,
            num_generated_constraints: a_constraints.len(),
        };

        // Objective: constant 0 (feasibility test)
        let c = SymmetricMatrix::zeros(n)?;
        let problem = SdpProblem::new(c, a_constraints, b_values)?;

        Ok((problem, report))
    }

    /// Compiles this CNF formula into a standard Semidefinite Programming relaxation (`SdpProblem`).
    pub fn build_moment_sdp_relaxation(&self, degree: usize) -> SciResult<SdpProblem> {
        self.build_moment_sdp_relaxation_with_report(degree)
            .map(|(prob, _)| prob)
    }

    /// Evaluates whether a candidate Boolean assignment (length `num_vars`) satisfies this CNF formula.
    pub fn evaluate_assignment(&self, assignment: &[bool]) -> SciResult<bool> {
        if assignment.len() < self.num_vars {
            return Err(SciError::InvalidParameter(
                "assignment length must be >= num_vars",
            ));
        }
        for clause in &self.clauses {
            let mut clause_satisfied = false;
            for lit in &clause.literals {
                let val = assignment[lit.variable];
                let lit_val = if lit.negated { !val } else { val };
                if lit_val {
                    clause_satisfied = true;
                    break;
                }
            }
            if !clause_satisfied {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Exact brute-force satisfiability oracle for small formulas ($n \le 20$).
    ///
    /// Returns `Some(assignment)` if satisfiable, or `None` if certifiably unsatisfiable.
    pub fn solve_brute_force(&self) -> SciResult<Option<Vec<bool>>> {
        if self.num_vars > 20 {
            return Err(SciError::InvalidParameter(
                "brute force oracle only supports up to 20 variables",
            ));
        }
        let total_states = 1usize << self.num_vars;
        for mask in 0..total_states {
            let assignment: Vec<bool> =
                (0..self.num_vars).map(|i| (mask & (1 << i)) != 0).collect();
            if self.evaluate_assignment(&assignment)? {
                return Ok(Some(assignment));
            }
        }
        Ok(None)
    }

    /// Parses a DIMACS CNF formatted string.
    ///
    /// Ignores comments (lines starting with 'c').
    /// Requires problem header line: `p cnf <vars> <clauses>`.
    pub fn from_dimacs(dimacs_str: &str) -> SciResult<Self> {
        let mut num_vars = 0;
        let mut expected_clauses = 0;
        let mut header_found = false;
        let mut clauses = Vec::new();
        let mut current_clause_lits = Vec::new();

        for line in dimacs_str.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('c') {
                continue;
            }
            if trimmed.starts_with('p') {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() != 4 || parts[1] != "cnf" {
                    return Err(SciError::ParseError(
                        "invalid DIMACS header format: expected 'p cnf <vars> <clauses>'"
                            .to_string(),
                    ));
                }
                num_vars = parts[2].parse::<usize>().map_err(|_| {
                    SciError::ParseError("invalid variable count in DIMACS header".to_string())
                })?;
                expected_clauses = parts[3].parse::<usize>().map_err(|_| {
                    SciError::ParseError("invalid clause count in DIMACS header".to_string())
                })?;
                header_found = true;
                continue;
            }

            if !header_found {
                return Err(SciError::ParseError(
                    "DIMACS clause encountered before header line".to_string(),
                ));
            }

            for token in trimmed.split_whitespace() {
                let val = token.parse::<isize>().map_err(|_| {
                    SciError::ParseError(format!("invalid literal integer token '{token}'"))
                })?;
                if val == 0 {
                    clauses.push(Clause::new(std::mem::take(&mut current_clause_lits)));
                } else {
                    let var_idx = val.unsigned_abs() - 1;
                    if var_idx >= num_vars {
                        return Err(SciError::ParseError(format!(
                            "literal index {} exceeds variable count {}",
                            var_idx + 1,
                            num_vars
                        )));
                    }
                    let negated = val < 0;
                    current_clause_lits.push(Literal {
                        variable: var_idx,
                        negated,
                    });
                }
            }
        }

        if !current_clause_lits.is_empty() {
            clauses.push(Clause::new(current_clause_lits));
        }

        if !header_found {
            return Err(SciError::ParseError(
                "no DIMACS header line found".to_string(),
            ));
        }

        if clauses.len() != expected_clauses {
            return Err(SciError::ParseError(format!(
                "clause count mismatch: header declared {}, found {}",
                expected_clauses,
                clauses.len()
            )));
        }

        Ok(Self { num_vars, clauses })
    }

    /// Serializes this CNF formula to DIMACS CNF format.
    pub fn to_dimacs(&self) -> String {
        let mut out = format!("p cnf {} {}\n", self.num_vars, self.clauses.len());
        for clause in &self.clauses {
            for lit in &clause.literals {
                let val = (lit.variable + 1) as isize;
                let signed = if lit.negated { -val } else { val };
                out.push_str(&format!("{signed} "));
            }
            out.push_str("0\n");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clause_violation_polynomial() {
        // Clause (x0 OR ~x1)
        // Satisfied unless (1 - x0) * x1 = 0 => x1 - x0 * x1 = 0
        let clause = Clause::new(vec![Literal::positive(0), Literal::negative(1)]);

        let poly = clause.to_violation_polynomial();
        assert_eq!(poly.len(), 2);

        let x1 = Monomial::single_var(1);
        let x0x1 = Monomial::from_powers(vec![(0, 1), (1, 1)]);

        let term_x1 = poly.iter().find(|(m, _)| *m == x1).unwrap();
        assert!((term_x1.1 - 1.0).abs() < 1e-10);

        let term_x0x1 = poly.iter().find(|(m, _)| *m == x0x1).unwrap();
        assert!((term_x0x1.1 - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cnf_relaxation_build() {
        let formula = CnfFormula::new(
            3,
            vec![Clause::new(vec![
                Literal::positive(0),
                Literal::positive(1),
                Literal::positive(2),
            ])],
        );

        let (sdp, report) = formula.build_moment_sdp_relaxation_with_report(2).unwrap();
        assert!(sdp.num_constraints() > 0);
        assert!(sdp.matrix_dim() > 0);
        assert_eq!(report.num_variables, 3);
        assert_eq!(report.relaxation_degree, 2);
        assert_eq!(report.clause_degrees, vec![3]);
    }
}
